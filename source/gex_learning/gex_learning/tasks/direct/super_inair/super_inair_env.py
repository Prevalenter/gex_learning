"""Fixed-base GX4M grasp collection. Actions are deliberately ignored: targets are held."""
from __future__ import annotations

import re
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaacsim.core.simulation_manager import SimulationManager
from isaaclab.utils import configclass

from .hand_env import Gx4mHandEnv, Gx4mHandEnvCfg
from .grasp_data import JOINT_NAMES, validate_seed
from .object_assets import ROOT, seed_object_size, seed_object_mass


@configclass
class Gx4mSuperInAirEnvCfg(Gx4mHandEnvCfg):
    seed_pose: dict = {}
    show_goal_marker = False
    show_thumb_marker = False
    show_visual_ground = False
    enable_rna = False
    force_scale = 0.0
    enable_object_size_dr = False
    enable_object_size_randomization = False
    action_noise_model = None
    observation_noise_model = None
    policy_obs_noise_model = None
    save_hand_trajectory = False
    # Check every physics step, retaining the original 60 Hz physics timestep.
    decimation = 1


class Gx4mSuperInAirEnv(Gx4mHandEnv):
    @staticmethod
    def _apply_hand_config(cfg):
        seed = validate_seed(cfg.seed_pose)
        if seed.get("object", {}).get("id") != "tennis_ball_60":
            raise ValueError("Only tennis_ball_60 is supported")
        cfg.default_hand_joint_pos = seed["joint_pos"]
        cfg.hand_start_pos = seed["hand_pose"][:3]
        cfg.object_start_pos = seed["object_pose"][:3]
        cfg.object_size = seed_object_size(seed)
        Gx4mHandEnv._apply_hand_config(cfg)
        cfg.robot_cfg.init_state.rot = tuple(seed["hand_pose"][3:])
        cfg.hand_start_rot = seed["hand_pose"][3:]
        cfg.robot_cfg.spawn.fix_base = True
        material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        cfg.sim.gravity = (0.0, 0.0, -9.81)
        cfg.sim.physics_material = material
        cfg.object_cfg.spawn = sim_utils.SphereCfg(
            radius=seed_object_size(seed) / 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=8, solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=seed_object_mass(seed)),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.12)),
        )
        if seed['version'] == 2:
            import xml.etree.ElementTree as ET
            path = ROOT / seed['object']['urdf_path']
            geometry = ET.parse(path).getroot().find('link/collision/geometry')[0]
            previous = cfg.object_cfg.spawn
            props = dict(rigid_props=previous.rigid_props, collision_props=previous.collision_props,
                         mass_props=previous.mass_props, visual_material=previous.visual_material,
                         physics_material=material)
            if geometry.tag == 'cylinder':
                cfg.object_cfg.spawn = sim_utils.CylinderCfg(radius=float(geometry.get('radius')),
                    height=float(geometry.get('length')), axis='Z', **props)
            elif geometry.tag == 'box':
                cfg.object_cfg.spawn = sim_utils.CuboidCfg(size=tuple(map(float, geometry.get('size').split())), **props)
            else:
                cfg.object_cfg.spawn = sim_utils.UrdfFileCfg(
                    asset_path=str(path), fix_base=False, joint_drive=None, collider_type='convex_hull',
                    make_instanceable=False, usd_dir=str(ROOT/'outputs/super_inair/object_usd'/seed['object']['id']),
                    rigid_props=previous.rigid_props, mass_props=previous.mass_props,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False))
        cfg.object_cfg.init_state.rot = tuple(seed["object_pose"][3:])
        cfg.object_base_scale = [1.0, 1.0, 1.0]
        cfg.scene.clone_in_fabric = False
        # Parent configuration has event fields; explicitly remove every randomizer.
        for name in vars(cfg.events):
            if not name.startswith("_"):
                setattr(cfg.events, name, None)
        cfg.observation_space = 45  # q, qdot, ball pose and velocity; no policy training interface.
        cfg.state_space = 0

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg.state_space = 0
        self._configure_gym_env_spaces()
        self.seed_pose = validate_seed(cfg.seed_pose)
        lower = self.hand_dof_lower_limits[0, self.actuated_dof_indices].cpu().numpy()
        upper = self.hand_dof_upper_limits[0, self.actuated_dof_indices].cpu().numpy()
        q = np.array(self.seed_pose["joint_pos"])
        if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
            raise ValueError("Saved seed joints exceed simulator joint limits; adjust them in viser")
        # Bind identical materials to robot and object shapes, including imported URDF colliders.
        for asset in (self.hand, self.object):
            materials = asset.root_physx_view.get_material_properties()
            materials[..., 0:2] = 1.0
            materials[..., 2] = 0.0
            asset.root_physx_view.set_material_properties(materials, torch.arange(self.num_envs, device="cpu"))

    def _setup_scene(self):
        self._enable_urdf_importer()
        self._patch_urdf_converter_for_local_importer()
        self.hand = Articulation(self.cfg.robot_cfg)
        if getattr(self.cfg, 'grasp_manifest', ''):
            # Clone the hand independently before spawning heterogeneous objects.
            # Inheritance from env_0 would add its object body to other URDF roots.
            self.scene.clone_environments(copy_from_source=True)
            self.object = RigidObject(self.cfg.object_cfg)
        else:
            self.object = RigidObject(self.cfg.object_cfg)
            self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.filter_collisions(global_prim_paths=[])
        if self.cfg.apply_visual_materials:
            self._apply_hand_visual_materials()
            from .object_visuals import make_visual_materials_opaque

            seeds = getattr(self.cfg, "grasp_seeds", []) or [self.cfg.seed_pose]
            stage = sim_utils.get_current_stage()
            object_name = self.cfg.object_cfg.prim_path.rsplit("/", 1)[-1]
            for index, env_path in enumerate(self.scene.env_prim_paths):
                seed = seeds[index % len(seeds)]
                if seed.get("object", {}).get("id") == "tennis_ball_60":
                    make_visual_materials_opaque(stage.GetPrimAtPath(f"{env_path}/{object_name}"))
        if getattr(self.cfg, "show_visual_ground", False):
            from pxr import Usd, UsdPhysics
            ground = sim_utils.spawn_ground_plane(
                prim_path="/World/visual_ground", cfg=sim_utils.GroundPlaneCfg(physics_material=None))
            for prim in Usd.PrimRange(ground):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)
        # The optional reference floor has no collisions; fallen objects gain no support.

    def _refresh_dr_state(self, env_ids=None):
        pass

    def _initialize_fingertip_contact_view(self):
        if self._fingertip_contact_physx_view is not None:
            return
        patterns = [f"/World/envs/env_*/Robot/{name}" for name in self.cfg.fingertip_body_names]
        object_pattern = re.sub(r'/env_\d+/', '/env_*/', self.object.root_physx_view.prim_paths[0])
        view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            patterns, filter_patterns=[[object_pattern] for _ in patterns])
        expected = self.num_envs * self.num_fingertips
        order = [-1] * expected
        for index, path in enumerate(view.sensor_paths):
            match = re.search(r"/env_(\d+)/Robot/([^/]+)$", str(path))
            if match and match[2] in self.cfg.fingertip_body_names:
                target = int(match[1]) * self.num_fingertips + self.cfg.fingertip_body_names.index(match[2])
                order[target] = index
        if view.sensor_count != expected or -1 in order:
            raise RuntimeError("Could not map fingertip-object contact sensors to environments")
        self._fingertip_contact_physx_view = view
        self._fingertip_contact_force_order = torch.tensor(order, device=self.device)

    def _get_fingertip_contact_forces_w(self):
        self._initialize_fingertip_contact_view()
        view = self._fingertip_contact_physx_view
        matrix = view.get_contact_force_matrix(dt=self.physics_dt)
        forces = matrix.reshape(view.sensor_count, -1, 3).sum(dim=1)
        return forces[self._fingertip_contact_force_order].reshape(self.num_envs, self.num_fingertips, 3)

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        self.hand.set_joint_position_target(self.cur_targets)

    def _get_dones(self):
        # Collector owns termination so it can read a terminal state before any reset.
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done.clone()

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self):
        return {"policy": torch.cat((self.hand.data.joint_pos[:, self.actuated_dof_indices],
                self.hand.data.joint_vel[:, self.actuated_dof_indices],
                self.object.data.root_pos_w - self.scene.env_origins,
                self.object.data.root_quat_w, self.object.data.root_vel_w), dim=1)}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        DirectRLEnv._reset_idx(self, env_ids)
        seed = self.seed_pose
        self.restore({"joint_pos": np.tile(seed["joint_pos"], (len(env_ids), 1)),
                      "joint_targets": np.tile(seed["joint_pos"], (len(env_ids), 1)),
                      "object_pose": np.tile(seed["object_pose"], (len(env_ids), 1))}, env_ids)

    def restore(self, states, env_ids=None):
        """Restore environment-local poses and zero velocities; targets retain contact preload."""
        ids = self.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        q = self.default_hand_joint_pos[ids].clone()
        targets = q.clone()
        q[:, self.actuated_dof_indices] = torch.as_tensor(states["joint_pos"], device=self.device, dtype=q.dtype)
        targets[:, self.actuated_dof_indices] = torch.as_tensor(states["joint_targets"], device=self.device, dtype=q.dtype)
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.cur_targets[ids] = targets
        self.prev_targets[ids] = targets
        self.hand.set_joint_position_target(targets, env_ids=ids)
        p = torch.as_tensor(states["object_pose"], device=self.device, dtype=q.dtype).clone()
        p[:, :3] += self.scene.env_origins[ids]
        self.object.write_root_pose_to_sim(p, env_ids=ids)
        self.object.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
        self.episode_length_buf[ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()

    def snapshot(self):
        tensors = dict(joint_pos=self.hand.data.joint_pos[:, self.actuated_dof_indices],
                       joint_vel=self.hand.data.joint_vel[:, self.actuated_dof_indices],
                       joint_targets=self.cur_targets[:, self.actuated_dof_indices],
                       object_pose=torch.cat((self.object.data.root_pos_w - self.scene.env_origins,
                                              self.object.data.root_quat_w), dim=1),
                       object_velocity=self.object.data.root_vel_w,
                       contact_forces=self._get_fingertip_contact_forces_w())
        return {key: value.detach().cpu().numpy().copy() for key, value in tensors.items()}

    def physics_metadata(self):
        return dict(joint_names=JOINT_NAMES, gravity=list(self.cfg.sim.gravity), physics_dt=self.physics_dt,
                    object_spawn_type=type(self.cfg.object_cfg.spawn).__name__,
                    object_rigid_body_path=self.object.root_physx_view.prim_paths[0],
                    fixed_base=True, static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                    stiffness=self.cfg.robot_cfg.actuators["fingers"].stiffness,
                    damping=self.cfg.robot_cfg.actuators["fingers"].damping,
                    effort_limit=self.cfg.robot_cfg.actuators["fingers"].effort_limit_sim,
                    velocity_limit=self.cfg.robot_cfg.actuators["fingers"].velocity_limit_sim)
