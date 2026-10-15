"""Cached-grasp HORA rotation of the 60 mm tennis ball."""
from pathlib import Path
import math
from copy import deepcopy

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .rotation_base import (
    Gx4mRotationBaseEnv, Gx4mRotationBaseEnvCfg,
)
from .super_inair_env import Gx4mSuperInAirEnv
from .grasp_data import ROOT, load_cache
from .reward_terms import grasp_stability_costs, rotation_velocity, hora_reward_terms, object_position_cost, cross_axis_orientation_cost, fingertip_contact_timeout
from .base_pose import sample_base_transform, transform_grasp, rotation_axis_world
from .object_assets import seed_object_size, seed_object_mass, normalized_single_body_urdf
from .mixed_cache import load_mixed_cache, sample_object_rows
from .object_observations import OBJECT_ID_NAMES, object_id_codes, write_object_id_observations


@configclass
class Gx4mSuperInAirRotationEnvCfg(Gx4mRotationBaseEnvCfg):
    grasp_cache = str(ROOT / "outputs/super_inair/tennis_grasps")
    grasp_manifest: str = str(ROOT / "experiment/super_inair/tennis_manifest.json")
    grasp_object_index: int = -1  # -1: all; otherwise zero-based manifest selection for playback.
    grasp_seeds: list[dict] = []
    seed_pose: dict = {}
    training_phase = "direction"
    direction_axis = "xz"
    rotation_task = "hora"
    control_mode = "implicit"  # "explicit_pd" is available for controlled comparisons.
    randomize_base_pose = False
    base_position_range_m = 0.0
    observe_base_pose = True
    observe_object_id = True  # Retain the original tennis run's six-slot observation layout.
    object_id_names: list[str] = list(OBJECT_ID_NAMES)
    hora_rotation_axis_frame = "wrist"
    hora_rotation_axis = (1.0, 0.0, 0.0)
    rotation_reward_scale = 1.0
    survival_reward_scale = 0.0  # Per control step; failure-terminal steps receive zero.
    reverse_rotation_penalty_multiplier = 1.0
    clip_reverse_rotation = True
    angular_velocity_clip_min = -0.5
    angular_velocity_clip_max = 0.5
    torque_penalty_scale = -0.1
    work_penalty_scale = -2.0
    direction_object_set = "grasp_cache"
    action_delta_rad = 0.05
    show_goal_marker = False
    show_visual_ground = False
    show_orientation_axes = True
    show_thumb_marker = False
    enable_rna = False
    force_scale = 0.0
    enable_object_size_dr = False
    enable_object_size_randomization = False
    action_noise_model = None
    observation_noise_model = None
    policy_obs_noise_model = None
    save_hand_trajectory = False
    fall_dist = 0.02
    fall_penalty = -30.0
    object_linvel_penalty_scale = -0.3
    initial_hand_pose_penalty_scale = -0.1
    object_position_penalty_scale = -0.2
    object_position_reference_m = 0.02
    # Penalize squared swing excess beyond 20 degrees, normalized by 20 degrees.
    cross_axis_penalty_scale = -0.1
    cross_axis_threshold_deg = 20.0
    fingertip_contact_timeout_s = 0.4
    fingertip_contact_force_threshold = 0.1
    fingertip_contact_timeout_penalty_scale = -0.15


class Gx4mSuperInAirRotationEnv(Gx4mRotationBaseEnv):
    _setup_scene = Gx4mSuperInAirEnv._setup_scene
    _initialize_fingertip_contact_view = Gx4mSuperInAirEnv._initialize_fingertip_contact_view
    _get_fingertip_contact_forces_w = Gx4mSuperInAirEnv._get_fingertip_contact_forces_w

    def _configure_dr_state_slices(self, object_shapes, robot_shapes, num_joints, include_object_scale):
        super()._configure_dr_state_slices(object_shapes, robot_shapes, num_joints, include_object_scale)
        # Compound URDFs have one material triplet per collider. Preserve these
        # privileged observations and reserve space for command/base/object ID.
        required = (self._full_observation_size_from_counts(num_joints, self.num_fingertips)
                    + 3 * self.num_fingertips + int(self.cfg.dr_state_space) + 3
                    + 10 + (len(self.cfg.object_id_names) if self.cfg.observe_object_id else 0))
        if required > int(self.cfg.state_space):
            self.cfg.state_space = required
            self._configure_gym_env_spaces()

    @staticmethod
    def _apply_hand_config(cfg):
        Gx4mSuperInAirEnv._apply_hand_config(cfg)
        if cfg.grasp_manifest:
            assets = []
            for seed in cfg.grasp_seeds:
                per_object = deepcopy(cfg)
                per_object.seed_pose = seed
                Gx4mSuperInAirEnv._apply_hand_config(per_object)
                spawn = per_object.object_cfg.spawn
                if isinstance(spawn, sim_utils.UrdfFileCfg):
                    directory = ROOT / 'outputs/super_inair/object_usd_mixed_v2' / seed['object']['id']
                    spawn.asset_path = str(normalized_single_body_urdf(spawn.asset_path, directory/'source.urdf'))
                    spawn.usd_dir = str(directory/'usd')
                assets.append(spawn)
            cfg.object_cfg.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets, random_choice=False)
            cfg.scene.replicate_physics = False
        if cfg.control_mode == "explicit_pd":
            previous = cfg.robot_cfg.actuators["fingers"]
            cfg.robot_cfg.actuators["fingers"] = IdealPDActuatorCfg(
                joint_names_expr=previous.joint_names_expr,
                stiffness=previous.stiffness, damping=previous.damping,
                effort_limit=previous.effort_limit_sim, effort_limit_sim=previous.effort_limit_sim,
                velocity_limit_sim=previous.velocity_limit_sim,
                armature=previous.armature, friction=previous.friction,
            )
        cfg.observation_space, cfg.state_space = 256, 300
        cfg.goal_object_cfg.markers = {"goal": sim_utils.SphereCfg(
            radius=seed_object_size(cfg.seed_pose) / 2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.65, 0.8)),
        )}
        if cfg.show_goal_marker:
            cfg.goal_marker_pos = [value + offset for value, offset in
                                   zip(cfg.seed_pose["object_pose"][:3], (.16, .08, 0.0))]

    def __init__(self, cfg, **kwargs):
        if not math.isfinite(cfg.cross_axis_threshold_deg) or not 0 < cfg.cross_axis_threshold_deg < 180:
            raise ValueError("cross_axis_threshold_deg must be between 0 and 180")
        if not math.isfinite(cfg.cross_axis_penalty_scale) or cfg.cross_axis_penalty_scale > 0:
            raise ValueError("cross_axis_penalty_scale must be finite and nonpositive")
        if not math.isfinite(cfg.object_position_reference_m) or cfg.object_position_reference_m <= 0:
            raise ValueError("object_position_reference_m must be finite and positive")
        if not math.isfinite(cfg.object_position_penalty_scale) or cfg.object_position_penalty_scale > 0:
            raise ValueError("object_position_penalty_scale must be finite and nonpositive")
        if not math.isfinite(cfg.base_position_range_m) or cfg.base_position_range_m < 0:
            raise ValueError("base_position_range_m must be finite and nonnegative")
        if cfg.control_mode not in ("explicit_pd", "implicit"):
            raise ValueError("control_mode must be explicit_pd or implicit")
        if cfg.rotation_task != "hora":
            raise ValueError("Only HORA rotation is supported")
        if cfg.hora_rotation_axis_frame not in ("world", "wrist"):
            raise ValueError("hora_rotation_axis_frame must be world or wrist")
        axis_tensor = torch.as_tensor(cfg.hora_rotation_axis, dtype=torch.float)
        if axis_tensor.shape != (3,) or not torch.isfinite(axis_tensor).all() or axis_tensor.norm() < 1e-8:
            raise ValueError("hora_rotation_axis must be a finite nonzero 3-vector")
        if cfg.angular_velocity_clip_min >= cfg.angular_velocity_clip_max:
            raise ValueError("Invalid angular velocity clip range")
        if cfg.training_phase != "direction" or cfg.direction_axis not in ("x", "z", "xz"):
            raise ValueError("SuperInAir supports direction training with x, z or xz")
        if cfg.grasp_object_index != -1 and not cfg.grasp_manifest:
            raise ValueError("Object index selection requires a mixed grasp manifest")
        self.grasp_object_ids = []
        if cfg.grasp_manifest:
            self.grasp_object_ids, seeds, states, offsets, counts = load_mixed_cache(cfg.grasp_manifest, cfg.grasp_object_index)
            cfg.grasp_seeds = seeds
            self._object_cache_offsets = offsets
            self._object_cache_counts = counts
            cfg.direction_object_ids = self.grasp_object_ids
            metadata = {'seed_pose': seeds[0]}
        else:
            states, metadata = load_cache(Path(cfg.grasp_cache))
        if len(states["joint_pos"]) == 0:
            raise ValueError("Grasp cache is empty")
        cfg.seed_pose = metadata["seed_pose"]
        if cfg.seed_pose.get("object", {}).get("id") != "tennis_ball_60":
            raise ValueError("Only tennis_ball_60 is supported")
        self.seed_pose = cfg.seed_pose
        self._cache_numpy = states
        self._cache_tensors = None
        axis = cfg.direction_axis
        # Preserve the original direction initialization and observation layout.
        cfg.direction_axis = "z" if axis == "xz" else axis
        super().__init__(cfg, **kwargs)
        cfg.direction_axis = axis
        if axis == "xz":
            self.direction_axis_indices = (torch.arange(self.num_envs, device=self.device) % 2) * 2
        for asset in (self.hand, self.object):
            material = asset.root_physx_view.get_material_properties()
            material[..., :2], material[..., 2] = 1.0, 0.0
            asset.root_physx_view.set_material_properties(material, torch.arange(self.num_envs, device="cpu"))
        if self.grasp_object_ids:
            # Stop on a wrong asset/order instead of silently training/evaluating it.
            actual = self.object.root_physx_view.get_masses().flatten().cpu()
            expected = torch.tensor([seed_object_mass(cfg.grasp_seeds[i % len(cfg.grasp_seeds)])
                                     for i in range(self.num_envs)])
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)
            from pxr import Usd, UsdPhysics
            stage = sim_utils.get_current_stage()
            for i, root in enumerate(self.scene.env_prim_paths):
                prim = stage.GetPrimAtPath(root + '/object')
                bodies = [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
                if bodies != [root + '/object/object']:
                    raise RuntimeError(f'Object rigid-body identity mismatch in env {i}: {bodies}')
            print('[super_inair] Verified one object body per environment and all assigned masses', flush=True)
        self._reset_idx(None)
        self.object_orientation_axes = self.goal_orientation_axes = None
        if cfg.show_goal_marker and cfg.show_orientation_axes:
            from .orientation_axes import OrientationAxes
            length = seed_object_size(cfg.seed_pose) * 0.9
            self.object_orientation_axes = OrientationAxes("/Visuals/super_inair/object_axes", length, self.device)
            self.goal_orientation_axes = OrientationAxes("/Visuals/super_inair/goal_axes", length, self.device)
            self._update_orientation_axes()
        if cfg.show_goal_marker:
            # Same viewing side as inhand_reorient, centered on this saved hand
            # pose and target rather than the old task's hard-coded world origin.
            points = torch.cat((self.hand.data.root_pos_w, self.object.data.root_pos_w,
                                self.goal_marker_pos + self.scene.env_origins))
            low, high = points.amin(dim=0), points.amax(dim=0)
            center = (low + high) * .5
            zoom = max(.7, float((high - low).max()) * 1.8)
            eye = center + torch.tensor([-.4, .78, .64], device=self.device) * zoom
            self.sim.set_camera_view(eye=eye.cpu().tolist(), target=center.cpu().tolist())
        print(f"[super_inair] Training from {len(states['joint_pos'])} cached grasps; "
              f"incremental joint targets {cfg.action_delta_rad} rad/step", flush=True)
        print(f"[super_inair] task={cfg.rotation_task}; HORA {cfg.hora_rotation_axis_frame} axis={cfg.hora_rotation_axis}; "
              f"controller={cfg.control_mode}; torque costs use "
              f"{'commanded effort' if cfg.control_mode == 'explicit_pd' else 'PD estimate'}", flush=True)
        print(f"[super_inair] randomize_base_pose={cfg.randomize_base_pose}; "
              f"observe_object_id={cfg.observe_object_id}; vocabulary={cfg.object_id_names}", flush=True)

    def _sample_commands(self, env_ids, initial=False):
        self.angular_velocity_command[env_ids] = rotation_axis_world(
            self.cfg.hora_rotation_axis, self.hand.data.root_quat_w[env_ids], self.cfg.hora_rotation_axis_frame)
        self.command_timer[env_ids] = 0
        self.command_switch_event[env_ids] = 0
        self.direction_initial_object_rot[env_ids] = self.object_rot[env_ids]
        self.direction_signed_progress_angle[env_ids] = 0

    def _set_direction_waypoint(self, env_ids):
        pass

    def set_angular_velocity_command(self, command, env_ids=None):
        # Generic play scripts may request an old direction command. HORA has
        # one configured axis, so retain it instead of silently changing tasks.
        ids = self.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        self._sample_commands(ids)

    def _reset_idx(self, env_ids):
        ids = self.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        DirectRLEnv._reset_idx(self, ids)
        if not hasattr(self, "failure_penalty_applied"):
            self.failure_penalty_applied = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure_penalty_applied[ids] = False
        if not hasattr(self, "fingertip_missed_contact_steps"):
            self.fingertip_missed_contact_steps = torch.zeros(
                (self.num_envs, self.num_fingertips), dtype=torch.long, device=self.device)
        self.fingertip_missed_contact_steps[ids] = 0
        if self._cache_tensors is None:
            self._cache_tensors = {key: torch.as_tensor(value, device=self.device) for key, value in self._cache_numpy.items()}
            if self.grasp_object_ids:
                self.grasp_object_indices = torch.arange(self.num_envs, device=self.device) % len(self.grasp_object_ids)
                self.grasp_cache_offsets = torch.tensor(self._object_cache_offsets, device=self.device)
                self.grasp_cache_counts = torch.tensor(self._object_cache_counts, device=self.device)
                self.grasp_nominal_bases = torch.tensor([seed['hand_pose'] for seed in self.cfg.grasp_seeds], device=self.device)
                self.grasp_env_masks = [torch.arange(i, self.num_envs, len(self.grasp_object_ids), device=self.device)
                                        for i in range(min(self.num_envs, len(self.grasp_object_ids)))]
            if self.cfg.observe_object_id:
                names = self.grasp_object_ids or ([self.seed_pose['object']['id']] if self.seed_pose['version'] == 2 else [])
                codes = object_id_codes(names, self.cfg.object_id_names, self.device)
                indices = (self.grasp_object_indices if self.grasp_object_ids
                           else torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
                self.object_id_obs = codes[indices]
        if self.grasp_object_ids:
            rows = sample_object_rows(self.grasp_object_indices[ids], self.grasp_cache_offsets, self.grasp_cache_counts)
        else:
            rows = torch.randint(len(self._cache_tensors["joint_pos"]), (len(ids),), device=self.device)
        if not hasattr(self, 'last_grasp_cache_rows'):
            self.last_grasp_cache_rows = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_grasp_cache_rows[ids] = rows
        states = {key: value[rows] for key, value in self._cache_tensors.items()}
        if self.cfg.randomize_base_pose or self.grasp_object_ids or hasattr(self, "episode_base_pose"):
            nominal = (self.grasp_nominal_bases[self.grasp_object_indices[ids]] if self.grasp_object_ids
                       else torch.tensor(self.seed_pose["hand_pose"], device=self.device))
            if not hasattr(self, "episode_base_pose"):
                self.episode_base_pose = torch.zeros((self.num_envs, 7), device=self.device)
            if self.cfg.randomize_base_pose:
                translation, delta = sample_base_transform(len(ids), self.cfg.base_position_range_m, self.device)
            else:
                translation = torch.zeros((len(ids), 3), device=self.device)
                delta = torch.zeros((len(ids), 4), device=self.device)
                delta[:, 0] = 1
            base_pose, object_pose = transform_grasp(nominal, states["object_pose"], translation, delta)
            self.episode_base_pose[ids] = base_pose
            base_world = base_pose.clone()
            base_world[:, :3] += self.scene.env_origins[ids]
            self.hand.write_root_pose_to_sim(base_world, env_ids=ids)
            self.hand.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
            states["object_pose"] = object_pose
        if not hasattr(self, "episode_initial_joint_pos"):
            self.episode_initial_joint_pos = torch.zeros(
                (self.num_envs, len(self.actuated_dof_indices)), device=self.device)
            self.reward_previous_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.episode_initial_joint_pos[ids] = states["joint_pos"]
        self.reward_previous_object_pos[ids] = states["object_pose"][:, :3]
        q = self.default_hand_joint_pos[ids].clone()
        targets = q.clone()
        q[:, self.actuated_dof_indices] = states["joint_pos"]
        targets[:, self.actuated_dof_indices] = states["joint_targets"]
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.cur_targets[ids] = self.prev_targets[ids] = targets
        self.hand.set_joint_position_target(targets, env_ids=ids)
        p = states["object_pose"].clone()
        if not hasattr(self, "episode_initial_object_quat"):
            self.episode_initial_object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.episode_initial_object_quat[ids] = p[:, 3:7]
        p[:, :3] += self.scene.env_origins[ids]
        self.object.write_root_pose_to_sim(p, env_ids=ids)
        self.object.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
        self.in_hand_pos[ids] = states["object_pose"][:, :3]
        self.goal_pos[ids] = self.in_hand_pos[ids]
        if not hasattr(self, "episode_initial_object_com_pos"):
            self.episode_initial_object_com_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Compute directly from the reset pose: COM pose buffers can still contain
        # the pre-reset frame until the next simulation step.
        self.episode_initial_object_com_pos[ids] = states["object_pose"][:, :3] + quat_apply(
            states["object_pose"][:, 3:7], self.object.data.body_com_pos_b[ids, 0])
        self.prev_hand_dof_pos[ids] = states["joint_pos"]
        self.actions[ids] = 0
        self.successes[ids] = 0
        self.reset_goal_buf[ids] = False
        self._refresh_dr_state(ids)
        self._compute_intermediate_values()
        self.goal_rot[ids] = self.object_rot[ids]
        if hasattr(self, "angular_velocity_command"):
            self._sample_commands(ids, initial=True)

    def _pre_physics_step(self, actions):
        self.reward_previous_object_pos[:] = self.object.data.root_pos_w - self.scene.env_origins
        self.reward_previous_object_rot = self.object.data.root_quat_w.clone()
        super()._pre_physics_step(actions)
        indices = self.actuated_dof_indices
        targets = self.prev_targets[:, indices] + self.cfg.action_delta_rad * self.actions
        self.cur_targets[:, indices] = torch.clamp(
            targets, self.hand_dof_lower_limits[:, indices], self.hand_dof_upper_limits[:, indices])
        self.prev_targets[:] = self.cur_targets

    def _apply_action(self):
        self.reward_previous_joint_pos = self.hand.data.joint_pos[:, self.actuated_dof_indices].clone()
        self.hand.set_joint_position_target(self.cur_targets)

    def _position_failure_penalty(self):
        exceeded = (self.object_pos - self.in_hand_pos).norm(dim=-1) >= float(self.cfg.fall_dist)
        apply = exceeded & ~self.failure_penalty_applied
        self.failure_penalty_applied |= apply
        penalty = torch.where(apply, torch.as_tensor(float(self.cfg.fall_penalty), device=self.device), 0.0)
        self.extras.setdefault("log", {}).update({
            "reward_failure": penalty.mean(),
            "position_failure_rate": apply.float().mean(),
        })
        return penalty

    def _contact_timeout_penalty(self):
        self.fingertip_missed_contact_steps, overdue = fingertip_contact_timeout(
            self.fingertip_missed_contact_steps, self.fingertip_contact_forces_w,
            self.control_dt, self.cfg.fingertip_contact_timeout_s,
            self.cfg.fingertip_contact_force_threshold)
        penalty = self.cfg.fingertip_contact_timeout_penalty_scale * overdue.sum(-1)
        self.extras.setdefault("log", {}).update({
            "reward_fingertip_contact_timeout": penalty.mean(),
            "fingertip_contact_timeout_count": overdue.float().sum(-1).mean(),
            **{f"fingertip_contact_timeout/{name}": overdue[:, i].float().mean()
               for i, name in enumerate(self.cfg.fingertip_body_names)},
        })
        return penalty

    def _get_rewards(self):
        return self._get_hora_rewards()

    def _get_hora_rewards(self):
        self._compute_intermediate_values()
        velocity = rotation_velocity(self.object_rot, self.reward_previous_object_rot, self.control_dt)
        axis = self.angular_velocity_command
        signed_speed = (velocity * axis).sum(-1)
        if self._direction_progress_last_step != int(self.common_step_counter):
            self.direction_signed_progress_angle += signed_speed * self.control_dt
            self._direction_progress_last_step = int(self.common_step_counter)
        linear_speed, pose_deviation = grasp_stability_costs(
            self.object_pos, self.reward_previous_object_pos, self.control_dt,
            self.hand_dof_pos[:, self.actuated_dof_indices], self.episode_initial_joint_pos)
        joint_velocity = (self.hand.data.joint_pos[:, self.actuated_dof_indices]
                          - self.reward_previous_joint_pos) / self.physics_dt
        torque = self.hand.data.applied_torque[:, self.actuated_dof_indices]
        terms = hora_reward_terms(velocity, axis, linear_speed, pose_deviation, torque, joint_velocity,
                                 self.cfg.angular_velocity_clip_min, self.cfg.angular_velocity_clip_max,
                                 self.cfg.reverse_rotation_penalty_multiplier, self.cfg.clip_reverse_rotation)
        # Retain pre-reset step metrics for deterministic first-episode evaluation.
        self.hora_step_metrics = dict(signed_speed=signed_speed, linear_speed=linear_speed,
                                     torque_squared=terms["torque"], power_squared=terms["work"])
        scales = dict(rotation=self.cfg.rotation_reward_scale,
                      object_linvel=self.cfg.object_linvel_penalty_scale,
                      initial_hand_pose=self.cfg.initial_hand_pose_penalty_scale,
                      torque=self.cfg.torque_penalty_scale, work=self.cfg.work_penalty_scale)
        weighted = {key: value * scales[key] for key, value in terms.items()}
        position_cost, com_position_error = object_position_cost(
            self.object.data.root_com_pos_w - self.scene.env_origins,
            self.episode_initial_object_com_pos, self.cfg.object_position_reference_m)
        weighted["object_position"] = self.cfg.object_position_penalty_scale * position_cost
        cross_axis_cost, cross_axis_angle = cross_axis_orientation_cost(
            self.object_rot, self.episode_initial_object_quat, axis, self.cfg.cross_axis_threshold_deg)
        if self.cfg.cross_axis_penalty_scale != 0:
            weighted["cross_axis"] = self.cfg.cross_axis_penalty_scale * cross_axis_cost
        weighted["fingertip_contact_timeout"] = self._contact_timeout_penalty()
        weighted["failure"] = self._position_failure_penalty()
        weighted["survival"] = (~self.reset_terminated).float() * self.cfg.survival_reward_scale
        reward = sum(weighted.values())
        self.reset_goal_buf[:] = False
        self.extras.setdefault("log", {}).update(
            {"reward": reward.mean(), "rotation_speed_rad_s": signed_speed.mean(),
             "net_rotation_turns": (self.direction_signed_progress_angle / (2 * torch.pi)).mean(),
             "object_linear_speed_l1": linear_speed.mean(),
             "initial_hand_pose_error_squared": pose_deviation.mean(),
             "object_com_displacement_m": com_position_error.mean(),
             **({"cross_axis_deviation_deg": torch.rad2deg(cross_axis_angle).mean(),
                "cross_axis_exceed_rate": (cross_axis_angle > math.radians(self.cfg.cross_axis_threshold_deg)).float().mean()}
               if self.cfg.cross_axis_penalty_scale != 0 else {}),
             **{f"reward_{key}": value.mean() for key, value in weighted.items()}})
        for i, (object_id, mask) in enumerate(zip(self.grasp_object_ids, getattr(self, 'grasp_env_masks', []))):
            if i < self.num_envs:
                self.extras['log'].update({
                    f'objects/{object_id}/reward': reward[mask].mean(),
                    f'objects/{object_id}/rotation_speed': signed_speed[mask].mean(),
                    f'objects/{object_id}/net_turns': (self.direction_signed_progress_angle[mask]/(2*torch.pi)).mean(),
                })
        return reward

    def _update_orientation_axes(self):
        if getattr(self, "object_orientation_axes", None) is not None:
            self.object_orientation_axes.update(self.object.data.root_pos_w, self.object.data.root_quat_w)
            self.goal_orientation_axes.update(self.goal_marker_pos + self.scene.env_origins, self.goal_rot)

    def _get_observations(self):
        observations = super()._get_observations()
        if self.cfg.rotation_task == "hora":
            # A wrist axis is generally not world-axis-aligned. Preserve its
            # full unit direction instead of the parent's component-wise sign.
            observations["policy"][:, 70:73] = self.angular_velocity_command
        if self.cfg.observe_base_pose:
            # Existing padding after the 3D command, preserving network dimensions.
            base_pose = self.hand.data.root_state_w[:, :7].clone()
            base_pose[:, :3] -= self.scene.env_origins
            base_pose[:, 3:] = torch.where(base_pose[:, 3:4] < 0, -base_pose[:, 3:], base_pose[:, 3:])
            observations["policy"][:, 73:80] = base_pose
            start = self._critic_command_obs_start + 3
            if start + 7 > observations["critic"].shape[-1]:
                raise RuntimeError("No critic padding available for base pose")
            observations["critic"][:, start:start+7] = base_pose
        if self.cfg.observe_object_id:
            write_object_id_observations(observations, self.object_id_obs, self._critic_command_obs_start)
        self._update_orientation_axes()
        return observations
