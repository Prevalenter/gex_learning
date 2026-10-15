from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import omni.kit.app
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from gex_learning.tasks.direct.dexanytwist.dataset import HandSpec, load_hand_spec
from gex_learning.tasks.direct.hand_trajectory_recorder import HandTrajectorySampler
from .initial_pose import DEFAULT_PATH as INITIAL_POSE_PATH, apply_initial_pose


def _count_urdf_collision_shapes(urdf_path: Path) -> int:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return 1
    return max(1, len(root.findall(".//collision")))


def _dr_state_size(object_shapes: int, robot_shapes: int, num_joints: int, include_object_scale: bool = True) -> int:
    return 1 + 3 * object_shapes + 3 * robot_shapes + num_joints + num_joints + 2 + 3 + 3 * include_object_scale


def _object_size_range_to_usd_scale(
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
) -> tuple[float, float] | dict[str, tuple[float, float]]:
    if isinstance(scale_range, dict):
        base_by_axis = {"x": DEXCUBE_SCALE[0], "y": DEXCUBE_SCALE[1], "z": DEXCUBE_SCALE[2]}
        return {
            axis: (bounds[0] * base_by_axis[axis], bounds[1] * base_by_axis[axis])
            for axis, bounds in scale_range.items()
        }
    if DEXCUBE_SCALE[0] == DEXCUBE_SCALE[1] == DEXCUBE_SCALE[2]:
        return (scale_range[0] * DEXCUBE_SCALE[0], scale_range[1] * DEXCUBE_SCALE[0])
    return {
        "x": (scale_range[0] * DEXCUBE_SCALE[0], scale_range[1] * DEXCUBE_SCALE[0]),
        "y": (scale_range[0] * DEXCUBE_SCALE[1], scale_range[1] * DEXCUBE_SCALE[1]),
        "z": (scale_range[0] * DEXCUBE_SCALE[2], scale_range[1] * DEXCUBE_SCALE[2]),
    }


DEFAULT_HAND = load_hand_spec("gx4m")
REPO_ROOT = Path(__file__).resolve().parents[6]
OBJECT_SIZE = (0.06, 0.06, 0.06)
DEXCUBE_SCALE = (1, 1, 1) # (0.85, 0.85, 0.85)
DEFAULT_OBJECT_DR_SHAPES = 1
DEFAULT_ROBOT_DR_SHAPES = _count_urdf_collision_shapes(DEFAULT_HAND.urdf_path)
DR_STATE_SPACE = _dr_state_size(DEFAULT_OBJECT_DR_SHAPES, DEFAULT_ROBOT_DR_SHAPES, DEFAULT_HAND.num_actions)
OBJECT_EXTERNAL_FORCE_STATE_SPACE = 3
DEXCUBE_USD_LOCAL = REPO_ROOT / "asset" / "nucleus_dexcube" / "dex_cube_instanceable.usd"
DEXCUBE_USD_URL = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
DEXCUBE_USD_PATH = DEXCUBE_USD_LOCAL.as_posix() if DEXCUBE_USD_LOCAL.exists() else DEXCUBE_USD_URL


def _quat_from_rot_xyz(rot_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles in radians to a quaternion in IsaacLab's wxyz order."""
    roll, pitch, yaw = rot_xyz
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    return (
        float(cy * cr * cp + sy * sr * sp),
        float(cy * sr * cp - sy * cr * sp),
        float(cy * cr * sp + sy * sr * cp),
        float(sy * cr * cp - cy * sr * sp),
    )


# GX4M palm/finger-pad side is local -Y. Rotate about X by -pi / 2 so it faces world +Z.
HAND_ROT_PALM_UP_ROT_XYZ = (-np.pi / 2.0 * 1, -np.pi / 2.0, 0.0)
HAND_ROT_PALM_UP = _quat_from_rot_xyz(HAND_ROT_PALM_UP_ROT_XYZ)
HAND_START_POS = (0.0, 0.0, 0.55)
OBJECT_START_POS = (0.01, 0.1, 0.6)
GOAL_MARKER_POS = (0.1, -0.09, 0.72)
GOAL_ROTATION_AXES = ("xyz", "x", "y", "z")
GOAL_MODES = ("random", "continuous_ccw")
INHAND_DEFAULT_JOINT_POS_DEG_BY_NAME = {
    "joint1": 20,
    "joint2": 80.0,
    "joint3": 10,
    "joint4": 10,
    "joint5": 0.0,
    "joint6": 10,
    "joint7": 10,
    "joint8": 10,
    "joint9": 0.0,
    "joint10": 10,
    "joint11": 10,
    "joint12": 10,
    "joint13": 10,
    "joint14": 0.0,
    "joint15": 10,
    "joint16": 10,
}


def _default_inhand_joint_pos(spec: HandSpec) -> list[float]:
    missing = sorted(set(spec.actuated_joint_names) - set(INHAND_DEFAULT_JOINT_POS_DEG_BY_NAME))
    if missing:
        raise ValueError(f"Missing in-hand default joint positions for: {missing}")
    return [float(np.deg2rad(INHAND_DEFAULT_JOINT_POS_DEG_BY_NAME[name])) for name in spec.actuated_joint_names]


class RandomNetworkAdversary(nn.Module):
    def __init__(self, num_envs: int, in_dims: int, out_dims: int, softmax_bins: int, device: str):
        super().__init__()
        self.num_envs = num_envs
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.softmax_bins = softmax_bins
        self.device = device
        self.num_feats1 = 512
        self.num_feats2 = 1024

        self.fc1 = nn.Linear(in_dims, self.num_feats1).to(self.device)
        self.fc1_1 = nn.Linear(self.num_feats1, self.num_feats1).to(self.device)
        self.fc2 = nn.Linear(self.num_feats1, self.num_feats2).to(self.device)
        self.fc2_1 = nn.Linear(self.num_feats2, self.num_feats2).to(self.device)
        self.fc3 = nn.Linear(self.num_feats2, out_dims * softmax_bins).to(self.device)

        self.dropout_masks1 = torch.ones((self.num_envs, self.num_feats1), device=self.device)
        self.dropout_masks2 = torch.ones((self.num_envs, self.num_feats2), device=self.device)
        self._refresh()

    def _refresh(self) -> None:
        self._init_weights()
        self.eval()
        self.refresh_dropout_masks()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc1_1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc2_1.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def refresh_dropout_masks(self) -> None:
        dropout_probs = torch.rand((2,), device=self.device)
        self.dropout_masks1 = torch.bernoulli(
            torch.ones((self.num_envs, self.num_feats1), device=self.device) * dropout_probs[0]
        )
        self.dropout_masks2 = torch.bernoulli(
            torch.ones((self.num_envs, self.num_feats2), device=self.device) * dropout_probs[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1_1(x)
        x = self.dropout_masks1 * x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2_1(x)
        x = self.dropout_masks2 * x
        x = self.fc3(x)
        x = x.view(-1, self.out_dims, self.softmax_bins)
        return F.softmax(x, dim=-1)


@configclass
class EventCfg:
    """Domain randomization using Isaac Lab's official event terms."""

    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": _object_size_range_to_usd_scale((0.9, 1.1)),
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.0, 0.01),
            "upper_limit_distribution_params": (0.0, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(24.0, 24.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class Gx4mInHandReorientEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0
    action_space = DEFAULT_HAND.num_actions
    observation_space = 47
    object_dr_shapes = DEFAULT_OBJECT_DR_SHAPES
    robot_dr_shapes = DEFAULT_ROBOT_DR_SHAPES
    dr_state_space = DR_STATE_SPACE
    state_space = (
        124
        + 3 * len(DEFAULT_HAND.fingertip_body_names)
        + dr_state_space
        + OBJECT_EXTERNAL_FORCE_STATE_SPACE
    )
    viewer = ViewerCfg(
        eye=(-0.4, 0.7, 1.2),
        lookat=(0.0, -0.08, 0.56),
        origin_type="world",
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2, gpu_max_rigid_patch_count=2**19),
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(DEFAULT_HAND.urdf_path),
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=DEFAULT_HAND.stiffness,
                    damping=DEFAULT_HAND.damping,
                ),
            ),
            collider_type="convex_hull",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_START_POS,
            rot=HAND_ROT_PALM_UP,
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(DEFAULT_HAND.actuated_joint_names),
                effort_limit_sim=DEFAULT_HAND.effort_limit,
                velocity_limit_sim=DEFAULT_HAND.velocity_limit,
                stiffness=DEFAULT_HAND.stiffness,
                damping=DEFAULT_HAND.damping,
            )
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEXCUBE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=450.0),
            scale=DEXCUBE_SCALE,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/gx4m_reorient_goal",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=DEXCUBE_USD_PATH,
                scale=DEXCUBE_SCALE,
            )
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    hand_name = DEFAULT_HAND.name
    actuated_joint_names = list(DEFAULT_HAND.actuated_joint_names)
    fingertip_body_names = list(DEFAULT_HAND.fingertip_body_names)
    fingertip_keypoint_offset = [list(offset) for offset in DEFAULT_HAND.fingertip_keypoint_offset]
    default_hand_joint_pos = _default_inhand_joint_pos(DEFAULT_HAND)
    initial_pose_file: str = str(INITIAL_POSE_PATH)
    use_initial_pose_config = True
    save_hand_trajectory = False
    hand_trajectory_dir = "experiment/inhand_reorient/hand_trajectories"
    hand_trajectory_task_name = "inhand_reorient"
    hand_trajectory_interval = 1000
    hand_trajectory_max_steps = 0
    hand_trajectory_quit_on_complete = False
    apply_visual_materials = False
    show_goal_marker = True
    enable_object_size_dr = True
    enable_object_size_randomization = True
    object_size_scale_range = (0.82, 0.88)
    hand_start_pos = list(HAND_START_POS)
    hand_start_rot = list(HAND_ROT_PALM_UP)
    object_start_pos = list(OBJECT_START_POS)
    object_start_rot = (1.0, 0.0, 0.0, 0.0)
    goal_marker_pos = list(GOAL_MARKER_POS)
    goal_rotation_axes = "xyz"
    goal_mode = "random"
    continuous_goal_step = 0.5 * np.pi
    # Random targets require the actor to observe the object and goal poses.
    # Joint/action-only observations are retained for legacy Z-spin checkpoints.
    policy_joint_action_only = False

    reset_position_noise = 0.01
    reset_object_rotation_noise = np.pi
    reset_position_noise_z = 0.0
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    ccw_angvel_reward_scale = 0.25
    clockwise_angvel_penalty_scale = -0.5
    action_penalty_scale = -0.0002
    reach_goal_bonus = 800.0
    fall_penalty = -5.0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.25
    goal_rotation_scale = 1.0
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 0.2
    action_residual_scale = 1.0
    force_scale = 2.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08
    enable_rna = True
    rna_action_perturb_prob = 0.15
    rna_weight_sample_freq = 1000
    rna_softmax_bins = 32
    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
    )


class Gx4mInHandReorientEnv(DirectRLEnv):
    cfg: Gx4mInHandReorientEnvCfg

    def __init__(self, cfg: Gx4mInHandReorientEnvCfg, render_mode: str | None = None, **kwargs):
        self.hand_trajectory_sampler = None
        if cfg.goal_rotation_axes not in GOAL_ROTATION_AXES:
            choices = ", ".join(GOAL_ROTATION_AXES)
            raise ValueError(f"goal_rotation_axes must be one of: {choices}; got {cfg.goal_rotation_axes!r}")
        if cfg.goal_mode not in GOAL_MODES:
            choices = ", ".join(GOAL_MODES)
            raise ValueError(f"goal_mode must be one of: {choices}; got {cfg.goal_mode!r}")
        if cfg.goal_mode == "random" and cfg.policy_joint_action_only:
            raise ValueError(
                "Random orientation goals require object and goal observations. "
                "Set env.policy_joint_action_only=False; joint/action-only policies cannot observe the target."
            )
        if cfg.goal_mode == "continuous_ccw" and cfg.goal_rotation_axes != "z":
            raise ValueError(
                "Cubic continuous_ccw supports only the Z axis. "
                "For XYZ targets use env.goal_mode=random and env.policy_joint_action_only=False."
            )
        if cfg.continuous_goal_step <= 0.0:
            raise ValueError(f"continuous_goal_step must be positive; got {cfg.continuous_goal_step}")
        self._apply_hand_config(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints
        self.actuated_dof_indices = [self._find_name(self.hand.joint_names, name) for name in cfg.actuated_joint_names]
        self.finger_bodies = [self._find_name(self.hand.body_names, name) for name in cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)
        self._fingertip_contact_physx_view = None
        self._fingertip_contact_force_order = None
        self.object_dr_shapes = int(self.object.root_physx_view.max_shapes)
        self.robot_dr_shapes = int(self.hand.root_physx_view.max_shapes)
        runtime_dr_state_space = _dr_state_size(
            self.object_dr_shapes,
            self.robot_dr_shapes,
            len(self.actuated_dof_indices),
            cfg.enable_object_size_dr,
        )
        if runtime_dr_state_space != cfg.dr_state_space:
            cfg.object_dr_shapes = self.object_dr_shapes
            cfg.robot_dr_shapes = self.robot_dr_shapes
            cfg.dr_state_space = runtime_dr_state_space
            cfg.state_space = (
                self._full_observation_size_from_counts(
                    len(self.actuated_dof_indices), len(cfg.fingertip_body_names)
                )
                + 3 * len(cfg.fingertip_body_names)
                + cfg.dr_state_space
                + OBJECT_EXTERNAL_FORCE_STATE_SPACE
            )
            self._configure_gym_env_spaces()
        self._configure_dr_state_slices(
            self.object_dr_shapes, self.robot_dr_shapes, len(self.actuated_dof_indices), cfg.enable_object_size_dr
        )

        joint_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_limits[..., 0]
        self.hand_dof_upper_limits = joint_limits[..., 1]

        default_hand_joint_pos = torch.tensor(
            self._default_joint_pos_in_articulation_order(cfg), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.default_hand_joint_pos = torch.clamp(
            default_hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        self.actions = torch.zeros((self.num_envs, cfg.action_space), dtype=torch.float, device=self.device)
        self.last_rna_action = torch.zeros_like(self.actions)
        self.rna_alpha = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.rna_perturb_mask = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        # Effective RNA perturb probability. The base task keeps this fixed at the configured
        # value; curriculum subclasses overwrite it once per policy step.
        self.rna_perturb_prob = float(cfg.rna_action_perturb_prob)
        self.discretized_dof_targets = torch.zeros(
            (self.num_envs, len(self.actuated_dof_indices), cfg.rna_softmax_bins),
            dtype=torch.float,
            device=self.device,
        )
        self.rna_network = (
            RandomNetworkAdversary(
                num_envs=self.num_envs,
                in_dims=len(self.actuated_dof_indices) + 7,
                out_dims=len(self.actuated_dof_indices),
                softmax_bins=cfg.rna_softmax_bins,
                device=self.device,
            )
            if cfg.enable_rna
            else None
        )
        self.random_force_prob = self._sample_random_force_prob(self.num_envs)
        self.object_external_forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
        self.object_external_torques = torch.zeros_like(self.object_external_forces)
        self.default_object_mass = self.object.data.default_mass.to(self.device).clone()
        self.object_mass = self.default_object_mass.clone()
        self.default_gravity = torch.tensor(self.cfg.sim.gravity, dtype=torch.float, device=self.device)
        self.object_scale_xyz = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.dr_state = torch.zeros((self.num_envs, cfg.dr_state_space), dtype=torch.float, device=self.device)
        self._refresh_dr_state()

        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.03
        self.goal_marker_pos = torch.tensor(cfg.goal_marker_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = self.goal_marker_pos.clone()
        self.goal_markers = VisualizationMarkers(cfg.goal_object_cfg) if cfg.show_goal_marker else None

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self._compute_intermediate_values()
        if cfg.save_hand_trajectory:
            self.hand_trajectory_sampler = HandTrajectorySampler(
                self,
                task_name=cfg.hand_trajectory_task_name,
                output_dir=cfg.hand_trajectory_dir,
                chunk_size=int(cfg.hand_trajectory_interval),
                actuated_joint_names=list(cfg.actuated_joint_names),
                actuated_joint_indices=self.actuated_dof_indices,
                fingertip_body_names=list(cfg.fingertip_body_names),
                fingertip_offsets=cfg.fingertip_keypoint_offset,
                run_label=cfg.hand_trajectory_task_name,
                max_frames=int(cfg.hand_trajectory_max_steps),
            )

    @staticmethod
    def _apply_hand_config(cfg: Gx4mInHandReorientEnvCfg) -> None:
        spec = load_hand_spec(cfg.hand_name)
        cfg.hand_name = spec.name
        apply_initial_pose(cfg, spec.actuated_joint_names)
        cfg.robot_cfg.spawn.asset_path = str(spec.urdf_path)
        cfg.robot_cfg.init_state.pos = cfg.hand_start_pos
        cfg.robot_cfg.init_state.rot = cfg.hand_start_rot
        cfg.object_cfg.init_state.pos = cfg.object_start_pos
        cfg.object_cfg.init_state.rot = cfg.object_start_rot
        cfg.actuated_joint_names = list(spec.actuated_joint_names)
        cfg.fingertip_body_names = list(spec.fingertip_body_names)
        cfg.fingertip_keypoint_offset = [list(offset) for offset in spec.fingertip_keypoint_offset]
        if len(cfg.default_hand_joint_pos) != spec.num_actions:
            raise ValueError("default_hand_joint_pos must match the actuated joints")
        cfg.action_space = spec.num_actions
        cfg.observation_space = Gx4mInHandReorientEnv._observation_size(
            spec, joint_action_only=cfg.policy_joint_action_only
        )
        cfg.object_dr_shapes = DEFAULT_OBJECT_DR_SHAPES
        cfg.robot_dr_shapes = _count_urdf_collision_shapes(spec.urdf_path)
        cfg.events.object_scale = None
        if cfg.enable_object_size_dr and cfg.enable_object_size_randomization:
            cfg.scene.replicate_physics = False
            cfg.scene.clone_in_fabric = False
            cfg.events.object_scale = EventTerm(
                func=mdp.randomize_rigid_body_scale,
                mode="prestartup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "scale_range": _object_size_range_to_usd_scale(cfg.object_size_scale_range),
                },
            )
        cfg.dr_state_space = _dr_state_size(
            cfg.object_dr_shapes, cfg.robot_dr_shapes, spec.num_actions, cfg.enable_object_size_dr
        )
        cfg.state_space = (
            Gx4mInHandReorientEnv._full_observation_size(spec)
            + 3 * len(spec.fingertip_body_names)
            + cfg.dr_state_space
            + OBJECT_EXTERNAL_FORCE_STATE_SPACE
        )
        cfg.robot_cfg.spawn.activate_contact_sensors = True

        actuator = cfg.robot_cfg.actuators["fingers"]
        actuator.joint_names_expr = list(spec.actuated_joint_names)
        actuator.effort_limit_sim = spec.effort_limit
        actuator.velocity_limit_sim = spec.velocity_limit
        actuator.stiffness = spec.stiffness
        actuator.damping = spec.damping

        joint_gains = cfg.robot_cfg.spawn.joint_drive.gains
        joint_gains.stiffness = spec.stiffness
        joint_gains.damping = spec.damping

    @staticmethod
    def _observation_size(spec: HandSpec, joint_action_only: bool = False) -> int:
        if joint_action_only:
            return 2 * spec.num_actions
        return 2 * spec.num_actions + 3 + 4 + 4 + 4

    @staticmethod
    def _full_observation_size(spec: HandSpec) -> int:
        return Gx4mInHandReorientEnv._full_observation_size_from_counts(
            spec.num_actions, len(spec.fingertip_body_names)
        )

    @staticmethod
    def _full_observation_size_from_counts(num_actions: int, num_fingertips: int) -> int:
        return (
            2 * num_actions
            + 3
            + 4
            + 3
            + 3
            + 3
            + 4
            + 4
            + 13 * num_fingertips
            + num_actions
        )

    @staticmethod
    def _find_name(names: Sequence[str], target: str) -> int:
        for i, name in enumerate(names):
            if name == target or name.endswith(target):
                return i
        raise ValueError(f"Could not find '{target}' in names: {names}")

    def _default_joint_pos_in_articulation_order(self, cfg: Gx4mInHandReorientEnvCfg) -> list[float]:
        joint_pos_by_name = dict(zip(cfg.actuated_joint_names, cfg.default_hand_joint_pos))
        default_joint_pos = []
        for joint_name in self.hand.joint_names:
            matched_value = 0.0
            for cfg_joint_name, value in joint_pos_by_name.items():
                if joint_name == cfg_joint_name or joint_name.endswith(cfg_joint_name):
                    matched_value = value
                    break
            default_joint_pos.append(matched_value)
        return default_joint_pos

    def _configure_dr_state_slices(
        self, object_shapes: int, robot_shapes: int, num_joints: int, include_object_scale: bool
    ) -> None:
        index = 0
        self._dr_object_mass_slice = slice(index, index + 1)
        index += 1
        self._dr_object_scale_slice = slice(index, index + 3 * include_object_scale)
        index += 3 * include_object_scale
        self._dr_object_material_slice = slice(index, index + 3 * object_shapes)
        index += 3 * object_shapes
        self._dr_robot_material_slice = slice(index, index + 3 * robot_shapes)
        index += 3 * robot_shapes
        self._dr_stiffness_slice = slice(index, index + num_joints)
        index += num_joints
        self._dr_damping_slice = slice(index, index + num_joints)
        index += num_joints
        self._dr_joint_limit_mean_slice = slice(index, index + 2)
        index += 2
        self._dr_gravity_slice = slice(index, index + 3)
        index += 3
        if index != self.cfg.dr_state_space:
            raise RuntimeError(f"DR state slice size {index} does not match cfg.dr_state_space {self.cfg.dr_state_space}.")

    @staticmethod
    def _flatten_materials(materials: torch.Tensor, expected_shapes: int) -> torch.Tensor:
        return materials.reshape(materials.shape[0], expected_shapes, 3).reshape(materials.shape[0], 3 * expected_shapes)

    def _read_object_scale_xyz(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_object_size_dr or not self.cfg.enable_object_size_randomization:
            return torch.ones((len(env_ids), 3), dtype=torch.float, device=self.device)

        from pxr import Sdf

        stage = sim_utils.get_current_stage()
        scales = []
        default_scale = torch.tensor(DEXCUBE_SCALE, dtype=torch.float, device=self.device)
        for env_id in env_ids.detach().cpu().tolist():
            prim_path = f"/World/envs/env_{env_id}/object"
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None or scale_spec.default is None:
                scales.append(torch.ones(3, dtype=torch.float, device=self.device))
            else:
                usd_scale = torch.tensor(tuple(scale_spec.default), dtype=torch.float, device=self.device)
                scales.append(usd_scale / torch.clamp(default_scale, min=1.0e-6))
        return torch.stack(scales, dim=0)

    def _setup_scene(self):
        self._enable_urdf_importer()
        self._patch_urdf_converter_for_local_importer()
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        if self.cfg.apply_visual_materials:
            self._apply_hand_visual_materials()
            self._hide_dexcube_collision_visuals()
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_hand_visual_materials(self) -> None:
        from gex_learning.tasks.direct.dexanytwist.dexanytwist_env import DexAnyTwistEnv
        from pxr import Gf, UsdGeom

        stage = sim_utils.get_current_stage()
        colors = DexAnyTwistEnv._load_urdf_link_colors(self.cfg.robot_cfg.spawn.asset_path)
        DexAnyTwistEnv._bind_urdf_link_materials(stage, "Robot", colors, Gf, UsdGeom)

    def _hide_dexcube_collision_visuals(self) -> None:
        from pxr import UsdGeom

        stage = sim_utils.get_current_stage()
        for env_index in range(self.num_envs):
            collision_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/object/collisions")
            if collision_prim.IsValid():
                UsdGeom.Imageable(collision_prim).MakeInvisible()

    @staticmethod
    def _enable_urdf_importer() -> None:
        manager = omni.kit.app.get_app().get_extension_manager()
        for ext_name in ("isaacsim.asset.importer.urdf", "isaacsim.asset.importer.urdf-2.4.31"):
            try:
                ext_id = manager.get_enabled_extension_id(ext_name)
                if not ext_id:
                    manager.set_extension_enabled_immediate(ext_name, True)
                return
            except Exception:
                continue
        raise RuntimeError("Could not enable the Isaac Sim URDF importer extension.")

    @staticmethod
    def _patch_urdf_converter_for_local_importer() -> None:
        from gex_learning.tasks.direct.dexanytwist.dexanytwist_env import DexAnyTwistEnv

        DexAnyTwistEnv._patch_urdf_converter_for_local_importer()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        action_apply = self._get_random_network_adversary_action(self.actions[:, : len(self.actuated_dof_indices)])
        targets = scale(action_apply, lower, upper)
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * targets
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices], lower, upper
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self._apply_random_object_force()

    def _sample_random_force_prob(self, count: int) -> torch.Tensor:
        force_prob_range = torch.tensor(self.cfg.force_prob_range, dtype=torch.float, device=self.device)
        log_low = torch.log(force_prob_range[0])
        log_high = torch.log(force_prob_range[1])
        return torch.exp((log_low - log_high) * torch.rand(count, dtype=torch.float, device=self.device) + log_high)

    def _refresh_discretized_dof_targets(self) -> None:
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        steps = torch.linspace(0.0, 1.0, self.cfg.rna_softmax_bins, dtype=torch.float, device=self.device)
        self.discretized_dof_targets[:] = lower[..., None] + (upper - lower)[..., None] * steps

    def _get_rna_alpha(self) -> torch.Tensor:
        return torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device)

    def _get_random_network_adversary_action(self, canonical_action: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_rna or self.rna_network is None or self.rna_perturb_prob <= 0.0:
            # No adversary, or the RNA curriculum has it switched off: skip the network forward
            # pass entirely and report zeroed perturbation statistics.
            self.last_rna_action[:, : canonical_action.shape[-1]] = canonical_action
            self.rna_alpha.zero_()
            self.rna_perturb_mask.zero_()
            return canonical_action

        if (
            self.common_step_counter > 0
            and self.cfg.rna_weight_sample_freq > 0
            and self.common_step_counter % self.cfg.rna_weight_sample_freq == 0
        ):
            self.rna_network._refresh()

        object_pose = torch.cat((self.object_pos, self.object_rot), dim=-1)
        rna_input = torch.cat((self.hand_dof_pos[:, self.actuated_dof_indices], object_pose), dim=-1)
        rand_action_softmax = self.rna_network(rna_input)
        rand_action_indices = torch.argmax(rand_action_softmax, dim=-1)
        rand_joint_targets = torch.gather(self.discretized_dof_targets, 2, rand_action_indices.unsqueeze(-1)).squeeze(-1)

        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        rand_perturbation = unscale(rand_joint_targets, lower, upper)
        self.rna_perturb_mask = (
            torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device)
            < float(self.rna_perturb_prob)
        )
        rand_perturbation = torch.where(self.rna_perturb_mask, rand_perturbation, canonical_action)
        self.rna_alpha = self._get_rna_alpha()
        action_apply = self.rna_alpha * rand_perturbation + (1.0 - self.rna_alpha) * canonical_action
        self.last_rna_action[:, : canonical_action.shape[-1]] = action_apply
        return action_apply

    def _apply_random_object_force(self) -> None:
        if self.cfg.force_scale <= 0.0:
            self.object_external_forces.zero_()
            self.object_external_torques.zero_()
            self.object.permanent_wrench_composer.set_forces_and_torques(
                forces=self.object_external_forces,
                torques=self.object_external_torques,
                is_global=False,
            )
            return

        force_decay = torch.pow(
            torch.tensor(float(self.cfg.force_decay), dtype=torch.float, device=self.device),
            torch.tensor(float(self.step_dt / self.cfg.force_decay_interval), dtype=torch.float, device=self.device),
        )
        self.object_external_forces *= force_decay
        force_env_ids = (torch.rand(self.num_envs, dtype=torch.float, device=self.device) < self.random_force_prob).nonzero(
            as_tuple=False
        ).squeeze(-1)
        if len(force_env_ids) > 0:
            object_mass = self.object_mass[force_env_ids].mean(dim=-1, keepdim=True).unsqueeze(-1)
            self.object_external_forces[force_env_ids] = (
                torch.randn((len(force_env_ids), 1, 3), dtype=torch.float, device=self.device)
                * object_mass
                * float(self.cfg.force_scale)
            )
        self.object.permanent_wrench_composer.set_forces_and_torques(
            forces=self.object_external_forces,
            torques=self.object_external_torques,
            is_global=False,
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        self._refresh_gravity_dr_state()
        if self.hand_trajectory_sampler is not None and self.common_step_counter > 0:
            self.hand_trajectory_sampler.record(int(self.common_step_counter))
            if self.hand_trajectory_sampler.complete and self.cfg.hand_trajectory_quit_on_complete:
                raise SystemExit(0)
        hand_pos_obs = unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        pose_error = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        if self.cfg.policy_joint_action_only:
            policy_obs = torch.cat((hand_pos_obs, self.actions), dim=-1)
        else:
            policy_obs = torch.cat(
                (
                    hand_pos_obs,
                    self.actions,
                    self.object_pos,
                    self.object_rot,
                    self.goal_rot,
                    pose_error,
                ),
                dim=-1,
            )
        full_obs = torch.cat(
            (
                hand_pos_obs,
                self.cfg.vel_obs_scale * self.hand_dof_vel[:, self.actuated_dof_indices],
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.in_hand_pos,
                self.goal_rot,
                pose_error,
                self.fingertip_pos.reshape(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.reshape(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.reshape(self.num_envs, self.num_fingertips * 6),
                self.actions,
            ),
            dim=-1,
        )
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)
        full_obs = torch.nan_to_num(full_obs, nan=0.0, posinf=0.0, neginf=0.0)
        contact_forces = self.fingertip_contact_forces_w.reshape(self.num_envs, self.num_fingertips * 3)
        object_external_forces = self.object_external_forces[:, 0, :]
        critic_obs = torch.cat(
            (
                full_obs,
                torch.nan_to_num(contact_forces, nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(self.dr_state, nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(object_external_forces, nan=0.0, posinf=0.0, neginf=0.0),
            ),
            dim=-1,
        )
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        dist_reward = self.cfg.dist_reward_scale * goal_dist
        rot_reward = self.cfg.rot_reward_scale / (torch.abs(rot_dist) + self.cfg.rot_eps)
        ccw_angvel = self.object_angvel[:, 2]
        if self.cfg.goal_mode == "continuous_ccw":
            ccw_angvel_reward = self.cfg.ccw_angvel_reward_scale * torch.clamp(ccw_angvel, min=0.0)
            clockwise_angvel_penalty = self.cfg.clockwise_angvel_penalty_scale * torch.clamp(-ccw_angvel, min=0.0)
        else:
            ccw_angvel_reward = torch.zeros_like(ccw_angvel)
            clockwise_angvel_penalty = torch.zeros_like(ccw_angvel)
        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)

        reward = dist_reward + rot_reward + ccw_angvel_reward + clockwise_angvel_penalty + action_penalty

        goal_reached = torch.abs(rot_dist) <= self.cfg.success_tolerance
        self.reset_goal_buf = goal_reached
        self.successes += goal_reached.float()
        success_reward = torch.where(goal_reached, torch.full_like(reward, self.cfg.reach_goal_bonus), 0.0)
        fall_reward = torch.where(goal_dist >= self.cfg.fall_dist, torch.full_like(reward, self.cfg.fall_penalty), 0.0)
        reward = reward + success_reward + fall_reward

        terminated, truncated = self._get_dones()
        resets = terminated | truncated
        num_resets = torch.sum(resets.float())
        finished_successes = torch.sum(self.successes * resets.float())
        self.consecutive_successes[:] = torch.where(
            num_resets > 0,
            self.cfg.av_factor * finished_successes / num_resets
            + (1.0 - self.cfg.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["reward"] = reward.mean()
        self.extras["log"]["reward_dist"] = dist_reward.mean()
        self.extras["log"]["reward_rot"] = rot_reward.mean()
        self.extras["log"]["reward_ccw_angvel"] = ccw_angvel_reward.mean()
        self.extras["log"]["penalty_clockwise_angvel"] = clockwise_angvel_penalty.mean()
        self.extras["log"]["object_angvel_z"] = ccw_angvel.mean()
        self.extras["log"]["reward_action"] = action_penalty.mean()
        self.extras["log"]["reward_success"] = success_reward.mean()
        self.extras["log"]["reward_fall"] = fall_reward.mean()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["successes"] = self.successes.mean()
        self.extras["log"]["successes_max"] = self.successes.max()
        self.extras["log"]["success_rate"] = goal_reached.float().mean()
        self.extras["log"]["episode_success_rate"] = (self.successes > 0.0).float().mean()
        self.extras["log"]["rotation_distance"] = rot_dist.mean()
        self.extras["log"]["object_goal_distance"] = goal_dist.mean()
        self.extras["log"]["object_goal_distance_max"] = goal_dist.max()
        self.extras["log"]["object_external_force_norm"] = torch.norm(
            self.object_external_forces[:, 0, :], p=2, dim=-1
        ).mean()
        self.extras["log"]["object_external_force_prob_mean"] = self.random_force_prob.mean()
        self.extras["log"]["rna_alpha_mean"] = self.rna_alpha.mean()
        self.extras["log"]["rna_action_delta_norm"] = torch.norm(
            self.last_rna_action[:, : len(self.actuated_dof_indices)]
            - self.actions[:, : len(self.actuated_dof_indices)],
            p=2,
            dim=-1,
        ).mean()
        self.extras["log"]["rna_perturb_mask_rate"] = self.rna_perturb_mask.float().mean()
        self.extras["log"]["rna_enabled"] = torch.tensor(
            float(self.cfg.enable_rna), dtype=torch.float, device=self.device
        )

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            max_success_reached = self.successes >= self.cfg.max_consecutive_success
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._refresh_dr_state(env_ids_t)
        self.object_external_forces[env_ids_t] = 0.0
        self.object_external_torques[env_ids_t] = 0.0
        self.random_force_prob[env_ids_t] = self._sample_random_force_prob(len(env_ids_t))
        self.object.permanent_wrench_composer.set_forces_and_torques(
            forces=self.object_external_forces[env_ids_t],
            torques=self.object_external_torques[env_ids_t],
            env_ids=env_ids_t,
            is_global=False,
        )

        object_state = self.object.data.default_root_state.clone()[env_ids_t]
        pos_noise_xy = sample_uniform(-1.0, 1.0, (len(env_ids_t), 2), device=self.device)
        pos_noise_z = sample_uniform(-1.0, 1.0, (len(env_ids_t), 1), device=self.device)
        object_state[:, 0:2] = (
            object_state[:, 0:2]
            + self.cfg.reset_position_noise * pos_noise_xy
            + self.scene.env_origins[env_ids_t, 0:2]
        )
        object_state[:, 2:3] = (
            object_state[:, 2:3]
            + self.cfg.reset_position_noise_z * pos_noise_z
            + self.scene.env_origins[env_ids_t, 2:3]
        )
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids_t),), device=self.device)
        rotation_delta = quat_from_angle_axis(
            rot_noise * float(self.cfg.reset_object_rotation_noise), self.z_unit_tensor[env_ids_t]
        )
        object_state[:, 3:7] = quat_mul(rotation_delta, object_state[:, 3:7])
        object_state[:, 7:] = torch.zeros_like(object_state[:, 7:])
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids_t)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids_t)
        self._reset_target_pose(env_ids_t, base_rot=object_state[:, 3:7])

        hand_pos = self.default_hand_joint_pos[env_ids_t].clone()
        lower = self.hand_dof_lower_limits[env_ids_t]
        upper = self.hand_dof_upper_limits[env_ids_t]
        rand = sample_uniform(-1.0, 1.0, (len(env_ids_t), self.num_hand_dofs), device=self.device)
        rand_delta = (lower - hand_pos) + (upper - lower) * 0.5 * (rand + 1.0)
        hand_pos = hand_pos + self.cfg.reset_dof_pos_noise * rand_delta
        hand_pos = torch.clamp(hand_pos, lower, upper)
        hand_vel = torch.zeros_like(hand_pos)

        self.prev_targets[env_ids_t] = hand_pos
        self.cur_targets[env_ids_t] = hand_pos
        self.hand.set_joint_position_target(hand_pos, env_ids=env_ids_t)
        self.hand.write_joint_state_to_sim(hand_pos, hand_vel, env_ids=env_ids_t)

        episode_success = self.successes[env_ids_t] > 0.0
        self.successes[env_ids_t] = 0.0
        self.reset_goal_buf[env_ids_t] = False
        self._compute_intermediate_values()
        if getattr(self, "hand_trajectory_sampler", None) is not None:
            self.hand_trajectory_sampler.mark_reset(env_ids_t, episode_success)

    def close(self):
        if getattr(self, "hand_trajectory_sampler", None) is not None:
            self.hand_trajectory_sampler.close()
            self.hand_trajectory_sampler = None
        super().close()

    def _reset_target_pose(self, env_ids: torch.Tensor, base_rot: torch.Tensor | None = None) -> None:
        if self.cfg.goal_mode == "continuous_ccw":
            if base_rot is None:
                base_rot = self.goal_rot[env_ids]
            goal_step = torch.full(
                (len(env_ids),), float(self.cfg.continuous_goal_step), dtype=torch.float, device=self.device
            )
            step_rot = quat_from_angle_axis(goal_step, self.z_unit_tensor[env_ids])
            self.goal_rot[env_ids] = quat_mul(step_rot, base_rot)
        else:
            self.goal_rot[env_ids] = self._sample_cubic_rotations(env_ids)
        self.goal_pos[:] = self.goal_marker_pos
        if self.goal_markers is not None:
            self.goal_markers.visualize(
                self.goal_pos + self.scene.env_origins,
                self.goal_rot,
                self.object_scale_xyz if self.cfg.enable_object_size_dr else None,
            )
        self.reset_goal_buf[env_ids] = False

    def _sample_cubic_rotations(self, env_ids: torch.Tensor) -> torch.Tensor:
        angles = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
        axis_indices = (
            range(3) if self.cfg.goal_rotation_axes == "xyz" else ("xyz".index(self.cfg.goal_rotation_axes),)
        )
        for axis_index in axis_indices:
            angles[:, axis_index] = (
                torch.randint(0, 4, (len(env_ids),), device=self.device).float() * (0.5 * np.pi)
            )
        return quat_mul(
            quat_from_angle_axis(angles[:, 2], self.z_unit_tensor[env_ids]),
            quat_mul(
                quat_from_angle_axis(angles[:, 1], self.y_unit_tensor[env_ids]),
                quat_from_angle_axis(angles[:, 0], self.x_unit_tensor[env_ids]),
            ),
        )

    def _refresh_dr_state(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

        if len(env_ids) == 0:
            return

        current_joint_limits = self.hand.data.joint_pos_limits.to(self.device)
        self.hand_dof_lower_limits = current_joint_limits[..., 0]
        self.hand_dof_upper_limits = current_joint_limits[..., 1]
        self._refresh_discretized_dof_targets()

        object_masses = self.object.root_physx_view.get_masses().to(self.device)[env_ids]
        self.object_mass[env_ids] = object_masses
        object_mass_scale = object_masses / torch.clamp(self.default_object_mass[env_ids], min=1.0e-6)
        self.dr_state[env_ids, self._dr_object_mass_slice] = object_mass_scale.mean(dim=-1, keepdim=True)

        self.object_scale_xyz[env_ids] = self._read_object_scale_xyz(env_ids)
        if self.cfg.enable_object_size_dr:
            self.dr_state[env_ids, self._dr_object_scale_slice] = self.object_scale_xyz[env_ids]

        object_materials = self.object.root_physx_view.get_material_properties().to(self.device)[env_ids]
        self.dr_state[env_ids, self._dr_object_material_slice] = self._flatten_materials(
            object_materials, self.object_dr_shapes
        )

        robot_materials = self.hand.root_physx_view.get_material_properties().to(self.device)[env_ids]
        self.dr_state[env_ids, self._dr_robot_material_slice] = self._flatten_materials(
            robot_materials, self.robot_dr_shapes
        )

        joint_ids = torch.as_tensor(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        stiffness = self.hand.data.joint_stiffness[env_ids[:, None], joint_ids]
        default_stiffness = self.hand.data.default_joint_stiffness[env_ids[:, None], joint_ids]
        damping = self.hand.data.joint_damping[env_ids[:, None], joint_ids]
        default_damping = self.hand.data.default_joint_damping[env_ids[:, None], joint_ids]
        self.dr_state[env_ids, self._dr_stiffness_slice] = stiffness / torch.clamp(default_stiffness, min=1.0e-6)
        self.dr_state[env_ids, self._dr_damping_slice] = damping / torch.clamp(default_damping, min=1.0e-6)

        joint_limits = current_joint_limits[env_ids[:, None], joint_ids]
        default_joint_limits = self.hand.data.default_joint_pos_limits[env_ids[:, None], joint_ids]
        joint_limit_offsets = torch.stack(
            (
                (joint_limits[..., 0] - default_joint_limits[..., 0]).mean(dim=-1),
                (joint_limits[..., 1] - default_joint_limits[..., 1]).mean(dim=-1),
            ),
            dim=-1,
        )
        self.dr_state[env_ids, self._dr_joint_limit_mean_slice] = joint_limit_offsets
        self._refresh_gravity_dr_state(env_ids)

        self.dr_state[env_ids] = torch.nan_to_num(self.dr_state[env_ids], nan=0.0, posinf=0.0, neginf=0.0)

    def _refresh_gravity_dr_state(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

        if len(env_ids) == 0:
            return

        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        gravity_value = physics_sim_view.get_gravity()
        gravity = torch.tensor(
            (gravity_value[0], gravity_value[1], gravity_value[2]), dtype=torch.float, device=self.device
        )
        gravity_delta = gravity - self.default_gravity
        self.dr_state[env_ids, self._dr_gravity_slice] = gravity_delta

    def _initialize_fingertip_contact_view(self) -> None:
        if self._fingertip_contact_physx_view is not None:
            return

        physics_sim_view = SimulationManager.get_physics_sim_view()
        prim_patterns = [f"/World/envs/env_*/Robot/{name}" for name in self.cfg.fingertip_body_names]
        self._fingertip_contact_physx_view = physics_sim_view.create_rigid_contact_view(prim_patterns)

        expected_count = self.num_envs * self.num_fingertips
        if self._fingertip_contact_physx_view.sensor_count != expected_count:
            raise RuntimeError(
                "Failed to initialize fingertip contact force view."
                f" Expected {expected_count} sensor bodies, got {self._fingertip_contact_physx_view.sensor_count}."
            )

        fingertip_index_by_name = {name: index for index, name in enumerate(self.cfg.fingertip_body_names)}
        order = torch.empty(expected_count, dtype=torch.long, device=self.device)
        filled = torch.zeros(expected_count, dtype=torch.bool, device=self.device)
        env_pattern = re.compile(r"/env_(\d+)/")
        for source_index, path in enumerate(self._fingertip_contact_physx_view.sensor_paths):
            path = str(path)
            env_match = env_pattern.search(path)
            link_name = path.rsplit("/", 1)[-1]
            if env_match is None or link_name not in fingertip_index_by_name:
                continue
            env_index = int(env_match.group(1))
            if env_index >= self.num_envs:
                continue
            target_index = env_index * self.num_fingertips + fingertip_index_by_name[link_name]
            order[target_index] = source_index
            filled[target_index] = True
        if not torch.all(filled):
            missing = torch.nonzero(~filled, as_tuple=False).squeeze(-1).detach().cpu().tolist()
            raise RuntimeError(f"Could not resolve fingertip contact force ordering for flat indices: {missing}")
        self._fingertip_contact_force_order = order

    def _get_fingertip_contact_forces_w(self) -> torch.Tensor:
        self._initialize_fingertip_contact_view()
        forces = self._fingertip_contact_physx_view.get_net_contact_forces(dt=self.physics_dt)
        forces = forces[self._fingertip_contact_force_order]
        return forces.reshape(self.num_envs, self.num_fingertips, 3)

    def _compute_intermediate_values(self) -> None:
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies] - self.scene.env_origins[:, None, :]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]
        self.fingertip_contact_forces_w = self._get_fingertip_contact_forces_w()
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0


@torch.jit.script
def randomize_z_rotation(rand: torch.Tensor, z_unit_tensor: torch.Tensor):
    return quat_from_angle_axis(rand * np.pi, z_unit_tensor)


@torch.jit.script
def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
