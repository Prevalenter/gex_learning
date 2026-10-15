from __future__ import annotations

from collections.abc import Sequence
import copy
import math
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import omni.kit.app

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .dataset import HandSpec, TwistObjectSpec, load_hand_spec, load_twist_object, load_twist_object_names
from gex_learning.tasks.direct.hand_trajectory_recorder import HandTrajectorySampler


DEFAULT_OBJECT_NAME = "bulb_custom_0017"
DEFAULT_OBJECT = load_twist_object(DEFAULT_OBJECT_NAME)
DEFAULT_HAND = load_hand_spec("gx4m")
# GX4M palm/finger-pad side is local -Y. Rotate about X by 90 deg so it faces world -Z.
HAND_ROT_PALM_DOWN = (90.0, 90.0, 0.0)  # XYZ Euler angles in degrees.
HAND_START_POS = (0.0, 0.0, 0.25)
# DexAnyTwist object offsets were authored around the GX10 base frame. GX4M uses a different
# base definition, so apply a hand-tuned translation compensation before training/evaluation.
GX4M_OBJECT_POSE_OFFSET = (0.0, -0.0, 0.04)
DEFAULT_HAND_POSE_OFFSET = (0.0, 0.0, 0.0)
DEFAULT_HAND_ROT_EULER_DEG = (20.0, 0.0, 0.0)
DEFAULT_ACTION_MEAN_JOINT_POS = (
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
)
DEFAULT_OBJECT_SCALE_RANGE = (0.90, 1.05)

def object_start_pos(spec: TwistObjectSpec, offset: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    return (
        HAND_START_POS[0] + spec.pose[0] + float(offset[0]),
        HAND_START_POS[1] + spec.pose[1] + float(offset[1]),
        HAND_START_POS[2] + spec.pose[2] + float(offset[2]),
    )


def _median_position(positions: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    values = []
    for axis in range(3):
        sorted_values = sorted(float(pos[axis]) for pos in positions)
        values.append(sorted_values[len(sorted_values) // 2])
    return (values[0], values[1], values[2])


def _vec3_add(base: Sequence[float], offset: Sequence[float]) -> tuple[float, float, float]:
    return (float(base[0]) + float(offset[0]), float(base[1]) + float(offset[1]), float(base[2]) + float(offset[2]))


def _same_float_sequence(lhs: Sequence[float] | None, rhs: Sequence[float] | None) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if len(lhs) != len(rhs):
        return False
    return all(abs(float(left) - float(right)) <= 1.0e-9 for left, right in zip(lhs, rhs))


def _uniform_scale_tuple(scale: float) -> tuple[float, float, float]:
    value = float(scale)
    return (value, value, value)


def _optional_uniform_scale_tuple(scale: float) -> tuple[float, float, float] | None:
    if abs(float(scale) - 1.0) <= 1.0e-9:
        return None
    return _uniform_scale_tuple(scale)


def _scale_range_samples(scale_range: Sequence[float], variant_count: int, label: str) -> list[float]:
    if len(scale_range) != 2:
        raise ValueError(f"{label} must contain exactly two values: [min, max].")
    lower = float(scale_range[0])
    upper = float(scale_range[1])
    if lower <= 0.0 or upper <= 0.0:
        raise ValueError(f"{label} values must be positive, got ({lower}, {upper}).")
    if lower > upper:
        raise ValueError(f"{label} lower bound must be <= upper bound, got ({lower}, {upper}).")
    count = int(variant_count)
    if count < 1:
        raise ValueError(f"scale_variant_count must be >= 1, got {variant_count}.")
    if abs(lower - upper) <= 1.0e-9:
        return [lower]
    if count == 1:
        return [0.5 * (lower + upper)]
    step = (upper - lower) / float(count - 1)
    return [lower + step * index for index in range(count)]


def _normalize_quat(quat: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quat))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _quat_mul(lhs: Sequence[float], rhs: Sequence[float]) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(value) for value in lhs)
    rw, rx, ry, rz = (float(value) for value in rhs)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_from_euler_xyz_deg(euler_deg: Sequence[float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(float(value)) for value in euler_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quat(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


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
    """Low-risk Isaac Lab domain randomization terms for the DexAnyTwist direct task."""

    hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("hand"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    hand_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
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
class DexAnyTwistEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = DEFAULT_HAND.num_actions
    observation_space = 2 * DEFAULT_HAND.num_actions
    state_space = (
        2 * DEFAULT_HAND.num_actions
        + 3 * len(DEFAULT_HAND.fingertip_body_names)
        + 1
        + 128
        + 3
        + 1
        + 3 * len(DEFAULT_OBJECT.keypoint_bias)
    )
    viewer = ViewerCfg(
        eye=(0.68, -0.94, 1.06),
        lookat=(0.03, -0.04, 0.61),
        origin_type="env",
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            enable_external_forces_every_iteration=True,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=0.75,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    hand_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(DEFAULT_HAND.urdf_path),
            fix_base=True,
            merge_fixed_joints=True,
            make_instanceable=False,
            self_collision=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=2.0, damping=0.1),
            ),
            collider_type="convex_hull",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_START_POS,
            rot=_quat_from_euler_xyz_deg(HAND_ROT_PALM_DOWN),
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

    object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(DEFAULT_OBJECT.urdf_path),
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            self_collision=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type={"joint1": "none", "joint1_rx": "position", "joint1_ry": "position"},
                drive_type="force",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness={"joint1": 0.0, "joint1_rx": 1000.0, "joint1_ry": 1000.0},
                    damping={"joint1": 0.001, "joint1_rx": 1000.0, "joint1_ry": 1000.0},
                ),
            ),
            collider_type="convex_decomposition",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=object_start_pos(DEFAULT_OBJECT, GX4M_OBJECT_POSE_OFFSET),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "main_joint": ImplicitActuatorCfg(
                joint_names_expr=["joint1"],
                effort_limit_sim=0.001,
                velocity_limit_sim=1.0,
                stiffness=0.0,
                damping=0.001,
            ),
            "xy_guard": ImplicitActuatorCfg(
                joint_names_expr=["joint1_rx", "joint1_ry"],
                effort_limit_sim=1000.0,
                velocity_limit_sim=1.0,
                stiffness=1000.0,
                damping=1000.0,
            ),
        },
    )

    object_keypoint_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/dexanytwist_object_keypoints",
        markers={
            "object_keypoint": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.15),
                    emissive_color=(0.0, 0.8, 0.12),
                ),
            )
        },
    )
    fingertip_keypoint_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/dexanytwist_fingertip_keypoints",
        markers={
            "fingertip_keypoint": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.1, 0.05),
                    emissive_color=(0.25, 0.02, 0.0),
                ),
            )
        },
    )

    hand_name = DEFAULT_HAND.name
    actuated_joint_names = list(DEFAULT_HAND.actuated_joint_names)
    fingertip_body_names = list(DEFAULT_HAND.fingertip_body_names)
    default_hand_joint_pos = list(DEFAULT_HAND.default_joint_pos)
    action_mean_joint_pos: list[float] | None = list(DEFAULT_ACTION_MEAN_JOINT_POS)
    policy_joint_action_only = True
    object_name = DEFAULT_OBJECT.name
    object_names = ""
    object_set = ""
    object_class_index = DEFAULT_OBJECT.class_index
    object_rotation_axis_sign = DEFAULT_OBJECT.rotation_axis_sign
    object_xy_axis_limit = DEFAULT_OBJECT.xy_axis_limit
    object_keypoint_bias = DEFAULT_OBJECT.keypoint_bias.tolist()
    object_points_feature = DEFAULT_OBJECT.points_feature.tolist()
    object_pose_offset = list(GX4M_OBJECT_POSE_OFFSET)
    hand_pose_offset = list(DEFAULT_HAND_POSE_OFFSET)
    hand_start_rot_xyz_deg = list(HAND_ROT_PALM_DOWN)
    hand_rot_euler_deg = list(DEFAULT_HAND_ROT_EULER_DEG)
    enable_scale_range = True
    object_scale_range = list(DEFAULT_OBJECT_SCALE_RANGE)
    scale_variant_count = 30
    fingertip_keypoint_offset = [list(offset) for offset in DEFAULT_HAND.fingertip_keypoint_offset]
    apply_visual_materials = True
    show_keypoint_markers = False
    save_hand_trajectory = False
    hand_trajectory_dir = "experiment/dexanytwist/hand_trajectories"
    hand_trajectory_interval = 1000
    hand_trajectory_max_steps = 0
    hand_trajectory_quit_on_complete = False
    print_hand_joint_limits = True
    hand_effort_scale = 3.0
    hand_velocity_scale = 1.0
    hand_stiffness_scale = 2.0
    hand_damping_scale = 1.0

    dof_speed_scale = 20.0
    act_moving_average = 0.2
    reset_position_noise = 0.005
    reset_dof_pos_noise = 0.2
    reset_dof_vel_noise = 0.0
    action_penalty_scale = -0.0001
    action_delta_penalty_scale = 0.0
    joint_velocity_penalty_scale = 0.0
    keypoints_reward_scale = -5.0
    progress_reward_scale = 0.0
    max_progress_reward_scale = 0.0
    success_progress_reward_scale = 0.0
    near_success_reward_scale = 0.0
    near_success_margin = 1.0
    success_margin_reward_scale = 0.0
    reset_hand_on_success = False
    reset_phase_on_success = False
    reach_goal_bonus = 3000.0
    success_threshold = 4.0
    object_fall_threshold = 0.25
    av_factor = 0.1

    enable_rna = True
    rna_action_perturb_prob = 0.15
    rna_weight_sample_freq = 1000
    rna_softmax_bins = 32
    force_scale = 2.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08

    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="abs"),
    )


class DexAnyTwistEnv(DirectRLEnv):
    cfg: DexAnyTwistEnvCfg

    def __init__(self, cfg: DexAnyTwistEnvCfg, render_mode: str | None = None, **kwargs):
        self._apply_hand_config(cfg)
        self._apply_object_config(cfg)
        self._object_specs = self._resolve_object_specs(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints
        self.actuated_dof_indices = [self._find_name(self.hand.joint_names, name) for name in self.cfg.actuated_joint_names]
        self.fingertip_body_ids = [self._find_name(self.hand.body_names, name) for name in self.cfg.fingertip_body_names]
        self.object_link_body_id = self._find_name(self.object.body_names, "link1", default=len(self.object.body_names) - 1)
        self.object_joint_ids = {
            "joint1": self._find_name(self.object.joint_names, "joint1"),
            "joint1_rx": self._find_name(self.object.joint_names, "joint1_rx"),
            "joint1_ry": self._find_name(self.object.joint_names, "joint1_ry"),
        }

        joint_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_limits[..., 0]
        self.hand_dof_upper_limits = joint_limits[..., 1]
        if self.cfg.print_hand_joint_limits:
            self._print_hand_joint_limits()

        default_hand_joint_pos = torch.tensor(
            self._default_joint_pos_in_articulation_order(self.cfg), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.default_hand_joint_pos = torch.clamp(
            default_hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.policy_actions = torch.zeros_like(self.actions)
        self.action_mean = torch.zeros_like(self.actions)
        self._refresh_action_mean()
        self.last_rna_action = torch.zeros_like(self.actions)
        self.discretized_dof_targets = torch.zeros(
            (self.num_envs, len(self.actuated_dof_indices), self.cfg.rna_softmax_bins),
            dtype=torch.float,
            device=self.device,
        )
        self.rna_network = (
            RandomNetworkAdversary(
                num_envs=self.num_envs,
                in_dims=len(self.actuated_dof_indices) + 7,
                out_dims=len(self.actuated_dof_indices),
                softmax_bins=self.cfg.rna_softmax_bins,
                device=self.device,
            )
            if self.cfg.enable_rna
            else None
        )
        self.rna_alpha = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device)
        self.rna_perturb_mask = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        self._refresh_discretized_dof_targets()

        object_specs = self._object_specs
        self.object_names = [spec.name for spec in object_specs]
        self.object_count = len(object_specs)
        self.spawn_scale_variants = self._spawn_scale_variants(self.cfg, object_specs)
        self.spawn_variant_count = len(self.spawn_scale_variants)
        spawn_variant_ids = self._spawn_variant_assignment_ids(self.spawn_variant_count)
        self.spawn_variant_ids = torch.tensor(spawn_variant_ids, dtype=torch.long, device=self.device)
        object_assignment_ids = [self.spawn_scale_variants[index][0] for index in spawn_variant_ids]
        object_scale_values = [self.spawn_scale_variants[index][1] for index in spawn_variant_ids]
        self.object_spec_ids = torch.tensor(object_assignment_ids, dtype=torch.long, device=self.device)
        self.object_scale = torch.tensor(object_scale_values, dtype=torch.float, device=self.device).view(self.num_envs, 1)
        object_assignments = [object_specs[index] for index in object_assignment_ids]
        self.keypoint_bias = torch.tensor(
            [spec.keypoint_bias.tolist() for spec in object_assignments], dtype=torch.float, device=self.device
        )
        self.keypoint_bias = self.keypoint_bias * self.object_scale[:, None, :]
        self.points_feature = torch.tensor(
            [spec.points_feature.tolist() for spec in object_assignments], dtype=torch.float, device=self.device
        )
        self.rotation_axis = torch.tensor(
            [[float(spec.rotation_axis_sign)] for spec in object_assignments], dtype=torch.float, device=self.device
        )
        self.class_index = torch.tensor(
            [[float(spec.class_index) / 100.0] for spec in object_assignments], dtype=torch.float, device=self.device
        )
        fingertip_offset = torch.tensor(self.cfg.fingertip_keypoint_offset, dtype=torch.float, device=self.device)
        if fingertip_offset.ndim == 1:
            if fingertip_offset.shape[0] != 3:
                raise ValueError(
                    "fingertip_keypoint_offset must be [x, y, z] or one [x, y, z] per fingertip, "
                    f"got shape {tuple(fingertip_offset.shape)}."
                )
            fingertip_offset = fingertip_offset.repeat(len(self.fingertip_body_ids), 1)
        if fingertip_offset.shape != (len(self.fingertip_body_ids), 3):
            raise ValueError(
                "fingertip_keypoint_offset must have shape "
                f"({len(self.fingertip_body_ids)}, 3), got {tuple(fingertip_offset.shape)}."
            )
        self.fingertip_offset = fingertip_offset.unsqueeze(0).repeat((self.num_envs, 1, 1))
        self.object_default_root_pose = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.object_default_root_pose[:, :3] = torch.tensor(
            [object_start_pos(spec, self.cfg.object_pose_offset) for spec in object_assignments],
            dtype=torch.float,
            device=self.device,
        )
        self.object_default_root_pose[:, 3:7] = torch.tensor(
            self.cfg.object_cfg.init_state.rot, dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.object_external_forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
        self.object_external_torques = torch.zeros_like(self.object_external_forces)
        self.default_object_mass = self.object.data.default_mass.to(self.device).clone()
        self.object_mass = self.default_object_mass.clone()
        self.random_force_prob = self._sample_random_force_prob(self.num_envs)

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_max_object_joint_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_max_object_joint_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.keypoints_closest_distance = torch.zeros(
            (self.num_envs, len(self.fingertip_body_ids)), dtype=torch.float, device=self.device
        )
        self.object_keypoint_markers = (
            VisualizationMarkers(self.cfg.object_keypoint_marker_cfg) if self.cfg.show_keypoint_markers else None
        )
        self.fingertip_keypoint_markers = (
            VisualizationMarkers(self.cfg.fingertip_keypoint_marker_cfg) if self.cfg.show_keypoint_markers else None
        )
        if self.cfg.show_keypoint_markers:
            print(
                "[DexAnyTwist] keypoint markers enabled: "
                f"{self.keypoint_bias.shape[1]} object keypoints/env, "
                f"{len(self.fingertip_body_ids)} fingertip keypoints/env.",
                flush=True,
            )
        self._hand_trajectory_enabled = False
        self.hand_trajectory_path = None
        self._last_hand_trajectory_step = None
        self._compute_intermediate_values()
        self._setup_hand_trajectory_recorder()

    @staticmethod
    def _apply_hand_config(cfg: DexAnyTwistEnvCfg) -> None:
        spec = load_hand_spec(cfg.hand_name)
        object_specs = DexAnyTwistEnv._resolve_object_specs(cfg)
        object_spec = object_specs[0]
        base_spawn = cfg.hand_cfg.spawn
        if hasattr(base_spawn, "assets_cfg"):
            base_spawn = base_spawn.assets_cfg[0]
        base_spawn = copy.deepcopy(base_spawn)
        if _same_float_sequence(cfg.hand_pose_offset, DEFAULT_HAND_POSE_OFFSET):
            cfg.hand_pose_offset = list(object_spec.hand_pose_offset)
        if _same_float_sequence(cfg.hand_rot_euler_deg, DEFAULT_HAND_ROT_EULER_DEG):
            cfg.hand_rot_euler_deg = list(object_spec.hand_rot_euler_deg)
        if cfg.action_mean_joint_pos is not None and _same_float_sequence(
            cfg.action_mean_joint_pos, DEFAULT_ACTION_MEAN_JOINT_POS
        ):
            cfg.action_mean_joint_pos = list(object_spec.action_mean_joint_pos)
        cfg.hand_name = spec.name
        base_spawn.asset_path = str(spec.urdf_path)
        cfg.hand_cfg.init_state.pos = _vec3_add(spec.start_pos, cfg.hand_pose_offset)
        start_rot = _quat_from_euler_xyz_deg(cfg.hand_start_rot_xyz_deg)
        cfg.hand_cfg.init_state.rot = _quat_mul(_quat_from_euler_xyz_deg(cfg.hand_rot_euler_deg), start_rot)
        cfg.actuated_joint_names = list(spec.actuated_joint_names)
        cfg.fingertip_body_names = list(spec.fingertip_body_names)
        cfg.default_hand_joint_pos = list(spec.default_joint_pos)
        cfg.fingertip_keypoint_offset = [list(offset) for offset in spec.fingertip_keypoint_offset]
        cfg.action_space = spec.num_actions
        full_observation_size = DexAnyTwistEnv._full_observation_size(spec, object_specs)
        cfg.observation_space = (
            2 * spec.num_actions if cfg.policy_joint_action_only else full_observation_size
        )
        cfg.state_space = full_observation_size

        actuator = cfg.hand_cfg.actuators["fingers"]
        actuator.joint_names_expr = list(spec.actuated_joint_names)
        actuator.effort_limit_sim = spec.effort_limit * float(cfg.hand_effort_scale)
        actuator.velocity_limit_sim = spec.velocity_limit * float(cfg.hand_velocity_scale)
        actuator.stiffness = spec.stiffness * float(cfg.hand_stiffness_scale)
        actuator.damping = spec.damping * float(cfg.hand_damping_scale)

        joint_gains = base_spawn.joint_drive.gains
        joint_gains.stiffness = spec.stiffness * float(cfg.hand_stiffness_scale)
        joint_gains.damping = spec.damping * float(cfg.hand_damping_scale)

        base_spawn.scale = None
        cfg.hand_cfg.spawn = base_spawn

    @staticmethod
    def _full_observation_size(hand_spec: HandSpec, object_specs: Sequence[TwistObjectSpec]) -> int:
        object_keypoint_count = len(object_specs[0].keypoint_bias)
        for object_spec in object_specs[1:]:
            if len(object_spec.keypoint_bias) != object_keypoint_count:
                raise ValueError(
                    "DexAnyTwist policy observation requires the same number of object keypoints for every object. "
                    f"{object_specs[0].name} has {object_keypoint_count}, "
                    f"but {object_spec.name} has {len(object_spec.keypoint_bias)}."
                )
        return (
            2 * hand_spec.num_actions
            + 3 * len(hand_spec.fingertip_body_names)
            + 1
            + 128
            + 3
            + 1
            + 3 * object_keypoint_count
        )

    @staticmethod
    def _apply_object_config(cfg: DexAnyTwistEnvCfg) -> None:
        specs = DexAnyTwistEnv._resolve_object_specs(cfg)
        spec = specs[0]
        cfg.object_name = spec.name
        cfg.object_names = ",".join(object_spec.name for object_spec in specs)
        base_spawn = cfg.object_cfg.spawn
        if hasattr(base_spawn, "assets_cfg"):
            base_spawn = base_spawn.assets_cfg[0]
        base_spawn = copy.deepcopy(base_spawn)
        joint_gains = base_spawn.joint_drive.gains
        xy_actuator = cfg.object_cfg.actuators["xy_guard"]
        if spec.xy_axis_limit > 0.1:
            joint_gains.stiffness["joint1_rx"] = 0.0
            joint_gains.stiffness["joint1_ry"] = 0.0
            joint_gains.damping["joint1_rx"] = 0.0
            joint_gains.damping["joint1_ry"] = 0.0
            xy_actuator.effort_limit_sim = 0.001
            xy_actuator.stiffness = 0.0
            xy_actuator.damping = 0.0
        else:
            joint_gains.stiffness["joint1_rx"] = 1000.0
            joint_gains.stiffness["joint1_ry"] = 1000.0
            joint_gains.damping["joint1_rx"] = 1000.0
            joint_gains.damping["joint1_ry"] = 1000.0
            xy_actuator.effort_limit_sim = 1000.0
            xy_actuator.stiffness = 1000.0
            xy_actuator.damping = 1000.0

        variants = DexAnyTwistEnv._spawn_scale_variants(cfg, specs)
        if len(variants) > 1:
            cfg.scene.replicate_physics = False
            cfg.scene.clone_in_fabric = False
            assets_cfg = []
            for object_spec_index, object_scale in variants:
                object_spec = specs[object_spec_index]
                asset_cfg = copy.deepcopy(base_spawn)
                asset_cfg.asset_path = str(object_spec.urdf_path)
                asset_cfg.scale = _optional_uniform_scale_tuple(object_scale)
                assets_cfg.append(asset_cfg)
            cfg.object_cfg.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets_cfg, random_choice=False)
            cfg.object_cfg.init_state.pos = _median_position(
                [object_start_pos(object_spec, cfg.object_pose_offset) for object_spec in specs]
            )
        else:
            base_spawn.asset_path = str(spec.urdf_path)
            base_spawn.scale = _optional_uniform_scale_tuple(variants[0][1])
            cfg.object_cfg.spawn = base_spawn
            cfg.object_cfg.init_state.pos = object_start_pos(spec, cfg.object_pose_offset)
        cfg.object_class_index = spec.class_index
        cfg.object_rotation_axis_sign = spec.rotation_axis_sign
        cfg.object_xy_axis_limit = spec.xy_axis_limit
        cfg.object_keypoint_bias = spec.keypoint_bias.tolist()
        cfg.object_points_feature = spec.points_feature.tolist()

    @staticmethod
    def _resolve_object_specs(cfg: DexAnyTwistEnvCfg) -> list[TwistObjectSpec]:
        if cfg.object_set not in ("", "bulb_custom_0017"):
            raise ValueError("This release supports only object_set='' or 'bulb_custom_0017'.")
        requested = cfg.object_names or [cfg.object_name]
        if isinstance(requested, str):
            requested = requested.strip().strip("\"'").replace(";", ",").split(",")
        names = [str(name).strip() for name in requested]
        if not names or any(name != "bulb_custom_0017" for name in names):
            raise ValueError("This release supports only bulb_custom_0017.")
        return [load_twist_object(name) for name in names]

    @staticmethod
    def _spawn_scale_variants(cfg: DexAnyTwistEnvCfg, specs: Sequence[TwistObjectSpec]) -> list[tuple[int, float]]:
        object_scales = DexAnyTwistEnv._object_scale_samples(cfg)
        return [
            (object_spec_index, object_scale)
            for object_spec_index in range(len(specs))
            for object_scale in object_scales
        ]

    @staticmethod
    def _object_scale_samples(cfg: DexAnyTwistEnvCfg) -> list[float]:
        if not cfg.enable_scale_range:
            return [1.0]
        return _scale_range_samples(cfg.object_scale_range, cfg.scale_variant_count, "object_scale_range")

    def _spawn_variant_assignment_ids(self, variant_count: int) -> list[int]:
        if variant_count <= 1:
            return [0 for _ in range(self.num_envs)]

        assignment_ids = [env_index % variant_count for env_index in range(self.num_envs)]
        object_root_expr = self.cfg.object_cfg.prim_path.rsplit("/", 1)[0]
        matched_env_paths = sim_utils.find_matching_prim_paths(object_root_expr)
        matched_count = 0
        for spawn_order, env_path in enumerate(matched_env_paths):
            env_name = env_path.rsplit("/", 1)[-1]
            if not env_name.startswith("env_"):
                continue
            try:
                env_index = int(env_name.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if 0 <= env_index < self.num_envs:
                assignment_ids[env_index] = spawn_order % variant_count
                matched_count += 1

        if matched_count != self.num_envs:
            raise RuntimeError(
                f"Expected {self.num_envs} DexAnyTwist environment prims, matched {matched_count} for {object_root_expr}."
            )
        return assignment_ids

    @staticmethod
    def _find_name(names: Sequence[str], target: str, default: int | None = None) -> int:
        for i, name in enumerate(names):
            if name == target or name.endswith(target):
                return i
        if default is not None:
            return default
        raise ValueError(f"Could not find '{target}' in names: {names}")

    def _default_joint_pos_in_articulation_order(self, cfg: DexAnyTwistEnvCfg) -> list[float]:
        if len(cfg.default_hand_joint_pos) != len(cfg.actuated_joint_names):
            raise ValueError(
                "default_hand_joint_pos must have one value per actuated joint: "
                f"got {len(cfg.default_hand_joint_pos)} values for {len(cfg.actuated_joint_names)} joints."
            )
        joint_pos_by_name = dict(zip(cfg.actuated_joint_names, cfg.default_hand_joint_pos))
        default_joint_pos = []
        for joint_name in self.hand.joint_names:
            matched_value = 0.0
            for cfg_joint_name, value in joint_pos_by_name.items():
                if joint_name == cfg_joint_name or joint_name.endswith(cfg_joint_name):
                    matched_value = float(value)
                    break
            default_joint_pos.append(matched_value)
        return default_joint_pos

    def _print_hand_joint_limits(self) -> None:
        lower = self.hand_dof_lower_limits[0].detach().cpu().tolist()
        upper = self.hand_dof_upper_limits[0].detach().cpu().tolist()
        actuated_indices = set(self.actuated_dof_indices)
        joint_rows = sorted(
            enumerate(self.hand.joint_names),
            key=lambda item: self._joint_number_sort_key(item[1], item[0]),
        )
        lines = [
            "[DexAnyTwist] hand joint limits from Isaac articulation (env 0):",
            "  dof_idx  joint_name                         lower(rad)   upper(rad)   actuated",
        ]
        for index, joint_name in joint_rows:
            actuated = "yes" if index in actuated_indices else "no"
            lines.append(f"  {index:>7d}   {joint_name:<32} {lower[index]:>10.6f} {upper[index]:>10.6f}   {actuated}")
        print("\n".join(lines), flush=True)

    @staticmethod
    def _joint_number_sort_key(joint_name: str, fallback_index: int) -> tuple[int, int, str]:
        match = re.search(r"joint_?(\d+)", joint_name)
        if match is not None:
            return (0, int(match.group(1)), joint_name)
        return (1, fallback_index, joint_name)

    def _refresh_action_mean(self) -> None:
        self.action_mean.zero_()
        if self.cfg.action_mean_joint_pos is None:
            return
        num_actions = len(self.actuated_dof_indices)
        if len(self.cfg.action_mean_joint_pos) != num_actions:
            raise ValueError(
                "action_mean_joint_pos must have one value per actuated joint: "
                f"got {len(self.cfg.action_mean_joint_pos)} values for {num_actions} joints."
            )
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        mean_joint_pos = torch.tensor(self.cfg.action_mean_joint_pos, dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        mean_joint_pos = torch.clamp(mean_joint_pos, lower, upper)
        self.action_mean[:, :num_actions] = torch.clamp(unscale(mean_joint_pos, lower, upper), -1.0, 1.0)

    def _setup_scene(self):
        self._enable_urdf_importer()
        self._patch_urdf_converter_for_local_importer()
        self.hand = Articulation(self.cfg.hand_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        if not self.cfg.scene.replicate_physics or self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["hand"] = self.hand
        self.scene.articulations["object"] = self.object
        if self.cfg.apply_visual_materials:
            self._apply_visual_materials()
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_visual_materials(self) -> None:
        from pxr import Gf, UsdGeom

        stage = sim_utils.get_current_stage()
        hand_spawn = self.cfg.hand_cfg.spawn
        hand_asset_path = getattr(hand_spawn, "asset_path", None)
        if hasattr(hand_spawn, "assets_cfg"):
            hand_asset_path = hand_spawn.assets_cfg[0].asset_path
        if hand_asset_path is None:
            raise ValueError("Could not determine DexAnyTwist hand asset path for visual material binding.")
        hand_link_colors = self._load_urdf_link_colors(hand_asset_path)
        object_spawn = self.cfg.object_cfg.spawn
        object_asset_path = getattr(object_spawn, "asset_path", None)
        if hasattr(object_spawn, "assets_cfg"):
            object_asset_path = object_spawn.assets_cfg[0].asset_path
        if object_asset_path is None:
            raise ValueError("Could not determine DexAnyTwist object asset path for visual material binding.")
        object_link_colors = self._load_urdf_link_colors(object_asset_path)

        self._bind_urdf_link_materials(stage, "Hand", hand_link_colors, Gf, UsdGeom)
        self._bind_urdf_link_materials(stage, "Object", object_link_colors, Gf, UsdGeom)

    @staticmethod
    def _load_urdf_link_colors(urdf_path: str) -> dict[str, tuple[float, float, float]]:
        import xml.etree.ElementTree as ET
        from pathlib import Path

        root = ET.parse(Path(urdf_path)).getroot()
        named_materials: dict[str, tuple[float, float, float]] = {}
        for material in root.findall("material"):
            name = material.get("name")
            color = DexAnyTwistEnv._read_urdf_color(material)
            if name and color is not None:
                named_materials[name] = color

        link_colors: dict[str, tuple[float, float, float]] = {}
        for link in root.findall("link"):
            link_name = link.get("name")
            if not link_name:
                continue
            for visual in link.findall("visual"):
                material = visual.find("material")
                if material is None:
                    continue
                color = DexAnyTwistEnv._read_urdf_color(material)
                if color is None:
                    material_name = material.get("name")
                    color = named_materials.get(material_name) if material_name else None
                if color is not None:
                    link_colors[link_name] = color
                    break
        return link_colors

    @staticmethod
    def _read_urdf_color(material) -> tuple[float, float, float] | None:
        color = material.find("color")
        if color is None:
            return None
        rgba = color.get("rgba")
        if not rgba:
            return None
        values = [float(value) for value in rgba.split()]
        if len(values) < 3:
            return None
        return (values[0], values[1], values[2])

    @staticmethod
    def _bind_urdf_link_materials(stage, asset_name: str, link_colors: dict[str, tuple[float, float, float]], Gf, UsdGeom):
        import re

        asset_roots = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/envs/env_") and path.endswith(f"/{asset_name}"):
                asset_roots.append(path)

        for asset_root in asset_roots:
            for link_name, color in link_colors.items():
                material_suffix = re.sub(r"[^0-9A-Za-z_]+", "_", f"{asset_name}_{link_name}")
                material_path = f"/World/Looks/gx4m_urdf_{material_suffix}"
                if not stage.GetPrimAtPath(material_path).IsValid():
                    material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5)
                    material_cfg.func(material_path, material_cfg)

                link_paths = DexAnyTwistEnv._find_link_prim_paths(stage, asset_root, link_name)
                for link_path in link_paths:
                    link_prefix = f"{link_path}/"
                    for prim in stage.Traverse():
                        path = str(prim.GetPath())
                        if path == link_path or path.startswith(link_prefix):
                            try:
                                sim_utils.bind_visual_material(path, material_path, stage=stage)
                            except ValueError:
                                pass
                            if prim.IsA(UsdGeom.Gprim):
                                gprim = UsdGeom.Gprim(prim)
                                gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                                gprim.CreateDisplayOpacityAttr([1.0])

    @staticmethod
    def _find_link_prim_paths(stage, asset_root: str, link_name: str) -> list[str]:
        direct_path = f"{asset_root}/{link_name}"
        if stage.GetPrimAtPath(direct_path).IsValid():
            return [direct_path]

        link_paths = []
        root_prefix = f"{asset_root}/"
        link_token = link_name.lower()
        mesh_tokens = (f"{link_name}.stl".lower(), f"{link_name}.obj".lower(), f"{link_name}.dae".lower())
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(root_prefix):
                continue
            leaf_name = path.rsplit("/", 1)[-1].lower()
            if leaf_name == link_token or leaf_name.startswith(f"{link_token}_") or leaf_name.endswith(f"_{link_token}"):
                link_paths.append(path)
                continue
            if DexAnyTwistEnv._prim_mentions_mesh_token(prim, mesh_tokens):
                link_paths.append(path)
        return link_paths

    @staticmethod
    def _prim_mentions_mesh_token(prim, mesh_tokens: Sequence[str]) -> bool:
        haystacks = [str(prim.GetPath()).lower()]
        try:
            haystacks.append(str(prim.GetAssetInfo()).lower())
        except Exception:
            pass
        for metadata_key in ("references", "payload", "assetInfo"):
            try:
                haystacks.append(str(prim.GetMetadata(metadata_key)).lower())
            except Exception:
                pass
        try:
            attributes = prim.GetAttributes()
        except Exception:
            attributes = []
        for attr in attributes:
            try:
                attr_name = attr.GetName().lower()
                haystacks.append(attr_name)
                if not any(key in attr_name for key in ("asset", "file", "path", "uri", "url", "reference", "source")):
                    continue
                value = attr.Get()
            except Exception:
                continue
            if value is None or isinstance(value, (bool, int, float)):
                continue
            text = str(value)
            if len(text) <= 2048:
                haystacks.append(text.lower())
        return any(token in haystack for token in mesh_tokens for haystack in haystacks)

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
        from isaaclab.sim.converters import urdf_converter
        from isaacsim.asset.importer.urdf._urdf import ImportConfig

        if hasattr(ImportConfig(), "set_merge_fixed_ignore_inertia"):
            return
        if getattr(urdf_converter.UrdfConverter, "_gx4m_merge_fixed_compat", False):
            return

        def _get_urdf_import_config_compat(self):
            import_config = ImportConfig()
            import_config.set_distance_scale(1.0)
            import_config.set_make_default_prim(True)
            import_config.set_create_physics_scene(False)

            import_config.set_density(self.cfg.link_density)
            import_config.set_convex_decomp(self.cfg.collider_type == "convex_decomposition")
            import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
            import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)

            import_config.set_fix_base(self.cfg.fix_base)
            import_config.set_self_collision(self.cfg.self_collision)
            import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
            import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)
            return import_config

        urdf_converter.UrdfConverter._get_urdf_import_config = _get_urdf_import_config_compat
        urdf_converter.UrdfConverter._gx4m_merge_fixed_compat = True

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_success_envs(goal_env_ids)
        self.policy_actions = actions.clone().clamp(-1.0, 1.0)
        self.actions = torch.clamp(self.policy_actions + self.action_mean, -1.0, 1.0)

    def _sample_random_force_prob(self, count: int) -> torch.Tensor:
        force_prob_range = torch.tensor(self.cfg.force_prob_range, dtype=torch.float, device=self.device)
        force_prob_range = torch.clamp(force_prob_range, min=1.0e-8)
        log_low = torch.log(force_prob_range[0])
        log_high = torch.log(force_prob_range[1])
        return torch.exp((log_low - log_high) * torch.rand(count, dtype=torch.float, device=self.device) + log_high)

    def _refresh_discretized_dof_targets(self) -> None:
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        steps = torch.linspace(0.0, 1.0, self.cfg.rna_softmax_bins, dtype=torch.float, device=self.device)
        self.discretized_dof_targets[:] = lower[..., None] + (upper - lower)[..., None] * steps

    def _get_random_network_adversary_action(self, canonical_action: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_rna or self.rna_network is None:
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

        object_pose = torch.cat((self.object_link_pos, self.object_link_rot), dim=-1)
        rna_input = torch.cat((self.hand_dof_pos[:, self.actuated_dof_indices], object_pose), dim=-1)
        rand_action_softmax = self.rna_network(rna_input)
        rand_action_indices = torch.argmax(rand_action_softmax, dim=-1)
        rand_joint_targets = torch.gather(self.discretized_dof_targets, 2, rand_action_indices.unsqueeze(-1)).squeeze(-1)

        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        rand_perturbation = unscale(rand_joint_targets, lower, upper)
        self.rna_perturb_mask = (
            torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device)
            < float(self.cfg.rna_action_perturb_prob)
        )
        rand_perturbation = torch.where(self.rna_perturb_mask, rand_perturbation, canonical_action)
        self.rna_alpha = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device)
        action_apply = self.rna_alpha * rand_perturbation + (1.0 - self.rna_alpha) * canonical_action
        action_apply = torch.clamp(action_apply, -1.0, 1.0)
        self.last_rna_action[:, : canonical_action.shape[-1]] = action_apply
        return action_apply

    def _apply_random_object_force(self) -> None:
        if self.cfg.force_scale <= 0.0:
            self.object_external_forces.zero_()
            self.object_external_torques.zero_()
            self.object.permanent_wrench_composer.set_forces_and_torques(
                forces=self.object_external_forces,
                torques=self.object_external_torques,
                body_ids=torch.tensor([self.object_link_body_id], dtype=torch.long, device=self.device),
                is_global=False,
            )
            return

        force_decay = torch.pow(
            torch.tensor(float(self.cfg.force_decay), dtype=torch.float, device=self.device),
            torch.tensor(
                float(self.step_dt / max(float(self.cfg.force_decay_interval), 1.0e-8)),
                dtype=torch.float,
                device=self.device,
            ),
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
            body_ids=torch.tensor([self.object_link_body_id], dtype=torch.long, device=self.device),
            is_global=False,
        )

    def _apply_action(self) -> None:
        lower = self.hand_dof_lower_limits[:, self.actuated_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.actuated_dof_indices]
        action_apply = self._get_random_network_adversary_action(self.actions[:, : len(self.actuated_dof_indices)])
        targets = scale(action_apply, lower, upper)
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * targets
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = torch.clamp(
            self.cur_targets[:, self.actuated_dof_indices], lower, upper
        )
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self._apply_random_object_force()

    def _setup_hand_trajectory_recorder(self) -> None:
        if not self.cfg.save_hand_trajectory:
            return
        object_name = self.cfg.object_name or "dexanytwist"
        extra_metadata = {
            "object_joint_names": np.asarray(list(self.object.joint_names), dtype=str),
            "actuated_joint_names": np.asarray(list(self.cfg.actuated_joint_names), dtype=str),
            "actuated_joint_indices": np.asarray(self.actuated_dof_indices, dtype=np.int64),
            "object_name": np.asarray(str(self.cfg.object_name), dtype=str),
            "object_names": np.asarray(list(self.object_names), dtype=str),
            "object_scale": self.object_scale.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        self.hand_trajectory_sampler = HandTrajectorySampler(
            self,
            task_name="dexanytwist",
            output_dir=self.cfg.hand_trajectory_dir,
            chunk_size=int(self.cfg.hand_trajectory_interval),
            actuated_joint_names=list(self.cfg.actuated_joint_names),
            actuated_joint_indices=self.actuated_dof_indices,
            fingertip_body_names=list(self.cfg.fingertip_body_names),
            fingertip_offsets=self.cfg.fingertip_keypoint_offset,
            extra_metadata=extra_metadata,
            run_label=object_name,
            max_frames=int(self.cfg.hand_trajectory_max_steps),
        )
        self._hand_trajectory_enabled = True
        self.hand_trajectory_path = self.hand_trajectory_sampler.writer.output_dir

    def _record_hand_trajectory_if_needed(self) -> None:
        if not getattr(self, "_hand_trajectory_enabled", False):
            return
        step = int(self.common_step_counter)
        if step <= 0:
            return
        if self._last_hand_trajectory_step == step:
            return
        self._last_hand_trajectory_step = step
        num_actions = len(self.actuated_dof_indices)
        hand_root_pose = torch.cat(
            (
                self.hand.data.root_pos_w - self.scene.env_origins,
                self.hand.data.root_quat_w,
            ),
            dim=-1,
        )
        object_root_pose = torch.cat(
            (
                self.object.data.root_pos_w - self.scene.env_origins,
                self.object.data.root_quat_w,
            ),
            dim=-1,
        )
        extra_fields = {
            "hand_root_pose": hand_root_pose,
            "object_joint_pos": self.object_joint_raw,
            "object_root_pose": object_root_pose,
            "policy_action": self.policy_actions[:, :num_actions],
            "applied_action": self.last_rna_action[:, :num_actions],
        }
        self.hand_trajectory_sampler.record(step, extra_fields=extra_fields)
        if self.hand_trajectory_sampler.complete and self.cfg.hand_trajectory_quit_on_complete:
            raise SystemExit(0)

    def close(self):
        if getattr(self, "hand_trajectory_sampler", None) is not None:
            self.hand_trajectory_sampler.close()
            self.hand_trajectory_sampler = None
        self._hand_trajectory_enabled = False
        super().close()

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        self._record_hand_trajectory_if_needed()
        num_actions = len(self.actuated_dof_indices)
        hand_pos_obs = unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        full_obs = torch.cat(
            (
                hand_pos_obs,
                self.actions[:, :num_actions],
                self.fingertip_pos.reshape(self.num_envs, -1),
                self.class_index,
                self.points_feature,
                self.object_joint_pos,
                self.rotation_axis,
                self.object_keypoints.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )
        policy_obs = (
            torch.cat((hand_pos_obs, self.actions[:, :num_actions]), dim=-1)
            if self.cfg.policy_joint_action_only
            else full_obs
        )
        if policy_obs.shape[1] != self.cfg.observation_space:
            raise RuntimeError(
                "DexAnyTwist policy observation size mismatch: "
                f"got {policy_obs.shape[1]}, expected {self.cfg.observation_space}."
            )
        if full_obs.shape[1] != self.cfg.state_space:
            raise RuntimeError(
                "DexAnyTwist critic observation size mismatch: "
                f"got {full_obs.shape[1]}, expected {self.cfg.state_space}."
            )
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)
        critic_obs = torch.nan_to_num(full_obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        valve_pos = self.object_joint_pos[:, 0]
        progress = torch.clamp(valve_pos - self.target_max_object_joint_pos, min=0.0)
        self.target_max_object_joint_pos = torch.maximum(self.target_max_object_joint_pos, valve_pos)
        self.episode_max_object_joint_pos = torch.maximum(self.episode_max_object_joint_pos, valve_pos)
        action_penalty = torch.sum(self.actions**2, dim=-1)

        action_delta = self.cur_targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]
        action_delta = torch.where(torch.abs(action_delta) < 0.2, torch.zeros_like(action_delta), action_delta)
        action_delta_penalty = torch.sum(action_delta**2, dim=-1)

        hand_dof_vel = self.hand_dof_vel[:, self.actuated_dof_indices].clone()
        hand_dof_vel = torch.where(torch.abs(hand_dof_vel) < 1.0, torch.zeros_like(hand_dof_vel), hand_dof_vel)
        joint_velocity_penalty = torch.sum(hand_dof_vel**2, dim=-1)

        keypoint_distance = torch.where(
            self.keypoints_closest_distance < 0.05,
            torch.zeros_like(self.keypoints_closest_distance),
            self.keypoints_closest_distance,
        )
        keypoints_reward = self.cfg.keypoints_reward_scale * keypoint_distance.mean(dim=1)
        fall_reward = torch.clamp(0.1 / torch.clamp(torch.abs(self.object_joint_pos[:, 1:]).sum(dim=1), min=1.0e-4), max=2.0)

        goal_reached = valve_pos > self.cfg.success_threshold
        self.reset_goal_buf = goal_reached.float()
        self.successes += self.reset_goal_buf
        success_reward = torch.where(goal_reached, torch.full_like(valve_pos, self.cfg.reach_goal_bonus), 0.0)
        near_success_start = self.cfg.success_threshold - float(self.cfg.near_success_margin)
        near_success_progress = torch.clamp(
            (self.episode_max_object_joint_pos - near_success_start)
            / max(float(self.cfg.near_success_margin), 1.0e-6),
            min=0.0,
            max=1.0,
        )

        reward = (
            valve_pos
            + self.cfg.progress_reward_scale * progress
            + self.cfg.max_progress_reward_scale * self.target_max_object_joint_pos
            + self.cfg.success_progress_reward_scale
            * torch.clamp(valve_pos / self.cfg.success_threshold, min=0.0, max=1.0)
            + self.cfg.near_success_reward_scale * near_success_progress
            + self.cfg.success_margin_reward_scale * torch.clamp(valve_pos - self.cfg.success_threshold, min=0.0)
            + self.cfg.action_penalty_scale * action_penalty
            + self.cfg.action_delta_penalty_scale * action_delta_penalty
            + self.cfg.joint_velocity_penalty_scale * joint_velocity_penalty
            + keypoints_reward
            + fall_reward
        )
        reward = reward + success_reward

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
        self.extras["log"]["reward_twist"] = valve_pos.mean()
        self.extras["log"]["reward_progress"] = (self.cfg.progress_reward_scale * progress).mean()
        self.extras["log"]["reward_max_progress"] = (
            self.cfg.max_progress_reward_scale * self.target_max_object_joint_pos
        ).mean()
        self.extras["log"]["reward_action"] = (self.cfg.action_penalty_scale * action_penalty).mean()
        self.extras["log"]["reward_action_delta"] = (
            self.cfg.action_delta_penalty_scale * action_delta_penalty
        ).mean()
        self.extras["log"]["reward_joint_velocity"] = (
            self.cfg.joint_velocity_penalty_scale * joint_velocity_penalty
        ).mean()
        self.extras["log"]["reward_keypoints"] = keypoints_reward.mean()
        self.extras["log"]["reward_fall"] = fall_reward.mean()
        self.extras["log"]["reward_success"] = success_reward.mean()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["keypoints_closest_distance"] = self.keypoints_closest_distance.mean()
        self.extras["log"]["object_joint_pos"] = valve_pos.mean()
        self.extras["log"]["max_object_joint_pos"] = self.episode_max_object_joint_pos.mean()
        self.extras["log"]["max_object_joint_pos_peak"] = self.episode_max_object_joint_pos.max()
        self.extras["log"]["target_max_object_joint_pos"] = self.target_max_object_joint_pos.mean()
        self.extras["log"]["success_rate"] = (valve_pos > self.cfg.success_threshold).float().mean()
        self.extras["log"]["episode_success_rate"] = (self.successes > 0.0).float().mean()
        self.extras["log"]["successes"] = self.successes.mean()
        self.extras["log"]["successes_max"] = self.successes.max()
        self.extras["log"]["object_count"] = torch.tensor(float(self.object_count), dtype=torch.float, device=self.device)
        self.extras["log"]["scale_range_enabled"] = torch.tensor(
            float(self.cfg.enable_scale_range), dtype=torch.float, device=self.device
        )
        self.extras["log"]["spawn_variant_count"] = torch.tensor(
            float(self.spawn_variant_count), dtype=torch.float, device=self.device
        )
        self.extras["log"]["object_scale_mean"] = self.object_scale.mean()
        self.extras["log"]["object_scale_range"] = self.object_scale.max() - self.object_scale.min()
        self.extras["log"]["object_external_force_norm"] = torch.norm(
            self.object_external_forces[:, 0, :], p=2, dim=-1
        ).mean()
        self.extras["log"]["object_external_force_prob_mean"] = self.random_force_prob.mean()
        self.extras["log"]["rna_action_delta_norm"] = torch.norm(
            self.last_rna_action[:, : len(self.actuated_dof_indices)]
            - self.actions[:, : len(self.actuated_dof_indices)],
            p=2,
            dim=-1,
        ).mean()
        self.extras["log"]["rna_perturb_rate"] = self.rna_perturb_mask.float().mean()
        if self.object_count > 1:
            per_object_success_rate = []
            per_object_max_joint = []
            for object_index in range(self.object_count):
                mask = self.object_spec_ids == object_index
                if torch.any(mask):
                    per_object_success_rate.append((self.successes[mask] > 0.0).float().mean())
                    per_object_max_joint.append(self.episode_max_object_joint_pos[mask].mean())
            if per_object_success_rate:
                self.extras["log"]["per_object_success_rate_min"] = torch.stack(per_object_success_rate).min()
                self.extras["log"]["per_object_max_object_joint_pos_min"] = torch.stack(per_object_max_joint).min()

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_fall = torch.any(torch.abs(self.object_joint_pos[:, 1:]) > self.cfg.object_fall_threshold, dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return object_fall, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        num_resets = len(env_ids_t)
        self._refresh_randomized_runtime_state(env_ids_t)

        hand_pos = self.default_hand_joint_pos[env_ids_t].clone()
        lower = self.hand_dof_lower_limits[env_ids_t]
        upper = self.hand_dof_upper_limits[env_ids_t]
        rand = torch.empty((num_resets, self.num_hand_dofs), device=self.device).uniform_(-1.0, 1.0)
        rand_delta = (lower - hand_pos) + (upper - lower) * 0.3 * (rand + 1.0)
        hand_pos = hand_pos + self.cfg.reset_dof_pos_noise * rand_delta
        hand_vel = self.cfg.reset_dof_vel_noise * torch.empty_like(hand_pos).uniform_(-1.0, 1.0)

        self.prev_targets[env_ids_t] = hand_pos
        self.cur_targets[env_ids_t] = hand_pos
        self.hand.set_joint_position_target(hand_pos, env_ids=env_ids_t)
        self.hand.write_joint_state_to_sim(hand_pos, hand_vel, env_ids=env_ids_t)

        self._reset_object_roots(env_ids_t)
        self._reset_object_joints(env_ids_t)

        episode_success = self.successes[env_ids_t] > 0.0
        self.successes[env_ids_t] = 0.0
        self.target_max_object_joint_pos[env_ids_t] = 0.0
        self.episode_max_object_joint_pos[env_ids_t] = 0.0
        self.reset_goal_buf[env_ids_t] = 0.0
        if getattr(self, "hand_trajectory_sampler", None) is not None:
            self.hand_trajectory_sampler.mark_reset(env_ids_t, episode_success)
        self._compute_intermediate_values()

    def _refresh_randomized_runtime_state(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        current_joint_limits = self.hand.data.joint_pos_limits.to(self.device)
        self.hand_dof_lower_limits = current_joint_limits[..., 0]
        self.hand_dof_upper_limits = current_joint_limits[..., 1]
        self._refresh_discretized_dof_targets()
        self._refresh_action_mean()

        self.object_mass[env_ids] = self.object.root_physx_view.get_masses().to(self.device)[env_ids]
        self.random_force_prob[env_ids] = self._sample_random_force_prob(len(env_ids))
        self.object_external_forces[env_ids] = 0.0
        self.object_external_torques[env_ids] = 0.0
        self.object.permanent_wrench_composer.set_forces_and_torques(
            forces=self.object_external_forces[env_ids],
            torques=self.object_external_torques[env_ids],
            body_ids=torch.tensor([self.object_link_body_id], dtype=torch.long, device=self.device),
            env_ids=env_ids,
            is_global=False,
        )

    def _reset_object_roots(self, env_ids: torch.Tensor) -> None:
        root_pose = self.object_default_root_pose[env_ids].clone()
        root_pose[:, :3] += self.cfg.reset_position_noise * torch.empty(
            (len(env_ids), 3), dtype=torch.float, device=self.device
        ).uniform_(-1.0, 1.0)
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = torch.zeros((len(env_ids), 6), dtype=torch.float, device=self.device)
        self.object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    def _reset_object_joints(self, env_ids: torch.Tensor) -> None:
        object_pos = torch.zeros((len(env_ids), self.object.num_joints), dtype=torch.float, device=self.device)
        object_vel = torch.zeros_like(object_pos)
        self.object.write_joint_state_to_sim(object_pos, object_vel, env_ids=env_ids)
        self.target_max_object_joint_pos[env_ids] = 0.0
        self.reset_goal_buf[env_ids] = 0.0

    def _reset_success_envs(self, env_ids: torch.Tensor) -> None:
        self._reset_object_joints(env_ids)
        self.random_force_prob[env_ids] = self._sample_random_force_prob(len(env_ids))
        self.object_external_forces[env_ids] = 0.0
        self.object_external_torques[env_ids] = 0.0
        self.object.permanent_wrench_composer.set_forces_and_torques(
            forces=self.object_external_forces[env_ids],
            torques=self.object_external_torques[env_ids],
            body_ids=torch.tensor([self.object_link_body_id], dtype=torch.long, device=self.device),
            env_ids=env_ids,
            is_global=False,
        )
        if self.cfg.reset_phase_on_success:
            self.episode_length_buf[env_ids] = 0
        if not self.cfg.reset_hand_on_success:
            return

        hand_pos = self.default_hand_joint_pos[env_ids].clone()
        hand_vel = torch.zeros_like(hand_pos)
        self.prev_targets[env_ids] = hand_pos
        self.cur_targets[env_ids] = hand_pos
        self.hand.set_joint_position_target(hand_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(hand_pos, hand_vel, env_ids=env_ids)

    def _compute_intermediate_values(self) -> None:
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.object_joint_raw = self.object.data.joint_pos
        self.object_joint_pos = torch.stack(
            (
                self.object_joint_raw[:, self.object_joint_ids["joint1"]],
                self.object_joint_raw[:, self.object_joint_ids["joint1_rx"]],
                self.object_joint_raw[:, self.object_joint_ids["joint1_ry"]],
            ),
            dim=-1,
        )
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_body_ids] - self.scene.env_origins[:, None, :]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.fingertip_body_ids]
        self.object_link_pos = self.object.data.body_pos_w[:, self.object_link_body_id] - self.scene.env_origins
        self.object_link_rot = self.object.data.body_quat_w[:, self.object_link_body_id]
        self.fingertip_keypoints = self.fingertip_pos + quat_apply(self.fingertip_rot, self.fingertip_offset)
        self.object_keypoints = self.object_link_pos[:, None, :] + self.keypoint_bias
        self.keypoints_closest_distance = torch.cdist(self.fingertip_keypoints, self.object_keypoints, p=2).min(
            dim=2
        ).values
        self._update_keypoint_markers()

    def _update_keypoint_markers(self) -> None:
        if self.object_keypoint_markers is None and self.fingertip_keypoint_markers is None:
            return
        env_origins = self.scene.env_origins[:, None, :]
        if self.object_keypoint_markers is not None:
            object_keypoints_w = (self.object_keypoints + env_origins).reshape(-1, 3)
            self.object_keypoint_markers.visualize(translations=object_keypoints_w)
        if self.fingertip_keypoint_markers is not None:
            fingertip_keypoints_w = (self.fingertip_keypoints + env_origins).reshape(-1, 3)
            self.fingertip_keypoint_markers.visualize(translations=fingertip_keypoints_w)

def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0
