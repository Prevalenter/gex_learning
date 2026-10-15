"""Shared direction state for cached-grasp rotation; no additional task registration."""
import math
import torch
from isaaclab.utils import configclass
from .hand_env import Gx4mHandEnv, Gx4mHandEnvCfg
from ..rotation_metrics import direction_net_rotation_deviations


def direction_rotation_failure_masks(reverse_angle, cross_axis_angle, failure_angle):
    return reverse_angle >= failure_angle, cross_axis_angle >= failure_angle


@configclass
class Gx4mRotationBaseEnvCfg(Gx4mHandEnvCfg):
    training_phase = "direction"
    direction_axis = "z"
    direction_object_ids: list[str] = []
    direction_failure_angle = math.radians(80.0)
    enable_direction_cross_axis_termination = False
    position_reward_scale = 10.0
    success_tolerance = 0.25
    reach_goal_bonus = 250.0
    max_consecutive_success = 0


class Gx4mRotationBaseEnv(Gx4mHandEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.angular_velocity_command = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)
        self.command_switch_event = torch.zeros(self.num_envs, device=self.device)
        self.direction_initial_object_rot = self.object_rot.clone()
        self.direction_signed_progress_angle = torch.zeros(self.num_envs, device=self.device)
        self._direction_progress_last_step = -1
        self.goal_rot[:] = self.object_rot
        self._sample_commands(torch.arange(self.num_envs, device=self.device), initial=True)
        self._critic_command_obs_start = (
            self._full_observation_size_from_counts(len(self.actuated_dof_indices), self.num_fingertips)
            + 3 * self.num_fingertips + int(self.cfg.dr_state_space) + 3)
        if self._critic_command_obs_start + 3 > int(cfg.state_space):
            raise RuntimeError("No critic padding available for the rotation command")
        self._reset_idx(None)

    @property
    def control_dt(self):
        return float(self.cfg.sim.dt) * int(self.cfg.decimation)

    def _reset_target_pose(self, env_ids):
        if hasattr(self, "angular_velocity_command"):
            self.reset_goal_buf[env_ids] = False
        else:
            super()._reset_target_pose(env_ids)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self.command_switch_event.zero_()

    def _get_observations(self):
        observations = super()._get_observations()
        observations["policy"][:, 70:73] = self.angular_velocity_command
        start = self._critic_command_obs_start
        observations["critic"][:, start:start+3] = self.angular_velocity_command
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        if self.cfg.training_phase == "direction" and hasattr(self, "direction_initial_object_rot"):
            reverse_deviation, cross_axis_deviation = direction_net_rotation_deviations(
                self.object_rot,
                self.direction_initial_object_rot,
                self.angular_velocity_command,
                self.direction_signed_progress_angle,
            )
            reverse_failure, cross_axis_failure = direction_rotation_failure_masks(
                reverse_deviation,
                cross_axis_deviation,
                float(self.cfg.direction_failure_angle),
            )
            cross_axis_exceeded = cross_axis_failure
            cross_axis_failure = cross_axis_failure & self.cfg.enable_direction_cross_axis_termination
            terminated = terminated | reverse_failure | cross_axis_failure
            log = self.extras.setdefault("log", {})
            log["direction_reverse_failure_rate"] = reverse_failure.float().mean()
            log["direction_cross_axis_failure_rate"] = cross_axis_failure.float().mean()
            log["direction_cross_axis_threshold_exceed_rate"] = cross_axis_exceeded.float().mean()
            log["direction_reverse_deviation"] = reverse_deviation.mean()
            log["direction_cross_axis_deviation"] = cross_axis_deviation.mean()
        return terminated, truncated
