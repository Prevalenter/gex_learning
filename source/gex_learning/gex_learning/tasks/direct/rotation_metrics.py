"""Quaternion metrics shared by base-task evaluation."""
from __future__ import annotations
import torch

def direction_net_rotation_deviations(
    object_orientation: torch.Tensor,
    initial_orientation: torch.Tensor,
    command: torch.Tensor,
    signed_progress_angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure net reverse progress and pose error from the commanded rotation path."""
    command_norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    active = command_norm.squeeze(-1) > 1.0e-6
    command_axis = command / command_norm.clamp_min(1.0e-6)
    expected_orientation = integrate_world_angular_velocity(
        initial_orientation,
        command_axis * signed_progress_angle.unsqueeze(-1),
        1.0,
    )

    qw, qx, qy, qz = object_orientation.unbind(dim=-1)
    iw, ix, iy, iz = expected_orientation.unbind(dim=-1)
    relative = torch.stack(
        (
            qw * iw + qx * ix + qy * iy + qz * iz,
            -qw * ix + qx * iw - qy * iz + qz * iy,
            -qw * iy + qx * iz + qy * iw - qz * ix,
            -qw * iz - qx * iy + qy * ix + qz * iw,
        ),
        dim=-1,
    )
    relative = relative / torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1.0e-8)
    relative = torch.where(relative[:, :1] < 0.0, -relative, relative)

    vector_norm = torch.linalg.vector_norm(relative[:, 1:], dim=-1)
    cross_axis_deviation = 2.0 * torch.atan2(vector_norm, relative[:, 0].clamp_min(0.0))
    reverse_deviation = torch.relu(-signed_progress_angle)
    reverse_deviation = torch.where(active, reverse_deviation, torch.zeros_like(reverse_deviation))
    cross_axis_deviation = torch.where(
        active, cross_axis_deviation, torch.zeros_like(cross_axis_deviation)
    )
    return reverse_deviation, cross_axis_deviation


def integrate_world_angular_velocity(
    orientation_wxyz: torch.Tensor, angular_velocity_xyz: torch.Tensor, dt: float
) -> torch.Tensor:
    """Left-multiply a quaternion by the world-frame angular displacement."""
    delta = angular_velocity_xyz * float(dt)
    angle = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(angle > 1.0e-8, torch.sin(half_angle) / angle, 0.5 - angle.square() / 48.0)
    delta_q = torch.cat((torch.cos(half_angle), delta * scale), dim=-1)
    w1, x1, y1, z1 = delta_q.unbind(-1)
    w2, x2, y2, z2 = orientation_wxyz.unbind(-1)
    result = torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1.0e-8)
