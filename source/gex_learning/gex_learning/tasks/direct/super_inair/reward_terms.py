"""HORA-style auxiliary costs for the cached-grasp rotation task."""
import torch
import math


def cross_axis_orientation_cost(current, initial, axis, threshold_deg=5.0):
    """Swing of q_current * inverse(q_initial), ignoring twist about world axis.

    Returns squared excess normalized by the deadband, and swing in radians.
    wxyz quaternion signs and full turns do not affect the result.
    """
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    initial = initial / initial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, v = current[..., :1], current[..., 1:]
    iw, iv = initial[..., :1], initial[..., 1:]
    dw = w * iw + (v * iv).sum(-1, keepdim=True)
    dv = -w * iv + iw * v - torch.linalg.cross(v, iv)
    parallel = (dv * axis).sum(-1, keepdim=True)
    perpendicular = (dv - parallel * axis).norm(dim=-1)
    twist_norm = torch.sqrt(dw.square() + parallel.square()).squeeze(-1)
    angle = 2 * torch.atan2(perpendicular, twist_norm)
    threshold = math.radians(threshold_deg)
    return ((angle - threshold).clamp_min(0) / threshold).square(), angle


def object_position_cost(com_position, initial_com_position, reference_m):
    """Normalized squared COM displacement and its distance in meters."""
    displacement = torch.linalg.vector_norm(com_position - initial_com_position, dim=-1)
    return (displacement / reference_m).square(), displacement


def grasp_stability_costs(object_position, previous_object_position, control_dt,
                          joint_position, initial_joint_position):
    """Finite-difference L1 linear speed and squared distance from reset hand pose."""
    linear_speed = torch.linalg.vector_norm(
        (object_position - previous_object_position) / control_dt, ord=1, dim=-1)
    pose_deviation = (joint_position - initial_joint_position).square().sum(dim=-1)
    return linear_speed, pose_deviation


def rotation_velocity(current, previous, dt):
    """World angular velocity from shortest wxyz quaternion displacement (HORA)."""
    w, v = current[:, :1], current[:, 1:]
    pw, pv = previous[:, :1], previous[:, 1:]
    delta = torch.cat((w * pw + (v * pv).sum(-1, keepdim=True),
                       -w * pv + pw * v - torch.linalg.cross(v, pv)), dim=-1)
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    delta = torch.where(delta[:, :1] < 0, -delta, delta)
    norm = delta[:, 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, delta[:, :1])
    factor = torch.where(norm > 1e-8, angle / norm.clamp_min(1e-12), 2.0)
    return delta[:, 1:] * factor / dt


def hora_reward_terms(angular_velocity, axis, linear_speed, pose_deviation,
                      torque, joint_velocity, clip_min=-0.5, clip_max=0.5, reverse_multiplier=1.0,
                      clip_reverse=True):
    """Unweighted HORA terms; work is squared total power, not sum of squares."""
    signed_speed = (angular_velocity * axis).sum(-1)
    rotation = signed_speed.clamp(clip_min, clip_max) if clip_reverse else signed_speed.clamp(max=clip_max)
    rotation = torch.where(rotation < 0, rotation * reverse_multiplier, rotation)
    return dict(rotation=rotation,
                object_linvel=linear_speed, initial_hand_pose=pose_deviation,
                torque=torque.square().sum(-1),
                work=(torque * joint_velocity).sum(-1).square())


def fingertip_contact_timeout(missed_steps, forces, control_dt, timeout_s, force_threshold):
    """One update per control step; touching resets each finger independently."""
    touching = forces.norm(dim=-1) > force_threshold
    steps = torch.where(touching, torch.zeros_like(missed_steps), missed_steps + 1)
    # Integer boundary avoids float32 6 * (1/30) spuriously exceeding 0.2 s.
    allowed_steps = math.floor(timeout_s / control_dt + 1e-9)
    overdue = steps > allowed_steps
    return steps, overdue
