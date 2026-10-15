"""Rigidly transform cached grasps; all poses are environment-local, wxyz."""
import torch


def multiply_quaternions(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw - (av*bv).sum(-1, keepdim=True),
                      aw*bv + bw*av + torch.linalg.cross(av, bv)), -1)


def rotate_vector(q, v):
    cross = 2 * torch.linalg.cross(q[..., 1:], v)
    return v + q[..., :1]*cross + torch.linalg.cross(q[..., 1:], cross)


def transform_grasp(nominal_hand_pose, object_poses, translation, delta_rotation):
    """Rotate the entire grasp about the hand base, then translate it."""
    nominal = nominal_hand_pose.expand(len(object_poses), -1)
    hand_pos = nominal[:, :3] + translation
    hand_quat = multiply_quaternions(delta_rotation, nominal[:, 3:])
    object_pos = hand_pos + rotate_vector(delta_rotation, object_poses[:, :3] - nominal[:, :3])
    object_quat = multiply_quaternions(delta_rotation, object_poses[:, 3:])
    return torch.cat((hand_pos, hand_quat), -1), torch.cat((object_pos, object_quat), -1)


def sample_base_transform(count, position_range_m, device):
    # A normalized isotropic Gaussian in R4 is uniform on S3, hence uniform SO(3).
    rotation = torch.randn((count, 4), device=device)
    rotation /= rotation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rotation = torch.where(rotation[:, :1] < 0, -rotation, rotation)
    translation = (2*torch.rand((count, 3), device=device)-1) * position_range_m
    return translation, rotation


def rotation_axis_world(axis, base_quaternion, frame):
    """Express a configured unit axis in world coordinates for reward projection."""
    axis = torch.as_tensor(axis, device=base_quaternion.device, dtype=base_quaternion.dtype)
    axis = (axis / axis.norm()).expand(base_quaternion.shape[0], -1)
    if frame == "wrist":
        return rotate_vector(base_quaternion, axis)
    if frame == "world":
        return axis
    raise ValueError("Rotation axis frame must be wrist or world")
