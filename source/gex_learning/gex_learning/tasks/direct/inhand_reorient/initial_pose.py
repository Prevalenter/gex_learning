"""Simulation-independent storage for the Cubic initial grasp editor."""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from ..dexanytwist.dataset import load_hand_spec

DEFAULT_PATH = Path(__file__).with_name("config") / "initial_pose.json"
ROOT = Path(__file__).resolve().parents[6]
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 17))


def joint_limits():
    root = ET.parse(load_hand_spec("gx4m").urdf_path).getroot()
    result = {}
    for name in JOINT_NAMES:
        limit = root.find(f"joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"Missing joint limits: {name}")
        result[name] = (float(limit.get("lower")), float(limit.get("upper")))
    return result


def _vector(raw, count, label):
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        raise ValueError(f"{label} requires {count} numbers")
    result = [float(v) for v in raw]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} must contain finite numbers")
    return result


def validate_pose(data):
    if data.get("version") != 1:
        raise ValueError("Unsupported initial pose version; expected 1")
    if type(data.get("enabled")) is not bool:
        raise ValueError("enabled must be a boolean")
    joints = data.get("joint_positions_rad", {})
    if set(joints) != set(JOINT_NAMES):
        raise ValueError("Initial pose must contain joint1 through joint16 exactly")
    limits = joint_limits()
    q = {}
    for name in JOINT_NAMES:
        value = float(joints[name])
        low, high = limits[name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be within [{low}, {high}] radians")
        q[name] = value
    position = _vector(data.get("object_position_m"), 3, "object_position_m")
    rotation = _vector(data.get("object_orientation_wxyz"), 4, "object_orientation_wxyz")
    norm = math.sqrt(sum(v * v for v in rotation))
    if norm < 1e-8:
        raise ValueError("Object orientation must be a nonzero quaternion")
    return dict(version=1, enabled=data["enabled"], joint_positions_rad=q,
                object_position_m=position, object_orientation_wxyz=[v / norm for v in rotation])


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_pose(path=DEFAULT_PATH):
    path = resolve_path(path)
    return validate_pose(json.loads(path.read_text(encoding="utf-8")))


def save_pose(data, path=DEFAULT_PATH):
    data = validate_pose(data)
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def apply_initial_pose(cfg, joint_names):
    """Freeze saved values into this environment configuration before spawning.

    A disabled file preserves the original random initialization. Saving a pose
    uses exact joint/object resets; target and physics randomization are separate.
    """
    if not cfg.use_initial_pose_config:
        return
    data = load_pose(cfg.initial_pose_file)
    if not data["enabled"]:
        return
    if set(joint_names) != set(JOINT_NAMES):
        raise ValueError("The initial pose editor supports GX4M joints only")
    cfg.default_hand_joint_pos = [data["joint_positions_rad"][name] for name in joint_names]
    cfg.object_start_pos = data["object_position_m"]
    cfg.object_start_rot = data["object_orientation_wxyz"]
    cfg.reset_position_noise = 0.0
    cfg.reset_position_noise_z = 0.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.reset_object_rotation_noise = 0.0
    cfg.robot_cfg.init_state.joint_pos = dict(data["joint_positions_rad"])
    # Values are now captured by params/env.yaml. Do not reload a later editor
    # change when reconstructing this already-resolved configuration.
    cfg.use_initial_pose_config = False
