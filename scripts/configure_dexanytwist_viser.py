"""Lightweight viser configurator for the GX4M DexAnyTwist task.

This tool previews the hand/object URDFs without launching Isaac Sim and writes
the selected object's DexAnyTwist defaults back to twist_anything JSON.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "gex_learning"
DEXANYTWIST_DIR = SOURCE_ROOT / "gex_learning" / "tasks" / "direct" / "dexanytwist"
DEXANYTWIST_ENV_PATH = DEXANYTWIST_DIR / "dexanytwist_env.py"
TWIST_CONFIG_DIR = DEXANYTWIST_DIR / "config" / "twist_anything"
ASSET_DIR = REPO_ROOT / "asset"
TWIST_ASSET_ROOT = ASSET_DIR / "twist_dataset"
TWIST_URDF_ROOT = TWIST_ASSET_ROOT / "twist_dataset" / "twist_dataset_urdf"
GX4M_URDF = ASSET_DIR / "gx4m" / "urdf" / "gx4m.urdf"
HAND_MESH_DIR = ASSET_DIR / "gx4m" / "meshes"
MAX_KEYPOINT_CONTROLS = 8
OBJECT_KEYPOINT_ANCHOR_PATH = "/object/isaaclab_link1_anchor"

DATASET_PATH = DEXANYTWIST_DIR / "dataset.py"
_DATASET_SPEC = importlib.util.spec_from_file_location("_gx4m_dexanytwist_dataset", DATASET_PATH)
if _DATASET_SPEC is None or _DATASET_SPEC.loader is None:
    raise RuntimeError(f"Could not load dataset module from {DATASET_PATH}")
_DATASET_MODULE = importlib.util.module_from_spec(_DATASET_SPEC)
sys.modules[_DATASET_SPEC.name] = _DATASET_MODULE
_DATASET_SPEC.loader.exec_module(_DATASET_MODULE)

HandSpec = _DATASET_MODULE.HandSpec
TwistObjectSpec = _DATASET_MODULE.TwistObjectSpec
load_hand_spec = _DATASET_MODULE.load_hand_spec
load_twist_object = _DATASET_MODULE.load_twist_object


@dataclass
class Defaults:
    object_name: str
    object_pose_offset: list[float]
    hand_pose_offset: list[float]
    hand_start_pos: list[float]
    hand_start_rot_xyz_deg: list[float]
    hand_rot_euler_deg: list[float]
    action_mean_joint_pos: list[float]


@dataclass
class ObjectPreviewSpec:
    name: str
    category: str
    urdf_path: Path
    pose: tuple[float, float, float]
    link1_origin: tuple[float, float, float]
    keypoints: list[list[float]]
    hand_pose_offset: tuple[float, float, float]
    hand_rot_euler_deg: tuple[float, float, float]
    action_mean_joint_pos: tuple[float, ...]
    registered: bool
    points_feature_path: Path | None = None


@dataclass
class ConfigState:
    object_name: str
    object_pose_offset: list[float]
    hand_pose_offset: list[float]
    hand_rot_euler_deg: list[float]
    action_mean_joint_pos: list[float]
    object_keypoints: list[list[float]] | None = None
    object_registered: bool = True
    object_urdf_path: str | None = None
    last_saved_path: Path | None = None


def _eval_static_expr(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_eval_static_expr(item, names) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -float(_eval_static_expr(node.operand, names))
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"unknown name: {node.id}")
        return names[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple"}:
        if len(node.args) != 1 or node.keywords:
            raise ValueError("only one-argument list()/tuple() calls are supported")
        value = _eval_static_expr(node.args[0], names)
        return list(value) if node.func.id == "list" else tuple(value)
    raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")


def load_env_defaults() -> Defaults:
    """Read simple DexAnyTwistEnvCfg defaults without importing Isaac Lab."""
    tree = ast.parse(DEXANYTWIST_ENV_PATH.read_text(encoding="utf-8"), filename=str(DEXANYTWIST_ENV_PATH))
    names: dict[str, Any] = {}
    cfg_values: dict[str, Any] = {}

    for node in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        if target_name is None or value_node is None:
            continue
        try:
            names[target_name] = _eval_static_expr(value_node, names)
        except ValueError:
            continue

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "DexAnyTwistEnvCfg":
            continue
        for stmt in node.body:
            target_name = None
            value_node = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                value_node = stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
                value_node = stmt.value
            if target_name is None or value_node is None:
                continue
            try:
                cfg_values[target_name] = _eval_static_expr(value_node, names)
            except ValueError:
                continue
        break

    action_mean_joint_pos = cfg_values.get("action_mean_joint_pos")
    if action_mean_joint_pos is None:
        action_mean_joint_pos = [0.0] * 16

    return Defaults(
        object_name=str(cfg_values.get("object_name", names.get("DEFAULT_OBJECT_NAME", "bulb_custom_0017"))),
        object_pose_offset=[float(v) for v in cfg_values.get("object_pose_offset", [0.0, 0.0, 0.04])],
        hand_pose_offset=[float(v) for v in cfg_values.get("hand_pose_offset", [0.0, 0.0, 0.0])],
        hand_start_pos=[float(v) for v in names.get("HAND_START_POS", [0.0, 0.0, 0.65])],
        hand_start_rot_xyz_deg=[float(v) for v in cfg_values.get("hand_start_rot_xyz_deg", [90.0, 90.0, 0.0])],
        hand_rot_euler_deg=[float(v) for v in cfg_values.get("hand_rot_euler_deg", [20.0, 0.0, 0.0])],
        action_mean_joint_pos=[float(v) for v in action_mean_joint_pos],
    )


def discover_twist_objects() -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for json_path in sorted(TWIST_CONFIG_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            categories[json_path.stem] = sorted(str(name) for name in data.keys())
    return categories


def _normalize_object_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _registered_object_category(name: str) -> str | None:
    for json_path in sorted(TWIST_CONFIG_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and name in data:
            return json_path.stem
    return None


def _asset_object_urdf(name: str) -> tuple[str, Path] | None:
    if not TWIST_URDF_ROOT.exists():
        return None
    for category_dir in sorted(path for path in TWIST_URDF_ROOT.iterdir() if path.is_dir()):
        object_dir = category_dir / name
        urdf_dir = object_dir / "urdf"
        if not urdf_dir.exists():
            continue
        urdf_paths = sorted(urdf_dir.glob("*.urdf"))
        if urdf_paths:
            return category_dir.name, urdf_paths[0]
    return None


def _category_default_raw_config(category: str) -> dict[str, Any] | None:
    config_path = TWIST_CONFIG_DIR / f"{category}.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        return None
    preferred_key = f"{category}_0028"
    if preferred_key in data:
        return data[preferred_key]
    return next(iter(data.values()))


def _keypoints_from_raw_config(raw: dict[str, Any] | None) -> list[list[float]]:
    if raw is None:
        return [[0.0, 0.03, 0.08], [0.0, -0.03, 0.08], [-0.03, 0.0, 0.08]]
    biases = raw.get("kp_bais", [])
    center = raw.get("kp_center", [[0.0, 0.0, 0.0]])
    cx, cy, cz = (float(value) for value in center[0])
    return [[float(point[0]) + cx, float(point[1]) + cy, float(point[2]) + cz] for point in biases]


def _raw_float_vector(raw: dict[str, Any] | None, key: str, default: tuple[float, ...], length: int) -> tuple[float, ...]:
    if raw is None or key not in raw:
        values = list(default)
    else:
        values = [float(value) for value in raw[key]]
    if len(values) < length:
        values.extend(default[len(values) : length])
    return tuple(float(value) for value in values[:length])


def _identity_rotation() -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rpy_rotation(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _rotation_mul(
    lhs: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    rhs: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return tuple(
        tuple(sum(lhs[row][inner] * rhs[inner][col] for inner in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _rotation_apply(
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(rotation[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _transform_mul(
    lhs: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        tuple[float, float, float],
    ],
    rhs: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        tuple[float, float, float],
    ],
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    tuple[float, float, float],
]:
    lhs_rot, lhs_pos = lhs
    rhs_rot, rhs_pos = rhs
    rotated_rhs_pos = _rotation_apply(lhs_rot, rhs_pos)
    return (
        _rotation_mul(lhs_rot, rhs_rot),
        (
            lhs_pos[0] + rotated_rhs_pos[0],
            lhs_pos[1] + rotated_rhs_pos[1],
            lhs_pos[2] + rotated_rhs_pos[2],
        ),
    )


def _origin_from_joint(joint: ET.Element) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    tuple[float, float, float],
]:
    origin = joint.find("origin")
    if origin is None:
        return (_identity_rotation(), (0.0, 0.0, 0.0))
    xyz = tuple(float(value) for value in origin.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(value) for value in origin.attrib.get("rpy", "0 0 0").split())
    if len(xyz) != 3 or len(rpy) != 3:
        return (_identity_rotation(), (0.0, 0.0, 0.0))
    return (_rpy_rotation(rpy), xyz)  # type: ignore[arg-type]


def object_link_origin(urdf_path: Path, link_name: str = "link1") -> tuple[float, float, float]:
    """Return target link origin in the URDF root frame at zero joint position."""
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return (0.0, 0.0, 0.0)

    links = {link.attrib.get("name", "") for link in root.iter("link") if link.attrib.get("name")}
    if link_name not in links:
        return (0.0, 0.0, 0.0)

    joints_by_parent: dict[str, list[tuple[str, Any]]] = {}
    child_links: set[str] = set()
    for joint in root.iter("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.attrib.get("link")
        child = child_el.attrib.get("link")
        if not parent or not child:
            continue
        joints_by_parent.setdefault(parent, []).append((child, _origin_from_joint(joint)))
        child_links.add(child)

    root_links = sorted(links - child_links)
    if not root_links:
        return (0.0, 0.0, 0.0)
    root_link = "world" if "world" in root_links else root_links[0]
    transforms = {root_link: (_identity_rotation(), (0.0, 0.0, 0.0))}
    stack = [root_link]
    while stack:
        parent = stack.pop()
        parent_transform = transforms[parent]
        for child, joint_transform in joints_by_parent.get(parent, []):
            if child in transforms:
                continue
            transforms[child] = _transform_mul(parent_transform, joint_transform)
            stack.append(child)
    return transforms.get(link_name, (_identity_rotation(), (0.0, 0.0, 0.0)))[1]


def load_object_preview_spec(name: str) -> ObjectPreviewSpec:
    normalized = _normalize_object_name(name)
    registered_category = _registered_object_category(normalized)
    if registered_category is not None:
        spec = load_twist_object(normalized)
        points_feature_path = getattr(spec, "points_feature_path", None)
        return ObjectPreviewSpec(
            name=spec.name,
            category=registered_category,
            urdf_path=spec.urdf_path,
            pose=spec.pose,
            link1_origin=object_link_origin(spec.urdf_path),
            keypoints=[[float(value) for value in point] for point in spec.keypoint_bias.tolist()],
            hand_pose_offset=tuple(float(value) for value in spec.hand_pose_offset),
            hand_rot_euler_deg=tuple(float(value) for value in spec.hand_rot_euler_deg),
            action_mean_joint_pos=tuple(float(value) for value in spec.action_mean_joint_pos),
            registered=True,
            points_feature_path=points_feature_path,
        )

    asset_match = _asset_object_urdf(normalized)
    if asset_match is None:
        raise KeyError(f"Object '{normalized}' was not found in twist configs or asset URDF directories.")
    category, urdf_path = asset_match
    raw = _category_default_raw_config(category)
    if raw is None:
        pose = (0.0, -0.04, -0.08)
    else:
        pose = (float(raw.get("dx", 0.0)), float(raw.get("dy", -0.07)) + 0.03, float(raw.get("dz", -0.08)))
    return ObjectPreviewSpec(
        name=normalized,
        category=category,
        urdf_path=urdf_path,
        pose=pose,
        link1_origin=object_link_origin(urdf_path),
        keypoints=_keypoints_from_raw_config(raw),
        hand_pose_offset=_raw_float_vector(raw, "hand_pose_offset", (0.0, 0.0, 0.0), 3),
        hand_rot_euler_deg=_raw_float_vector(raw, "hand_rot_euler_deg", (0.0, 0.0, 0.0), 3),
        action_mean_joint_pos=_raw_float_vector(raw, "action_mean_joint_pos", (0.0,) * 16, 16),
        registered=False,
    )


def _object_start_pos(spec: ObjectPreviewSpec, defaults: Defaults, state: ConfigState) -> tuple[float, float, float]:
    return (
        defaults.hand_start_pos[0] + float(spec.pose[0]) + state.object_pose_offset[0],
        defaults.hand_start_pos[1] + float(spec.pose[1]) + state.object_pose_offset[1],
        defaults.hand_start_pos[2] + float(spec.pose[2]) + state.object_pose_offset[2],
    )


def _vec3_add(lhs: list[float] | tuple[float, float, float], rhs: list[float]) -> tuple[float, float, float]:
    return (float(lhs[0]) + float(rhs[0]), float(lhs[1]) + float(rhs[1]), float(lhs[2]) + float(rhs[2]))


def _normalize_quat(quat: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in quat))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(float(value) / norm for value in quat)  # type: ignore[return-value]


def _quat_mul(lhs: list[float] | tuple[float, float, float, float], rhs: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(value) for value in lhs)
    rw, rx, ry, rz = (float(value) for value in rhs)
    return _normalize_quat(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_inv(quat: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = _normalize_quat(quat)
    return (w, -x, -y, -z)


def _quat_from_euler_xyz_deg(euler_deg: list[float] | tuple[float, float, float]) -> tuple[float, float, float, float]:
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


def _euler_xyz_deg_from_quat(quat: list[float] | tuple[float, float, float, float]) -> list[float]:
    w, x, y, z = _normalize_quat(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _hand_world_pose(hand_spec: HandSpec, defaults: Defaults, state: ConfigState) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position = _vec3_add(hand_spec.start_pos, state.hand_pose_offset)
    start_rot = _quat_from_euler_xyz_deg(defaults.hand_start_rot_xyz_deg)
    rotation = _quat_mul(_quat_from_euler_xyz_deg(state.hand_rot_euler_deg), start_rot)
    return position, rotation


def _replace_urdf_package_meshes(urdf_path: Path, package_mesh_dirs: dict[str, Path], fallback_mesh_dir: Path | None = None) -> Path:
    output_dir = Path(tempfile.gettempdir()) / "gex_learning_viser_urdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{urdf_path.stem}_viser.urdf"
    text = urdf_path.read_text(encoding="utf-8")
    for package_name, mesh_dir in package_mesh_dirs.items():
        text = text.replace(f"package://{package_name}/meshes/", f"{mesh_dir.resolve().as_posix()}/")
    if fallback_mesh_dir is not None:
        fallback = fallback_mesh_dir.resolve().as_posix()
        text = re.sub(r"package://[^/]+/meshes/", f"{fallback}/", text)
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != text:
        output_path.write_text(text, encoding="utf-8")
    return output_path


def prepare_hand_urdf() -> Path:
    return _replace_urdf_package_meshes(
        GX4M_URDF,
        {
            "gx4m": HAND_MESH_DIR,
            "gx4m_01": HAND_MESH_DIR,
            "gx4m_03": HAND_MESH_DIR,
        },
    )


def prepare_object_urdf(spec: ObjectPreviewSpec) -> Path:
    mesh_dir = spec.urdf_path.parent.parent / "meshes"
    return _replace_urdf_package_meshes(spec.urdf_path, {}, fallback_mesh_dir=mesh_dir)


def load_joint_limits(urdf_path: Path, joint_names: list[str] | tuple[str, ...]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    wanted = set(joint_names)
    tree = ET.parse(urdf_path)
    for joint in tree.getroot().iter("joint"):
        name = joint.attrib.get("name", "")
        if name not in wanted:
            continue
        limit_el = joint.find("limit")
        if limit_el is None:
            out[name] = (-math.pi, math.pi)
            continue
        lower = float(limit_el.attrib.get("lower", str(-math.pi)))
        upper = float(limit_el.attrib.get("upper", str(math.pi)))
        if upper < lower:
            lower, upper = upper, lower
        out[name] = (lower, upper)
    return out


def _round_float(value: float) -> float:
    if abs(float(value)) < 5.0e-12:
        return 0.0
    return float(f"{float(value):.6g}")


def _format_float(value: float) -> str:
    return f"{_round_float(value):.6g}"


def _format_list(values: list[float] | tuple[float, ...]) -> str:
    return "[" + ",".join(_format_float(value) for value in values) + "]"


def _normalized_keypoints(points: list[list[float]] | None, fallback: list[list[float]]) -> list[list[float]]:
    source = fallback if not points else points
    normalized = []
    for point in source[:MAX_KEYPOINT_CONTROLS]:
        values = [float(value) for value in point[:3]]
        if len(values) < 3:
            values.extend([0.0] * (3 - len(values)))
        normalized.append(values)
    return normalized or [[0.0, 0.0, 0.08]]


def _keypoints_to_config_patch(points: list[list[float]]) -> dict[str, list[list[float]]]:
    if not points:
        return {"kp_bais": [], "kp_center": [[0.0, 0.0, 0.0]]}
    center = [
        sum(float(point[axis]) for point in points) / float(len(points))
        for axis in range(3)
    ]
    biases = [
        [_round_float(float(point[axis]) - center[axis]) for axis in range(3)]
        for point in points
    ]
    return {
        "kp_bais": biases,
        "kp_center": [[_round_float(value) for value in center]],
    }


def _object_pose_to_config_patch(state: ConfigState, spec: ObjectPreviewSpec, defaults: Defaults) -> dict[str, float]:
    pose_with_default_offset = [
        float(spec.pose[axis]) + float(state.object_pose_offset[axis]) - float(defaults.object_pose_offset[axis])
        for axis in range(3)
    ]
    return {
        "dx": _round_float(pose_with_default_offset[0]),
        "dy": _round_float(pose_with_default_offset[1] - 0.03),
        "dz": _round_float(pose_with_default_offset[2]),
    }


def _object_config_patch(state: ConfigState, defaults: Defaults) -> dict[str, Any]:
    patch: dict[str, Any] = _keypoints_to_config_patch(state.object_keypoints or [])
    try:
        spec = load_object_preview_spec(state.object_name)
    except Exception:
        return patch
    patch = {**_object_pose_to_config_patch(state, spec, defaults), **patch}
    patch["hand_pose_offset"] = [_round_float(value) for value in state.hand_pose_offset]
    patch["hand_rot_euler_deg"] = [_round_float(value) for value in state.hand_rot_euler_deg]
    patch["action_mean_joint_pos"] = [_round_float(value) for value in state.action_mean_joint_pos]
    if not spec.registered:
        try:
            patch["urdf_path"] = spec.urdf_path.resolve().relative_to(TWIST_ASSET_ROOT.resolve()).as_posix()
        except ValueError:
            patch["urdf_path"] = spec.urdf_path.as_posix()
    return patch


def hydra_overrides(state: ConfigState) -> list[str]:
    return [
        f"env.object_name={state.object_name}",
        f"env.object_pose_offset={_format_list(state.object_pose_offset)}",
        f"env.hand_pose_offset={_format_list(state.hand_pose_offset)}",
        f"env.hand_rot_euler_deg={_format_list(state.hand_rot_euler_deg)}",
        f"env.action_mean_joint_pos={_format_list(state.action_mean_joint_pos)}",
    ]


def apply_object_config_to_twist_json(state: ConfigState, defaults: Defaults) -> tuple[Path, dict[str, Any]]:
    spec = load_object_preview_spec(state.object_name)
    if not spec.registered:
        raise ValueError(
            f"Object '{state.object_name}' is asset-only. Add a full twist_anything JSON entry with points_feature "
            "before applying edits to source object configs."
        )

    config_path = TWIST_CONFIG_DIR / f"{spec.category}.json"
    with config_path.open("r", encoding="utf-8") as f:
        category_cfg = json.load(f)
    if state.object_name not in category_cfg:
        raise KeyError(f"Object '{state.object_name}' not found in {config_path}")

    patch = _object_config_patch(state, defaults)
    raw = category_cfg[state.object_name]
    for key in (
        "dx",
        "dy",
        "dz",
        "kp_bais",
        "kp_center",
        "hand_pose_offset",
        "hand_rot_euler_deg",
        "action_mean_joint_pos",
    ):
        if key in patch:
            raw[key] = patch[key]

    config_path.write_text(json.dumps(category_cfg, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    state.object_pose_offset = list(defaults.object_pose_offset)
    updated_spec = load_object_preview_spec(state.object_name)
    state.hand_pose_offset = list(updated_spec.hand_pose_offset)
    state.hand_rot_euler_deg = list(updated_spec.hand_rot_euler_deg)
    state.action_mean_joint_pos = list(updated_spec.action_mean_joint_pos)
    state.object_keypoints = _normalized_keypoints(None, updated_spec.keypoints)
    state.object_registered = updated_spec.registered
    state.object_urdf_path = str(updated_spec.urdf_path)
    state.last_saved_path = config_path
    return config_path, patch


def _normalize_state_lengths(state: ConfigState, defaults: Defaults, num_actions: int) -> ConfigState:
    state.object_pose_offset = (state.object_pose_offset + defaults.object_pose_offset)[:3]
    state.hand_pose_offset = (state.hand_pose_offset + defaults.hand_pose_offset)[:3]
    state.hand_rot_euler_deg = (state.hand_rot_euler_deg + defaults.hand_rot_euler_deg)[:3]
    mean = list(state.action_mean_joint_pos)
    if len(mean) < num_actions:
        mean.extend(defaults.action_mean_joint_pos[len(mean) : num_actions])
    if len(mean) < num_actions:
        mean.extend([0.0] * (num_actions - len(mean)))
    state.action_mean_joint_pos = mean[:num_actions]
    if state.object_keypoints is not None:
        state.object_keypoints = _normalized_keypoints(state.object_keypoints, [])
    return state


def build_initial_state(args: argparse.Namespace, defaults: Defaults, categories: dict[str, list[str]], hand_spec: HandSpec) -> ConfigState:
    object_names = {name for names in categories.values() for name in names}
    object_name = _normalize_object_name(args.object_name or defaults.object_name)
    if object_name not in object_names:
        raise SystemExit(f"Unknown object '{object_name}'. Run with --list-objects to inspect available names.")
    spec = load_object_preview_spec(object_name)
    state = ConfigState(
        object_name=spec.name,
        object_pose_offset=list(defaults.object_pose_offset),
        hand_pose_offset=list(spec.hand_pose_offset),
        hand_rot_euler_deg=list(spec.hand_rot_euler_deg),
        action_mean_joint_pos=list(spec.action_mean_joint_pos),
        object_keypoints=_normalized_keypoints(None, spec.keypoints),
        object_registered=spec.registered,
        object_urdf_path=str(spec.urdf_path),
    )
    state = _normalize_state_lengths(state, defaults, hand_spec.num_actions)
    if state.object_name not in object_names:
        raise SystemExit(f"Unknown object '{state.object_name}'. Run with --list-objects to inspect available names.")
    return state


def _missing_viser_message(error: ImportError) -> str:
    return (
        f"Missing dependency: {error.name or error}. Install the viser extra first:\n"
        '  python -m pip install -e ".[viser]"\n'
        "or install the runtime packages directly:\n"
        "  python -m pip install viser yourdfpy"
    )


def _markdown_for_state(state: ConfigState, defaults: Defaults) -> str:
    source_path = "asset-only preview"
    if state.object_registered:
        try:
            spec = load_object_preview_spec(state.object_name)
            source_path = f"`{TWIST_CONFIG_DIR / f'{spec.category}.json'}`"
        except Exception:
            source_path = "unknown"
    saved = f"`{state.last_saved_path}`" if state.last_saved_path is not None else "not saved yet"
    overrides = " ".join(f'"{override}"' for override in hydra_overrides(state))
    registration_note = (
        "Registered object: yes"
        if state.object_registered
        else "Registered object: no - asset-only preview. Add this object to twist_anything JSON and generate points_feature before using the Hydra object override for training."
    )
    patch = json.dumps(_object_config_patch(state, defaults), indent=2, ensure_ascii=False)
    return (
        f"**Object:** `{state.object_name}`  \n"
        f"**{registration_note}**  \n"
        f"**Source config:** {source_path}  \n"
        f"**Last saved:** {saved}\n\n"
        "Object JSON patch that Save will apply:\n"
        f"```json\n{patch}\n```\n\n"
        "Hydra overrides:\n"
        f"```powershell\n{overrides}\n```"
    )


def _print_config_result(path: Path, state: ConfigState) -> None:
    print(f"[configure_dexanytwist_viser] source object config: {path}")
    print("[configure_dexanytwist_viser] hydra overrides:")
    for override in hydra_overrides(state):
        print(f"  {override}")


def run_viser(args: argparse.Namespace, defaults: Defaults, categories: dict[str, list[str]], hand_spec: HandSpec, state: ConfigState) -> None:
    try:
        import numpy as np
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise SystemExit(_missing_viser_message(exc)) from exc

    server = viser.ViserServer(host=args.host, port=args.port, label="DexAnyTwist Configurator")
    server.scene.add_grid(
        "/ground",
        width=1.6,
        height=1.6,
        plane="xy",
        cell_size=0.05,
        section_size=0.25,
        position=(0.0, 0.0, 0.0),
    )
    server.scene.add_frame("/world", axes_length=0.12, axes_radius=0.004)

    hand_urdf_path = prepare_hand_urdf()
    hand_joint_limits = load_joint_limits(GX4M_URDF, hand_spec.actuated_joint_names)
    hand_position, hand_rotation = _hand_world_pose(hand_spec, defaults, state)
    hand_frame = server.scene.add_frame("/hand", show_axes=False, position=hand_position, wxyz=hand_rotation)
    hand_urdf = ViserUrdf(server, hand_urdf_path, root_node_name="/hand")
    hand_urdf_names = hand_urdf.get_actuated_joint_names()

    object_to_category = {name: category for category, names in categories.items() for name in names}
    current_object_frame: Any = None
    current_object_urdf: Any = None
    current_keypoint_cloud: Any = None
    current_keypoint_controls: list[Any] = []
    current_object_spec: ObjectPreviewSpec | None = None
    suppress_callbacks = False
    initial_object_spec = load_object_preview_spec(state.object_name)
    initial_object_position = _object_start_pos(initial_object_spec, defaults, state)

    def hand_cfg_vector() -> Any:
        by_name = dict(zip(hand_spec.actuated_joint_names, state.action_mean_joint_pos))
        return np.array([by_name.get(name, 0.0) for name in hand_urdf_names], dtype=np.float64)

    def update_hand_urdf() -> None:
        hand_urdf.update_cfg(hand_cfg_vector())

    update_hand_urdf()

    def refresh_keypoint_cloud(show: bool) -> None:
        nonlocal current_keypoint_cloud
        if current_keypoint_cloud is not None:
            current_keypoint_cloud.remove()
            current_keypoint_cloud = None
        if not show or current_object_spec is None or not state.object_keypoints:
            return
        points = np.asarray(state.object_keypoints, dtype=np.float32)
        current_keypoint_cloud = server.scene.add_point_cloud(
            f"{OBJECT_KEYPOINT_ANCHOR_PATH}/keypoints",
            points=points,
            colors=(20, 240, 70),
            point_size=0.018,
            point_shape="circle",
        )

    def remove_keypoint_controls() -> None:
        nonlocal current_keypoint_controls
        for control in current_keypoint_controls:
            control.remove()
        current_keypoint_controls = []

    def load_object_scene(object_name: str, reset_keypoints: bool) -> None:
        nonlocal current_keypoint_cloud, current_object_frame, current_object_urdf, current_object_spec
        if current_object_frame is not None:
            if current_keypoint_cloud is not None:
                current_keypoint_cloud.remove()
            current_object_frame.remove()
            current_keypoint_cloud = None
            remove_keypoint_controls()
            current_object_frame = None
            current_object_urdf = None
        current_object_spec = load_object_preview_spec(object_name)
        state.object_name = current_object_spec.name
        state.object_registered = current_object_spec.registered
        state.object_urdf_path = str(current_object_spec.urdf_path)
        if reset_keypoints or state.object_keypoints is None:
            state.object_pose_offset = list(defaults.object_pose_offset)
            state.hand_pose_offset = list(current_object_spec.hand_pose_offset)
            state.hand_rot_euler_deg = list(current_object_spec.hand_rot_euler_deg)
            state.action_mean_joint_pos = list(current_object_spec.action_mean_joint_pos)
            _normalize_state_lengths(state, defaults, hand_spec.num_actions)
            state.object_keypoints = _normalized_keypoints(None, current_object_spec.keypoints)
        object_position = _object_start_pos(current_object_spec, defaults, state)
        object_urdf_path = prepare_object_urdf(current_object_spec)
        current_object_frame = server.scene.add_frame("/object", show_axes=False, position=object_position)
        current_object_urdf = ViserUrdf(server, object_urdf_path, root_node_name="/object")
        server.scene.add_frame(
            OBJECT_KEYPOINT_ANCHOR_PATH,
            position=current_object_spec.link1_origin,
            axes_length=0.045,
            axes_radius=0.002,
        )
        object_joint_names = current_object_urdf.get_actuated_joint_names()
        if object_joint_names:
            current_object_urdf.update_cfg(np.zeros((len(object_joint_names),), dtype=np.float64))

    with server.gui.add_folder("Object", order=0.0):
        initial_category = object_to_category.get(state.object_name, sorted(categories)[0])
        category_dropdown = server.gui.add_dropdown(
            "Category",
            sorted(categories),
            initial_value=initial_category,
            order=0.0,
        )
        object_dropdown = server.gui.add_dropdown(
            "Object",
            categories[initial_category],
            initial_value=state.object_name,
            order=1.0,
        )
        show_keypoints = server.gui.add_checkbox("Show object keypoints", initial_value=True, order=2.0)
        object_transform = server.scene.add_transform_controls(
            "/object_transform",
            scale=0.12,
            position=initial_object_position,
            disable_rotations=True,
            translation_limits=((-0.5, 0.5), (-0.5, 0.5), (0.2, 1.2)),
            depth_test=False,
        )
        object_offset_handles = [
            server.gui.add_number(
                f"object offset {axis}",
                state.object_pose_offset[index],
                min=-0.5,
                max=0.5,
                step=0.001,
                order=3.0 + float(index),
            )
            for index, axis in enumerate(("x", "y", "z"))
        ]

    with server.gui.add_folder("Object Keypoints", order=0.5):
        show_keypoint_controls = server.gui.add_checkbox("Show keypoint controls", initial_value=True, order=0.0)
        add_keypoint_button = server.gui.add_button("Add keypoint", order=1.0)
        remove_keypoint_button = server.gui.add_button("Remove last keypoint", order=2.0)
        reset_keypoints_button = server.gui.add_button("Reset from object", order=3.0)
        keypoint_number_handles = []
        for index in range(MAX_KEYPOINT_CONTROLS):
            row_handles = []
            for axis_index, axis in enumerate(("x", "y", "z")):
                row_handles.append(
                    server.gui.add_number(
                        f"kp{index} {axis}",
                        0.0,
                        min=-1.0,
                        max=1.0,
                        step=0.001,
                        visible=False,
                        order=10.0 + index * 3.0 + axis_index,
                    )
                )
            keypoint_number_handles.append(row_handles)

    with server.gui.add_folder("Hand Pose", order=1.0):
        hand_transform = server.scene.add_transform_controls(
            "/hand_transform",
            scale=0.12,
            position=hand_position,
            wxyz=hand_rotation,
            translation_limits=((-0.35, 0.35), (-0.35, 0.35), (0.25, 1.1)),
            rotation_limits=((-math.pi, math.pi), (-math.pi, math.pi), (-math.pi, math.pi)),
        )
        offset_handles = [
            server.gui.add_number(
                f"offset {axis}",
                state.hand_pose_offset[index],
                min=-0.35,
                max=0.35,
                step=0.001,
                order=float(index),
            )
            for index, axis in enumerate(("x", "y", "z"))
        ]
        rot_handles = [
            server.gui.add_number(
                f"rot {axis} deg",
                state.hand_rot_euler_deg[index],
                min=-180.0,
                max=180.0,
                step=0.5,
                order=10.0 + float(index),
            )
            for index, axis in enumerate(("x", "y", "z"))
        ]

    with server.gui.add_folder("Action Mean Joint Position", order=2.0):
        joint_handles = []
        for index, joint_name in enumerate(hand_spec.actuated_joint_names):
            lower, upper = hand_joint_limits.get(joint_name, (-math.pi, math.pi))
            initial = min(max(state.action_mean_joint_pos[index], lower), upper)
            state.action_mean_joint_pos[index] = initial
            joint_handles.append(
                server.gui.add_slider(
                    joint_name,
                    min=lower,
                    max=upper,
                    step=0.001,
                    initial_value=initial,
                    order=float(index),
                )
            )

    with server.gui.add_folder("Export", order=3.0):
        save_button = server.gui.add_button("Save object config", order=0.0)
        output_markdown = server.gui.add_markdown(_markdown_for_state(state, defaults), order=1.0)

    def refresh_output() -> None:
        output_markdown.content = _markdown_for_state(state, defaults)

    def sync_object_controls_from_state(sync_transform: bool = True) -> None:
        nonlocal suppress_callbacks
        if current_object_spec is None:
            return
        object_pos = _object_start_pos(current_object_spec, defaults, state)
        if current_object_frame is not None:
            current_object_frame.position = object_pos
        if sync_transform:
            object_transform.position = object_pos
        suppress_callbacks = True
        try:
            for handle, value in zip(object_offset_handles, state.object_pose_offset):
                handle.value = float(value)
        finally:
            suppress_callbacks = False

    def sync_keypoint_number_controls() -> None:
        nonlocal suppress_callbacks
        points = state.object_keypoints or []
        suppress_callbacks = True
        try:
            for point_index, row_handles in enumerate(keypoint_number_handles):
                visible = point_index < len(points)
                for axis_index, handle in enumerate(row_handles):
                    handle.visible = visible
                    if visible:
                        handle.value = float(points[point_index][axis_index])
        finally:
            suppress_callbacks = False

    def rebuild_keypoint_controls(show: bool) -> None:
        nonlocal current_keypoint_controls
        remove_keypoint_controls()
        if not show or current_object_spec is None:
            return
        points = state.object_keypoints or []
        for point_index, point in enumerate(points):
            control = server.scene.add_transform_controls(
                f"{OBJECT_KEYPOINT_ANCHOR_PATH}/keypoint_{point_index}",
                scale=0.035,
                position=tuple(point),
                disable_rotations=True,
                translation_limits=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
                depth_test=False,
                opacity=0.85,
            )

            @control.on_update
            def _(_: Any, index: int = point_index, handle: Any = control) -> None:
                if suppress_callbacks or state.object_keypoints is None or index >= len(state.object_keypoints):
                    return
                state.object_keypoints[index] = [float(value) for value in handle.position]
                sync_keypoint_number_controls()
                refresh_keypoint_cloud(bool(show_keypoints.value))
                refresh_output()

            current_keypoint_controls.append(control)

    def refresh_keypoint_scene(rebuild_controls: bool = True) -> None:
        refresh_keypoint_cloud(bool(show_keypoints.value))
        if rebuild_controls:
            rebuild_keypoint_controls(bool(show_keypoint_controls.value))
        sync_keypoint_number_controls()

    def sync_hand_controls_from_state(sync_transform: bool = True) -> None:
        nonlocal suppress_callbacks
        hand_pos, hand_rot = _hand_world_pose(hand_spec, defaults, state)
        hand_frame.position = hand_pos
        hand_frame.wxyz = hand_rot
        if sync_transform:
            hand_transform.position = hand_pos
            hand_transform.wxyz = hand_rot
        suppress_callbacks = True
        try:
            for handle, value in zip(offset_handles, state.hand_pose_offset):
                handle.value = float(value)
            for handle, value in zip(rot_handles, state.hand_rot_euler_deg):
                handle.value = float(value)
        finally:
            suppress_callbacks = False

    def sync_joint_controls_from_state() -> None:
        nonlocal suppress_callbacks
        suppress_callbacks = True
        try:
            for handle, value in zip(joint_handles, state.action_mean_joint_pos):
                handle.value = float(value)
        finally:
            suppress_callbacks = False
        update_hand_urdf()

    load_object_scene(state.object_name, reset_keypoints=False)
    sync_object_controls_from_state(sync_transform=True)
    refresh_keypoint_scene(rebuild_controls=True)
    sync_hand_controls_from_state()
    sync_joint_controls_from_state()
    refresh_output()

    @category_dropdown.on_update
    def _(_: Any) -> None:
        nonlocal suppress_callbacks
        if suppress_callbacks:
            return
        category = str(category_dropdown.value)
        names = categories[category]
        suppress_callbacks = True
        try:
            object_dropdown.options = names
            object_dropdown.value = names[0]
        finally:
            suppress_callbacks = False
        state.object_name = names[0]
        load_object_scene(state.object_name, reset_keypoints=True)
        sync_object_controls_from_state(sync_transform=True)
        sync_hand_controls_from_state(sync_transform=True)
        sync_joint_controls_from_state()
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    @object_dropdown.on_update
    def _(_: Any) -> None:
        if suppress_callbacks:
            return
        state.object_name = _normalize_object_name(str(object_dropdown.value))
        load_object_scene(state.object_name, reset_keypoints=True)
        sync_object_controls_from_state(sync_transform=True)
        sync_hand_controls_from_state(sync_transform=True)
        sync_joint_controls_from_state()
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    @object_transform.on_update
    def _(_: Any) -> None:
        if suppress_callbacks or current_object_spec is None:
            return
        final_position = [float(value) for value in object_transform.position]
        state.object_pose_offset = [
            final_position[0] - defaults.hand_start_pos[0] - float(current_object_spec.pose[0]),
            final_position[1] - defaults.hand_start_pos[1] - float(current_object_spec.pose[1]),
            final_position[2] - defaults.hand_start_pos[2] - float(current_object_spec.pose[2]),
        ]
        sync_object_controls_from_state(sync_transform=False)
        refresh_output()

    def _object_offset_update(_: Any) -> None:
        if suppress_callbacks:
            return
        state.object_pose_offset = [float(handle.value) for handle in object_offset_handles]
        sync_object_controls_from_state(sync_transform=True)
        refresh_output()

    for handle in object_offset_handles:
        handle.on_update(_object_offset_update)

    @show_keypoints.on_update
    def _(_: Any) -> None:
        refresh_keypoint_cloud(bool(show_keypoints.value))

    @show_keypoint_controls.on_update
    def _(_: Any) -> None:
        rebuild_keypoint_controls(bool(show_keypoint_controls.value))

    def _keypoint_number_update(_: Any) -> None:
        if suppress_callbacks or state.object_keypoints is None:
            return
        points = []
        for point_index, row_handles in enumerate(keypoint_number_handles):
            if point_index >= len(state.object_keypoints):
                break
            points.append([float(handle.value) for handle in row_handles])
        state.object_keypoints = points
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    for row in keypoint_number_handles:
        for handle in row:
            handle.on_update(_keypoint_number_update)

    @add_keypoint_button.on_click
    def _(_: Any) -> None:
        points = list(state.object_keypoints or [])
        if len(points) >= MAX_KEYPOINT_CONTROLS:
            return
        points.append(list(points[-1]) if points else [0.0, 0.0, 0.08])
        state.object_keypoints = points
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    @remove_keypoint_button.on_click
    def _(_: Any) -> None:
        points = list(state.object_keypoints or [])
        if len(points) <= 1:
            return
        points.pop()
        state.object_keypoints = points
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    @reset_keypoints_button.on_click
    def _(_: Any) -> None:
        if current_object_spec is None:
            return
        state.object_keypoints = _normalized_keypoints(None, current_object_spec.keypoints)
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()

    @hand_transform.on_update
    def _(_: Any) -> None:
        nonlocal suppress_callbacks
        if suppress_callbacks:
            return
        final_position = [float(value) for value in hand_transform.position]
        final_rotation = _normalize_quat([float(value) for value in hand_transform.wxyz])
        state.hand_pose_offset = [
            final_position[0] - float(hand_spec.start_pos[0]),
            final_position[1] - float(hand_spec.start_pos[1]),
            final_position[2] - float(hand_spec.start_pos[2]),
        ]
        start_rot = _quat_from_euler_xyz_deg(defaults.hand_start_rot_xyz_deg)
        state.hand_rot_euler_deg = _euler_xyz_deg_from_quat(_quat_mul(final_rotation, _quat_inv(start_rot)))
        sync_hand_controls_from_state(sync_transform=False)
        refresh_output()

    def _pose_number_update(_: Any) -> None:
        if suppress_callbacks:
            return
        state.hand_pose_offset = [float(handle.value) for handle in offset_handles]
        state.hand_rot_euler_deg = [float(handle.value) for handle in rot_handles]
        sync_hand_controls_from_state(sync_transform=True)
        refresh_output()

    for handle in [*offset_handles, *rot_handles]:
        handle.on_update(_pose_number_update)

    def _joint_update(_: Any) -> None:
        if suppress_callbacks:
            return
        state.action_mean_joint_pos = [float(handle.value) for handle in joint_handles]
        update_hand_urdf()
        refresh_output()

    for handle in joint_handles:
        handle.on_update(_joint_update)

    @save_button.on_click
    def _(_: Any) -> None:
        path, _patch = apply_object_config_to_twist_json(state, defaults)
        load_object_scene(state.object_name, reset_keypoints=False)
        sync_object_controls_from_state(sync_transform=True)
        sync_hand_controls_from_state(sync_transform=True)
        sync_joint_controls_from_state()
        refresh_keypoint_scene(rebuild_controls=True)
        refresh_output()
        _print_config_result(path, state)

    print(f"[configure_dexanytwist_viser] open http://{args.host}:{args.port}")
    print("[configure_dexanytwist_viser] press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[configure_dexanytwist_viser] stopping.")
    finally:
        server.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", "--object_name", dest="object_name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Apply the selected object's current JSON values back to its source config and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    categories = discover_twist_objects()
    defaults = load_env_defaults()
    hand_spec = load_hand_spec("gx4m")

    if args.list_objects:
        for category, names in categories.items():
            print(f"{category}:")
            for name in names:
                print(f"  {name}")
        return

    state = build_initial_state(args, defaults, categories, hand_spec)
    if args.export_only:
        path, _patch = apply_object_config_to_twist_json(state, defaults)
        _print_config_result(path, state)
        return

    run_viser(args, defaults, categories, hand_spec, state)


if __name__ == "__main__":
    main()
