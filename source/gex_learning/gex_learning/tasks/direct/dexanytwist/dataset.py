from __future__ import annotations

import json
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[6]
ASSET_DIR = REPO_ROOT / "asset"
GX4M_URDF = ASSET_DIR / "gx4m" / "urdf" / "gx4m.urdf"
TWIST_ASSET_ROOT = ASSET_DIR / "twist_dataset"
TWIST_CONFIG_DIR = Path(__file__).resolve().parent / "config" / "twist_anything"
TWIST_SPLIT_DIR = TWIST_CONFIG_DIR / "split"


@dataclass(frozen=True)
class TwistObjectSpec:
    name: str
    urdf_path: Path
    pose: tuple[float, float, float]
    keypoint_bias: np.ndarray
    hand_pose_offset: tuple[float, float, float]
    hand_rot_euler_deg: tuple[float, float, float]
    action_mean_joint_pos: tuple[float, ...]
    points_feature: np.ndarray
    rotation_axis_sign: float
    xy_axis_limit: float
    class_index: int


@dataclass(frozen=True)
class HandSpec:
    name: str
    urdf_path: Path
    actuated_joint_names: tuple[str, ...]
    fingertip_body_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    fingertip_keypoint_offset: tuple[tuple[float, float, float], ...]
    effort_limit: float
    velocity_limit: float
    stiffness: float
    damping: float
    start_pos: tuple[float, float, float] = (0.0, 0.0, 0.3)
    start_rot: tuple[float, float, float, float] = (1.0, 0, 0.0, 0.0)

    @property
    def num_actions(self) -> int:
        return len(self.actuated_joint_names)


def _prepared_gx4m_urdf() -> Path:
    """Return a local URDF whose package:// mesh URIs resolve without ROS package lookup."""
    checkout_id = hashlib.sha256(str(REPO_ROOT).encode()).hexdigest()[:12]
    output_dir = Path(gettempdir()) / "gex_learning_assets" / checkout_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gx4m_isaac.urdf"
    if not GX4M_URDF.is_file():
        raise FileNotFoundError(f"Missing hand asset: {GX4M_URDF}. Run scripts/prepare_assets.py; see asset/README.md.")
    text = GX4M_URDF.read_text(encoding="utf-8")
    mesh_dir = (ASSET_DIR / "gx4m" / "meshes").resolve().as_posix()
    text = text.replace("package://gx4m/meshes/", f"{mesh_dir}/")
    text = text.replace("package://gx4m_01/meshes/", f"{mesh_dir}/")
    text = text.replace("package://gx4m_03/meshes/", f"{mesh_dir}/")
    # The converted URDF lives in a temporary directory, so its mesh references
    # must be resolved before moving it away from the source asset directory.
    text = text.replace("../meshes/", f"{mesh_dir}/")
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != text:
        output_path.write_text(text, encoding="utf-8")
    return output_path


def load_hand_spec(name: str = "gx4m") -> HandSpec:
    """Load the local dexterous hand spec used by direct tasks."""
    normalized = name.lower()
    if normalized in {"gx4m", "gx4max"}:
        return HandSpec(
            name="gx4m",
            urdf_path=_prepared_gx4m_urdf(),
            actuated_joint_names=tuple(f"joint{i}" for i in range(1, 17)),
            fingertip_body_names=("link4", "link8", "link12", "link16"),
            default_joint_pos=(
                0.0, 0, 0.0, 0.0,
                0.0, 0.7, 0.75, 0.45,
                0.0, 0.7, 0.75, 0.45,
                0.25, 0.0, 0.7, 0.45,
            ),
            fingertip_keypoint_offset=(
                (0.05, 0.0, -0.02),
                (0.05, 0.0, -0.02),
                (0.05, 0.0, -0.02),
                (0.05, 0.0, -0.02),
            ),
            effort_limit=1.0,
            velocity_limit=3.0,
            stiffness=2.0,
            damping=0.1,
        )
    raise ValueError(f"Unknown hand spec: {name}")


def _parse_joint_axis_and_limit(urdf_path: Path, joint_name: str) -> tuple[tuple[float, float, float], float]:
    tree = ET.parse(urdf_path)
    for joint in tree.getroot().iter("joint"):
        if joint.attrib.get("name") != joint_name:
            continue
        axis_el = joint.find("axis")
        limit_el = joint.find("limit")
        axis = (0.0, 0.0, 1.0)
        if axis_el is not None:
            axis = tuple(float(v) for v in axis_el.attrib.get("xyz", "0 0 1").split())
        width = 0.0
        if limit_el is not None:
            lower = float(limit_el.attrib.get("lower", "0"))
            upper = float(limit_el.attrib.get("upper", "0"))
            width = abs(upper - lower)
        return axis, width
    return (0.0, 0.0, 1.0), 0.0


def _load_class_order() -> list[str]:
    names: list[str] = []
    for json_path in sorted(TWIST_CONFIG_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            names.extend(json.load(f).keys())
    return names


def _load_object_category_config(name: str) -> tuple[Path, dict]:
    inferred_category = name.rsplit("_", 1)[0]
    inferred_path = TWIST_CONFIG_DIR / f"{inferred_category}.json"
    if inferred_path.exists():
        with inferred_path.open("r", encoding="utf-8") as f:
            category_cfg = json.load(f)
        if name in category_cfg:
            return inferred_path, category_cfg

    for config_path in sorted(TWIST_CONFIG_DIR.glob("*.json")):
        if config_path == inferred_path:
            continue
        with config_path.open("r", encoding="utf-8") as f:
            category_cfg = json.load(f)
        if name in category_cfg:
            return config_path, category_cfg

    raise KeyError(f"Object '{name}' not found in {TWIST_CONFIG_DIR}")


def _float_tuple(raw: object, default: tuple[float, ...], length: int) -> tuple[float, ...]:
    if raw is None:
        values = list(default)
    else:
        values = [float(value) for value in raw]  # type: ignore[union-attr]
    if len(values) < length:
        values.extend(default[len(values) : length])
    return tuple(float(value) for value in values[:length])


def load_twist_object_names(split: str = "train_all") -> list[str]:
    """Load object names from the DexAnyTwist split files or all JSON configs."""
    if split in {"train", "train_all"}:
        split_path = TWIST_SPLIT_DIR / "twist_train_list.txt"
    elif split in {"all", "dataset_all", "cls"}:
        split_path = TWIST_SPLIT_DIR / "twist_cls_list.txt"
    elif split == "configs":
        return _load_class_order()
    else:
        raise ValueError(f"Unknown DexAnyTwist split: {split}")

    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_twist_object(name: str = "bulb_custom_0017") -> TwistObjectSpec:
    config_path, category_cfg = _load_object_category_config(name)
    raw = category_cfg[name]
    urdf_path = TWIST_ASSET_ROOT / raw["urdf_path"]
    points_path = TWIST_ASSET_ROOT / raw["points_feature"]
    for path in (urdf_path, points_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing task asset: {path}. Run scripts/prepare_assets.py; see asset/README.md.")
    points_feature = np.load(points_path)[0].astype(np.float32)

    keypoint_bias = np.asarray(raw["kp_bais"], dtype=np.float32) + np.asarray(raw["kp_center"], dtype=np.float32)
    axis, _ = _parse_joint_axis_and_limit(urdf_path, "joint1")
    _, xy_axis_limit = _parse_joint_axis_and_limit(urdf_path, "joint1_rx")
    class_order = _load_class_order()

    return TwistObjectSpec(
        name=name,
        urdf_path=urdf_path,
        pose=(float(raw["dx"]), float(raw["dy"]) + 0.03, float(raw["dz"])),
        keypoint_bias=keypoint_bias,
        hand_pose_offset=_float_tuple(raw.get("hand_pose_offset"), (0.0, 0.0, 0.0), 3),  # type: ignore[arg-type]
        hand_rot_euler_deg=_float_tuple(raw.get("hand_rot_euler_deg"), (0.0, 0.0, 0.0), 3),  # type: ignore[arg-type]
        action_mean_joint_pos=_float_tuple(raw.get("action_mean_joint_pos"), (0.0,) * 16, 16),  # type: ignore[arg-type]
        points_feature=points_feature,
        rotation_axis_sign=1.0 if axis[2] >= 0.0 else -1.0,
        xy_axis_limit=float(xy_axis_limit),
        class_index=int(raw["class_index"]) if "class_index" in raw else (class_order.index(name) if name in class_order else 0),
    )
