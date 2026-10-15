"""Simulation-independent seed, screening and cache contracts. SI units, wxyz quaternions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CONFIG = ROOT / "experiment/super_inair/objects/tennis_ball_60.json"
JOINT_NAMES = [f"joint{i}" for i in range(1, 17)]
STATE_FIELDS = {"joint_pos": 16, "joint_targets": 16, "object_pose": 7}


def joint_order(source_names, target_names):
    if len(set(source_names)) != len(source_names) or set(source_names) != set(target_names):
        raise ValueError("Joint names must be unique and match the GX4M actuated joints")
    return [source_names.index(name) for name in target_names]


def vector(value, size, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return out


def pose(value, name):
    out = vector(value, 7, name).copy()
    norm = np.linalg.norm(out[3:])
    if norm < 1e-8:
        raise ValueError(f"{name} has a zero quaternion")
    out[3:] /= norm
    return out.tolist()


def validate_seed(raw, require_saved=True):
    if raw.get("version") not in (1, 2) or (require_saved and raw.get("saved_by_user") is not True):
        raise ValueError("Save a seed pose in the super_inair viser editor before collecting")
    names = list(raw["joint_names"])
    order = joint_order(names, JOINT_NAMES)
    q = vector(raw["joint_pos"], 16, "joint_pos")[order]
    common = dict(version=raw['version'], saved_by_user=bool(raw.get('saved_by_user')),
                  joint_names=JOINT_NAMES.copy(), joint_pos=q.tolist(),
                  hand_pose=pose(raw['hand_pose'], 'hand_pose'), object_pose=pose(raw['object_pose'], 'object_pose'))
    if raw['version'] == 2:
        from .object_assets import validate_object
        return dict(common, object=validate_object(raw['object']))
    diameter, mass = float(raw["ball_diameter_m"]), float(raw["ball_mass_kg"])
    if not np.isfinite([diameter, mass]).all() or diameter <= 0 or mass <= 0:
        raise ValueError("Ball diameter and mass must be positive and finite")
    return dict(common, ball_diameter_m=diameter, ball_mass_kg=mass)


def load_seed(path=DEFAULT_CONFIG):
    return validate_seed(json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def local_pose(world_pose, origins):
    result = np.array(world_pose, copy=True)
    result[..., :3] -= origins
    return result


def world_pose(env_pose, origins):
    result = np.array(env_pose, copy=True)
    result[..., :3] += origins
    return result


@dataclass
class Screening:
    settle_s: float = 0.5
    hold_s: float = 1.5
    replay_s: float = 1.0
    contact_force_n: float = 0.1
    min_contacts: int = 2
    linear_speed_m_s: float = 0.02
    angular_speed_rad_s: float = 0.5
    drift_m: float = 0.01
    displacement_m: float = 0.05
    drop_m: float = 0.05
    joint_noise_rad: float = 0.1
    position_noise_m: float = 0.002

    def validate(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid screening parameter: {key}")
        if not 1 <= self.min_contacts <= 4 or int(self.min_contacts) != self.min_contacts:
            raise ValueError("min_contacts must be an integer from 1 to 4")
        if min(self.hold_s, self.replay_s) <= 0:
            raise ValueError("Validation durations must be positive")
        return self


def failure_masks(state, initial_position, hold_position, limits, stable):
    """Independent reasons; caller latches the first failing step for each candidate."""
    pos = state["object_pose"][:, :3]
    forces = np.linalg.norm(state["contact_forces"], axis=-1)
    finite = np.ones(len(pos), dtype=bool)
    for key in ("joint_pos", "joint_vel", "joint_targets", "object_pose", "object_velocity", "contact_forces"):
        finite &= np.isfinite(state[key]).reshape(len(pos), -1).all(axis=1)
    masks = {
        "nonfinite": ~finite,
        "displacement": np.linalg.norm(pos - initial_position, axis=1) > limits.displacement_m,
        "drop": initial_position[:, 2] - pos[:, 2] > limits.drop_m,
    }
    if stable:
        masks.update(
            contacts=(forces > limits.contact_force_n).sum(axis=1) < limits.min_contacts,
            linear_speed=np.linalg.norm(state["object_velocity"][:, :3], axis=1) >= limits.linear_speed_m_s,
            angular_speed=np.linalg.norm(state["object_velocity"][:, 3:], axis=1) >= limits.angular_speed_rad_s,
            drift=np.linalg.norm(pos - hold_position, axis=1) > limits.drift_m,
        )
    return masks


def empty_states():
    return {key: np.empty((0, width), dtype=np.float32) for key, width in STATE_FIELDS.items()}


def save_cache(directory, states, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(states[key], dtype=np.float32) for key in STATE_FIELDS}
    count = len(arrays["joint_pos"])
    for key, width in STATE_FIELDS.items():
        if arrays[key].shape != (count, width) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid cache array: {key}")
    seed = validate_seed(metadata["seed_pose"])
    norms = np.linalg.norm(arrays["object_pose"][:, 3:], axis=1)
    if not np.allclose(norms, 1, atol=1e-4):
        raise ValueError("Cache quaternions must be unit wxyz")
    with (directory / "grasps.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays, joint_names=np.array(JOINT_NAMES),
                            hand_pose=np.asarray(seed["hand_pose"], dtype=np.float64))
    (directory / "grasps.npz.tmp").replace(directory / "grasps.npz")
    write_json(directory / "metadata.json", dict(metadata, version=1, count=count,
               coordinate_frame="environment", quaternion_order="wxyz", joint_units="rad"))


def load_cache(path):
    path = Path(path)
    if path.is_dir():
        path = path / "grasps.npz"
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("version") != 1 or metadata.get("coordinate_frame") != "environment"
            or metadata.get("quaternion_order") != "wxyz" or metadata.get("joint_units") != "rad"):
        raise ValueError("Unsupported grasp cache format")
    seed = validate_seed(metadata["seed_pose"])
    with np.load(path, allow_pickle=False) as archive:
        order = joint_order(archive["joint_names"].tolist(), JOINT_NAMES)
        states = {key: archive[key].copy() for key in STATE_FIELDS}
        if not np.allclose(archive["hand_pose"], seed["hand_pose"]):
            raise ValueError("Cache and metadata hand poses differ")
    for key, width in STATE_FIELDS.items():
        if states[key].shape != (metadata["count"], width) or not np.isfinite(states[key]).all():
            raise ValueError(f"Corrupt cache: {key}")
    for key in ("joint_pos", "joint_targets"):
        states[key] = states[key][:, order]
    if not np.allclose(np.linalg.norm(states["object_pose"][:, 3:], axis=1), 1, atol=1e-4):
        raise ValueError("Corrupt cache quaternion")
    return states, metadata
