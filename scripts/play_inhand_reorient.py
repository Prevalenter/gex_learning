"""Play a GX4M in-hand reorientation checkpoint with Isaac Lab's RSL-RL play.py."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
import json

import yaml

import h5py  # noqa: F401  # Preload HDF5 DLLs before Kit changes DLL search paths on Windows.
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import isaaclab_root as resolve_isaaclab_root
DEFAULT_ISAACLAB_ROOT = resolve_isaaclab_root()
TASK = "Gx4m-InHandReorient-Direct-v0"
EXPERIMENT_NAME = "gx4m_inhand_reorient"


def _has_option(argv: list[str], names: tuple[str, ...]) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in argv)


def _has_hydra_override(argv: list[str], key: str) -> bool:
    return any(arg.lstrip("+").startswith(f"{key}=") for arg in argv)


def _with_default_option(argv: list[str], names: tuple[str, ...], value: str) -> list[str]:
    if _has_option(argv, names):
        return argv
    return [*argv, names[0], value]


def _require_checkpoint(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--checkpoint":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit("--checkpoint requires a model path")
            return
        if arg.startswith("--checkpoint="):
            if arg.split("=", 1)[1]:
                return
            raise SystemExit("--checkpoint requires a model path")
    raise SystemExit(
        "A checkpoint path is required.\n"
        "Example:\n"
        "  python scripts/play_inhand_reorient.py --checkpoint logs/rsl_rl/gx4m_inhand_reorient/<run>/model_1000.pt"
    )


def _with_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    _require_checkpoint(result)
    result = _with_default_option(result, ("--task",), TASK)
    result = _with_default_option(result, ("--experiment_name",), EXPERIMENT_NAME)
    result = _with_default_option(result, ("--num_envs",), "1")
    if not _has_hydra_override(result, "env.hand_name"):
        result.append("env.hand_name=gx4m")
    if not _has_hydra_override(result, "env.apply_visual_materials"):
        result.append("env.apply_visual_materials=True")
    # Preserve the training articulation instead of changing it for rendering.
    if not _has_hydra_override(result, "env.robot_cfg.spawn.merge_fixed_joints"):
        result.append("env.robot_cfg.spawn.merge_fixed_joints=True")
    return result


def _option_value(argv: list[str], name: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            return argv[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _with_training_configuration(argv: list[str]) -> list[str]:
    """Restore observation and reset settings from this checkpoint's saved run.

    BaseLoader reads Python-tagged Isaac Lab YAML as plain data. Only selected
    scalar/list fields are converted; no Python constructors are executed.
    """
    checkpoint = _option_value(argv, "--checkpoint")
    if not checkpoint:
        return argv
    params = Path(checkpoint).expanduser().resolve().parent / "params/env.yaml"
    if not params.is_file():
        return argv
    saved = yaml.load(params.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    fields = {
        "hand_name": str, "goal_rotation_axes": str, "goal_mode": str,
        "policy_joint_action_only": lambda value: value.lower() == "true",
        "continuous_goal_step": float,
        "hand_start_pos": lambda values: list(map(float, values)),
        "hand_start_rot": lambda values: list(map(float, values)),
        "default_hand_joint_pos": lambda values: list(map(float, values)),
        "object_start_pos": lambda values: list(map(float, values)),
        "object_start_rot": lambda values: list(map(float, values)),
        "reset_position_noise": float, "reset_position_noise_z": float,
        "reset_dof_pos_noise": float, "reset_object_rotation_noise": float,
        "reach_goal_bonus": float, "dist_reward_scale": float,
    }
    result = list(argv)
    custom_pose = (_has_hydra_override(argv, "env.initial_pose_file")
                   or any(arg.lstrip("+").lower() == "env.use_initial_pose_config=true" for arg in argv))
    pose_fields = {"default_hand_joint_pos", "object_start_pos", "object_start_rot",
                   "reset_position_noise", "reset_position_noise_z", "reset_dof_pos_noise",
                   "reset_object_rotation_noise"}
    for key, convert in fields.items():
        if key not in saved or _has_hydra_override(argv, f"env.{key}"):
            continue
        if custom_pose and key in pose_fields:
            continue
        value = convert(saved[key])
        result.append(f"env.{key}={json.dumps(value, separators=(',', ':'))}")
    if not custom_pose and not _has_hydra_override(argv, "env.use_initial_pose_config"):
        # The run YAML contains the resolved pose; it must not be replaced by a
        # newer Viser edit or depend on the original temporary snapshot path.
        result.append("env.use_initial_pose_config=False")
    merge = saved.get("robot_cfg", {}).get("spawn", {}).get("merge_fixed_joints")
    if merge is not None and not _has_hydra_override(argv, "env.robot_cfg.spawn.merge_fixed_joints"):
        result.append(f"env.robot_cfg.spawn.merge_fixed_joints={merge}")
    print(f"[play_inhand_reorient] Restored training settings from {params}", flush=True)
    return result


def _with_local_isaacsim_ext_folders(argv: list[str]) -> list[str]:
    import isaacsim

    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    ext_folders = [
        isaacsim_root / "exts",
        isaacsim_root / "extsDeprecated",
        isaacsim_root / "extscache",
        isaacsim_root / "extsPhysics",
        isaacsim_root / "kit" / "data" / "Kit" / "Isaac-Sim" / "5.1" / "exts" / "3",
    ]
    kit_args = _option_value(argv, "--kit_args")
    if kit_args is None:
        kit_args = " ".join(f'--ext-folder="{path.as_posix()}"' for path in ext_folders if path.exists())
    # Vulkan crashes during viewport startup on this Windows Isaac Sim setup.
    # Respect an explicit graphics backend selection supplied by the user.
    if sys.platform == "win32" and "--/app/vulkan" not in kit_args:
        kit_args += " --/app/vulkan=false"
    result = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--kit_args":
            skip = True
        elif not arg.startswith("--kit_args="):
            result.append(arg)
    return [*result, "--kit_args", kit_args] if kit_args else result


def main() -> None:
    os.chdir(REPO_ROOT)
    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT))
    play_script = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    if not play_script.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab RSL-RL play script at: {play_script}")

    sys.path.insert(0, str(REPO_ROOT / "source" / "gex_learning"))
    import gex_learning.tasks  # noqa: F401

    forwarded_argv = _with_local_isaacsim_ext_folders(
        _with_defaults(_with_training_configuration(sys.argv[1:]))
    )
    sys.path.insert(0, str(play_script.parent))
    sys.argv = [str(play_script), *forwarded_argv]
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
