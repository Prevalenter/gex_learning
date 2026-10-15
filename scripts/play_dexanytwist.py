"""Play a GX4M DexAnyTwist checkpoint with Isaac Lab's RSL-RL play.py."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import h5py  # noqa: F401  # Preload HDF5 DLLs before Kit changes DLL search paths on Windows.
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import isaaclab_root as resolve_isaaclab_root
DEFAULT_ISAACLAB_ROOT = resolve_isaaclab_root()
TASK = "Gx4m-DexAnyTwist-Direct-v0"
EXPERIMENT_NAME = "gx4m_dexanytwist"
OBJECT_NAME = "bulb_custom_0017"
DEBUG_NUM_ENVS = "9"


def _has_option(argv: list[str], names: tuple[str, ...]) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in argv)


def _has_hydra_override(argv: list[str], key: str) -> bool:
    return any(arg.lstrip("+").startswith(f"{key}=") for arg in argv)


def _with_default_option(argv: list[str], names: tuple[str, ...], value: str) -> list[str]:
    if _has_option(argv, names):
        return argv
    return [*argv, names[0], value]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for --debug: {value}")


def _consume_debug(argv: list[str]) -> tuple[list[str], bool]:
    result: list[str] = []
    debug = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--debug":
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                debug = _parse_bool(argv[index + 1])
                index += 2
            else:
                debug = True
                index += 1
            continue
        if arg.startswith("--debug="):
            debug = _parse_bool(arg.split("=", 1)[1])
            index += 1
            continue
        result.append(arg)
        index += 1
    return result, debug


def _consume_option(argv: list[str], names: tuple[str, ...]) -> tuple[list[str], str | None]:
    result: list[str] = []
    value: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in names:
            if index + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        matched = False
        for name in names:
            if arg.startswith(f"{name}="):
                value = arg.split("=", 1)[1]
                matched = True
                break
        if not matched:
            result.append(arg)
        index += 1
    return result, value


def _remove_option(argv: list[str], names: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in names:
            if index + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            index += 2
            continue
        if any(arg.startswith(f"{name}=") for name in names):
            index += 1
            continue
        result.append(arg)
        index += 1
    return result


def _remove_flag(argv: list[str], names: tuple[str, ...]) -> list[str]:
    return [arg for arg in argv if arg not in names and not any(arg.startswith(f"{name}=") for name in names)]


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
        "  python scripts/play_dexanytwist.py --checkpoint logs/rsl_rl/gx4m_dexanytwist/<run>/model_1000.pt"
    )


def _with_defaults(argv: list[str]) -> list[str]:
    result, debug = _consume_debug(list(argv))
    result, object_name = _consume_option(result, ("--object-name", "--object_name"))
    _require_checkpoint(result)
    if debug:
        os.environ["HEADLESS"] = "0"
        result = _remove_flag(result, ("--headless",))
        result = _remove_option(result, ("--num_envs",))
        result.extend(("--num_envs", DEBUG_NUM_ENVS))
        if not _has_hydra_override(result, "env.scene.clone_in_fabric"):
            result.append("env.scene.clone_in_fabric=false")
        if not _has_hydra_override(result, "env.print_hand_joint_limits"):
            result.append("env.print_hand_joint_limits=True")
        print(
            f"[play_dexanytwist] debug=True: using {DEBUG_NUM_ENVS} envs with Isaac Sim viewer"
            " and clone_in_fabric=false.",
            flush=True,
        )
    elif not _has_option(result, ("--headless",)):
        os.environ["HEADLESS"] = "0"
    result = _with_default_option(result, ("--task",), TASK)
    result = _with_default_option(result, ("--experiment_name",), EXPERIMENT_NAME)
    result = _with_default_option(result, ("--num_envs",), "1")
    if not _has_hydra_override(result, "env.hand_name"):
        result.append("env.hand_name=gx4m")
    if not any(
        _has_hydra_override(result, key)
        for key in ("env.object_name", "env.object_names", "env.object_set")
    ):
        result.append(f"env.object_name={object_name or OBJECT_NAME}")
    if not _has_hydra_override(result, "env.enable_scale_range"):
        result.append("env.enable_scale_range=False")
    if not _has_hydra_override(result, "env.apply_visual_materials"):
        result.append("env.apply_visual_materials=True")
    # Preserve link17 as a separate visual prim so the fourth red fingertip is
    # not recolored by its gray fixed-joint parent (link16).
    if not _has_hydra_override(result, "env.hand_cfg.spawn.merge_fixed_joints"):
        result.append("env.hand_cfg.spawn.merge_fixed_joints=False")
    if not _has_hydra_override(result, "env.show_keypoint_markers"):
        result.append(f"env.show_keypoint_markers={str(debug)}")
    return result


def _with_local_isaacsim_ext_folders(argv: list[str]) -> list[str]:
    if _has_option(argv, ("--kit_args",)):
        return argv

    import isaacsim

    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    ext_folders = [
        isaacsim_root / "exts",
        isaacsim_root / "extsDeprecated",
        isaacsim_root / "extscache",
        isaacsim_root / "extsPhysics",
        isaacsim_root / "kit" / "data" / "Kit" / "Isaac-Sim" / "5.1" / "exts" / "3",
    ]
    kit_args = " ".join(f'--ext-folder="{path.as_posix()}"' for path in ext_folders if path.exists())
    return [*argv, "--kit_args", kit_args] if kit_args else argv


def main() -> None:
    os.chdir(REPO_ROOT)
    isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT))
    play_script = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    if not play_script.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab RSL-RL play script at: {play_script}")

    sys.path.insert(0, str(REPO_ROOT / "source" / "gex_learning"))
    import gex_learning.tasks  # noqa: F401

    forwarded_argv = _with_local_isaacsim_ext_folders(_with_defaults(sys.argv[1:]))
    sys.path.insert(0, str(play_script.parent))
    sys.argv = [str(play_script), *forwarded_argv]
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
