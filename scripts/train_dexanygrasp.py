"""Train the GX4M DexAnyGrasp cube task with Isaac Lab RSL-RL."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import h5py  # noqa: F401  # Preload HDF5 DLLs before Kit changes DLL search paths on Windows.
import tensordict  # noqa: F401
from rsl_rl.runners import OnPolicyRunner  # noqa: F401
from training_time_limit import consume_time_limit, install_training_time_limit


REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import isaaclab_root as resolve_isaaclab_root
DEFAULT_ISAACLAB_ROOT = resolve_isaaclab_root()
TASK = "Gx4m-DexAnyGrasp-Direct-v0"
EXPERIMENT_NAME = "gx4m_dexanygrasp_ducky"
DEBUG_NUM_ENVS = "256"


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


def _with_defaults(argv: list[str]) -> list[str]:
    result, debug = _consume_debug(list(argv))
    if debug:
        os.environ["HEADLESS"] = "0"
        result = _remove_flag(result, ("--headless",))
        result = _remove_option(result, ("--num_envs",))
        result.extend(("--num_envs", DEBUG_NUM_ENVS))
        if not _has_hydra_override(result, "env.scene.clone_in_fabric"):
            result.append("env.scene.clone_in_fabric=false")
        if not _has_hydra_override(result, "env.apply_visual_materials"):
            result.append("env.apply_visual_materials=True")
        # Preserve fixed fingertip links so their URDF colors can be bound independently.
        if not _has_hydra_override(result, "env.robot_cfg.spawn.merge_fixed_joints"):
            result.append("env.robot_cfg.spawn.merge_fixed_joints=false")
        print(
            f"[train_dexanygrasp] debug=True: using {DEBUG_NUM_ENVS} envs with Isaac Sim viewer"
            ", clone_in_fabric=false, visual materials enabled, and merge_fixed_joints=false.",
            flush=True,
        )
    elif not _has_option(result, ("--headless",)):
        result.append("--headless")
    result = _with_default_option(result, ("--task",), TASK)
    result = _with_default_option(result, ("--experiment_name", "--experiment-name"), EXPERIMENT_NAME)
    if not _has_hydra_override(result, "env.hand_name"):
        result.append("env.hand_name=gx4m")
    return result


def _option_value(argv: list[str], names: tuple[str, ...], default: str) -> str:
    for index, arg in enumerate(argv):
        if arg in names and index + 1 < len(argv):
            return argv[index + 1]
        for name in names:
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return default


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
    train_script = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab RSL-RL train script at: {train_script}")

    sys.path.insert(0, str(REPO_ROOT / "source" / "gex_learning"))
    import gex_learning.tasks  # noqa: F401

    training_argv, max_training_hours = consume_time_limit(sys.argv[1:])
    if max_training_hours is not None:
        install_training_time_limit(OnPolicyRunner, max_training_hours)
    forwarded_argv = _with_local_isaacsim_ext_folders(_with_defaults(training_argv))
    print(
        "[train_dexanygrasp] task="
        f"{_option_value(forwarded_argv, ('--task',), TASK)} "
        "experiment="
        f"{_option_value(forwarded_argv, ('--experiment_name', '--experiment-name'), EXPERIMENT_NAME)}",
        flush=True,
    )
    sys.path.insert(0, str(train_script.parent))
    sys.argv = [str(train_script), *forwarded_argv]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
