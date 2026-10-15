"""Play a GX4M DexAnyGrasp checkpoint with Isaac Lab's RSL-RL play.py."""

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
TASK = "Gx4m-DexAnyGrasp-Direct-v0"
EXPERIMENT_NAME = "gx4m_dexanygrasp_ducky"


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
        "  python scripts/play_dexanygrasp.py --checkpoint logs/rsl_rl/gx4m_dexanygrasp_ducky/<run>/model_1000.pt"
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
