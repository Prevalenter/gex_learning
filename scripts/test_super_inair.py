"""Run bounded inference for a GX4M Super In-Air tennis checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import re
import sys
import time
from pathlib import Path

import h5py  # noqa: F401  # Preload HDF5 DLLs before Kit changes DLL search paths on Windows.
import tensordict  # noqa: F401
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
from runtime_paths import isaaclab_root as resolve_isaaclab_root
DEFAULT_ISAACLAB_ROOT = resolve_isaaclab_root()
TASK = "Gx4m-SuperInAir-Rotation-Direct-v0"
EXPERIMENT_NAME = "gx4m_super_inair_tennis"
MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to the latest model_*.pt.")
    parser.add_argument("--run-dir", default=None, help="Run directory to search for the latest checkpoint.")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--experiment-name", "--experiment_name", dest="experiment_name", default=EXPERIMENT_NAME)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. <=0 runs until the window closes.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.set_defaults(real_time=True)
    parser.add_argument("--show-goal-marker", dest="show_goal_marker", action="store_true")
    parser.add_argument("--hide-goal-marker", dest="show_goal_marker", action="store_false")
    parser.set_defaults(show_goal_marker=False)
    parser.add_argument("--apply-visual-materials", dest="apply_visual_materials", action="store_true")
    parser.add_argument("--no-visual-materials", dest="apply_visual_materials", action="store_false")
    parser.set_defaults(apply_visual_materials=True)
    parser.add_argument("--keep-ui-windows", action="store_true", help="Do not hide Isaac Sim side panels on startup.")
    parser.add_argument("--show-ground", dest="show_ground", action="store_true")
    parser.add_argument("--hide-ground", dest="show_ground", action="store_false")
    parser.set_defaults(show_ground=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--experience", default="")
    parser.add_argument("--kit_args", default=None)
    parser.add_argument("--logger", default=None)
    parser.add_argument("--log_project_name", default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--load_run", default=None)
    return parser.parse_known_args()


def _default_kit_args() -> str:
    import isaacsim

    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    ext_folders = [
        isaacsim_root / "exts",
        isaacsim_root / "extsDeprecated",
        isaacsim_root / "extscache",
        isaacsim_root / "extsPhysics",
        isaacsim_root / "kit" / "data" / "Kit" / "Isaac-Sim" / "5.1" / "exts" / "3",
        DEFAULT_ISAACLAB_ROOT / "apps",
        DEFAULT_ISAACLAB_ROOT / "source",
    ]
    args = [
        "--/app/extensions/registryEnabled=0",
        "--/telemetry/enableAnonymousData=false",
        "--/telemetry/enableAnonymousAppName=false",
    ]
    args.extend(f'--ext-folder="{path.as_posix()}"' for path in ext_folders if path.exists())
    return " ".join(args)


def _has_hydra_override(argv: list[str], key: str) -> bool:
    return any(arg.lstrip("+").startswith(f"{key}=") for arg in argv)


def _checkpoint_key(path: Path) -> tuple[int, float]:
    match = MODEL_RE.match(path.name)
    iteration = int(match.group(1)) if match else -1
    return iteration, path.stat().st_mtime


def _latest_checkpoint_in_run(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_key)
    return checkpoints[-1] if checkpoints else None


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()

    if args.run_dir:
        run_dirs = [Path(args.run_dir).expanduser().resolve()]
    else:
        root = REPO_ROOT / "logs" / "rsl_rl" / args.experiment_name
        if not root.exists():
            raise FileNotFoundError(f"Experiment log directory does not exist: {root}")
        run_dirs = [path for path in root.iterdir() if path.is_dir()]
        if args.run_name:
            run_dirs = [path for path in run_dirs if args.run_name in path.name]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        checkpoint = _latest_checkpoint_in_run(run_dir)
        if checkpoint is not None:
            return checkpoint.resolve()

    searched = args.run_dir or str(REPO_ROOT / "logs" / "rsl_rl" / args.experiment_name)
    raise FileNotFoundError(f"No model_*.pt checkpoint found under: {searched}")


ARGS, HYDRA_ARGS = _parse_args()
if ARGS.kit_args is None:
    ARGS.kit_args = _default_kit_args()
if ARGS.device is None:
    delattr(ARGS, "device")
if not _has_hydra_override(HYDRA_ARGS, "env.hand_name"):
    HYDRA_ARGS.append("env.hand_name=gx4m")

AppLauncher._hide_stop_button = lambda self: None
AppLauncher._hide_play_button = lambda self, hide=True: None

sys.argv = [sys.argv[0], *HYDRA_ARGS]
app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

import gymnasium as gym
import carb
if not ARGS.headless:
    import omni.ui as ui
import omni.usd
import torch
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(REPO_ROOT / "source" / "gex_learning"))
import gex_learning.tasks  # noqa: E402, F401

isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", DEFAULT_ISAACLAB_ROOT))
sys.path.insert(0, str(isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402


def _hide_startup_windows() -> None:
    if ARGS.keep_ui_windows or ARGS.headless:
        return

    for name in (
        "Stage",
        "Property",
        "Content",
        "Console",
        "Layer",
        "Isaac Sim Assets",
        "Samples",
        "Render Settings",
        "Simulation Settings",
        "Physics Debug",
        "Physics Inspector",
        "Physics API Editor",
        "AppSelector",
    ):
        try:
            ui.Workspace.show_window(name, False)
            window = ui.Workspace.get_window(name)
            if window is not None:
                window.visible = False
        except Exception:
            pass

    try:
        viewport = ui.Workspace.get_window("Viewport")
        if viewport is not None:
            viewport.visible = True
            viewport.focus()
    except Exception:
        pass


def _set_viewer_grid_visible(visible: bool) -> None:
    try:
        settings = carb.settings.get_settings()
        for key in (
            "/app/viewport/defaults/guide/grid/visible",
            "/persistent/app/viewport/grid/enabled",
            "/persistent/app/viewport/displayOptions/showGrid",
        ):
            settings.set_bool(key, visible)
    except Exception:
        pass


def _set_ground_visible(visible: bool) -> None:
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        from pxr import UsdGeom

        for path in ("/World/ground", "/World/defaultGroundPlane"):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
    except Exception:
        pass


@hydra_task_config(ARGS.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    checkpoint = _resolve_checkpoint(ARGS)
    print(f"[test_super_inair] checkpoint: {checkpoint}", flush=True)

    agent_cfg.experiment_name = ARGS.experiment_name
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, ARGS)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg.scene.num_envs = int(ARGS.num_envs)
    env_cfg.log_dir = str((REPO_ROOT / "outputs" / "test_super_inair").resolve())
    if ARGS.seed is not None:
        env_cfg.seed = ARGS.seed
        agent_cfg.seed = ARGS.seed
    if getattr(ARGS, "device", None) is not None:
        env_cfg.sim.device = ARGS.device
    if hasattr(env_cfg, "apply_visual_materials"):
        env_cfg.apply_visual_materials = bool(ARGS.apply_visual_materials)
    if hasattr(env_cfg, "show_goal_marker"):
        env_cfg.show_goal_marker = bool(ARGS.show_goal_marker)

    _set_viewer_grid_visible(bool(ARGS.show_ground))
    _hide_startup_windows()
    env = gym.make(ARGS.task, cfg=env_cfg, render_mode=None)
    _set_ground_visible(bool(ARGS.show_ground))
    _set_viewer_grid_visible(bool(ARGS.show_ground))
    _hide_startup_windows()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(str(checkpoint)))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = float(env.unwrapped.step_dt)
    deadline = None if ARGS.duration <= 0.0 else time.time() + ARGS.duration
    steps = 0
    print(
        f"[test_super_inair] running {ARGS.num_envs} env(s). Close the Isaac Sim window to stop.",
        flush=True,
    )
    _set_ground_visible(bool(ARGS.show_ground))
    _set_viewer_grid_visible(bool(ARGS.show_ground))
    _hide_startup_windows()
    try:
        while simulation_app.is_running() and (deadline is None or time.time() < deadline):
            step_start = time.time()
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=ARGS.stochastic)
                obs, _, dones, _ = env.step(actions)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
            steps += 1
            if ARGS.real_time:
                sleep_time = dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        print(f"[test_super_inair] finished after {steps} steps.", flush=True)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
