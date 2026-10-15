# gex_learning

**English** | [简体中文](README.zh-CN.md)

Isaac Lab simulation training and evaluation for the GX4M dexterous hand. This repository includes four tasks:

| Task | Module | Gym task ID |
| --- | --- | --- |
| AnyGrasp: cube grasping with Franka + GX4M | `dexanygrasp` | `Gx4m-DexAnyGrasp-Direct-v0` |
| Twist: light bulb twisting | `dexanytwist` | `Gx4m-DexAnyTwist-Direct-v0` |
| Cubic: in-hand DexCube reorientation | `inhand_reorient` | `Gx4m-InHandReorient-Direct-v0` |
| Super In-Air: 60 mm tennis ball rotation | `super_inair` | `Gx4m-SuperInAir-Rotation-Direct-v0` |

Super In-Air supports only the tennis ball. Prepare its assets and a valid grasp cache before training; see the [tennis task guide (Chinese)](docs/SUPER_INAIR.md). Once prepared, `.\experiment\super_inair\train.ps1` starts a fresh one-hour run.

Twist supports only `bulb_custom_0017`, preserving its custom initial pose, joint targets, point features, and original class index **93**. Task IDs and robot names retain their original spelling. Here, AnyGrasp refers to this repository's reinforcement learning task. Real robot deployment, pretrained weights, velocity/student extensions, and other experiments are excluded.

## Installation

Use an existing Isaac Lab environment. Local validation used Windows, Python 3.11, Isaac Lab 2.3.2, Isaac Sim 5.1.0, PyTorch 2.7.0+cu128, RSL-RL 5.0.1, and NumPy 1.26.0. Other versions have not been validated here.

Activate your Python environment with Isaac Lab, Isaac Sim, and RSL-RL installed, then run from the repository root:

```powershell
$env:ISAACLAB_ROOT = "D:\tools\IsaacLab"  # Your Isaac Lab checkout; omit if its editable installation is discoverable
python -m pip install -e ".[viser,test]"
python scripts/prepare_assets.py --check
```

Assets for AnyGrasp, Twist, and Cubic are included in the source checkout. Prepare the tennis assets and grasp cache separately using the guide above. AnyGrasp generates its cube in code; Cubic uses the included DexCube mesh and texture. See the [asset inventory (Chinese)](asset/README.md). To restore files from an asset bundle:

```powershell
python scripts/prepare_assets.py --source <asset-directory>
```

The supported installation is an editable source checkout. Keep scripts and assets together in that checkout.

PowerShell launchers use `python` from the current environment. To select an interpreter explicitly:

```powershell
$env:PYTHON_EXECUTABLE = "D:\envs\isaaclab\python.exe"
```

On Linux, use the Python entry points below and set `export ISAACLAB_ROOT=/path/to/IsaacLab`. Linux simulation has not been validated here.

## Training

These entry points run headless by default. Adjust `--num_envs` to fit your GPU memory:

```powershell
python scripts/train_dexanygrasp.py --num_envs 4096 --max_iterations 10000
python scripts/train_dexanytwist.py --num_envs 4096 --max_iterations 10000
python scripts/train_inhand_reorient.py --num_envs 8192 --max_iterations 10000
python scripts/train_super_inair.py --num_envs 8192 --max_iterations 100000 --max-training-hours 1
```

For a first smoke check, use `--num_envs 4 --max_iterations 1`. For Twist, also add `env.scale_variant_count=2`: the default 30 size variants previously timed out during conversion on the local machine.

Use `--debug true` to open the viewer with the entry point's small environment count. Alternatively, run `experiment/<module>/train.ps1`; override its environment count as needed for your GPU.

Default log directories under `logs/rsl_rl/` are `gx4m_dexanygrasp_ducky`, `gx4m_dexanytwist`, `gx4m_inhand_reorient`, and `gx4m_super_inair_tennis`. The `ducky` suffix is historical: AnyGrasp currently uses a cube. Logs, generated checkpoints, and sampled trajectories are excluded by `.gitignore`.

Training can stop and save after a specified wall time:

```powershell
python scripts/train_dexanygrasp.py --num_envs 4096 --max_iterations 1000000 --max-training-hours 0.5
python scripts/train_dexanytwist.py --num_envs 4096 --max_iterations 1000000 --max-training-hours 1
python scripts/train_inhand_reorient.py --num_envs 8192 --max_iterations 1000000 --max-training-hours 1
```

Timing starts at the learning loop, excluding simulation initialization. At the deadline, training finishes the current iteration, saves the model, and exits. The run directory's `training_time_limit.json` records status and the expected deadline.

## Playback and evaluation

Replace `<checkpoint>` with a `model_*.pt` produced by training:

```powershell
python scripts/play_dexanygrasp.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_dexanytwist.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_inhand_reorient.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_super_inair.py --checkpoint <checkpoint> --num_envs 1
```

The `play_*.py` scripts use Isaac Lab's playback runner and require an explicit checkpoint. The `test_*.py` scripts provide time-limited playback, not pytest tests. They accept `--duration 10 --headless --no-real-time` and select the latest checkpoint in the corresponding local log directory when none is specified.

```powershell
python scripts/test_dexanytwist.py --checkpoint <checkpoint> --num_envs 1 --duration 10 --headless --no-real-time
```

Headless playback has been checked. Cubic GUI playback was subsequently verified after fixing its Windows graphics backend; GUI playback has not been validated for every task. See the [validation record (Chinese)](docs/VALIDATION.md) for historical renderer issues and validation scope.

### Cubic / InHandReorient

Defaults are `env.goal_rotation_axes=xyz` and `env.goal_mode=random`. Goal orientations combine multiples of 90 degrees about X, Y, and Z. With `env.policy_joint_action_only=False`, the actor receives 47 observations: joint positions, previous actions, object position and orientation, goal orientation, and orientation error. Random goals require this goal-aware observation layout.

The training entry point defaults to 8192 environments, adjustable with `--num_envs`. Cube scale is randomized over `0.82–0.88`, giving a side length of **49.2–52.8 mm**. Existing 32-observation models cannot be resumed or played as 47-observation policies; train a new policy instead. To replay an older Z-axis model, explicitly pass:

```text
env.goal_rotation_axes=z env.goal_mode=continuous_ccw env.policy_joint_action_only=True
```

See the [Cubic configuration notes (Chinese)](docs/CUBIC_TRAINING_FIX.md) for the comparison and validation limits.

### Configuration tools

```powershell
python scripts/configure_dexanytwist_viser.py
python scripts/configure_inhand_reorient_viser.py
```

These tools edit task JSON files. The Cubic editor defaults to `http://127.0.0.1:8083` and supports 16 initial joint positions plus cube translation and rotation. You can also launch it through `experiment/inhand_reorient/configure_viser.ps1`.

Click **保存初始姿态** (Save initial pose) to use the saved pose in new training runs. Cubic playback restores settings from the checkpoint's training configuration when available; pass `env.use_initial_pose_config=True` to use the latest editor configuration explicitly. See the [initial pose guide (Chinese)](docs/INHAND_INITIAL_POSE.md).

Use `experiment/<module>/sample.ps1` to collect trajectories and `tensorboard.ps1` to inspect training curves.

## Repository layout and validation

```text
source/gex_learning/gex_learning/   Four tasks, registrations, and shared modules
scripts/                           Training, playback, evaluation, configuration, and asset checks
experiment/                        PowerShell launchers for the four tasks
asset/                             Asset manifest, preparation instructions, and local models
tests/                             Unit tests that do not start simulation
docs/                              Sources, migration inventory, and validation results
```

```powershell
python -m pytest -q
python scripts/prepare_assets.py --check
python scripts/check_usd_assets.py
```

See the [validation record](docs/VALIDATION.md) and [migration scope](docs/MIGRATION.md) (Chinese). Project-owned code uses the [MIT license](LICENSE). Third-party assets and dependencies retain their own terms; see [sources and licenses (Chinese)](docs/THIRD_PARTY_NOTICES.md).
