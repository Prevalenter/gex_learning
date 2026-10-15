# gex_learning

[English](README.md) | **简体中文**

GX4M 灵巧手的 Isaac Lab 仿真训练与评估代码。包含以下四个任务：

| 功能 | 模块 | Gym 任务 ID |
| --- | --- | --- |
| AnyGrasp，Franka + GX4M 抓取方块 | `dexanygrasp` | `Gx4m-DexAnyGrasp-Direct-v0` |
| Twist，灯泡旋拧 | `dexanytwist` | `Gx4m-DexAnyTwist-Direct-v0` |
| Cubic，DexCube 掌内重定向 | `inhand_reorient` | `Gx4m-InHandReorient-Direct-v0` |
| Super In-Air，60 mm 网球旋转 | `super_inair` | `Gx4m-SuperInAir-Rotation-Direct-v0` |

Super In-Air 只支持网球；首次使用需准备网球资产和有效抓握缓存，详见
[网球训练与迁移说明](docs/SUPER_INAIR.md)。本机已准备完毕，运行
`.\experiment\super_inair\train.ps1` 即从头训练一小时。

Twist 仅支持 `bulb_custom_0017`，保留定制初始姿态、关节目标、点特征和原类别编号 93。
原有任务 ID 和机器人名称保持不变。这里的 AnyGrasp 指本仓库的强化学习任务。
不包含真机部署、预训练权重、velocity/student 扩展及其他实验。

## 安装

以已有 Isaac Lab 环境为基础。本次验证环境：Windows、Python 3.11、Isaac Lab 2.3.2、
Isaac Sim 5.1.0、PyTorch 2.7.0+cu128、RSL-RL 5.0.1、NumPy 1.26.0。
这些版本描述本次本地环境，不保证其他版本已通过仿真验证。

先激活安装了 Isaac Lab、Isaac Sim 和 RSL-RL 的 Python 环境，然后在仓库根目录执行：

```powershell
$env:ISAACLAB_ROOT = "D:\tools\IsaacLab"  # 改成自己的 Isaac Lab 源码目录；editable 安装可自动定位，省略此行
python -m pip install -e ".[viser,test]"
python scripts/prepare_assets.py --check
```

AnyGrasp、Twist 和 Cubic 所需资产已随源码提供；网球资产及抓握缓存按上方准备说明恢复。
AnyGrasp 的方块由环境代码生成；Cubic 使用随附的 DexCube 网格与纹理。
完整对应关系见 [任务资产](asset/README.md)。如需从原资产包恢复文件，可运行
`python scripts/prepare_assets.py --source <资产目录>`。
本项目支持在源码目录中 editable 安装；运行脚本和资产需保留在同一份 checkout 中。

PowerShell 启动器默认调用当前环境的 `python`。需要指定解释器时：

```powershell
$env:PYTHON_EXECUTABLE = "D:\envs\isaaclab\python.exe"
```

Linux 可使用下面的 Python 入口并设置 `export ISAACLAB_ROOT=/path/to/IsaacLab`；本次未进行 Linux 仿真验证。

## 训练

下面的入口默认无窗口训练，`--num_envs` 根据显存调整：

```powershell
python scripts/train_dexanygrasp.py --num_envs 4096 --max_iterations 10000
python scripts/train_dexanytwist.py --num_envs 4096 --max_iterations 10000
python scripts/train_inhand_reorient.py --num_envs 8192 --max_iterations 10000
python scripts/train_super_inair.py --num_envs 8192 --max_iterations 100000 --max-training-hours 1
```

首次检查可用 `--num_envs 4 --max_iterations 1`，Twist 另加 `env.scale_variant_count=2`；
本机默认 30 个尺寸变体曾在转换阶段超时，详见验证记录。
`--debug true` 打开窗口并使用入口预设的小批量环境。
同样可运行 `experiment/<模块>/train.ps1`；这些便捷启动器保留原训练规模，应按机器显存覆盖参数。

默认日志分别写入 `logs/rsl_rl/gx4m_dexanygrasp_ducky`、`gx4m_dexanytwist`、
`gx4m_inhand_reorient`、`gx4m_super_inair_tennis`。AnyGrasp 日志名中的 `ducky` 是历史命名，当前物体是方块。
这些日志、生成权重和采样数据被 `.gitignore` 排除。

AnyGrasp、Twist 和 Cubic 可按实际训练时间停止并保存检查点，例如：

```powershell
python scripts/train_dexanygrasp.py --num_envs 4096 --max_iterations 1000000 --max-training-hours 0.5
python scripts/train_dexanytwist.py --num_envs 4096 --max_iterations 1000000 --max-training-hours 1
python scripts/train_inhand_reorient.py --num_envs 8192 --max_iterations 1000000 --max-training-hours 1
```

计时从学习循环开始，不包含仿真初始化；达到时限后完成当前 iteration、保存模型并正常退出。
运行状态和预计截止时间写入训练目录的 `training_time_limit.json`。

## 回放与评估

将 `<checkpoint>` 替换为上述训练产生的 `model_*.pt`：

```powershell
python scripts/play_dexanygrasp.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_dexanytwist.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_inhand_reorient.py --checkpoint <checkpoint> --num_envs 1
python scripts/play_super_inair.py --checkpoint <checkpoint> --num_envs 1
```

`play_*.py` 使用 Isaac Lab 官方回放器，要求显式提供检查点。`test_*.py` 是带时长限制的回放入口，
不是 pytest 测试；可指定 `--duration 10 --headless --no-real-time`。
`test_*.py` 未提供检查点时从当前仓库对应日志目录选择最新模型。
已验证无窗口回放；Cubic 修复 Windows 图形后端后也已验证窗口回放。
尚未完成所有任务的 GUI 验证；历史渲染问题和验证范围见验证记录。

```powershell
python scripts/test_dexanytwist.py --checkpoint <checkpoint> --num_envs 1 --duration 10 --headless --no-real-time
```

Cubic / InHandReorient 默认使用 `env.goal_rotation_axes=xyz` 和 `env.goal_mode=random`，
在 X、Y、Z 三轴上组合采样 90° 的整数倍旋转作为目标姿态。默认
`env.policy_joint_action_only=False`：策略使用 47 维观测，包含关节位置、上一帧动作、
物体位置与姿态、目标姿态及姿态误差。随机目标不允许搭配看不到目标的 32 维观测。
训练入口默认 8192 个环境，与原目录成功的 Z 轴实验一致，可用 `--num_envs` 覆盖。
方块尺寸倍率为 `0.82–0.88`，实际边长为 **49.2–52.8 mm**。
此前生成的 32 维模型不能直接续训或回放为新的 47 维策略，需要从头训练。
若要按旧 Z 轴配置回放，显式传入
`env.goal_rotation_axes=z env.goal_mode=continuous_ccw env.policy_joint_action_only=True`。
配置对比和验证边界见 [Cubic 训练修正记录](docs/CUBIC_TRAINING_FIX.md)。

配置工具：

```powershell
python scripts/configure_dexanytwist_viser.py
python scripts/configure_inhand_reorient_viser.py
```

配置工具会修改对应 JSON。Cubic / InHandReorient 的专用配置器默认位于
`http://127.0.0.1:8083`，支持 16 个初始关节位置、方块平移与旋转，
也可通过 `experiment/inhand_reorient/configure_viser.ps1` 启动。
点击“保存初始姿态”后，新启动的训练使用保存姿态重置。
Cubic 回放优先恢复检查点旁记录的训练配置；显式传入
`env.use_initial_pose_config=True` 可使用编辑器最新保存的配置。
详细行为见 [Cubic 初始姿态配置](docs/INHAND_INITIAL_POSE.md)。
`experiment/<模块>/sample.ps1` 采集轨迹；`tensorboard.ps1` 查看该任务的训练曲线。

## 目录与验证

```text
source/gex_learning/gex_learning/   四个环境、任务注册及公共模块
scripts/                           训练、回放、评估、配置和资产检查
experiment/                        四个任务的 PowerShell 启动器
asset/                             资产清单、准备说明及本地模型
tests/                             不启动仿真的单元测试
docs/                              来源、迁移清单及验证结果
```

```powershell
python -m pytest -q
python scripts/prepare_assets.py --check
python scripts/check_usd_assets.py
```

详细结果见 [验证记录](docs/VALIDATION.md)，保留和排除范围见 [迁移说明](docs/MIGRATION.md)。
项目自身代码采用 [MIT](LICENSE)；第三方资产与外部依赖见 [来源与许可说明](docs/THIRD_PARTY_NOTICES.md)。
