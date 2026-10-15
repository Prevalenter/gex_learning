# Cubic XYZ 目标与策略观测修正

## 原训练记录对比

2026-10-14 对比原项目保留的 `gx4m_inhand_reorient` 训练日志、`params/env.yaml`、
`params/agent.yaml`，以及当前环境实现。原目录仅做读取。

原目录 2026-08-13 的四次长训练均为 **Z 轴连续逆时针旋转**，并非 XYZ 随机目标的成功记录。
其中 `2026-08-13_23-33-01_train` 的最后 100 个 iteration：

- `Episode/consecutive_successes` 平均约 18.69；
- `Episode/episode_success_rate` 平均约 0.9197；
- 平均 Z 轴角速度约 1.649 rad/s。

这些是训练中的日志统计，不是独立评估的成功率。该运行配置为 8192 个环境、
32 维策略观测（关节位置与上一帧动作）、20 秒 episode、LSTM 256 隐状态、
MLP `[512, 256, 128]`。PPO、奖励、物体尺寸范围 `[1.05, 1.12]`、动作平滑系数 0.2、
RNA 和随机扰动参数均与迁移版本一致。

停止的 `2026-10-14_14-27-35_cubic_xyz_2hours_20261014_142729` 运行至 iteration 2645，
最后 100 个 iteration 的 `Episode/episode_success_rate` 平均约 0.0270，
`Episode/consecutive_successes` 约 0.0305，姿态误差约 2.283 rad。
不同目标任务的这些指标不能直接作为性能排名，但确认此次随机目标训练没有取得明显进展。

## 问题与修改

之前只把 `goal_rotation_axes` 改为 `xyz`、`goal_mode` 改为 `random`，
却保留了 `policy_joint_action_only=True`。策略看不到物体姿态、目标姿态和姿态误差。
目标是独立随机采样的，仅凭关节位置和历史动作无法判断应转向哪个目标。
Critic 能看到目标也不能弥补 Actor 输入中缺失的信息。

修正后的默认配置：

| 配置 | 值 |
| --- | --- |
| `goal_rotation_axes` | `xyz` |
| `goal_mode` | `random` |
| `policy_joint_action_only` | `False` |
| Actor 观测 | 47 维：关节位置 16 + 上一帧动作 16 + 物体位置 3 + 物体姿态 4 + 目标姿态 4 + 姿态误差 4 |
| Critic 观测 | 237 维，未变 |
| Python / PowerShell 训练入口默认环境数 | 8192，可显式覆盖 |

复用环境已有的完整策略观测分支，保留原成功 Z 轴实验的 PPO、奖励和物理参数。
新增初始化检查：随机目标拒绝 `policy_joint_action_only=True`；Cubic 的 `continuous_ccw`
实现只支持 Z 轴，其他轴组合会明确报错，避免配置写 XYZ 实际仍沿 Z 轴旋转。

原 32 维检查点无法直接加载为 47 维网络，修正配置需要从头训练。旧 Z 轴模型仍可使用：

```powershell
python scripts/play_inhand_reorient.py --checkpoint <旧模型> env.goal_rotation_axes=z env.goal_mode=continuous_ccw env.policy_joint_action_only=True
```

## 验证

后续用户调整：Cubic 默认到达目标奖励先从 500 改为 1200，位置保持项改为 `-5 × d`。
2026-10-14 最新调整为到达奖励 **800**、位置保持项 **`-10 × d`**，其中 `d` 为物体到
掌内参考位置的欧氏距离（米）。继续使用 XYZ 随机目标及完整目标观测，从头开始训练。

`python scripts/check_inhand_goals.py` 使用真实 Isaac Lab 环境，固定物理状态、分别改变 X/Y/Z
目标，检查关节与物体输入不变而目标和误差观测随之变化；同时检查三轴采样及有限数值。
64 个环境、32 步检查通过。该检查覆盖此前短回放没有发现的目标输入遗漏。

已有 pytest 检查全部通过（32 项）。默认 8192 环境短训练完成 2 个 iteration，
实际保存配置为 `xyz / random / policy_joint_action_only=False`，Actor/Critic 为 47/237 维。
新 `model_1.pt` 在 4 个环境中完成无窗口回放 59 步，正常退出。
这些检查仅验证输入维度、训练更新和检查点兼容性，不代表 XYZ 策略已经收敛；
没有重新启动长期训练。

本地验证日志（不进入交付内容）：

- `outputs/training/cubic_config_comparison_metrics.json`：原训练和停止运行的 TensorBoard 指标；
- `outputs/training/cubic_goal_observation_check.log`：真实环境的目标观测检查；
- `outputs/training/cubic_xyz_observation_fix_smoke.log`：8192 环境短训练；
- `outputs/training/cubic_xyz_observation_fix_play.log`：新检查点回放。
