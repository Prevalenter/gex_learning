# 验证记录

2026-10-15：Super In-Hand 已删除。下文 2026-10-14 的相关结果仅为历史记录。
当前四任务包含 Super In-Air 网球，详见 [网球说明](SUPER_INAIR.md)。

本次删除验证：70 项 pytest 通过，注册集合恰为四个保留任务；Python 语法及包发现检查通过，
包内没有已删除任务。网球继承的 47 个方法在类名替换后 AST 一致，PPO 参数保持原值。
使用原网球 `model_250.pt`、2 个环境完成 3 秒无窗口回放，实际推理循环完成且进程退出码为 0。
日志位于 `outputs/validation/remove_super_inhand/tennis_verify.log`，仍有本机图形插件启动警告；
本次没有重跑四任务训练或 GUI 回放。原 Cubic 三小时训练持续运行。

日期：2026-10-14。Windows 11、RTX 3090 24 GB、Python 3.11、Isaac Lab 2.3.2、
Isaac Sim 5.1.0、PyTorch 2.7.0+cu128、RSL-RL 5.0.1、NumPy 1.26.0。

## 已通过

- 23 项 pytest 检查：轨迹记录、六轴评估终止快照、仅四个任务注册、灯泡白名单、类别编号 93、
  128 维点特征、URDF 网格、临时手模型路径、个人路径排除、外部工作目录及运行环境定位。
- Python 语法编译和所有保留 PowerShell 入口的语法解析。
- 隔离构建 wheel 成功；检查包内包含 Super In-Hand JSON、灯泡 JSON 和两个 split 文件，
  不包含模型资产或训练权重。运行方式为源码 checkout + editable 安装，wheel 不是独立资产发行包。
- 71 个白名单资产文件的存在性和 URDF 网格引用检查（含补入的 DexCube 来源记录）；USD 层及其外部依赖检查通过。
- Twist 配置器只列出 `bulb_custom_0017`；Super In-Hand 配置器可输出当前 Hydra 参数。
- 迁移清单内源文件的 SHA-256 与原目录一致；原仓库的未提交修改状态保持不变。
- 后续资产补齐：所有白名单物体、纹理和机器人模型已纳入 Git 文件范围和源码 ZIP；
  源码包解压后可独立通过资产检查。wheel 仍只包含 Python 包，完整运行使用随附资产的源码目录。

## 短训练与检查点回放

每个任务用 4 个环境、seed 42、1 次 PPO iteration 训练；随后加载新生成的 `model_0.pt`，
无窗口回放 3 秒。启动进程使用仓库之外的临时工作目录，验证入口路径不依赖调用位置。
检查退出码、完成标记和日志异常，不以退出码 0 单独判定成功。

| 任务 | 训练 | 回放 | 验证配置 |
| --- | --- | --- | --- |
| AnyGrasp | 通过 | 189 步 | 默认任务配置 |
| Cubic / InHandReorient | 通过 | 58 步 | 默认任务配置 |
| Super In-Hand | 通过 | 52 步 | 默认任务配置 |
| Twist | 通过 | 168 步 | `env.scale_variant_count=2`，其余任务参数保持默认 |

前三项使用 `outputs/validation/migration_20261014_084717_*` 日志；Twist 使用
`outputs/validation/migration_20261014_085245_*` 日志。日志、权重和报告仅保存在本地，不进入 Git。

Super In-Hand 六轴评估另用 6 个环境、每方向 1 个 episode、`env.episode_length_s=1.0` 完成，
生成 `outputs/validation/axes.json`、`axes.episodes.csv` 和 `axes.env.yaml`。
六个方向都完成统计。短训练策略没有收敛，本次验证不代表抓取或重定向性能。

## 已知限制

2026-10-14 Cubic 改为 XYZ 随机目标后，另修正了策略看不到目标的配置遗漏。
真实环境目标观测检查、8192 环境 2 iteration 短训练、新检查点 59 步回放均通过；
详见 [Cubic 训练修正记录](CUBIC_TRAINING_FIX.md)。这些检查不代表 XYZ 任务已收敛。

- **Twist 默认 30 个尺寸变体未通过本次完整启动验证**：一次运行在 URDF/USD 转换阶段超过 240 秒，
  已按测试超时终止。2 个变体的短训练和回放通过，仍覆盖尺寸范围两个端点。保留原默认 30，
  没有为通过检查而改动训练算法；快速验证时显式使用 `env.scale_variant_count=2`。
- **可视化交互未验证通过**：本机强制在 headless 进程中加载带视口的自定义及官方 experience 时，
  Kit/Hydra 渲染插件发生原生访问冲突。纯 headless 流程通过。最终代码采用官方 AppLauncher
  自动选择 experience，不再保留旧自定义查看器。不能将无窗口回放结果视为 GUI 验收。
- Isaac Sim 对灯泡无几何的辅助关节会输出空 visual prim 引用警告；最终成功运行未发现
  `Failed to resolve mesh`、缺失资产异常或 Python traceback。
- 非隔离 wheel 构建因本机 `setuptools` / `packaging` 不匹配失败；标准隔离构建通过。
  未修改已有仿真环境的软件版本。
- 尚未验证 Linux、长时间训练收敛、真机运行和外部未提供的模型资产包。

## 复现

先准备资产并设置 `ISAACLAB_ROOT`，然后执行：

```powershell
python -m pytest -q
python scripts/prepare_assets.py --check
python scripts/check_usd_assets.py
python scripts/smoke_test.py
```

`smoke_test.py` 按顺序运行四个任务，Twist 的短测试显式使用两个尺寸变体；默认单进程超时为 240 秒。
结果写到 `outputs/validation/*_results.json`。可用 `--tasks dexanytwist` 单独验证，
或用 `--playback-only` 加载此前 migration 测试生成的检查点。
