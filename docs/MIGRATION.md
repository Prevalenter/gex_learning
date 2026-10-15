# 迁移范围

2026-10-14 追加迁移：第五个任务 Super In-Air，仅保留 `tennis_ball_60`。
代码、入口、物体配置、原许可及最小资产引用已整理；本地准备 1000 条原有效抓握用于训练。
抓握缓存、权重、日志不进入交付；YCB 网格许可未独立确认，网格和纹理由本地授权资产包准备。
详细运行配置与验证结果见 [SUPER_INAIR.md](SUPER_INAIR.md)。

2026-10-15 已移除 Super In-Hand 任务及专用入口、配置和评估。当前保留以下四个任务。
网球使用的底层环境提取为 `super_inair/hand_env.py`，不注册独立任务；
Cubic 配置器的旋转函数提取为 `scripts/pose_math.py`。

从原 `gx4m_learning` 工作目录的当前文件内容选择性迁移，新项目名和 Python 包名为 `gex_learning`。
原 Git HEAD 为 `cc9d8e1`，所选文件按工作目录内容复制，不以该提交覆盖未提交内容。
逐文件来源与源内容 SHA-256 见 `migration_manifest.json`；该表记录迁移输入，新目录文件另有必要修改。

## 保留

- `dexanygrasp`、`dexanytwist`、`inhand_reorient`、仅网球的 `super_inair` 环境和 PPO 配置。
- 四个任务的训练、回放、测试、采样及 TensorBoard 入口。
- Twist / Cubic 的 viser 配置工具及网球抓握准备工具。
- 轨迹记录公共模块；从 velocity 提取的两个纯旋转数学函数，不包含 velocity 环境。
- 使用 Isaac Lab 官方查看器配置及对应测试。
- 白名单内的物体模型、纹理、机器人模型和点特征，一并纳入 Git 文件范围及源码压缩包。

## 排除

- `super_inhand` 及其 velocity/student 扩展、六轴评估、distillation、`inair_reorient`、
  `infinger_reorient`、`dexlong`、`dexrio_grasp` 及其入口。
- 真机 `deploy`、驱动库、其他机器人、其他 Twist 物体、Craftnet/EGAD/HORA 数据。
- `ref`、备份、压缩包、旧 Git 历史、编辑器配置、训练权重、日志、轨迹、视频和历史分析产物。
- cubic 下误指向鸭子任务的配置启动器。

## 有意调整

- 仅包名更改，保留 `Gx4m-…-Direct-v0` ID、`gx4m` 硬件名称及默认实验日志名称。
- Twist 全部入口统一为 `bulb_custom_0017`，原类别编号显式设为 93；取消对未携带物体的选择。
- 个人路径改为环境变量或仓库相对路径，必要 URDF 网格引用改为相对地址。
- 回放根据 headless/camera 选项自动使用 Isaac Lab 官方启动配置，去掉会触发本机 Vulkan 崩溃的旧自定义查看器。
- 配置 JSON 和 split 文件纳入 Python 包资源；源码 checkout + editable 安装是支持的使用方式。
- 不复用旧项目的 Git 历史，不创建远程仓库，不上传文件。
