# Super In-Air：网球旋转

迁移自原工作目录的 `super_inair` 任务，仅支持 `tennis_ball_60`。任务 ID 为
`Gx4m-SuperInAir-Rotation-Direct-v0`，不注册抓握采集任务或其他物体任务。

保留原 `train_tennis.ps1` 的固定手腕、隐式 PD 控制、手腕坐标系 +X 旋转轴、
8192 个环境及一小时训练时限。训练从随机初始化的策略开始，初始抓握从经过验证的缓存采样。
Actor/critic 均使用 LSTM；保留原 256 维 actor 观测，以及网球在六槽物体编码中的第六槽。
其他五槽只是为保持原观测布局预留，不开放其他物体。

奖励沿用迁移时原代码的 HORA 实现，包括沿目标轴角速度奖励，以及位移、偏轴、关节姿态、
力矩、功率和指尖失去接触的惩罚。当前位置失败阈值是 0.02 m，失败惩罚为 -30，
初始关节姿态惩罚系数为 -0.1。原 2026-10-01 历史训练保存的这三个值分别为
0.04 m、0、-0.3；本次以原目录当前代码为准，没有套用 Cubic 的奖励修改。

## 资产与抓握缓存

本机已准备网球 URDF、视觉网格、碰撞网格、材质和纹理，以及原网球专用缓存中的
1000 个有效抓握。缓存位于 `outputs/super_inair/tennis_grasps/`，不进入开源交付。
网球网格来源为 HORA 中的 YCB `056_tennis_ball/google_16k`。保留原 HORA 许可，
但该许可不能单独证明 YCB 网格的分发授权，因此这些网格和纹理只在本地准备，
由 Git 和源码包排除。没有复制其他 HORA 物体或参考仓库。

新机器先从有使用授权的资产包恢复 `asset/manifest.json` 中的文件：

```powershell
python scripts/prepare_assets.py --source <资产目录>
python scripts/prepare_assets.py --check
```

已有网球缓存时，安装并检查：

```powershell
python scripts/prepare_super_inair_tennis.py --cache <网球缓存目录>
python scripts/prepare_super_inair_tennis.py --check
```

没有缓存时，可基于附带的 `experiment/super_inair/objects/tennis_ball_60.json` 生成：

```powershell
python scripts/super_inair_grasps.py generate --headless --num_envs 256 --count 1000
python scripts/prepare_super_inair_tennis.py --cache outputs/super_inair/<生成时间>_generate
```

生成器执行稳定抓握和恢复检查；只有状态为 `complete` 的非空网球缓存才能用于训练。
本次正式训练使用原有的 1000 个有效抓握，没有重新采集这些数据。

## 训练与回放

```powershell
.\experiment\super_inair\train.ps1
# 或显式指定规模和时限
python scripts/train_super_inair.py --num_envs 8192 --max_iterations 100000 --max-training-hours 1
.\experiment\super_inair\play.ps1 --checkpoint <model_*.pt>
# 有界无窗口回放检查
python scripts/test_super_inair.py --checkpoint <model_*.pt> --headless --duration 5
```

日志写入 `logs/rsl_rl/gx4m_super_inair_tennis/`。计时从学习循环开始，一小时后完成
当前迭代、保存模型并退出。PowerShell 使用 `PYTHON_EXECUTABLE` 或当前环境的 `python`；
Isaac Lab 使用 `ISAACLAB_ROOT` 或已有 editable 安装的位置。

迁移仅抽出了原父类中实际使用的命令状态、观测写入和终止判断，复用公共旋转数学模块。
未引入原 velocity/student 任务、其他物体、真机部署或模型权重。

## 本次验证（2026-10-14）

- 60 项测试通过，包含迁移的抓握缓存、接触超时和偏轴奖励测试，以及新增的网球白名单、
  原物体编码槽、入口默认值和跨工作目录缓存解析检查。未迁移不再提供的旧编辑器及旧回放兼容模式测试。
- 77 个所需资产文件存在；URDF、OBJ 材质及纹理引用均可在本仓库内解析。
- Isaac Lab 2.3.2 / Isaac Sim 5.1 / RSL-RL 5.0.1：8 环境短训练完成 2 个迭代，
  生成 `model_1.pt`；从临时工作目录启动 1 环境无窗口回放，完成 54 步后正常结束。
- 正式训练为 8192 环境、随机初始化策略、1000 个有效抓握、一小时时限。
  学习循环北京时间 18:02:21 开始，预计 19:02:21 后完成当前迭代并保存退出。
  运行目录：`logs/rsl_rl/gx4m_super_inair_tennis/2026-10-14_18-01-35_tennis_fixed_base_1h_20261014_180129/`。
- 本次未等待一小时训练结束，未声称策略已收敛；未重新完整生成 1000 个抓握，
  也未对 GUI 回放进行视觉验收。短训练与回放日志、权重及缓存均在忽略目录中。
- Kit 日志仍有本机图形缓存初始化报错 `DriverShaderCacheManager::init()`；
  本次训练与有界回放均越过该消息继续执行，以实际学习迭代和回放步数作为运行验证依据。
