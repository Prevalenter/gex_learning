# 四个任务的资产

`manifest.json` 是本项目需要的资产白名单。仅准备 GX4M、Franka + GX4M、DexCube、一个定制灯泡和网球；
不需要完整的 Twist 数据集，也不需要鸭子、GX7、Craftnet、EGAD 等模型。

AnyGrasp、Twist 和 Cubic 的资产已随仓库和源码压缩包提供。网球资产已在本机准备；
其 YCB 网格分发许可尚未独立确认，因此网格与纹理不进入 Git 或源码包，
新机器按 [Super In-Air 准备步骤](../docs/SUPER_INAIR.md) 从授权资产包恢复。对应关系如下：

| 任务 | 操作物体 | 模型与依赖位置 |
| --- | --- | --- |
| AnyGrasp | 76 mm 方块，由环境的 `CuboidCfg` 直接生成 | 无需单独物体文件；机器人使用 `franka_gx4m/` |
| Twist | `bulb_custom_0017` 定制灯泡 | `twist_dataset/twist_dataset/twist_dataset_urdf/bulb/bulb_0017/` |
| Cubic / InHandReorient | 60 mm DexCube，训练缩放后 49.2–52.8 mm | `nucleus_dexcube/`，包含主 USD、网格 USD 和纹理 |
| Super In-Air | `tennis_ball_60`，约 60 mm、50 g | `HORA/selected/tennis_ball_60.urdf`；视觉/碰撞网格、材质、纹理来自 `HORA/assets/ycb/056_tennis_ball/google_16k/` |

`gx4m/` 提供手模型；`franka_gx4m/` 提供 AnyGrasp 所需的组合机器人 URDF、网格和 USD 层。
原仓库的鸭子属于未迁移的 in-air 任务。

## 检查和恢复

正常使用只需运行 `python scripts/prepare_assets.py --check`。如需恢复删除的文件，可从同布局的资产包导入。

资产包根目录应包含：

```text
gx4m/urdf/gx4m.urdf
gx4m/meshes/...
franka_gx4m/urdf/franka_gx4m.urdf
franka_gx4m/meshes/...
franka_gx4m/usd/franka_gx4m.usd
franka_gx4m/usd/configuration/...
nucleus_dexcube/dex_cube_instanceable.usd
nucleus_dexcube/Props/instanceable_meshes.usd
nucleus_dexcube/Materials/dex_cube_mod.png
twist_dataset/twist_dataset/twist_dataset_urdf/bulb/bulb_0017/...
```

用 `python scripts/prepare_assets.py --source <资产包根目录>` 导入。
已有旧工作目录时，传入其 `asset` 子目录即可；脚本只复制白名单文件，运行不依赖旧仓库。
导入时会将 URDF 的 `package://.../meshes/` 改成相对引用，避免依赖 ROS 包查找或个人绝对路径。
源文件保持不变。`--check` 只检查，不复制或修改文件。

灯泡必需文件只有 `urdf/bulb_0017_modified.urdf`、`meshes/base_link.STL`、
`meshes/link1.STL`、`meshes/link1_poisson_1024_feature.npy`。
配置名 `bulb_custom_0017` 与底层模型目录 `bulb_0017` 的差异是有意保留的。

## 来源与重新准备

- GX4M：向机器人模型提供方获取 `gx4m.urdf` 及对应网格。
- Franka + GX4M：需要现有组合 URDF、网格和 USD 层。它由 Isaac Sim URDF importer 的
  `franka_description` 示例与 GX4M 组合而成。自行重新导出时必须保留原来的挂载变换、关节顺序及物理配置，
  普通 Franka 模型不能直接替换组合模型。当前准备脚本不生成该组合模型。
- DexCube：来自 Isaac Sim 5.1 的 `Isaac/Props/Blocks/DexCube/`，保留上述三文件的相对布局。
  可通过已安装 Isaac Sim 的资产服务获取；环境在本地 DexCube 缺失时也保留 Isaac Lab 官方资产 URL 回退。
- Twist：来自 DexAnyTwist 数据集及本地定制。上游标准 `bulb_0017` 不保证等同于这里的 modified URDF；
  应使用与该配置匹配的授权资产包。

取得文件后执行 `python scripts/prepare_assets.py --check` 和
`python scripts/check_usd_assets.py`。第二个命令需要 Isaac Sim，检查 USD 层及纹理引用。

DexCube 的原下载记录保存在 `nucleus_dexcube/DOWNLOAD_SOURCE.txt`。
第三方资产保留原来源及许可说明，项目 MIT 不覆盖第三方资产许可。
