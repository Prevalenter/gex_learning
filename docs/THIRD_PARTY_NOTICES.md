# 第三方来源与许可

根目录 MIT 适用于本项目自行编写和拥有版权的代码，不重新许可第三方模型或外部依赖。
本说明根据迁移时的本地文件记录来源，不代表额外授予分发权。

| 内容 | 本地记录的来源 | 本次处理 |
| --- | --- | --- |
| GX4M URDF、网格 | 原工作目录的 GX4M 模型 | 随项目保留所需模型，资产许可不由项目 MIT 替代 |
| Franka + GX4M | Isaac Sim URDF importer 中的 `franka_description` 示例与 GX4M 组合 | 随项目保留必要组合资源及来源说明 |
| DexCube | Isaac Sim 5.1 资产服务，`Isaac/Props/Blocks/DexCube/` | 随项目保留 USD、纹理和原下载来源记录；保留原资产条款 |
| Twist 灯泡与配置 | DexAnyTwist 数据集及本地定制 | 随项目保留指定灯泡模型、配置和点特征 |
| Isaac Lab / Isaac Sim / RSL-RL | 用户安装的外部运行环境 | 不复制外部源码或安装文件，保留原有许可 |

DexAnyTwist 本地参考 README 声明 MIT，数据集独立授权文件未在本次选定资产中找到。
因此不将这个 README 声明扩大解释为灯泡网格的分发许可。
研究来源：[DexAnyTwist 项目页](https://prevalenter.github.io/dexanytwist.github.io/)。

DexCube 原下载记录中的公共资源前缀：
`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Blocks/DexCube/`。
包含 `dex_cube_instanceable.usd`、`Props/instanceable_meshes.usd`、`Materials/dex_cube_mod.png`。

本次没有将整个 `ref` 参考仓库或 SDK 拷入新仓库。

Super In-Air 网球来源为原 HORA 资产目录的 YCB `056_tennis_ball/google_16k`，
使用缩放后的 `tennis_ball_60.urdf`。HORA 来源记录为 `HaozhiQi/hora`，
保留其原许可于 `asset/HORA/LICENSE`。本地没有找到能独立确认 YCB 网格分发授权的文件，
因此网格、碰撞模型和纹理仅复制到本地运行目录，Git 和源码包排除这些文件；
准备步骤见 [Super In-Air](SUPER_INAIR.md)。不以项目 MIT 或 HORA 的许可替代网球资产许可。
