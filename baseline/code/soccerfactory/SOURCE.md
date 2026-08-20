# SoccerFactory vendored source

本目录只保存固定G10链路需要修改和本地import的源码包，不包含模型权重、训练数据、历史outputs、`.git`或Python缓存。远端原始目录保持只读。

复制日期：2026-08-19。

| 本地目录 | 只读来源 | 来源revision | 本次内容 |
|---|---|---|---|
| `tracklab/` | `/remote-home/haolinyang/sports/soccernet/tracklab` | `8987e8525709b5d37751ba60f135760fc91792ad` | `tracklab/`、`plugins/`、`pyproject.toml`、`LICENSE` |
| `sn-gamestate/` | `/remote-home/haolinyang/sports/soccernet/sn-gamestate` | `bda5ef3f37b66d7d16c22dba234745c05c629ac9` | `sn_gamestate/`、`plugins/`、`pyproject.toml` |
| `refiner/` | `/remote-home/haolinyang/sports/soccernet/Refiner` | `d4a06f77aebcb45eea1e54b47991dc80ee0f239a` | `inference.py`、`requirements.txt`、`Radar.png`、`configs/`、`model/`、`dataset/` |

核心源目录复制前后均为963个非缓存文件、10,871,371 bytes；加上入口、`Radar.png`和元数据后，本地`vendor/soccerfactory`共约11.80 MB。

首次`rsync -a`因远端`nobody:nogroup`无法映射到本地group而退出23；没有改变远端。随后仅增加`--no-owner --no-group`完成一次补全，三个组件复制命令均退出0，源/目标核心文件数和总字节数一致。

TrackLab来源包含LICENSE。sn-gamestate和Refiner来源根目录在本次两层搜索中没有发现LICENSE，因此这些副本只用于当前私人工作区的复现和修改，在许可明确前不对外再分发。

本地import smoke已经确认9个关键模块均从本目录解析，CUDA未初始化且没有加载模型/checkpoint；唯一局部适配是抑制OpenMIM写入只读home缓存。现有G10配置和本地环境的`.pth`仍指向远端路径，权重和固定数据也尚未迁移。只有正式入口改成本地`PYTHONPATH`并完成固定样本验证后，才能称为运行时已切换到本地源码。
