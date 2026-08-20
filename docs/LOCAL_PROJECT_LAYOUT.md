# 本地项目结构

更新：2026-08-20 UTC。

## 一级目录

| 目录 | 内容 | 是否日常修改 |
|---|---|---|
| `baseline/` | 冻结的原始SoccerMaster、SoccerFactory、上游配置和官方checkpoint入口 | 否 |
| `research/` | 自己的源码、配置、复现和实验 | 是 |
| `assets/` | checkpoint、预训练模型和固定数据 | 只新增，不覆盖官方资产 |
| `runs/` | 一次运行的日志、结果、图片和checkpoint | 由运行生成 |
| `archive/` | 重构前的Gate、实验、配置和工具 | 否 |
| `docs/` | 当前说明、代码地图和研究笔记 | 按需 |

## 根目录保留项

根目录只保留稳定入口和全局账本，不再放单次实验脚本：

| 文件 | 职责 |
|---|---|
| `README.md` | 项目入口和导航 |
| `AGENTS.md` | 长期安全边界与执行规则 |
| `REPRODUCTION_STATUS.md` | 唯一复现状态账本 |
| `research_log.md` | 研究假设、修改、结果和去留 |
| `train.py`、`eval.py` | 指向`research/src/soccermaster/`的稳定薄入口 |
| `setup.py` | 本地研究包安装入口 |
| `pyrightconfig.json` | 当前研究源码的编辑器分析范围 |

隐藏目录`.envs/`和`.runtime/`分别存放本地环境与可重建运行时数据。`.local_assets`、`.local_envs`、`.local_deps`和`.conda_pkgs`只是旧命令兼容链接，不是第二份资产。

## 代码边界

```text
baseline/code/soccermaster/       原始SoccerMaster参考
baseline/code/soccerfactory/      TrackLab、sn-gamestate、Refiner参考

research/src/soccermaster/        当前可修改代码
├── models/                        现有模型与五个任务头
├── data/                          数据读取
├── backbones/                     新Backbone入口
├── tasks/                         新子任务入口
├── pipelines/                     多任务组合
└── integrations/soccerfactory/   SoccerFactory到SoccerMaster的接口
```

现有上游模型暂时保留在`models/`，避免一次重构同时改变所有模型语义。新任务进入`tasks/<name>/`，新Backbone进入`backbones/<name>/`；确认接口后，再把旧模块逐个迁移，而不是复制一份。

## checkpoint

```text
assets/checkpoints/
├── official/
│   ├── soccermaster/epoch_19/
│   └── soccerfactory/
├── reproduced/
└── experimental/
```

`baseline/checkpoints`只是指向`assets/checkpoints/official`的链接，不产生第二份模型文件。`.local_assets`是旧命令兼容链接；新配置只使用`assets/`。

## 实验与日志

实验定义和运行结果分开：

```text
research/experiments/event_localization_v1/
├── README.md
├── config.yaml
└── run.py

runs/experiments/event_localization_v1/20260820_seed42/
├── resolved_config.yaml
├── run.log
├── result.json
├── figures/
└── checkpoints/
```

根目录`research_log.md`只记录假设、修改、结果、原因判断和是否保留，并链接对应run。原始日志不复制进研究日志。

## 历史材料

重构前路径没有删除，映射见`archive/PATH_MAP.md`。历史Gate结论仍由根目录`REPRODUCTION_STATUS.md`解释；归档脚本不会作为新实验模板继续扩散。

## 文档结构

```text
docs/
├── README.md                    文档总索引
├── LOCAL_PROJECT_LAYOUT.md      当前目录架构
├── ASSET_INVENTORY.md           资产位置和验证边界
├── PIPELINE_MAP_*.md            两条系统代码地图
├── HARNESS.md                   正式Gate协议
├── guides/                      当前操作指南
├── research/                    术语、组会材料和每日研究记录
├── operations/                  集群与运维知识
├── adr/                         架构决策
└── future_improvements/         尚未验证的研究方向
```
