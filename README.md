# SoccerMaster

这是 SoccerMaster 的个人复现与研究工作区。目录已经分成冻结基线、可修改研究代码、统一资产、运行结果和历史归档五部分。

## 当前结构

```text
baseline/   原始代码、上游配置和官方checkpoint入口；冻结不改
research/   当前开发区：源码、配置、复现、实验、工具和测试
assets/     checkpoint和固定数据；不进Git
runs/       日志、指标、图片、训练输出和导出结果
archive/    2026-08-20重构前的Gate、实验、配置和工具历史
docs/       架构、资产、代码地图、Harness和研究笔记
```

根目录只保留项目入口和状态文件。`train.py`与`eval.py`是指向`research/src/soccermaster/`的薄启动器。

## 研究代码

```text
research/src/soccermaster/
├── models/          当前多任务模型实现
├── data/            Dataset、transform和DataLoader
├── backbones/       新Backbone扩展入口
├── tasks/           新子任务扩展入口
├── pipelines/       Backbone与任务的组合
├── integrations/    SoccerFactory等外部系统接口
├── training/
├── evaluation/
└── utils/
```

新增实验从`research/experiments/_template/`开始。不要复制完整源码；修改正式模块，再用实验目录记录小配置和运行入口。

## 当前复现边界

G0–G8达到既定最小证明范围；G9完整训练未通过；G10固定SNGS-10004接口链已跑通，但角色、球队、号码和坐标语义质量仍未确认。唯一权威状态见[REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

## 重要入口

- [文档总索引](docs/README.md)
- [目录说明](docs/LOCAL_PROJECT_LAYOUT.md)
- [资产清单](docs/ASSET_INVENTORY.md)
- [SoccerMaster代码地图](docs/PIPELINE_MAP_SOCCERMASTER.md)
- [SoccerFactory代码地图](docs/PIPELINE_MAP_SOCCERFACTORY.md)
- [历史路径映射](archive/PATH_MAP.md)
- [执行纪律](docs/HARNESS.md)

原始远端目录`/remote-home/haolinyang/sports/Soccer-Backbone`永久只读。任何GPU推理、评估或训练仍需重新检查资源并获得当次明确授权。
