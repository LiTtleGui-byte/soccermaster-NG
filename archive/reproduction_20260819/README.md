# SoccerMaster 复现入口

`reproduction/` 只收纳已定义的 Gate 复现入口和固定输入，不放研究设想、大型权重或运行日志。

开始任何 Gate 前，依次阅读：

1. [`AGENTS.md`](../AGENTS.md)：安全与授权边界。
2. [`docs/HARNESS.md`](../docs/HARNESS.md)：Gate 协议和通过标准。
3. [`REPRODUCTION_STATUS.md`](../REPRODUCTION_STATUS.md)：唯一当前状态账本。

## Gate 入口

| Gate | 入口 | 固定输入 | 证据目录 |
| --- | --- | --- | --- |
| G2 | `gates/g2_checkpoint_load.py` | high-resolution config + `epoch_19` | `reports/g2/` |
| G3 | `gates/g3_random_forward.py` | 固定随机张量协议 | `reports/g3/` |
| G4 | `gates/g4_real_video.py` | 脚本内固定的单个真实视频 | `reports/g4/` |
| G5 | `gates/g5_fixed_eval.py` | `manifests/g5_fixed_eval.json` | `reports/g5/` |
| G6 | `gates/g6_tiny_overfit.py` | `manifests/g6_tiny_overfit.json` | `reports/g6/` |
| G7 | `gates/g7_single_task.py` | `manifests/g7_single_task.json` | `reports/g7/` |
| G8 静态预检 | `gates/g8_multitask.py` | `manifests/g8_multitask.json` | `reports/g8/` |
| G8 两卡运行 | `gates/g8_multitask_run.py` | `manifests/g8_multitask.json` | run5 功能性恢复通过；位级 exact resume 为 false |
| G10-B Step 1、Refiner契约/谱系审计与兼容转换 | `gates/g10_soccerfactory_step1_run2.py`、`gates/g10_refiner_input_contract_audit.py`、`gates/g10_prerefiner_lineage_audit.py`、`gates/g10_soccerfactory_convert.py` | `manifests/g10_soccerfactory_step1_run5_sngs10004.json`、`manifests/g10_refiner_input_contract_audit_run5.json`、`manifests/g10_prerefiner_lineage_audit.json`、`manifests/g10_soccerfactory_sngs10004.json` | `reports/g10/20260814_step1_run5/`、`reports/g10/20260814_refiner_input_contract_audit/`、`reports/g10/20260814_prerefiner_lineage_audit/`、`reports/g10/20260813_static_conversion/` |
| G10-B Refiner前置enrichment run1 | `gates/g10_prerefiner_enrichment.py` + `gates/g10_prerefiner_enrichment_worker.py` | `manifests/g10_prerefiner_enrichment_run1_sngs10004.json` + `configs/g10/g10_prerefiner_enrichment_sngs10004_run1.yaml` | GPU 0获批run1在dataset阶段超时；未生成兼容归档，禁止盲目重跑 |
| G10-B Refiner前置enrichment run2 | `gates/g10_prerefiner_enrichment_run2.py` + `gates/g10_prerefiner_enrichment_run2_worker.py` | `manifests/g10_prerefiner_enrichment_run2_sngs10004.json` + `configs/g10/g10_prerefiner_enrichment_sngs10004_run2.yaml` | GPU 0实际运行通过；39,016,979-byte新归档具备核心Refiner列 |
| G10-B enrichment run2静态契约审计 | `gates/g10_prerefiner_enriched_contract_audit.py` | `manifests/g10_prerefiner_enriched_contract_audit_run2.json` | CPU审计通过；显式255帧覆盖下静态入口就绪，默认750帧不兼容，语义质量/Refiner forward未验证 |
| G10-B enrichment dataset诊断run2 | `gates/g10_prerefiner_dataset_diagnosis.py` + `gates/g10_prerefiner_dataset_diagnosis_worker.py` | `manifests/g10_prerefiner_dataset_diagnosis_run2_sngs10004.json` + run1 dataset config | 实际run2在dataset wrapper前因Hydra完整快照解析sweep-only字段失败；现场保留 |
| G10-B enrichment dataset诊断run3 | `gates/g10_prerefiner_dataset_diagnosis_run3.py` + `gates/g10_prerefiner_dataset_diagnosis_run3_worker.py` | `manifests/g10_prerefiner_dataset_diagnosis_run3_sngs10004.json` + run1 dataset config | CPU实际诊断通过：双快照、1视频/255图像、frame 0..254及空GT契约成立 |
| G10-B Step 1 | `gates/g10_soccerfactory_step1.py` | `manifests/g10_soccerfactory_step1_sngs10004.json` + `configs/g10/g10_step1_sngs10004.yaml` | 静态预检通过；run1 在启动/数据实例化边界超时失败 |
| G10-B 启动诊断 | `gates/g10_soccerfactory_step1_diagnose.py` | run1/run2/run3 manifests | run3通过：TrackLab导入、Hydra、固定255帧dataset契约已确认 |
| G10-B Step 1 run2分阶段Harness | `gates/g10_soccerfactory_step1_run2.py` + `gates/g10_soccerfactory_step1_run2_worker.py` | `manifests/g10_soccerfactory_step1_run2_sngs10004.json` + run2 config | 最终静态预检通过；尚未获得本次GPU运行授权 |
| G10-C 当前产物 DataLoader 消费 | `gates/g10_current_pipeline_dataloader_smoke.py` | `manifests/g10_current_pipeline_dataloader_smoke_sngs10004.json` | CPU smoke 通过：准备图片到 SoccerMaster DataLoader 已通；原视频到准备图片仍缺历史切片实现 |

G0/G1主要是资产和环境盘点，当前没有独立Python Gate入口。G8 run5已完成两卡、两epoch、8步五头多任务训练、validation和功能性恢复；位级exact resume仍为false。G9完整训练仍未通过。G10已隔离提前启动；固定SNGS-10004现已从255张准备图片经Step 1、enrichment、Refiner、适配后Step 3、PKL转换并被真实SoccerMaster DataLoader消费。G10-C只证明准备图片到训练消费端的代码和格式链路；原始视频到255张准备图片的历史切片/填充实现仍未恢复，角色、球队、号码和坐标质量也仍未验证。下一步严格以状态账本为准。

## G8 交接点

G8 的 manifest、静态 Harness、已确认显存限制、候选固定输入、机器断言和明确未验证范围记录在 [`REPRODUCTION_STATUS.md`](../REPRODUCTION_STATUS.md) 的“G8 小规模多任务只读设计审查”中。

run5 的日志和机器结果位于 `reports/g8/20260812_run5/`。当前工作已经按批准的隔离例外提前进入 G10；新会话必须以 `REPRODUCTION_STATUS.md` 的“唯一下一步”为准，不能从本节的历史交接点直接启动 G9 或 G10 运行。

## 固定基线

- 工作区：`/home/tianlin/SoccerMaster`
- 本地 Python：`/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`
- 原始资产：`/remote-home/haolinyang/sports/Soccer-Backbone`（永久只读）
- 模型基线：high-resolution `unisoccer_part_temporal`
- checkpoint：真实 `epoch_19`，`ckpt_type="soccer_master"`
- Gate 运行默认使用绝对路径、offline、30 秒心跳和明确 timeout。

不要从本文复制历史命令盲目运行。GPU、评估或训练必须先按 `AGENTS.md` 重新检查资源并获得当次授权。

## 产物规则

- 原始日志、JSON 和图片：`reports/gN/`
- 一次运行需要多个文件时：`reports/gN/<run_id>/`
- checkpoint：`outputs/gN/<run_id>/`，Git 忽略
- 临时数据视图：`.runtime/data_views/gN/`，Git 忽略

历史证据一旦生成就不覆盖。retry 必须使用新的 `run_id`，并在状态账本中说明是成功、失败还是用户中止。

整理前生成的日志和 JSON 可能在内部记录当时的旧绝对路径；这些证据保持原字节不改写，当前归档位置以 `REPRODUCTION_STATUS.md` 和本索引为准。
