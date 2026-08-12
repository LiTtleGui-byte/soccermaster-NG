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

G0/G1 主要是资产和环境盘点，当前没有独立 Python Gate 入口。G8–G10 尚未建立入口；不为尚未设计的 Gate 预先创建空脚本。

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
