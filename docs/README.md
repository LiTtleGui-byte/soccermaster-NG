# SoccerMaster 文档索引

这里保存复现协议、研究假设和项目说明。运行产生的原始证据位于 `reports/`；当前进度的唯一状态账本是仓库根目录的 `REPRODUCTION_STATUS.md`。

## 核心文档

| 文档 | 职责 |
|---|---|
| [README.md](../README.md) | 项目入口、总体进度、目录说明和工作导航 |
| [AGENTS.md](../AGENTS.md) | 长期规则、路径边界、授权条件和报告要求 |
| [HARNESS.md](HARNESS.md) | Gate 执行模板、断言标准和最小证明范围 |
| [REPRODUCTION_STATUS.md](../REPRODUCTION_STATUS.md) | 已确认事实、当前 Gate、未知项、风险和下一步 |
| [reproduction/README.md](../reproduction/README.md) | Gate 入口、manifest、固定基线和证据导航 |
| [experiments/README.md](../experiments/README.md) | 后续改进与消融实验登记模板 |
| [RESUME_TRAINING.md](../RESUME_TRAINING.md) | 上游已有的恢复训练说明；使用前仍需按 Harness 验证 |

## 研究候选

`future_improvements/` 中的内容是研究假设，不代表当前代码已经实现，也不代表问题已经被运行证实。

- [方向总览](future_improvements/README.md)
- [数据与训练正确性审计](future_improvements/data_training_correctness.md)
- [可复现环境、资产与 checkpoint 管理](future_improvements/reproducibility_and_storage.md)
- [统一评估与错误分析协议](future_improvements/evaluation_protocol.md)
- [多任务训练冲突与任务特征分配](future_improvements/multitask_training.md)
- [保持长宽比的多分辨率输入](future_improvements/native_aspect_ratio.md)
- [解说生成的视觉事实约束](future_improvements/commentary_grounding.md)
- [检测、小目标与跨帧跟踪](future_improvements/detection_and_tracking.md)

## 证据与产物

```text
reports/      可提交的小型结构化摘要和经过审计的证据
outputs/      本地 checkpoint 和大型运行产物；Git 永久忽略
.runtime/     可重建的临时数据视图和链接；Git 永久忽略
```

原始 `.log` 文件默认不提交。如果某段日志对结论不可替代，应提取不包含秘密和受限路径的结构化摘要，并记录生成命令、退出码和源日志位置。

## 阅读顺序

准备执行某个 Gate 时：

1. 从 `REPRODUCTION_STATUS.md` 确认当前状态。
2. 从 `AGENTS.md` 确认授权和安全边界。
3. 完整阅读 `HARNESS.md`，为目标 Gate 填写执行模板。
4. 运行后把事实写回状态文档，把原始证据放到合适的报告或本地产物目录。
5. 完成当前 Gate 后停止，等待下一步指示。
