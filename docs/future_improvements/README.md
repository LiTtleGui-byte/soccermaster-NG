# 后续改进方向

这个目录记录 SoccerMaster 基线复现完成后值得验证的改进方向。

这里的内容是研究候选，而不是已经确定的实现方案。每项改进都应先保留原始基线，明确控制变量，并通过小规模实验验证后再进入完整训练。

## 状态说明

- `待研究`：只有问题和初步想法，尚未确认实现方案。
- `待实现`：方案已经明确，但还没有修改代码。
- `验证中`：已有实现，正在进行小规模实验。
- `已验证`：实验结果支持该改进，且复现记录完整。
- `不采用`：实验或分析表明暂不值得继续。

## 改进列表

| 优先级 | 方向 | 状态 | 进入条件 | 文档 |
|---:|---|---|---|---|
| P0 | 数据与训练正确性审计 | 待实现 | 立即；先于模型改进 | [data_training_correctness.md](data_training_correctness.md) |
| P0 | 可复现环境、资产与 checkpoint 管理 | 待研究 | 基线运行脚本整理时 | [reproducibility_and_storage.md](reproducibility_and_storage.md) |
| P1 | 保持长宽比的多分辨率视觉输入 | 待研究 | 完成 `512×512` 论文基线复现 | [native_aspect_ratio.md](native_aspect_ratio.md) |
| P1 | 解说生成的视觉事实约束 | 待研究 | 完成解说基线复现 | [commentary_grounding.md](commentary_grounding.md) |
| P1 | 检测、足球小目标与跨帧跟踪 | 待研究 | 完成人员检测基线复现 | [detection_and_tracking.md](detection_and_tracking.md) |
| P2 | 多任务训练冲突与任务特征分配 | 待研究 | 各单任务基线可信后 | [multitask_training.md](multitask_training.md) |
| P0 | 统一评估与错误分析协议 | 待研究 | 与每项基线同步建立 | [evaluation_protocol.md](evaluation_protocol.md) |

## 推荐推进顺序

```text
正确性审计与可复现环境
→ 单任务论文基线
→ 输入与任务头改进
→ 多任务联合优化
→ 完整规模实验
```

## 记录原则

每个方向至少应记录：当前行为、问题假设、候选方案、影响范围、最小实验、评价指标、风险和未决问题。

同一项改进在没有通过单元测试、单样本 forward 和 tiny overfit 前，不直接进入完整训练。
