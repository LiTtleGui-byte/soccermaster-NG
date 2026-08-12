# SoccerMaster

SoccerMaster 是一个面向足球视频理解的多任务视觉模型复现工作区。本仓库保留上游 [Soccer-Backbone](https://github.com/haolinyang-hlyang/Soccer-Backbone) 的 Git 历史，并在此基础上逐步建立可审计、可恢复和可重复的复现流程。

本仓库当前为个人私有复现仓库，不是上游项目的官方发布，也不宣称已经复现论文的完整训练结果。

## 当前状态

复现采用逐级 Gate：每个 Gate 只验证一个有限问题，完成后停止，不自动进入下一阶段。

- G0：资产定位基本完成。
- G1：Python、依赖与 CUDA 环境检查通过。
- G2：目标 checkpoint 的七个组件完成 CPU-only 加载，missing/unexpected keys 均为空。
- G3：随机张量 forward 尚未开始。
- G4–G10：尚未开始。

当前结论、证据范围、未知项和下一步以 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) 为准。Gate 的判定协议见 [docs/HARNESS.md](docs/HARNESS.md)。

## 仓库结构

```text
configs/                    模型、数据与实验配置
data/                       Dataset、transform 和 collate 源码；不包含数据集资产
models/                     共享视觉 backbone、任务头和原生扩展源码
scripts/                    复现、分析和可视化脚本
sn_calibration/             球场标定相关代码
utils/                      日志、分布式和训练辅助代码
docs/                       Harness、研究假设与复现说明
reports/                    可提交的结构化证据；原始 .log 文件保持本地
```

根目录暂时保留上游训练、评估和可视化入口。它们会在建立兼容测试后再分类整理，避免仅为目录整洁而破坏现有 import 和运行方式。

## 开始工作前

1. 阅读 [AGENTS.md](AGENTS.md)，确认路径、安全、授权和报告规则。
2. 阅读 [docs/HARNESS.md](docs/HARNESS.md)，确认目标 Gate 的输入、断言和证明范围。
3. 阅读 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)，确认当前已经验证到哪里。
4. 使用状态文档中已确认的环境和资产；不要从 README 推断某个候选环境已经等价可用。
5. 每次只推进一个 Gate，并保留命令、退出码、断言和日志。

当前没有提供“一键训练”命令。训练链路尚未通过 G3–G8 的逐层验证，未经明确授权也不会启动训练或 GPU 操作。

## 版本与资产边界

Git 只保存：

- 源码、配置和小型 manifest；
- 文档、测试和复现脚本；
- 不包含秘密或受限资产的结构化报告。

以下内容不进入 Git：

- 数据集、视频和派生特征；
- pretrained weights、checkpoint 和模型导出；
- Conda/venv、本地依赖 staging 和编译缓存；
- 原始运行日志及实验输出；
- 未确认许可允许再分发的论文材料或第三方资产。

具体忽略边界见 [.gitignore](.gitignore)，资产与 checkpoint 的目标管理方案见 [docs/future_improvements/reproducibility_and_storage.md](docs/future_improvements/reproducibility_and_storage.md)。

## Git 远端约定

```text
origin    个人私有 SoccerMaster 仓库
upstream  原作者 Soccer-Backbone 仓库
```

上游同步与本地复现提交应保持可区分。现有上游历史不改写；新的复现工作使用小而完整、能够说明证据边界的提交。

## 文档导航

完整索引见 [docs/README.md](docs/README.md)。主要入口：

- [长期工作规则](AGENTS.md)
- [Harness 协议](docs/HARNESS.md)
- [复现状态](REPRODUCTION_STATUS.md)
- [未来改进方向](docs/future_improvements/README.md)
- [历史恢复训练说明](RESUME_TRAINING.md)

## 来源与许可说明

本仓库基于上游 Soccer-Backbone 的代码历史开展复现。当前快照未在仓库根目录发现明确的 LICENSE 文件；在许可和第三方组件分发边界得到确认前，本仓库保持私有，不对外重新发布数据、权重、论文材料或未确认授权的第三方源码。
