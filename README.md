# SoccerMaster

SoccerMaster 是一个面向足球视频理解的多任务模型复现与改进工作区。当前基线来自服务器上真实运行过的 Soccer-Backbone 代码、high-resolution 配置、SigLIP2 初始权重和 `epoch_19` checkpoint。

本仓库是个人复现工作区，不是上游官方发布，也不宣称已复现论文的完整指标或完整训练结果。

## 当前进度

| Gate | 状态 | 已证明的范围 |
| --- | --- | --- |
| G0 资产定位 | 基本完成 | 真实代码、目标配置、SigLIP2 和 `epoch_19` 已定位 |
| G1 环境 | 通过 | 共享参考环境和本地高速环境的核心依赖可用 |
| G2 checkpoint 加载 | 通过 | backbone、text model 和五个任务头完整加载，keys 全部匹配 |
| G3 随机张量 forward | 通过 | 两个 dataset 分支和五个任务头完成 float32 forward |
| G4 单个真实视频 | 通过 | 固定 SoccerReplay 视频完成解码、预处理和 Caption 推理 |
| G5 固定小规模评估 | 通过 | 五类任务指标结构完整，两遍评估重复性通过 |
| G6 tiny overfit | 通过 | 固定真实样本的 loss、梯度和 optimizer 最小链路通过 |
| G7 单任务训练 | 通过 | retry2 完整运行 2 epochs/184 optimizer steps，固定 train/valid、scheduler、事务式 checkpoint 和 exact-resume 断言全部通过 |
| G8–G10 | 未开始 | 小规模多任务、完整训练和 SoccerFactory 均未验证 |

详细证据、限制和唯一当前状态以 [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md) 为准。

## 固定基线

- 工作区：`/home/tianlin/SoccerMaster`
- 原始只读资产：`/remote-home/haolinyang/sports/Soccer-Backbone`
- 当前 Gate Python：`/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python`
- 共享参考 Python：`/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python`
- 目标配置：`configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml`
- checkpoint：原始只读目录中 high-resolution 实验的 `epoch_19`
- checkpoint 类型：`soccer_master`
- 输入基线：30 帧、512×512、float32

本地环境用于减少 GPFS 上的 Python 导入延迟，但模型、checkpoint 和数据仍从原始目录只读访问。

## 仓库导航

```text
models/                 上游模型和任务头，保持原布局
data/                   上游 Dataset、transform 和 collate
configs/                上游配置；未来本地配置按复现/改进分类
scripts/                上游原有工具脚本
reproduction/           Gate 入口、固定 manifest 和复现导航
experiments/            后续改进与消融实验登记
docs/                   Harness 和未验证研究假设
reports/gN/             按 Gate 归类的日志、JSON 和图片
outputs/gN/             本地 checkpoint，Git 忽略
.runtime/data_views/gN/ 可重建的临时数据视图，Git 忽略
.local_envs/            实际本地 Python 环境，Git 忽略
```

上游根目录的 `train.py`、`eval.py`、`eval_loss.py`、`vis_detr.py` 和若干 shell 脚本仍保持原位置。部分入口已确认存在参数或路径假设问题，不能仅根据文件名直接运行。

## 开始复现

1. 阅读 [`AGENTS.md`](AGENTS.md)，确认安全边界和授权条件。
2. 完整阅读 [`docs/HARNESS.md`](docs/HARNESS.md)，确定目标 Gate 的证明范围。
3. 阅读 [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md)，确认当前进度。
4. 从 [`reproduction/README.md`](reproduction/README.md) 定位对应 Gate 的脚本、manifest 和证据。
5. 涉及 GPU、评估或训练时，先检查最新资源并获得当次明确授权。

## 开始改进

先从 [`docs/future_improvements/`](docs/future_improvements/) 选择一个研究假设，再在 [`experiments/`](experiments/) 中建立独立实验。每个实验必须记录基线 commit、checkpoint、配置差异、数据、命令、结果和未验证范围。

复现 Gate 不得为了成功而静默加入研究改进；改进实验也不得反向改写已确认的基线结论。

## Git 与资产边界

Git 可以保存源码、配置、小型 manifest、文档和不含秘密的结构化报告。数据集、视频、权重、checkpoint、实际 Python 环境、临时数据视图和原始日志不应提交。

当前根目录未发现明确 LICENSE 文件。在许可和第三方组件分发边界确认前，仓库应保持私有，不对外重新发布数据、权重或第三方材料。
