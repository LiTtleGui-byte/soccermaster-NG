# SoccerMaster Harness 协议

本文档规定 SoccerMaster 复现过程中如何执行、观察和判定各个 Gate。它描述验证协议，不替代 `AGENTS.md` 的安全与授权规则，也不宣称任何尚未执行的 Gate 已经通过。

## 1. 目标与非目标

Harness 的目标是让每个结论都满足以下条件：

- 可复现：另一位执行者可以用记录的输入和命令重新运行。
- 可判定：成功和失败由断言与退出码决定，而不是依赖阅读者猜测日志。
- 可定位：失败能够定位到资产、环境、数据、模型、任务、rank 或执行阶段。
- 不改变语义：验证代码不会为了成功而静默改变目标实验。
- 有边界：一个 Gate 只证明其成功条件明确覆盖的内容。

Harness 不负责证明论文结论、替代完整评估，或自动推进后续 Gate。

本文档中的规则按状态标签解释：

- **现行规则**：从现在开始执行相关工作时必须遵守。
- **目标协议**：期望代码和测试最终满足，但不能据此宣称当前已经实现。
- **兼容例外**：用于无法直接满足新协议的历史资产，必须显式记录验证范围。
- **当前行为**：只能由代码与运行证据确认，并应记录在 `REPRODUCTION_STATUS.md`；本文档不以规范性文字代替事实证据。

## 2. 文档与证据层级

```text
AGENTS.md                         长期规则与授权边界
docs/HARNESS.md                   执行协议与目标不变量
REPRODUCTION_STATUS.md            当前状态和结论
docs/future_improvements/         研究假设与候选方案
reports/                          原始日志和机器可读结果
```

发生冲突时，先服从 `AGENTS.md`。当前行为以代码和运行证据为准；future-improvements 文档中的内容默认不是已确认事实。

## 3. 系统模型

一次完整实验被视为以下有状态数据流：

```text
资产定位
→ 环境构建
→ 模型构建
→ checkpoint 恢复
→ 输入构造与数据增强
→ forward
→ loss 与 backward
→ optimizer 与 scheduler
→ evaluation
→ checkpoint 与结果落盘
```

Gate 是对其中一个有限边界的验证。前一 Gate 通过不自动证明后一 Gate，也不授权执行后一 Gate。

## 4. 术语

- **已确认**：有代码、日志、测试结果或资产清单直接支持。
- **推断**：根据代码路径或已有事实作出的判断，尚未用目标运行验证。
- **未知**：当前证据不足以判断。
- **目标不变量**：系统应满足、但仍需要测试证明的性质。
- **风险**：可能破坏正确性、可复现性、资源安全或结论范围的条件。
- **exact resume**：恢复后能够延续原训练状态，而不只是重新加载部分权重。
- **实验语义**：模型、任务、数据范围、采样策略、精度、设备、batch size、loss 权重和优化边界等会影响结论的设置。

## 5. Gate 执行模板

**状态：现行规则。**

每次执行前应先形成以下记录：

```text
Gate:
目标:
用户授权范围:
输入资产:
实际 Python:
代码版本与 dirty 状态:
解析后的配置:
前置条件:
执行命令:
目标不变量:
超时:
心跳间隔和内容:
成功条件:
失败条件:
预期输出:
明确不验证的内容:
```

执行后补充：

```text
开始与结束时间:
退出码:
墙钟时间:
峰值 CPU 内存:
GPU 与峰值显存（如适用）:
断言结果:
日志和报告路径:
已确认:
推断:
未知:
风险:
结论:
唯一建议的下一步:
```

## 6. 通过与失败标准

**状态：现行规则。**

一个 Gate 只有在以下条件全部满足时才能标记为通过：

1. 前置条件全部满足。
2. 命令与实际输入被完整记录。
3. 进程在规定 timeout 内结束并报告退出码 0。
4. 所有必需断言通过。
5. 没有发生未声明的 fallback 或实验语义改变。
6. 日志和输出产物存在，并能定位到本次运行。
7. 报告明确写出该 Gate 没有验证什么。

下列情况不能判定通过：

- 只有“没有报错”，没有验证输出。
- 捕获异常后继续执行并最终返回 0。
- 缺少资产时静默跳过样本、组件或任务头。
- 实际使用了与记录不同的环境、配置或 checkpoint。
- 长时间命令没有明确退出码，或者被外部中断但原因未知。
- 不同 rank 进入不同控制流，而日志不足以证明 collective 顺序一致。

## 7. 实验身份与运行清单

**状态：现行规则。**

每次运行至少记录：

- Git commit；若工作区 dirty，记录相关文件列表，不读取或覆盖无关改动。
- Python 可执行文件绝对路径。
- Python、PyTorch、CUDA、Transformers、Accelerate 和关键扩展版本。
- resolved config，不能只记录配置文件名。
- 数据、基础权重和 checkpoint 的逻辑名称与解析后绝对路径。
- 可行时记录资产大小、mtime 和哈希；大型资产是否计算完整哈希应事先评估成本。
- seed、world size、rank、设备、dtype 和确定性设置。
- 入口命令、timeout、心跳、日志和输出目录。

运行清单不得包含密码、token、私钥或其他秘密。

## 8. 数据契约

### 8.1 Batch 接口

**状态：目标协议。**

训练和验证入口应依赖有名称的字段，而不是依赖字典插入顺序。每个任务至少定义：

- 必需字段及可选字段；
- tensor shape、dtype 和 device；
- batch、frame、channel、高宽等维度的含义；
- 坐标系、归一化方式和 padding 约定；
- 空标注与无效样本的处理方式。

### 8.2 视频与几何增强

**状态：目标协议。**

需要作为目标不变量验证：

- 同一 clip 中需要时间一致的随机几何参数被共享；不同 clip 可以独立采样。
- crop、resize、flip、affine 和 perspective 后，图像与 bounding boxes、球场线、关键点、相机参数保持同步。
- 被裁剪、越界、面积为零或无效的标注使用显式规则处理。

在相应测试通过前，不得把上述性质写成已确认行为。验证应使用人工可计算的简单样本和明确数值断言。当前代码路径分析、最小测试建议和候选产物见[数据与训练正确性审计](future_improvements/data_training_correctness.md)。

### 8.3 资产失败策略

**状态：现行规则；现有代码是否完全满足仍需验证。**

缺失、损坏或无法解析的必需资产应尽早失败，报告逻辑名称和精确路径。除非实验协议显式定义，否则不得在 DataLoader worker 中静默丢弃样本。

## 9. 随机性与恢复契约

**状态：目标协议。**

初始复现至少记录并设置：

- Python `random`；
- NumPy RNG；
- PyTorch CPU RNG；
- 每个 CUDA device 的 RNG；
- DataLoader generator 与 worker seed 策略；
- distributed sampler 的 epoch 和 rank 相关状态。

exact resume 还需要保存和恢复：

- 模型；
- optimizer；
- scheduler；
- AMP scaler（如使用）；
- epoch、global step、micro-step 或梯度累积位置；
- 所有必要 RNG 状态；
- sampler、任务调度器及其他会改变下一批输入的状态；
- 当前配置和依赖的基础权重标识。

exact resume 的最小验证不是“能够继续运行”，而是从同一保存点恢复后，下一批输入、loss 和参数更新满足事先定义的等价条件。

资产边界、保存范围和轻量 checkpoint 候选策略见[可复现环境、资产与 checkpoint 管理](future_improvements/reproducibility_and_storage.md)。该文档中的候选方案不是当前实现证据。

## 10. 多任务调度契约

**状态：目标协议。**

多任务训练必须显式记录：

- 一个 optimizer step 包含哪些任务和数据集；
- 每个任务每步使用的 batch 数与有效样本数；
- DataLoader 耗尽后的行为：停止、循环、重新采样或跳过；
- 任务顺序是否固定，以及不同 rank 是否一致；
- raw loss、loss weight、加权 loss、归一化规则和优化边界；
- 各任务累计采样次数。

具体观测项、特征分配消融和动态平衡候选方案见[多任务训练冲突与任务特征分配](future_improvements/multitask_training.md)。在数据、优化器和单任务基线可信前，这些候选方案不得被当作基线要求。

## 11. 分布式与并发契约

**状态：目标协议；collective 顺序和异常处理原则为现行安全要求。**

- 所有 rank 必须以兼容的顺序进入 barrier、gather、all-reduce 和其他 collective。
- 遇到空 batch、坏样本、OOM 或任务级异常时，不允许只有部分 rank 静默跳过当前阶段。
- 分布式心跳至少包含时间、rank、epoch、global step、dataset/task 和当前 phase。
- phase 名称建议使用 `data_wait`、`decode_augment`、`h2d`、`forward_backbone`、`forward_head`、`loss`、`backward`、`all_reduce`、`optimizer`、`evaluate`、`checkpoint`。
- 发生 hang 或 timeout 时，先定位各 rank 最后完成的 phase 和 collective；不得仅通过增加 timeout 将问题标记为解决。

## 12. 可观测性与性能测量

**状态：现行规则，适用于形成性能结论的运行。**

性能结论应来自分阶段测量，至少区分：

```text
共享存储读取
→ 视频解码与增强
→ DataLoader 等待
→ Host-to-Device
→ shared backbone
→ task heads
→ loss/backward
→ distributed communication
→ optimizer
→ evaluation/checkpoint I/O
```

记录吞吐量时同时记录 batch size、帧数、分辨率、任务组合、dtype、worker 数、prefetch 设置和设备。单独的 GPU 利用率不足以定位瓶颈。

## 13. Checkpoint 完整性契约

**状态：目标协议。**

新实现的 checkpoint 保存应采用事务式流程：

```text
写入同一目标文件系统中的临时目录
→ 保存所有组件
→ 生成 manifest
→ 校验必需文件、大小和可读取性
→ 最后写入 COMPLETE 标记
→ 原子 rename 为正式目录
```

manifest 至少描述 checkpoint 版本、模型组件、训练状态、配置、基础权重标识和必要文件列表。只有校验通过的 checkpoint 才能自动恢复。

**兼容例外：**现有历史 checkpoint 可以通过显式的 legacy compatibility 检查使用，但报告必须说明它没有遵循新协议，以及本次实际检查了哪些完整性条件。

只保存模型权重、部分任务头或 adapter 是允许的，但文件名、manifest 和报告必须准确表达恢复范围，不得标记为完整训练状态。

## 14. 失败分类

**状态：现行规则。**

失败报告至少选择一个主要类别：

- `asset_missing_or_corrupt`
- `environment_or_abi`
- `configuration`
- `data_contract`
- `model_construction`
- `checkpoint_load`
- `forward_or_numerical`
- `backward_or_optimizer`
- `distributed_order_or_timeout`
- `resource_exhaustion`
- `evaluation_contract`
- `artifact_write_or_integrity`
- `external_interruption`
- `unknown`

使用 `unknown` 时必须保留已有日志，并说明下一项能够缩小范围的只读或最小验证动作。

## 15. Gate 最小证明范围

**状态：现行规则。具体命令、输入和断言在对应 Gate 获得授权时确定。**

- **G0**：证明目标代码、配置、数据和权重已定位且访问边界明确。
- **G1**：证明目标 Python 和必要依赖满足既定导入/版本检查；不证明模型可运行。
- **G2**：证明完整 checkpoint 按目标策略加载，组件和 keys 检查通过；不证明 forward 正确。
- **G3**：证明规定形状的随机张量能够完成目标 forward，输出结构、shape 和有限性断言通过；不证明真实数据语义。
- **G4**：证明一个固定真实视频能通过完整输入和推理路径，并保存可检查结果；不证明总体指标。
- **G5**：证明固定小规模样本清单上的评估流程可重复，并产生结构完整的指标与预测。
- **G6**：证明一个固定微型数据集可以按预期过拟合，用于检查标签、梯度和优化器链路。
- **G7**：证明单任务训练协议、恢复和评估可信。
- **G8**：证明小规模多任务调度、梯度、分布式行为和指标记录可信。
- **G9**：在前述证据成立后执行完整训练，并记录完整资源与恢复策略。
- **G10**：在主分支基线稳定后，独立验证 SoccerFactory 分支，不反向污染基线结论。

每个 Gate 完成后更新 `REPRODUCTION_STATUS.md` 并停止。任何下一 Gate 都需要新的用户指示以及 `AGENTS.md` 要求的资源授权。
