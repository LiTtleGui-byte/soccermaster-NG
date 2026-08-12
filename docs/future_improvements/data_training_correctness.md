# 数据与训练正确性审计

## 状态

- 当前状态：`待实现`
- 优先级：`P0`
- 目标：在改变模型结构前，确认基线的数据、标签、梯度和优化器行为可信

## 为什么优先

如果输入、标签、mask 或优化器存在问题，模型结构对照实验的结果将无法可靠解释。这里的工作以 CPU 静态检查和小规模单元测试为主，不需要先启动完整训练。

## 待核对事项

### 1. 视频增强是否在时间上保持一致

当前检测数据在视频模式下逐帧调用 transform，同时 `ClearAugmentationMetas` 会清除已经保存的随机增强参数。按代码路径推断，同一段视频中的帧可能分别采样仿射、透视、裁剪、翻转和颜色参数。

最小测试：

1. 构造 30 张完全相同的测试帧。
2. 使用固定随机种子执行一次训练 transform。
3. 检查 30 帧的几何变换参数是否完全一致。
4. 分别验证检测数据和 VideoCaption 数据路径。

通过条件：同一 clip 内共享随机几何参数，不同 clip 之间仍然独立采样。

### 2. 几何增强与标注是否同步

需要逐项验证：

- bounding boxes 在 crop、resize、flip、affine、perspective 后的位置。
- 球场线和关键点坐标及热图。
- 相机内参、旋转、平移和视场角。
- 被裁掉或只剩一部分的目标如何过滤。

最小测试应使用人工可计算的简单图像，例如一个矩形框和几条已知坐标的直线，并检查变换后的数值。

### 3. 解说 tokenizer、labels 与 attention mask

需要确认：

- tokenizer 是否自动添加 `<|begin_of_text|>`。
- collater 是否再次手动添加 BOS，造成重复。
- `<|end_of_text|>` 是否被 attention mask 错误屏蔽。
- padding 位置是否在 labels 中正确设为 `-100`。
- `[PLAYER]`、`[TEAM]`、`[COACH]` 和 `[REFEREE]` 是否各自成为单一 token。
- 新增特殊 token 的 embedding 是否真的参与训练并进入优化器。

最小输出应同时打印：原始文本、token、token ID、attention mask、labels 和每个特殊 token 的计数。

### 4. 优化器参数覆盖与学习率

为每个参数记录：

```text
参数名
requires_grad
所在优化器参数组
学习率
weight decay
```

检查目标：

- 所有可训练参数恰好进入一个参数组。
- 冻结参数不进入优化器。
- 没有参数被重复加入。
- 时间位置编码、视觉 backbone、Q-Former、投影层和 LoRA 的学习率符合配置。
- 配置中的学习率不会被优化器代码中的常量静默覆盖。

### 5. 数据资产完整性

为每个 split 统计：

- 样本数、缺失文件数和损坏文件数。
- 视频时长、分辨率、帧率和长宽比分布。
- 文本为空或事件标签缺失的数量。
- 检测框为空、越界或面积为零的数量。
- train、valid、test 之间可能重复的视频或片段。
- 软链接目标是否存在。

资产缺失应在启动训练前失败，并给出明确路径，不能在 DataLoader worker 中静默跳过。

### 6. 验证与随机性

检查：

- validation/test DataLoader 是否应当 `shuffle=False`。
- Python、NumPy、PyTorch 和 worker 的随机种子是否统一。
- 分布式采样是否遗漏或重复样本。
- 同一 checkpoint、同一输入和确定性解码能否产生相同结果。

## 建议产物

```text
tests/data/
tests/tokenization/
tests/optimizers/
scripts/audit_assets.py
reports/audits/
```

具体目录应在后续代码结构确定后再创建；此文档不代表已经建立这些文件。

## 完成标准

- [ ] 所有最小测试可以在 CPU 上运行。
- [ ] 每项检查都有明确的 pass/fail，而不是只打印日志。
- [ ] 已确认的问题先形成基线兼容修复或对照实验。
- [ ] 审计报告记录代码提交、配置、环境和数据版本。
