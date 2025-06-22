# DETR可视化脚本使用说明

## 概述
`vis_detr.py` 是一个用于可视化DETR模型在SoccerNetGSR_Detection数据集上预测效果的脚本。它可以从训练集和测试集中抽样图像，并生成并排的可视化结果，左边显示模型预测，右边显示真实标注。

## 功能特性
- 从train和test数据集中随机抽样
- 并排可视化：左侧为预测结果，右侧为真实标注
- 支持多种目标检测类别（球员、守门员、裁判、球等）
- 可配置的分数阈值过滤
- 自动保存高质量的可视化图像

## 使用方法

### 基本用法
```bash
python vis_detr.py \
    --config configs/default.yaml \
    --checkpoint outputs/debug_video/epoch_9 \
    --output_dir vis_results \
    --num_samples 10 \
    --score_threshold 0.3
```

### 使用提供的脚本
```bash
# 修改 run_vis_example.sh 中的路径
chmod +x run_vis_example.sh
./run_vis_example.sh
```

### 参数说明
- `--config`: 配置文件路径（必需）
- `--checkpoint`: checkpoint文件或目录路径（必需）
- `--output_dir`: 输出目录路径（可选，默认自动生成）
- `--num_samples`: 每个数据集的抽样数量（默认：10）
- `--score_threshold`: 预测分数阈值，过滤低分预测（默认：0.3）
- `--super_config`: Super config文件路径（可选）

## 输出结果

### 文件结构
```
vis_results/
├── train_sample_000_batch_5_idx_2.png
├── train_sample_001_batch_12_idx_0.png
├── ...
├── test_sample_000_batch_3_idx_1.png
├── test_sample_001_batch_8_idx_4.png
└── ...
```

### 可视化说明
每张输出图像包含：
- **左侧**：模型预测结果
  - 红色边界框表示预测的目标
  - 标签显示类别名称和置信度分数
  - 只显示高于设定阈值的预测

- **右侧**：真实标注
  - 绿色边界框表示真实目标
  - 标签显示真实类别名称

### 目标类别
根据SoccerNetGSR数据集的role_mapping：
- 0: ball (球)
- 1: goalkeeper (守门员)
- 2: other (其他)
- 3: player (球员)
- 4: referee (裁判)

## 配置要求

### 环境依赖
- torch
- matplotlib
- numpy
- PIL (Pillow)
- accelerate

### 数据结构
脚本期望以下数据结构：
- 模型输出包含：pred_logits, pred_boxes
- 真实标注包含：boxes, labels
- 图像格式：RGB, 已归一化

### Checkpoint格式
支持两种checkpoint格式：
1. **目录结构**（推荐）：
   ```
   epoch_9/
   ├── backbone/           # SigLIP backbone
   ├── SoccerNetGSR_Detection.pt  # Detection head
   └── ...
   ```

2. **单文件**：包含完整模型状态字典的.pt文件

## 高级配置

### 修改类别名称
如果需要修改类别名称，编辑 `vis_detr.py` 中的 `class_names` 字典：
```python
class_names = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}
```

### 调整可视化参数
在脚本中可以修改：
- 图像大小：`figsize=(20, 10)`
- 边界框颜色：`edgecolor='red'` (预测), `edgecolor='green'` (真实)
- 线条粗细：`linewidth=2`
- 字体大小：`fontsize=10`

### 处理大数据集
对于大型数据集，建议：
- 减少 `num_samples` 数量
- 增加 `score_threshold` 过滤噪声预测
- 使用更高的GPU内存

## 故障排除

### 常见问题
1. **CUDA内存不足**：
   - 减少batch size
   - 减少样本数量
   - 使用更小的图像尺寸

2. **Checkpoint加载失败**：
   - 检查checkpoint路径是否正确
   - 确认模型配置与checkpoint匹配

3. **可视化结果为空**：
   - 降低score_threshold
   - 检查数据集是否正确加载
   - 确认模型已正确训练

### 调试技巧
- 启用详细日志：在脚本中添加更多print语句
- 检查中间结果：保存模型输出到文件
- 验证数据格式：确认boxes坐标格式（相对/绝对坐标）

## 示例输出

运行成功后，你将看到类似以下的输出：
```
正在加载checkpoint: outputs/debug_video/epoch_9
加载backbone: outputs/debug_video/epoch_9/backbone
加载SoccerNetGSR_Detection头: outputs/debug_video/epoch_9/SoccerNetGSR_Detection.pt
Checkpoint加载成功!
可视化结果将保存到: vis_results
可用的训练任务: ['SoccerNetGSR_Detection']
可用的测试任务: ['SoccerNetGSR_Detection']
开始从数据集中抽样...
从训练集抽样 10 个样本...
保存 train 样本 1/10: vis_results/train_sample_000_batch_5_idx_2.png
  预测框数量: 15 (阈值>0.3: 8)
  真实框数量: 12
...
可视化完成! 结果保存在: vis_results
```

每个样本还会显示统计信息，帮助你了解模型性能。 