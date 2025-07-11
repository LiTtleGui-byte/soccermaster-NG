# Resume Training 功能说明

本文档介绍如何使用训练中断后自动恢复训练的功能。

## 功能概述

Resume Training 功能允许你在训练中断后（如服务器重启、程序崩溃等）自动从最近的checkpoint继续训练，而不需要从头开始。该功能会保存和恢复完整的训练状态，包括：

- 模型权重（backbone + 各个head）
- 优化器状态（Adam动量等）
- 学习率调度器状态
- 训练进度（epoch数、global step）
- 随机数状态（保证训练的可重现性）
- **TensorBoard日志连续性**：自动在TensorBoard中标记resume点，确保日志连续记录

## 配置方法

### 方法1：自动恢复最新checkpoint

在配置文件中设置：

```yaml
RESUME_TRAINING: True
OUTPUTS_DIR: ./outputs/your_experiment_name
```

这样程序会自动在 `OUTPUTS_DIR` 中寻找最新的有效checkpoint并恢复训练。

### 方法2：从指定checkpoint恢复

在配置文件中设置：

```yaml
RESUME_FROM_CHECKPOINT_DIR: ./outputs/your_experiment_name/epoch_10
```

这样程序会从指定的checkpoint目录恢复训练。

### 方法3：通过命令行参数

```bash
# 自动恢复
python train.py --config-path configs/your_config.yaml --resume-training True

# 从指定checkpoint恢复
python train.py --config-path configs/your_config.yaml --resume-from-checkpoint-dir ./outputs/experiment/epoch_10
```

## Checkpoint结构

每个checkpoint目录包含以下文件：

```
epoch_X/
├── backbone/                    # SigLIP backbone权重
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── SoccerNetGSR_ReID.pt        # ReID head权重
├── SoccerNetGSR_Detection.pt   # Detection head权重
├── LinesDetection.pt           # Lines head权重
├── KeypointsDetection.pt       # Keypoints head权重
├── CameraRegression.pt         # Camera head权重
├── VideoCaption.pt             # Caption head权重（如果启用）
├── CaptionClassification.pt    # Caption Classification head权重（如果启用）
├── optimizer_state.pt          # 优化器状态
├── scheduler_state.pt          # 学习率调度器状态
└── training_state.json         # 训练进度和随机数状态
```

## 使用示例

### 示例1：正常训练并保存checkpoint

```yaml
# configs/my_experiment.yaml
SUPER_CONFIG_PATH: ./configs/default.yaml

EXP_NAME: my_experiment
EPOCHS: 30
SAVE_CHECKPOINT_PER_EPOCH: 5  # 每5个epoch保存一次

DATASETS_TO_HEADS:
  SoccerNetGSR_Detection: [SoccerNetGSR_Detection, LinesDetection, KeypointsDetection, CameraRegression]
```

运行训练：
```bash
python train.py --config-path configs/my_experiment.yaml
```

### 示例2：训练中断后恢复

如果训练在epoch 15中断了，修改配置文件：

```yaml
# configs/my_experiment.yaml
SUPER_CONFIG_PATH: ./configs/default.yaml

# 添加resume配置
RESUME_TRAINING: True

EXP_NAME: my_experiment
EPOCHS: 30
SAVE_CHECKPOINT_PER_EPOCH: 5
```

重新运行训练，程序会自动从epoch 10的checkpoint恢复（最新的完整checkpoint）：
```bash
python train.py --config-path configs/my_experiment.yaml
```

### 示例3：从特定checkpoint恢复

如果你想从特定的checkpoint恢复（比如epoch 5而不是最新的）：

```yaml
# configs/my_experiment.yaml
SUPER_CONFIG_PATH: ./configs/default.yaml

# 指定特定checkpoint
RESUME_FROM_CHECKPOINT_DIR: ./outputs/my_experiment/epoch_5

EXP_NAME: my_experiment
EPOCHS: 30
SAVE_CHECKPOINT_PER_EPOCH: 5
```

## TensorBoard日志连续性

### 自动连续记录
当你resume训练时，TensorBoard日志会自动保持连续性：

1. **相同日志目录**：resume训练会使用与原训练相同的日志目录（`outputs_dir/logs`）
2. **连续global_step**：恢复的global_step会从之前的训练继续，确保时间轴连续
3. **Resume标记**：在TensorBoard中自动添加resume标记，包括：
   - 文本日志：显示resume的epoch和global_step
   - Resume标记：在曲线图中标记resume点
   - 时间戳：记录resume的具体时间

### 在TensorBoard中查看Resume信息
1. 在**SCALARS**标签页中，可以看到：
   - `training/resume_marker`：显示resume点的标记
   - 所有metrics曲线都会连续显示，resume点清晰可见

2. 在**TEXT**标签页中，可以看到：
   - `training/resume_info`：显示resume的详细信息
   - `training/resume_timestamp`：显示resume的时间戳

### 测试TensorBoard连续性
可以使用提供的测试脚本验证TensorBoard日志连续性：

```bash
# 运行TensorBoard连续性测试
python test_tensorboard_resume.py --test-dir ./test_tensorboard_output --keep-files

# 查看测试结果
tensorboard --logdir ./test_tensorboard_output/logs
```

## 注意事项

1. **配置一致性**：恢复训练时，模型结构相关的配置（如网络架构、head设置等）必须与原训练保持一致。

2. **数据路径**：确保数据路径仍然有效，数据集没有发生变化。

3. **优先级**：`RESUME_FROM_CHECKPOINT_DIR` 的优先级高于 `RESUME_TRAINING`，如果两者都设置了，会优先使用指定的checkpoint目录。

4. **有效性检查**：程序会检查checkpoint的完整性，只有包含 `training_state.json` 的checkpoint才被视为有效。

5. **跨设备恢复**：checkpoint保存时使用CPU映射，因此可以在不同的GPU设备上恢复训练。

6. **TensorBoard连续性**：TensorBoard会自动在相同目录中创建新的事件文件，但会保持日志连续性。建议在resume前备份重要的TensorBoard日志。

## 故障排除

### 问题：找不到有效checkpoint
```
RESUME_TRAINING is True but no valid checkpoint found, starting from scratch
```

**解决方案**：
- 检查 `OUTPUTS_DIR` 路径是否正确
- 确认checkpoint目录中包含 `training_state.json` 文件
- 检查checkpoint是否被意外删除或损坏

### 问题：指定的checkpoint目录不存在
```
Specified checkpoint directory ./outputs/experiment/epoch_10 does not exist
```

**解决方案**：
- 检查路径是否正确
- 确认该epoch的checkpoint确实存在

### 问题：模型加载失败
```
RuntimeError: Error(s) in loading state_dict...
```

**解决方案**：
- 确认当前配置与checkpoint保存时的配置一致
- 检查模型结构是否发生了变化
- 如果模型结构有变化，可能需要从该checkpoint手动提取权重

## 最佳实践

1. **定期保存**：设置合理的 `SAVE_CHECKPOINT_PER_EPOCH` 值，既不会占用太多磁盘空间，也能在中断时不丢失太多训练进度。

2. **备份重要checkpoint**：对于长时间训练，建议定期备份关键checkpoint到其他位置。

3. **磁盘空间管理**：checkpoint会占用较多磁盘空间，可以定期清理旧的checkpoint（但要保留几个最新的）。

4. **日志记录**：程序会在日志中记录resume的详细信息，有助于调试和确认恢复状态。 