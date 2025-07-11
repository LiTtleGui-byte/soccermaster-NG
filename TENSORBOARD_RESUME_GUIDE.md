# TensorBoard Resume 使用指南

## 快速开始

### 1. 正常启动训练
```bash
python train.py --config-path configs/your_config.yaml
```

训练会在 `outputs/your_experiment/logs` 目录下创建TensorBoard日志文件。

### 2. 查看训练进度
在另一个终端中启动TensorBoard：
```bash
tensorboard --logdir ./outputs/your_experiment/logs
```

然后在浏览器中访问 `http://localhost:6006`

### 3. 训练中断后恢复
修改配置文件，添加resume设置：
```yaml
RESUME_TRAINING: True
```

重新启动训练：
```bash
python train.py --config-path configs/your_config.yaml
```

### 4. 查看resume后的日志
刷新TensorBoard页面，你会看到：
- 训练曲线连续显示
- 在resume点有清晰的标记
- TEXT标签页中有resume信息

## TensorBoard中的Resume信息

### SCALARS标签页
- **training/resume_marker**: 显示resume点的标记（值为1.0）
- **train/loss**: 连续的loss曲线，resume点清晰可见
- **train/learning_rate**: 学习率曲线，包含resume点
- **所有其他metrics**: 都会连续显示

### TEXT标签页
- **training/resume_info**: 显示resume的epoch和global_step
- **training/resume_timestamp**: 显示resume的具体时间

### 示例视图
```
Loss曲线示例：
    1.0 |     \
        |      \
        |       \___
    0.5 |           \___
        |               \___
        |                   \___[Resume点]
    0.0 |________________________\___
        0    100   200   300   400   500
                    Global Step
```

## 测试功能

### 运行测试脚本
```bash
# 测试TensorBoard连续性
python test_tensorboard_resume.py --keep-files

# 查看测试结果
tensorboard --logdir ./test_tensorboard_output/logs
```

### 验证项目
1. ✅ 多个事件文件被创建
2. ✅ 训练曲线连续显示
3. ✅ Resume点清晰标记
4. ✅ 文本日志包含resume信息
5. ✅ 时间戳正确记录

## 常见问题

### Q: Resume后看不到之前的训练曲线？
A: 确保使用相同的OUTPUTS_DIR和EXP_NAME，TensorBoard会自动合并相同目录的日志。

### Q: 曲线在resume点有断裂？
A: 检查global_step是否正确恢复，应该从中断点继续而不是重新开始。

### Q: 如何区分不同的训练会话？
A: 查看TEXT标签页中的resume_info和resume_timestamp，可以清楚看到每次resume的信息。

### Q: TensorBoard显示多个runs？
A: 这是正常的，TensorBoard会为每个训练会话创建单独的事件文件，但它们属于同一个experiment。

## 最佳实践

1. **定期保存**: 设置合适的 `SAVE_CHECKPOINT_PER_EPOCH` 值
2. **备份日志**: 在重要的训练节点备份TensorBoard日志
3. **命名规范**: 使用有意义的 `EXP_NAME` 便于管理
4. **监控资源**: 长时间训练时注意磁盘空间和TensorBoard日志大小

## 配置参考

```yaml
# 基本设置
USE_TENSORBOARD: True
TENSORBOARD_FLUSH_SECS: 30
RESUME_TENSORBOARD_CONTINUOUS: True

# Resume设置
RESUME_TRAINING: True
OUTPUTS_DIR: ./outputs/your_experiment
SAVE_CHECKPOINT_PER_EPOCH: 5
```

这样配置可以确保TensorBoard日志的连续性和resume功能的正常工作。 