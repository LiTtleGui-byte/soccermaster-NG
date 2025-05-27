# TensorBoard Integration for Soccer-Backbone

本项目已经完全集成了TensorBoard来记录训练过程中的loss和其他重要指标。

## 功能特性

### 自动记录的指标
1. **Loss指标**
   - 总体loss (train/total_loss)
   - 各个任务的具体loss (train/{task}_{loss_type})
   - Epoch级别的平均loss (epoch/{loss_type})

2. **训练指标**
   - 学习率 (lr/group_0)
   - 梯度范数 (train/grad_norm)
   - 训练进度信息

3. **模型参数统计** (可选)
   - 参数统计 (params/{param_name}_mean/std/norm)
   - 梯度统计 (grads/{param_name}_mean/std/norm)

## 配置选项

在`configs/default.yaml`中可以配置以下选项：

```yaml
# TensorBoard设置
USE_TENSORBOARD: True        # 是否启用TensorBoard日志
TENSORBOARD_LOG_DIR:         # 日志目录（如果为空，将使用OUTPUTS_DIR/logs）
TENSORBOARD_FLUSH_SECS: 30   # 多久刷新一次日志
LOG_PARAMS_GRADS: False      # 是否记录参数和梯度统计信息
LOGGING_INTERVAL: 20         # 每多少个iteration记录一次日志
```

## 使用方法

### 1. 训练时自动记录
只需正常运行训练脚本，TensorBoard日志会自动记录：

```bash
python train.py --config_path configs/default.yaml --exp_name my_experiment
```

### 2. 查看TensorBoard
在另一个终端中启动TensorBoard：

```bash
tensorboard --logdir ./outputs/logs
```

然后在浏览器中打开 `http://localhost:6006` 查看训练日志。

### 3. 自定义日志目录
如果想指定特定的日志目录：

```yaml
TENSORBOARD_LOG_DIR: /path/to/your/logs
```

## 日志结构

TensorBoard日志将保存在以下结构中：

```
outputs/
├── logs/
│   └── [timestamp]/
│       ├── events.out.tfevents.*
│       └── ...
└── checkpoints/
    └── ...
```

## 监控的关键指标

### 训练过程监控
- `train/total_loss`: 总体训练损失
- `train/grad_norm`: 梯度范数（用于监控梯度爆炸/消失）
- `lr/group_0`: 学习率变化

### 任务特定监控
- `train/SoccerNetGSR_*`: SoccerNetGSR任务的各种损失

### Epoch级别监控
- `epoch/*`: 每个epoch的平均指标

## 故障排除

### 常见问题
1. **TensorBoard无法启动**
   - 确保已安装TensorBoard: `pip install tensorboard`
   - 检查日志目录是否存在且有读写权限

2. **日志为空**
   - 检查配置文件中`USE_TENSORBOARD`是否设为`True`
   - 确保训练正常进行且没有提前退出

3. **性能影响**
   - 如果训练速度受影响，可以增大`TENSORBOARD_FLUSH_SECS`
   - 将`LOG_PARAMS_GRADS`设为`False`可以减少日志量

## 扩展功能

### 添加自定义指标
可以在训练代码中添加自定义指标记录：

```python
# 在train_one_epoch函数中
if logger:
    logger.log_scalar("custom/my_metric", my_value, step)
    logger.log_histogram("custom/my_histogram", my_tensor, step)
```

### 记录图像
可以记录训练过程中的图像：

```python
if logger:
    logger.log_image("train/sample_image", image_tensor, step)
```

这个TensorBoard集成提供了完整的训练过程可视化，帮助更好地理解和调试模型训练。 