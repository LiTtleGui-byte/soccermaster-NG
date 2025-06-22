# 模型评估脚本使用说明

## 概述

`eval.py` 脚本用于评估已训练好的模型checkpoint，计算各种损失和性能指标（如mAP、precision、recall等），并将结果保存到文件中。

## 功能特性

- ✅ 支持加载训练期间保存的checkpoint（目录结构或单文件）
- ✅ 计算所有任务的损失指标
- ✅ 计算detection任务的性能指标（mAP、precision、recall、F1等）
- ✅ 格式化输出易读的评估结果
- ✅ 自动保存结果到txt文件
- ✅ 支持多GPU评估

## 使用方法

### 命令行参数

```bash
python eval.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH [--log_dir LOG_DIR] [--super_config SUPER_CONFIG_PATH]
```

**必需参数：**
- `--config`: 配置文件路径（与训练时使用的配置文件相同）
- `--checkpoint`: checkpoint路径（可以是文件或目录）

**可选参数：**
- `--log_dir`: 日志目录路径（默认自动生成）
- `--super_config`: super config文件路径（如果与配置文件中指定的不同）

### 示例使用

#### 1. 基本使用

```bash
python eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/my_experiment/epoch_19 \
    --log_dir eval_logs
```

#### 2. 使用训练期间保存的checkpoint目录

```bash
python eval.py \
    --config configs/debug_dataset.yaml \
    --checkpoint outputs/exp_name/epoch_9
```

#### 3. 使用脚本运行

```bash
# 修改run_eval_example.sh中的路径
bash run_eval_example.sh
```

## Checkpoint格式

脚本支持两种checkpoint格式：

### 1. 目录结构（推荐，与train.py保存格式一致）

```
epoch_X/
├── backbone/          # SigLIP backbone权重
├── SoccerNetGSR_Detection.pt  # Detection head权重
├── SoccerNetGSR_ReID.pt       # ReID head权重（如果有）
└── ...
```

### 2. 单文件格式

```
model.pt  # 包含完整模型权重的单个文件
```

## 输出格式

### 控制台输出

```
================================================================================
EVALUATION RESULTS
================================================================================

OVERALL RESULTS:
  Overall Weighted Loss: 1.2345
  Overall Unweighted Loss: 2.3456
  Total Samples: 1000

SOCCERNETGSR_DETECTION RESULTS:
  Sample Count: 500
  Weighted Losses:
    loss_ce: 0.1234
    loss_bbox: 0.2345
    loss_giou: 0.3456
    Total Weighted Loss: 0.7035
  Evaluation Metrics:
    mAP: 0.4567
    mAP@0.5: 0.5678
    mAP@0.75: 0.3456
    precision: 0.6789
    recall: 0.5432
    f1: 0.6020
    AP@0.50: 0.5678
    AP@0.55: 0.5234
    ...
================================================================================
```

### 文件输出

结果会自动保存到指定的日志目录中，包含：
- `eval_results.txt`: 格式化的评估结果文本文件（包含时间戳和所有控制台输出）
- `eval_results.json`: 详细的JSON格式评估结果
- `log.txt`: 运行过程的日志记录
- `config.yaml`: 使用的配置文件备份

输出目录结构：
```
log_dir/
├── eval_results.txt    # 易读的文本结果
├── eval_results.json   # 详细的JSON结果
├── log.txt            # 运行日志
└── config.yaml        # 配置备份
```

## 评估指标说明

### Detection任务指标

- **mAP**: 所有IoU阈值(0.5-0.95)的平均精度
- **mAP@0.5**: IoU阈值0.5时的平均精度
- **mAP@0.75**: IoU阈值0.75时的平均精度
- **precision**: 整体精确率
- **recall**: 整体召回率
- **f1**: F1分数
- **AP@X.XX**: 特定IoU阈值下的平均精度

### Loss指标

- **Weighted Losses**: 训练时使用的加权损失
- **Unweighted Losses**: 未加权的原始损失
- **Total Loss**: 各项损失的总和

## 配置要求

1. **配置文件**: 必须与训练时使用的配置文件一致
2. **数据路径**: 确保测试数据路径在配置文件中正确设置
3. **任务设置**: TASKS列表必须包含要评估的任务

## 常见问题

### Q: 如何确定checkpoint路径？
A: 查看训练输出目录中的`epoch_X`文件夹，选择你想要评估的epoch。

### Q: 为什么某些指标显示0？
A: 可能原因：
- 测试数据集为空
- 模型预测全部为负样本
- 配置文件中的阈值设置过高

### Q: 如何只评估特定任务？
A: 在配置文件中修改`TASKS`列表，只包含要评估的任务。

### Q: 评估过程太慢怎么办？
A: 可以：
- 减小batch size
- 使用更少的数据（修改配置中的数据路径）
- 在配置中调整IoU阈值数量

## 扩展功能

如果需要添加新的评估指标：

1. 在相应的模型文件中实现metrics计算类
2. 在`models/build.py`的`build_metrics_fn`中注册
3. 重新运行evaluation即可

## 注意事项

- 确保有足够的GPU内存
- 测试数据预处理必须与训练时一致
- 多GPU环境下会自动使用所有可用GPU 