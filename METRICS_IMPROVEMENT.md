# Detection Metrics 改进说明

## 问题描述

之前的实现存在两个主要问题：

1. **多进程重复输出**：在多卡训练时，每个进程都会输出evaluation结果，导致重复输出
2. **AP计算不准确**：AP是在单个batch内计算的，而不是在整个数据集上计算，这不符合标准的mAP计算方式

## 解决方案

### 1. 修改DetectionMetrics类

**新增功能：**
- `reset()`: 重置收集的数据
- `update()`: 在当前batch计算TP/FP/scores并收集到CPU
- `gather_tp_fp_scores()`: 在所有进程间聚合TP/FP/scores结果
- `compute_metrics_from_gathered_tp_fp()`: 从聚合的TP/FP数据计算metrics
- `compute_final_metrics()`: 计算最终的metrics（在所有数据收集完成后调用）

**工作流程：**
1. 在每个batch中调用`update()`在GPU上计算TP/FP，然后将结果转移到CPU收集
2. 在evaluation结束后调用`compute_final_metrics()`
3. 使用accelerator聚合所有进程的TP/FP/scores数据（已在CPU上）
4. 只在主进程计算最终的AP等指标

### 2. 修改evaluate_one_epoch函数

**主要变更：**
- 使用`metrics_fn_dict[task_name].update()`而不是直接计算metrics
- 在所有数据收集完成后调用`compute_final_metrics()`
- 只在主进程记录和返回最终metrics

### 3. 修改eval.py

**主要变更：**
- 只在主进程输出和保存结果
- 添加`accelerator.wait_for_everyone()`确保所有进程同步

## 使用方法

### 训练时的evaluation
```python
# 在train.py中，evaluation会自动使用新的metrics计算方式
# 不需要额外的代码修改
```

### 独立evaluation
```bash
# 使用多卡evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/debug_video/epoch_19 \
    --log_dir outputs/debug_video/epoch_19/eval_logs
```

## 技术细节

### AP计算改进
- **之前**：在每个batch内分别计算AP，然后平均
- **现在**：收集整个数据集的所有预测和真值，然后计算全局AP

### 多进程处理
- **之前**：每个进程独立计算并输出结果
- **现在**：使用`accelerator.gather_for_metrics()`聚合所有进程的数据，只在主进程计算最终结果

### 内存优化
- **GPU内存优化**：每个batch的TP/FP计算完成后立即转移到CPU，避免在GPU上累积大量预测数据
- **数据传输优化**：只传输必要的TP/FP/scores数据，而不是完整的预测框和标注
- **进程间通信优化**：传输的数据量大大减少，提高多进程聚合效率
- 提供`reset()`方法清理收集的数据

## 兼容性

- 保持了原有的`forward()`方法用于向后兼容
- 原有的调用方式仍然有效，但建议使用新的方式获得更准确的结果

## 性能优化亮点

### 🚀 内存效率提升
- **GPU内存使用**：从存储完整预测数据改为只计算TP/FP结果，大幅减少GPU内存占用
- **数据传输**：进程间只传输轻量级的TP/FP/scores数据，而不是完整的预测框和标注
- **即时处理**：每个batch处理完立即转移到CPU，避免GPU内存累积

### ⚡ 计算效率提升
- **分布式计算**：每张卡独立计算自己的TP/FP，充分利用多GPU并行能力
- **减少冗余**：避免在最终聚合时重复计算IoU和匹配
- **数据局部性**：在GPU上完成计算密集型操作，CPU上只做轻量级聚合

### 📊 准确性保证
- **全局AP计算**：确保AP是基于整个数据集计算，而不是batch平均
- **正确排序**：所有预测按置信度全局排序后计算precision-recall曲线
- **标准实现**：遵循COCO evaluation的标准流程
- **属性准确度**：对匹配成功的检测框计算role、jersey、digit_head、digit_tail的分类准确度

## 对比总结

| 方面 | 优化前 | 优化后 |
|------|--------|--------|
| GPU内存 | 存储所有预测框和标注 | 只存储TP/FP结果 |
| 数据传输 | 完整预测数据 | 轻量级TP/FP数据 |
| 计算分布 | 主进程集中计算 | 各进程并行计算 |
| AP准确性 | batch内平均 | 全局数据集计算 |
| 多进程输出 | 重复输出 | 主进程单一输出 |
| 属性评估 | 无 | role/jersey/digit准确度 |

## 新增功能：属性准确度计算

### 🎯 功能说明
- **匹配策略**：只有IoU@0.5匹配成功的检测框才会计算属性准确度
- **支持属性**：role、jersey number、digit_head、digit_tail
- **计算方式**：比较预测类别（argmax）与真实标签的一致性
- **输出指标**：
  - `{attribute}_accuracy`: 该属性的分类准确度
  - `{attribute}_matched_count`: 参与该属性评估的匹配框数量

### 📈 输出示例
```
role_accuracy: 0.8750
role_matched_count: 120
jersey_accuracy: 0.6500  
jersey_matched_count: 98
digit_head_accuracy: 0.7200
digit_head_matched_count: 85
digit_tail_accuracy: 0.8100
digit_tail_matched_count: 95
``` 