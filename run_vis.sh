#!/bin/bash

# 示例脚本：运行vis_detr.py进行DETR模型可视化
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/default.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="outputs/debug_video/epoch_14"

# 输出目录路径（可选，如果不设置会自动生成）
OUTPUT_DIR="outputs/debug_video/epoch_14/vis_results"

# 每个数据集的抽样数量
NUM_SAMPLES=10

# 预测分数阈值（过滤低分预测）
SCORE_THRESHOLD=0.3

# GPU设备
# GPU_ID=4

# 运行可视化
CUDA_VISIBLE_DEVICES=4 python vis_detr.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --score_threshold $SCORE_THRESHOLD

echo "可视化完成! 结果保存在: $OUTPUT_DIR" 