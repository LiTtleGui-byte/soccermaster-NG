#!/bin/bash

# 示例脚本：运行eval.py进行模型评估
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/probe_semantic_video_unisoccer_freeze_vision_2cards.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="./pretrained_models/unisoccer/pretrained_both.pth"

# 日志目录路径（可选）
LOG_DIR="outputs/eval_unisoccer_alignment/eval_logs"

# 运行evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 