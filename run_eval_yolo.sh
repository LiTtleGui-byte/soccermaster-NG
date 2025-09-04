#!/bin/bash

# 示例脚本：运行eval.py进行模型评估
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/image_224_eval_yolo.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="outputs/video_224_lr_align_open_text_loss_weight_4_unisoccer/epoch_17"

# 日志目录路径（可选）
LOG_DIR="outputs/image_224_eval_yolo/eval_logs"

# 运行evaluation
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 