#!/bin/bash

# 示例脚本：运行eval.py进行模型评估
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/video_224_video_detection_all_align.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="outputs/video_224_video_detection_all_align/epoch_10"

# 日志目录路径（可选）
LOG_DIR="outputs/video_224_video_detection_all_align/epoch_10/eval_logs"

# 运行evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=29519 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 