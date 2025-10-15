#!/bin/bash

# 示例脚本：运行eval.py进行模型评估
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/eval_downstream_detection_video_512_siglip_aug_color_flip_noise_blur_random_crop_affine_perspective_open_vision_on_image.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="outputs/eval_downstream_detection_video_512_siglip_aug_color_flip_noise_blur_random_crop_affine_perspective_open_vision_on_image"

# 日志目录路径（可选）
LOG_DIR="outputs/eval_downstream_detection_video_512_siglip_aug_color_flip_noise_blur_random_crop_affine_perspective_open_vision_on_image/eval_logs"

# 运行evaluation
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 