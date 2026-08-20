#!/bin/bash

# 示例脚本：运行eval.py进行模型评估
# 请根据你的实际情况修改以下路径

# 配置文件路径
CONFIG_PATH="configs/pretrain_base_224_multitask_aug_consine_part_temporal_early_freeze_text_extra_7000_6cards.yaml"

# checkpoint路径（可以是文件或目录）
CHECKPOINT_PATH="outputs/pretrain_base_224_multitask_aug_consine_part_temporal_early_freeze_text_extra_7000_6cards/epoch_8"

# 日志目录路径（可选）
LOG_DIR="outputs/eval_pretrain_base_224_multitask_aug_consine_part_temporal_early_freeze_text_extra_7000_6cards_epoch_8/eval_logs"

# 运行evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 eval.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --log_dir $LOG_DIR

echo "Evaluation completed! Check the results in $LOG_DIR" 