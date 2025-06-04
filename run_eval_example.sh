#!/bin/bash

# Example script to run evaluation using eval_loss.py
# Usage: bash run_eval_example.sh

# Set the path to your trained checkpoint
# This should be the path to a directory containing the saved model (from training script's save_pretrained)
# or a .pth/.pt file containing the model state dict
CHECKPOINT_PATH="./outputs/default_2_tasks_camera/epoch_19"

# Set the config file (same as used during training)
CONFIG_PATH="./configs/default.yaml"

# Set experiment name for evaluation outputs
EXP_NAME="eval_experiment"

# Run evaluation
python eval_loss.py \
    --config-path $CONFIG_PATH \
    --eval-model $CHECKPOINT_PATH \
    --exp-name $EXP_NAME \
    --batch-size 4 \
    --num-workers 4

echo "Evaluation completed! Check results in ./outputs/$EXP_NAME/eval_results/" 