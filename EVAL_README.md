# Model Evaluation Script (eval_loss.py)

This script allows you to evaluate trained models on the test dataset and compute evaluation losses.

## Features

- **Multi-task Support**: Evaluates models on multiple tasks (SoccerNetGSR_Detection, SoccerNetGSR_ReID, etc.)
- **Flexible Checkpoint Loading**: Supports both HuggingFace `save_pretrained` format and standard PyTorch checkpoint files
- **Comprehensive Metrics**: Computes weighted, unweighted, and log-only losses for each task
- **Distributed Evaluation**: Supports multi-GPU evaluation using Accelerate
- **Detailed Logging**: Provides progress logs and saves results to JSON files

## Usage

### Basic Usage

```bash
python eval_loss.py \
    --config-path ./configs/your_config.yaml \
    --eval-model ./outputs/your_experiment/epoch_19 \
    --exp-name evaluation_experiment
```

### Command Line Arguments

Required arguments:
- `--config-path`: Path to the configuration YAML file (same as used during training)
- `--eval-model`: Path to the trained model checkpoint
- `--exp-name`: Name for the evaluation experiment (determines output directory)

Optional arguments:
- `--batch-size`: Batch size for evaluation (default from config)
- `--num-workers`: Number of data loading workers (default from config)
- `--outputs-dir`: Base directory for outputs (default: ./outputs/)
- `--data-root`: Root directory of datasets (default from config)
- `--use-tensorboard`: Enable TensorBoard logging (default: False)

### Checkpoint Formats

The script supports two checkpoint formats:

1. **HuggingFace format** (recommended): Directory containing model files saved with `save_pretrained`
   ```
   ./outputs/experiment_name/epoch_19/
   ├── config.json
   ├── model.safetensors
   └── ...
   ```

2. **PyTorch state dict**: Single `.pth` or `.pt` file
   ```
   ./outputs/experiment_name/model_checkpoint.pth
   ```

## Output Structure

The evaluation script creates the following output structure:

```
./outputs/{exp_name}/eval_results/
├── logs/                          # TensorBoard logs and text logs
│   ├── events.out.tfevents.*     # TensorBoard event files
│   └── log.txt                   # Text log file
├── eval_results.json             # Detailed evaluation results
└── config.yaml                  # Configuration used for evaluation
```

## Evaluation Results

The `eval_results.json` file contains:

### Structure
```json
{
  "evaluation_summary": {
    "total_samples": 12345,
    "tasks_evaluated": ["SoccerNetGSR_Detection", "SoccerNetGSR_ReID"],
    "task_sample_counts": {
      "SoccerNetGSR_Detection": 8000,
      "SoccerNetGSR_ReID": 4345
    }
  },
  "task_results": {
    "SoccerNetGSR_Detection": {
      "total_loss": 2.456,
      "weighted_loss_ce": 1.234,
      "weighted_loss_bbox": 0.678,
      "unweighted_loss_ce": 0.987,
      "num_samples": 8000,
      "num_batches": 1000
    },
    "SoccerNetGSR_ReID": {
      "total_loss": 1.789,
      "weighted_focal_loss": 0.456,
      "weighted_triplet_loss": 1.333,
      "num_samples": 4345,
      "num_batches": 543
    }
  },
  "overall_metrics": {
    "weighted_average_loss": 2.123,
    "SoccerNetGSR_Detection_total_loss": 2.456,
    "SoccerNetGSR_ReID_total_loss": 1.789
  }
}
```

### Key Metrics

- **total_loss**: Average total loss for each task
- **weighted_xxx**: Losses multiplied by their respective weights (used in training)
- **unweighted_xxx**: Raw losses before applying weights
- **log_only_xxx**: Metrics that are logged but not used in loss computation
- **weighted_average_loss**: Overall loss weighted by the number of samples per task

## Example Usage Scenarios

### 1. Evaluate Final Model
```bash
python eval_loss.py \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
    --eval-model ./outputs/final_experiment/epoch_19 \
    --exp-name final_evaluation
```

### 2. Compare Multiple Checkpoints
```bash
# Evaluate epoch 10
python eval_loss.py \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
    --eval-model ./outputs/experiment/epoch_9 \
    --exp-name eval_epoch10

# Evaluate epoch 15
python eval_loss.py \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
    --eval-model ./outputs/experiment/epoch_14 \
    --exp-name eval_epoch15
```

### 3. Custom Batch Size for Memory Constraints
```bash
python eval_loss.py \
    --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
    --eval-model ./outputs/experiment/epoch_19 \
    --exp-name memory_constrained_eval \
    --batch-size 2
```

## Multi-GPU Evaluation

The script automatically detects and uses available GPUs through Accelerate:

```bash
# Single GPU
python eval_loss.py --config-path ... --eval-model ...

# Multi-GPU
accelerate launch eval_loss.py --config-path ... --eval-model ...
```

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   - Verify the checkpoint path exists
   - Check if it's a directory (HuggingFace format) or file (PyTorch format)

2. **CUDA out of memory**
   - Reduce batch size: `--batch-size 1`
   - Reduce number of workers: `--num-workers 0`

3. **Missing test datasets**
   - Check if test datasets are properly configured in your config file
   - Verify dataset paths in the configuration

4. **Model architecture mismatch**
   - Ensure the configuration file matches the one used during training
   - Check if the model architecture in config matches the checkpoint

### Debug Mode

For debugging, you can enable more verbose logging:

```bash
python eval_loss.py \
    --config-path ./configs/debug_config.yaml \
    --eval-model ./path/to/checkpoint \
    --exp-name debug_eval \
    --use-tensorboard True
```

## Integration with Training Pipeline

This evaluation script is designed to work seamlessly with the training pipeline:

1. Train your model using `train.py`
2. After training completes, use `eval_loss.py` to evaluate the final checkpoint
3. The checkpoint path format matches what's saved by the training script
4. Use the same configuration file for both training and evaluation

## Performance Considerations

- **Batch Size**: Larger batch sizes are generally faster but use more memory
- **Number of Workers**: More workers can speed up data loading but use more CPU/memory
- **Mixed Precision**: Automatically enabled when supported by your hardware
- **Memory Management**: The script includes aggressive memory cleanup options for constrained environments 