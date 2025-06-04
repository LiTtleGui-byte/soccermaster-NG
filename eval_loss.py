# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs
from collections import defaultdict
import json

from data.build import build_dataloader
from utils.logger import Logger, MetricsTracker, TPS, Metrics
from models.multi_task import MultiTaskingSigLIP
from runtime_option import runtime_option
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn

def eval_loss_engine(config: dict):
    """
    Evaluation engine for computing loss on test set using trained checkpoint
    
    Args:
        config: Configuration dictionary containing model and data settings
    """
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    assert "EVAL_MODEL" in config and config["EVAL_MODEL"] is not None, "Please set the model checkpoint path for evaluation."
    
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])
    
    # Create evaluation specific output directory
    eval_outputs_dir = os.path.join(outputs_dir, "eval_results")
    os.makedirs(eval_outputs_dir, exist_ok=True)

    # Init Accelerator at beginning:
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    state = PartialState()
    
    # Set the seed:
    set_seed(config["SEED"])
    
    # Set the sharing strategy:
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Init Logger:
    log_dir = os.path.join(eval_outputs_dir, "logs")
    logger = Logger(
        log_dir=log_dir,
        accelerator=accelerator,
        config=config,
        # use_tensorboard=config.get("USE_TENSORBOARD", False),
        use_tensorboard=False,
        tensorboard_flush_secs=config.get("TENSORBOARD_FLUSH_SECS", 30)
    )
    logger.config(config=config)
    
    # Build test dataset only:
    _, dataloader_test_dict = build_dataloader(config=config)
    
    # Filter out None dataloaders (some tasks might not have test sets)
    dataloader_test_dict = {task: dataloader for task, dataloader in dataloader_test_dict.items() 
                           if dataloader is not None}
    
    if not dataloader_test_dict:
        logger.warning("No test dataloaders found. Please check your dataset configuration.")
        return
    
    # Build loss functions:
    loss_fn_dict = build_loss_fn(config=config)
    
    # Build model:
    model = MultiTaskingSigLIP(config=config)
    
    # Load checkpoint:
    logger.info(f"Loading checkpoint from: {config['EVAL_MODEL']}")
    
    # Check if the checkpoint path exists
    if not os.path.exists(config["EVAL_MODEL"]):
        raise FileNotFoundError(f"Checkpoint not found at: {config['EVAL_MODEL']}")
    
    # Load checkpoint based on the format used in training
    # The training script saves backbone weights using save_pretrained
    try:
        # Try loading as a directory with model files (save_pretrained format)
        if os.path.isdir(config["EVAL_MODEL"]):
            from transformers import AutoModel
            logger.info("Loading checkpoint as pretrained model directory...")
            # Load the backbone weights
            model.backbone.model = AutoModel.from_pretrained(config["EVAL_MODEL"])
        else:
            # Try loading as a single checkpoint file
            logger.info("Loading checkpoint as state dict...")
            checkpoint = torch.load(config["EVAL_MODEL"], map_location="cpu")
            
            # Handle different checkpoint formats
            if "model" in checkpoint:
                model_state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
            else:
                model_state_dict = checkpoint
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logger.info("Checkpoint loaded successfully!")
        
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        logger.info("Proceeding with randomly initialized model...")
    
    # Prepare model and dataloaders with accelerator:
    model = accelerator.prepare(model)
    dataloader_test_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_test_dict.items()}
    
    # Print model parameter info:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,} ({100 - trainable_percentage:.2f}%)")
    
    # Run evaluation:
    eval_results = evaluate_model(
        config=config,
        accelerator=accelerator,
        dataloader_dict=dataloader_test_dict,
        loss_fn_dict=loss_fn_dict,
        model=model,
        logger=logger
    )
    
    # Save evaluation results:
    results_path = os.path.join(eval_outputs_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {results_path}")
    logger.info("Evaluation completed!")
    
    # Close logger
    if logger:
        logger.close_tb_writer()
    
    return eval_results

def evaluate_model(
    config: dict,
    accelerator: Accelerator,
    dataloader_dict: dict[str, DataLoader],
    loss_fn_dict: dict[str, nn.Module],
    model,
    logger: Logger = None
):
    """
    Evaluate model on test dataset and compute losses
    
    Args:
        config: Configuration dictionary
        accelerator: Accelerator instance
        dataloader_dict: Dictionary of test dataloaders for each task
        loss_fn_dict: Dictionary of loss functions for each task
        model: Model to evaluate
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Starting evaluation...")
    
    model.eval()
    device = accelerator.device
    
    # Initialize metrics tracker for evaluation
    eval_metrics = MetricsTracker()
    task_metrics = {task: MetricsTracker() for task in dataloader_dict.keys()}
    
    total_samples = 0
    task_sample_counts = {task: 0 for task in dataloader_dict.keys()}
    
    # Process each task separately
    for task_name, dataloader in dataloader_dict.items():
        logger.info(f"Evaluating task: {task_name}")
        
        task_total_loss = 0.0
        task_loss_components = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, annotations, metas = batch.values()
                batch_size = images.size(0)
                
                # Forward pass
                with accelerator.autocast():
                    outputs = model(images, task_name)
                    
                    # Compute loss
                    loss_output = loss_fn_dict[task_name](outputs[task_name], annotations)
                    
                    # Parse loss output based on task type
                    if task_name in ["SoccerNetGSR_Detection"]:
                        loss_task_raw, weight_dict, _ = loss_output
                        
                        # Compute weighted losses
                        weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                        unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                        log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                        
                        # Total weighted loss for this batch
                        batch_total_loss = sum(weighted_losses.values())
                        
                        # Accumulate losses
                        for k, v in weighted_losses.items():
                            task_loss_components[f"weighted_{k}"] += v.item()
                        for k, v in unweighted_losses.items():
                            task_loss_components[f"unweighted_{k}"] += v.item()
                        for k, v in log_only_losses.items():
                            task_loss_components[f"log_only_{k}"] += v.item()
                            
                    elif task_name in ["SoccerNetGSR_ReID"]:
                        loss_task_raw, weight_dict = loss_output
                        
                        # Compute weighted losses
                        weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                        unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                        log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                        
                        # Total weighted loss for this batch
                        batch_total_loss = sum(weighted_losses.values())
                        
                        # Accumulate losses
                        for k, v in weighted_losses.items():
                            task_loss_components[f"weighted_{k}"] += v.item()
                        for k, v in unweighted_losses.items():
                            task_loss_components[f"unweighted_{k}"] += v.item()
                        for k, v in log_only_losses.items():
                            task_loss_components[f"log_only_{k}"] += v.item()
                            
                    else:
                        # For other tasks, assume loss_output is a dictionary of losses
                        if isinstance(loss_output, dict):
                            batch_total_loss = sum(loss_output.values())
                            for k, v in loss_output.items():
                                task_loss_components[k] += v.item()
                        else:
                            # Single loss value
                            batch_total_loss = loss_output
                            task_loss_components["total_loss"] += loss_output.item()
                
                task_total_loss += batch_total_loss.item()
                num_batches += 1
                task_sample_counts[task_name] += batch_size
                
                # Log progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = task_total_loss / num_batches
                    logger.info(f"Task {task_name} - Batch [{batch_idx + 1}/{len(dataloader)}], "
                              f"Avg Loss: {avg_loss:.4f}")
        
        # Compute average losses for this task
        if num_batches > 0:
            task_avg_loss = task_total_loss / num_batches
            avg_loss_components = {k: v / num_batches for k, v in task_loss_components.items()}
            
            # Update task metrics
            task_metrics[task_name].update({
                "total_loss": task_avg_loss,
                "num_samples": task_sample_counts[task_name],
                "num_batches": num_batches,
                **avg_loss_components
            })
            
            # Update global metrics
            eval_metrics.update({
                f"{task_name}_total_loss": task_avg_loss,
                f"{task_name}_num_samples": task_sample_counts[task_name]
            })
            for k, v in avg_loss_components.items():
                eval_metrics.update({f"{task_name}_{k}": v})
            
            logger.info(f"Task {task_name} completed - Avg Loss: {task_avg_loss:.4f}, "
                       f"Samples: {task_sample_counts[task_name]}")
    
    # Compute overall statistics
    total_samples = sum(task_sample_counts.values())
    
    # Prepare results dictionary
    results = {
        "evaluation_summary": {
            "total_samples": total_samples,
            "tasks_evaluated": list(dataloader_dict.keys()),
            "task_sample_counts": task_sample_counts
        },
        "task_results": {},
        "overall_metrics": {}
    }
    
    # Add task-specific results
    for task_name in dataloader_dict.keys():
        task_avg_metrics = task_metrics[task_name].get_averages()
        results["task_results"][task_name] = task_avg_metrics
        
        # Log task results
        logger.info(f"\n=== {task_name} Results ===")
        for metric_name, value in task_avg_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric_name}: {value:.6f}")
            else:
                logger.info(f"{metric_name}: {value}")
    
    # Add overall metrics
    overall_avg_metrics = eval_metrics.get_averages()
    results["overall_metrics"] = overall_avg_metrics
    
    # Compute weighted average loss across all tasks
    total_weighted_loss = 0.0
    total_weight = 0
    for task_name in dataloader_dict.keys():
        if f"{task_name}_total_loss" in overall_avg_metrics:
            task_loss = overall_avg_metrics[f"{task_name}_total_loss"]
            task_weight = task_sample_counts[task_name]
            total_weighted_loss += task_loss * task_weight
            total_weight += task_weight
    
    if total_weight > 0:
        weighted_avg_loss = total_weighted_loss / total_weight
        results["overall_metrics"]["weighted_average_loss"] = weighted_avg_loss
        logger.info(f"\n=== Overall Results ===")
        logger.info(f"Weighted Average Loss: {weighted_avg_loss:.6f}")
        logger.info(f"Total Samples: {total_samples}")
    
    return results

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:   # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:                                   # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)
    
    # Set the evaluation model path from command line argument
    if opt.eval_model is not None:
        cfg["EVAL_MODEL"] = opt.eval_model
    
    # Call the "eval_loss_engine" function:
    eval_results = eval_loss_engine(config=cfg) 