# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os
os.environ["NCCL_TIMEOUT"] = "7200"   # 7200秒 = 120分钟
# os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import timedelta
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from collections import defaultdict
from itertools import cycle
import json
import glob

from data.build import build_dataloader
from utils.logger import Logger, MetricsTracker, TPS, Metrics
from models.multi_task import MultiTaskingSigLIP
from runtime_option import runtime_option
from utils.misc import is_distributed, set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn, build_metrics_fn

def find_latest_checkpoint(outputs_dir):
    """
    Find the latest checkpoint directory in outputs_dir
    
    Args:
        outputs_dir: Output directory path
        
    Returns:
        tuple: (latest_epoch_dir, epoch_number) or (None, None) if no checkpoint found
    """
    if not os.path.exists(outputs_dir):
        return None, None
    
    # Find all epoch directories
    epoch_dirs = glob.glob(os.path.join(outputs_dir, "epoch_*"))
    if not epoch_dirs:
        return None, None
    
    # Extract epoch numbers and find the latest one
    valid_epochs = []
    for epoch_dir in epoch_dirs:
        try:
            epoch_num = int(os.path.basename(epoch_dir).split("_")[1])
            # Check if this epoch has a complete checkpoint (training_state.json should exist)
            training_state_file = os.path.join(epoch_dir, "training_state.json")
            if os.path.exists(training_state_file):
                valid_epochs.append((epoch_num, epoch_dir))
        except (ValueError, IndexError):
            continue
    
    if not valid_epochs:
        return None, None
    
    # Sort by epoch number and return the latest
    valid_epochs.sort(key=lambda x: x[0])
    latest_epoch, latest_dir = valid_epochs[-1]
    
    return latest_dir, latest_epoch

def save_training_state(outputs_dir, epoch, model, optimizer, scheduler, train_states, accelerator, logger):
    """
    Save complete training state including model, optimizer, scheduler and training progress
    
    Args:
        outputs_dir: Output directory
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        train_states: Training states dictionary
        accelerator: Accelerator instance
        logger: Logger instance
    """
    # Use model.module to access original model attributes when using DDP
    original_model = model.module if hasattr(model, 'module') else model
    
    # Create epoch directory
    epoch_dir = os.path.join(outputs_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save backbone weights in backbone subdirectory
    backbone_dir = os.path.join(epoch_dir, 'backbone')
    original_model.backbone.vision_model.save_pretrained(backbone_dir)
    
    # Save each head separately
    for head_name, head in original_model.multi_task_head.items():
        torch.save(head.state_dict(), os.path.join(epoch_dir, f'{head_name}.pt'))
    
    # Save optimizer state
    optimizer_state_file = os.path.join(epoch_dir, "optimizer_state.pt")
    torch.save(optimizer.state_dict(), optimizer_state_file)
    
    # Save scheduler state
    scheduler_state_file = os.path.join(epoch_dir, "scheduler_state.pt")
    torch.save(scheduler.state_dict(), scheduler_state_file)
    
    # Save training states and other metadata
    training_state = {
        "epoch": epoch,
        "global_step": train_states["global_step"],
        "start_epoch": train_states["start_epoch"],
        "random_states": {
            "python": torch.get_rng_state().tolist(),
            "numpy": torch.random.get_rng_state().tolist() if hasattr(torch.random, 'get_rng_state') else None,
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None,
        }
    }
    
    training_state_file = os.path.join(epoch_dir, "training_state.json")
    with open(training_state_file, 'w') as f:
        json.dump(training_state, f, indent=2)
    
    logger.info(f"Saved complete training state for epoch {epoch} to {epoch_dir}")

def load_training_state(checkpoint_dir, model, optimizer, scheduler, train_states, accelerator, logger):
    """
    Load complete training state from checkpoint
    
    Args:
        checkpoint_dir: Checkpoint directory path
        model: Model to load state into
        optimizer: Optimizer to load state into  
        scheduler: Learning rate scheduler to load state into
        train_states: Training states dictionary to update
        accelerator: Accelerator instance
        logger: Logger instance
        
    Returns:
        int: Loaded epoch number
    """
    # Use model.module to access original model attributes when using DDP
    original_model = model.module if hasattr(model, 'module') else model
    
    logger.info(f"Loading training state from {checkpoint_dir}")
    
    original_model.load_checkpoint(checkpoint_dir, logger)
    # # Load backbone weights
    # backbone_dir = os.path.join(checkpoint_dir, 'backbone')
    # if os.path.exists(backbone_dir):
    #     # Load backbone weights using the from_pretrained method
    #     backbone_model = original_model.backbone.vision_model.__class__.from_pretrained(backbone_dir)
    #     original_model.backbone.vision_model.load_state_dict(backbone_model.state_dict())
    #     logger.info(f"Loaded backbone weights from {backbone_dir}")
    
    # # Load each head
    # for head_name, head in original_model.multi_task_head.items():
    #     head_file = os.path.join(checkpoint_dir, f'{head_name}.pt')
    #     if os.path.exists(head_file):
    #         head.load_state_dict(torch.load(head_file, map_location='cpu'))
    #         logger.info(f"Loaded {head_name} head weights from {head_file}")
    
    # Load optimizer state
    optimizer_state_file = os.path.join(checkpoint_dir, "optimizer_state.pt")
    optimizer.load_state_dict(torch.load(optimizer_state_file, map_location='cpu'))
    logger.info(f"Loaded optimizer state from {optimizer_state_file}")
    
    # Load scheduler state
    scheduler_state_file = os.path.join(checkpoint_dir, "scheduler_state.pt")
    scheduler.load_state_dict(torch.load(scheduler_state_file, map_location='cpu'))
    logger.info(f"Loaded scheduler state from {scheduler_state_file}")
    
    # Load training states
    training_state_file = os.path.join(checkpoint_dir, "training_state.json")
    resumed_epoch = 0
    with open(training_state_file, 'r') as f:
        training_state = json.load(f)
    
    resumed_epoch = training_state["epoch"]
    train_states["global_step"] = training_state["global_step"]
    train_states["start_epoch"] = training_state["start_epoch"]
    
    # Restore random states for reproducibility
    if "random_states" in training_state:
        random_states = training_state["random_states"]
        if random_states["python"]:
            torch.set_rng_state(torch.tensor(random_states["python"], dtype=torch.uint8))
        if random_states["numpy"] and hasattr(torch.random, 'set_rng_state'):
            torch.random.set_rng_state(torch.tensor(random_states["numpy"], dtype=torch.uint8))
        if random_states["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.tensor(random_states["cuda"], dtype=torch.uint8))
    
    logger.info(f"Loaded training state: epoch={resumed_epoch}, global_step={train_states['global_step']}")
    
    return resumed_epoch

# Create parameter groups with different learning rates
def create_param_groups(model, config):
    # Get the original model (handle DDP wrapper)
    original_model = model.module if hasattr(model, 'module') else model
    
    param_groups = []
    
    # Backbone parameters
    backbone_params = []
    for param in original_model.backbone.parameters():
        if param.requires_grad:
            backbone_params.append(param)
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': config["LR_BACKBONE"],
            'weight_decay': config["WEIGHT_DECAY"],
            'name': 'backbone'
        })
    
    # Head parameters with different learning rates
    head_lr_mapping = {
        'SoccerNetGSR_ReID': config["LR_SOCCERNET_GSR_REID"],
        'SoccerNetGSR_Detection': config["LR_SOCCERNET_GSR_DETECTION"],
        'LinesDetection': config["LR_LINES_DETECTION"],
        'KeypointsDetection': config["LR_KEYPOINTS_DETECTION"],
        'CameraRegression': config["LR_CAMERA_REGRESSION"],
        'VideoCaption': config["LR_VIDEO_CAPTION"],
        'CaptionClassification': config["LR_CAPTION_CLASSIFICATION"],
        'CaptionClassificationAlign': config["LR_CAPTION_CLASSIFICATION"]
    }
    
    for head_name, head in original_model.multi_task_head.items():
        # Special handling for CaptionClassificationAlign head to separate classifier parameters
        if head_name in ['CaptionClassification', 'CaptionClassificationAlign']:
            # Separate classifier parameters from other parameters
            classifier_params = []
            other_params = []
            
            for name, param in head.named_parameters():
                if param.requires_grad:
                    if 'classifier' in name.lower():
                        classifier_params.append(param)
                    else:
                        other_params.append(param)
            
            # Add classifier parameters with specific learning rate
            if classifier_params:
                param_groups.append({
                    'params': classifier_params,
                    'lr': config["LR_CAPTION_CLASSIFICATION_CLASSIFIER"],
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': f'{head_name}_classifier'
                })
            
            # Add other parameters with default head learning rate
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': head_lr_mapping[head_name],
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': f'{head_name}_other'
                })
        else:
            # Regular head parameter handling
            head_params = []
            for param in head.parameters():
                if param.requires_grad:
                    head_params.append(param)
            
            if head_params:
                lr = head_lr_mapping[head_name]  # Default to base LR if not specified
                param_groups.append({
                    'params': head_params,
                    'lr': lr,
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': head_name
                })
    
    return param_groups

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    # accelerator = Accelerator()
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(minutes=120))
        ]
    )
    # accelerator = Accelerator(
    #     kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)]
    # )
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    
    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.
    # torch.autograd.set_detect_anomaly(True)
    
    # Init TensorBoard Logger:
    log_dir = os.path.join(outputs_dir, "logs")
    logger = Logger(
        log_dir=log_dir,
        accelerator=accelerator,
        config=config,
        use_tensorboard=config["USE_TENSORBOARD"],
        tensorboard_flush_secs=config["TENSORBOARD_FLUSH_SECS"]
    )
    logger.config(config=config)
    
    # Build training dataset:
    dataloader_train_dict, dataloader_test_dict = build_dataloader(config=config)
    
    # Filter out None test dataloaders (some tasks might not have test sets)
    dataloader_test_dict = {dataset: dataloader for dataset, dataloader in dataloader_test_dict.items() 
                           if dataloader is not None}
    
    if dataloader_test_dict:
        logger.info(f"Test datasets available for tasks: {list(dataloader_test_dict.keys())}")
    else:
        logger.warning("No test datasets available. Evaluation will be skipped.")
    
    # Build loss functions:
    loss_fn_dict = build_loss_fn(config=config)
    
    # Build metrics functions:
    metrics_fn_dict = build_metrics_fn(config=config)
    
    model = MultiTaskingSigLIP(config=config, logger=logger)
    if config["LOAD_CHECKPOINTS"]:
        model.load_checkpoint(config["STAGE_1_CKPT_DIR"], logger)
    if config['USE_GRADIENT_CHECKPOINTING']:
        model.backbone.vision_model.gradient_checkpointing_enable()
    
    # Create optimizer with parameter groups
    param_groups = create_param_groups(model, config)
    optimizer = AdamW(param_groups)
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )
    
    # torch.cuda.init()  # 显式初始化CUDA
    # device = torch.device("cuda")
    # torch.tensor([1.0]).to(device)  # 创建虚拟张量强制初始化上下文
    
    # if config["USE_GRADIENT_CHECKPOINTING"]:
    #     accelerator.gradient_checkpointing_enable()
    
    # Fix DDP parameter sharing issue by setting static graph
    # if hasattr(model, 'module') and hasattr(model.module, '_set_static_graph'):
    #     model.module._set_static_graph()
    # elif hasattr(model, '_set_static_graph'):
    #     model._set_static_graph()
    
    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }
    
    # Check for resume training from checkpoint
    resume_from_checkpoint = False
    resume_epoch = 0
    if config["RESUME_TRAINING"]:
        checkpoint_dir, latest_epoch = find_latest_checkpoint(outputs_dir)
        if checkpoint_dir is not None:
            logger.info(f"Found checkpoint at epoch {latest_epoch}, will resume training from {checkpoint_dir}")
            resume_from_checkpoint = True
            # We'll load the state after preparing model, optimizer, scheduler
        else:
            logger.info("RESUME_TRAINING is True but no valid checkpoint found, starting from scratch")
    
    # Alternative: resume from specific checkpoint directory
    if config["RESUME_FROM_CHECKPOINT_DIR"] is not None:
        resume_checkpoint_dir = config["RESUME_FROM_CHECKPOINT_DIR"]
        if os.path.exists(resume_checkpoint_dir):
            training_state_file = os.path.join(resume_checkpoint_dir, "training_state.json")
            if os.path.exists(training_state_file):
                logger.info(f"Will resume training from specified checkpoint: {resume_checkpoint_dir}")
                resume_from_checkpoint = True
                checkpoint_dir = resume_checkpoint_dir
            else:
                logger.warning(f"Specified checkpoint directory {resume_checkpoint_dir} does not contain valid training state")
        else:
            logger.warning(f"Specified checkpoint directory {resume_checkpoint_dir} does not exist")

    # Load checkpoint state if resuming training
    if resume_from_checkpoint:
        resume_epoch = load_training_state(
            checkpoint_dir, model, optimizer, scheduler, train_states, accelerator, logger
        )
        # Update start_epoch to resume from the next epoch
        train_states["start_epoch"] = resume_epoch + 1
        logger.info(f"Resuming training from epoch {resume_epoch + 1}, global_step {train_states['global_step']}")
        
        # Mark resume point in tensorboard
        logger.mark_resume(resume_epoch, train_states['global_step'])
            
    model, optimizer = accelerator.prepare(model, optimizer)
    dataloader_train_dict = {dataset: accelerator.prepare(dataloader) for dataset, dataloader in dataloader_train_dict.items()}
    if dataloader_test_dict:
        dataloader_test_dict = {dataset: accelerator.prepare(dataloader) for dataset, dataloader in dataloader_test_dict.items()}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    
    # Get original model (handle DDP wrapper)
    original_model = model.module if hasattr(model, 'module') else model
    
    # Calculate vision_model parameters
    vision_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters())
    vision_trainable_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters() if p.requires_grad)
    
    # Calculate each head parameters
    head_params = {}
    head_trainable_params = {}
    total_head_params = 0
    total_head_trainable_params = 0
    
    for head_name, head in original_model.multi_task_head.items():
        head_total = sum(p.numel() for p in head.parameters())
        head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
        head_params[head_name] = head_total
        head_trainable_params[head_name] = head_train
        total_head_params += head_total
        total_head_trainable_params += head_train
    
    # Log parameter statistics (in millions)
    logger.info(f"=== Model Parameter Statistics (Unit: M) ===")
    logger.info(f"Total parameters: {total_params/1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M ({trainable_percentage:.2f}%)")
    logger.info(f"Non-trainable parameters: {(total_params - trainable_params)/1e6:.2f}M ({100 - trainable_percentage:.2f}%)")
    logger.info(f"")
    logger.info(f"Vision Model parameters: {vision_params/1e6:.2f}M")
    logger.info(f"Vision Model trainable: {vision_trainable_params/1e6:.2f}M ({vision_trainable_params/vision_params*100:.2f}%)")
    logger.info(f"")
    logger.info(f"Total Head parameters: {total_head_params/1e6:.2f}M")
    logger.info(f"Total Head trainable: {total_head_trainable_params/1e6:.2f}M ({total_head_trainable_params/total_head_params*100:.2f}%)")
    logger.info(f"")
    for head_name in head_params:
        logger.info(f"{head_name} Head parameters: {head_params[head_name]/1e6:.2f}M")
        logger.info(f"{head_name} Head trainable: {head_trainable_params[head_name]/1e6:.2f}M ({head_trainable_params[head_name]/head_params[head_name]*100:.2f}%)")
    logger.info(f"============================================")
    
    # Log parameter groups and their learning rates
    logger.info(f"=== Learning Rate Configuration ===")
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group['name']
        group_lr = param_group['lr']
        group_params = len(param_group['params'])
        logger.info(f"{group_name}: LR={group_lr:.0e}, Parameters={group_params}")
    logger.info(f"=====================================")
    
    # Print names of non-trainable parameters
    # logger.info("Non-trainable layers:")
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         logger.info(f"  {name}")
    
    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        # Train one epoch:
        train_one_epoch(
            config=config,
            accelerator=accelerator,
            states=train_states,
            epoch=epoch,
            dataloader_dict=dataloader_train_dict,
            loss_fn_dict=loss_fn_dict,
            model=model,
            optimizer=optimizer,
            logger=logger,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            # lr_warmup_tgt_lr=config["LR"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            max_clip_norm=config["MAX_CLIP_NORM"],
            use_accelerate_clip_norm=config["USE_ACCELERATE_CLIP_NORM"],
            logging_interval=config["LOGGING_INTERVAL"],
        )
        scheduler.step()
        if is_distributed():
            torch.distributed.barrier()
        
        # Evaluate after each epoch if test datasets are available
        if dataloader_test_dict and (epoch + 1) % config["EVAL_PER_EPOCH"] == 0:
            logger.info(f"Starting evaluation for epoch {epoch}...")
            eval_results = evaluate_one_epoch(
                config=config,
                accelerator=accelerator,
                epoch=epoch,
                dataloader_dict=dataloader_test_dict,
                loss_fn_dict=loss_fn_dict,
                metrics_fn_dict=metrics_fn_dict,
                model=model,
                logger=logger
            )
            logger.info(f"Evaluation completed for epoch {epoch}")
        if is_distributed():
            torch.distributed.barrier()
        
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            # Save complete training state including model, optimizer, scheduler
            save_training_state(
                outputs_dir, epoch, model, optimizer, scheduler, train_states, accelerator, logger
            )
    
    # Close logger at the end of training
    if logger:
        logger.close_tb_writer()
        logger.info("Training completed. TensorBoard logger closed.")

def evaluate_one_epoch(
    config: dict,
    accelerator: Accelerator,
    epoch: int,
    dataloader_dict: dict[str, DataLoader],
    loss_fn_dict: dict[str, nn.Module],
    metrics_fn_dict: dict[str, nn.Module],
    model,
    logger: Logger
):
    """
    Evaluate model on test dataset for one epoch and log results to tensorboard
    
    Args:
        config: Configuration dictionary
        accelerator: Accelerator instance
        epoch: Current epoch number
        dataloader_dict: Dictionary of test dataloaders for each dataset
        loss_fn_dict: Dictionary of loss functions for each head
        metrics_fn_dict: Dictionary of metrics functions for each head
        model: Model to evaluate
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    device = accelerator.device
    
    # Get dataset to heads mapping
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    
    # Collect all heads
    all_heads = []
    for dataset_name, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    # Initialize metrics tracker for evaluation - 使用二级字典结构，按head分组
    eval_weighted_losses = {head: MetricsTracker() for head in all_heads}
    eval_unweighted_losses = {head: MetricsTracker() for head in all_heads}
    eval_log_only_losses = {head: MetricsTracker() for head in all_heads}
    
    head_sample_counts = {head: 0 for head in all_heads}
    
    # Process each dataset separately
    for dataset_name, dataloader in dataloader_dict.items():
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, annotations, metas = batch.values()
                batch_size = images.size(0)
                if dataset_name in ["VideoCaption"]:
                    text = [annotation['text'] for annotation in annotations]
                else:
                    text = None
                
                # Forward pass
                with accelerator.autocast():
                    outputs = model(images, dataset_name, metas, text)
                    
                    # Process each head for this dataset
                    for head_name in datasets_to_heads[dataset_name]:
                        # Compute loss
                        loss_output = loss_fn_dict[head_name](outputs[head_name], annotations)
                        
                        # Parse loss output based on task type
                        if head_name in ["SoccerNetGSR_Detection"]:
                            loss_task_raw, weight_dict, _ = loss_output
                            
                            # Compute weighted and unweighted losses
                            weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                            unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                            log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                            
                            # Update metrics trackers
                            eval_weighted_losses[head_name].update(weighted_losses)
                            eval_unweighted_losses[head_name].update(unweighted_losses)
                            eval_log_only_losses[head_name].update(log_only_losses)
                            
                            # Compute metrics if metrics function is available
                            if metrics_fn_dict[head_name] is not None:
                                if 'target_sizes' in metas:
                                    target_sizes = metas['target_sizes']
                                else:
                                    target_sizes = torch.tensor([[512, 512]] * batch_size, device=device)
                                metrics_fn_dict[head_name].update(outputs[head_name], annotations, target_sizes)
                                
                        elif head_name in ["SoccerNetGSR_ReID", "VideoCaption", "CaptionClassification", "CaptionClassificationAlign", "LinesDetection", "KeypointsDetection", "CameraRegression"]:
                            loss_task_raw, weight_dict = loss_output
                            
                            # Compute weighted and unweighted losses
                            weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                            unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                            log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                            
                            # Update metrics trackers
                            eval_weighted_losses[head_name].update(weighted_losses)
                            eval_unweighted_losses[head_name].update(unweighted_losses)
                            eval_log_only_losses[head_name].update(log_only_losses)
                            
                            # Compute metrics if metrics function is available
                            if metrics_fn_dict[head_name] is not None:
                                # For VideoCaption, pass loss_task_raw to extract accuracy values
                                if head_name == "VideoCaption":
                                    metrics_fn_dict[head_name].update(outputs[head_name], annotations, loss_task_raw)
                                else:
                                    # ReID, CaptionClassification, Camera metrics can be implemented later
                                    metrics_fn_dict[head_name].update(outputs[head_name], annotations)
                                
                        else:
                            raise ValueError(f"Unknown head name: {head_name}")
                        
                        head_sample_counts[head_name] += batch_size
        
        logger.info(f"Dataset {dataset_name} eval completed")
    
    # 计算最终的metrics（在所有数据收集完成后）
    final_metrics_results = {}
    for head_name in all_heads:
        if metrics_fn_dict[head_name] is not None:
            # 计算最终metrics并只在主进程返回结果
            if is_distributed():
                torch.distributed.barrier()
            final_metrics = metrics_fn_dict[head_name].compute_final_metrics(accelerator)
            if accelerator.is_main_process:
                final_metrics_results[head_name] = final_metrics
            else:
                final_metrics_results[head_name] = {}
            metrics_fn_dict[head_name].reset()
    
    # Log evaluation results to tensorboard
    if logger:
        # Calculate overall losses
        total_weighted_loss = 0.0
        total_unweighted_loss = 0.0
        total_samples = sum(head_sample_counts.values())
        
        # Log head-specific results and accumulate overall losses
        for head_name in all_heads:
            # Get average metrics for this head
            head_weighted_avg = eval_weighted_losses[head_name].get_averages()
            head_unweighted_avg = eval_unweighted_losses[head_name].get_averages()
            head_log_only_avg = eval_log_only_losses[head_name].get_averages()
            
            # Calculate head total losses
            head_weighted_total = sum(head_weighted_avg.values()) if head_weighted_avg else 0.0
            head_unweighted_total = sum(head_unweighted_avg.values()) if head_unweighted_avg else 0.0
            
            # Weight by sample count for overall calculation
            sample_weight = head_sample_counts[head_name] / total_samples if total_samples > 0 else 0.0
            total_weighted_loss += head_weighted_total * sample_weight
            total_unweighted_loss += head_unweighted_total * sample_weight
            
            # Log head-specific loss metrics
            if head_weighted_avg:
                for metric_name, value in head_weighted_avg.items():
                    logger.log_scalar(f"eval_weighted_{head_name}/{metric_name}", value, epoch)
                logger.log_scalar(f"eval_weighted_{head_name}/total_loss", head_weighted_total, epoch)
            
            if head_unweighted_avg:
                for metric_name, value in head_unweighted_avg.items():
                    logger.log_scalar(f"eval_unweighted_{head_name}/{metric_name}", value, epoch)
                logger.log_scalar(f"eval_unweighted_{head_name}/total_loss", head_unweighted_total, epoch)
            
            if head_log_only_avg:
                for metric_name, value in head_log_only_avg.items():
                    logger.log_scalar(f"eval_unweighted_{head_name}/{metric_name}", value, epoch)
            
            # Log head-specific detection/evaluation metrics (mAP, precision, recall等)
            if head_name in final_metrics_results:
                for metric_name, value in final_metrics_results[head_name].items():
                    logger.log_scalar(f"eval_metrics_{head_name}/{metric_name}", value, epoch)
        
        # Log overall metrics
        logger.log_scalar("eval_overall/weighted_total_loss", total_weighted_loss, epoch)
        logger.log_scalar("eval_overall/unweighted_total_loss", total_unweighted_loss, epoch)
        logger.log_scalar("eval_overall/total_samples", total_samples, epoch)
        
        # Flush logger
        logger.flush_tb_writer()
    
    # Return evaluation results
    results = {
        "head_weighted_results": {head: eval_weighted_losses[head].get_averages() for head in all_heads},
        "head_unweighted_results": {head: eval_unweighted_losses[head].get_averages() for head in all_heads},
        "head_log_only_results": {head: eval_log_only_losses[head].get_averages() for head in all_heads},
        "head_metrics_results": final_metrics_results,  # 使用最终计算的metrics
        "head_sample_counts": head_sample_counts,
        "overall_weighted_loss": total_weighted_loss if logger else 0.0,
        "overall_unweighted_loss": total_unweighted_loss if logger else 0.0
    }
    
    return results
    
def train_one_epoch(
        # Infos:
        config: dict,
        accelerator: Accelerator,
        states: dict,
        epoch: int,
        dataloader_dict: dict[str, DataLoader],
        loss_fn_dict: dict[str, nn.Module],
        model,
        optimizer,
        logger: Logger,
        lr_warmup_epochs: int = 0,
        accumulate_steps: int = 1,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
):
    epoch_start_timestamp = TPS.timestamp()
    current_last_checkpoint_idx = 0
    model.train()
    tps = TPS()
    metrics = Metrics()
    step_timestamp = tps.timestamp()
    optimizer.zero_grad()
    device = accelerator.device
    
    assert accumulate_steps == 1, "accumulate_steps must be 1 for now."
    
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset_name, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    max_iterations = max(len(dataloader) for dataloader in dataloader_dict.values())
    
    # Initialize metrics tracker for epoch-level logging
    epoch_metrics = MetricsTracker()
    
    logger.info(f"Training epoch {epoch} with {max_iterations} iterations...")
    
    # 备选方案：预计算采样策略 (适用于训练稳定后的优化)
    # def create_balanced_sampling_plan(dataloader_lengths, max_iterations):
    #     """创建平衡采样计划，避免运行时的动态重置"""
    #     sampling_plan = {}
    #     for task, length in dataloader_lengths.items():
    #         # 计算每个任务需要重复多少轮
    #         repeats = (max_iterations + length - 1) // length
    #         indices = list(range(length)) * repeats
    #         sampling_plan[task] = indices[:max_iterations]
    #     return sampling_plan
    # 
    # # 使用示例：
    # # sampling_plan = create_balanced_sampling_plan(dataloader_lengths, max_iterations)
    # # 然后根据sampling_plan来索引数据

    dataloader_iters = {task: iter(dataloader) for task, dataloader in dataloader_dict.items()}
    len_tasks = len(dataloader_iters)

    # Track which dataloaders have been exhausted and reset
    dataloader_lengths = {task: len(dataloader) for task, dataloader in dataloader_dict.items()}
    # dataloader_counters = {task: 0 for task in dataloader_dict.keys()}
    logger.info(f"Dataloader lengths: {dataloader_lengths}")

    for cur_iter in range(max_iterations):
        # {head_name: loss_dict}
        weighted_loss_dict = {}  # 用于backward的加权loss
        unweighted_loss_dict = {}  # 记录加权前的原始loss
        log_only_loss_dict = {}
        
        # 逐任务处理，每个任务计算完立即backward以减少显存占用
        for dataset_name, dataloader_iter in dataloader_iters.items():
            with accelerator.autocast():
                # batch = next(dataloader)
                try:
                    batch = next(dataloader_iter)
                    # dataloader_counters[dataset_name] += 1
                except StopIteration:
                    # Reset the dataloader iterator and counter when exhausted
                    logger.info(f"Dataset {dataset_name} dataloader exhausted at iteration {cur_iter}, resetting...")
                    dataloader_iters[dataset_name] = iter(dataloader_dict[dataset_name])
                    # dataloader_counters[dataset_name] = 1  # Reset counter to 1 (current batch)
                    batch = next(dataloader_iters[dataset_name])
                    
                images, annotations, metas = batch.values()
                if dataset_name in ["VideoCaption"]:
                    text = [annotation['text'] for annotation in annotations]
                else:
                    text = None
                
                # Learning rate warmup:
                if epoch < lr_warmup_epochs:
                    # Do warmup:
                    lr_warmup_multi_groups(
                        optimizer=optimizer,
                        epoch=epoch, curr_iter=cur_iter,
                        warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=max_iterations,
                    )
                
                # 可选：使用梯度检查点进一步减少显存（会增加计算时间）
                # if config["USE_GRADIENT_CHECKPOINTING"]:
                #     outputs = torch.utils.checkpoint.checkpoint(model, images, task, metas, text, use_reentrant=False)
                # else:
                outputs = model(images, dataset_name, metas, text)
                
                loss_outputs = {}
                for head in datasets_to_heads[dataset_name]:
                    loss_outputs[head] = loss_fn_dict[head](outputs[head], annotations)
                    
                # loss_output = loss_fn_dict[dataset_name](outputs[dataset_name], annotations)
                for head_name in datasets_to_heads[dataset_name]:
                    loss_output = loss_outputs[head_name]
                    if head_name in ["SoccerNetGSR_Detection"]:
                        loss_task_raw, weight_dict, _ = loss_output
                    elif head_name in ["SoccerNetGSR_ReID", "VideoCaption", "CaptionClassification", "CaptionClassificationAlign", "KeypointsDetection", "LinesDetection", "CameraRegression"]:
                        loss_task_raw, weight_dict = loss_output
                    else:
                        raise ValueError(f"Head {head_name} not supported.")
                    unweighted_loss_dict[head_name] = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                    weighted_loss_dict[head_name] = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                    log_only_loss_dict[head_name] = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
            
            # 立即对当前dataset所属的所有heads计算的loss进行backward，减少显存占用
            dataset_total_loss = sum(sum(weighted_loss_dict[head].values()) for head in datasets_to_heads[dataset_name])
            # dataset_total_loss /= (accumulate_steps * len_tasks)  # 除以任务数量进行平均
            accelerator.backward(dataset_total_loss)
            
            # 可选：每个任务后清理CUDA缓存（会影响性能，但最大化显存释放）
            if config.get("AGGRESSIVE_MEMORY_CLEANUP", False):
                torch.cuda.empty_cache()
        
        # 梯度裁剪和参数更新
        if (cur_iter + 1) % accumulate_steps == 0:
            if use_accelerate_clip_norm:
                # Clip gradients separately for backbone and each head
                # Use model.module to access original model attributes when using DDP
                original_model = model.module if hasattr(model, 'module') else model
                backbone_grad_norm = accelerator.clip_grad_norm_(original_model.backbone.parameters(), max_norm=max_clip_norm)
                head_grad_norms = {}
                for head_name, head in original_model.multi_task_head.items():
                    head_grad_norms[head_name] = accelerator.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
            else:
                accelerator.unscale_gradients()
                # Clip gradients separately for backbone and each head
                # Use model.module to access original model attributes when using DDP
                original_model = model.module if hasattr(model, 'module') else model
                backbone_grad_norm = torch.nn.utils.clip_grad_norm_(original_model.backbone.parameters(), max_clip_norm)
                head_grad_norms = {}
                for head_name, head in original_model.multi_task_head.items():
                    head_grad_norms[head_name] = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
                
        # Update global step
        states["global_step"] += 1
        tps.update(TPS.timestamp() - step_timestamp)
        step_timestamp = TPS.timestamp()
        
        # Add logging
        if logger and (cur_iter + 1) % logging_interval == 0:
            # 计算overall losses
            total_weighted_loss = sum(sum(task_losses.values()) for task_losses in weighted_loss_dict.values())
            total_unweighted_loss = sum(sum(task_losses.values()) for task_losses in unweighted_loss_dict.values())
            
            # Log overall losses
            logger.log_scalar("train_overall/weighted_total_loss", total_weighted_loss, states["global_step"])
            logger.log_scalar("train_overall/unweighted_total_loss", total_unweighted_loss, states["global_step"])
            
            # Log task-specific losses
            for head_name in weighted_loss_dict.keys():
                if weighted_loss_dict[head_name]:
                    logger.log_loss_dict(weighted_loss_dict[head_name], states["global_step"], prefix=f"train_weighted_{head_name}")
                if unweighted_loss_dict[head_name]:
                    logger.log_loss_dict(unweighted_loss_dict[head_name], states["global_step"], prefix=f"train_unweighted_{head_name}")
                if log_only_loss_dict[head_name]:
                    logger.log_loss_dict(log_only_loss_dict[head_name], states["global_step"], prefix=f"train_unweighted_{head_name}", count_sum=False)
            
            logger.log_learning_rate(optimizer, states["global_step"])
            # Log separate gradient norms for backbone and each head
            logger.log_scalar("train_grad_norm/backbone_grad_norm", backbone_grad_norm, states["global_step"])
            for head_name, head_grad_norm in head_grad_norms.items():
                logger.log_scalar(f"train_grad_norm/{head_name}_head_grad_norm", head_grad_norm, states["global_step"])
            
            # Log parameter and gradient statistics if enabled
            # if config.get("LOG_PARAMS_GRADS", False):
            #     logger.log_model_parameters(model, states["global_step"])
            
            # Update metrics tracker (epoch metrics) - 记录所有类型的loss
            # 为epoch级别的metrics更新，需要将二级字典展平
            for head_name, task_losses in weighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"weighted_{head_name}_{loss_name}": loss_value})
            
            for head_name, task_losses in unweighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"unweighted_{head_name}_{loss_name}": loss_value})
            
            for head_name, task_losses in log_only_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"{head_name}_{loss_name}": loss_value})
            
            # Update metrics (iteration metrics)
            metrics.update(name="weighted_total_loss", value=total_weighted_loss.detach())
            metrics.update(name="unweighted_total_loss", value=total_unweighted_loss.detach())
            
            # 为iteration级别的metrics更新，需要将二级字典展平
            for head_name, task_losses in weighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"weighted_{head_name}_{loss_name}", value=loss_value.detach())
            
            for head_name, task_losses in unweighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"unweighted_{head_name}_{loss_name}", value=loss_value.detach())
            
            for head_name, task_losses in log_only_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"{head_name}_{loss_name}", value=loss_value.detach())
            
            # Log learning rates for all parameter groups
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group['name']
                _lr = param_group['lr']
                metrics[f"lr_{group_name}"].clear()
                metrics.update(name=f"lr_{group_name}", value=_lr)
            # Add gradient norm metrics
            # if 'backbone_grad_norm' in locals():
            metrics.update(name="backbone_grad_norm", value=backbone_grad_norm.detach())
            for head_name, head_grad_norm in head_grad_norms.items():
                metrics.update(name=f"{head_name}_head_grad_norm", value=head_grad_norm.detach())
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            metrics["max_cuda_mem(MB)"].clear()
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            metrics.sync()
            eta = tps.eta(total_steps=max_iterations, current_steps=cur_iter)
            eta = TPS.format(eta)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{cur_iter}/{max_iterations}] "
                    f"[tps: {tps.average:.2f}s] [eta: {eta}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        
        # 清理当前iteration的变量，防止显存累积
        # del weighted_loss_dict, unweighted_loss_dict, log_only_loss_dict
        # 定期清理CUDA缓存
        # if (cur_iter + 1) % (logging_interval * 2) == 0:
        #     torch.cuda.empty_cache()
        
    states["start_epoch"] += 1
    time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
    # Log epoch-level metrics
    if logger:
        epoch_avg_metrics = epoch_metrics.get_averages()
        
        # 计算epoch级别的overall losses
        epoch_weighted_total = 0.0
        epoch_unweighted_total = 0.0
        
        for key, value in epoch_avg_metrics.items():
            if key.startswith("weighted_") and not key.startswith("weighted_total"):
                epoch_weighted_total += value
            elif key.startswith("unweighted_") and not key.startswith("unweighted_total"):
                epoch_unweighted_total += value
        
        # Log epoch overall metrics
        logger.log_scalar("epoch_overall/weighted_total_loss", epoch_weighted_total, epoch)
        logger.log_scalar("epoch_overall/unweighted_total_loss", epoch_unweighted_total, epoch)
        
        # Log epoch task-specific metrics
        task_weighted_totals = {}
        task_unweighted_totals = {}
        
        for key, value in epoch_avg_metrics.items():
            if key.startswith("weighted_"):
                # Extract task name from key like "weighted_TaskName_loss_type"
                parts = key.split("_", 2)  # Split into ["weighted", "TaskName", "loss_type"]
                if len(parts) >= 3:
                    head_name = parts[1]
                    if head_name not in task_weighted_totals:
                        task_weighted_totals[head_name] = 0.0
                    task_weighted_totals[head_name] += value
            elif key.startswith("unweighted_"):
                # Extract task name from key like "unweighted_TaskName_loss_type"
                parts = key.split("_", 2)  # Split into ["unweighted", "TaskName", "loss_type"]
                if len(parts) >= 3:
                    head_name = parts[1]
                    if head_name not in task_unweighted_totals:
                        task_unweighted_totals[head_name] = 0.0
                    task_unweighted_totals[head_name] += value
        
        # Log task totals
        for head_name, total_loss in task_weighted_totals.items():
            logger.log_scalar(f"epoch_weighted_{head_name}/total_loss", total_loss, epoch)
        
        for head_name, total_loss in task_unweighted_totals.items():
            logger.log_scalar(f"epoch_unweighted_{head_name}/total_loss", total_loss, epoch)
        
        # Flush logger at the end of epoch
        logger.flush_tb_writer()
        logger.info(f"Epoch {epoch} completed. Time per epoch: {time_per_epoch}")

def lr_warmup_multi_groups(optimizer, epoch: int, curr_iter: int, warmup_epochs: int, num_iter_per_epoch: int):
    """
    Learning rate warmup for multiple parameter groups with different target learning rates.
    Each parameter group's initial learning rate is used as the target learning rate.
    """
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    
    for param_group in optimizer.param_groups:
        # Use the group's initial learning rate as the target learning rate
        if 'initial_lr' not in param_group:
            # Store the initial learning rate on first call
            param_group['initial_lr'] = param_group['lr']
        
        tgt_lr = param_group['initial_lr']
        current_lr = tgt_lr * current_lr_ratio
        
        if "lr_scale" in param_group:
            param_group["lr"] = current_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = current_lr
    return

def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    # min_lr = 1e-8
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return
    
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

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

    # Call the "train_engine" function:
    train_engine(config=cfg)