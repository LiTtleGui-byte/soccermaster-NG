# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs
from collections import defaultdict
from itertools import cycle

from data.build import build_dataloader
from utils.logger import Logger, MetricsTracker, TPS, Metrics
from models.multi_task import MultiTaskingSigLIP
from runtime_option import runtime_option
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    # accelerator = Accelerator()
    # accelerator = Accelerator(
    #     kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    # )
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)]
    )
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
        use_tensorboard=config.get("USE_TENSORBOARD", False),
        tensorboard_flush_secs=config.get("TENSORBOARD_FLUSH_SECS", 30)
    )
    logger.config(config=config)
    
    # Build training dataset:
    dataloader_train_dict, dataloader_test_dict = build_dataloader(config=config)
    
    # Build loss functions:
    loss_fn_dict = build_loss_fn(config=config)
    
    # num_tasks = get_world_size()
    # global_rank = get_rank()
    # sampler_train = DistributedBatchTaskBalancedSampler(dataset_train_dict, config["BATCH_SIZE"], num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    model = MultiTaskingSigLIP(config=config)
    
    # TODO: set params groups
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )
    
    model, optimizer = accelerator.prepare(model, optimizer)
    dataloader_train_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_train_dict.items()}
    dataloader_test_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_test_dict.items()}
    
    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,} ({100 - trainable_percentage:.2f}%)")
    
    # Print names of non-trainable parameters
    logger.info("Non-trainable layers:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.info(f"  {name}")
    
    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        epoch_start_timestamp = TPS.timestamp()
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
            lr_warmup_tgt_lr=config["LR"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            logging_interval=config.get("LOGGING_INTERVAL", 20),
        )
        
        scheduler.step()
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
    
    # Close logger at the end of training
    if logger:
        logger.close_tb_writer()
        logger.info("Training completed. TensorBoard logger closed.")
    
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
        logger: Logger = None,
        lr_warmup_epochs: int = 0,
        lr_warmup_tgt_lr: float = 1e-4,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
):
    current_last_checkpoint_idx = 0
    model.train()
    tps = TPS()
    metrics = Metrics()
    step_timestamp = tps.timestamp()
    optimizer.zero_grad()
    device = accelerator.device
    
    assert accumulate_steps == 1, "accumulate_steps must be 1 for now."
    
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
    dataloader_counters = {task: 0 for task in dataloader_dict.keys()}
    logger.info(f"Dataloader lengths: {dataloader_lengths}")

    for cur_iter in range(max_iterations):
        loss_dict = {}  # 用于backward的加权loss
        unweighted_loss_dict = {}  # 记录加权前的原始loss
        log_only_loss_dict = {}
        
        # 逐任务处理，每个任务计算完立即backward以减少显存占用
        for task, dataloader_iter in dataloader_iters.items():
            with accelerator.autocast():
                # batch = next(dataloader)
                try:
                    batch = next(dataloader_iter)
                    dataloader_counters[task] += 1
                except StopIteration:
                    # Reset the dataloader iterator and counter when exhausted
                    logger.info(f"Task {task} dataloader exhausted at iteration {cur_iter}, resetting...")
                    dataloader_iters[task] = iter(dataloader_dict[task])
                    dataloader_counters[task] = 1  # Reset counter to 1 (current batch)
                    batch = next(dataloader_iters[task])
                    
                images, annotations, metas = batch.values()
                
                # Learning rate warmup:
                if epoch < lr_warmup_epochs:
                    # Do warmup:
                    lr_warmup(
                        optimizer=optimizer,
                        epoch=epoch, curr_iter=cur_iter, tgt_lr=lr_warmup_tgt_lr,
                        warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=max_iterations,
                    )
                
                # 可选：使用梯度检查点进一步减少显存（会增加计算时间）
                if config.get("USE_GRADIENT_CHECKPOINTING", False):
                    outputs = torch.utils.checkpoint.checkpoint(model, images, task, use_reentrant=False)
                else:
                    outputs = model(images, task)
                    
                loss_output = loss_fn_dict[task](outputs[task], annotations)
                if task in ["SoccerNetGSR_Detection", ]:
                    loss_task_raw, weight_dict, _ = loss_output
                    unweighted_loss_dict.update({f"{task}_{k}": v for k, v in loss_task_raw.items() if k in weight_dict})
                    loss_task = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                    log_only_loss_dict.update({f"{task}_{k}": v for k, v in loss_task_raw.items() if k not in weight_dict})
                elif task in ["SoccerNetGSR_ReID"]:
                    loss_task_raw, weight_dict = loss_output
                    unweighted_loss_dict.update({f"{task}_{k}": v for k, v in loss_task_raw.items() if k in weight_dict})
                    loss_task = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                    log_only_loss_dict.update({f"{task}_{k}": v for k, v in loss_task_raw.items() if k not in weight_dict})
                else:
                    loss_task = loss_output
                    # 对于其他任务，未加权和加权的loss相同
                    unweighted_loss_dict.update({f"{task}_{k}": v for k, v in loss_task.items()})
                
                loss_dict.update({f"{task}_{k}": v for k, v in loss_task.items()})
            
            # 立即对当前任务进行backward，减少显存占用
            task_total_loss = sum(loss_task.values())
            task_total_loss /= (accumulate_steps * len_tasks)  # 除以任务数量进行平均
            accelerator.backward(task_total_loss)
            
            # 显式释放当前任务的中间变量，防止显存泄漏
            # del images, annotations, metas, outputs, loss_output
            # if 'loss_task_raw' in locals():
            #     del loss_task_raw
            # del loss_task, task_total_loss
            
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
                for task_name, head in original_model.multi_task_head.items():
                    head_grad_norms[task_name] = accelerator.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
                # For logging purposes, we can use the backbone grad norm as the main grad_norm
                grad_norm = backbone_grad_norm
            else:
                accelerator.unscale_gradients()
                # Clip gradients separately for backbone and each head
                # Use model.module to access original model attributes when using DDP
                original_model = model.module if hasattr(model, 'module') else model
                backbone_grad_norm = torch.nn.utils.clip_grad_norm_(original_model.backbone.parameters(), max_clip_norm)
                head_grad_norms = {}
                for task_name, head in original_model.multi_task_head.items():
                    head_grad_norms[task_name] = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
                # For logging purposes, we can use the backbone grad norm as the main grad_norm
                grad_norm = backbone_grad_norm
            optimizer.step()
            optimizer.zero_grad()
                
        # Update global step
        states["global_step"] += 1
        tps.update(TPS.timestamp() - step_timestamp)
        step_timestamp = TPS.timestamp()
        
        # Add logging
        if logger and (cur_iter + 1) % logging_interval == 0:
            logger.log_loss_dict(loss_dict, states["global_step"], prefix="train_weighted")
            logger.log_loss_dict(unweighted_loss_dict, states["global_step"], prefix="train_unweighted")
            logger.log_loss_dict(log_only_loss_dict, states["global_step"], prefix="train_unweighted", count_sum=False)
            logger.log_learning_rate(optimizer, states["global_step"])
            # Log gradient norm
            # logger.log_scalar("train/grad_norm", grad_norm, states["global_step"])
            # Log separate gradient norms for backbone and each head
            logger.log_scalar("train_grad_norm/backbone_grad_norm", backbone_grad_norm, states["global_step"])
            for task_name, head_grad_norm in head_grad_norms.items():
                logger.log_scalar(f"train_grad_norm/{task_name}_head_grad_norm", head_grad_norm, states["global_step"])
            
            # Log dataloader progress for each task
            # for task_name, counter in dataloader_counters.items():
            #     progress_ratio = counter / dataloader_lengths[task_name]
            #     reset_count = counter // dataloader_lengths[task_name]
            #     logger.log_scalar(f"dataloader_progress/{task_name}_progress", progress_ratio, states["global_step"])
            #     logger.log_scalar(f"dataloader_progress/{task_name}_resets", reset_count, states["global_step"])
            
            # Log parameter and gradient statistics if enabled
            if config.get("LOG_PARAMS_GRADS", False):
                logger.log_model_parameters(model, states["global_step"])
            
            # Print progress
            total_loss = sum(loss_dict.values())
            total_unweighted_loss = sum(unweighted_loss_dict.values())
            
            # Update metrics tracker (epoch metrics) - 记录所有类型的loss
            epoch_metrics.update(loss_dict)
            epoch_metrics.update({f"unweighted_{k}": v for k, v in unweighted_loss_dict.items()})
            epoch_metrics.update(log_only_loss_dict)

            
            # Update metrics (iteration metrics)
            metrics.update(name="weighted_total_loss", value=total_loss.detach())
            metrics.update(name="unweighted_total_loss", value=total_unweighted_loss.detach())
            for k, v in loss_dict.items():
                metrics.update(name=f"weighted_{k}", value=v.detach())
            for k, v in unweighted_loss_dict.items():
                metrics.update(name=f"unweighted_{k}", value=v.detach())
            for k, v in log_only_loss_dict.items():
                metrics.update(name=k, value=v.detach())
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            metrics["lr"].clear()
            metrics.update(name="lr", value=_lr)
            # Add gradient norm metrics
            if 'backbone_grad_norm' in locals():
                metrics.update(name="backbone_grad_norm", value=backbone_grad_norm.detach())
                for task_name, head_grad_norm in head_grad_norms.items():
                    metrics.update(name=f"{task_name}_head_grad_norm", value=head_grad_norm.detach())
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
        # del loss_dict, unweighted_loss_dict, log_only_loss_dict
        # 定期清理CUDA缓存
        if (cur_iter + 1) % (logging_interval * 2) == 0:
            torch.cuda.empty_cache()
        
    states["start_epoch"] += 1
    # Log epoch-level metrics
    if logger:
        epoch_avg_metrics = epoch_metrics.get_averages()
        for key, value in epoch_avg_metrics.items():
            logger.log_scalar(f"epoch/{key}", value, epoch)
        
        # Flush logger at the end of epoch
        logger.flush_tb_writer()
        logger.info(f"Epoch {epoch} completed. Average losses: {epoch_avg_metrics}")
       
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