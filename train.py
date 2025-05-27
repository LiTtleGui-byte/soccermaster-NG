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
from data.sampler import DistributedBatchTaskBalancedSampler
from utils import get_world_size, get_rank
from models.multi_task import MultiTaskingSigLIP
from runtime_option import runtime_option
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn

# TODO: add logger (tensorboard)

def train_engine(config: dict):
    # Init some settings:
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # Init Accelerator at beginning:
    # accelerator = Accelerator()
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    state = PartialState()
    # Also, we set the seed:
    set_seed(config["SEED"])
    
    # from issue: https://github.com/pytorch/pytorch/issues/11201
    # Set the sharing strategy (to avoid error: too many open files):
    torch.multiprocessing.set_sharing_strategy('file_system')   # if not, raise error: too many open files.
    # torch.autograd.set_detect_anomaly(True)
    
    # Init Logger:
    # TODO: finish logger code
    
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
    dataloader_train_dict.values() = accelerator.prepare(dataloader_train_dict.values())
    dataloader_test_dict.values() = accelerator.prepare(dataloader_test_dict.values())
    
    # Init the training states:
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }
    
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
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            separate_clip_norm=config.get("SEPARATE_CLIP_NORM", True),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            use_accelerate_clip_norm=config.get("USE_ACCELERATE_CLIP_NORM", True),
            logging_interval=config.get("LOGGING_INTERVAL", 20),
        )
        
        scheduler.step()
    
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
        lr_warmup_epochs: int,
        lr_warmup_tgt_lr: float,
        accumulate_steps: int = 1,
        separate_clip_norm: bool = True,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        logging_interval: int = 20,
):
    current_last_checkpoint_idx = 0
    model.train()
    optimizer.zero_grad()
    device = accelerator.device
    
    max_iterations = max(len(dataloader) for dataloader in dataloader_dict.values())
    
    # fetch data until one of the dataloaders is exhausted:
    # for iteration in range(max_iterations):
    #     is_exhausted = False
    #     for task, dataloader in dataloader_dict.items():
    #         try:
    #             batch = next(dataloader)
    #         except StopIteration:
    #             is_exhausted = True
    #             break
    #     if is_exhausted:
    #         break
        
    # fetch data until reaching max iterations
    cycle_dataloader_dict = {task: cycle(dataloader) for task, dataloader in dataloader_dict.items() }
    for cur_iter in range(max_iterations):
        loss_dict = {}
        for task, dataloader in cycle_dataloader_dict.items():
            batch = next(dataloader)
            images, annotations, metas = batch
            assert task == metas["task"]
            
            # Learning rate warmup:
            if epoch < lr_warmup_epochs:
                # Do warmup:
                lr_warmup(
                    optimizer=optimizer,
                    epoch=epoch, curr_iter=cur_iter, tgt_lr=lr_warmup_tgt_lr,
                    warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=max_iterations,
                )
                
            # TODO: prepare DETR annotations to GT

            # forward pass:
            outputs = model(images, task)
            
            # TODO: compute loss
            loss_task = loss_fn_dict[task](outputs, annotations)
            loss_dict.update({f"{task}_{k}": v for k, v in loss_task.items()})
            
        # backward pass:
                # Backward:
        with accelerator.autocast():
            loss = sum(loss_dict.values())
            loss /= accumulate_steps
            accelerator.backward(loss)
            
            if (cur_iter + 1) % accumulate_steps == 0:
                if use_accelerate_clip_norm:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
                else:
                    accelerator.unscale_gradients()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
        
                # update optimizer:
                optimizer.step()
                optimizer.zero_grad()
        
       
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
    # train_engine(config=cfg)