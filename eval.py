# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Evaluation script for trained checkpoints
# ------------------------------------------------------------------------
import torch
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs

from data.build import build_dataloader
from utils.logger import Logger, MetricsTracker
from models.multi_task import MultiTaskingSigLIP
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn, build_metrics_fn
from train import evaluate_one_epoch


def load_checkpoint(model, checkpoint_path, device):
    """
    加载checkpoint到模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 检查是否是目录结构的checkpoint（包含backbone和各个head）
    if os.path.isdir(checkpoint_path):
        # 加载backbone
        backbone_path = os.path.join(checkpoint_path, 'backbone')
        if os.path.exists(backbone_path):
            print(f"Loading backbone from: {backbone_path}")
            model.backbone.vision_model = model.backbone.vision_model.from_pretrained(backbone_path)
        
        # 加载各个task head
        for task_name in model.multi_task_head.keys():
            head_path = os.path.join(checkpoint_path, f'{task_name}.pt')
            if os.path.exists(head_path):
                print(f"Loading {task_name} head from: {head_path}")
                head_state_dict = torch.load(head_path, map_location=device)
                model.multi_task_head[task_name].load_state_dict(head_state_dict)
            else:
                print(f"Warning: {task_name} head checkpoint not found at {head_path}")
    else:
        # 加载单个checkpoint文件
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully!")
    return model


def format_results(results):
    """
    格式化evaluation结果为易读的字符串
    """
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("EVALUATION RESULTS")
    output_lines.append("="*80)
    
    # 整体结果
    output_lines.append(f"\nOVERALL RESULTS:")
    output_lines.append(f"  Overall Weighted Loss: {results['overall_weighted_loss']:.4f}")
    output_lines.append(f"  Overall Unweighted Loss: {results['overall_unweighted_loss']:.4f}")
    output_lines.append(f"  Total Samples: {sum(results['task_sample_counts'].values())}")
    
    # 每个任务的结果
    for task_name in results['task_sample_counts'].keys():
        output_lines.append(f"\n{task_name.upper()} RESULTS:")
        output_lines.append(f"  Sample Count: {results['task_sample_counts'][task_name]}")
        
        # Loss results
        weighted_results = results['task_weighted_results'][task_name]
        unweighted_results = results['task_unweighted_results'][task_name]
        metrics_results = results['task_metrics_results'][task_name]
        
        if weighted_results:
            output_lines.append(f"  Weighted Losses:")
            for metric_name, value in weighted_results.items():
                output_lines.append(f"    {metric_name}: {value:.4f}")
            total_weighted = sum(weighted_results.values())
            output_lines.append(f"    Total Weighted Loss: {total_weighted:.4f}")
        
        if unweighted_results:
            output_lines.append(f"  Unweighted Losses:")
            for metric_name, value in unweighted_results.items():
                output_lines.append(f"    {metric_name}: {value:.4f}")
            total_unweighted = sum(unweighted_results.values())
            output_lines.append(f"    Total Unweighted Loss: {total_unweighted:.4f}")
        
        # Evaluation metrics (mAP, precision, recall等)
        if metrics_results:
            output_lines.append(f"  Evaluation Metrics:")
            
            # 按重要性排序metrics
            important_metrics = ['mAP', 'mAP@0.5', 'mAP@0.75', 'precision', 'recall', 'f1']
            
            # 先显示重要metrics
            for metric_name in important_metrics:
                if metric_name in metrics_results:
                    value = metrics_results[metric_name]
                    output_lines.append(f"    {metric_name}: {value:.4f}")
            
            # 然后显示其他metrics
            for metric_name, value in metrics_results.items():
                if metric_name not in important_metrics:
                    output_lines.append(f"    {metric_name}: {value:.4f}")
    
    output_lines.append("="*80)
    return "\n".join(output_lines)


def save_results_with_logger(results, logger):
    """
    使用Logger将结果保存到文件
    """
    formatted_results = format_results(results)
    
    # 保存格式化的文本结果
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger._save(f"Evaluation Results - {timestamp}", filename="eval_results.txt", mode="w")
    
    # 将格式化结果按行保存
    for line in formatted_results.split('\n'):
        logger._save(line, filename="eval_results.txt", mode="a")
    
    # 保存详细的JSON结果
    # logger._write_dict_to_json(results, filename="eval_results.json", mode="w")
    
    logger.success(f"Results saved to: {logger.log_dir}/eval_results.txt and eval_results.json")


def evaluation_engine(config: dict, checkpoint_path: str, log_dir: str = None):
    """
    主要的evaluation引擎
    """
    # Init Accelerator
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    state = PartialState()
    
    # Set seed
    set_seed(config["SEED"])
    
    # Set sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 设置默认log_dir
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
        log_dir = f"eval_logs_{checkpoint_name}_{timestamp}"
    
    # Init Logger
    logger = Logger(
        log_dir=log_dir,
        accelerator=accelerator,
        config=config,
        use_tensorboard=False,
        tensorboard_flush_secs=30
    )
    
    # Build test dataloader only
    _, dataloader_test_dict = build_dataloader(config=config, only_test=True)
    
    # Filter out None test dataloaders
    dataloader_test_dict = {task: dataloader for task, dataloader in dataloader_test_dict.items() 
                           if dataloader is not None}
    
    if not dataloader_test_dict:
        raise ValueError("No test datasets available for evaluation!")
    
    logger.info(f"Test datasets available for tasks: {list(dataloader_test_dict.keys())}")
    
    # Build loss and metrics functions
    loss_fn_dict = build_loss_fn(config=config)
    metrics_fn_dict = build_metrics_fn(config=config)
    
    # Build model
    model = MultiTaskingSigLIP(config=config)
    
    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path, accelerator.device)
    
    # Prepare model and dataloaders
    model = accelerator.prepare(model)
    dataloader_test_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_test_dict.items()}
    
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
    
    for task_name, head in original_model.multi_task_head.items():
        head_total = sum(p.numel() for p in head.parameters())
        head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
        head_params[task_name] = head_total
        head_trainable_params[task_name] = head_train
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
    for task_name in head_params:
        logger.info(f"{task_name} Head parameters: {head_params[task_name]/1e6:.2f}M")
        logger.info(f"{task_name} Head trainable: {head_trainable_params[task_name]/1e6:.2f}M ({head_trainable_params[task_name]/head_params[task_name]*100:.2f}%)")
    logger.info(f"============================================")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    eval_results = evaluate_one_epoch(
        config=config,
        accelerator=accelerator,
        epoch=0,  # 设置为0，因为这是单独的evaluation
        dataloader_dict=dataloader_test_dict,
        loss_fn_dict=loss_fn_dict,
        metrics_fn_dict=metrics_fn_dict,
        model=model,
        logger=logger  # 传入logger用于记录evaluation过程
    )
    logger.info("Evaluation completed!")
    
    # Format and print results (只在主进程输出)
    if accelerator.is_main_process:
        formatted_results = format_results(eval_results)
        print(formatted_results)
        
        # Save results using logger
        save_results_with_logger(eval_results, logger)
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    return eval_results


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Evaluate trained model checkpoint')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Log directory path for results (default: auto-generated)')
    parser.add_argument('--super_config', type=str, default=None,
                       help='Path to super config file')
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Parse arguments
    args = parse_args()
    
    # Load config
    cfg = yaml_to_dict(args.config)
    
    # Load super config
    if args.super_config is not None:
        cfg = load_super_config(cfg, args.super_config)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    # Update config with command line args
    cfg["CHECKPOINT_PATH"] = args.checkpoint
    if args.log_dir:
        cfg["LOG_DIR"] = args.log_dir
    
    # Run evaluation
    try:
        eval_results = evaluation_engine(
            config=cfg,
            checkpoint_path=args.checkpoint,
            log_dir=args.log_dir
        )
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 