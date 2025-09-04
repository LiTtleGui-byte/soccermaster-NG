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
from models.yolo import YOLOModel
from utils.misc import set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn, build_metrics_fn
from train import evaluate_one_epoch

def save_results_with_logger(eval_results: dict, logger: Logger):
    """
    将evaluation结果保存到log.txt文件中
    
    Args:
        eval_results: 包含evaluation结果的字典
        logger: Logger实例
    """
    logger.info("=== Evaluation Results Summary ===")
    
    # 保存head_metrics_results
    if "head_metrics_results" in eval_results:
        logger.info("--- Head Metrics Results ---")
        head_metrics = eval_results["head_metrics_results"]
        
        if head_metrics:
            for head_name, metrics_dict in head_metrics.items():
                if metrics_dict:  # 确保metrics_dict不为空
                    logger.info(f"{head_name} Metrics:")
                    for metric_name, metric_value in metrics_dict.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"{head_name}: No metrics available")
        else:
            logger.info("No head metrics results available")
    
    # # 保存head_weighted_results
    # if "head_weighted_results" in eval_results:
    #     logger.info("--- Head Weighted Loss Results ---")
    #     head_weighted = eval_results["head_weighted_results"]
        
    #     if head_weighted:
    #         for head_name, loss_dict in head_weighted.items():
    #             if loss_dict:  # 确保loss_dict不为空
    #                 logger.info(f"{head_name} Weighted Losses:")
    #                 for loss_name, loss_value in loss_dict.items():
    #                     logger.info(f"  {loss_name}: {loss_value:.4f}")
    #             else:
    #                 logger.info(f"{head_name}: No weighted losses available")
    #     else:
    #         logger.info("No head weighted results available")
    
    # # 保存head_unweighted_results
    # if "head_unweighted_results" in eval_results:
    #     logger.info("--- Head Unweighted Loss Results ---")
    #     head_unweighted = eval_results["head_unweighted_results"]
        
    #     if head_unweighted:
    #         for head_name, loss_dict in head_unweighted.items():
    #             if loss_dict:  # 确保loss_dict不为空
    #                 logger.info(f"{head_name} Unweighted Losses:")
    #                 for loss_name, loss_value in loss_dict.items():
    #                     logger.info(f"  {loss_name}: {loss_value:.4f}")
    #             else:
    #                 logger.info(f"{head_name}: No unweighted losses available")
    #     else:
    #         logger.info("No head unweighted results available")
    
    # # 保存head_log_only_results
    # if "head_log_only_results" in eval_results:
    #     logger.info("--- Head Log-Only Results ---")
    #     head_log_only = eval_results["head_log_only_results"]
        
    #     if head_log_only:
    #         for head_name, log_dict in head_log_only.items():
    #             if log_dict:  # 确保log_dict不为空
    #                 logger.info(f"{head_name} Log-Only Values:")
    #                 for log_name, log_value in log_dict.items():
    #                     logger.info(f"  {log_name}: {log_value:.4f}")
    #             else:
    #                 logger.info(f"{head_name}: No log-only values available")
    #     else:
    #         logger.info("No head log-only results available")
    
    # # 保存head_sample_counts
    # if "head_sample_counts" in eval_results:
    #     logger.info("--- Head Sample Counts ---")
    #     head_counts = eval_results["head_sample_counts"]
        
    #     if head_counts:
    #         for head_name, count in head_counts.items():
    #             logger.info(f"{head_name}: {count} samples")
    #     else:
    #         logger.info("No head sample counts available")
    
    # # 保存overall results
    # logger.info("--- Overall Results ---")
    # if "overall_weighted_loss" in eval_results:
    #     logger.info(f"Overall Weighted Loss: {eval_results['overall_weighted_loss']:.4f}")
    # if "overall_unweighted_loss" in eval_results:
    #     logger.info(f"Overall Unweighted Loss: {eval_results['overall_unweighted_loss']:.4f}")
    
    logger.info("=== End of Evaluation Results ===")

def evaluation_engine(config: dict, checkpoint_path: str, log_dir: str = None, 
                     save_video_caption_failures: bool = False, failure_save_path: str = None):
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
    if config["MODEL_ARCH"] == "multitask":
        model = MultiTaskingSigLIP(config=config, logger=logger)
    elif config["MODEL_ARCH"] == "yolo":
        model = YOLOModel(config=config, logger=logger)
    else:
        raise ValueError(f"Invalid model architecture: {config['MODEL_ARCH']}")
    
    # Load checkpoint
    if config["MODEL_ARCH"] == "multitask":
        model.load_checkpoint(checkpoint_path)
    
    # Prepare model and dataloaders
    model = accelerator.prepare(model)
    dataloader_test_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_test_dict.items()}
    
    if config["MODEL_ARCH"] == "multitask":
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
        logger=logger,  # 传入logger用于记录evaluation过程
        save_video_caption_failures=save_video_caption_failures,
        failure_save_path=failure_save_path
    )
    logger.info("Evaluation completed!")
    
    # Format and print results (只在主进程输出)
    if accelerator.is_main_process:
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
    parser.add_argument('--save_video_caption_failures', action='store_true',
                       help='Save video caption failure cases to file')
    parser.add_argument('--failure_save_path', type=str, default=None,
                       help='Path to save video caption failure cases (required if --save_video_caption_failures is used)')
    
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
    
    # Set failure save path
    failure_save_path = args.failure_save_path
    if args.save_video_caption_failures and failure_save_path is None:
        failure_save_path = f"video_caption_failures_{os.path.basename(args.checkpoint)}.txt"
    
    # Run evaluation
    try:
        eval_results = evaluation_engine(
            config=cfg,
            checkpoint_path=args.checkpoint,
            log_dir=args.log_dir,
            save_video_caption_failures=args.save_video_caption_failures,
            failure_save_path=failure_save_path
        )
        print("\n✅ Evaluation completed successfully!")
        
        if args.save_video_caption_failures:
            print(f"📝 Video caption failure cases saved to: {failure_save_path}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 