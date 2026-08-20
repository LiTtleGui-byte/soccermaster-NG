# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Visualization script for DETR model predictions
# ------------------------------------------------------------------------
import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs
import random

from soccermaster.data.build import build_dataloader
from soccermaster.utils.logger import Logger
from soccermaster.models.multi_task import MultiTaskingSigLIP
from soccermaster.models.deformable_detr.deformable_detr import PostProcess
from soccermaster.utils.misc import set_seed
from soccermaster.config import load_super_config, update_config, yaml_to_dict
from soccermaster.data.soccernet_gsr_reid import role_mapping, jn_mapping

# def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
def denormalize_image(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    反归一化图像张量到[0,1]范围
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0)
    
    # 转换为numpy格式 (H, W, C)
    if image_tensor.shape[0] == 3:  # (C, H, W)
        image_tensor = image_tensor.permute(1, 2, 0)
    
    image = image_tensor.cpu().numpy()
    
    # 反归一化
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    return image


def draw_boxes_on_image(image, boxes, labels, scores, class_names, color=(1, 0, 0), thickness=2):
    """
    在图像上绘制边界框
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # 绘制边界框
        rect = Rectangle((x1, y1), width, height, linewidth=thickness, 
                        edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标签和分数
        class_name = class_names.get(int(label), f"class_{int(label)}")
        text = f"{class_name}: {score:.2f}"
        ax.text(x1, y1-10, text, fontsize=10, color=color, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.axis('off')
    return fig, ax


def visualize_predictions_and_gt(image, pred_results, gt_annotations, class_names, score_threshold=0.5):
    """
    可视化预测结果和真实标注
    """
    # 创建subplot: 左边是预测，右边是真实标注
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左边：预测结果
    ax1.imshow(image)
    ax1.set_title("预测结果 (Predictions)", fontsize=16, fontweight='bold')
    ax1.set_xlim(0, image.shape[1])
    ax1.set_ylim(image.shape[0], 0)
    
    if pred_results and len(pred_results['boxes']) > 0:
        # 过滤低分预测
        valid_mask = pred_results['scores'] > score_threshold
        pred_boxes = pred_results['boxes'][valid_mask]
        pred_labels = pred_results['labels'][valid_mask]
        pred_scores = pred_results['scores'][valid_mask]
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box.cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            
            # 绘制预测框 (红色)
            rect = Rectangle((x1, y1), width, height, linewidth=2, 
                           edgecolor='red', facecolor='none')
            ax1.add_patch(rect)
            
            # 添加标签
            class_name = class_names.get(int(label.cpu().item()), f"class_{int(label.cpu().item())}")
            text = f"{class_name}: {score.cpu().item():.2f}"
            ax1.text(x1, y1-10, text, fontsize=10, color='red', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 右边：真实标注
    ax2.imshow(image)
    ax2.set_title("真实标注 (Ground Truth)", fontsize=16, fontweight='bold')
    ax2.set_xlim(0, image.shape[1])
    ax2.set_ylim(image.shape[0], 0)
    
    if len(gt_annotations['boxes']) > 0:
        # 转换gt boxes到像素坐标
        gt_boxes = gt_annotations['boxes'].cpu().numpy()
        gt_labels = gt_annotations['labels'].cpu().numpy()
        
        # 假设gt_boxes是相对坐标(0-1)，需要转换为绝对坐标
        if gt_boxes.max() <= 1.0:
            h, w = image.shape[:2]
            # 假设是cxcywh格式，转换为xyxy
            if gt_boxes.shape[1] == 4:
                # 从cxcywh转换为xyxy
                gt_boxes_xyxy = np.zeros_like(gt_boxes)
                gt_boxes_xyxy[:, 0] = (gt_boxes[:, 0] - gt_boxes[:, 2]/2) * w  # x1
                gt_boxes_xyxy[:, 1] = (gt_boxes[:, 1] - gt_boxes[:, 3]/2) * h  # y1
                gt_boxes_xyxy[:, 2] = (gt_boxes[:, 0] + gt_boxes[:, 2]/2) * w  # x2
                gt_boxes_xyxy[:, 3] = (gt_boxes[:, 1] + gt_boxes[:, 3]/2) * h  # y2
                gt_boxes = gt_boxes_xyxy
        
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # 绘制真实框 (绿色)
            rect = Rectangle((x1, y1), width, height, linewidth=2, 
                           edgecolor='green', facecolor='none')
            ax2.add_patch(rect)
            
            # 添加标签
            class_name = class_names.get(int(label), f"class_{int(label)}")
            ax2.text(x1, y1-10, class_name, fontsize=10, color='green', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    
    return fig


# def visualize_engine(config: dict, checkpoint_path: str, output_dir: str, num_samples: int = 10, score_threshold: float = 0.5):
def visualize_engine(config: dict, output_dir: str, num_samples: int = 10, score_threshold: float = 0.5):
    """
    主要的可视化引擎
    """
    # Init Accelerator
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    
    # Set seed
    set_seed(config["SEED"])
    
    # Set sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"可视化结果将保存到: {output_dir}")
    
    # Build dataloaders
    dataloader_train_dict, dataloader_test_dict = build_dataloader(config=config)
    
    # 确保有SoccerNetGSR_Detection任务
    task_name = "SoccerNetGSR_Detection"
    if task_name not in dataloader_train_dict:
        raise ValueError(f"任务 {task_name} 在训练集中不存在!")
    
    print(f"可用的训练任务: {list(dataloader_train_dict.keys())}")
    if dataloader_test_dict:
        print(f"可用的测试任务: {list(dataloader_test_dict.keys())}")
    
    # Build model
    model = MultiTaskingSigLIP(config=config)
    if config["LOAD_CHECKPOINTS"]:
        model.load_checkpoint(config["STAGE_1_CKPT_DIR"])
    
    # Prepare model
    model = accelerator.prepare(model)
    model.eval()
    
    # Build postprocessor
    postprocessor = PostProcess()
    
    # 定义类别名称（根据SoccerNetGSR数据集的role_mapping）
    # 从data/SoccerNetGSR_ReID.py中的role_mapping获取
    # class_names = {0: "ball", 1: "goalkeeper", 2: "other", 3: "player", 4: "referee"}
    # class_names = {v: k for k, v in role_mapping.items()}
    # class_names = {0: "person", 1: "other"}
    class_names = {0: "person", 1: "ball", 2: "other"}
    
    print(f"开始从数据集中抽样...")
    
    # Prepare dataloaders with accelerator
    train_dataloader = accelerator.prepare(dataloader_train_dict[task_name])
    print(f"从训练集抽样 {num_samples} 个样本...")
    visualize_samples_from_dataloader(
        model, train_dataloader, postprocessor, class_names,
        output_dir, "train", num_samples, score_threshold, accelerator
    )
    
    # 从测试集抽样（如果存在）
    if dataloader_test_dict and task_name in dataloader_test_dict:
        test_dataloader = accelerator.prepare(dataloader_test_dict[task_name])
        print(f"从测试集抽样 {num_samples} 个样本...")
        visualize_samples_from_dataloader(
            model, test_dataloader, postprocessor, class_names,
            output_dir, "test", num_samples, score_threshold, accelerator
        )
    else:
        print("测试集不可用，跳过测试集可视化")
    
    print(f"可视化完成! 结果保存在: {output_dir}")


def visualize_samples_from_dataloader(model, dataloader, postprocessor, class_names, 
                                   output_dir, split_name, num_samples, score_threshold, accelerator):
    """
    从数据加载器中抽样并可视化 - 优化版，直接选择需要的样本
    """
    device = accelerator.device
    task_name = "SoccerNetGSR_Detection"
    
    # 直接获取原始数据集
    original_dataset = dataloader.dataset
    total_samples = len(original_dataset)
    
    # 随机选择要可视化的样本索引
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # 创建只包含选定索引的子数据集
    subset_dataset = torch.utils.data.Subset(original_dataset, indices)
    
    # 创建一个新的DataLoader，只加载选定的样本
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=1,  # 每个样本单独作为一个batch
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory
    )
    subset_loader = accelerator.prepare(subset_loader)
    
    sample_count = 0
    global_indices = indices  # 保存全局索引用于文件名
    
    with torch.no_grad():
        # 现在只遍历选定的样本
        for batch_idx, batch in enumerate(subset_loader):
            # 获取当前样本的全局索引
            sample_global_idx = global_indices[batch_idx]
            
            # 提取单个样本 (batch_size=1)
            images, annotations, metas = batch.values()
            sample_image = images  # [1, C, H, W]
            sample_annotations = {
                'boxes': annotations[0]['boxes'],
                'labels': annotations[0]['labels']
            }
            sample_metas = metas[0]
            
            # 确保数据在正确的设备上
            sample_image = sample_image.to(device)
            
            # 前向推理
            with accelerator.autocast():
                outputs = model(sample_image, task_name, sample_metas)
            
            # 后处理预测结果
            model_outputs = outputs[task_name]
            
            # 获取target_sizes用于后处理
            if 'target_sizes' in sample_metas:
                target_sizes = sample_metas['target_sizes']
            else:
                # 使用图像原始尺寸
                _, _, h, w = sample_image.shape
                target_sizes = torch.tensor([[h, w]], device=device)
            
            # 使用PostProcess处理输出
            predictions = postprocessor(model_outputs, target_sizes)
            pred_result = predictions[0]  # 获取第一个样本的预测结果
            
            # 反归一化图像
            original_image = denormalize_image(sample_image[0])
            
            # 可视化
            fig = visualize_predictions_and_gt(
                original_image, pred_result, sample_annotations, 
                class_names, score_threshold
            )
            
            # 保存图像
            save_path = os.path.join(output_dir, f"{split_name}_sample_{sample_count:03d}_idx_{sample_global_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
            # 打印一些统计信息
            num_pred = len(pred_result['boxes']) if pred_result['boxes'] is not None else 0
            num_gt = len(sample_annotations['boxes'])
            num_pred_valid = torch.sum(pred_result['scores'] > score_threshold).item() if pred_result['scores'] is not None else 0
            
            print(f"保存 {split_name} 样本 {sample_count + 1}/{num_samples} (索引={sample_global_idx}): {save_path}")
            print(f"  预测框数量: {num_pred} (阈值>{score_threshold}: {num_pred_valid})")
            print(f"  真实框数量: {num_gt}")
            
            sample_count += 1


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='可视化DETR模型预测结果')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    # parser.add_argument('--checkpoint', type=str, required=True,
    #                    help='Checkpoint文件或目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径 (默认: auto-generated)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='每个数据集(train/test)的抽样数量 (默认: 10)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                       help='预测分数阈值 (默认: 0.3)')
    parser.add_argument('--super_config', type=str, default=None,
                       help='Super config文件路径')
    
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
    
    # 设置默认输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(args.checkpoint.rstrip('/'))
        args.output_dir = f"vis_results_{checkpoint_name}_{timestamp}"
    
    # Run visualization
    try:
        visualize_engine(
            config=cfg,
            # checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            score_threshold=args.score_threshold
        )
        print(f"\n✅ 可视化完成! 结果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 可视化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 