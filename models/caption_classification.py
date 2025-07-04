from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List
import math
from models.utils.flatten_data import flatten_data
from data.video_caption import keywords_list

class CaptionClassificationHead(nn.Module):
    def __init__(self, input_dim=768, backbone_type='image', dropout_rate=0.1):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 分类类别数
            backbone_type: backbone类型，'image'或'video'
            dropout_rate: dropout比率
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.num_classes = len(keywords_list)
        
        # 分类头网络
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_outputs, metas):
        """
        Args:
            backbone_outputs: backbone输出，包含global_features
            metas: 元数据
            
        Returns:
            包含logits的字典
        """
        global_features = backbone_outputs['global_features']
        
        if self.backbone_type == 'video':
            vision_features = global_features.mean(dim=1)  # [N, D]
        else:
            vision_features = global_features[:, 0]  # [N, D]
        
        logits = self.classifier(vision_features)  # [N, num_classes]
        
        output = {
            'logits': logits,
            'features': vision_features
        }
        return output


class CaptionClassificationLoss(nn.Module):
    def __init__(self, weight_dict, label_smoothing=0.0):
        """
        Caption分类损失
        
        Args:
            weight_dict: 损失权重字典
            num_classes: 分类类别数
            label_smoothing: 标签平滑系数
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.num_classes = len(keywords_list)
        self.label_smoothing = label_smoothing
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction='mean')

    def forward(self, outputs, targets):
        """
        计算分类损失
        
        Args:
            outputs: 模型输出，包含logits
            targets: 目标列表，每个包含caption_index
            
        Returns:
            losses: 损失字典
            weight_dict: 权重字典
        """
        logits = outputs['logits']  # [N, num_classes]
        
        # 提取caption_index作为标签
        labels = torch.stack([t['caption_index'] for t in targets], dim=0)  # [N]
        labels = labels.to(logits.device)
        
        # 计算交叉熵损失
        classification_loss = self.criterion(logits, labels)
        
        # 计算top-k准确率（用于监控）
        with torch.no_grad():
            top1_acc, top3_acc, top5_acc = self.calculate_top_k_accuracy(logits, labels)
        
        losses = {
            'classification_loss': classification_loss,
            'top_1_accuracy': top1_acc,
            'top_3_accuracy': top3_acc,
            'top_5_accuracy': top5_acc
        }
        
        return losses, self.weight_dict
    
    def calculate_top_k_accuracy(self, logits, labels):
        """
        计算Top-K准确率
        
        Args:
            logits: 预测logits [N, num_classes]
            labels: 真实标签 [N]
            
        Returns:
            top1_acc, top3_acc, top5_acc: Top-1, Top-3, Top-5准确率
        """
        batch_size = logits.size(0)
        
        # 获取top-5预测
        _, top5_pred = torch.topk(logits, k=min(5, self.num_classes), dim=1)
        
        # 扩展标签以匹配top-k形状
        labels_expanded = labels.unsqueeze(1).expand_as(top5_pred)
        
        # 计算匹配
        correct = (top5_pred == labels_expanded).float()
        
        # Top-1准确率
        top1_acc = correct[:, 0].sum() / batch_size
        
        # Top-3准确率
        if top5_pred.size(1) >= 3:
            top3_acc = correct[:, :3].sum(dim=1).clamp(max=1).sum() / batch_size
        else:
            top3_acc = correct.sum(dim=1).clamp(max=1).sum() / batch_size
        
        # Top-5准确率
        top5_acc = correct.sum(dim=1).clamp(max=1).sum() / batch_size
        
        return top1_acc, top3_acc, top5_acc


class CaptionClassificationMetrics(nn.Module):
    """Caption分类任务的指标计算类"""
    
    def __init__(self):
        super().__init__()
        self.num_classes = len(keywords_list)
        self.reset()
        
    def reset(self):
        """重置收集的数据"""
        self.metrics_data = {
            'predictions': [],      # 预测结果
            'targets': [],         # 真实标签
            'confidences': [],     # 预测置信度
            'total_samples': 0     # 总样本数
        }

    def update(self, outputs, targets):
        """
        更新指标数据
        
        Args:
            outputs: 模型输出，包含logits
            targets: 目标数据列表
        """
        logits = outputs['logits']  # [N, num_classes]
        labels = torch.stack([t['caption_index'] for t in targets], dim=0)  # [N]
        labels = labels.to(logits.device)
        
        # 获取预测结果
        probs = F.softmax(logits, dim=1)  # [N, num_classes]
        predictions = torch.argmax(logits, dim=1)  # [N]
        confidences = torch.max(probs, dim=1)[0]  # [N]
        
        # 转移到CPU并添加到收集器
        self.metrics_data['predictions'].extend(predictions.cpu().tolist())
        self.metrics_data['targets'].extend(labels.cpu().tolist())
        self.metrics_data['confidences'].extend(confidences.cpu().tolist())
        self.metrics_data['total_samples'] += len(predictions)

    def gather_metrics_data(self, accelerator):
        """收集所有进程的指标数据"""
        gathered_metrics = {}
        for key in ['predictions', 'targets', 'confidences']:
            gathered_metrics[key] = accelerator.gather_for_metrics(self.metrics_data[key])
        gathered_metrics['total_samples'] = accelerator.gather_for_metrics([self.metrics_data['total_samples']])
        return gathered_metrics

    def compute_metrics_from_gathered_data(self, gathered_metrics):
        """从收集的数据计算最终指标"""
        metrics = {}
        
        # 展平所有进程的数据
        all_predictions = flatten_data(gathered_metrics['predictions'])
        all_targets = flatten_data(gathered_metrics['targets'])
        all_confidences = flatten_data(gathered_metrics['confidences'])
        
        # 计算总样本数
        total_samples = sum(gathered_metrics['total_samples'])
        
        if total_samples > 0:
            # 转换为tensor
            predictions = torch.tensor(all_predictions, dtype=torch.long)
            targets = torch.tensor(all_targets, dtype=torch.long)
            confidences = torch.tensor(all_confidences, dtype=torch.float32)
            
            # 计算整体准确率
            accuracy = (predictions == targets).float().mean().item()
            metrics['classification_accuracy'] = accuracy
            
            # 计算平均置信度
            avg_confidence = confidences.mean().item()
            metrics['avg_confidence'] = avg_confidence
            
            # 计算每个类别的准确率
            class_correct = torch.zeros(self.num_classes)
            class_total = torch.zeros(self.num_classes)
            
            for target_class in range(self.num_classes):
                mask = (targets == target_class)
                if mask.sum() > 0:
                    class_total[target_class] = mask.sum().float()
                    class_correct[target_class] = (predictions[mask] == target_class).sum().float()
            
            # 计算每类准确率（避免除零）
            class_accuracies = class_correct / (class_total + 1e-8)
            
            # 计算宏平均准确率（只考虑有样本的类别）
            valid_classes = class_total > 0
            if valid_classes.sum() > 0:
                macro_accuracy = class_accuracies[valid_classes].mean().item()
                metrics['macro_accuracy'] = macro_accuracy
            else:
                metrics['macro_accuracy'] = 0.0
            
            # 计算混淆矩阵相关指标
            # 计算精确率、召回率、F1分数的宏平均
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for target_class in range(self.num_classes):
                # True Positives
                tp = ((predictions == target_class) & (targets == target_class)).sum().float()
                # False Positives
                fp = ((predictions == target_class) & (targets != target_class)).sum().float()
                # False Negatives
                fn = ((predictions != target_class) & (targets == target_class)).sum().float()
                
                # 精确率
                precision = tp / (tp + fp + 1e-8)
                # 召回率
                recall = tp / (tp + fn + 1e-8)
                # F1分数
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                precision_scores.append(precision.item())
                recall_scores.append(recall.item())
                f1_scores.append(f1.item())
            
            # 宏平均
            metrics['macro_precision'] = sum(precision_scores) / len(precision_scores)
            metrics['macro_recall'] = sum(recall_scores) / len(recall_scores)
            metrics['macro_f1'] = sum(f1_scores) / len(f1_scores)
            
            # 记录样本数量和类别分布
            metrics['total_samples'] = total_samples
            metrics['num_classes_with_samples'] = valid_classes.sum().item()
            
            # # 计算置信度相关统计
            # metrics['confidence_std'] = confidences.std().item()
            # metrics['confidence_min'] = confidences.min().item()
            # metrics['confidence_max'] = confidences.max().item()
            
        else:
            # 没有有效样本
            for metric_name in ['classification_accuracy', 'avg_confidence',
                                'macro_accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
                                # 'confidence_std', 'confidence_min', 'confidence_max']:
                metrics[metric_name] = 0.0
            metrics['total_samples'] = 0
            metrics['num_classes_with_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        前向传播计算指标（评估时使用）
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            
        Returns:
            空字典（实际指标在compute_final_metrics中计算）
        """
        self.update(outputs, targets)
        return {}

    def compute_final_metrics(self, accelerator):
        """
        计算最终指标（在所有数据收集完成后调用）
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            最终指标字典
        """
        gathered_data = self.gather_metrics_data(accelerator)
        if accelerator.is_main_process:
            final_metrics = self.compute_metrics_from_gathered_data(gathered_data)
            return final_metrics
        else:
            return {}


def build_caption_classification_head(config: dict):
    """构建Caption分类头"""
    return CaptionClassificationHead(
        input_dim=768,
        backbone_type=config["BACKBONE_TYPE"],
        dropout_rate=config["CAPTION_CLASSIFICATION_DROPOUT_RATE"]
    )


def build_caption_classification_loss(config: dict):
    """构建Caption分类损失函数"""
    weight_dict = {
        'classification_loss': config["CAPTION_CLASSIFICATION_LOSS_WEIGHT"]
    }
    
    return CaptionClassificationLoss(
        weight_dict=weight_dict,
        label_smoothing=config["CAPTION_CLASSIFICATION_LABEL_SMOOTHING"]
    )


def build_caption_classification_metrics(config: dict):
    return CaptionClassificationMetrics() 