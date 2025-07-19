from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List
import math
from models.utils.flatten_data import flatten_data
from data.video_caption import keywords_list
from accelerate.utils.operations import gather_object

class CaptionClassificationHead(nn.Module):
    def __init__(self, input_dim=768, backbone_type='image', dropout_rate=0.1, use_attn_pool=False, 
                 use_transformers=False, num_transformer_encoder=2, use_mlp=True, use_layer_norm=False):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 分类类别数
            backbone_type: backbone类型，'image'或'video'
            dropout_rate: dropout比率
            use_attn_pool: 是否使用attention pooling，默认False
            use_transformers: 是否在pooling前使用transformer encoder，默认False
            num_transformer_encoder: transformer encoder的层数，默认2
            use_mlp: 是否使用MLP分类器，默认True。如果False则使用单层Linear
            use_layer_norm: 是否使用LayerNorm，默认False
        """
        super().__init__()
        assert backbone_type == 'video'
        self.backbone_type = backbone_type
        self.use_attn_pool = use_attn_pool
        self.use_transformers = use_transformers
        self.use_mlp = use_mlp
        self.use_layer_norm = use_layer_norm
        num_classes = len(keywords_list)
        
        # Transformer encoder layers (optional)
        if self.use_transformers:
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=12,
                dim_feedforward=input_dim * 4,
                dropout=dropout_rate,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer,
                num_layers=num_transformer_encoder
            )
        
        if self.use_attn_pool:
            self.query_token = nn.Parameter(torch.randn(1, 1, input_dim))
            # Multi-head attention for pooling
            self.attn_pool = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=12,
                dropout=dropout_rate,
                batch_first=True
            )
            # Layer norm after attention pooling
            self.attn_pool_ln = nn.LayerNorm(input_dim)
        
        # 分类头网络
        if self.use_mlp:
            # MLP分类器（多层神经网络）
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(input_dim // 4, num_classes)
            )
        else:
            # 单层Linear分类器
            self.classifier = nn.Linear(input_dim, num_classes)
        
        # LayerNorm层（可选）
        if self.use_layer_norm:
            self.classifier_ln1 = nn.LayerNorm(input_dim)
            self.classifier_ln2 = nn.LayerNorm(input_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化query token
        if self.use_attn_pool:
            nn.init.normal_(self.query_token, std=0.02)

    def forward(self, backbone_outputs, metas):
        """
        Args:
            backbone_outputs: backbone输出，包含global_features
            metas: 元数据
            
        Returns:
            包含logits的字典
        """
        global_features = backbone_outputs['global_features']
        
        # 可选的LayerNorm处理（第一个）
        if self.use_layer_norm:
            global_features = self.classifier_ln1(global_features)
        
        # 可选的transformer encoder处理
        if self.use_transformers:
            global_features = self.transformer_encoder(global_features)
        
        # 根据是否使用attention pooling选择不同的特征提取方式
        if self.use_attn_pool:
            batch_size = global_features.size(0)
            query = self.query_token.expand(batch_size, -1, -1)  # [N, 1, D]
            
            attn_output, _ = self.attn_pool(
                query=query,  # [N, 1, D]
                key=global_features,  # [N, seq_len, D]
                value=global_features  # [N, seq_len, D]
            )
            
            vision_features = self.attn_pool_ln(attn_output.squeeze(1))  # [N, D]
        else:
            vision_features = global_features.mean(dim=1)  # [N, D]
        
        # 可选的LayerNorm处理（第二个）
        if self.use_layer_norm:
            vision_features = self.classifier_ln2(vision_features)
        
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
        
        # 计算准确率（用于监控）
        with torch.no_grad():
            accuracy = self.calculate_accuracy(logits, labels)
        
        losses = {
            'classification_loss': classification_loss,
            'accuracy': accuracy
        }
        
        return losses, self.weight_dict
    
    def calculate_accuracy(self, logits, labels):
        """
        计算准确率
        
        Args:
            logits: 预测logits [N, num_classes]
            labels: 真实标签 [N]
            
        Returns:
            accuracy: 准确率
        """
        batch_size = logits.size(0)
        
        # 获取预测结果
        predictions = torch.argmax(logits, dim=1)
        
        # 计算准确率
        correct = (predictions == labels).float()
        accuracy = correct.sum() / batch_size
        
        return accuracy


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
            gathered_metrics[key] = gather_object(self.metrics_data[key])
        gathered_metrics['total_samples'] = gather_object([self.metrics_data['total_samples']])
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
        dropout_rate=config["CAPTION_CLASSIFICATION_DROPOUT_RATE"],
        use_attn_pool=config["CAPTION_CLASSIFICATION_USE_ATTN_POOL"],
        use_transformers=config["CAPTION_CLASSIFICATION_USE_TRANSFORMERS"],
        num_transformer_encoder=config["CAPTION_CLASSIFICATION_NUM_TRANSFORMER_ENCODER"],
        use_mlp=config["CAPTION_CLASSIFICATION_USE_MLP"],
        use_layer_norm=config["CAPTION_CLASSIFICATION_USE_LAYER_NORM"]
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