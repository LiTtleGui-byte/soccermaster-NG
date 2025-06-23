from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional

class VideoCaptionHead(nn.Module):
    def __init__(self, loss_type='siglip_loss', backbone_type='image'):
        super().__init__()
        self.loss_type = loss_type
        self.backbone_type = backbone_type
        
        if loss_type == 'siglip_loss':
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
            self.logits_bias = nn.Parameter(torch.tensor(-10.0))
        elif loss_type == 'infonce_loss':
            self.temperature = nn.Parameter(torch.tensor(0.3))
        else:
            raise ValueError(f"Loss type {loss_type} not supported.")

    def forward(self, 
                backbone_outputs, 
                is_training: bool = False,
                gather_distributed: bool = True):
        # 提取特征
        global_features = backbone_outputs['global_features']
        text_features = backbone_outputs['text_features']
        
        # 处理视觉特征
        if self.backbone_type == 'video':
            vision_features = global_features.mean(dim=1)  # [N, D]
        else:
            vision_features = global_features[:, 0]  # [N, D]
        
        # 归一化特征
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # DDP模式下收集全局特征（仅在训练时启用）
        if dist.is_initialized() and is_training and gather_distributed:
            vision_features, text_features = self._gather_features_distributed(
                vision_features, 
                text_features
            )
        
        # 计算相似度矩阵
        base_similarity_matrix = vision_features @ text_features.t()
        
        # 根据损失类型处理相似度矩阵
        if self.loss_type == 'siglip_loss':
            processed_similarity_matrix = base_similarity_matrix * self.logit_scale.exp() + self.logits_bias
        elif self.loss_type == 'infonce_loss':
            processed_similarity_matrix = base_similarity_matrix / self.temperature.clamp(min=1e-6)
        else:
            processed_similarity_matrix = base_similarity_matrix
        
        output = {
            'vision_features': vision_features,
            'text_features': text_features,
            'base_similarity_matrix': base_similarity_matrix,
            'processed_similarity_matrix': processed_similarity_matrix
        }
        return output
    
    def _gather_features_distributed(self, vision, text):
        """
        跨GPU收集特征张量
        返回形状: [world_size * local_batch, D]
        """
        world_size = dist.get_world_size()
        
        # 准备收集容器
        vision_list = [torch.zeros_like(vision) for _ in range(world_size)]
        text_list = [torch.zeros_like(text) for _ in range(world_size)]
        
        # 全收集操作
        dist.all_gather(vision_list, vision)
        dist.all_gather(text_list, text)
        
        # 保留当前进程特征的梯度
        vision_list[dist.get_rank()] = vision
        text_list[dist.get_rank()] = text
        
        # 拼接所有特征
        vision_global = torch.cat(vision_list, dim=0)
        text_global = torch.cat(text_list, dim=0)
        
        return vision_global, text_global

class VideoCaptionLoss(nn.Module):
    def __init__(self, 
                 weight_dict, 
                 loss_type='siglip_loss', 
                 distributed_gather=True):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_type = loss_type
        self.distributed_gather = distributed_gather

    def forward(self, outputs, targets):
        # 从输出中获取特征
        proc_sim = outputs['processed_similarity_matrix']
        base_sim = outputs['base_similarity_matrix']
        
        # 收集全局标签矩阵
        target_label = self._create_global_label(targets, base_sim.device)
        
        # 计算损失
        if self.loss_type == 'siglip_loss':
            loss = self.compute_siglip_loss(proc_sim, target_label)
        elif self.loss_type == 'infonce_loss':
            loss = self.compute_infonce_loss(proc_sim, target_label)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_type}")
        
        # 计算精度指标
        top_1_acc, top_3_acc, top_5_acc = self.calculate_top_k_accuracy(base_sim, target_label)
        
        losses = {
            f'{self.loss_type}': loss,
            'top_1_accuracy': top_1_acc,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc
        }
        return losses, self.weight_dict
    
    def _create_global_label(self, targets, device):
        """创建全局标签矩阵，支持DDP模式"""
        # Step 1: 本地caption收集
        local_captions = [t['caption'] for t in targets]
        
        # Step 2: 分布式收集所有caption
        if dist.is_initialized() and self.distributed_gather:
            global_captions = self._gather_captions_distributed(local_captions)
        else:
            global_captions = local_captions
            
        # Step 3: 创建全局标签矩阵
        return create_label_from_comment(global_captions).to(device)
    
    def _gather_captions_distributed(self, local_captions):
        """跨GPU收集caption字符串"""
        world_size = dist.get_world_size()
        
        # 当前进程的数据包装为对象
        local_data = [local_captions]
        all_data = [None] * world_size
        
        # 使用PyTorch的跨进程对象收集
        dist.all_gather_object(all_data, local_data)
        
        # 展平收集的数据
        flat_captions = []
        for data in all_data:
            flat_captions.extend(data[0])
        return flat_captions

    def compute_siglip_loss(self, logits, target_label):
        logits_per_image = logits.t()
        return -F.logsigmoid(target_label * logits_per_image).mean()

    def compute_infonce_loss(self, logits, target_label):
        logits = logits / self.temperature
        pos_mask = (target_label > 0).float()
        neg_mask = (target_label < 0).float()
        
        pos_loss = -torch.log(torch.sigmoid(logits)) * pos_mask
        neg_loss = -torch.log(1 - torch.sigmoid(logits)) * neg_mask
        
        # 计算有效样本数
        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)
        
        return (pos_loss.sum() / n_pos + neg_loss.sum() / n_neg) / 2

    def calculate_top_k_accuracy(self, sim_matrix, labels):
        batch_size = sim_matrix.size(0)
        topk_indices = torch.topk(sim_matrix, k=5, dim=1)[1]
        
        # 创建正样本标签 (1表示匹配)
        pos_mask = (labels > 0).float()
        
        # 计算Top-K准确率
        correct = torch.gather(pos_mask, 1, topk_indices)
        top1_acc = correct[:, 0].sum() / batch_size
        top3_acc = correct[:, :3].sum(dim=1).clamp(max=1).sum() / batch_size
        top5_acc = correct.sum(dim=1).clamp(max=1).sum() / batch_size
        
        return top1_acc, top3_acc, top5_acc

def create_label_from_comment(captions, special_categories=None):
    """创建全局标签矩阵，1=正样本，-1=负样本"""
    N = len(captions)
    labels = -torch.ones(N, N)  # 默认全负样本
    
    if special_categories is None:
        special_categories = {"end of half game", "off side", "start of half game",
                             "ball possession", "substitution"}
    
    # 对角线总是正样本 (自监督基础假设)
    for i in range(N):
        labels[i, i] = 1.0
        
    # 特殊类别匹配
    for i, cap_i in enumerate(captions):
        for j, cap_j in enumerate(captions):
            if i != j and cap_i in special_categories and cap_i == cap_j:
                labels[i, j] = 1.0
                labels[j, i] = 1.0
    
    return labels

def build_video_caption_loss(config):
    loss_type = config["VIDEO_CAPTION_LOSS_TYPE"]
    
    if loss_type == "siglip_loss":
        weight_dict = {'siglip_loss': config["VIDEO_CAPTION_SIGLIP_LOSS_WEIGHT"]}
    elif loss_type == "infonce_loss":
        weight_dict = {'infonce_loss': config["VIDEO_CAPTION_INFONCE_LOSS_WEIGHT"]}
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return VideoCaptionLoss(
        weight_dict=weight_dict,
        loss_type=loss_type,
        distributed_gather=config["DISTRIBUTED_GATHER"]  # 配置文件中添加此选项
    )

def build_video_caption_head(config):
    return VideoCaptionHead(
        loss_type=config["VIDEO_CAPTION_LOSS_TYPE"],
        backbone_type=config["BACKBONE_TYPE"]
    )