from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

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
                metas,
                gather_distributed: bool = True):
        global_features = backbone_outputs['global_features']
        text_features = backbone_outputs['text_features']
        
        # 处理视觉特征
        if self.backbone_type == 'video':
            vision_features = global_features.mean(dim=1)  # [N, D]
        else:
            vision_features = global_features[:, 0]  # [N, D]
        
        # 归一化特征
        vision_features = F.normalize(vision_features, dim=-1)
        
        assert text_features is not None
        # if text_features is not None:
        # 检测哪些位置有有效的text特征（非零向量）
        text_norms = torch.norm(text_features, dim=-1)
        valid_text_mask = text_norms > 1e-6  # 零向量的norm接近0
        
        # if valid_text_mask.any():
        # 无论是否有，都要进行gather，不然会死锁
        text_features = F.normalize(text_features, dim=-1)
        
        # DDP模式下收集全局特征
        if dist.is_initialized() and gather_distributed:
            vision_features, text_features, valid_text_mask = self._gather_features_distributed_with_mask(
                vision_features, 
                text_features,
                valid_text_mask
            )
        
        # 此时valid_text_mask是全局的
        if valid_text_mask.any():
            # 计算相似度矩阵
            base_similarity_matrix = vision_features @ text_features.t()
            
            # 根据损失类型处理相似度矩阵
            if self.loss_type == 'siglip_loss':
                processed_similarity_matrix = base_similarity_matrix * self.logit_scale.exp() + self.logits_bias
            elif self.loss_type == 'infonce_loss':
                processed_similarity_matrix = base_similarity_matrix / self.temperature.clamp(min=1e-6)
            else:
                processed_similarity_matrix = base_similarity_matrix
        else:
            # 没有有效的text特征
            base_similarity_matrix = None
            processed_similarity_matrix = None
            valid_text_mask = torch.zeros(vision_features.shape[0], dtype=torch.bool, device=vision_features.device)
        # else:
        #     base_similarity_matrix = None
        #     processed_similarity_matrix = None
        #     valid_text_mask = torch.zeros(vision_features.shape[0], dtype=torch.bool, device=vision_features.device)
        
        output = {
            'vision_features': vision_features,
            'text_features': text_features,
            'base_similarity_matrix': base_similarity_matrix,
            'processed_similarity_matrix': processed_similarity_matrix,
            'valid_text_mask': valid_text_mask
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
    
    def _gather_features_distributed_with_mask(self, vision, text, valid_mask):
        """
        跨GPU收集特征张量和mask信息
        返回形状: [world_size * local_batch, D]
        """
        world_size = dist.get_world_size()
        
        # 准备收集容器
        vision_list = [torch.zeros_like(vision) for _ in range(world_size)]
        text_list = [torch.zeros_like(text) for _ in range(world_size)]
        mask_list = [torch.zeros_like(valid_mask) for _ in range(world_size)]
        
        # 全收集操作
        dist.all_gather(vision_list, vision)
        dist.all_gather(text_list, text)
        dist.all_gather(mask_list, valid_mask)
        
        # 保留当前进程特征的梯度
        vision_list[dist.get_rank()] = vision
        text_list[dist.get_rank()] = text
        mask_list[dist.get_rank()] = valid_mask
        
        # 拼接所有特征
        vision_global = torch.cat(vision_list, dim=0)
        text_global = torch.cat(text_list, dim=0)
        mask_global = torch.cat(mask_list, dim=0)
        
        return vision_global, text_global, mask_global

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
        # 获取有效text的mask
        valid_text_mask = outputs.get('valid_text_mask', None)
        proc_sim = outputs['processed_similarity_matrix']
        base_sim = outputs['base_similarity_matrix']
        
        # 检查是否有有效的text样本
        if valid_text_mask is None or not valid_text_mask.any() or proc_sim is None or base_sim is None:
            # 如果没有有效的text样本，返回零loss
            device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
            losses = {
                f'{self.loss_type}': dummy_loss,
                'top_1_accuracy': torch.tensor(0.0, device=device),
                'top_3_accuracy': torch.tensor(0.0, device=device),
                'top_5_accuracy': torch.tensor(0.0, device=device),
                'top_1_accuracy_type': torch.tensor(0.0, device=device),
                'top_3_accuracy_type': torch.tensor(0.0, device=device),
                'top_5_accuracy_type': torch.tensor(0.0, device=device)
            }
            return losses, self.weight_dict
        
        # 只保留有效的text样本
        global_captions = self._gather_captions_distributed(targets)
        valid_indices = torch.where(valid_text_mask)[0]
        valid_captions = [global_captions[i] for i in valid_indices.cpu().tolist()]
        valid_proc_sim = proc_sim[valid_text_mask][:, valid_text_mask]
        valid_base_sim = base_sim[valid_text_mask][:, valid_text_mask]
        
        # 收集全局标签矩阵
        # target_label = self._create_global_label(valid_captions, valid_base_sim.device)
        target_label = create_label_from_comment(valid_captions).to(valid_base_sim.device)
        
        # 计算损失
        if self.loss_type == 'siglip_loss':
            loss = self.compute_siglip_loss(valid_proc_sim, target_label)
        elif self.loss_type == 'infonce_loss':
            loss = self.compute_infonce_loss(valid_proc_sim, target_label)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_type}")
        
        # 计算精度指标
        top_1_acc, top_3_acc, top_5_acc = self.calculate_top_k_accuracy(valid_base_sim, target_label)
        with torch.no_grad():
            target_label_type = create_label_from_type(valid_captions).to(valid_base_sim.device)
            top_1_acc_type, top_3_acc_type, top_5_acc_type = self.calculate_top_k_accuracy(valid_base_sim, target_label_type)
        
        losses = {
            f'{self.loss_type}': loss,
            'top_1_accuracy': top_1_acc,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc,
            'top_1_accuracy_type': top_1_acc_type,
            'top_3_accuracy_type': top_3_acc_type,
            'top_5_accuracy_type': top_5_acc_type
        }
        return losses, self.weight_dict
    
    def _gather_captions_distributed(self, targets):
        local_captions = [t['caption'] for t in targets]
        if dist.is_initialized() and self.distributed_gather:
            global_captions = self._gather_list_distributed(local_captions)
        else:
            global_captions = local_captions
        return global_captions
    
    def _create_global_label(self, targets, device):
        """创建全局标签矩阵，支持DDP模式"""
        # Step 1: 本地caption收集
        local_captions = [t['caption'] for t in targets]
        
        # Step 2: 分布式收集所有caption
        if dist.is_initialized() and self.distributed_gather:
            global_captions = self._gather_list_distributed(local_captions)
        else:
            global_captions = local_captions
            
        # Step 3: 创建全局标签矩阵
        return create_label_from_comment(global_captions).to(device)
    
    def _gather_list_distributed(self, local_list):
        world_size = dist.get_world_size()
        
        # 当前进程的数据包装为对象
        local_data = [local_list]
        all_data = [None] * world_size
        
        # 使用PyTorch的跨进程对象收集
        dist.all_gather_object(all_data, local_data)
        
        # 展平收集的数据
        flat_list = []
        for data in all_data:
            flat_list.extend(data[0])
        return flat_list

    def compute_siglip_loss(self, logits, target_label):
        """
        SigLIP损失：对每个样本分别计算损失，然后平均
        logits: [batch_size, batch_size] - 相似度矩阵
        target_label: [batch_size, batch_size] - 标签矩阵 (1=正样本, -1=负样本)
        """
        logits_per_image = logits.t()
        # return -F.logsigmoid(target_label * logits_per_image).mean()
        return -F.logsigmoid(target_label * logits_per_image).sum() / logits.shape[0]

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
        
        # 根据batch_size动态调整k值
        max_k = min(5, batch_size)
        topk_indices = torch.topk(sim_matrix, k=max_k, dim=1)[1]
        
        # 创建正样本标签 (1表示匹配)
        pos_mask = (labels > 0).float()
        
        # 计算Top-K准确率
        correct = torch.gather(pos_mask, 1, topk_indices)
        
        # Top-1准确率
        top1_acc = correct[:, 0].sum() / batch_size
        
        # Top-3准确率：如果batch_size < 3，设置为1.0
        if batch_size < 3:
            top3_acc = torch.tensor(1.0, device=sim_matrix.device)
        else:
            top3_acc = correct[:, :3].sum(dim=1).clamp(max=1).sum() / batch_size
        
        # Top-5准确率：如果batch_size < 5，设置为1.0
        if batch_size < 5:
            top5_acc = torch.tensor(1.0, device=sim_matrix.device)
        else:
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
            if cap_i in special_categories and cap_i == cap_j:
                labels[i, j] = 1.0
                labels[j, i] = 1.0
    
    return labels

def create_label_from_type(captions):
    """创建全局标签矩阵，1=正样本，-1=负样本"""
    N = len(captions)
    labels = -torch.ones(N, N)  # 默认全负样本
    
    # 对角线总是正样本 (自监督基础假设)
    for i in range(N):
        labels[i, i] = 1.0
        
    # 特殊类别匹配
    for i, cap_i in enumerate(captions):
        for j, cap_j in enumerate(captions):
            if cap_i == cap_j:
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


class VideoCaptionMetrics(nn.Module):
    """VideoCaption的指标计算类"""
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        """重置收集的数据"""
        self.video_caption_metrics_data = {
            'top_1_accuracy': [],
            'top_3_accuracy': [],
            'top_5_accuracy': [],
            'top_1_accuracy_type': [],
            'top_3_accuracy_type': [],
            'top_5_accuracy_type': [],
            'sample_count': 0
        }

    def update(self, outputs, targets, loss_task_raw=None):
        """
        更新指标数据
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            loss_task_raw: 损失字典，包含top_k_accuracy值
        """
        # 检查是否有有效的text样本
        valid_text_mask = outputs.get('valid_text_mask', None)
        if valid_text_mask is None or not valid_text_mask.any():
            # 如果没有有效的text样本，跳过更新
            return
        
        top_1_acc = loss_task_raw['top_1_accuracy']
        top_3_acc = loss_task_raw['top_3_accuracy'] 
        top_5_acc = loss_task_raw['top_5_accuracy']
        top_1_acc_type = loss_task_raw['top_1_accuracy_type']
        top_3_acc_type = loss_task_raw['top_3_accuracy_type']
        top_5_acc_type = loss_task_raw['top_5_accuracy_type']
        
        # 转移到CPU并添加到收集器
        self.video_caption_metrics_data['top_1_accuracy'].append(top_1_acc.cpu().item())
        self.video_caption_metrics_data['top_3_accuracy'].append(top_3_acc.cpu().item())
        self.video_caption_metrics_data['top_5_accuracy'].append(top_5_acc.cpu().item())
        self.video_caption_metrics_data['top_1_accuracy_type'].append(top_1_acc_type.cpu().item())
        self.video_caption_metrics_data['top_3_accuracy_type'].append(top_3_acc_type.cpu().item())
        self.video_caption_metrics_data['top_5_accuracy_type'].append(top_5_acc_type.cpu().item())
        
        # 更新样本计数（只计算有效的text样本数）
        valid_sample_count = valid_text_mask.sum().item()
        self.video_caption_metrics_data['sample_count'] += valid_sample_count

    def gather_metrics_data(self, accelerator):
        """收集所有进程的指标数据"""
        video_caption_key_list = ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_1_accuracy_type', 'top_3_accuracy_type', 'top_5_accuracy_type']
        gathered_video_caption_metrics = {}
        
        for key in video_caption_key_list:
            gathered_video_caption_metrics[key] = gather_object(self.video_caption_metrics_data[key])
        
        gathered_video_caption_metrics['sample_count'] = gather_object([self.video_caption_metrics_data['sample_count']])
        return gathered_video_caption_metrics

    def compute_metrics_from_gathered_data(self, gathered_video_caption_metrics):
        """从收集的数据计算最终指标"""
        metrics = {}

        # 展平所有进程的video caption数据
        all_top_1_accuracy = flatten_data(gathered_video_caption_metrics['top_1_accuracy'])
        all_top_3_accuracy = flatten_data(gathered_video_caption_metrics['top_3_accuracy'])
        all_top_5_accuracy = flatten_data(gathered_video_caption_metrics['top_5_accuracy'])
        all_top_1_accuracy_type = flatten_data(gathered_video_caption_metrics['top_1_accuracy_type'])
        all_top_3_accuracy_type = flatten_data(gathered_video_caption_metrics['top_3_accuracy_type'])
        all_top_5_accuracy_type = flatten_data(gathered_video_caption_metrics['top_5_accuracy_type'])
        
        # 计算总的样本数
        total_sample_count = sum(gathered_video_caption_metrics['sample_count'])
        
        if total_sample_count > 0:
            # 计算平均accuracy
            top_1_accuracy_tensor = torch.tensor(all_top_1_accuracy, dtype=torch.float32)
            top_3_accuracy_tensor = torch.tensor(all_top_3_accuracy, dtype=torch.float32)
            top_5_accuracy_tensor = torch.tensor(all_top_5_accuracy, dtype=torch.float32)
            top_1_accuracy_type_tensor = torch.tensor(all_top_1_accuracy_type, dtype=torch.float32)
            top_3_accuracy_type_tensor = torch.tensor(all_top_3_accuracy_type, dtype=torch.float32)
            top_5_accuracy_type_tensor = torch.tensor(all_top_5_accuracy_type, dtype=torch.float32)
            
            metrics['video_caption_top_1_accuracy'] = top_1_accuracy_tensor.mean().item()
            metrics['video_caption_top_3_accuracy'] = top_3_accuracy_tensor.mean().item()
            metrics['video_caption_top_5_accuracy'] = top_5_accuracy_tensor.mean().item()
            metrics['video_caption_top_1_accuracy_type'] = top_1_accuracy_type_tensor.mean().item()
            metrics['video_caption_top_3_accuracy_type'] = top_3_accuracy_type_tensor.mean().item()
            metrics['video_caption_top_5_accuracy_type'] = top_5_accuracy_type_tensor.mean().item()
            
            # 记录样本数量
            metrics['video_caption_total_samples'] = total_sample_count
            
            # 计算标准差
            metrics['video_caption_top_1_accuracy_std'] = top_1_accuracy_tensor.std().item()
            metrics['video_caption_top_3_accuracy_std'] = top_3_accuracy_tensor.std().item()
            metrics['video_caption_top_5_accuracy_std'] = top_5_accuracy_tensor.std().item()
            metrics['video_caption_top_1_accuracy_type_std'] = top_1_accuracy_type_tensor.std().item()
            metrics['video_caption_top_3_accuracy_type_std'] = top_3_accuracy_type_tensor.std().item()
            metrics['video_caption_top_5_accuracy_type_std'] = top_5_accuracy_type_tensor.std().item()
            
        else:
            # 没有有效的video caption数据
            for metric_name in ['video_caption_top_1_accuracy', 'video_caption_top_3_accuracy', 'video_caption_top_5_accuracy',
                                'video_caption_top_1_accuracy_type', 'video_caption_top_3_accuracy_type', 'video_caption_top_5_accuracy_type',
                                'video_caption_top_1_accuracy_std', 'video_caption_top_3_accuracy_std', 'video_caption_top_5_accuracy_std',
                                'video_caption_top_1_accuracy_type_std', 'video_caption_top_3_accuracy_type_std', 'video_caption_top_5_accuracy_type_std']:
                metrics[metric_name] = 0.0
            metrics['video_caption_total_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets, loss_task_raw=None):
        """
        前向传播计算指标（评估时使用）
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            loss_task_raw: 损失字典，包含accuracy值
            
        Returns:
            空字典（实际指标在compute_final_metrics中计算）
        """
        self.update(outputs, targets, loss_task_raw)
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


def build_video_caption_metrics(config):
    """构建VideoCaption指标计算器"""
    return VideoCaptionMetrics()