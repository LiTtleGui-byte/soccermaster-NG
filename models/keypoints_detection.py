# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# KeypointsDetection模块，从deformable_detr.py中独立出来
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import List, Tuple
import copy


class KeypointsDetection(nn.Module):
    """KeypointsDetection模块，用于关键点检测"""
    
    def __init__(self, backbone_num_channels, num_keypoints, backbone_type='image'):
        """
        初始化KeypointsDetection模块
        
        Args:
            backbone_num_channels: backbone输出通道数
            num_keypoints: 关键点数量
            backbone_type: backbone类型，'image'或'video'
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.keypoints_head = KeypointsHead(dim_in=backbone_num_channels[0], num_keypoints=num_keypoints)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        """
        前向传播
        
        Args:
            backbone_outputs: backbone的输出，包含global_features和local_features
            metas: 元数据
            is_training: 是否为训练模式
            
        Returns:
            包含pred_keypoints_heatmap的字典
        """
        global_features, local_features = backbone_outputs['global_features'], backbone_outputs['local_features']
        
        if self.backbone_type == 'video':
            global_features = global_features[:, 0]
            local_features = local_features[:, 0]

        # 将local_features从(N, L, D)重塑为(N, D, H, W)
        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        
        keypoints_heatmap = self.keypoints_head(reshaped_local_features)

        out = {'pred_keypoints_heatmap': keypoints_heatmap}
        return out


class KeypointsHead(nn.Module):
    """关键点检测头，使用子像素卷积进行可学习上采样"""
    
    def __init__(self, dim_in=768, num_keypoints=58):
        super(KeypointsHead, self).__init__()
        self.dim_in = dim_in
        # 使用子像素卷积(pixel shuffle)进行可学习上采样
        # 这种方法参数效率更高，通常比转置卷积效果更好
        
        # Stage 1: (768, 32, 32) -> (192, 64, 64) 使用2x上采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),  # 4x通道数用于2x上采样
            nn.PixelShuffle(2),  # (192*4, 32, 32) -> (192, 64, 64)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: (192, 64, 64) -> (96, 128, 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 96 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (96*4, 64, 64) -> (96, 128, 128)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: (96, 128, 128) -> (48, 256, 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 48 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (48*4, 128, 128) -> (48, 256, 256)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_keypoints),
            nn.ReLU(inplace=True)
        )
        
        # 最终卷积层: (num_keypoints, 256, 256) -> (num_keypoints, 256, 256)
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        使用可学习上采样的前向传播
        
        Args:
            x: 输入特征，形状为(N, 768, 32, 32)
            
        Returns:
            output: 重构的特征，形状为(N, num_keypoints, 256, 256)
        """
        x = self.stage1(x)      # (N, 192, 64, 64)
        x = self.stage2(x)      # (N, 96, 128, 128)
        x = self.stage3(x)      # (N, num_keypoints, 256, 256)
        x = self.final_conv(x)  # (N, num_keypoints, 256, 256)
        
        return x


class KeypointsDetectionLoss(nn.Module):
    """KeypointsDetection的损失计算类"""
    
    def __init__(self, weight_dict):
        """
        创建损失计算器
        
        Args:
            weight_dict: 包含损失权重的字典
        """
        super().__init__()
        self.weight_dict = weight_dict

    def forward(self, outputs, targets, **kwargs):
        """
        计算损失
        
        Args:
            outputs: 模型输出，包含pred_keypoints_heatmap
            targets: 目标标签列表
            
        Returns:
            losses: 损失字典
            weight_dict: 权重字典
            None: 为了与其他loss函数接口保持一致
        """
        losses = {}
        
        # Keypoints loss
        keypoints_gt = torch.stack([t["keypoints_target"] for t in targets], dim=0)
        keypoints_mask = torch.stack([t["keypoints_mask"] for t in targets], dim=0)
        keypoints_pred = outputs["pred_keypoints_heatmap"]
        
        # 使用MSE loss计算keypoints损失
        loss_keypoints = F.mse_loss(keypoints_pred, keypoints_gt, reduction='none')
        # 应用mask，只在有效的keypoints上计算损失
        loss_keypoints = (loss_keypoints * keypoints_mask.unsqueeze(-1).unsqueeze(-1)).sum() / (keypoints_mask.sum() + 1e-6)
        losses["loss_keypoints"] = loss_keypoints
        
        return losses, self.weight_dict


class KeypointsDetectionMetrics(nn.Module):
    """KeypointsDetection的指标计算类"""
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        """重置收集的数据"""
        self.keypoints_metrics_data = {
            'accuracy': [],     # 准确度
            'precision': [],     # 精确度
            'recall': [],        # 召回率
            'f1': [],      # F1分数
            'valid_count': 0      # 有效样本数量
        }

    # def get_keypoints_from_heatmap_batch_maxpool(
    #         self, 
    #         heatmap: torch.Tensor,
    #         scale: int = 2,
    #         max_keypoints: int = 1,
    #         min_keypoint_pixel_distance: int = 15,
    #         return_scores: bool = True,
    # ):
    #     """从批量热力图中快速提取关键点，使用maxpooling"""
    #     batch_size, n_channels, height, width = heatmap.shape
    #     device = heatmap.device
        
    #     # 获取每个通道的max_keypoints个局部最大值(使用maxpool)
    #     kernel_size = min_keypoint_pixel_distance + 1
    #     kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    #     padding = kernel_size // 2
        
    #     # 通过在边界填充最高可能值来排除边界关键点
    #     heatmap_padded = F.pad(heatmap, [padding] * 4, mode='constant', value=1.0)
        
    #     # 应用maxpool以获得局部最大值
    #     local_maxima = F.max_pool2d(heatmap_padded, kernel_size, stride=1, padding=0)
        
    #     # 创建掩码以识别局部最大值
    #     maxima_mask = (heatmap == local_maxima).float()
        
    #     # 从热力图中提取top-k(可能包括非局部最大值，如果峰值数量少于max_keypoints)
    #     scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        
    #     # 将展平的索引转换回2D坐标
    #     y_coords = (indices // width) * scale
    #     x_coords = (indices % width) * scale
        
    #     # 应用局部最大值掩码
    #     flat_maxima_mask = maxima_mask.view(batch_size, n_channels, -1)
    #     maxima_scores = torch.gather(flat_maxima_mask, 2, indices)
        
    #     # 将非局部最大值的分数设置为0
    #     scores = scores * maxima_scores
        
    #     if return_scores:
    #         return list(zip(x_coords.tolist(), y_coords.tolist(), scores.tolist()))
    #     else:
    #         return list(zip(x_coords.tolist(), y_coords.tolist()))

    def get_keypoints_from_heatmap_batch_maxpool(
            self, 
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 1,
            min_keypoint_pixel_distance: int = 15,
            return_scores: bool = True,
    ):
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling."""
        batch_size, n_channels, height, width = heatmap.shape

        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = min_keypoint_pixel_distance
        
        # exclude border keypoints by padding with highest possible value
        padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
        max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
        
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)

        # moving to CPU
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        
        filtered_indices = []
        for batch_idx in range(batch_size):
            batch_keypoints = []
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                batch_keypoints.append(locs)
            filtered_indices.append(batch_keypoints)

        return torch.tensor(filtered_indices)

    # def calculate_keypoints_metrics(self, gt, pred, mask, conf_th=0.1, dist_th=5):
    #     """
    #     计算关键点检测的指标
        
    #     Args:
    #         gt: ground truth关键点 (batch_size, num_keypoints, num_coords)
    #         pred: 预测的关键点 (batch_size, num_keypoints, num_coords)
    #         mask: 有效性掩码 (batch_size, num_keypoints)
    #         conf_th: 置信度阈值
    #         dist_th: 距离阈值
            
    #     Returns:
    #         包含accuracy, precision, recall, f1的指标字典
    #     """
    #     batch_size = gt.shape[0]
    #     all_metrics = {
    #         'accuracy': [],
    #         'precision': [],
    #         'recall': [],
    #         'f1': []
    #     }
        
    #     for b in range(batch_size):
    #         gt_batch = gt[b]  # (num_keypoints, num_coords)
    #         pred_batch = pred[b]  # (num_keypoints, num_coords) 
    #         mask_batch = mask[b]  # (num_keypoints,)
            
    #         # 只考虑有效的关键点
    #         valid_indices = mask_batch > 0
    #         if valid_indices.sum() == 0:
    #             continue
                
    #         gt_valid = gt_batch[valid_indices]  # (valid_keypoints, num_coords)
    #         pred_valid = pred_batch[valid_indices]  # (valid_keypoints, num_coords)
            
    #         # 计算预测关键点的置信度(这里假设在第3列，如果没有则使用1.0)
    #         if gt_valid.shape[1] > 2:
    #             pred_conf = pred_valid[:, 2]
    #         else:
    #             pred_conf = torch.ones(pred_valid.shape[0], device=pred_valid.device)
            
    #         # 应用置信度阈值
    #         conf_mask = pred_conf > conf_th
    #         if conf_mask.sum() == 0:
    #             # 没有高置信度的预测
    #             all_metrics['accuracy'].append(0.0)
    #             all_metrics['precision'].append(0.0)
    #             all_metrics['recall'].append(0.0)
    #             all_metrics['f1'].append(0.0)
    #             continue
            
    #         # 计算距离
    #         gt_coords = gt_valid[:, :2]  # (valid_keypoints, 2)
    #         pred_coords = pred_valid[:, :2]  # (valid_keypoints, 2)
            
    #         # 计算欧氏距离
    #         distances = torch.norm(gt_coords - pred_coords, dim=1)  # (valid_keypoints,)
            
    #         # 计算正确预测(距离小于阈值且置信度高)
    #         correct_predictions = (distances < dist_th) & conf_mask
            
    #         # 计算指标
    #         num_correct = correct_predictions.sum().item()
    #         num_predicted = conf_mask.sum().item()
    #         num_gt = len(gt_valid)
            
    #         accuracy = num_correct / num_gt if num_gt > 0 else 0.0
    #         precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    #         recall = num_correct / num_gt if num_gt > 0 else 0.0
    #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
    #         all_metrics['accuracy'].append(accuracy)
    #         all_metrics['precision'].append(precision)
    #         all_metrics['recall'].append(recall)
    #         all_metrics['f1'].append(f1)
        
    #     # 计算平均指标
    #     if all_metrics['accuracy']:
    #         avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    #     else:
    #         avg_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
    #     return avg_metrics

    def calculate_keypoints_metrics(self, gt, pred, mask, conf_th=0.1, dist_th=5):
        """计算keypoints的metrics"""
        # Convert mask to geometry mask (excluding last channel if needed)
        geometry_mask = (mask > 0).cpu()
            
        # Ensure gt and pred are on CPU for computation
        gt = gt.cpu()
        pred = pred.cpu()
        
        batch_size = gt.shape[0]
        batch_metrics = []
        
        for batch_idx in range(batch_size):
            if not geometry_mask[batch_idx].any():
                # No valid keypoints in this sample
                batch_metrics.append((0.0, 0.0, 0.0, 0.0))
                continue
                
            # Get valid keypoints for this batch
            valid_mask = geometry_mask[batch_idx]
            
            # Extract positions and confidence scores
            gt_batch = gt[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            pred_batch = pred[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            
            # Check confidence thresholds
            gt_conf_mask = gt_batch[:, -1] > conf_th  # GT confidence > threshold
            pred_conf_mask = pred_batch[:, -1] > conf_th  # Pred confidence > threshold
            
            # Calculate distances between predicted and GT positions
            gt_pos = gt_batch[:, :2]  # [valid_kp, 2] (x, y)
            pred_pos = pred_batch[:, :2]  # [valid_kp, 2] (x, y)
            distances = torch.norm(pred_pos - gt_pos, dim=1)  # [valid_kp]
            
            # Count true positives, false positives, and false negatives
            true_positives = ((distances < dist_th) & pred_conf_mask & gt_conf_mask).sum().item()
            true_negatives = (~pred_conf_mask & ~gt_conf_mask).sum().item()
            false_positives = ((pred_conf_mask & ~gt_conf_mask) | ((distances >= dist_th) & pred_conf_mask & gt_conf_mask)).sum().item()
            false_negatives = (~pred_conf_mask & gt_conf_mask).sum().item()
            
            # Calculate metrics
            total_valid = valid_mask.sum().item()
            if total_valid > 0:
                accuracy = (true_positives + true_negatives) / total_valid
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                accuracy = precision = recall = f1 = 0.0
                
            batch_metrics.append((accuracy, precision, recall, f1))
        
        return batch_metrics

    def compute_keypoints_metrics(self, pred_keypoints_heatmap, targets):
        """
        计算keypoints的metrics
        
        Args:
            pred_keypoints_heatmap: 预测的keypoints heatmap [B, num_keypoints, H, W]
            targets: 目标数据列表
        """
        # 检查是否有keypoints数据
        if pred_keypoints_heatmap is None:
            return
            
        # 获取GT keypoints heatmap和mask
        keypoints_gt_list = [t.get("keypoints_target", None) for t in targets]
        keypoints_mask_list = [t.get("keypoints_mask", None) for t in targets]
        
        # 过滤掉None值
        valid_indices = [i for i, (kp_gt, kp_mask) in enumerate(zip(keypoints_gt_list, keypoints_mask_list)) 
                        if kp_gt is not None and kp_mask is not None]
        
        if not valid_indices:
            return  # 没有有效的keypoints数据
        
        # 只处理有效的数据
        keypoints_gt = torch.stack([keypoints_gt_list[i] for i in valid_indices])
        keypoints_mask = torch.stack([keypoints_mask_list[i] for i in valid_indices])
        pred_keypoints_valid = pred_keypoints_heatmap[valid_indices]
        
        # 从heatmap中提取keypoints
        kp_gt = self.get_keypoints_from_heatmap_batch_maxpool(keypoints_gt[:,:-1,:,:], return_scores=True, max_keypoints=1)
        kp_pred = self.get_keypoints_from_heatmap_batch_maxpool(pred_keypoints_valid[:,:-1,:,:], return_scores=True, max_keypoints=1)
        
        # 计算metrics
        batch_metrics = self.calculate_keypoints_metrics(kp_gt, kp_pred, keypoints_mask[:, :-1])
        
        # 收集metrics
        for accuracy, precision, recall, f1 in batch_metrics:
            self.keypoints_metrics_data['accuracy'].append(accuracy)
            self.keypoints_metrics_data['precision'].append(precision)
            self.keypoints_metrics_data['recall'].append(recall)
            self.keypoints_metrics_data['f1'].append(f1)
        
        self.keypoints_metrics_data['valid_count'] += len(batch_metrics)
        

    def update(self, outputs, targets):
        """
        更新指标数据
        
        Args:
            outputs: 模型输出
            targets: 目标标签
        """
        self.compute_keypoints_metrics(outputs['pred_keypoints_heatmap'], targets)

    def gather_metrics_data(self, accelerator):
        """聚合多进程的指标数据"""
        gathered_data = {}
        
        for key, values in self.keypoints_metrics_data.items():
            if key == 'valid_count':
                gathered_data[key] = accelerator.gather_for_metrics([values])
            else:
                gathered_data[key] = accelerator.gather_for_metrics(values)
        
        return gathered_data

    def compute_metrics_from_gathered_data(self, gathered_keypoints_metrics):
        """从聚合的数据计算最终指标"""
        def flatten_data(data):
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return result
            else:
                return data if isinstance(data, list) else [data]
        
        # 计算keypoints指标
        keypoints_results = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            values = flatten_data(gathered_keypoints_metrics[metric_name])
            keypoints_results[f'keypoints_{metric_name}'] = sum(values) / len(values)
                    
        # 添加有效样本数量
        keypoints_results['keypoints_valid_samples'] = sum(gathered_keypoints_metrics['valid_count'])
        
        return keypoints_results

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        前向传播，计算指标
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            
        Returns:
            指标字典
        """
        # 更新数据
        self.update(outputs, targets)
        
        # 返回当前批次的指标(如果需要)
        return {}

    def compute_final_metrics(self, accelerator):
        """
        计算最终指标(在所有数据收集完成后调用)
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            最终指标字典
        """
        # 聚合多进程数据
        gathered_keypoints_metrics = self.gather_metrics_data(accelerator)
        
        # 计算最终指标
        if accelerator.is_main_process:
            final_metrics = self.compute_metrics_from_gathered_data(gathered_keypoints_metrics)
            return final_metrics
        else:
            return {}


def build_keypoints_detection_head(config: dict):
    """构建KeypointsDetection头"""
    backbone_num_channels = [config['BACKBONE_NUM_CHANNELS'] if 'BACKBONE_NUM_CHANNELS' in config else 768]
    num_keypoints = config['NUM_KEYPOINTS']
    backbone_type = config['BACKBONE_TYPE']
    
    return KeypointsDetection(
        backbone_num_channels=backbone_num_channels,
        num_keypoints=num_keypoints,
        backbone_type=backbone_type
    )


def build_keypoints_detection_loss(config: dict):
    """构建KeypointsDetection损失函数"""
    weight_dict = {
        "loss_keypoints": config["GSR_KEYPOINTS_LOSS_WEIGHT"]
    }
    
    return KeypointsDetectionLoss(weight_dict=weight_dict)


def build_keypoints_detection_metrics(config: dict):
    """构建KeypointsDetection指标计算器"""
    return KeypointsDetectionMetrics() 