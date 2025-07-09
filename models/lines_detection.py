# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import os
from typing import List, Tuple
import copy

from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

class LinesDetection(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone_num_channels, num_lines, backbone_type='image', head_type='default', selected_layers=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            head_type: str, 'default' for LinesHead, 'dpt' for DPTLinesHead
            selected_layers: list, 当使用DPT时选择的层索引
        """
        # TODO: find a way to handle positional encoding, strides, channels, etc.
        super().__init__()
        self.backbone_type = backbone_type
        self.head_type = head_type
        self.selected_layers = selected_layers
        
        if head_type == 'dpt':
            self.lines_head = DPTLinesHead(dim_in=backbone_num_channels[0], num_lines=num_lines)
        else:
            self.lines_head = LinesHead(dim_in=backbone_num_channels[0], num_lines=num_lines)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        global_features, local_features = backbone_outputs['global_features'], backbone_outputs['local_features']
        hidden_states = backbone_outputs['hidden_states']
        
        bs, num_frames = None, None
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = local_features.shape
            # 将 [bs, num_frames, ...] reshape为 [bs*num_frames, ...]
            local_features = local_features.reshape(bs * num_frames, *local_features.shape[2:])
            if global_features is not None:
                global_features = global_features.reshape(bs * num_frames, -1)
            if hidden_states is not None:
                hidden_states = [hs.reshape(bs * num_frames, *hs.shape[2:]) for hs in hidden_states]

        if self.head_type == 'dpt':
            # DPT使用多层hidden_states
            # hidden_states是一个tuple/list，包含所有层的输出
            multi_layer_features = []
            
            for layer_idx in self.selected_layers:
                if layer_idx < len(hidden_states):
                    layer_feat = hidden_states[layer_idx]  # (N, L, D)
                    N, L, D = layer_feat.shape
                    # 重塑为2D特征图格式
                    layer_feat = layer_feat.permute(0, 2, 1).contiguous()  # (N, D, L)
                    Hf = Wf = int(math.sqrt(L))
                    layer_feat = layer_feat.reshape(N, D, Hf, Wf)  # (N, D, Hf, Wf)
                    multi_layer_features.append(layer_feat)
            
            lines_heatmap = self.lines_head(multi_layer_features)
        else:
            # 默认使用最后一层local_features
            N, L, D = local_features.shape
            reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
            Hf = Wf = int(math.sqrt(L))
            reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
            
            lines_heatmap = self.lines_head(reshaped_local_features)

        # 如果是video模式，将输出reshape回[bs, num_frames, ...]
        if self.backbone_type == 'video':
            lines_heatmap = lines_heatmap.reshape(bs, num_frames, *lines_heatmap.shape[1:])

        out = {'pred_lines_heatmap': lines_heatmap}
        return out

class LinesDetectionLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, backbone_type='image'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            backbone_type: 'image' or 'video'
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.backbone_type = backbone_type

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      For video mode: list of lists, where each inner list contains annotations for each frame
        """
        losses = {}
        
        # Handle video mode: flatten targets if needed
        if self.backbone_type == 'video':
            # Check if targets is list of lists (video mode)
            if targets and isinstance(targets[0], list):
                # Flatten targets: convert list of list of dicts to list of dicts
                flattened_targets = []
                for batch_targets in targets:
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                targets_for_loss = flattened_targets
            else:
                # Already flattened
                targets_for_loss = targets
        else:
            targets_for_loss = targets
        
        # 检查哪些样本的valid_lines为True
        valid_lines_mask = torch.stack([t["valid_lines"] for t in targets_for_loss], dim=0)  # [batch_size]
        
        if valid_lines_mask.any():
            # 只对valid_lines为True的样本计算loss
            lines_gt = torch.stack([t["lines_target"] for t in targets_for_loss], dim=0)  # [batch_size, num_lines, H, W]
            lines_pred = outputs["pred_lines_heatmap"]  # [batch_size, num_lines, H, W]
            
            # 对于video模式，pred的shape是[bs, num_frames, num_lines, H, W]，需要reshape
            if self.backbone_type == 'video' and len(lines_pred.shape) == 5:
                bs, num_frames = lines_pred.shape[:2]
                lines_pred = lines_pred.reshape(bs * num_frames, *lines_pred.shape[2:])
            
            # 使用mask来过滤有效的样本
            # 扩展mask的维度以匹配lines_gt和lines_pred的维度
            expanded_mask = valid_lines_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [batch_size, 1, 1, 1]
            expanded_mask = expanded_mask.expand_as(lines_gt)  # [batch_size, num_lines, H, W]
            
            # 只计算有效样本的loss
            loss_lines = F.mse_loss(lines_pred * expanded_mask, lines_gt * expanded_mask, reduction='sum')
            # 归一化：除以有效样本数量和每个样本的元素数量
            valid_elements = expanded_mask.sum()
            if valid_elements > 0:
                loss_lines = loss_lines / valid_elements
            else:
                loss_lines = torch.tensor(0.0, device=lines_pred.device, requires_grad=True)
        else:
            # 如果没有有效样本，loss为0
            loss_lines = torch.tensor(0.0, device=outputs["pred_lines_heatmap"].device, requires_grad=True)
        
        losses["loss_lines"] = loss_lines
        
        # losses = {k: (v * self.weight_dict[k] if k in self.weight_dict else v) for k, v in losses.items()}

        return losses, self.weight_dict

class LinesHead(nn.Module):
    def __init__(self, dim_in=768, num_lines=24):
        super(LinesHead, self).__init__()
        self.dim_in = dim_in
        # Using sub-pixel convolution (pixel shuffle) for learnable upsampling
        # This is more parameter-efficient and often works better than transposed convolution
        
        # Stage 1: (768, 32, 32) -> (192, 64, 64) using 2x upsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),  # 4x channels for 2x upsampling
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
            # nn.Conv2d(48, 48, kernel_size=3, padding=1),
            # nn.BatchNorm2d(48),
            nn.Conv2d(48, num_lines, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_lines),
            nn.ReLU(inplace=True)
        )
        
        # Final stage: (24, 512, 512) -> (output_channels, 512, 512)
        self.final_conv = nn.Sequential(
            # nn.Conv2d(24, num_lines, kernel_size=3, padding=1),
            nn.Conv2d(num_lines, num_lines, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
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
        Forward pass using learnable upsampling
        Args:
            x: Input features of shape (N, 768, 32, 32)
        Returns:
            output: Reconstructed features of shape (N, output_channels, 512, 512)
        """
        x = self.stage1(x)      # (N, 192, 64, 64)
        x = self.stage2(x)      # (N, 96, 128, 128)
        x = self.stage3(x)      # (N, 48, 256, 256)
        # x = self.stage4(x)      # (N, 24, 512, 512)
        x = self.final_conv(x)  # (N, output_channels, 512, 512)
        
        return x



class LinesDetectionMetrics(nn.Module):
    """
    计算lines相关的metrics，包括accuracy、precision、recall、F1等指标
    支持多进程聚合和整个数据集上的lines性能计算
    """
    def __init__(self, backbone_type='image'):
        super().__init__()
        self.backbone_type = backbone_type
        
        # 为lines收集数据
        self.lines_metrics_data = {
            'accuracies': [],     # 准确度
            'precisions': [],     # 精确度
            'recalls': [],        # 召回率
            'f1_scores': [],      # F1分数
            'valid_count': 0      # 有效样本数量
        }
        
    def reset(self):
        """重置收集的数据"""
        self.lines_metrics_data = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'valid_count': 0
        }

    def get_keypoints_from_heatmap_batch_maxpool_l(
            self,
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 2,
            min_keypoint_pixel_distance: int = 10,
            return_scores: bool = True,
    ) -> List[List[List[Tuple[int, int]]]]:
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

        Inspired by mmdetection and CenterNet:
        https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

        Args:
            heatmap (torch.Tensor): NxCxHxW heatmap batch
            max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
            min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

            Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
            abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
            rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

        Returns:
            The extracted keypoints for each batch, channel and heatmap; and their scores
        """
        batch_size, n_channels, _, width = heatmap.shape
        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = int((kernel-1)/2)

        max_pooled_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap

        # all values to zero that are not local maxima
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
        # at this point either score > 0.0, in which case the index is a local maximum
        # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

        #  remove top-k that are not local maxima and threshold (if required)
        # thresholding shouldn't be done during training

        #  moving them to CPU now to avoid multiple GPU-mem accesses!
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
        filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

        # have to do this manually as the number of maxima for each channel can be different
        for batch_idx in range(batch_size):
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
                filtered_indices[batch_idx][channel_idx] = locs

        return torch.tensor(filtered_indices)

    def calculate_lines_metrics(self, gt, pred, conf_th=0.1, dist_th=5):
        """计算lines的metrics，按batch处理"""
        # Ensure gt and pred are on CPU for computation
        gt = gt.cpu()
        pred = pred.cpu()
        
        batch_size = gt.shape[0]
        batch_metrics = []
        
        for batch_idx in range(batch_size):
            # Get data for current batch
            gt_batch = gt[batch_idx]  # [num_lines, max_keypoints, 3]
            pred_batch = pred[batch_idx]  # [num_lines, max_keypoints, 3]
            
            # Extract positions and confidence scores
            pred_pos = pred_batch[:, :, :-1]  # [num_lines, max_keypoints, 2]
            gt_pos = gt_batch[:, :, :-1]  # [num_lines, max_keypoints, 2]
            
            pred_mask = torch.all((pred_batch[:, :, -1] > conf_th), dim=-1)  # [num_lines]
            gt_mask = torch.all((gt_batch[:, :, -1] > conf_th), dim=-1)  # [num_lines]
            
            gt_flip = torch.flip(gt_pos, dims=[1])  # [num_lines, max_keypoints, 2]
            
            distances1 = torch.norm(pred_pos - gt_pos, dim=-1)  # [num_lines, max_keypoints]
            distances2 = torch.norm(pred_pos - gt_flip, dim=-1)  # [num_lines, max_keypoints]
            
            distances1_bool = torch.all((distances1 < dist_th), dim=-1)  # [num_lines]
            distances2_bool = torch.all((distances2 < dist_th), dim=-1)  # [num_lines]
            
            # Count true positives, false positives, and false negatives based on distance threshold
            true_positives = ((distances1_bool | distances2_bool) & pred_mask & gt_mask).sum().item()
            true_negatives = (~pred_mask & ~gt_mask).sum().item()
            false_positives = (
                    (pred_mask & ~gt_mask) | ((~distances1_bool & ~distances2_bool) & pred_mask & gt_mask)).sum().item()
            false_negatives = (~pred_mask & gt_mask).sum().item()
            
            # Calculate metrics for this batch
            total_lines = gt_batch.shape[0]
            if total_lines > 0:
                accuracy = (true_positives + true_negatives) / total_lines
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                accuracy = precision = recall = f1 = 0.0
            
            batch_metrics.append((accuracy, precision, recall, f1))
        
        return batch_metrics

    def compute_lines_metrics(self, pred_lines_heatmap, targets):
        """
        计算lines的metrics
        
        Args:
            pred_lines_heatmap: 预测的lines heatmap [B, num_lines, H, W] or [B, num_frames, num_lines, H, W]
            targets: 目标数据列表 or list of lists for video mode
        """
        # 检查是否有lines数据
        if pred_lines_heatmap is None:
            return
        
        # Handle video mode: flatten targets if needed
        if self.backbone_type == 'video':
            # Check if targets is list of lists (video mode)
            if targets and isinstance(targets[0], list):
                # Flatten targets: convert list of list of dicts to list of dicts
                flattened_targets = []
                for batch_targets in targets:
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                targets_for_metrics = flattened_targets
            else:
                # Already flattened
                targets_for_metrics = targets
                
            # Reshape pred_lines_heatmap from [bs, num_frames, ...] to [bs*num_frames, ...]
            if len(pred_lines_heatmap.shape) == 5:  # [bs, num_frames, num_lines, H, W]
                bs, num_frames = pred_lines_heatmap.shape[:2]
                pred_lines_heatmap = pred_lines_heatmap.reshape(bs * num_frames, *pred_lines_heatmap.shape[2:])
        else:
            targets_for_metrics = targets
        
        # 只处理valid_lines为True的样本
        valid_lines_mask = torch.stack([t["valid_lines"] for t in targets_for_metrics], dim=0)  # [batch_size]
        if not valid_lines_mask.any():
            return  # 没有有效的lines数据
            
        # 获取GT lines heatmap，只处理有效样本
        lines_gt_list = [t["lines_target"] for i, t in enumerate(targets_for_metrics) if valid_lines_mask[i]]
        if not lines_gt_list:
            return
            
        # 只处理有效的数据
        lines_gt = torch.stack(lines_gt_list, dim=0)
        pred_lines_heatmap_valid = pred_lines_heatmap[valid_lines_mask]
        
        # 从heatmap中提取lines
        l_gt = self.get_keypoints_from_heatmap_batch_maxpool_l(lines_gt[:,:-1,:,:], return_scores=True, max_keypoints=2)
        lines_pred = self.get_keypoints_from_heatmap_batch_maxpool_l(pred_lines_heatmap_valid[:,:-1,:,:], return_scores=True, max_keypoints=2)
        
        # 计算metrics
        batch_metrics = self.calculate_lines_metrics(l_gt, lines_pred)
        
        # 收集metrics
        for accuracy, precision, recall, f1 in batch_metrics:
            self.lines_metrics_data['accuracies'].append(accuracy)
            self.lines_metrics_data['precisions'].append(precision)
            self.lines_metrics_data['recalls'].append(recall)
            self.lines_metrics_data['f1_scores'].append(f1)
        
        self.lines_metrics_data['valid_count'] += len(batch_metrics)

    def update(self, outputs, targets):
        """
        在当前batch上更新lines metrics
        
        Args:
            outputs: 模型输出，包含pred_lines_heatmap
            targets: 真实标注，包含lines_target
        """
        self.compute_lines_metrics(outputs['pred_lines_heatmap'], targets)

    def gather_metrics_data(self, accelerator):
        """
        在所有进程间聚合lines metrics数据
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            gathered_lines_metrics: 聚合后的lines metrics数据
        """
        # 聚合lines metrics数据
        lines_key_list = ['accuracies', 'precisions', 'recalls', 'f1_scores']
        gathered_lines_metrics = {}
        for key in lines_key_list:
            gathered_lines_metrics[key] = gather_object(self.lines_metrics_data[key])
        gathered_lines_metrics['valid_count'] = gather_object([self.lines_metrics_data['valid_count']])
        
        return gathered_lines_metrics

    def compute_metrics_from_gathered_data(self, gathered_lines_metrics):
        """
        从聚合的数据计算最终的lines metrics
        
        Args:
            gathered_lines_metrics: 聚合的lines metrics数据
            
        Returns:
            dict: 包含lines metrics的字典
        """
        metrics = {}
        
        # 计算lines metrics
        if gathered_lines_metrics is not None:
            # 展平所有进程的lines数据
            all_accuracies = flatten_data(gathered_lines_metrics['accuracies'])
            all_precisions = flatten_data(gathered_lines_metrics['precisions'])
            all_recalls = flatten_data(gathered_lines_metrics['recalls'])
            all_f1_scores = flatten_data(gathered_lines_metrics['f1_scores'])
            
            # 计算总的有效样本数
            total_valid_count = sum(gathered_lines_metrics['valid_count'])
            
            if total_valid_count > 0 and len(all_accuracies) > 0:
                # 计算lines metrics的平均值
                accuracies = torch.tensor(all_accuracies, dtype=torch.float32)
                precisions = torch.tensor(all_precisions, dtype=torch.float32)
                recalls = torch.tensor(all_recalls, dtype=torch.float32)
                f1_scores = torch.tensor(all_f1_scores, dtype=torch.float32)
                
                metrics['lines_accuracy'] = accuracies.mean().item()
                metrics['lines_precision'] = precisions.mean().item()
                metrics['lines_recall'] = recalls.mean().item()
                metrics['lines_f1'] = f1_scores.mean().item()
                
                # 计算高精度阈值下的性能
                # 精度 > 0.8 的比例
                high_acc_ratio = (accuracies > 0.8).float().mean().item()
                metrics['lines_high_accuracy_ratio'] = high_acc_ratio
                
                # F1 > 0.7 的比例
                high_f1_ratio = (f1_scores > 0.7).float().mean().item()
                metrics['lines_high_f1_ratio'] = high_f1_ratio
                
                # 记录样本数量
                metrics['lines_valid_samples'] = total_valid_count
            else:
                # 没有有效的lines数据
                for metric_name in ['lines_accuracy', 'lines_precision', 'lines_recall', 'lines_f1',
                                  'lines_high_accuracy_ratio', 'lines_high_f1_ratio']:
                    metrics[metric_name] = 0.0
                metrics['lines_valid_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        计算lines metrics (保持向后兼容)
        这个方法现在只是调用update来收集数据
        """
        self.update(outputs, targets)
        # 返回空字典，实际的metrics计算在compute_final_metrics中进行
        return {}
        
    def compute_final_metrics(self, accelerator):
        """
        计算最终的metrics（在所有数据收集完成后调用）
        
        Args:
            accelerator: Accelerator实例
            
        Returns:
            dict: 包含lines metrics的字典
        """
        # 聚合所有进程的lines metrics数据
        gathered_lines_metrics = self.gather_metrics_data(accelerator)
        
        # 只在主进程计算metrics
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_data(gathered_lines_metrics)
        else:
            return {}


def build_lines_detection_head(config: dict):
    """
    构建SoccerNetGSR Lines任务的head
    """
    # 获取backbone输出通道数
    backbone_num_channels = [768]  # 根据SigLIP backbone的输出通道数
    num_lines = config["NUM_LINES"]
    backbone_type = config["BACKBONE_TYPE"]
    head_type = config.get("LINES_HEAD_TYPE", "default")  # 默认使用原来的LinesHead
    selected_layers = config["DPT_SELECTED_LAYERS"]  # DPT选择的层
    
    head = LinesDetection(
        backbone_num_channels=backbone_num_channels,
        num_lines=num_lines,
        backbone_type=backbone_type,
        head_type=head_type,
        selected_layers=selected_layers
    )
    return head

def build_lines_detection_loss(config: dict):
    """
    构建lines criterion (loss function)
    """
    weight_dict = {
        "loss_lines": config["GSR_LINES_LOSS_WEIGHT"]
    }
    
    criterion = LinesDetectionLoss(weight_dict=weight_dict, backbone_type=config["BACKBONE_TYPE"])
    return criterion

def build_lines_detection_metrics(config: dict):
    """
    构建lines metrics计算器
    """
    metrics = LinesDetectionMetrics(backbone_type=config["BACKBONE_TYPE"])
    return metrics

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[x[i].shape[2], x[i].shape[3]],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        # extra = config['MODEL']['EXTRA']
        # 硬编码EXTRA配置
        extra = {
            'FINAL_CONV_KERNEL': 1,
            'STAGE1': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 1,
                'BLOCK': 'BOTTLENECK',
                'NUM_BLOCKS': [4],
                'NUM_CHANNELS': [64],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [48, 96],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192, 384],
                'FUSE_METHOD': 'SUM'
            }
        }
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sf = nn.Softmax(dim=1)
        self.layer1 = self._make_layer(Bottleneck, self.inplanes, self.inplanes, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        final_inp_channels = sum(pre_stage_channels) + self.inplanes

        self.head = nn.Sequential(nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=24,
                kernel_size=extra['FINAL_CONV_KERNEL']),
            nn.Sigmoid()))
        self.init_weights()

    def _make_head(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.head(x)

        return x

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, task, metas, text=None):
        # h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x_skip = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Head Part
        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self._make_head(x, x_skip)

        out = {}
        out['pred_lines_heatmap'] = x
        return {'SoccerNetGSR_Lines': out}

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            # print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                # logger.info(
                #     '=> loading {} pretrained model {}'.format(k, pretrained))
                #print('=> loading {} pretrained model {}'.format(k, pretrained))
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)


class DPTLinesHead(nn.Module):
    """
    DPT-based Lines Head for dense line prediction
    采用DPT架构进行更好的密集预测性能
    
    DPT特点：
    1. 多层特征融合（第2、5、8、11层）
    2. 渐进式特征解码：32x32 -> 64x64 -> 128x128 -> 256x256 
    3. 更强的特征表示能力
    """
    def __init__(self, dim_in=768, num_lines=24, hidden_dim=256):
        super(DPTLinesHead, self).__init__()
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim
        self.num_lines = num_lines
        
        # DPT-style projection layers for different transformer layers
        # 为4个不同层级创建投影层
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(dim_in, hidden_dim, kernel_size=1) for _ in range(4)
        ])
        
        # Stage 1: 32x32 融合最深两层特征 (layer11 + layer8)
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1->2: 32x32 -> 64x64
        self.upsample_stage1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 64x64 融合上采样特征和layer5特征
        self.stage2_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2->3: 64x64 -> 128x128
        self.upsample_stage2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: 128x128 融合上采样特征和layer2特征
        self.stage3_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3->4: 128x128 -> 256x256
        self.upsample_stage3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层：256x256 -> 256x256
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, num_lines, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, multi_layer_features):
        """
        Forward pass using DPT-based architecture with progressive upsampling
        Args:
            multi_layer_features: List of 4 feature maps from different layers
                                Each feature map has shape (N, 768, 32, 32)
                                [layer2, layer5, layer8, layer11]
        Returns:
            output: Reconstructed features of shape (N, num_lines, 256, 256)
        """
        # 对每层特征进行投影
        projected_features = []
        for i, layer_feat in enumerate(multi_layer_features):
            projected_feat = self.proj_layers[i](layer_feat)  # (N, hidden_dim, 32, 32)
            projected_features.append(projected_feat)
        
        # 按深度顺序排列：[layer2, layer5, layer8, layer11]
        feat_layer2 = projected_features[0]   # [N, hidden_dim, 32, 32]
        feat_layer5 = projected_features[1]   # [N, hidden_dim, 32, 32]
        feat_layer8 = projected_features[2]   # [N, hidden_dim, 32, 32]
        feat_layer11 = projected_features[3]  # [N, hidden_dim, 32, 32]
        
        # Stage 1: 32x32 - 融合最深两层特征 (layer11 + layer8)
        stage1_concat = torch.cat([feat_layer11, feat_layer8], dim=1)  # [N, hidden_dim*2, 32, 32]
        stage1_fused = self.stage1_fusion(stage1_concat)  # [N, hidden_dim, 32, 32]
        
        # Stage 1->2: 32x32 -> 64x64 上采样
        stage1_upsampled = self.upsample_stage1(stage1_fused)  # [N, hidden_dim, 64, 64]
        
        # Stage 2: 64x64 - 融合上采样特征和layer5特征
        # 首先将layer5特征上采样到64x64
        feat_layer5_up = F.interpolate(feat_layer5, size=(64, 64), mode='bilinear', align_corners=False)
        stage2_concat = torch.cat([stage1_upsampled, feat_layer5_up], dim=1)  # [N, hidden_dim*2, 64, 64]
        stage2_fused = self.stage2_fusion(stage2_concat)  # [N, hidden_dim, 64, 64]
        
        # Stage 2->3: 64x64 -> 128x128 上采样
        stage2_upsampled = self.upsample_stage2(stage2_fused)  # [N, hidden_dim, 128, 128]
        
        # Stage 3: 128x128 - 融合上采样特征和layer2特征
        # 首先将layer2特征上采样到128x128
        feat_layer2_up = F.interpolate(feat_layer2, size=(128, 128), mode='bilinear', align_corners=False)
        stage3_concat = torch.cat([stage2_upsampled, feat_layer2_up], dim=1)  # [N, hidden_dim*2, 128, 128]
        stage3_fused = self.stage3_fusion(stage3_concat)  # [N, hidden_dim, 128, 128]
        
        # Stage 3->4: 128x128 -> 256x256 最终上采样
        stage3_upsampled = self.upsample_stage3(stage3_fused)  # [N, hidden_dim//2, 256, 256]
        
        # 最终输出
        output = self.final_conv(stage3_upsampled)  # [N, num_lines, 256, 256]

        return output
