# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Camera模块，从deformable_detr.py中独立出来
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import List, Tuple, Optional
from models.deformable_detr.vggt.head_act import activate_pose
from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

class Camera(nn.Module):
    """Camera检测模块，用于相机姿态估计"""
    
    def __init__(self, backbone_num_channels, backbone_type='image'):
        """
        初始化Camera模块
        
        Args:
            backbone_num_channels: backbone输出通道数
            backbone_type: backbone类型，'image'或'video'
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.camera_head = ConvCameraHead(input_channels=backbone_num_channels[0])

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        """
        前向传播
        
        Args:
            backbone_outputs: backbone的输出，包含global_features和local_features
            metas: 元数据
            is_training: 是否为训练模式
            
        Returns:
            包含quaternion, translation, fov的字典
        """
        global_features, local_features = backbone_outputs['global_features'], backbone_outputs['local_features']
        
        bs, num_frames = None, None
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = local_features.shape
            # 将 [bs, num_frames, ...] reshape为 [bs*num_frames, ...]
            local_features = local_features.reshape(bs * num_frames, *local_features.shape[2:])
            if global_features is not None:
                global_features = global_features.reshape(bs * num_frames, -1)

        # 将local_features从(N, L, D)重塑为(N, D, H, W)
        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        
        quaternion, translation, fov = self.camera_head(reshaped_local_features)

        # 如果是video模式，将输出reshape回[bs, num_frames, ...]
        if self.backbone_type == 'video':
            quaternion = quaternion.reshape(bs, num_frames, *quaternion.shape[1:])
            translation = translation.reshape(bs, num_frames, *translation.shape[1:])
            fov = fov.reshape(bs, num_frames, *fov.shape[1:])

        out = {
            'quaternion': quaternion,
            'translation': translation, 
            'fov': fov
        }
        return out


class ConvCameraHead(nn.Module):
    """卷积相机头，用于预测相机姿态参数"""
    
    def __init__(
        self, 
        input_channels=768,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "linear",
        ):
        super(ConvCameraHead, self).__init__()
        
        self.input_channels = input_channels
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        
        # Define convolutional layers similar to PoseCNN
        self.convs = {}
        self.convs[0] = nn.Conv2d(input_channels, 256, 7, 2, 3)
        self.convs[1] = nn.Conv2d(256, 256, 5, 2, 2)
        self.convs[2] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[3] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[4] = nn.Conv2d(256, 256, 3, 2, 1)
        
        # Final prediction layer: 4 (quaternion) + 3 (translation) + 2 (fov) = 9
        self.camera_conv = nn.Conv2d(256, 9, 1)
        
        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        
        # Convert to ModuleList for proper parameter registration
        self.net = nn.ModuleList(list(self.convs.values()))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the camera head"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for camera head
        Args:
            x: input features of shape (N, C, H, W)
        Returns:
            quaternion: (N, 4) - camera rotation as quaternion
            translation: (N, 3) - camera translation
            fov: (N, 2) - field of view parameters
        """
        # Apply convolutional layers with ReLU activation
        for i in range(self.num_convs):
            x = self.convs[i](x)
            x = self.relu(x)
        
        # Final prediction layer
        x = self.camera_conv(x)
        
        # Global average pooling to get a single prediction per image
        x = x.mean(3).mean(2)  # Shape: (N, 9)
        
        x = activate_pose(x, self.trans_act, self.quat_act, self.fl_act)
        
        # Split the output into quaternion, translation, and fov
        quaternion = x[:, :4]  # First 4 values
        translation = x[:, 4:7]  # Next 3 values  
        fov = x[:, 7:9]  # Last 2 values
        
        # Normalize quaternion to unit length
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        return quaternion, translation, fov


class CameraLoss(nn.Module):
    """Camera的损失计算类"""
    
    def __init__(self, weight_dict, backbone_type='image'):
        """
        创建损失计算器
        
        Args:
            weight_dict: 包含损失权重的字典
            backbone_type: 'image' or 'video'
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.backbone_type = backbone_type

    def forward(self, outputs, targets, **kwargs):
        """
        计算损失
        
        Args:
            outputs: 模型输出，包含quaternion, translation, fov
            targets: 目标标签列表
                    For video mode: list of lists, where each inner list contains annotations for each frame
            
        Returns:
            losses: 损失字典
            weight_dict: 权重字典
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
        
        valid_camera_mask = torch.stack([t["valid_camera"] for t in targets_for_loss], dim=0)
        if valid_camera_mask.any():
            quaternion_gt = torch.stack([t["quaternion"] for t in targets_for_loss], dim=0)[valid_camera_mask]
            translation_gt = torch.stack([t["translation"] for t in targets_for_loss], dim=0)[valid_camera_mask]
            fov_hw_gt = torch.stack([t["fov_hw"] for t in targets_for_loss], dim=0)[valid_camera_mask]
            
            # 对于video模式，pred的shape可能是[bs, num_frames, ...]，需要reshape
            quaternion_pred = outputs["quaternion"]
            translation_pred = outputs["translation"]
            fov_hw_pred = outputs["fov"]
            
            if self.backbone_type == 'video' and len(quaternion_pred.shape) >= 3:
                bs, num_frames = quaternion_pred.shape[:2]
                quaternion_pred = quaternion_pred.reshape(bs * num_frames, *quaternion_pred.shape[2:])
                translation_pred = translation_pred.reshape(bs * num_frames, *translation_pred.shape[2:])
                fov_hw_pred = fov_hw_pred.reshape(bs * num_frames, *fov_hw_pred.shape[2:])
            
            quaternion_pred = quaternion_pred[valid_camera_mask]
            translation_pred = translation_pred[valid_camera_mask]
            fov_hw_pred = fov_hw_pred[valid_camera_mask]
            
            cur_pred_pose_enc = torch.cat([translation_pred, quaternion_pred, fov_hw_pred], dim=-1)
            gt_pose_encoding = torch.cat([translation_gt, quaternion_gt, fov_hw_gt], dim=-1)
            
            loss_T, loss_R, loss_fl = camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="huber")
            losses["loss_T"] = loss_T
            losses["loss_R"] = loss_R
            losses["loss_fl"] = loss_fl
        else:
            losses["loss_T"] = torch.tensor(0.0, device=outputs['quaternion'].device)
            losses["loss_R"] = torch.tensor(0.0, device=outputs['quaternion'].device)
            losses["loss_fl"] = torch.tensor(0.0, device=outputs['quaternion'].device)
        
        return losses, self.weight_dict


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    """
    计算单个相机损失
    
    Args:
        cur_pred_pose_enc: 预测的姿态编码 [N, 9] (translation[3] + quaternion[4] + fov[2])
        gt_pose_encoding: 真实的姿态编码 [N, 9]
        loss_type: 损失类型 ("l1", "l2", "huber")
        
    Returns:
        loss_T: 平移损失
        loss_R: 旋转损失  
        loss_fl: 视野损失
    """
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = F.smooth_l1_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3], reduction='none')
        loss_R = F.smooth_l1_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7], reduction='none')
        loss_fl = F.smooth_l1_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:], reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    检查loss_tensor是否包含inf或nan。如果包含，用零替换这些值并打印损失张量的名称。

    Args:
        loss_tensor (torch.Tensor): 要检查的损失张量
        loss_name (str): 损失的名称（用于诊断打印）
        hard_max (float): 硬性最大值限制

    Returns:
        torch.Tensor: 检查并修复的损失张量，inf/nan被替换为0
    """
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


class CameraMetrics(nn.Module):
    """Camera的指标计算类"""
    
    def __init__(self, backbone_type='image'):
        super().__init__()
        self.backbone_type = backbone_type
        self.reset()
        
    def reset(self):
        """重置收集的数据"""
        self.camera_metrics_data = {
            'translation_errors': [],  # 欧氏距离误差
            'rotation_errors': [],     # 角度误差(degrees)
            'fov_errors': [],         # FOV误差
            'valid_count': 0          # 有效样本数量
        }

    def quaternion_angular_difference(self, q1, q2):
        """
        计算两个四元数之间的角度差（以度为单位）
        
        Args:
            q1: 第一个四元数 [N, 4] (w, x, y, z)
            q2: 第二个四元数 [N, 4] (w, x, y, z)
            
        Returns:
            角度差（度）[N]
        """
        # 确保四元数是归一化的
        q1 = F.normalize(q1, p=2, dim=1)
        q2 = F.normalize(q2, p=2, dim=1)
        
        # 计算点积
        dot_product = torch.sum(q1 * q2, dim=1)
        
        # 限制到有效范围，避免数值误差
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        
        # 计算角度（弧度）
        angle_rad = 2 * torch.acos(dot_product)
        
        # 转换为度
        angle_deg = angle_rad * 180.0 / math.pi
        
        return angle_deg

    def compute_camera_metrics(self, outputs, targets):
        """
        计算相机参数的指标
        
        Args:
            outputs: 预测的相机参数字典，包含quaternion, translation, fov
            targets: 目标列表，每个包含camera_params，或list of lists for video mode
            
        Returns:
            指标字典
        """
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
        else:
            targets_for_metrics = targets
        
        # 获取有效相机mask
        valid_camera_mask = torch.stack([t.get("valid_camera", torch.tensor(False)) for t in targets_for_metrics], dim=0)
        
        if not valid_camera_mask.any():
            return  # 没有有效的相机数据
        
        # 获取GT相机参数
        quaternion_gt = torch.stack([t["quaternion"] for t in targets_for_metrics], dim=0)[valid_camera_mask]
        translation_gt = torch.stack([t["translation"] for t in targets_for_metrics], dim=0)[valid_camera_mask]
        fov_hw_gt = torch.stack([t["fov_hw"] for t in targets_for_metrics], dim=0)[valid_camera_mask]
        
        # 获取预测相机参数 - 处理video模式
        quaternion_pred = outputs["quaternion"]
        translation_pred = outputs["translation"]
        fov_hw_pred = outputs["fov"]
        
        # 如果是video模式，reshape预测结果
        if self.backbone_type == 'video' and len(quaternion_pred.shape) >= 3:
            bs, num_frames = quaternion_pred.shape[:2]
            quaternion_pred = quaternion_pred.reshape(bs * num_frames, *quaternion_pred.shape[2:])
            translation_pred = translation_pred.reshape(bs * num_frames, *translation_pred.shape[2:])
            fov_hw_pred = fov_hw_pred.reshape(bs * num_frames, *fov_hw_pred.shape[2:])
        
        # 应用有效相机mask
        quaternion_pred = quaternion_pred[valid_camera_mask]
        translation_pred = translation_pred[valid_camera_mask]
        fov_hw_pred = fov_hw_pred[valid_camera_mask]
        
        # 计算平移误差（欧氏距离）
        translation_errors = torch.norm(translation_pred - translation_gt, dim=-1)  # [N]
        
        # 计算旋转误差（角度差异）
        rotation_errors = self.quaternion_angular_difference(quaternion_pred, quaternion_gt)  # [N]
        
        # 计算FOV误差（L2距离）
        fov_errors = torch.norm(fov_hw_pred - fov_hw_gt, dim=-1)  # [N]
        
        # 转移到CPU并添加到收集器
        self.camera_metrics_data['translation_errors'].extend(translation_errors.cpu().tolist())
        self.camera_metrics_data['rotation_errors'].extend(rotation_errors.cpu().tolist())
        self.camera_metrics_data['fov_errors'].extend(fov_errors.cpu().tolist())
        self.camera_metrics_data['valid_count'] += len(translation_errors)

    def update(self, outputs, targets):
        """
        更新指标数据
        
        Args:
            outputs: 模型输出
            targets: 目标数据
        """
        self.compute_camera_metrics(outputs, targets)

    def gather_metrics_data(self, accelerator):
        """收集所有进程的指标数据"""
        camera_key_list = ['translation_errors', 'rotation_errors', 'fov_errors']
        gathered_camera_metrics = {}
        for key in camera_key_list:
            gathered_camera_metrics[key] = gather_object(self.camera_metrics_data[key])
        gathered_camera_metrics['valid_count'] = gather_object([self.camera_metrics_data['valid_count']])
        return gathered_camera_metrics

    def compute_metrics_from_gathered_data(self, gathered_camera_metrics):
        """从收集的数据计算最终指标"""
        metrics = {}

        # 展平所有进程的相机数据
        all_translation_errors = flatten_data(gathered_camera_metrics['translation_errors'])
        all_rotation_errors = flatten_data(gathered_camera_metrics['rotation_errors'])
        all_fov_errors = flatten_data(gathered_camera_metrics['fov_errors'])
        
        # 计算总的有效样本数
        total_valid_count = sum(gathered_camera_metrics['valid_count'])
        
        if total_valid_count > 0:
            # 计算平移误差统计
            translation_errors = torch.tensor(all_translation_errors, dtype=torch.float32)
            metrics['camera_translation_mae'] = translation_errors.mean().item()  # 平均绝对误差
            metrics['camera_translation_rmse'] = torch.sqrt(translation_errors.pow(2).mean()).item()  # 均方根误差
            metrics['camera_translation_median'] = translation_errors.median().item()  # 中位数误差
            
            # 计算旋转误差统计
            rotation_errors = torch.tensor(all_rotation_errors, dtype=torch.float32)
            metrics['camera_rotation_mae'] = rotation_errors.mean().item()  # 平均绝对角度误差(度)
            metrics['camera_rotation_rmse'] = torch.sqrt(rotation_errors.pow(2).mean()).item()  # 均方根角度误差
            metrics['camera_rotation_median'] = rotation_errors.median().item()  # 中位数角度误差
            
            # 计算FOV误差统计
            fov_errors = torch.tensor(all_fov_errors, dtype=torch.float32)
            metrics['camera_fov_mae'] = fov_errors.mean().item()  # 平均绝对FOV误差
            metrics['camera_fov_rmse'] = torch.sqrt(fov_errors.pow(2).mean()).item()  # 均方根FOV误差
            metrics['camera_fov_median'] = fov_errors.median().item()  # 中位数FOV误差
            
            # 记录样本数量
            metrics['camera_valid_samples'] = total_valid_count
            
            # 计算精度阈值内的准确度
            # 平移误差 < 1.0 的比例
            translation_acc_1 = (translation_errors < 1.0).float().mean().item()
            metrics['camera_translation_acc@1.0'] = translation_acc_1
            
            # 旋转误差 < 5度的比例
            rotation_acc_5 = (rotation_errors < 5.0).float().mean().item()
            metrics['camera_rotation_acc@5deg'] = rotation_acc_5
            
            # 旋转误差 < 10度的比例
            rotation_acc_10 = (rotation_errors < 10.0).float().mean().item()
            metrics['camera_rotation_acc@10deg'] = rotation_acc_10
        else:
            # 没有有效的相机数据
            for metric_name in ['camera_translation_mae', 'camera_translation_rmse', 'camera_translation_median',
                                'camera_rotation_mae', 'camera_rotation_rmse', 'camera_rotation_median', 
                                'camera_fov_mae', 'camera_fov_rmse', 'camera_fov_median',
                                'camera_translation_acc@1.0', 'camera_rotation_acc@5deg', 'camera_rotation_acc@10deg']:
                metrics[metric_name] = 0.0
            metrics['camera_valid_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        前向传播计算指标（评估时使用）
        
        Args:
            outputs: 模型输出
            targets: 目标数据
            
        Returns:
            指标字典
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


def build_camera_head(config: dict):
    """构建Camera检测头"""
    return Camera(
        backbone_num_channels=[config["BACKBONE_HIDDEN_DIM"]],
        backbone_type=config["BACKBONE_TYPE"]
    )


def build_camera_loss(config: dict):
    """构建Camera损失函数"""
    weight_dict = {
        'loss_T': config["CAMERA_T_LOSS_WEIGHT"],
        'loss_R': config["CAMERA_R_LOSS_WEIGHT"],
        'loss_fl': config["CAMERA_FL_LOSS_WEIGHT"],
    }
    
    return CameraLoss(weight_dict=weight_dict, backbone_type=config["BACKBONE_TYPE"])


def build_camera_metrics(config: dict):
    """构建Camera指标计算器"""
    return CameraMetrics(backbone_type=config["BACKBONE_TYPE"])