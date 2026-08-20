import torch
import numpy as np
import pickle
import zipfile
import os
import json
from typing import Dict, List, Tuple

from .geometry import compute_iou_batch

# Constants for pitch dimensions
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_X_MARGIN = 10.0
PITCH_Y_MARGIN = 5.0
COORD_X_MIN = -((PITCH_LENGTH / 2) + PITCH_X_MARGIN)  # -62.5
COORD_X_MAX = ((PITCH_LENGTH / 2) + PITCH_X_MARGIN)   # 62.5
COORD_Y_MIN = -((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)   # -39.0
COORD_Y_MAX = ((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)    # 39.0

# Feature mapping tables
ROLE_MAP = {'player': 0, 'goalkeeper': 1, 'referee': 2, 'unknown': 3}
TEAM_MAP = {'left': 0, 'right': 1, 'nan': 2}


def flip_x_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """
    将坐标的x值翻转
    
    Args:
        coords (torch.Tensor): 坐标 [..., 2]
    
    Returns:
        torch.Tensor: x翻转后的坐标
    """
    flipped_coords = coords.clone()
    flipped_coords[..., 0] = -flipped_coords[..., 0]
    return flipped_coords


def flip_y_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """
    将坐标的y值翻转
    
    Args:
        coords (torch.Tensor): 坐标 [..., 2]
    
    Returns:
        torch.Tensor: y翻转后的坐标
    """
    flipped_coords = coords.clone()
    flipped_coords[..., 1] = -flipped_coords[..., 1]
    return flipped_coords


def flip_teams(teams: torch.Tensor) -> torch.Tensor:
    """
    翻转队伍标签（left变right，right变left）
    
    Args:
        teams (torch.Tensor): 队伍标签，one-hot编码 [..., 3]，
                             teams[..., 0] = left
                             teams[..., 1] = right
                             teams[..., 2] = nan
    
    Returns:
        torch.Tensor: 翻转后的队伍标签
    """
    flipped_teams = teams.clone()
    # 交换left和right
    left_values = flipped_teams[..., 0].clone()
    flipped_teams[..., 0] = flipped_teams[..., 1]  # left变为right
    flipped_teams[..., 1] = left_values            # right变为left
    return flipped_teams


def load_pipeline_data(pipeline_dir, pklz_files, vid) -> tuple:
    """加载单个视频的管道预测数据"""
    with zipfile.ZipFile(os.path.join(pipeline_dir, pklz_files[0])) as zf:
        with zf.open(f'{vid}.pkl') as f:
            pipeline_preds = pickle.load(f)
        with zf.open(f'{vid}_image.pkl') as f:
            pipeline_image_preds = pickle.load(f)
    return pipeline_preds, pipeline_image_preds


def load_gt_data(gt_root, split, vid) -> Dict:
    """加载单个视频的GT数据"""
    path = os.path.join(gt_root, split, f'SNGS-{vid}', 'Labels-GameState.json')
    with open(path) as f:
        gt_data = json.load(f)
    return gt_data


def process_pipeline_frame(frame_preds, frame_image_preds, max_detections, role_map, team_map, 
                          coord_x_min, coord_x_max, coord_y_min, coord_y_max) -> Dict:
    """处理单帧管道数据"""
    num_det = min(len(frame_preds), max_detections)
    frame_data = {
        'feats': torch.zeros((max_detections, 256)),
        'bbox_ltwh': torch.zeros((max_detections, 4)),
        'coords': torch.zeros((max_detections, 2)),
        'roles': torch.zeros((max_detections, len(role_map))),
        'teams': torch.zeros((max_detections, len(team_map))),
        'JNs': torch.zeros(max_detections, dtype=torch.long),
        'track_ids': torch.zeros(max_detections, dtype=torch.long),
        'visible_mask': torch.zeros(max_detections, dtype=torch.bool),
    }
    
    det_ids = frame_preds.index.tolist()
    detection_mappings = {}
    
    if num_det > 0:
        frame_data['visible_mask'][:num_det] = True
        # 处理feat
        frame_data['feats'][:num_det] = torch.from_numpy(np.stack(frame_preds['embeddings'].values).squeeze(1))
        
        # 处理bbox
        bboxes = np.stack(frame_preds['bbox_ltwh'].values)
        frame_data['bbox_ltwh'][:num_det] = torch.from_numpy(bboxes)
        
        # 处理坐标
        coords = []
        for i, bbox_pitch in enumerate(frame_preds['bbox_pitch'].values):
            if isinstance(bbox_pitch, dict):
                # 对bbox_pitch进行clip，确保坐标在合理范围内
                x = np.clip(bbox_pitch['x_bottom_middle'], coord_x_min, coord_x_max)
                y = np.clip(bbox_pitch['y_bottom_middle'], coord_y_min, coord_y_max)
                coords.append([x, y])
            else:
                coords.append([0.0, 0.0])
                frame_data['visible_mask'][i] = False
        frame_data['coords'][:num_det] = torch.tensor(coords)
        
        # 处理角色和队伍
        for i, (role, team) in enumerate(zip(frame_preds['role'].values, frame_preds['team'].values)):
            if i >= num_det:
                break
            # 处理role，如果不在预定义的role_map中，则设为'unknown'(值为3)
            role_idx = role_map.get(role, role_map['unknown'])
            frame_data['roles'][i, role_idx] = 1.0
            # 处理team
            team_idx = team_map.get(team, team_map['nan'])
            frame_data['teams'][i, team_idx] = 1.0
            
            # 处理球衣号
            jn = frame_preds['jersey_number'].values[i]
            frame_data['JNs'][i] = int(jn) if jn and 0 < int(jn) < 100 else 0
            
            # 处理track_id
            frame_data['track_ids'][i] = int(frame_preds['track_id'].values[i]) if frame_preds['track_id'].values[i] is not None and 0 < int(frame_preds['track_id'].values[i]) < 150 else 0
            
        for i in range(num_det):
            if frame_data['visible_mask'][i]:
                detection_mappings[i] = det_ids[i]
    
    # 处理相机参数
    if frame_image_preds is not None:
        camera_params = frame_image_preds['parameters'].iloc[0]
        camera_params_valid_mask = True if 'x_focal_length' in camera_params else False
        frame_data['camera_params_valid_mask'] = camera_params_valid_mask
        
        if camera_params_valid_mask:
            x_focal_length = camera_params['x_focal_length']
            y_focal_length = camera_params['y_focal_length']
            principal_point = np.array(camera_params['principal_point'])
            position_meters = np.array(camera_params['position_meters'])
            rotation = np.array(camera_params['rotation_matrix'])
            
            # 创建K, R, T矩阵
            frame_data['T'] = torch.from_numpy(position_meters)
            frame_data['R'] = torch.from_numpy(rotation)
            frame_data['K'] = torch.tensor([[x_focal_length, 0, principal_point[0]],
                                        [0, y_focal_length, principal_point[1]],
                                        [0, 0, 1]])
    
    return frame_data, detection_mappings


def process_pipeline_video(vid, preds, image_preds, max_frames, max_detections, role_map, team_map,
                          coord_x_min=COORD_X_MIN, coord_x_max=COORD_X_MAX, 
                          coord_y_min=COORD_Y_MIN, coord_y_max=COORD_Y_MAX) -> Tuple[Dict, Dict, List[int]]:
    """处理单个视频的管道数据，不进行normalization，将原始数据返回"""
    image_ids = sorted(image_preds.id.unique().tolist(), key=lambda x: int(x))
    assert len(image_ids) == max_frames, f"Video {vid} has {len(image_ids)} frames"
    
    # 初始化输出容器
    processed = {
        'feats': torch.zeros((max_frames, max_detections, 256)),
        'bbox_ltwh': torch.zeros((max_frames, max_detections, 4)),
        'coords': torch.zeros((max_frames, max_detections, 2)),
        'roles': torch.zeros((max_frames, max_detections, len(role_map))),
        'teams': torch.zeros((max_frames, max_detections, len(team_map))),
        'JNs': torch.zeros((max_frames, max_detections), dtype=torch.long),
        'track_ids': torch.zeros((max_frames, max_detections), dtype=torch.long),
        'visible_mask': torch.zeros((max_frames, max_detections), dtype=torch.bool),
        'camera_params_valid_mask': torch.zeros(max_frames, dtype=torch.bool),
        'K': torch.zeros((max_frames, 3, 3)),
        'R': torch.zeros((max_frames, 3, 3)),
        'T': torch.zeros((max_frames, 3)),
    }
    
    # Detection ID mappings to restore predictions
    detection_mappings = {}
    
    # 逐帧处理
    for t, image_id in enumerate(image_ids[:max_frames]):
        frame_preds = preds[preds.image_id == image_id]
        frame_image_preds = image_preds[image_preds.id == image_id]
        frame_data, detection_mappings[t] = process_pipeline_frame(
            frame_preds, frame_image_preds, max_detections, 
            role_map, team_map, coord_x_min, coord_x_max, coord_y_min, coord_y_max
        )
        
        # 复制数据到结果张量
        for key in frame_data:
            if key == 'camera_params_valid_mask':
                processed[key][t] = frame_data[key]
            elif key in ['K', 'R', 'T'] and 'camera_params_valid_mask' in frame_data and frame_data['camera_params_valid_mask']:
                processed[key][t] = frame_data[key]
            elif key in processed:
                processed[key][t] = frame_data[key]
    
    return processed, detection_mappings


def process_gt_video(gt_data, image_ids, max_frames, max_detections, role_map, team_map,
                    coord_x_min, coord_x_max, coord_y_min, coord_y_max, load_embeddings=False, embeddings=None) -> Dict:
    """处理单个视频的GT数据"""
    processed = {
        'bbox_ltwh': torch.zeros((max_frames, max_detections, 4)),
        'coords': torch.zeros((max_frames, max_detections, 2)),
        'roles': torch.zeros((max_frames, max_detections, len(role_map))),
        'teams': torch.zeros((max_frames, max_detections, len(team_map))),
        'JNs': torch.zeros((max_frames, max_detections), dtype=torch.long),
        'track_ids': torch.zeros((max_frames, max_detections), dtype=torch.long),
        'visible_mask': torch.zeros((max_frames, max_detections), dtype=torch.bool),
    }
    if load_embeddings:
        assert embeddings is not None
        processed['feats'] = torch.zeros((max_frames, max_detections, 256))
    
    # 创建图像ID到索引的映射
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
    
    is_labeled = {image_id: False for image_id in image_ids}
    for image_anno in gt_data['images']:
        image_id = image_anno['image_id']
        is_labeled[image_id] = image_anno['has_labeled_person'] and image_anno['has_labeled_pitch'] and image_anno['has_labeled_camera']
    
    # 遍历所有标注
    for anno in gt_data['annotations']:
        if anno['supercategory'] != 'object' or anno['attributes']['role'] == 'ball':
            continue
        
        anno_id = anno['id']
        img_id = anno['image_id']
        if not is_labeled[img_id]:
            continue
        assert img_id in image_id_to_idx, f"Image ID {img_id} not found in image_id_to_idx"
            
        frame_idx = image_id_to_idx[img_id]
        det_idx = processed['visible_mask'][frame_idx].sum().item()
        if det_idx >= max_detections:
            continue
            
        # 填充bbox
        bbox = anno['bbox_image']
        processed['bbox_ltwh'][frame_idx, det_idx] = torch.tensor([
            bbox['x'], bbox['y'], bbox['w'], bbox['h']
        ])
        
        # 填充坐标
        bbox_pitch = anno['bbox_pitch']
        # 对bbox_pitch进行clip，确保坐标在合理范围内
        x = np.clip(bbox_pitch['x_bottom_middle'], coord_x_min, coord_x_max)
        y = np.clip(bbox_pitch['y_bottom_middle'], coord_y_min, coord_y_max)
        processed['coords'][frame_idx, det_idx] = torch.tensor([x, y])
        
        # 填充角色和队伍
        role = anno['attributes']['role']
        role_idx = role_map.get(role, role_map['unknown'])
        processed['roles'][frame_idx, det_idx, role_idx] = 1.0
        
        team = anno['attributes']['team']
        team_idx = team_map.get(team, team_map['nan'])
        processed['teams'][frame_idx, det_idx, team_idx] = 1.0
            
        # 填充球衣号
        jn = anno['attributes']['jersey']
        processed['JNs'][frame_idx, det_idx] = int(jn) if jn and 0 < int(jn) < 100 else 0
        
        # 填充track_id
        processed['track_ids'][frame_idx, det_idx] = anno['track_id'] if 0 < anno['track_id'] < 150 else 0
        
        # 更新可见性掩码
        processed['visible_mask'][frame_idx, det_idx] = True
        
        if load_embeddings:
            processed['feats'][frame_idx, det_idx] = torch.tensor(embeddings[img_id][anno_id])
    
    return processed


def compute_video_matches(input_data, gt_data, max_frames, iou_threshold=0.0) -> Dict:
    """计算视频级的匹配映射"""
    match_mapping = {}
    for frame_idx in range(max_frames):
        # 获取预测和GT的可见区域
        pred_mask = input_data['visible_mask'][frame_idx]
        gt_mask = gt_data['visible_mask'][frame_idx]
        
        pred_boxes = input_data['bbox_ltwh'][frame_idx][pred_mask].numpy()
        gt_boxes = gt_data['bbox_ltwh'][frame_idx][gt_mask].numpy()
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # 计算IoU矩阵
        iou_matrix = compute_iou_batch(pred_boxes, gt_boxes)
        
        # 贪心匹配
        matches = {}
        used_gt_indices = set()
        for pred_idx in range(iou_matrix.shape[0]):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx in range(iou_matrix.shape[1]):
                if gt_idx in used_gt_indices:
                    continue
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold and best_gt_idx != -1:
                # 记录绝对索引
                abs_pred_idx = pred_mask.nonzero()[:, 0][pred_idx].item()
                abs_gt_idx = gt_mask.nonzero()[:, 0][best_gt_idx].item()
                matches[abs_pred_idx] = abs_gt_idx
                used_gt_indices.add(best_gt_idx)
        
        if matches:
            match_mapping[frame_idx] = matches
    return match_mapping


def permute_gt_video(gt_data, match_mapping) -> Dict:
    """根据匹配映射重新排列GT数据，并更新visible_mask"""
    # 初始化permuted_gt为零张量，与gt_data相同的shape和type
    permuted_gt = {}
    for key in gt_data:
        if key == 'visible_mask':
            permuted_gt[key] = torch.zeros_like(gt_data[key], dtype=torch.bool)
        else:
            permuted_gt[key] = torch.zeros_like(gt_data[key])
    
    # 根据匹配关系重排
    for frame_idx, frame_matches in match_mapping.items():
        for pred_idx, gt_idx in frame_matches.items():
            # 将匹配的GT数据移动到与预测相同的位置
            for key in permuted_gt:
                permuted_gt[key][frame_idx, pred_idx] = gt_data[key][frame_idx, gt_idx]
    
    return permuted_gt


def augment_camera_params(R, T, rotation_std, translation_std, rotation_mean=0.):
    """
    对相机参数进行数据增强
    
    参数:
    - R: 旋转矩阵 [3, 3]
    - T: 平移向量 [3]
    - rotation_std: 旋转角度标准差（度）
    - translation_std: 平移标准差
    - rotation_mean: 旋转角度平均值（度）
    
    返回:
    - R_aug: 增强后的旋转矩阵
    - T_aug: 增强后的平移向量
    """
    # 旋转增强
    # 从正态分布采样旋转角度变化（度）
    rotation_noise_deg = torch.normal(
        mean=torch.tensor(rotation_mean), 
        std=torch.tensor(rotation_std)
    )
    
    # 随机旋转轴
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    
    # 将角度转换为弧度
    angle_rad = rotation_noise_deg * torch.pi / 180.0
    
    # 构建旋转矩阵（Rodrigues公式）
    K = torch.zeros(3, 3)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    R_noise = torch.eye(3) + torch.sin(angle_rad) * K + (1 - torch.cos(angle_rad)) * (K @ K)
    
    # 应用旋转噪声
    R_aug = R @ R_noise.to(R.dtype)
    
    # 平移增强
    # 从正态分布采样xyz方向的平移变化
    translation_noise = torch.normal(
        mean=torch.zeros(3),
        std=torch.tensor(translation_std)
    )
    
    # 应用平移噪声
    T_aug = T + translation_noise
    
    return R_aug, T_aug


def backproject_using_ray_method(K, R, T, image_point):
    """
    通过射线相交法将图像点反投影到世界坐标系z=0平面
    
    Args:
        K (torch.Tensor): 3x3内参矩阵
        R (torch.Tensor): 3x3旋转矩阵
        T (torch.Tensor): 3D平移向量（相机位置）
        image_point (tuple): 图像坐标 (u, v)
    
    Returns:
        torch.Tensor: 世界坐标系点 [X, Y, 0.0]
    """
    u, v = image_point
    
    # 转换为齐次坐标 (3D向量)
    Pi_homogeneous = torch.tensor([u, v, 1.0], device=K.device)
    
    # 计算射线方向
    K_inv = torch.inverse(K)                # 内参逆矩阵
    P_cam_normalized = K_inv @ Pi_homogeneous  # 相机坐标系归一化坐标
    R_inv = torch.inverse(R)                # 旋转逆矩阵
    ray_dir_world = R_inv @ P_cam_normalized # 世界坐标系射线方向
    
    # 计算射线参数 (s = (0 - T_z) / dir_z)
    s = (-T[2]) / ray_dir_world[2]
    
    # 计算交点坐标
    X = T[0] + s * ray_dir_world[0]
    Y = T[1] + s * ray_dir_world[1]
    
    return torch.tensor([X, Y, 0.0], device=K.device)


def normalize_coordinates(coords, normalization_method, coord_mean=None, coord_std=None,
                         coord_x_min=COORD_X_MIN, coord_x_max=COORD_X_MAX, 
                         coord_y_min=COORD_Y_MIN, coord_y_max=COORD_Y_MAX):
    """标准化坐标"""
    if normalization_method == "z_score" and coord_mean is not None and coord_std is not None:
        # Z-score normalization
        return (coords - torch.from_numpy(coord_mean).float()) / torch.from_numpy(coord_std).float()
    elif normalization_method == "minmax":
        # Min-max normalization to [-1, 1]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Scale x coordinates to [-1, 1]
        x_norm = 2.0 * (x_coords - coord_x_min) / (coord_x_max - coord_x_min) - 1.0
        
        # Scale y coordinates to [-1, 1]
        y_norm = 2.0 * (y_coords - coord_y_min) / (coord_y_max - coord_y_min) - 1.0
        
        return torch.stack([x_norm, y_norm], dim=1)
    else:
        return coords


def denormalize_coordinates(coords, normalization_method, coord_mean=None, coord_std=None,
                           coord_x_min=COORD_X_MIN, coord_x_max=COORD_X_MAX, 
                           coord_y_min=COORD_Y_MIN, coord_y_max=COORD_Y_MAX):
    """将标准化的坐标还原为原始坐标"""
    if normalization_method == "z_score" and coord_mean is not None and coord_std is not None:
        # 还原Z-score标准化
        return coords * torch.from_numpy(coord_std).float() + torch.from_numpy(coord_mean).float()
    elif normalization_method == "minmax":
        # 还原min-max标准化
        x_orig = (coords[:, 0] + 1.0) / 2.0 * (coord_x_max - coord_x_min) + coord_x_min
        y_orig = (coords[:, 1] + 1.0) / 2.0 * (coord_y_max - coord_y_min) + coord_y_min
        return torch.stack([x_orig, y_orig], dim=1)
    else:
        return coords 