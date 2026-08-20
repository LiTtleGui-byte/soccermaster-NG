import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
import json
import os
import zipfile
from tqdm import tqdm
import warnings

from .dataset_utils import (
    PITCH_LENGTH, PITCH_WIDTH, PITCH_X_MARGIN, PITCH_Y_MARGIN,
    COORD_X_MIN, COORD_X_MAX, COORD_Y_MIN, COORD_Y_MAX,
    ROLE_MAP, TEAM_MAP,
    load_pipeline_data, load_gt_data, process_pipeline_video,
    process_gt_video, compute_video_matches, permute_gt_video,
    augment_camera_params, backproject_using_ray_method,
    normalize_coordinates, denormalize_coordinates,
    flip_x_coordinates, flip_y_coordinates, flip_teams
)

class SoccerSequenceDataset(Dataset):
    def __init__(self,
                 pipeline_outputs_root: str,
                 pipeline_exp_name: str,
                 pipeline_exp_name_test: str,
                 gt_root: str,
                 metadata_path: str, 
                 split: str,
                 pipeline_exp_name_valid: str = None,
                 max_frames: int = 750,
                 max_clip_frames: int = 50,
                 max_detections_per_frame: int = 30,
                 simulate_missing: bool = False,
                 normalize_bbox: bool = True,
                 normalize_coords: bool = True,
                 normalization_method: str = "z_score",
                 bbox_mean: List[float] = None,
                 bbox_std: List[float] = None,
                 coord_mean: List[float] = None,
                 coord_std: List[float] = None,
                 use_all_frames_as_start: bool = False,
                 augment_coords: bool = False,
                 augment_coords_mode: str = 'individual',
                 augment_coords_sigma: float = 1.0,
                 rotation_std: float = 5.0,
                 translation_std: float = 0.1,
                 flip_x_prob: float = 0.0,
                 flip_y_prob: float = 0.0,
                 max_affinity_frame_distance: int = 25,
                 track_enabled: bool = False,
                 precompute_track_masks: bool = True,
                 random_augment_track_ids: bool = False,
                 random_track_id_change_prob: float = 0.1,
                 random_augment_roles: bool = False,
                 random_role_change_prob: float = 0.1,
                 camera_drift_prob: float = 0.03,
                 drift_rotation_std: float = 2.0,
                 drift_translation_std: float = 1.0,
                 wipe_mode: str = 'none',
                 wipe_random_prob: float = 0.03,
                 wipe_continuous_prob: float = 0.05):
        super().__init__()
        
        # 初始化元数据和路径
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            self.metadata['valid'] = self.metadata['validation']
        assert split in ['train', 'valid', 'test']
        self.split = split
        self.vid_id_list = [vid["name"].split('-')[1] for vid in self.metadata[split]]
        
        # 初始化参数
        self.max_frames = max_frames
        self.max_clip_frames = max_clip_frames
        self.max_detections = max_detections_per_frame
        self.simulate_missing = simulate_missing
        self.use_all_frames_as_start = use_all_frames_as_start
        self.iou_threshold = 0.0
        self.normalize_bbox = normalize_bbox
        self.normalize_coords = normalize_coords
        self.normalization_method = normalization_method
        assert normalization_method in ["z_score", "minmax"], f"Unknown normalization method: {normalization_method}"
        
        # 新增：最大帧间距参数
        self.max_affinity_frame_distance = max_affinity_frame_distance
        
        # 新增：轨迹关联任务是否启用
        self.track_enabled = track_enabled
        
        # 新增：是否预计算轨迹关联掩码
        self.precompute_track_masks = precompute_track_masks
        
        # Track ID augmentation parameters
        self.random_augment_track_ids = random_augment_track_ids
        self.random_track_id_change_prob = random_track_id_change_prob
        
        # Role augmentation parameters
        self.random_augment_roles = random_augment_roles
        self.random_role_change_prob = random_role_change_prob
        
        # 坐标增强参数
        self.augment_coords = augment_coords
        self.augment_coords_mode = augment_coords_mode
        assert any(mode in augment_coords_mode for mode in ['individual', 'frame', 'camera', 'drift']), f"Unknown augment_coords_mode: {augment_coords_mode}"
        self.augment_coords_sigma = augment_coords_sigma
        
        # 翻转增强参数
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob
        
        # 相机参数增强参数
        self.rotation_std = rotation_std
        self.translation_std = translation_std
        
        # 相机漂移增强参数
        self.camera_drift_prob = camera_drift_prob
        self.drift_rotation_std = drift_rotation_std
        self.drift_translation_std = drift_translation_std
        
        # 帧擦除增强参数
        self.wipe_mode = wipe_mode
        assert wipe_mode in ['none', 'random', 'continuous', 'random+continuous'], f"Unknown wipe_mode: {wipe_mode}"
        self.wipe_random_prob = wipe_random_prob
        self.wipe_continuous_prob = wipe_continuous_prob
        
        # 初始化球场尺寸和边界
        self.pitch_length = PITCH_LENGTH
        self.pitch_width = PITCH_WIDTH
        self.pitch_x_margin = PITCH_X_MARGIN
        self.pitch_y_margin = PITCH_Y_MARGIN
        self.coord_x_min = COORD_X_MIN
        self.coord_x_max = COORD_X_MAX
        self.coord_y_min = COORD_Y_MIN
        self.coord_y_max = COORD_Y_MAX
        
        # 初始化normalization参数
        self.bbox_mean = np.array(bbox_mean) if bbox_mean is not None else None
        self.bbox_std = np.array(bbox_std) if bbox_std is not None else None
        self.coord_mean = np.array(coord_mean) if coord_mean is not None else None
        self.coord_std = np.array(coord_std) if coord_std is not None else None

        # 特征映射表
        self.role_map = ROLE_MAP
        self.team_map = TEAM_MAP
        
        # 保存路径信息，而不是预加载数据
        self.pipeline_outputs_root = pipeline_outputs_root
        # Choose the appropriate pipeline_exp_name based on split
        if split == 'train':
            self.pipeline_exp_name = pipeline_exp_name
        elif split == 'valid' and pipeline_exp_name_valid:
            # Use valid-specific pipeline if available
            self.pipeline_exp_name = pipeline_exp_name_valid
        else:
            # Default to test pipeline for other splits
            self.pipeline_exp_name = pipeline_exp_name_test
        self.gt_root = gt_root
        
        # 找到pipeline文件
        self.pipeline_dir = os.path.join(pipeline_outputs_root, self.pipeline_exp_name, 'states')
        self.pklz_files = [f for f in os.listdir(self.pipeline_dir) if f.endswith('.pklz')]
        assert len(self.pklz_files) > 0, "No pipeline files found"
        
        # 如果需要预计算统计量，则进行一次数据扫描
        if (self.normalize_bbox and (self.bbox_mean is None or self.bbox_std is None)) or \
           (self.normalize_coords and self.normalization_method == "z_score" and (self.coord_mean is None or self.coord_std is None)):
            # self._compute_normalization_stats()
            # 如果需要归一化但没有提供统计量，则直接报错
            if self.normalize_bbox and (self.bbox_mean is None or self.bbox_std is None):
                raise ValueError("normalize_bbox is True but bbox_mean or bbox_std is not provided")
            if self.normalize_coords and self.normalization_method == "z_score" and (self.coord_mean is None or self.coord_std is None):
                raise ValueError("normalize_coords is True with z_score but coord_mean or coord_std is not provided")
            
        # 创建索引列表，每个视频分成多个clip
        self.clip_indices = []
        for vid_idx, vid in enumerate(self.vid_id_list):
            if self.use_all_frames_as_start and split != 'test':
                # 使用每一帧作为clip的起始位置
                for start_frame in range(self.max_frames):
                    self.clip_indices.append((vid_idx, start_frame))
            else:
                # 原来的逻辑：按照max_clip_frames切割
                num_clips = (self.max_frames + self.max_clip_frames - 1) // self.max_clip_frames
                for clip_idx in range(num_clips):
                    start_frame = clip_idx * self.max_clip_frames
                    if start_frame < self.max_frames:
                        self.clip_indices.append((vid_idx, start_frame))

    def _compute_normalization_stats(self):
        """计算数据集的统计量，用于归一化"""
        print("Computing normalization statistics...")
        all_bboxes = []
        all_coords = []
        
        with zipfile.ZipFile(os.path.join(self.pipeline_dir, self.pklz_files[0])) as zf:
            for vid in tqdm(self.vid_id_list[:min(10, len(self.vid_id_list))]):  # 使用部分数据计算统计量
                with zf.open(f'{vid}.pkl') as f:
                    preds = pickle.load(f)
                with zf.open(f'{vid}_image.pkl') as f:
                    image_preds = pickle.load(f)
                
                image_ids = sorted(image_preds.id.unique().tolist(), key=lambda x: int(x))
                image_ids = image_ids[:self.max_frames]
                
                if self.normalize_bbox and (self.bbox_mean is None or self.bbox_std is None):
                    for image_id in image_ids:
                        frame_preds = preds[preds.image_id == image_id]
                        if len(frame_preds) > 0:
                            bboxes = np.stack(frame_preds['bbox_ltwh'].values)
                            all_bboxes.append(bboxes)
                
                if self.normalize_coords and self.normalization_method == "z_score" and (self.coord_mean is None or self.coord_std is None):
                    for image_id in image_ids:
                        frame_preds = preds[preds.image_id == image_id]
                        if len(frame_preds) > 0:
                            coords = []
                            for bbox_pitch in frame_preds['bbox_pitch'].values:
                                if isinstance(bbox_pitch, dict):
                                    coords.append([bbox_pitch['x_bottom_middle'], bbox_pitch['y_bottom_middle']])
                            if coords:
                                all_coords.append(np.array(coords))
        
        if self.normalize_bbox and (self.bbox_mean is None or self.bbox_std is None) and all_bboxes:
            all_bboxes = np.concatenate(all_bboxes, axis=0)
            self.bbox_mean = all_bboxes.mean(axis=0)
            self.bbox_std = all_bboxes.std(axis=0)
            self.bbox_std[self.bbox_std == 0] = 1  # 避免除零
        
        if self.normalize_coords and self.normalization_method == "z_score" and (self.coord_mean is None or self.coord_std is None) and all_coords:
            all_coords = np.concatenate(all_coords, axis=0)
            self.coord_mean = all_coords.mean(axis=0)
            self.coord_std = all_coords.std(axis=0)
            self.coord_std[self.coord_std == 0] = 1  # 避免除零

    def _load_pipeline_data(self, vid: str) -> tuple:
        """加载单个视频的管道预测数据"""
        return load_pipeline_data(self.pipeline_dir, self.pklz_files, vid)

    def _load_gt_data(self, vid: str) -> Dict:
        """加载单个视频的GT数据"""
        return load_gt_data(self.gt_root, self.split, vid)

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> Dict:
        # 获取当前索引对应的视频ID和起始帧
        vid_idx, start_frame = self.clip_indices[idx]
        vid = self.vid_id_list[vid_idx]
        end_frame = min(start_frame + self.max_clip_frames, self.max_frames)
        actual_clip_frames = end_frame - start_frame
        
        # 在获取数据时才加载和处理
        # 加载管道数据
        preds, image_preds = self._load_pipeline_data(vid)
        image_ids = sorted(image_preds.id.unique().tolist(), key=lambda x: int(x))
        
        # 处理管道数据 - 不进行normalization
        input_data, detection_mappings = process_pipeline_video(
            vid, preds, image_preds,
            self.max_frames, self.max_detections,
            self.role_map, self.team_map,
            coord_x_min=self.coord_x_min, coord_x_max=self.coord_x_max, 
            coord_y_min=self.coord_y_min, coord_y_max=self.coord_y_max
        )
        
        # 加载并处理GT数据
        gt_data = self._load_gt_data(vid)
        gt_batch = process_gt_video(
            gt_data, image_ids,
            self.max_frames, self.max_detections,
            self.role_map, self.team_map,
            self.coord_x_min, self.coord_x_max, self.coord_y_min, self.coord_y_max
        )
        
        # 计算匹配映射
        match_mapping = compute_video_matches(input_data, gt_batch, self.max_frames, self.iou_threshold)
        
        # 根据映射排列GT数据
        permuted_gt = permute_gt_video(gt_batch, match_mapping)
        
        # 提取clip数据
        clip_data = {}
        for key in input_data:
            clip_data[key] = input_data[key][start_frame:end_frame]
        
        # 添加GT数据
        for key in permuted_gt:
            clip_data[f'gt_{key}'] = permuted_gt[key][start_frame:end_frame]
        
        # 在训练模式下进行数据增强
        if self.split != 'test':
            # 1. 决定是否执行x翻转
            do_flip_x = torch.rand(1).item() < self.flip_x_prob
            if do_flip_x:
                # 翻转坐标的x值
                clip_data['coords'] = flip_x_coordinates(clip_data['coords'])
                # 翻转GT坐标的x值
                if 'gt_coords' in clip_data:
                    clip_data['gt_coords'] = flip_x_coordinates(clip_data['gt_coords'])
                # 翻转队伍标签（left变right，right变left）
                clip_data['teams'] = flip_teams(clip_data['teams'])
                if 'gt_teams' in clip_data:
                    clip_data['gt_teams'] = flip_teams(clip_data['gt_teams'])
            
            # 2. 决定是否执行y翻转
            do_flip_y = torch.rand(1).item() < self.flip_y_prob
            if do_flip_y:
                # 翻转坐标的y值
                clip_data['coords'] = flip_y_coordinates(clip_data['coords'])
                # 翻转GT坐标的y值
                if 'gt_coords' in clip_data:
                    clip_data['gt_coords'] = flip_y_coordinates(clip_data['gt_coords'])
            
            # 3. 坐标噪声增强
            if self.augment_coords:
                # 根据增强模式选择不同的增强方法
                if 'camera' in self.augment_coords_mode or 'drift' in self.augment_coords_mode:
                    # 相机参数增强方式 - 使用相机映射重新计算坐标
                    for t in range(actual_clip_frames):
                        # 确保当前帧有有效的相机参数
                        if not clip_data['camera_params_valid_mask'][t]:
                            continue
                        
                        # 获取原始相机参数
                        K = clip_data['K'][t]
                        R_orig = clip_data['R'][t]
                        T_orig = clip_data['T'][t]
                        
                        do_aug_camera = False
                        
                        if 'drift' in self.augment_coords_mode:
                            do_drift = torch.rand(1).item() < self.camera_drift_prob
                        else:
                            do_drift = False
                        if do_drift:
                            # 使用较大的标准差进行相机漂移增强
                            R_aug, T_aug = augment_camera_params(R_orig, T_orig, self.drift_rotation_std, self.drift_translation_std)
                            do_aug_camera = True
                        elif 'camera' in self.augment_coords_mode:
                            # 使用标准的相机参数增强
                            R_aug, T_aug = augment_camera_params(R_orig, T_orig, self.rotation_std, self.translation_std)
                            do_aug_camera = True
                            
                        if do_aug_camera:
                            # 更新增强后的相机参数
                            clip_data['R'][t] = R_aug
                            clip_data['T'][t] = T_aug
                            
                            # 遍历所有可见的检测框
                            for det_idx in range(clip_data['visible_mask'][t].shape[0]):
                                if clip_data['visible_mask'][t][det_idx]:
                                    # 获取检测框坐标
                                    bbox = clip_data['bbox_ltwh'][t][det_idx]
                                    
                                    # 计算底边中点坐标
                                    l, top, w, h = bbox
                                    bottom_middle = (l + w/2, top + h)
                                    
                                    try:
                                        # 反投影到pitch坐标系
                                        world_point = backproject_using_ray_method(K, R_aug, T_aug, bottom_middle)
                                        
                                        # 将计算得到的世界坐标赋值给coords
                                        x, y = world_point[0].item(), world_point[1].item()
                                        
                                        # 对坐标进行clip，确保在合理范围内
                                        x = np.clip(x, self.coord_x_min, self.coord_x_max)
                                        y = np.clip(y, self.coord_y_min, self.coord_y_max)
                                        
                                        # 直接赋值，不进行normalization
                                        clip_data['coords'][t, det_idx, 0] = x
                                        clip_data['coords'][t, det_idx, 1] = y
                                    except Exception as e:
                                        # 处理投影失败的情况，保持原坐标不变
                                        print(f"投影失败: {e}")
                
                else:  # 'individual' 或 'frame' 模式 - 直接在坐标上添加噪声
                    # 提取原始坐标 - 已经是未标准化的
                    coords = clip_data['coords']  # [T, N, 2]
                    visible_mask = clip_data['visible_mask']  # [T, N]
                    
                    # 生成高斯噪声
                    if self.augment_coords_mode == 'frame':
                        # 每帧生成一个噪声，应用于所有检测框
                        T, N, _ = coords.shape
                        frame_noise = torch.randn(T, 1, 2) * self.augment_coords_sigma
                        # 对所有可见检测框应用相同的噪声
                        frame_noise = frame_noise.expand(T, N, 2)
                        # 只对可见的检测框应用噪声
                        noise_mask = visible_mask.unsqueeze(-1).expand(T, N, 2)
                        noise = torch.zeros_like(coords)
                        noise[noise_mask] = frame_noise[noise_mask]
                    else:  # 'individual'
                        # 为每个检测框独立生成噪声
                        noise = torch.randn_like(coords) * self.augment_coords_sigma
                        # 只对可见的检测框应用噪声
                        noise_mask = visible_mask.unsqueeze(-1).expand_as(noise)
                        noise = noise * noise_mask
                    
                    # 应用噪声
                    augmented_coords = coords + noise
                    
                    # 将坐标限制在合理范围内
                    augmented_coords[:, :, 0] = torch.clamp(augmented_coords[:, :, 0], self.coord_x_min, self.coord_x_max)
                    augmented_coords[:, :, 1] = torch.clamp(augmented_coords[:, :, 1], self.coord_y_min, self.coord_y_max)
                    
                    # 更新clip数据中的坐标
                    clip_data['coords'] = augmented_coords

            # Track ID augmentation
            if self.random_augment_track_ids and 'track_ids' in clip_data:
                track_ids = clip_data['track_ids']  # [T, N]
                visible_mask = clip_data['visible_mask']  # [T, N]
                
                # 找出clip中所有不同的track_id（非零值）
                unique_track_ids = torch.unique(track_ids[visible_mask])
                unique_track_ids = unique_track_ids[unique_track_ids > 0]
                
                if len(unique_track_ids) > 0:
                    # 为每个轨迹决定是否替换（根据random_track_id_change_prob）
                    tracks_to_change = []
                    for track_id in unique_track_ids:
                        if torch.rand(1).item() < self.random_track_id_change_prob:
                            tracks_to_change.append(track_id.item())
                    
                    if tracks_to_change:
                        # 找出所有不在当前视频中使用的track_id
                        all_possible_ids = set(range(1, 150))
                        existing_ids = set([tid.item() for tid in unique_track_ids])
                        available_ids = list(all_possible_ids - existing_ids)
                        
                        # 为每个要替换的track_id随机选择一个新ID
                        replacement_map = {}
                        for track_id in tracks_to_change:
                            if available_ids:
                                new_id = available_ids.pop(0)
                            else:
                                # 如果没有可用ID，就随机选择一个1-149的ID
                                new_id = torch.randint(1, 150, (1,)).item()
                            replacement_map[track_id] = new_id
                        
                        # 应用替换
                        for track_id, new_id in replacement_map.items():
                            # 创建掩码，标识所有匹配当前track_id且可见的位置
                            id_mask = (track_ids == track_id) & visible_mask
                            # 批量替换这些位置的ID
                            track_ids[id_mask] = new_id
                
                clip_data['track_ids'] = track_ids
                
            # Role augmentation - 全新的角色增强功能
            if self.random_augment_roles and 'track_ids' in clip_data and 'roles' in clip_data:
                track_ids = clip_data['track_ids']  # [T, N]
                visible_mask = clip_data['visible_mask']  # [T, N]
                roles = clip_data['roles']  # [T, N, num_roles]
                teams = clip_data['teams']  # [T, N, num_teams]
                jersey_numbers = clip_data['JNs']  # [T, N]
                
                # 找出clip中所有不同的track_id（非零值）
                unique_track_ids = torch.unique(track_ids[visible_mask])
                unique_track_ids = unique_track_ids[unique_track_ids > 0]
                
                if len(unique_track_ids) > 0:
                    # 为每个轨迹决定是否改变角色
                    for track_id in unique_track_ids:
                        if torch.rand(1).item() < self.random_role_change_prob:
                            # 创建该track_id的掩码
                            id_mask = (track_ids == track_id) & visible_mask
                            
                            # 获取第一个可见的实例，确定当前角色
                            t_idx, n_idx = torch.where(id_mask)
                            if len(t_idx) == 0:
                                continue  # 如果没有可见实例，则跳过
                                
                            t, n = t_idx[0].item(), n_idx[0].item()
                            curr_role_idx = roles[t, n].argmax().item()
                            if curr_role_idx == 3: # 暂时先不考虑对unknown进行增强
                                continue
                            
                            # 从角色列表中排除当前角色，选择新角色
                            avail_roles = [0, 1, 2]  # player, goalkeeper, referee
                            avail_roles.remove(curr_role_idx)
                            new_role_idx = avail_roles[torch.randint(0, len(avail_roles), (1,)).item()]
                            
                            # 为每个位置创建新的one-hot编码角色向量
                            new_roles = torch.zeros_like(roles[0, 0])
                            new_roles[new_role_idx] = 1.0
                            
                            # 根据新角色，确定是否需要更新team和jersey_number
                            # 当前团队属性
                            curr_team_idx = teams[t, n].argmax().item()
                            curr_jn = jersey_numbers[t, n].item()
                            
                            # 为referee准备团队属性 (2 = nan)
                            ref_team = torch.zeros_like(teams[0, 0])
                            ref_team[2] = 1.0  # nan team
                            
                            # 对于player和goalkeeper的team属性
                            # 如果新角色是player或goalkeeper，需要一个实际的team
                            if new_role_idx in [0, 1]:  # player or goalkeeper
                                if curr_role_idx == 2:  # 如果之前是referee
                                    # 随机分配一个team (0 = left, 1 = right)
                                    rand_team_idx = torch.randint(0, 2, (1,)).item()
                                    new_team = torch.zeros_like(teams[0, 0])
                                    new_team[rand_team_idx] = 1.0
                                else:
                                    # 保持原来的team
                                    new_team = teams[t, n].clone()
                            else:  # referee
                                new_team = ref_team
                            
                            # 对于jersey number:
                            # player: 保持原来的，或者分配新的（如果是从referee转变）
                            # goalkeeper, referee: 设为0（约定）
                            if new_role_idx == 0:  # player
                                new_jn = torch.randint(2, 100, (1,)).item()
                            else:  # referee or goalkeeper
                                new_jn = 0
                            
                            # 应用新的角色、团队和球衣号码
                            roles[id_mask] = new_roles
                            teams[id_mask] = new_team
                            jersey_numbers[id_mask] = new_jn
                    
                    # 更新数据
                    clip_data['roles'] = roles
                    clip_data['teams'] = teams
                    clip_data['JNs'] = jersey_numbers 
        
        # 应用帧擦除增强 - 这是所有增强之后的最后一步
        if self.split != 'test' and self.wipe_mode != 'none':
            # 1. 随机帧擦除
            if 'random' in self.wipe_mode:
                # 为每一帧生成随机概率
                random_probs = torch.rand(actual_clip_frames)
                # 确定哪些帧需要擦除
                frames_to_wipe = random_probs < self.wipe_random_prob
                
                # 应用擦除到可见性掩码
                for t in range(actual_clip_frames):
                    if frames_to_wipe[t]:
                        # 输入数据擦除
                        clip_data['visible_mask'][t, :] = False
                        # GT数据擦除
                        if 'gt_visible_mask' in clip_data:
                            clip_data['gt_visible_mask'][t, :] = False
            
            # 2. 连续帧擦除
            if 'continuous' in self.wipe_mode:
                # 决定是否执行连续擦除
                if torch.rand(1).item() < self.wipe_continuous_prob:
                    # 随机选择连续擦除的起始帧
                    continuous_start = torch.randint(0, actual_clip_frames, (1,)).item()
                    # 随机决定连续擦除的长度（3-50帧）
                    low = 3
                    high = min(51, actual_clip_frames - continuous_start + 1)
                    if low < high:
                        continuous_length = torch.randint(low, high, (1,)).item()
                        # 计算结束帧
                        continuous_end = min(continuous_start + continuous_length, actual_clip_frames)
                        
                        # 应用连续擦除
                        for t in range(continuous_start, continuous_end):
                            # 输入数据擦除
                            clip_data['visible_mask'][t, :] = False
                            # GT数据擦除
                            clip_data['gt_visible_mask'][t, :] = False
        
        # 添加时间掩码 - 只考虑实际有效的帧
        clip_data['time_mask'] = torch.zeros(self.max_clip_frames, dtype=torch.bool)
        clip_data['time_mask'][:actual_clip_frames] = True
        
        # 如果clip长度小于max_clip_frames，需要填充
        if actual_clip_frames < self.max_clip_frames:
            for key in clip_data:
                if key != 'time_mask':
                    # 创建填充张量
                    pad_shape = list(clip_data[key].shape)
                    pad_shape[0] = self.max_clip_frames - actual_clip_frames
                    padding = torch.zeros(pad_shape, dtype=clip_data[key].dtype)
                    # 连接原始数据和填充
                    clip_data[key] = torch.cat([clip_data[key], padding], dim=0)
        
        # 在返回之前应用normalization
        if self.normalize_bbox and self.bbox_mean is not None and self.bbox_std is not None:
            clip_data['bbox_ltwh'] = (clip_data['bbox_ltwh'] - torch.from_numpy(self.bbox_mean).float()) / torch.from_numpy(self.bbox_std).float()
            
        if self.normalize_coords:
            if self.normalization_method == "z_score" and self.coord_mean is not None and self.coord_std is not None:
                # Z-score normalization
                clip_data['coords'] = (clip_data['coords'] - torch.from_numpy(self.coord_mean).float()) / torch.from_numpy(self.coord_std).float()
            elif self.normalization_method == "minmax":
                # Min-max normalization to [-1, 1]
                x_coords = clip_data['coords'][:, :, 0]
                y_coords = clip_data['coords'][:, :, 1]
                
                # Scale x coordinates to [-1, 1]
                x_norm = 2.0 * (x_coords - self.coord_x_min) / (self.coord_x_max - self.coord_x_min) - 1.0
                
                # Scale y coordinates to [-1, 1]
                y_norm = 2.0 * (y_coords - self.coord_y_min) / (self.coord_y_max - self.coord_y_min) - 1.0
                
                clip_data['coords'] = torch.stack([x_norm, y_norm], dim=-1)
        
        # 预计算轨迹关联掩码 - 只有在track_enabled为True且track_ids存在时才计算
        if self.track_enabled and self.precompute_track_masks and 'gt_track_ids' in clip_data and 'gt_visible_mask' in clip_data:
                
            # 获取轨迹ID和可见性掩码
            track_ids = clip_data['gt_track_ids']  # [T, N]
            visible_mask = clip_data['gt_visible_mask']  # [T, N]
            T, N = track_ids.shape
            
            # 预先计算所有有效的帧对
            valid_pairs = []
            for t1 in range(T-1):
                max_t2 = min(t1 + self.max_affinity_frame_distance + 1, T)
                for t2 in range(t1+1, max_t2):
                    valid_pairs.append((t1, t2))
            
            # 如果有有效帧对，创建存储预计算的mask
            if valid_pairs:
                track_valid_pairs = {}
                track_pos_masks = {}
                track_neg_masks = {}
                
                # 为每个帧对计算mask
                for t1, t2 in valid_pairs:
                    # 扩展轨迹ID和可见性掩码以进行比较
                    track_t1 = track_ids[t1].unsqueeze(1)  # [N, 1]
                    track_t2 = track_ids[t2].unsqueeze(0)  # [1, N]
                    
                    mask_t1 = visible_mask[t1].unsqueeze(1)  # [N, 1]
                    mask_t2 = visible_mask[t2].unsqueeze(0)  # [1, N]
                    
                    # 计算有效的检测对
                    valid_pairs_mask = mask_t1 & mask_t2  # [N, N]
                    
                    # 计算正样本对：相同track_id的检测
                    pos_pairs = (track_t1 == track_t2) & valid_pairs_mask  # [N, N]
                    
                    # 计算负样本对：不同track_id的检测
                    neg_pairs = (track_t1 != track_t2) & valid_pairs_mask  # [N, N]
                    
                    # 转换为浮点掩码
                    pos_mask = pos_pairs.float()  # [N, N]
                    neg_mask = neg_pairs.float()  # [N, N]
                    
                    # 存储计算结果
                    track_valid_pairs[(t1, t2)] = valid_pairs_mask
                    track_pos_masks[(t1, t2)] = pos_mask
                    track_neg_masks[(t1, t2)] = neg_mask
                
                # 将预计算的mask添加到clip_data
                clip_data['track_valid_pairs'] = track_valid_pairs
                clip_data['track_pos_masks'] = track_pos_masks
                clip_data['track_neg_masks'] = track_neg_masks
                clip_data['track_frame_pairs'] = valid_pairs
        
        # 检查gt_visible_mask是否全为False
        if torch.any(clip_data['gt_visible_mask']):
            return clip_data
        else:
            warnings.warn(f"Warning: Video {vid} clip {start_frame}-{end_frame} has all gt_visible_mask=False. Retrying...")
            # 对于训练集，随机选择另一个索引
            idx = np.random.randint(0, len(self.clip_indices))
            return self.__getitem__(idx)