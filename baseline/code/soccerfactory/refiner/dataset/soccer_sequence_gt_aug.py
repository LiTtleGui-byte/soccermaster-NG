import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from tqdm import tqdm
import warnings
import pickle

from .dataset_utils import (
    PITCH_LENGTH, PITCH_WIDTH, PITCH_X_MARGIN, PITCH_Y_MARGIN,
    COORD_X_MIN, COORD_X_MAX, COORD_Y_MIN, COORD_Y_MAX,
    ROLE_MAP, TEAM_MAP,
    load_gt_data, process_gt_video,
    augment_camera_params, backproject_using_ray_method,
    normalize_coordinates, denormalize_coordinates,
    flip_x_coordinates, flip_y_coordinates, flip_teams
)

class SoccerSequenceGTAugDataset(Dataset):
    def __init__(self,
                 pipeline_outputs_root: str,  # 保持接口兼容性，但不会使用
                 pipeline_exp_name: str,      # 保持接口兼容性，但不会使用
                 pipeline_exp_name_test: str, # 保持接口兼容性，但不会使用
                 pipeline_exp_name_valid: str, # 保持接口兼容性，但不会使用
                 gt_root: str,
                 metadata_path: str, 
                 split: str,
                 max_frames: int = 750,
                 max_clip_frames: int = 50,
                 max_detections_per_frame: int = 30,
                 simulate_missing: bool = False,
                 normalize_bbox: bool = True,
                 normalize_coords: bool = True,
                 normalization_method: str = "z_score",
                 bbox_mean: list = None,
                 bbox_std: list = None,
                 coord_mean: list = None,
                 coord_std: list = None,
                 use_all_frames_as_start: bool = False,
                 augment_coords: bool = True,  # 默认开启
                 augment_coords_mode: str = 'camera',  # 默认使用相机增强
                 augment_coords_sigma: float = 1.0,
                 rotation_std: float = 0.5,
                 translation_std: float = 0.2,
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
        assert split in ['train', 'valid', 'test', 'train+valid'], f"Unsupported split: {split}"
        self.split = split
        
        # Handle special case for train+valid split
        if split == 'train+valid':
            # Combine video IDs from both train and validation splits
            train_vid_ids = [vid["name"].split('-')[1] for vid in self.metadata['train']]
            valid_vid_ids = [vid["name"].split('-')[1] for vid in self.metadata['validation']]
            self.vid_id_list = train_vid_ids + valid_vid_ids
            print(f"Using combined train+valid split with {len(train_vid_ids)} train videos and {len(valid_vid_ids)} validation videos")
            
            # Load both train and validation embeddings
            train_embeddings_path = os.path.join(gt_root, '..', 'ReID_features', 'train_embeddings.pkl')
            valid_embeddings_path = os.path.join(gt_root, '..', 'ReID_features', 'valid_embeddings.pkl')
            
            with open(train_embeddings_path, 'rb') as f:
                train_embeddings = pickle.load(f)
            with open(valid_embeddings_path, 'rb') as f:
                valid_embeddings = pickle.load(f)
                
            # Merge embeddings
            self.reid_embeddings = {**train_embeddings, **valid_embeddings}
        else:
            self.vid_id_list = [vid["name"].split('-')[1] for vid in self.metadata[split]]
            # Load embeddings for the specific split
            split_embeddings_path = os.path.join(gt_root, '..', 'ReID_features', f'{split}_embeddings.pkl')
            with open(split_embeddings_path, 'rb') as f:
                self.reid_embeddings = pickle.load(f)
        
        # 初始化参数
        self.max_frames = max_frames
        self.max_clip_frames = max_clip_frames
        self.max_detections = max_detections_per_frame
        self.simulate_missing = simulate_missing
        self.use_all_frames_as_start = use_all_frames_as_start
        self.normalize_bbox = normalize_bbox
        self.normalize_coords = normalize_coords
        self.normalization_method = normalization_method
        assert normalization_method in ["z_score", "minmax"], f"Unknown normalization method: {normalization_method}"
        
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
        
        # Track ID augmentation parameters
        self.random_augment_track_ids = random_augment_track_ids
        self.random_track_id_change_prob = random_track_id_change_prob
        
        # Role augmentation parameters
        self.random_augment_roles = random_augment_roles
        self.random_role_change_prob = random_role_change_prob
        
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
        
        # 保存路径信息
        self.gt_root = gt_root
        
        # 添加相机参数根目录
        self.camera_params_root = os.path.join(gt_root, '..', 'camera_params')
        
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
        
        # 预处理所有数据
        print("Preprocessing all data...")
        self.all_gt_data = {}
        
        for vid_idx, vid in enumerate(tqdm(self.vid_id_list, desc="Processing videos")):
            # 加载GT数据
            gt_data = self._load_gt_data(vid)
            
            # 获取图像ID列表
            image_ids = sorted([img['image_id'] for img in gt_data['images']])
            assert len(image_ids) == self.max_frames, f"Video {vid} has {len(image_ids)} frames, but max_frames is {self.max_frames}"
            
            # 处理GT数据
            gt_batch = process_gt_video(
                gt_data, image_ids,
                self.max_frames, self.max_detections,
                self.role_map, self.team_map,
                self.coord_x_min, self.coord_x_max, self.coord_y_min, self.coord_y_max,
                load_embeddings=True, embeddings=self.reid_embeddings
            )
            
            # 加载相机参数
            camera_params = self._load_camera_params(vid, image_ids)
            
            # 添加相机参数到GT数据中
            gt_batch.update(camera_params)
            camera_valid_mask = gt_batch['camera_params_valid_mask']  # [max_frames]
            visible_mask = gt_batch['visible_mask']  # [max_frames, max_detections]
            for frame_idx in range(self.max_frames):
                if not camera_valid_mask[frame_idx]:
                    visible_mask[frame_idx, :] = False
            gt_batch['visible_mask'] = visible_mask
            
            # 存储处理后的数据
            self.all_gt_data[vid_idx] = gt_batch

    def _load_gt_data(self, vid: str) -> dict:
        """加载单个视频的GT数据"""
        if self.split == 'train+valid':
            # Determine which original split this video belongs to
            if vid in [v["name"].split('-')[1] for v in self.metadata['train']]:
                actual_split = 'train'
            else:
                actual_split = 'valid'
            return load_gt_data(self.gt_root, actual_split, vid)
        else:
            return load_gt_data(self.gt_root, self.split, vid)

    def _load_camera_params(self, vid: str, image_ids: list) -> dict:
        """加载单个视频的相机参数"""
        # 确定哪个实际split
        if self.split == 'train+valid':
            if vid in [v["name"].split('-')[1] for v in self.metadata['train']]:
                actual_split = 'train'
            else:
                actual_split = 'valid'
        else:
            actual_split = self.split
        
        # 构建相机参数文件路径
        camera_params_path = os.path.join(self.camera_params_root, actual_split, f"SNGS-{vid}.json")
        
        # 初始化相机参数张量
        K = torch.zeros((self.max_frames, 3, 3))  # 相机内参矩阵
        R = torch.zeros((self.max_frames, 3, 3))  # 旋转矩阵
        T = torch.zeros((self.max_frames, 3))     # 平移向量
        camera_params_valid_mask = torch.zeros(self.max_frames, dtype=torch.bool)  # 有效性掩码
        
        # 如果相机参数文件存在，则加载
        assert os.path.exists(camera_params_path), f"Camera params file not found: {camera_params_path}"
        with open(camera_params_path, 'r') as f:
            camera_params_dict = json.load(f)
        
        # 遍历所有图像ID
        for frame_idx, image_id in enumerate(image_ids):
            assert frame_idx <= self.max_frames
            
            # 如果当前图像ID存在相机参数
            if image_id in camera_params_dict:
                cam_params = camera_params_dict[image_id]
                
                if cam_params['ransac_params'] is None:
                    params = cam_params['all_points_params']
                else:
                    if cam_params['all_reprojection_error_by_ransac'] < cam_params['all_points_params']['reprojection_error']:
                        params = cam_params['ransac_params']
                    else:
                        params = cam_params['all_points_params']
                    
                if min(cam_params['all_reprojection_error_by_ransac'], cam_params['all_points_params']['reprojection_error']) > 20:
                    continue
                
                # 转换相机参数为张量
                x_focal_length = params['x_focal_length']
                y_focal_length = params['y_focal_length']
                principal_point = np.array(params['principal_point'])
                position_meters = np.array(params['position_meters'])
                rotation = np.array(params['rotation_matrix'])
                
                # 保存相机参数
                T[frame_idx] = torch.from_numpy(position_meters)
                R[frame_idx] = torch.from_numpy(rotation)
                K[frame_idx] = torch.tensor([[x_focal_length, 0, principal_point[0]],
                                            [0, y_focal_length, principal_point[1]],
                                            [0, 0, 1]], dtype=torch.float32)
                camera_params_valid_mask[frame_idx] = True
        
        return {
            'K': K, 
            'R': R, 
            'T': T, 
            'camera_params_valid_mask': camera_params_valid_mask
        }

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _generate_augmented_data(self, gt_data: dict, start_frame: int, end_frame: int) -> dict:
        """
        基于GT数据生成增强后的输入数据
        
        Args:
            gt_data (dict): 完整的GT数据
            start_frame (int): 起始帧索引
            end_frame (int): 结束帧索引
            
        Returns:
            dict: 包含增强后输入数据和GT数据的字典
        """
        # 提取需要处理的clip数据
        clip_data = {}
        clip_gt_data = {}
        
        # 获取原始帧范围
        actual_clip_frames = end_frame - start_frame
        
        # 提取输入数据
        for key in ['feats', 'bbox_ltwh', 'coords', 'roles', 'teams', 'JNs', 'track_ids', 
                    'visible_mask', 'camera_params_valid_mask', 'K', 'R', 'T']:
            if key in gt_data:
                clip_data[key] = gt_data[key][start_frame:end_frame].clone()
        
        # 提取GT数据
        for key in gt_data:
            clip_gt_data[f'gt_{key}'] = gt_data[key][start_frame:end_frame].clone()
        
        # 应用数据增强
        # 1. 坐标噪声增强 - 只对输入数据应用
        if self.augment_coords:
            if 'camera' in self.augment_coords_mode or 'drift' in self.augment_coords_mode:
                # 对相机参数进行增强
                for t in range(actual_clip_frames):
                    # 只对有效的相机参数进行增强
                    if clip_data['camera_params_valid_mask'][t]:
                        # 获取原始相机参数
                        K = clip_data['K'][t]
                        R_orig = clip_data['R'][t]
                        T_orig = clip_data['T'][t]
                        
                        if 'drift' in self.augment_coords_mode:
                            do_drift = torch.rand(1).item() < self.camera_drift_prob
                        else:
                            do_drift = False
                            
                        do_aug_camera = False
                            
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
                        
                            # 遍历所有可见的检测框，使用增强后的相机参数重新计算世界坐标
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
                                        
                                        # 更新坐标
                                        clip_data['coords'][t, det_idx, 0] = x
                                        clip_data['coords'][t, det_idx, 1] = y
                                    except Exception as e:
                                        # 处理投影失败的情况，保持原坐标不变
                                        pass
            else:  # 'individual' 或 'frame' 模式
                coords = clip_data['coords']
                visible_mask = clip_data['visible_mask']
                
                if self.augment_coords_mode == 'frame':
                    T, N, _ = coords.shape
                    frame_noise = torch.randn(T, 1, 2) * self.augment_coords_sigma
                    frame_noise = frame_noise.expand(T, N, 2)
                    noise_mask = visible_mask.unsqueeze(-1).expand(T, N, 2)
                    noise = torch.zeros_like(coords)
                    noise[noise_mask] = frame_noise[noise_mask]
                else:  # 'individual'
                    noise = torch.randn_like(coords) * self.augment_coords_sigma
                    noise_mask = visible_mask.unsqueeze(-1).expand_as(noise)
                    noise = noise * noise_mask
                
                clip_data['coords'] = coords + noise
                clip_data['coords'][:, :, 0] = torch.clamp(
                    clip_data['coords'][:, :, 0], self.coord_x_min, self.coord_x_max
                )
                clip_data['coords'][:, :, 1] = torch.clamp(
                    clip_data['coords'][:, :, 1], self.coord_y_min, self.coord_y_max
                )
        
        # 2. 决定是否执行x翻转
        do_flip_x = torch.rand(1).item() < self.flip_x_prob
        if do_flip_x:
            # 翻转输入数据
            clip_data['coords'] = flip_x_coordinates(clip_data['coords'])
            clip_data['teams'] = flip_teams(clip_data['teams'])
            # 同时翻转GT数据
            clip_gt_data['gt_coords'] = flip_x_coordinates(clip_gt_data['gt_coords'])
            clip_gt_data['gt_teams'] = flip_teams(clip_gt_data['gt_teams'])
        
        # 3. 决定是否执行y翻转
        do_flip_y = torch.rand(1).item() < self.flip_y_prob
        if do_flip_y:
            # 翻转输入数据
            clip_data['coords'] = flip_y_coordinates(clip_data['coords'])
            # 同时翻转GT数据
            clip_gt_data['gt_coords'] = flip_y_coordinates(clip_gt_data['gt_coords'])
        
        # 4. 应用帧擦除增强 - 这是所有增强之后的最后一步
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
                        clip_gt_data['gt_visible_mask'][t, :] = False
            
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
                            clip_gt_data['gt_visible_mask'][t, :] = False
        
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
        
        # 将增强后的输入数据和GT数据合并
        result = {}
        result.update(clip_data)
        result.update(clip_gt_data)
        
        return result

    def __getitem__(self, idx: int) -> dict:
        # 获取当前索引对应的视频ID和起始帧
        vid_idx, start_frame = self.clip_indices[idx]
        vid = self.vid_id_list[vid_idx]
        end_frame = min(start_frame + self.max_clip_frames, self.max_frames)
        actual_clip_frames = end_frame - start_frame
        
        # 从预处理的数据中获取对应的视频数据
        gt_data = self.all_gt_data[vid_idx]
        
        # 生成增强后的输入数据和GT数据
        clip_data = self._generate_augmented_data(gt_data, start_frame, end_frame)
        
        # 添加时间掩码
        clip_data['time_mask'] = torch.zeros(self.max_clip_frames, dtype=torch.bool)
        clip_data['time_mask'][:actual_clip_frames] = True
        
        # 如果clip长度小于max_clip_frames，需要填充
        if actual_clip_frames < self.max_clip_frames:
            for key in clip_data:
                if key != 'time_mask':
                    pad_shape = list(clip_data[key].shape)
                    pad_shape[0] = self.max_clip_frames - actual_clip_frames
                    padding = torch.zeros(pad_shape, dtype=clip_data[key].dtype)
                    clip_data[key] = torch.cat([clip_data[key], padding], dim=0)
        
        # 在返回之前应用normalization
        if self.normalize_bbox and self.bbox_mean is not None and self.bbox_std is not None:
            clip_data['bbox_ltwh'] = (clip_data['bbox_ltwh'] - torch.from_numpy(self.bbox_mean).float()) / torch.from_numpy(self.bbox_std).float()
            
        if self.normalize_coords:
            if self.normalization_method == "z_score" and self.coord_mean is not None and self.coord_std is not None:
                clip_data['coords'] = (clip_data['coords'] - torch.from_numpy(self.coord_mean).float()) / torch.from_numpy(self.coord_std).float()
            elif self.normalization_method == "minmax":
                x_coords = clip_data['coords'][:, :, 0]
                y_coords = clip_data['coords'][:, :, 1]
                
                x_norm = 2.0 * (x_coords - self.coord_x_min) / (self.coord_x_max - self.coord_x_min) - 1.0
                y_norm = 2.0 * (y_coords - self.coord_y_min) / (self.coord_y_max - self.coord_y_min) - 1.0
                
                clip_data['coords'] = torch.stack([x_norm, y_norm], dim=-1)
        
        # 检查gt_visible_mask是否全为False
        if torch.any(clip_data['gt_visible_mask']):
            return clip_data
        else:
            warnings.warn(f"Warning: Video {vid} clip {start_frame}-{end_frame} has all gt_visible_mask=False. Retrying...")
            idx = np.random.randint(0, len(self.clip_indices))
            return self.__getitem__(idx) 