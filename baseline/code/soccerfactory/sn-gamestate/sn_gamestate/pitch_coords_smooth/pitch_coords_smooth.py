from functools import partial
from pathlib import Path
from typing import Any
from PIL import Image

import os
import sys
import yaml
import copy
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from tracklab.pipeline.videolevel_module import VideoLevelModule

from scipy.spatial.distance import euclidean

class PitchCoordsSmooth(VideoLevelModule):
    input_columns = {
        "detection": ['track_bbox_pred_kf_ltwh', 'track_id', 'jersey_number_detection',
       'legibility_score', 'track_bbox_kf_ltwh', 'costs', 'age', 'bbox_conf',
       'video_id', 'state', 'embeddings', 'ignored', 'body_masks', 'hits',
       'jersey_number_confidence', 'matched_with', 'bbox_pitch',
       'visibility_scores', 'category_id', 'time_since_update',
       'jersey_number_full_detection', 'image_id', 'bbox_ltwh',
       'role_confidence', 'role_detection', 'jersey_number', 'role',
       'team_cluster', 'team'],
    }
    output_columns = {
        "detection": ['track_bbox_pred_kf_ltwh', 'track_id', 'jersey_number_detection',
       'legibility_score', 'track_bbox_kf_ltwh', 'costs', 'age', 'bbox_conf',
       'video_id', 'state', 'embeddings', 'ignored', 'body_masks', 'hits',
       'jersey_number_confidence', 'matched_with', 'bbox_pitch',
       'visibility_scores', 'category_id', 'time_since_update',
       'jersey_number_full_detection', 'image_id', 'bbox_ltwh',
       'role_confidence', 'role_detection', 'jersey_number', 'role',
       'team_cluster', 'team'],
    }

    def __init__(self, batch_size=None, **kwargs):
        super().__init__()
        
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        image_id_list = detections.image_id.unique()
        image_id_map = {image_id: i for i, image_id in enumerate(image_id_list)}
        detections['image_idx'] = detections.image_id.map(image_id_map)
        sorted_image_idx = sorted(detections.image_idx.unique())
        detections['x'] = detections['bbox_pitch'].apply(lambda d: d.get('x_bottom_middle', np.nan) if isinstance(d, dict) else np.nan)
        detections['y'] = detections['bbox_pitch'].apply(lambda d: d.get('y_bottom_middle', np.nan) if isinstance(d, dict) else np.nan)
        
        # detections_image_idx_groups = detections[(detections['role'].isin(['player', 'goalkeeper', 'referee']))].groupby('image_idx')
        
        # anomaly_frames = set()
        # for i in range(1, len(sorted_image_idx)):
        #     image_idx1 = sorted_image_idx[i-1]
        #     image_idx2 = sorted_image_idx[i]
        #     detections_frame1 = detections_image_idx_groups.get_group(image_idx1)
        #     detections_frame2 = detections_image_idx_groups.get_group(image_idx2)
        #     if detections_frame1.empty or detections_frame2.empty:
        #         continue
            
        #     # 求detections_frame1和detections_frame2中重合的track_id
        #     track_ids_frame1 = set(detections_frame1['track_id'].values)
        #     track_ids_frame2 = set(detections_frame2['track_id'].values)
        #     overlapping_track_ids = track_ids_frame1.intersection(track_ids_frame2)
        #     total_players = len(overlapping_track_ids)
        #     if total_players == 0:
        #         continue
            
        #     moved_players = 0
        #     for track_id in overlapping_track_ids:
        #         point1 = (detections_frame1[detections_frame1['track_id'] == track_id]['x'].values[0], detections_frame1[detections_frame1['track_id'] == track_id]['y'].values[0])
        #         point2 = (detections_frame2[detections_frame2['track_id'] == track_id]['x'].values[0], detections_frame2[detections_frame2['track_id'] == track_id]['y'].values[0])
        #         if euclidean(point1, point2) > 10:
        #             moved_players += 1
        #     if moved_players / total_players >= 0.15:
        #         anomaly_frames.add(image_idx1)
        #         anomaly_frames.add(image_idx2)
        
        # # 找出连续异常段（1-3帧）
        # anomaly_ranges = []
        # current_range = []
        # for image_idx in sorted_image_idx:
        #     if image_idx in anomaly_frames:
        #         current_range.append(image_idx)
        #     else:
        #         if 1 <= len(current_range) <= 3:
        #             anomaly_ranges.append(current_range)
        #             current_range = []
        # if 3 <= len(current_range):
        #     anomaly_ranges.append(current_range)
            
        # detections_image_idx_groups = detections[(detections['role'].isin(['player', 'goalkeeper', 'referee']))].groupby('image_idx')
        # for anomaly_range in anomaly_ranges:
        #     prev_frame = anomaly_range[0]
        #     next_frame = anomaly_range[-1]
        #     # 获取参考帧数据
        #     prev_players = detections_image_idx_groups.get_group(prev_frame) if prev_frame in detections_image_idx_groups.groups else pd.DataFrame()
        #     next_players = detections_image_idx_groups.get_group(next_frame) if next_frame in detections_image_idx_groups.groups else pd.DataFrame()
            
        #     for frame_idx in anomaly_range[1:-1]:
        #         frame_data = detections_image_idx_groups.get_group(frame_idx)
        #         for idx, row in frame_data.iterrows():
        #             track_id = row['track_id']
        #             prev_player = prev_players[prev_players['track_id'] == track_id]
        #             next_player = next_players[next_players['track_id'] == track_id]
        #             if prev_player.empty or next_player.empty or len(prev_player) != 1 or len(next_player) != 1:
        #                 continue
        #             prev_player_x = prev_player['x'].values[0]
        #             prev_player_y = prev_player['y'].values[0]
        #             next_player_x = next_player['x'].values[0]
        #             next_player_y = next_player['y'].values[0]
        #             ratio = (frame_idx - prev_frame) / (next_frame - prev_frame)
        #             new_x = prev_player_x + ratio * (next_player_x - prev_player_x)
        #             new_y = prev_player_y + ratio * (next_player_y - prev_player_y)
        #             # 使用loc正确修改DataFrame
        #             detections.loc[idx, 'x'] = new_x
        #             detections.loc[idx, 'y'] = new_y
                
        tracklets_groups = detections[(detections['role'].isin(['player', 'goalkeeper', 'referee']))].groupby('track_id')
        smoothed_indices = []
        x_smoothed_values = []
        y_smoothed_values = []
        
        for track_id, tracklet in tracklets_groups:
            tracklet = tracklet.sort_values('image_idx')
            image_idx_diff = np.diff(tracklet['image_idx'])
            segment_breaks = np.where(image_idx_diff > 5)[0] + 1
            segments = np.split(tracklet, segment_breaks) if len(segment_breaks) > 0 else [tracklet]
            
            for seg in segments:
                if len(seg) < 2:
                    continue
                
                seg = seg.sort_values('image_idx')
                x_vals = seg['x'].values
                y_vals = seg['y'].values
                indices = seg.index.values
                
                alpha = 0.2
                
                x_ewma_fwd = np.zeros_like(x_vals)
                y_ewma_fwd = np.zeros_like(y_vals)
                x_current = x_vals[0] if not np.isnan(x_vals[0]) else 0
                y_current = y_vals[0] if not np.isnan(y_vals[0]) else 0
                
                for i in range(len(x_vals)):
                    if not np.isnan(x_vals[i]) and not np.isnan(y_vals[i]):
                        x_current = alpha * x_vals[i] + (1 - alpha) * x_current
                        y_current = alpha * y_vals[i] + (1 - alpha) * y_current
                
                    x_ewma_fwd[i] = x_current
                    y_ewma_fwd[i] = y_current
                    
                x_ewma_bwd = np.zeros_like(x_vals)
                y_ewma_bwd = np.zeros_like(y_vals)
                x_current = x_vals[-1] if not np.isnan(x_vals[-1]) else 0
                y_current = y_vals[-1] if not np.isnan(y_vals[-1]) else 0
                
                for i in range(len(x_vals)-1, -1, -1):
                    if not np.isnan(x_vals[i]) and not np.isnan(y_vals[i]):
                        x_current = alpha * x_vals[i] + (1 - alpha) * x_current
                        y_current = alpha * y_vals[i] + (1 - alpha) * y_current
                
                    x_ewma_bwd[i] = x_current
                    y_ewma_bwd[i] = y_current
                    
                x_smoothed = (x_ewma_fwd + x_ewma_bwd) / 2
                y_smoothed = (y_ewma_fwd + y_ewma_bwd) / 2
                
                smoothed_indices.extend(indices)
                x_smoothed_values.extend(x_smoothed)
                y_smoothed_values.extend(y_smoothed)
                
        smoothed_x_map = pd.Series(x_smoothed_values, index=smoothed_indices)
        smoothed_y_map = pd.Series(y_smoothed_values, index=smoothed_indices)
        
        # 使用loc正确修改DataFrame
        for idx in smoothed_x_map.index:
            detections.loc[idx, 'x'] = smoothed_x_map[idx]
            detections.loc[idx, 'y'] = smoothed_y_map[idx]
        
        # 正确更新bbox_pitch字典
        for idx, row in detections.iterrows():
            if 'bbox_pitch' in row and isinstance(row['bbox_pitch'], dict):
                if 'x' in row and not pd.isna(row['x']):
                    detections.loc[idx, 'bbox_pitch']['x_bottom_middle'] = row['x']
                if 'y' in row and not pd.isna(row['y']):
                    detections.loc[idx, 'bbox_pitch']['y_bottom_middle'] = row['y']
        
        detections = detections.drop(columns=['x', 'y', 'image_idx'], errors='ignore')
        return detections