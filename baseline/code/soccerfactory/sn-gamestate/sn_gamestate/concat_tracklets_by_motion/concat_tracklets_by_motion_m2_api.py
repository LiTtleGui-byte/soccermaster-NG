from pathlib import Path

import cv2
import pandas as pd
import torch
import requests
import numpy as np
from tqdm import tqdm
from tracklab.utils.cv2 import cv2_load_image, crop_bbox_ltwh
from tracklab.utils.attribute_voting import select_highest_voted_att

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.openmmlab import get_checkpoint

from collections import Counter


import logging


log = logging.getLogger(__name__)
    
def try_merge_segment(detections, track_to_image_ids, tracklet_segments, segment, image_height, image_width, margin=30, max_frames=50, max_dist=5):
    # segment自己必须不在边缘
    detections_b = detections.iloc[segment.ids]
    fisrt_detection_b = detections_b[detections_b.image_id == str(segment.start_image_id)]
    fisrt_detection_b_bbox_ltrb = fisrt_detection_b.bbox.ltrb().values[0]
    if fisrt_detection_b_bbox_ltrb[0] < margin or fisrt_detection_b_bbox_ltrb[1] < margin or fisrt_detection_b_bbox_ltrb[2] > image_width - margin or fisrt_detection_b_bbox_ltrb[3] > image_height - margin:
        return False
    
    # Check if bbox_pitch is nan
    if pd.isna(fisrt_detection_b.bbox_pitch.values[0]) or fisrt_detection_b.bbox_pitch.values[0] is None:
        return False
    fisrt_detection_b_bbox_pitch = np.array(fisrt_detection_b.bbox_pitch.values[0]['x_bottom_middle'], fisrt_detection_b.bbox_pitch.values[0]['y_bottom_middle'])
    
    use_team = (len(segment.image_ids) > 3)
    if use_team:
        team_b = detections_b.team.mode()[0]
    
    judge_list = [True] * len(tracklet_segments)
    dist_list = [1000000] * len(tracklet_segments)
    for i, segment_i in enumerate(tracklet_segments):
        # 不考虑已经被合并的
        if segment_i.being_merged:
            judge_list[i] = False
            continue
        # 不考虑start_image_id更大的，即时间轴上更晚的
        if segment_i.start_image_id >= segment.start_image_id:
            judge_list[i] = False
            continue
        # 不考虑有重叠的
        if (segment_i.track_id != segment.track_id) and (len(set(track_to_image_ids[segment_i.track_id]) & set(segment.image_ids)) > 0):
            judge_list[i] = False
            continue
        
        concat_last_image_id_a = max([image_id for image_id in segment_i.image_ids if image_id < segment.start_image_id])
        # 超过max_frames帧则认为不能合并
        if concat_last_image_id_a - segment.start_image_id > max_frames:
            judge_list[i] = False
            continue
        detections_a = detections.iloc[segment_i.ids]
        if use_team:
            team_a = detections_a.team.mode()[0]
            if team_a != team_b:
                judge_list[i] = False
                continue
        
        concat_last_detection_a = detections_a[detections_a.image_id == str(concat_last_image_id_a)]
        concat_last_detection_a_bbox_ltrb = concat_last_detection_a.bbox.ltrb().values[0]
        # 如果a在图片边缘margin个像素内，则不能合并
        if concat_last_detection_a_bbox_ltrb[0] < margin or concat_last_detection_a_bbox_ltrb[1] < margin or concat_last_detection_a_bbox_ltrb[2] > image_width - margin or concat_last_detection_a_bbox_ltrb[3] > image_height - margin:
            judge_list[i] = False
            continue
        if pd.isna(concat_last_detection_a.bbox_pitch.values[0]) or concat_last_detection_a.bbox_pitch.values[0] is None:
            continue
        concat_last_detection_a_bbox_pitch = np.array(concat_last_detection_a.bbox_pitch.values[0]['x_bottom_middle'], concat_last_detection_a.bbox_pitch.values[0]['y_bottom_middle'])
        # 不考虑bbox_pitch欧几里得距离超过max_dist的
        dist = np.linalg.norm(concat_last_detection_a_bbox_pitch - fisrt_detection_b_bbox_pitch)
        if dist > max_dist:
            judge_list[i] = False
            continue
        dist_list[i] = dist
        
    if sum(judge_list) == 0:
        return False
        
    # 如果judge_list中多于1个True，则选择dist最小的
    true_indices = [i for i, x in enumerate(judge_list) if x]
    min_dist_index = true_indices[np.argmin([dist_list[i] for i in true_indices])]
    tracklet_segments[min_dist_index].merge(track_to_image_ids, segment)
    return True
        
class TrackletSegment:
    def __init__(self, track_id, ids, image_ids):
        self.track_id = track_id
        self.ids = ids
        self.image_ids = image_ids
        self.start_image_id = min(image_ids)
        self.end_image_id = max(image_ids)
        
        self.being_merged = False
        
    def merge(self, track_to_image_ids, segment):
        assert not self.being_merged
        segment.being_merged = True
        
        self.ids = np.concatenate([self.ids, segment.ids])
        self.image_ids = np.concatenate([self.image_ids, segment.image_ids])
        self.start_image_id = min(self.image_ids)
        self.end_image_id = max(self.image_ids)
        
        # track_id不同则需要对track_to_image_ids进行一定的变换
        if self.track_id != segment.track_id:
            track_to_image_ids[self.track_id] = np.concatenate([track_to_image_ids[self.track_id], segment.image_ids])
            track_to_image_ids[segment.track_id] = [image_id for image_id in track_to_image_ids[segment.track_id] if image_id not in segment.image_ids]
        
class ConcatTrackletsByMotion2(VideoLevelModule):
    input_columns = {
        "image": [],
        "detection": ["track_id", "bbox_pitch", "bbox_ltwh", "team"] 
        
    }
    output_columns = {
        "detection": ["track_id"]
    }
    
    def __init__(self, image_width=1920, image_height=1080, margin=30, max_frames=50, max_dist = 10, **kwargs):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.margin = margin
        self.max_frames = max_frames
        self.max_dist = max_dist
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return detections
        
        # 统计每个track_id对应的image_ids
        track_to_image_ids = {}
        tracklets = detections.track_id.unique()
        for track_id in tracklets:
            track_to_image_ids[track_id] = detections[detections.track_id == track_id].image_id.astype(int)
        
        # 先按照连续性分成segments，然后再进行匹配
        tracklets = detections.track_id.unique()
        tracklet_segments = []
        for track_id in tracklets:
            tracklet = detections[detections.track_id == track_id]
            # Convert image_id to int and sort
            sorted_frames = sorted(tracklet.image_id.astype(int))
            
            # Initialize first segment
            current_segment = [sorted_frames[0]]
            current_segment_ids = [tracklet[tracklet.image_id == str(sorted_frames[0])].index[0]]
            
            # Iterate through sorted frames to find breaks in continuity
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] == sorted_frames[i-1] + 1:
                    # Frame is continuous, add to current segment
                    current_segment.append(sorted_frames[i])
                    current_segment_ids.append(tracklet[tracklet.image_id == str(sorted_frames[i])].index[0])
                else:
                    # Break in continuity, create new segment
                    tracklet_segments.append(TrackletSegment(track_id, current_segment_ids, detections.iloc[current_segment_ids].image_id.astype(int)))
                    current_segment = [sorted_frames[i]]
                    current_segment_ids = [tracklet[tracklet.image_id == str(sorted_frames[i])].index[0]]
            
            # Add final segment
            tracklet_segments.append(TrackletSegment(track_id, current_segment_ids, detections.iloc[current_segment_ids].image_id.astype(int)))
            
        # 按照start_image_id排序
        tracklet_segments = sorted(tracklet_segments, key=lambda x: x.start_image_id)
            
        # 两两判断是否可以合并
        # FIXME 添加距离最近优先
        for i in range(len(tracklet_segments)):
            segment = tracklet_segments[i]
            assert not segment.being_merged
            try_merge_segment(detections, track_to_image_ids, tracklet_segments, segment, self.image_height, self.image_width, self.margin, self.max_frames, self.max_dist)
                    
        # 应用合并结果
        for i in range(len(tracklet_segments)):
            segment = tracklet_segments[i]
            if not segment.being_merged:
                detections.loc[segment.ids, 'track_id'] = segment.track_id
                    
        return detections