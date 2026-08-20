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

def is_convex_polygon(points):
    """
    Check if a polygon formed by points is convex.
    
    Args:
        points: numpy array of shape (4,2) containing x,y coordinates of 4 points
        
    Returns:
        bool: True if polygon is convex, False otherwise
    """
    if points is None:
        return False
        
    # Need at least 3 points to form a polygon
    if len(points) < 3:
        return False
        
    # Get vectors between consecutive points
    vectors = []
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        vectors.append([p2[0] - p1[0], p2[1] - p1[1]])
        
    # Check cross products have same sign
    cross_products = []
    for i in range(n):
        v1 = vectors[i]
        v2 = vectors[(i + 1) % n]
        cross_products.append(v1[0] * v2[1] - v1[1] * v2[0])
        
    # All cross products should have same sign for convex polygon
    all_positive = all(x >= 0 for x in cross_products)
    all_negative = all(x <= 0 for x in cross_products)
    
    return all_positive or all_negative


def is_detection_b_belongs_to_detection_a(detections_a, detections_b, image_height, image_width, margin=30, max_frames=50, max_dist=5):
    """
    Check if detection_b belongs to detection_a
    """
    if len(detections_a) == 0 or len(detections_b) == 0:
        return False
    
    # 首先判断b的第一个image_id是否比a的第一个image_id大，大说明b在a之后，否则不能合并
    image_id_a = detections_a.image_id.astype(int)
    image_id_b = detections_b.image_id.astype(int)
    # print(image_id_a, image_id_b)
    
    # assert int(detections_a.iloc[0].image_id) == min(image_id_a), f"{detections_a.iloc[0].image_id} != {min(image_id_a)}"
    # assert int(detections_b.iloc[0].image_id) == min(image_id_b), f"{detections_b.iloc[0].image_id} != {min(image_id_b)}"
    if min(image_id_a) > min(image_id_b):
        return False
    
    # 如果a和b中有重叠的image_id，则不能合并
    if len(set(image_id_a) & set(image_id_b)) > 0:
        return False
    
    # a中image_id小于b中image_id的最后一帧的bbox_ltwh
    concat_last_image_id_a = max([i for i in image_id_a if i < min(image_id_b)])
    concat_last_detection_a = detections_a[detections_a.image_id == str(concat_last_image_id_a)]
    concat_last_detection_a_bbox_ltrb = concat_last_detection_a.bbox.ltrb().values[0]
    # 如果a在图片边缘margin个像素内，则不能合并
    if concat_last_detection_a_bbox_ltrb[0] < margin or concat_last_detection_a_bbox_ltrb[1] < margin or concat_last_detection_a_bbox_ltrb[2] > image_width - margin or concat_last_detection_a_bbox_ltrb[3] > image_height - margin:
        return False
    
    # b中image_id最小的那一帧的bbox_ltwh
    fisrt_detection_b = detections_b[detections_b.image_id == str(min(image_id_b))]
    fisrt_detection_b_bbox_ltrb = fisrt_detection_b.bbox.ltrb().values[0]
    # 如果b在图片边缘margin个像素内，则不能合并
    if fisrt_detection_b_bbox_ltrb[0] < margin or fisrt_detection_b_bbox_ltrb[1] < margin or fisrt_detection_b_bbox_ltrb[2] > image_width - margin or fisrt_detection_b_bbox_ltrb[3] > image_height - margin:
        return False
    
    use_team = (len(image_id_b) > 3)
    if use_team:
        team_a = concat_last_detection_a.team.mode()[0]
        team_b = fisrt_detection_b.team.mode()[0]
        if team_a != team_b:
            return False

    # 超过max_frames帧则认为不能合并
    if concat_last_image_id_a - min(image_id_b) > max_frames:
        return False
    
    # TODO 需要check是否有pitch不存在的情况
    # bbox_pitch的欧几里得距离超过max_dist，则不能合并
    if pd.isna(fisrt_detection_b.bbox_pitch.values[0]) or fisrt_detection_b.bbox_pitch.values[0] is None or pd.isna(concat_last_detection_a.bbox_pitch.values[0]) or concat_last_detection_a.bbox_pitch.values[0] is None:
        return False
    fisrt_detection_b_bbox_pitch = np.array(fisrt_detection_b.bbox_pitch.values[0]['x_bottom_middle'], fisrt_detection_b.bbox_pitch.values[0]['y_bottom_middle'])
    concat_last_detection_a_bbox_pitch = np.array(concat_last_detection_a.bbox_pitch.values[0]['x_bottom_middle'], concat_last_detection_a.bbox_pitch.values[0]['y_bottom_middle'])
    if np.linalg.norm(fisrt_detection_b_bbox_pitch - concat_last_detection_a_bbox_pitch) > max_dist:
        return False
    
    return True

class ConcatTrackletsByMotion(VideoLevelModule):
    input_columns = {
        # "image": ["pitch_corners"],
        "image": [],
        # "detection": ["track_id", "jersey_number", "bbox_pitch", "bbox_ltwh", "team"] 
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
        
        # 用image bbox上来判断是否在边界也行，应该更准
        # "pitch_corners" is a numpy array of 4 points, left_top, right_top, right_bottom, left_bottom
        # valid_pitch_corners = [True] * len(metadatas)
        # pitch_corners_video = metadatas['pitch_corners']
        # # 如果pitch_corners围城的四边形不是凸四边形
        # for i, pitch_corners in enumerate(pitch_corners_video):
        #     if not is_convex_polygon(pitch_corners):
        #         valid_pitch_corners[i] = False
        #     # 如果左下和右下的点y值均小于左上和右上的点y值中小的那个
        #     if pitch_corners[2][1] < min(pitch_corners[0][1], pitch_corners[1][1]) and pitch_corners[3][1] < min(pitch_corners[0][1], pitch_corners[1][1]):
        #         valid_pitch_corners[i] = False
        
        # image边界各30个像素
        # 逻辑上是检测每个tracklet中，连续段落的最后一帧是否在边界内
        
        # 统计每个track_id对应的detection_ids
        track_to_detection_ids = {}
        tracklets = detections.track_id.unique()
        for track_id in tracklets:
            detection_ids = detections[detections.track_id == track_id].index
            track_to_detection_ids[track_id] = detection_ids

        # 两两判断是否可以合并
        # FIXME 添加距离最近优先
        concat_matrix = np.zeros((len(tracklets), len(tracklets)), dtype=bool)
        for i, track_id_i in enumerate(tracklets):
            for j, track_id_j in enumerate(tracklets):
                if i == j:
                    concat_matrix[i, j] = False
                else:
                    # detection_i = detections.iloc[track_to_detection_ids[track_id_i]]
                    # detection_j = detections.iloc[track_to_detection_ids[track_id_j]]
                    detections_i = detections[detections.track_id == track_id_i]
                    detections_j = detections[detections.track_id == track_id_j]
                    concat_matrix[i, j] = is_detection_b_belongs_to_detection_a(detections_i, detections_j, self.image_height, self.image_width, self.margin, self.max_frames, self.max_dist)
                        
                if concat_matrix[i, j]:
                    # 合并detection_i和detection_j，即track_id转变
                    detections.loc[track_to_detection_ids[track_id_j], 'track_id'] = track_id_i
                    track_to_detection_ids[track_id_i] = np.concatenate([track_to_detection_ids[track_id_i], track_to_detection_ids[track_id_j]])
                    track_to_detection_ids[track_id_j] = []
                    
                    # 其他的合并留给后面的tracklet_agg, team之类的模块
                    
        return detections
