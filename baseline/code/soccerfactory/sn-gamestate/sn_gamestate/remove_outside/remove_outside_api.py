import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

# Constants for pitch dimensions
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_X_MARGIN = 10.0
PITCH_Y_MARGIN = 5.0
COORD_X_MIN = -((PITCH_LENGTH / 2) + PITCH_X_MARGIN)  # -62.5
COORD_X_MAX = ((PITCH_LENGTH / 2) + PITCH_X_MARGIN)   # 62.5
COORD_Y_MIN = -((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)   # -39.0
COORD_Y_MAX = ((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)    # 39.0

log = logging.getLogger(__name__)

class RemoveOutside(VideoLevelModule):
    input_columns = ["track_id", "bbox_pitch"]
    output_columns = []
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # 用于存储要删除的track_id
        track_ids_to_remove = []
        
        for track_id, group in detections.groupby("track_id"):
            total_frames = len(group)
            valid_frames = group[group['bbox_pitch'].apply(lambda x: isinstance(x, dict))]
            if len(valid_frames) == 0:
                continue
            
            in_range_range_count = 0
            for _, frame in valid_frames.iterrows():
                bbox = frame['bbox_pitch']
                # Check if the frame is outside the allowed range
                if (COORD_X_MIN <= bbox['x_bottom_middle'] <= COORD_X_MAX) and (COORD_Y_MIN <= bbox['y_bottom_middle'] <= COORD_Y_MAX):
                    in_range_range_count += 1
                    
            # 如果有效帧中在场内的帧数小于一半，添加到要删除的列表中
            if in_range_range_count < len(valid_frames) / 2:
                track_ids_to_remove.append(track_id)
        
        # 从detections中删除这些track_id对应的所有行
        if track_ids_to_remove:
            detections = detections[~detections['track_id'].isin(track_ids_to_remove)]
                
        return detections
                

        
        
        
        