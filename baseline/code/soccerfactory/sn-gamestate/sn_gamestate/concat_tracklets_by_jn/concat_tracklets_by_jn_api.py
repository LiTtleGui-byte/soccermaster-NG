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

class ConcatTrackletsByJN(VideoLevelModule):
    
    input_columns = ["track_id", "jersey_number"] 
    output_columns = ["track_id"]
    
    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return detections
            
        # Create mapping of jersey numbers to track IDs and their image_ids
        jn_to_tracks = {}
        track_to_images = {}
        
        # Get unique track IDs and their most common jersey number
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            jersey_numbers = tracklet.jersey_number.dropna()
            
            if len(jersey_numbers) > 0:
                # Get most common jersey number for this tracklet
                most_common_jn = jersey_numbers.mode()[0]
                
                # Store image_ids for this tracklet
                track_to_images[track_id] = set(tracklet.image_id.values)
                
                if most_common_jn not in jn_to_tracks:
                    jn_to_tracks[most_common_jn] = []
                jn_to_tracks[most_common_jn].append(track_id)
        
        # Create new track ID mapping
        track_id_map = {}
        next_track_id = 1
        
        # Check each jersey number group for frame overlaps
        for jersey_number, track_ids in jn_to_tracks.items():
            # Skip if only one tracklet has this jersey number
            if len(track_ids) <= 1:
                continue
                
            # Check for frame overlaps between tracklets
            has_overlap = False
            for i in range(len(track_ids)):
                for j in range(i+1, len(track_ids)):
                    if track_to_images[track_ids[i]] & track_to_images[track_ids[j]]:
                        has_overlap = True
                        break
                if has_overlap:
                    break
                    
            # Only merge tracklets if no overlaps found
            if not has_overlap:
                for old_track_id in track_ids:
                    track_id_map[old_track_id] = next_track_id
                next_track_id += 1
        
        # Map all remaining tracklets to new IDs
        for track_id in detections.track_id.unique():
            if track_id not in track_id_map:
                track_id_map[track_id] = next_track_id
                next_track_id += 1
                
        # Update track IDs
        detections["track_id"] = detections["track_id"].map(track_id_map)
        
        return detections
