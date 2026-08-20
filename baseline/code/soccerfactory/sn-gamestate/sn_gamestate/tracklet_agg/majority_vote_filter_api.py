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

class MajorityVoteTrackletFilter(VideoLevelModule):
    
    input_columns = []
    output_columns = []
    
    def __init__(self, cfg, device, tracking_dataset=None):
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.input_columns.append(f"{attribute}_detection")
            self.input_columns.append(f"{attribute}_confidence")
            self.output_columns.append(attribute)
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections[self.output_columns] = np.nan
        
        if "track_id" not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            for attribute in self.attributes:
                attribute_detection = tracklet[f"{attribute}_detection"]
                attribute_confidence = tracklet[f"{attribute}_confidence"]
                
                if attribute == 'jersey_number':
                    # Convert to list for easier manipulation
                    detection_list = list(attribute_detection)
                    confidence_list = list(attribute_confidence)
                    
                    # First pass: filter out values without 3 consecutive matches
                    filtered_detection = detection_list.copy()
                    for i in range(len(detection_list)):
                        # Get window of 3 centered at current position
                        start = max(0, i-1)
                        end = min(len(detection_list), i+2)
                        window = detection_list[start:end]
                        
                        # Check if current value matches all values in window
                        current_val = detection_list[i]
                        if current_val is None or not all(v == current_val for v in window):
                            filtered_detection[i] = None
                            
                    # Create filtered lists removing None values
                    final_detection = []
                    final_confidence = []
                    for d, c in zip(filtered_detection, confidence_list):
                        if d is not None:
                            final_detection.append(d)
                            final_confidence.append(c)
                            
                    # Get majority vote from filtered values
                    if final_detection:
                        attribute_value = select_highest_voted_att(final_detection, final_confidence)
                    else:
                        attribute_value = None
                        
                    attribute_value = [attribute_value] * len(tracklet)
                else:
                    attribute_value = [select_highest_voted_att(attribute_detection, attribute_confidence)] * len(tracklet)
                            
                detections.loc[tracklet.index, attribute] = attribute_value
            
        return detections

class MajorityVoteTrackletFilter2(VideoLevelModule):
    
    input_columns = []
    output_columns = []
    
    def __init__(self, cfg, device, tracking_dataset=None):
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.input_columns.append(f"{attribute}_detection")
            self.input_columns.append(f"{attribute}_confidence")
            self.output_columns.append(attribute)
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections[self.output_columns] = np.nan
        
        if "track_id" not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            for attribute in self.attributes:
                attribute_detection = tracklet[f"{attribute}_detection"]
                attribute_confidence = tracklet[f"{attribute}_confidence"]
                
                if attribute == 'jersey_number':
                    # Convert to list for easier manipulation
                    detection_list = list(attribute_detection)
                    confidence_list = list(attribute_confidence)
                    
                    # First pass: filter out values without 3 consecutive matches
                    filtered_detection = detection_list.copy()
                    for i in range(len(detection_list)):
                        # Get window of 3 centered at current position
                        start = max(0, i-1)
                        end = min(len(detection_list), i+2)
                        window = detection_list[start:end]
                        
                        # Check if current value matches all values in window
                        current_val = detection_list[i]
                        if current_val is None or sum(v == current_val for v in window) < 2:
                            filtered_detection[i] = None
                            
                    # Create filtered lists removing None values
                    final_detection = []
                    final_confidence = []
                    for d, c in zip(filtered_detection, confidence_list):
                        if d is not None:
                            final_detection.append(d)
                            final_confidence.append(c)
                            
                    # Get majority vote from filtered values
                    if final_detection:
                        attribute_value = select_highest_voted_att(final_detection, final_confidence)
                    else:
                        attribute_value = None
                        
                    attribute_value = [attribute_value] * len(tracklet)
                else:
                    attribute_value = [select_highest_voted_att(attribute_detection, attribute_confidence)] * len(tracklet)
                            
                detections.loc[tracklet.index, attribute] = attribute_value
            
        return detections

class MajorityVoteTrackletFilter10(VideoLevelModule):
    
    input_columns = []
    output_columns = []
    
    def __init__(self, cfg, device, tracking_dataset=None):
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.input_columns.append(f"{attribute}_detection")
            self.input_columns.append(f"{attribute}_confidence")
            self.output_columns.append(attribute)
        
    def check_consecutive_ten(self, values):
        if not isinstance(values, (list, np.ndarray)):
            values = list(values)
        count = 0
        for v in values:
            if v == '10':
                count += 1
            else:
                count = 0
            if count >= 3:
                return True
        return False
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        
        detections[self.output_columns] = np.nan
        
        if "track_id" not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            for attribute in self.attributes:
                attribute_detection = tracklet[f"{attribute}_detection"]
                attribute_confidence = tracklet[f"{attribute}_confidence"]
                
                if attribute == 'jersey_number':
                    # Special handling for jersey number 10
                    attribute_value = select_highest_voted_att(attribute_detection, attribute_confidence)
                    if attribute_value == '10':
                        # Only keep '10' if it appears in at least 3 consecutive frames
                        if not self.check_consecutive_ten(attribute_detection):
                            # Find the next most voted value that's not '10'
                            filtered_detection = [v for v in attribute_detection if v != '10']
                            filtered_confidence = [c for v, c in zip(attribute_detection, attribute_confidence) if v != '10']
                            if filtered_detection:
                                attribute_value = select_highest_voted_att(filtered_detection, filtered_confidence)
                            else:
                                attribute_value = None
                    attribute_value = [attribute_value] * len(tracklet)
                else:
                    attribute_value = [select_highest_voted_att(attribute_detection, attribute_confidence)] * len(tracklet)
                            
                detections.loc[tracklet.index, attribute] = attribute_value
            
        return detections
