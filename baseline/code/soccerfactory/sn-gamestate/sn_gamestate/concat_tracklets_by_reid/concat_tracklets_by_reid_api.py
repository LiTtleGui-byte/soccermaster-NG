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
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

class ConcatTrackletsByReid(VideoLevelModule):
    
    input_columns = ["track_id", "embeddings"] 
    output_columns = ["track_id"]
    
    def __init__(self, threshold=0.05, **kwargs):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return detections
            
        track_id_2_embeddings = {}
        track_id_2_image_ids = {}
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            embeddings = np.mean(tracklet.embeddings.values, axis=0).squeeze()
            track_id_2_embeddings[track_id] = embeddings
            track_id_2_image_ids[track_id] = set(tracklet.image_id.values)
        sorted_track_ids = sorted(track_id_2_embeddings.keys())
        sorted_track_id_2_embeddings = {track_id: track_id_2_embeddings[track_id] for track_id in sorted_track_ids}
        track_id_2_embeddings = sorted_track_id_2_embeddings
        
        reach_threshold = False
        while not reach_threshold:
            track_ids = list(track_id_2_embeddings.keys())
            n = len(track_ids)
            embeddings_matrix = np.vstack([track_id_2_embeddings[tid] for tid in track_ids])
            similarity_matrix = cosine_similarity(embeddings_matrix, embeddings_matrix)
            
            distances = []
            for i in range(n):
                for j in range(i+1, n):  # 只计算上三角部分以避免重复
                    track_id1 = track_ids[i]
                    track_id2 = track_ids[j]
                    similarity = similarity_matrix[i, j]
                    distance = 1.0 - similarity
                    distances.append((track_id1, track_id2, distance))
            
            sorted_distances = sorted(distances, key=lambda x: x[2])
            
            merged = False
            for track_id1, track_id2, distance in sorted_distances:
                if distance > self.threshold:
                    reach_threshold = True
                    break
                else:
                    if track_id_2_image_ids[track_id1].isdisjoint(track_id_2_image_ids[track_id2]):
                        # 按照各自占有的image数量进行加权平均
                        weight1 = len(track_id_2_image_ids[track_id1])
                        weight2 = len(track_id_2_image_ids[track_id2])
                        total_weight = weight1 + weight2
                        track_id_2_embeddings[track_id1] = (weight1 * track_id_2_embeddings[track_id1] + 
                                                           weight2 * track_id_2_embeddings[track_id2]) / total_weight
                        track_id_2_image_ids[track_id1] = track_id_2_image_ids[track_id1] | track_id_2_image_ids[track_id2]
                        track_id_2_embeddings.pop(track_id2)
                        track_id_2_image_ids.pop(track_id2)
                        
                        # 执行merge
                        detections.loc[detections.track_id == track_id2, 'track_id'] = track_id1
                        merged = True
                        break
                    else:
                        pass
            
            # If no tracklets were merged in this iteration, exit the loop
            if not merged:
                reach_threshold = True
        
        return detections
