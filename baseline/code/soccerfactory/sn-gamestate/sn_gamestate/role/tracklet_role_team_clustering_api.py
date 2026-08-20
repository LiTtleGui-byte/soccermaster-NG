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

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosinie_distance(a, b):
    return 1 - cosine_similarity(a, b)

class TrackletRoleTeamClustering(VideoLevelModule):
    input_columns = ["track_id", "embeddings", "bbox_pitch"]
    output_columns = ["team_cluster", "team", "role"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        track_id_list = detections.track_id.unique()
        # 清空
        detections['team_cluster'] = np.nan
        detections['role'] = 'other'
        detections['team'] = None

        center_filtered_detections = detections[detections['bbox_pitch'].apply(
            lambda bbox: 'x_bottom_middle' in bbox and -30 <= bbox['x_bottom_middle'] <= 30 if isinstance(bbox, dict) else False
        )]
        
        center_tracklet_list = []
        for track_id, group in center_filtered_detections.groupby("track_id"):
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            center_tracklet_list.append({'track_id': track_id, 'embeddings': embeddings})
        
        if len(center_tracklet_list) < 3:
            detections['team'] = np.nan
            detections['role'] = np.nan
            return detections
        
        center_tracklets = pd.DataFrame(center_tracklet_list)
        center_embeddings = np.vstack(center_tracklets.embeddings.values)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(center_embeddings)
        label_counts = np.bincount(kmeans.labels_)
        sorted_indices = np.argsort(-label_counts)
        label_mapping = {
            sorted_indices[0]: 0,  # Most frequent cluster -> 0
            sorted_indices[1]: 1,  # Second most frequent cluster -> 1
            sorted_indices[2]: 2,  # Least frequent cluster -> 2
        }
        center_tracklets['team_cluster'] = [label_mapping[label] for label in kmeans.labels_]
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = np.array([cluster_centers[label_mapping[i]] for i in range(3)])
        
        # 中间的匹配结果，propagate到所有detections上
        track_id_2_team_cluster = dict(zip(center_tracklets.track_id, center_tracklets.team_cluster))
        detections['team_cluster'] = detections.track_id.map(track_id_2_team_cluster)
        
        # 对于没有匹配上的detections，计算tracklet_embeddings，通过和cluster_centers的距离进行匹配
        unmatched_detections = detections[detections['team_cluster'].isna()]
        unmatched_tracklets = []
        for track_id, group in unmatched_detections.groupby("track_id"):
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            unmatched_tracklets.append({'track_id': track_id, 'embeddings': embeddings})
        unmatched_tracklets = pd.DataFrame(unmatched_tracklets)
        
        unmatched_tracklet_distances = []
        for idx, row in unmatched_tracklets.iterrows():
            track_id = row['track_id']
            embedding = row['embeddings']
            for cluster_id in range(len(cluster_centers)):
                center = cluster_centers[cluster_id]
                distance = cosinie_distance(embedding, center)
                unmatched_tracklet_distances.append({
                'track_id': track_id,
                'cluster': cluster_id,
                'cosine_distance': distance
            })
        unmatched_tracklet_distances_df = pd.DataFrame(unmatched_tracklet_distances)
        if len(unmatched_tracklet_distances_df) > 0:
            closest_clusters = unmatched_tracklet_distances_df.loc[
                unmatched_tracklet_distances_df.groupby('track_id')['cosine_distance'].idxmin()
            ]
            assigned_tracks = closest_clusters[closest_clusters['cosine_distance'] < 0.2]
            track_id_2_team_cluster.update(dict(zip(assigned_tracks['track_id'], assigned_tracks['cluster'])))
            detections['team_cluster'] = detections.track_id.map(track_id_2_team_cluster)
        
        # 再次计算没有match的detections
        unmatched_detections = detections[detections['team_cluster'].isna()]
        
        # 处理左边禁区中没有匹配上的detections，计算cluster_center
        left_zone_detections = unmatched_detections[detections['bbox_pitch'].apply(
            lambda bbox: 'x_bottom_middle' in bbox and -52.5 <= bbox['x_bottom_middle'] <= -36 and -20.16 <= bbox['y_bottom_middle'] <= 20.16 if isinstance(bbox, dict) else False
        )]
        if len(left_zone_detections) > 0:
            new_center = np.mean(np.vstack(left_zone_detections.embeddings.values), axis=0)
        else:
            new_center = np.zeros(cluster_centers.shape[1])
        cluster_centers = np.vstack([cluster_centers, new_center])
        
        # 处理右边禁区中没有匹配上的detections，计算cluster_center
        right_zone_detections = unmatched_detections[detections['bbox_pitch'].apply(
            lambda bbox: 'x_bottom_middle' in bbox and 52.5 >= bbox['x_bottom_middle'] >= 36 and -20.16 <= bbox['y_bottom_middle'] <= 20.16 if isinstance(bbox, dict) else False
        )]
        if len(right_zone_detections) > 0:
            new_center = np.mean(np.vstack(right_zone_detections.embeddings.values), axis=0)
        else:
            new_center = np.zeros(cluster_centers.shape[1])
        cluster_centers = np.vstack([cluster_centers, new_center])
        
        # 剩下的所有的detections，只要tracklet上超过一半的xy都在范围内，就选择最近的那个cluster
        unmatched_tracklets = []
        for track_id, group in unmatched_detections.groupby("track_id"):
            # Filter out frames where bbox_pitch is not a dict
            valid_frames = group[group['bbox_pitch'].apply(lambda x: isinstance(x, dict))]
            if len(valid_frames) == 0:
                continue
            
            outside_range_count = 0
            for _, frame in valid_frames.iterrows():
                bbox = frame['bbox_pitch']
                # Check if the frame is outside the allowed range
                if (not (COORD_X_MIN <= bbox['x_bottom_middle'] <= COORD_X_MAX) or 
                    not (COORD_Y_MIN <= bbox['y_bottom_middle'] <= COORD_Y_MAX)):
                    outside_range_count += 1
                    
            if outside_range_count > len(valid_frames) / 2:
                continue
            
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            unmatched_tracklets.append({'track_id': track_id, 'embeddings': embeddings})
        unmatched_tracklets = pd.DataFrame(unmatched_tracklets)
        
        unmatched_tracklet_distances = []
        for idx, row in unmatched_tracklets.iterrows():
            track_id = row['track_id']
            embedding = row['embeddings']
            for cluster_id in range(len(cluster_centers)):
                center = cluster_centers[cluster_id]
                distance = cosinie_distance(embedding, center)
                unmatched_tracklet_distances.append({
                'track_id': track_id,
                'cluster': cluster_id,
                'cosine_distance': distance
            })
        if len(unmatched_tracklet_distances) > 0:
            unmatched_tracklet_distances_df = pd.DataFrame(unmatched_tracklet_distances)
            closest_clusters = unmatched_tracklet_distances_df.loc[
                unmatched_tracklet_distances_df.groupby('track_id')['cosine_distance'].idxmin()
            ]
            track_id_2_team_cluster.update(dict(zip(closest_clusters['track_id'], closest_clusters['cluster'])))
            detections['team_cluster'] = detections.track_id.map(track_id_2_team_cluster)
        
        # 计算0和1，觉得谁left 谁right
        team_a = detections[detections.team_cluster == 0]
        team_b = detections[detections.team_cluster == 1]
        xa_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_a.bbox_pitch]  # (x, y) are the center of a bbox
        xb_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_b.bbox_pitch]  # (x, y) are the center of a bbox
        avg_a = np.nanmean(xa_coordinates)
        avg_b = np.nanmean(xb_coordinates)
        if avg_a > avg_b:
            temp_mapping = {0: 1, 1: 0, 2: 2, 3:3, 4:4}  # Keep cluster 2 (referee) unchanged
            detections['team_cluster'] = detections['team_cluster'].map(lambda x: temp_mapping.get(x, x))
            cluster_centers[0], cluster_centers[1] = cluster_centers[1], cluster_centers[0]
        
        # Assign team and role based on team_cluster values
        role_mapping = {
            0: 'player',
            1: 'player',
            2: 'referee',
            3: 'goalkeeper',
            4: 'goalkeeper'
        }
        
        team_mapping = {
            0: 'left',
            1: 'right',
            2: None,
            3: 'left',
            4: 'right'
        }
        
        # Apply mappings to detections dataframe
        detections['role'] = detections['team_cluster'].map(role_mapping)
        detections['team'] = detections['team_cluster'].map(team_mapping)
                
        return detections
                

        
        
        
        