from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import bpbreid_pitch_deep_sort.deep_sort as deep_sort
import logging

from tracklab.pipeline import ImageLevelModule

log = logging.getLogger(__name__)


class BPBReIDPitchDeepSORT(ImageLevelModule):
    input_columns = [
        "bbox_pitch",
        "embeddings",
    ]
    output_columns = [
        "track_id",
        "track_position",
        "track_velocity",
        "hits",
        "age",
        "time_since_update",
    ]

    def __init__(self, cfg, device, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = deep_sort.DeepSORT(
            max_dist=self.cfg.max_dist,
            max_distance=self.cfg.max_distance,
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            nn_budget=self.cfg.nn_budget,
            min_confidence=self.cfg.min_bbox_confidence,
        )

    def prepare_next_frame(self, next_frame: np.ndarray):
        # Propagate the state distribution to the current time step using a Kalman filter prediction step.
        self.model.tracker.predict()

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        if len(detections) == 0:
            return {
                "id": [],
                "pitch_positions": [],
                "reid_features": [],
                "scores": [],
                "frame": [],
            }
        if hasattr(detections, "bbox_conf"):
            score = detections.bbox_conf
        else:
            score = detections.keypoints_conf
        try:
            pitch_positions = np.array([[pos['x_bottom_middle'], pos['y_bottom_middle']] for pos in detections.bbox_pitch])
        except: # TODO load_gt时可能出现未标记camera的帧，会导致他们没有bbox_pitch，要不要在load_gt时把他们过滤掉？
            return {
                "id": [],
                "pitch_positions": [],
                "reid_features": [],
                "scores": [],
                "frame": [],
            }
            
        input_tuple = {
            "id": detections.index.to_numpy(),
            "pitch_positions": pitch_positions,
            "reid_features": np.stack(detections.embeddings),
            "scores": np.stack(score),
            "frame": np.ones(len(detections.index)) * metadata.frame,
        }
        return input_tuple

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0 or len(batch["id"]) == 0:
            return []
        results = self.model.update(
            batch["id"][0],
            batch["pitch_positions"][0],
            batch["reid_features"][0],
            batch["scores"][0],
            batch["frame"][0],
        )
        assert set(results.index).issubset(
            detections.index
        ), "Mismatch of indexes during the tracking. The results should match the detections."
        return results
