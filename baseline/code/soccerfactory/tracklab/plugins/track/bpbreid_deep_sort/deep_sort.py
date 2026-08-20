import numpy as np
import pandas as pd

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ["DeepSORT"]


class DeepSORT(object):
    def __init__(
        self,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        min_bbox_confidence=0.2,
    ):
        self.max_dist = max_dist
        self.min_bbox_confidence = min_bbox_confidence
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
        )

    def update(
        self,
        ids,
        bbox_ltwh,
        reid_features,
        confidences,
        frame,
    ):
        # generate detections
        detections = [
            Detection(
                ids[i].cpu().detach().numpy(),
                np.asarray(bbox_ltwh[i].cpu().detach().numpy(), dtype=float),
                conf.cpu().detach().numpy(),
                0,
                reid_features[i][0].cpu().detach().numpy(),
            )
            for i, conf in enumerate(confidences)
        ]

        detections = self.filter_detections(detections)

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        ids = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            
            det = track.last_detection
            result_det = {
                "track_id": track.track_id,
                "track_bbox_kf_ltwh": track.to_tlwh(),
                "track_bbox_pred_kf_ltwh": track.last_kf_pred_tlwh,
                "hits": track.hits,
                "age": track.age,
                "time_since_update": track.time_since_update,
            }
            ids.append(det.id)
            outputs.append(result_det)

        outputs = pd.DataFrame(
            outputs,
            index=np.array(ids),
            columns=[
                "track_id",
                "track_bbox_kf_ltwh",
                "track_bbox_pred_kf_ltwh",
                "hits",
                "age",
                "time_since_update",
            ],
        )
        return outputs

    def filter_detections(self, detections):
        detections = [
            det for det in detections if det.confidence > self.min_bbox_confidence
        ]
        return detections
