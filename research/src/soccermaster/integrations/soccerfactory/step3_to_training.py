"""Convert SoccerFactory Step-3 tables to SoccerMaster's frame dictionary.

This module owns only the interface contract. Detection, tracking, role/team
inference, calibration, and Refiner remain SoccerFactory responsibilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _optional_scalar(value: Any) -> Any:
    return None if pd.isna(value) else value


def convert_step3_to_training_frames(
    detections: pd.DataFrame,
    images: pd.DataFrame,
    expected_video_id: str,
) -> dict[int, dict[str, Any]]:
    """Return the historical SoccerMaster per-frame training structure."""

    required_detection = {
        "image_id",
        "bbox_ltwh",
        "role",
        "legibility_score",
        "jersey_number",
    }
    required_image = {"id", "frame", "video_id", "parameters"}
    missing_detection = sorted(required_detection - set(detections.columns))
    missing_image = sorted(required_image - set(images.columns))
    if missing_detection or missing_image:
        raise ValueError(
            f"Missing adapter fields: detections={missing_detection}, images={missing_image}"
        )

    images = images.sort_values("frame", kind="stable")
    if images["frame"].astype(int).tolist() != list(range(len(images))):
        raise ValueError("Image frames must be zero-based and contiguous")
    if set(images["video_id"].astype(str)) != {str(expected_video_id)}:
        raise ValueError("Image video_id does not match expected_video_id")

    converted: dict[int, dict[str, Any]] = {}
    for image in images.itertuples(index=False):
        frame_id = int(image.frame) + 1
        image_detections = detections.loc[detections["image_id"] == image.id]
        people = []
        for index, detection in image_detections.iterrows():
            people.append(
                {
                    "id": int(index),
                    "bbox_ltwh": np.asarray(detection["bbox_ltwh"]).copy(),
                    "role": _optional_scalar(detection["role"]),
                    "legibility_score": float(detection["legibility_score"]),
                    "jersey_number": _optional_scalar(detection["jersey_number"]),
                }
            )

        parameters = image.parameters
        if parameters is None or (
            isinstance(parameters, float) and np.isnan(parameters)
        ):
            intrinsic = extrinsic = projection = None
            valid_camera = False
        else:
            principal = np.asarray(parameters["principal_point"], dtype=np.float64)
            intrinsic = np.array(
                [
                    [parameters["x_focal_length"], 0.0, principal[0]],
                    [0.0, parameters["y_focal_length"], principal[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            rotation = np.asarray(parameters["rotation_matrix"], dtype=np.float64)
            position = np.asarray(parameters["position_meters"], dtype=np.float64)
            translation = -rotation @ position
            extrinsic = np.concatenate((rotation, translation[:, None]), axis=1)
            projection = intrinsic @ extrinsic
            valid_camera = True

        converted[frame_id] = {
            "people": people,
            "K": intrinsic,
            "R": extrinsic,
            "P": projection,
            "valid_cam_params": valid_camera,
        }
    return converted
