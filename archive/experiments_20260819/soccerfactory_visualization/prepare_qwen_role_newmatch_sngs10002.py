#!/usr/bin/env python3
"""Freeze label-blind single-view Qwen inputs for the independent SNGS-10002 match."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from prepare_qwen_role_swap_sngs10004 import PROMPT, atomic_json, select_representatives


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = Path(
    "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
    "step_3_sn500_1000/states/sn-gamestate.pklz"
)
VIDEO_ID = "10002"
IMAGE_DIR = Path(
    "/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/"
    "SoccerNetGS/sn500/SNGS-10002/img1"
)
OUTPUT_DIR = REPO / "reports/g10/20260819_qwen_role_newmatch_sngs10002"
MANIFEST = OUTPUT_DIR / "input_manifest.json"


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {MANIFEST}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    if len(detections) != 10985 or detections.track_id.nunique() != 33:
        raise AssertionError("Predetermined SNGS-10002 detection identity changed")
    if len(images) != 722 or int(images.frame.min()) != 0 or int(images.frame.max()) != 721:
        raise AssertionError("Predetermined SNGS-10002 image identity changed")
    image_by_id = {row.id: row for row in images.itertuples(index=False)}

    samples = []
    track_summary = []
    for track_id, group in detections.groupby("track_id", sort=True):
        numeric_id = int(float(track_id))
        selected = select_representatives(group)
        selected_ids = []
        for ordinal, row in enumerate(selected.itertuples(index=False), start=1):
            metadata = image_by_id[row.image_id]
            image_path = IMAGE_DIR / Path(str(metadata.file_path)).name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            sample_id = f"{VIDEO_ID}_track_{numeric_id:03d}_view_{ordinal}"
            selected_ids.append(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "video_id": VIDEO_ID,
                    "track_id": numeric_id,
                    "view_ordinal": ordinal,
                    "image_id": int(row.image_id),
                    "frame": int(metadata.frame),
                    "image_path_read_only": str(image_path),
                    "bbox_ltwh": [float(value) for value in row.bbox_ltwh],
                    "bbox_conf": float(row.bbox_conf),
                }
            )
        track_summary.append(
            {"track_id": numeric_id, "archive_rows": int(len(group)), "sample_ids": selected_ids}
        )

    if len(track_summary) != 33 or len(samples) != 89:
        raise AssertionError(f"Expected 33 tracks/89 views, found {len(track_summary)}/{len(samples)}")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise AssertionError("Duplicate sample IDs")
    value = {
        "status": "prepared",
        "schema_version": 1,
        "experiment": "independent-match single-view Qwen2.5-VL role validation on SNGS-10002",
        "video_id": VIDEO_ID,
        "selection_reason": "lowest numeric historical Step-3 match not used in SNGS-10001/SNGS-10004 role diagnostics",
        "source_archive_read_only": str(ARCHIVE),
        "image_root_read_only": str(IMAGE_DIR),
        "historical_role_fields_read_by_gpu": False,
        "manual_annotations_read": False,
        "gpu_used": False,
        "selection": "Three temporal bins per track; highest bbox_conf row in each bin.",
        "prompt": PROMPT,
        "candidate_roles": ["player", "referee", "goalkeeper", "other"],
        "tracks": len(track_summary),
        "samples": len(samples),
        "track_summary": track_summary,
        "sample_manifest": samples,
        "evaluation_boundary": "Freeze GPU predictions before reading or creating independent role labels.",
    }
    atomic_json(MANIFEST, value)
    print(json.dumps({"status": "prepared", "manifest": str(MANIFEST), "tracks": 33, "samples": 89}))


if __name__ == "__main__":
    main()
