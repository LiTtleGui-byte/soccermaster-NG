#!/usr/bin/env python3
"""Freeze the CPU-only input manifest for the SNGS-10004 Qwen role swap."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004"
MANIFEST = OUTPUT_DIR / "input_manifest.json"
VIDEO_ID = "10004"
PROMPT = (
    "There are an image and a crop from it.\n"
    "Analyze this image and determine the role of the person in this image.\n"
    "Respond ONLY with a single word in ['player', 'referee', 'goalkeeper', 'other'].\n"
    "If there is no person in the image, or the person is not an athlete on the "
    "pitch, respond 'other'."
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def select_representatives(group: pd.DataFrame) -> pd.DataFrame:
    """Use the same deterministic three-bin selection as the annotation page."""
    ordered = group.sort_values("image_id", kind="stable")
    if len(ordered) <= 3:
        return ordered
    selections = []
    for positions in np.array_split(np.arange(len(ordered)), 3):
        part = ordered.iloc[positions]
        selections.append(part.loc[part.bbox_conf.astype(float).idxmax()])
    return pd.DataFrame(selections).drop_duplicates(subset=["image_id"])


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {MANIFEST}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    image_by_id = {row.id: row for row in images.itertuples(index=False)}

    samples = []
    track_summary = []
    missing_images = []
    for track_id, group in detections.groupby("track_id", sort=True):
        numeric_id = int(float(track_id))
        selected = select_representatives(group)
        selected_ids = []
        for ordinal, row in enumerate(selected.itertuples(index=False), start=1):
            metadata = image_by_id[row.image_id]
            image_path = Path(str(metadata.file_path))
            if not image_path.is_file():
                missing_images.append(str(image_path))
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
                    "existing_prtreid_role_detection": str(row.role_detection),
                    "existing_prtreid_role_confidence": float(row.role_confidence),
                }
            )
        track_summary.append(
            {
                "track_id": numeric_id,
                "archive_rows": int(len(group)),
                "sample_ids": selected_ids,
            }
        )

    if len(track_summary) != 49:
        raise AssertionError(f"Expected 49 tracks, found {len(track_summary)}")
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} source images")
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise AssertionError("Duplicate sample IDs")

    value = {
        "status": "prepared",
        "schema_version": 1,
        "experiment": "same-match Qwen2.5-VL role-module swap on SNGS-10004",
        "video_id": VIDEO_ID,
        "source_archive_read_only": str(ARCHIVE),
        "manual_annotations_read": False,
        "gpu_used": False,
        "selection": (
            "Up to three temporal bins per track; select the highest bbox_conf row "
            "inside each bin, identical to the existing annotation-page selector."
        ),
        "prompt": PROMPT,
        "candidate_roles": ["player", "referee", "goalkeeper", "other"],
        "tracks": len(track_summary),
        "samples": len(samples),
        "track_summary": track_summary,
        "sample_manifest": samples,
        "evaluation_boundary": (
            "The GPU inference stage must write predictions before the separate CPU "
            "evaluation stage reads manual role labels."
        ),
    }
    atomic_json(MANIFEST, value)
    print(
        json.dumps(
            {"status": "prepared", "manifest": str(MANIFEST), "tracks": 49, "samples": len(samples)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
