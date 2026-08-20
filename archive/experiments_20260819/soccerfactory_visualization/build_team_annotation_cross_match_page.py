#!/usr/bin/env python3
"""Build the boxed track annotation page for predetermined SNGS-10001."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from build_team_annotation_page import build_html, make_strip, select_representatives


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = Path(
    "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
    "step_3_sn500_1000/states/sn-gamestate.pklz"
)
VIDEO_ID = "10001"
IMAGE_DIR = Path(
    "/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/"
    "SoccerNetGS/sn500/SNGS-10001/img1"
)
REPORT_DIR = REPO / "reports/g10/20260818_team_color_cross_match_sngs10001"
CROP_DIR = REPORT_DIR / "annotation_crops"
PAGE = REPORT_DIR / "annotate_tracks.html"


def main() -> None:
    if PAGE.exists() or CROP_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite annotation outputs: {PAGE}, {CROP_DIR}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    if len(detections) != 11494 or detections.track_id.nunique() != 40 or len(images) != 750:
        raise AssertionError("Predetermined SNGS-10001 archive identity changed")
    path_by_id = {row.id: str(IMAGE_DIR / Path(str(row.file_path)).name) for row in images.itertuples(index=False)}
    if any(not Path(path).is_file() for path in path_by_id.values()):
        raise FileNotFoundError("One or more source JPEGs are missing")
    CROP_DIR.mkdir(parents=True)
    tracks = []
    for track_id, group in detections.groupby("track_id"):
        numeric_id = int(float(track_id))
        selected = select_representatives(group)
        relative = f"annotation_crops/track_{numeric_id:03d}.jpg"
        make_strip(selected, path_by_id).save(REPORT_DIR / relative, quality=92)
        tracks.append(
            {
                "track_id": numeric_id,
                "rows": int(len(group)),
                "eligible": True,
                "image": relative,
            }
        )
    tracks.sort(key=lambda item: item["track_id"])
    if len(tracks) != 40:
        raise AssertionError("Expected 40 tracks")
    PAGE.write_text(build_html(tracks, video_id=VIDEO_ID, priority_count=40), encoding="utf-8")
    print(json.dumps({"status": "passed", "page": str(PAGE), "tracks": 40, "boxed_targets": True}))


if __name__ == "__main__":
    main()
