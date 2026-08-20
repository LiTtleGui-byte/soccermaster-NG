#!/usr/bin/env python3
"""Render representative crops and distance diagnostics for Step-3 ReID merges."""

from __future__ import annotations

import json
import pickle
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path("/home/tianlin/SoccerMaster")
STATE = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
STEP3_RESULT = REPO / "reports/g10/20260819_step3_refiner_preserving_cpu/result.json"
REPORT = STEP3_RESULT.parent
GROUP_DIR = REPORT / "reid_merge_groups"
SUMMARY = REPORT / "merge_review_summary.json"
OVERVIEW = REPORT / "merge_groups_overview.png"
CELL = (180, 270)
HEADER = 48


def load_state() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(STATE) as archive:
        with archive.open("10004.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open("10004_image.pkl") as handle:
            images = pickle.load(handle)
    return detections, images


def crop_person(row: pd.Series, image_paths: dict[Any, str], size: tuple[int, int]) -> Image.Image:
    with Image.open(image_paths[row.image_id]) as source:
        image = source.convert("RGB")
    x, y, width, height = (float(value) for value in row.bbox_ltwh)
    margin_x, margin_y = width * 0.12, height * 0.08
    box = (
        max(0, int(x - margin_x)), max(0, int(y - margin_y)),
        min(image.width, int(x + width + margin_x)),
        min(image.height, int(y + height + margin_y)),
    )
    crop = image.crop(box)
    return ImageOps.pad(crop, size, method=Image.Resampling.LANCZOS, color="white")


def select_views(track: pd.DataFrame, count: int = 3) -> list[pd.Series]:
    ordered = track.sort_values("image_id")
    if len(ordered) <= count:
        return [row for _, row in ordered.iterrows()]
    bins = np.array_split(np.arange(len(ordered)), count)
    selected = []
    for positions in bins:
        subset = ordered.iloc[positions]
        score = subset.bbox_conf.astype(float) * subset.bbox_ltwh.apply(lambda box: float(box[3]))
        selected.append(subset.loc[score.idxmax()])
    return selected


def mode_text(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    return values.mode().iloc[0] if not values.empty else "?"


def track_embedding(track: pd.DataFrame) -> np.ndarray:
    return np.mean(np.vstack(track.embeddings.values), axis=0)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 - float(np.dot(left, right) / denominator)


def labeled_cell(crop: Image.Image, title: str, width: int = CELL[0]) -> Image.Image:
    canvas = Image.new("RGB", (width, crop.height + HEADER), "white")
    canvas.paste(crop, ((width - crop.width) // 2, HEADER))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((5, 5), title, fill="black", font=ImageFont.load_default(), spacing=2)
    return canvas


def paste_grid(cells: list[list[Image.Image]], path: Path, cell_width: int, cell_height: int) -> None:
    rows = len(cells)
    columns = max(len(row) for row in cells)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#dddddd")
    for row_index, row in enumerate(cells):
        for column_index, cell in enumerate(row):
            canvas.paste(cell, (column_index * cell_width, row_index * cell_height))
    canvas.save(path)


def main() -> int:
    if SUMMARY.exists() or OVERVIEW.exists():
        raise FileExistsError("Merge-review output already exists")
    GROUP_DIR.mkdir(exist_ok=True)
    detections, images = load_state()
    image_paths = images.set_index("id").file_path.to_dict()
    result = json.loads(STEP3_RESULT.read_text(encoding="utf-8"))
    groups = result["transitions"]["concat_by_reid"]["merged_groups_from_original_tracks"]
    if len(groups) != 9:
        raise AssertionError(f"Expected 9 ReID merge groups, found {len(groups)}")

    records: list[dict[str, Any]] = []
    overview_rows: list[list[Image.Image]] = []
    for group in groups:
        source_ids = group["source_track_ids"]
        track_frames = {track_id: detections[detections.track_id == track_id] for track_id in source_ids}
        embeddings = {track_id: track_embedding(frame) for track_id, frame in track_frames.items()}
        pairwise = [
            {
                "left": int(left), "right": int(right),
                "cosine_distance": cosine_distance(embeddings[left], embeddings[right]),
            }
            for left, right in combinations(source_ids, 2)
        ]
        group_rows = []
        overview_cells = []
        for track_id in source_ids:
            track = track_frames[track_id]
            views = select_views(track)
            cells = []
            for view in views:
                crop = crop_person(view, image_paths, CELL)
                title = f"old ID {track_id}\nframe {int(images.loc[view.image_id, 'frame'])}\nJN {mode_text(track.jersey_number)}"
                cells.append(labeled_cell(crop, title))
            group_rows.append(cells)
            best = max(views, key=lambda row: float(row.bbox_conf) * float(row.bbox_ltwh[3]))
            overview_crop = crop_person(best, image_paths, (150, 220))
            overview_title = f"ID {track_id}  f{int(images.loc[best.image_id, 'frame'])}\nJN {mode_text(track.jersey_number)}"
            overview_cells.append(labeled_cell(overview_crop, overview_title, width=160))
        group_path = GROUP_DIR / f"new_track_{int(group['new_track_id']):02d}.png"
        if not group_path.exists():
            paste_grid(group_rows, group_path, CELL[0], CELL[1] + HEADER)
        label = Image.new("RGB", (160, 268), "white")
        ImageDraw.Draw(label).multiline_text(
            (8, 8),
            f"new ID {group['new_track_id']}\n{len(source_ids)} old tracks",
            fill="black", font=ImageFont.load_default(), spacing=4,
        )
        overview_rows.append([label, *overview_cells])
        records.append({
            "new_track_id": int(group["new_track_id"]),
            "source_track_ids": source_ids,
            "source_frame_ranges": {
                str(track_id): [
                    int(images.loc[track.image_id, "frame"].min()),
                    int(images.loc[track.image_id, "frame"].max()),
                ]
                for track_id, track in track_frames.items()
            },
            "pairwise_distances": pairwise,
            "minimum_pairwise_distance": min(item["cosine_distance"] for item in pairwise),
            "maximum_pairwise_distance": max(item["cosine_distance"] for item in pairwise),
            "pairs_above_merge_threshold_0_1": sum(item["cosine_distance"] > 0.1 for item in pairwise),
            "contact_sheet": str(group_path),
        })

    paste_grid(overview_rows, OVERVIEW, 160, 268)
    summary = {
        "status": "passed",
        "merge_threshold": 0.1,
        "groups": records,
        "groups_with_nonlocal_pairs_above_threshold": sum(
            record["pairs_above_merge_threshold_0_1"] > 0 for record in records
        ),
        "interpretation": (
            "Pairs above 0.1 inside a final merge group indicate transitive chaining: "
            "the implementation only requires each accepted merge edge to pass the threshold."
        ),
        "gpu_used": False,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
