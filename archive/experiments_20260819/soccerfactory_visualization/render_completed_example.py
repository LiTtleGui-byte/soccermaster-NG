#!/usr/bin/env python3
"""Render one CPU-only example from the completed pre-Refiner pipeline.

This reads the immutable G10 enrichment run2 archive and one source JPEG.  It
does not import torch, run a model, alter the archive, or write outside the
workspace report directory.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260818_completed_pipeline_example"
VIDEO_ID = "10004"
IMAGE_ID = "610004000148"


def scalar_or_none(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def team_color(team: str) -> tuple[int, int, int]:
    return (255, 86, 86) if team == "left" else (65, 210, 255)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"{VIDEO_ID}.pkl") as handle:
            detections = pd.read_pickle(handle)
        with archive.open(f"{VIDEO_ID}_image.pkl") as handle:
            images = pd.read_pickle(handle)
    return detections, images


def draw_overlay(source: Image.Image, rows: pd.DataFrame) -> Image.Image:
    canvas = source.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    label_font = font(18)
    title_font = font(24)

    draw.rectangle((0, 0, canvas.width, 42), fill=(0, 0, 0))
    draw.text(
        (12, 8),
        "Pre-Refiner candidate labels | red=left, cyan=right | NOT ground truth",
        fill="white",
        font=title_font,
    )

    for row in rows.sort_values("bbox_conf", ascending=False).itertuples(index=False):
        x, y, width, height = (float(v) for v in row.bbox_ltwh)
        x2, y2 = x + width, y + height
        team = str(row.team)
        color = team_color(team)
        track_id = int(float(row.track_id))
        jersey = scalar_or_none(row.jersey_number)
        jersey_text = "?" if jersey is None else str(int(float(jersey)))
        confidence = float(row.bbox_conf)
        role = str(row.role)
        text = f"ID {track_id} | {team} | #{jersey_text} | {role} | det {confidence:.2f}"

        draw.rectangle((x, y, x2, y2), outline=color, width=4)
        foot_x, foot_y = x + width / 2, y2
        draw.ellipse((foot_x - 5, foot_y - 5, foot_x + 5, foot_y + 5), fill=(255, 230, 50))

        text_box = draw.textbbox((0, 0), text, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0, min(x, canvas.width - text_width - 8))
        label_y = y - text_height - 8 if y - text_height - 8 > 44 else y2 + 3
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 8, label_y + text_height + 6),
            fill=(0, 0, 0),
        )
        draw.text((label_x + 4, label_y + 2), text, fill=color, font=label_font)

    return canvas


def pitch_point(x: float, y: float, width: int, height: int, margin: int) -> tuple[float, float]:
    # Standard 105 x 68 m display. These are candidate projected coordinates.
    px = margin + (x + 52.5) / 105.0 * (width - 2 * margin)
    py = margin + (34.0 - y) / 68.0 * (height - 2 * margin)
    return px, py


def draw_pitch(rows: pd.DataFrame) -> Image.Image:
    width, height, margin = 1200, 820, 70
    canvas = Image.new("RGB", (width, height), (26, 116, 58))
    draw = ImageDraw.Draw(canvas)
    line = (240, 245, 240)
    title_font = font(25)
    label_font = font(18)

    draw.rectangle((margin, margin, width - margin, height - margin), outline=line, width=4)
    draw.line((width / 2, margin, width / 2, height - margin), fill=line, width=3)
    draw.ellipse((width / 2 - 90, height / 2 - 90, width / 2 + 90, height / 2 + 90), outline=line, width=3)
    draw.rectangle((margin, height / 2 - 180, margin + 170, height / 2 + 180), outline=line, width=3)
    draw.rectangle((width - margin - 170, height / 2 - 180, width - margin, height / 2 + 180), outline=line, width=3)
    draw.text((margin, 20), "Candidate pitch projection before Refiner (not validated)", fill="white", font=title_font)

    for row in rows.itertuples(index=False):
        pitch = row.bbox_pitch
        x = float(pitch["x_bottom_middle"])
        y = float(pitch["y_bottom_middle"])
        px, py = pitch_point(x, y, width, height, margin)
        team = str(row.team)
        color = team_color(team)
        track_id = int(float(row.track_id))
        draw.ellipse((px - 12, py - 12, px + 12, py + 12), fill=color, outline="black", width=2)
        draw.text((px + 15, py - 10), str(track_id), fill="white", font=label_font, stroke_width=2, stroke_fill="black")

    draw.rectangle((80, height - 52, 1050, height - 12), fill=(0, 0, 0))
    draw.text(
        (92, height - 45),
        "Position = bbox_pitch bottom-middle. A plausible plot does not prove calibration accuracy.",
        fill="white",
        font=label_font,
    )
    return canvas


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#151a21")
    x = (size[0] - result.width) // 2
    y = (size[1] - result.height) // 2
    panel.paste(result, (x, y))
    return panel


def make_overview(original: Image.Image, overlay: Image.Image, pitch: Image.Image) -> Image.Image:
    panel_size = (720, 430)
    title_height = 54
    canvas = Image.new("RGB", (panel_size[0] * 2, title_height + panel_size[1] * 2), "#0f141a")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "SoccerFactory completed stages: one visual example", fill="white", font=font(28))
    canvas.paste(fit_panel(original, panel_size), (0, title_height))
    canvas.paste(fit_panel(overlay, panel_size), (panel_size[0], title_height))
    canvas.paste(fit_panel(pitch, panel_size), (0, title_height + panel_size[1]))

    note_x = panel_size[0]
    note_y = title_height + panel_size[1]
    draw.rectangle((note_x, note_y, canvas.width, canvas.height), fill="#202934")
    lines = [
        "What has run:",
        "- person detection",
        "- ReID and tracking",
        "- camera / pitch projection",
        "- role, team and jersey enrichment",
        "",
        "What has NOT run:",
        "- Refiner forward",
        "- Step 3 / final label conversion",
        "- human ground-truth accuracy review",
        "",
        "This is a candidate-label visualization.",
    ]
    y = note_y + 25
    for line in lines:
        draw.text((note_x + 30, y), line, fill="#f3f5f7", font=font(22))
        y += 31
    return canvas


def serializable_row(row: Any) -> dict[str, Any]:
    pitch = row.bbox_pitch
    return {
        "track_id": int(float(row.track_id)),
        "bbox_ltwh": [round(float(value), 4) for value in row.bbox_ltwh],
        "detection_confidence": round(float(row.bbox_conf), 6),
        "role": str(row.role),
        "role_confidence": scalar_or_none(row.role_confidence),
        "team": str(row.team),
        "jersey_number": scalar_or_none(row.jersey_number),
        "jersey_number_confidence": scalar_or_none(row.jersey_number_confidence),
        "legibility_score": scalar_or_none(row.legibility_score),
        "pitch_bottom_middle": {
            "x": round(float(pitch["x_bottom_middle"]), 6),
            "y": round(float(pitch["y_bottom_middle"]), 6),
        },
    }


def main() -> None:
    expected_outputs = [
        OUTPUT_DIR / "frame_0148_original.jpg",
        OUTPUT_DIR / "frame_0148_overlay.jpg",
        OUTPUT_DIR / "frame_0148_pitch_projection.jpg",
        OUTPUT_DIR / "overview.jpg",
        OUTPUT_DIR / "frame_0148_labels.json",
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    detections, images = load_tables()
    image_rows = images.loc[images["id"] == IMAGE_ID]
    if len(image_rows) != 1:
        raise AssertionError(f"Expected one image row for {IMAGE_ID}, got {len(image_rows)}")
    rows = detections.loc[detections["image_id"] == IMAGE_ID].copy()
    if len(rows) != 19 or rows["track_id"].nunique() != 19:
        raise AssertionError("Selected frame identity or detection count changed")

    source_path = Path(str(image_rows.iloc[0]["file_path"]))
    source = Image.open(source_path).convert("RGB")
    overlay = draw_overlay(source, rows)
    pitch = draw_pitch(rows)
    overview = make_overview(source, overlay, pitch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source.save(expected_outputs[0], quality=91)
    overlay.save(expected_outputs[1], quality=92)
    pitch.save(expected_outputs[2], quality=92)
    overview.save(expected_outputs[3], quality=92)

    payload = {
        "status": "candidate_labels_before_refiner",
        "sequence": "SNGS-10004",
        "video_id": VIDEO_ID,
        "source_frame_zero_based": int(image_rows.iloc[0]["frame"]),
        "source_filename": source_path.name,
        "source_path_read_only": str(source_path),
        "state_archive_read_only": str(ARCHIVE),
        "detection_count": len(rows),
        "unique_track_ids": int(rows["track_id"].nunique()),
        "non_null_jersey_numbers": int(rows["jersey_number"].count()),
        "team_counts": {str(key): int(value) for key, value in rows["team"].value_counts().items()},
        "role_counts": {str(key): int(value) for key, value in rows["role"].value_counts().items()},
        "labels": [serializable_row(row) for row in rows.itertuples(index=False)],
        "limitations": [
            "Refiner has not run.",
            "Step 3 and final training-label conversion have not run.",
            "No human ground-truth comparison has established semantic accuracy.",
        ],
    }
    expected_outputs[4].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for path in expected_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")
    print(json.dumps({"status": "passed", "output_dir": str(OUTPUT_DIR), "files": [path.name for path in expected_outputs]}))


if __name__ == "__main__":
    main()
