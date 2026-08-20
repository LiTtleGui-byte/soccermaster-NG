#!/usr/bin/env python3
"""Render three CPU-only views of actual Step-1 intermediate columns."""

from __future__ import annotations

import argparse
import colorsys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_step1/run5/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260818_completed_pipeline_example"
YOLO_OUTPUT = OUTPUT_DIR / "frame_0148_step1a_yolo_detection.jpg"
REID_OUTPUT = OUTPUT_DIR / "frame_0148_step1b_reid_features.jpg"
TRACK_OUTPUT = OUTPUT_DIR / "frame_0148_step1c_strongsort_tracks.jpg"
VIDEO_ID = "10004"
IMAGE_ID = "610004000148"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def load_state() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"{VIDEO_ID}.pkl") as handle:
            detections = pd.read_pickle(handle)
        with archive.open(f"{VIDEO_ID}_image.pkl") as handle:
            images = pd.read_pickle(handle)
    return detections, images


def draw_label(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    x: float,
    y: float,
    y2: float,
    text: str,
    color: tuple[int, int, int],
    text_font: ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    label_x = max(0, min(x, canvas.width - width - 8))
    label_y = y - height - 8 if y - height - 8 > 44 else y2 + 3
    draw.rectangle((label_x, label_y, label_x + width + 8, label_y + height + 6), fill="black")
    draw.text((label_x + 4, label_y + 2), text, fill=color, font=text_font)


def render_yolo(source: Image.Image, rows: pd.DataFrame) -> Image.Image:
    canvas = source.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    color = (255, 190, 45)
    draw.rectangle((0, 0, canvas.width, 42), fill="black")
    draw.text(
        (12, 8),
        "STEP 1A / YOLO ACTUAL OUTPUT | person boxes + confidence | D = frame-local key",
        fill="white",
        font=font(24),
    )
    for row in rows.itertuples(index=False):
        x, y, width, height = (float(value) for value in row.bbox_ltwh)
        x2, y2 = x + width, y + height
        draw.rectangle((x, y, x2, y2), outline=color, width=4)
        draw_label(
            draw, canvas, x, y, y2,
            f"D{int(row.local_detection):02d} | {float(row.bbox_conf):.2f}",
            color, font(18),
        )
    return canvas


def crop_person(source: Image.Image, bbox: Any, size: tuple[int, int]) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    crop = source.crop((
        max(0, int(np.floor(x))),
        max(0, int(np.floor(y))),
        min(source.width, int(np.ceil(x + width))),
        min(source.height, int(np.ceil(y + height))),
    )).convert("RGB")
    crop.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (28, 28, 28))
    panel.paste(crop, ((size[0] - crop.width) // 2, (size[1] - crop.height) // 2))
    return panel


def embedding_grid(embedding: Any, side: int = 176) -> Image.Image:
    values = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if values.size != 256 or not np.isfinite(values).all():
        raise AssertionError("Expected one finite 256-dimensional embedding")
    centered = values - float(values.mean())
    scale = float(np.max(np.abs(centered)))
    normalized = centered / scale if scale > 0 else centered
    red = np.clip(normalized, 0.0, 1.0) * 255.0
    blue = np.clip(-normalized, 0.0, 1.0) * 255.0
    green = (1.0 - np.abs(normalized)) * 35.0
    rgb = np.stack([red, green, blue], axis=1).reshape(16, 16, 3).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((side, side), Image.Resampling.NEAREST)


def render_reid(source: Image.Image, rows: pd.DataFrame) -> Image.Image:
    canvas = Image.new("RGB", (1920, 1320), (10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (14, 10),
        "STEP 1B / PRTReID ACTUAL OUTPUT | player crop + 256-D appearance embedding",
        fill="white", font=font(25),
    )
    draw.text(
        (14, 45),
        "Embedding grids are centered and normalized per detection for display; colors have no semantic label.",
        fill=(190, 190, 190), font=font(16),
    )
    for index, row in enumerate(rows.itertuples(index=False)):
        x0 = 14 + (index % 5) * 381
        y0 = 82 + (index // 5) * 306
        draw.rounded_rectangle(
            (x0, y0, x0 + 374, y0 + 300), radius=8,
            fill=(24, 24, 24), outline=(80, 80, 80), width=2,
        )
        canvas.paste(crop_person(source, row.bbox_ltwh, (138, 218)), (x0 + 10, y0 + 42))
        canvas.paste(embedding_grid(row.embeddings), (x0 + 184, y0 + 55))
        role = str(row.role_detection)
        role_confidence = float(row.role_confidence)
        visibility = float(np.asarray(row.visibility_scores, dtype=np.float32).reshape(-1)[0])
        draw.text(
            (x0 + 10, y0 + 10),
            f"D{int(row.local_detection):02d}  role={role} | raw score {role_confidence:.2f}",
            fill=(255, 205, 80), font=font(16),
        )
        draw.text(
            (x0 + 184, y0 + 239), f"visibility {visibility:.2f}",
            fill=(210, 210, 210), font=font(15),
        )
        draw.text(
            (x0 + 184, y0 + 265), "16 x 16 = 256 dims",
            fill=(150, 150, 150), font=font(14),
        )
    return canvas


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def render_tracks(
    source: Image.Image,
    current_rows: pd.DataFrame,
    detections: pd.DataFrame,
    images: pd.DataFrame,
) -> Image.Image:
    canvas = source.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 42), fill="black")
    draw.text(
        (12, 8),
        "STEP 1C / StrongSORT ACTUAL OUTPUT | Track ID + preceding 12-frame trail",
        fill="white", font=font(24),
    )
    frame_by_image = images["frame"].to_dict()
    history = detections.loc[detections["track_id"].isin(current_rows["track_id"])].copy()
    history["_frame"] = history["image_id"].map(frame_by_image)
    current_frame = int(frame_by_image[IMAGE_ID])
    history = history.loc[history["_frame"].between(current_frame - 12, current_frame)]
    for row in current_rows.sort_values("track_id").itertuples(index=False):
        track_id = int(float(row.track_id))
        color = track_color(track_id)
        trail = history.loc[history["track_id"] == row.track_id].sort_values("_frame")
        points = []
        for box in trail["bbox_ltwh"]:
            x, y, width, height = (float(value) for value in box)
            points.append((x + width / 2, y + height))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for point in points[:-1]:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        x, y, width, height = (float(value) for value in row.bbox_ltwh)
        x2, y2 = x + width, y + height
        draw.rectangle((x, y, x2, y2), outline=color, width=4)
        draw.ellipse((x + width / 2 - 5, y2 - 5, x + width / 2 + 5, y2 + 5), fill=color)
        draw_label(draw, canvas, x, y, y2, f"ID {track_id}", color, font(18))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace only this renderer's three named output images.",
    )
    args = parser.parse_args()
    outputs = (YOLO_OUTPUT, REID_OUTPUT, TRACK_OUTPUT)
    occupied = [str(path) for path in outputs if path.exists() or path.is_symlink()]
    if occupied and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {occupied}")

    detections, images = load_state()
    rows = detections.loc[detections["image_id"] == IMAGE_ID].copy()
    if len(rows) != 19:
        raise AssertionError("Expected 19 actual detections in frame 148")
    required = {
        "bbox_ltwh", "bbox_conf", "embeddings", "visibility_scores",
        "role_detection", "role_confidence", "track_id",
    }
    if not required.issubset(rows.columns):
        raise AssertionError(f"Missing actual intermediate columns: {sorted(required - set(rows.columns))}")
    if rows["track_id"].isna().any():
        raise AssertionError("StrongSORT output has a null track ID")
    rows = rows.sort_index().copy()
    rows["local_detection"] = np.arange(1, len(rows) + 1)

    image_row = images.loc[images.index == IMAGE_ID]
    if len(image_row) != 1:
        raise AssertionError("Source frame identity is not unique")
    source_path = Path(str(image_row.iloc[0]["file_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = Image.open(source_path).convert("RGB")

    render_yolo(source, rows).save(YOLO_OUTPUT, quality=92)
    render_reid(source, rows).save(REID_OUTPUT, quality=92)
    render_tracks(source, rows, detections, images).save(TRACK_OUTPUT, quality=92)
    for output in outputs:
        if not output.is_file() or output.stat().st_size == 0:
            raise AssertionError(f"Missing output: {output}")
    print(
        "passed: " + ", ".join(path.name for path in outputs)
        + "; 19 detections with real YOLO/ReID/StrongSORT columns"
    )


if __name__ == "__main__":
    main()
