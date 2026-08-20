#!/usr/bin/env python3
"""Render CPU-only Step-1 and enrichment views of the same real frame."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
STEP1_ARCHIVE = REPO / ".runtime/g10/sngs10004_step1/run5/states/sn-gamestate.pklz"
ENRICHED_ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260818_completed_pipeline_example"
STEP1_OUTPUT = OUTPUT_DIR / "frame_0148_step1_only.jpg"
ENRICHED_OUTPUT = OUTPUT_DIR / "frame_0148_enrichment_only.jpg"
VIDEO_ID = "10004"
IMAGE_ID = "610004000148"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def value_or_none(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def load_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
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
    draw.rectangle(
        (label_x, label_y, label_x + width + 8, label_y + height + 6),
        fill=(0, 0, 0),
    )
    draw.text((label_x + 4, label_y + 2), text, fill=color, font=text_font)


def draw_step1(source: Image.Image, rows: pd.DataFrame) -> Image.Image:
    canvas = source.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    title_font = font(24)
    text_font = font(19)
    box_color = (255, 190, 45)
    draw.rectangle((0, 0, canvas.width, 42), fill=(0, 0, 0))
    draw.text(
        (12, 8),
        "STEP 1 ACTUAL OUTPUT | person bbox + track ID + detection confidence",
        fill="white",
        font=title_font,
    )
    for row in rows.sort_values("bbox_conf", ascending=False).itertuples(index=False):
        x, y, width, height = (float(value) for value in row.bbox_ltwh)
        x2, y2 = x + width, y + height
        track_id = int(float(row.track_id))
        confidence = float(row.bbox_conf)
        draw.rectangle((x, y, x2, y2), outline=box_color, width=4)
        draw_label(
            draw, canvas, x, y, y2,
            f"ID {track_id} | det {confidence:.2f}",
            box_color, text_font,
        )
    return canvas


def team_color(team: str) -> tuple[int, int, int]:
    if team == "left":
        return (255, 86, 86)
    if team == "right":
        return (65, 210, 255)
    return (235, 235, 235)


def draw_enrichment(source: Image.Image, rows: pd.DataFrame) -> Image.Image:
    legend_height = 300
    canvas = Image.new("RGB", (source.width, source.height + legend_height), (0, 0, 0))
    canvas.paste(source.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = font(24)
    text_font = font(17)
    legend_font = font(17)
    draw.rectangle((0, 0, canvas.width, 42), fill=(0, 0, 0))
    draw.text(
        (12, 8),
        "ENRICHMENT ACTUAL OUTPUT | team + role + jersey + pitch position | NOT ground truth",
        fill="white",
        font=title_font,
    )
    for row in rows.sort_values("bbox_conf", ascending=False).itertuples(index=False):
        x, y, width, height = (float(value) for value in row.bbox_ltwh)
        x2, y2 = x + width, y + height
        track_id = int(float(row.track_id))
        team = str(row.team)
        role = str(row.role)
        color = team_color(team)
        jersey = value_or_none(row.jersey_number)
        jersey_text = "?" if jersey is None else str(int(float(jersey)))
        pitch = row.bbox_pitch
        pitch_x = float(pitch["x_bottom_middle"])
        pitch_y = float(pitch["y_bottom_middle"])
        draw.rectangle((x, y, x2, y2), outline=color, width=4)
        foot_x, foot_y = x + width / 2, y2
        draw.ellipse(
            (foot_x - 5, foot_y - 5, foot_x + 5, foot_y + 5),
            fill=(255, 230, 50),
        )
        draw_label(draw, canvas, x, y, y2, f"ID {track_id}", color, text_font)

    legend_top = source.height
    draw.text(
        (16, legend_top + 10),
        "Per-track enrichment values (track ID is the join key; pitch = candidate x,y)",
        fill="white",
        font=font(19),
    )
    ordered = rows.sort_values("track_id").reset_index(drop=True)
    column_width = source.width // 3
    for index, row in enumerate(ordered.itertuples(index=False)):
        column = index // 7
        line = index % 7
        track_id = int(float(row.track_id))
        team = str(row.team)
        role = str(row.role)
        jersey = value_or_none(row.jersey_number)
        jersey_text = "?" if jersey is None else str(int(float(jersey)))
        pitch_x = float(row.bbox_pitch["x_bottom_middle"])
        pitch_y = float(row.bbox_pitch["y_bottom_middle"])
        text = (
            f"ID {track_id:02d} | {team:<5} | {role:<6} | "
            f"#{jersey_text:<2} | pitch ({pitch_x:5.1f},{pitch_y:5.1f})"
        )
        draw.text(
            (16 + column * column_width, legend_top + 48 + line * 33),
            text,
            fill=team_color(team),
            font=legend_font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this renderer's two named output images.",
    )
    args = parser.parse_args()
    occupied = [
        str(path) for path in (STEP1_OUTPUT, ENRICHED_OUTPUT)
        if path.exists() or path.is_symlink()
    ]
    if occupied and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {occupied}")
    step1, step1_images = load_archive(STEP1_ARCHIVE)
    enriched, enriched_images = load_archive(ENRICHED_ARCHIVE)
    step1_rows = step1.loc[step1["image_id"] == IMAGE_ID].copy()
    enriched_rows = enriched.loc[enriched["image_id"] == IMAGE_ID].copy()
    if len(step1_rows) != 19 or len(enriched_rows) != 19:
        raise AssertionError("Expected 19 detections in both real stage outputs")
    step1_rows = step1_rows.sort_values("track_id").reset_index(drop=True)
    enriched_rows = enriched_rows.sort_values("track_id").reset_index(drop=True)
    if step1_rows["track_id"].tolist() != enriched_rows["track_id"].tolist():
        raise AssertionError("Track identities changed between stage archives")
    if not np.allclose(
        np.asarray(
            [[float(value) for value in box] for box in step1_rows["bbox_ltwh"]],
            dtype=np.float64,
        ),
        np.asarray(
            [[float(value) for value in box] for box in enriched_rows["bbox_ltwh"]],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Bounding boxes changed between stage archives")
    if not np.allclose(
        step1_rows["bbox_conf"].to_numpy(dtype=np.float64),
        enriched_rows["bbox_conf"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Detection confidence changed between stage archives")

    step1_image_row = step1_images.loc[step1_images["id"] == IMAGE_ID]
    enriched_image_row = enriched_images.loc[enriched_images["id"] == IMAGE_ID]
    if len(step1_image_row) != 1 or len(enriched_image_row) != 1:
        raise AssertionError("Frame identity is not unique in both archives")
    step1_source = Path(str(step1_image_row.iloc[0]["file_path"]))
    enriched_source = Path(str(enriched_image_row.iloc[0]["file_path"]))
    if step1_source != enriched_source or not step1_source.is_file():
        raise AssertionError("Source image identity changed between stages")
    source = Image.open(step1_source).convert("RGB")
    draw_step1(source, step1_rows).save(STEP1_OUTPUT, quality=92)
    draw_enrichment(source, enriched_rows).save(ENRICHED_OUTPUT, quality=92)
    for output in (STEP1_OUTPUT, ENRICHED_OUTPUT):
        if not output.is_file() or output.stat().st_size == 0:
            raise AssertionError(f"Missing output: {output}")
    print(
        f"passed: {STEP1_OUTPUT.name}, {ENRICHED_OUTPUT.name}; "
        "19 matched real detections"
    )


if __name__ == "__main__":
    main()
