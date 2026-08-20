#!/usr/bin/env python3
"""Render three CPU-only, stage-by-stage F06 diagrams from saved predictions."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import textwrap
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
PREPARED = Path(
    "/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/"
    "SoccerNetGS/sn500/SNGS-10004/img1"
)
OUTPUT_DIR = REPO / "reports/one_match/20260819_sngs10004_end_to_end/visuals"
OUTPUTS = {
    "role": OUTPUT_DIR / "F06a_factory_role_pipeline.png",
    "jersey": OUTPUT_DIR / "F06b_factory_jersey_pipeline.png",
    "team": OUTPUT_DIR / "F06c_factory_team_pipeline.png",
}

BG = "#0f172a"
PANEL = "#172033"
PANEL_2 = "#202b41"
TEXT = "#f8fafc"
MUTED = "#b7c5db"
BLUE = "#60a5fa"
ORANGE = "#fb923c"
GREEN = "#34d399"
RED = "#fb7185"
YELLOW = "#facc15"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def load_state() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pickle.loads(archive.read("10004.pkl"))
        images = pickle.loads(archive.read("10004_image.pkl"))
    required = {
        "track_id", "image_id", "bbox_ltwh", "embeddings", "role_detection",
        "role_confidence", "role", "legibility_score", "jersey_number_detection",
        "jersey_number_confidence", "jersey_number", "team_cluster", "team", "bbox_pitch",
    }
    if not required.issubset(detections.columns):
        raise AssertionError(f"Missing columns: {sorted(required - set(detections.columns))}")
    return detections, images


def image_frame_map(images: pd.DataFrame) -> dict[Any, int]:
    return {idx: int(row["frame"]) for idx, row in images.iterrows()}


def crop(row: pd.Series, frame_by_image: dict[Any, int], size: tuple[int, int]) -> Image.Image:
    frame = frame_by_image[row["image_id"]]
    source = Image.open(PREPARED / f"{frame + 1:06d}.jpg").convert("RGB")
    x, y, width, height = [float(v) for v in row["bbox_ltwh"]]
    person = source.crop((
        max(0, math.floor(x)), max(0, math.floor(y)),
        min(source.width, math.ceil(x + width)), min(source.height, math.ceil(y + height)),
    ))
    person.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#080d16")
    panel.paste(person, ((size[0] - person.width) // 2, (size[1] - person.height) // 2))
    return panel


def wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, width: int,
            size: int = 22, fill: str = TEXT, spacing: int = 7) -> int:
    lines: list[str] = []
    for paragraph in value.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    draw.multiline_text(xy, "\n".join(lines), font=font(size), fill=fill, spacing=spacing)
    return len(lines) * (size + spacing)


def stage(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], number: int, title: str) -> None:
    draw.rounded_rectangle(box, radius=20, fill=PANEL, outline="#40516d", width=3)
    x1, y1, _, _ = box
    draw.ellipse((x1 + 22, y1 + 20, x1 + 70, y1 + 68), fill=BLUE)
    draw.text((x1 + 39, y1 + 27), str(number), font=font(22, True), fill="#07111f", anchor="ma")
    draw.text((x1 + 86, y1 + 24), title, font=font(27, True), fill=TEXT)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill=BLUE, width=6)
    x, y = end
    draw.polygon(((x, y), (x - 18, y - 11), (x - 18, y + 11)), fill=BLUE)


def title(draw: ImageDraw.ImageDraw, heading: str, subtitle: str) -> None:
    draw.text((35, 26), heading, font=font(34, True), fill=TEXT)
    draw.text((35, 74), subtitle, font=font(20), fill=MUTED)


def save_new(image: Image.Image, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    image.save(temp, format="PNG")
    os.replace(temp, destination)


def frame_row(track: pd.DataFrame, frame_by_image: dict[Any, int], frame: int) -> pd.Series:
    rows = track.loc[track["image_id"].map(frame_by_image) == frame]
    if len(rows) != 1:
        raise AssertionError(f"track/frame matched {len(rows)} rows: frame={frame}")
    return rows.iloc[0]


def render_role(detections: pd.DataFrame, frames: dict[Any, int]) -> Image.Image:
    canvas = Image.new("RGB", (2400, 1180), BG)
    draw = ImageDraw.Draw(canvas)
    title(draw, "F06a  Role pipeline — actual track 17", "PRTReID predicts each crop; SoccerFactory aggregates one label per track")
    boxes = [(30, 125, 570, 1125), (625, 125, 1165, 1125), (1220, 125, 1760, 1125), (1815, 125, 2370, 1125)]
    for i, (box, label) in enumerate(zip(boxes, ["Person crops", "Per-frame role", "Track voting", "Final role"]), 1):
        stage(draw, box, i, label)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 8, 625), (right[0] - 8, 625))

    track = detections.loc[detections.track_id == 17].copy()
    chosen_frames = [41, 42, 43, 47, 127, 149]
    selected = [frame_row(track, frames, value) for value in chosen_frames]
    for index, row in enumerate(selected):
        x = 58 + (index % 3) * 168
        y = 220 + (index // 3) * 395
        canvas.paste(crop(row, frames, (135, 260)), (x, y))
        draw.text((x, y + 270), f"frame {chosen_frames[index]}", font=font(18), fill=MUTED)

    for index, row in enumerate(selected):
        x = 655 + (index % 2) * 250
        y = 220 + (index // 2) * 255
        label = str(row["role_detection"])
        color = GREEN if label == "referee" else (YELLOW if label == "other" else BLUE)
        draw.rounded_rectangle((x, y, x + 220, y + 180), radius=12, fill=PANEL_2, outline=color, width=3)
        draw.text((x + 15, y + 17), f"frame {chosen_frames[index]}", font=font(18), fill=MUTED)
        draw.text((x + 15, y + 62), label, font=font(28, True), fill=color)
        draw.text((x + 15, y + 112), f"raw max {float(row['role_confidence']):.2f}", font=font(18), fill=TEXT)
    wrapped(draw, (655, 1000), "Raw max is not a calibrated probability.", 42, 18, MUTED)

    counts = track["role_detection"].astype(str).value_counts()
    sums = track.groupby("role_detection")["role_confidence"].sum().sort_values(ascending=False)
    draw.text((1255, 220), "All 221 observations", font=font(24, True), fill=TEXT)
    y = 280
    colors = {"player": BLUE, "referee": GREEN, "other": YELLOW, "goalkeeper": ORANGE}
    for label in ["player", "referee", "other", "goalkeeper"]:
        count = int(counts.get(label, 0))
        score = float(sums.get(label, 0.0))
        draw.text((1260, y), f"{label:10s} {count:3d} frames", font=font(23), fill=colors[label])
        draw.text((1260, y + 37), f"summed raw score: {score:.2f}", font=font(18), fill=MUTED)
        y += 120
    wrapped(draw, (1255, 800), "MajorityVoteTrackletFilter2 sums the available label scores for the same track and broadcasts the winner back to every row.", 39, 21, TEXT)

    draw.rounded_rectangle((1875, 260, 2310, 680), radius=24, fill="#102a2a", outline=GREEN, width=4)
    draw.text((2092, 330), "track 17", font=font(27), fill=MUTED, anchor="ma")
    draw.text((2092, 435), "player", font=font(52, True), fill=BLUE, anchor="ma")
    draw.text((2092, 520), "221 / 221 rows", font=font(23), fill=TEXT, anchor="ma")
    wrapped(draw, (1870, 745), "Interpretation: the pipeline works mechanically, but this track is visibly referee-like in several frames. The final role is therefore not trustworthy.", 39, 22, RED)
    return canvas


def filter_jersey_like_pipeline(track: pd.DataFrame) -> Counter[str]:
    detections = list(track["jersey_number_detection"])
    kept = detections.copy()
    for i, current in enumerate(detections):
        start, end = max(0, i - 1), min(len(detections), i + 2)
        window = detections[start:end]
        if current is None or (isinstance(current, float) and math.isnan(current)) or sum(v == current for v in window) < 2:
            kept[i] = None
    return Counter(str(value) for value in kept if value is not None and not (isinstance(value, float) and math.isnan(value)))


def render_jersey(detections: pd.DataFrame, frames: dict[Any, int]) -> Image.Image:
    canvas = Image.new("RGB", (2400, 1180), BG)
    draw = ImageDraw.Draw(canvas)
    title(draw, "F06b  Jersey-number pipeline — actual track 15", "Legibility gate → Qwen2.5-VL OCR → temporal filtering/vote → one track-level number")
    boxes = [(30, 125, 570, 1125), (625, 125, 1165, 1125), (1220, 125, 1760, 1125), (1815, 125, 2370, 1125)]
    for i, (box, label) in enumerate(zip(boxes, ["Legibility", "Per-frame OCR", "Filter + vote", "Final number"]), 1):
        stage(draw, box, i, label)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 8, 625), (right[0] - 8, 625))

    track = detections.loc[detections.track_id == 15].copy()
    chosen_frames = [16, 18, 26, 43, 61, 122]
    selected = [frame_row(track, frames, value) for value in chosen_frames]
    for index, row in enumerate(selected):
        x = 58 + (index % 3) * 168
        y = 205 + (index // 3) * 410
        canvas.paste(crop(row, frames, (135, 260)), (x, y))
        score = float(row["legibility_score"])
        color = GREEN if score >= 0.5 else RED
        draw.text((x, y + 270), f"f{chosen_frames[index]}  {score:.3f}", font=font(18), fill=color)
        draw.text((x, y + 299), "OCR" if score >= 0.5 else "SKIP", font=font(18, True), fill=color)
    draw.text((58, 1010), "gate threshold = 0.50", font=font(21, True), fill=TEXT)

    for index, row in enumerate(selected):
        x = 660
        y = 215 + index * 125
        raw = row["jersey_number_detection"]
        value = "none" if pd.isna(raw) else str(raw)
        color = RED if value == "none" else (GREEN if value == "8" else YELLOW)
        draw.rounded_rectangle((x, y, x + 450, y + 92), radius=12, fill=PANEL_2)
        draw.text((x + 16, y + 14), f"frame {chosen_frames[index]:3d}", font=font(19), fill=MUTED)
        draw.text((x + 235, y + 14), f"OCR → {value}", font=font(25, True), fill=color)
        if value == "none":
            draw.text((x + 235, y + 55), "not sent / no parse", font=font(15), fill=MUTED)

    raw_counts = track["jersey_number_detection"].dropna().astype(str).value_counts()
    kept_counts = filter_jersey_like_pipeline(track)
    draw.text((1255, 215), "Raw OCR counts", font=font(23, True), fill=TEXT)
    draw.text((1255, 265), "  ".join(f"{k}:{v}" for k, v in raw_counts.head(8).items()), font=font(18), fill=MUTED)
    draw.text((1255, 340), "After local 2-of-3 filter", font=font(23, True), fill=TEXT)
    draw.text((1255, 390), "  ".join(f"{k}:{v}" for k, v in kept_counts.most_common(8)), font=font(18), fill=MUTED)
    draw.text((1255, 500), "Vote winner", font=font(23, True), fill=TEXT)
    draw.text((1475, 610), "8", font=font(84, True), fill=GREEN, anchor="mm")
    wrapped(draw, (1255, 720), "Only locally repeated readings survive. Remaining values are voted using jersey_number_confidence.", 39, 22, TEXT)
    wrapped(draw, (1255, 920), "Risk: if track 15 switches to another person, both players' OCR observations enter this same vote.", 39, 21, RED)

    draw.rounded_rectangle((1885, 275, 2300, 690), radius=24, fill="#102a2a", outline=GREEN, width=4)
    draw.text((2092, 345), "track 15", font=font(27), fill=MUTED, anchor="ma")
    draw.text((2092, 455), "8", font=font(86, True), fill=GREEN, anchor="ma")
    draw.text((2092, 570), "applied to 184 rows", font=font(22), fill=TEXT, anchor="ma")
    wrapped(draw, (1875, 760), "This is a track-level consensus, not proof that every frame visibly contains number 8.", 38, 22, YELLOW)
    return canvas


def embedding_grid(values: Any, size: int = 112) -> Image.Image:
    array = np.asarray(values, dtype=np.float32).reshape(16, 16)
    array = array - float(array.mean())
    scale = float(np.max(np.abs(array)))
    array = array / scale if scale > 0 else array
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    rgb[..., 2] = (np.clip(-array, 0, 1) * 255).astype(np.uint8)
    rgb[..., 1] = ((1 - np.abs(array)) * 35).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def representative(group: pd.DataFrame) -> pd.Series:
    areas = group["bbox_ltwh"].apply(lambda b: float(b[2]) * float(b[3]))
    return group.loc[areas.idxmax()]


def pitch_x(value: Any) -> float:
    return float(value["x_bottom_middle"]) if isinstance(value, dict) else float("nan")


def render_team(detections: pd.DataFrame, frames: dict[Any, int]) -> Image.Image:
    canvas = Image.new("RGB", (2400, 1180), BG)
    draw = ImageDraw.Draw(canvas)
    title(draw, "F06c  Team pipeline — actual 49 saved tracks", "Track appearance embedding → mean per track → KMeans(2) → spatial names left/right")
    boxes = [(30, 125, 570, 1125), (625, 125, 1165, 1125), (1220, 125, 1760, 1125), (1815, 125, 2370, 1125)]
    for i, (box, label) in enumerate(zip(boxes, ["Track crops", "Mean embedding", "KMeans clusters", "Side naming"]), 1):
        stage(draw, box, i, label)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 8, 625), (right[0] - 8, 625))

    selected_ids = [7, 15, 17, 39, 41, 34]
    groups = {int(tid): group for tid, group in detections.groupby("track_id")}
    for index, track_id in enumerate(selected_ids):
        group = groups[track_id]
        row = representative(group)
        x = 58 + (index % 3) * 168
        y = 210 + (index // 3) * 410
        canvas.paste(crop(row, frames, (135, 260)), (x, y))
        draw.text((x, y + 270), f"track {track_id}", font=font(19, True), fill=TEXT)
        draw.text((x, y + 300), f"{len(group)} frames", font=font(16), fill=MUTED)

        mean_embedding = np.mean(np.vstack(group["embeddings"].values), axis=0)
        ex = 655 + (index % 2) * 250
        ey = 215 + (index // 2) * 260
        canvas.paste(embedding_grid(mean_embedding), (ex, ey))
        cluster = int(float(group["team_cluster"].iloc[0]))
        draw.text((ex, ey + 122), f"track {track_id}", font=font(18, True), fill=TEXT)
        draw.text((ex, ey + 151), f"mean of {len(group)} × 256D", font=font(15), fill=MUTED)
        draw.text((ex, ey + 181), f"saved cluster {cluster}", font=font(18), fill=BLUE if cluster == 0 else ORANGE)
    wrapped(draw, (655, 1000), "Grid colors only visualize numeric embedding values; they do not directly mean jersey color.", 43, 18, MUTED)

    by_track = detections.groupby("track_id").agg(cluster=("team_cluster", "first"), rows=("track_id", "size"))
    cluster_track_counts = by_track["cluster"].value_counts().sort_index()
    cluster_row_counts = detections["team_cluster"].value_counts().sort_index()
    for cluster, color, y in [(0, BLUE, 300), (1, ORANGE, 620)]:
        draw.rounded_rectangle((1260, y, 1720, y + 230), radius=18, fill=PANEL_2, outline=color, width=4)
        draw.text((1300, y + 28), f"cluster {cluster}", font=font(31, True), fill=color)
        draw.text((1300, y + 92), f"{int(cluster_track_counts.get(cluster, 0))} tracks", font=font(24), fill=TEXT)
        draw.text((1300, y + 140), f"{int(cluster_row_counts.get(cluster, 0))} detections", font=font(21), fill=MUTED)
    wrapped(draw, (1255, 930), "KMeans is forced to make two clusters; it does not know the club names and does not guarantee balanced groups.", 40, 20, RED)

    cluster_x: dict[int, float] = {}
    for cluster, group in detections.groupby("team_cluster"):
        cluster_x[int(float(cluster))] = float(np.nanmean([pitch_x(v) for v in group["bbox_pitch"]]))
    team_counts = detections["team"].value_counts()
    draw.text((1855, 235), "Mean pitch x", font=font(24, True), fill=TEXT)
    draw.text((1855, 295), f"cluster 0: {cluster_x.get(0, float('nan')):+.2f} m", font=font(22), fill=BLUE)
    draw.text((1855, 345), f"cluster 1: {cluster_x.get(1, float('nan')):+.2f} m", font=font(22), fill=ORANGE)
    wrapped(draw, (1855, 420), "The cluster with the smaller mean x is named left; the other is named right.", 38, 22, TEXT)
    draw.rounded_rectangle((1865, 610, 2315, 850), radius=20, fill="#241a12", outline=ORANGE, width=4)
    draw.text((2090, 665), f"left  {int(team_counts.get('left', 0))}", font=font(30, True), fill=BLUE, anchor="ma")
    draw.text((2090, 735), f"right  {int(team_counts.get('right', 0))}", font=font(30, True), fill=ORANGE, anchor="ma")
    draw.text((2090, 795), "not Chelsea / Burnley", font=font(18), fill=MUTED, anchor="ma")
    wrapped(draw, (1855, 920), "46 vs 3 tracks is a clear failure signal for this sample, even though the fields were produced successfully.", 38, 21, RED)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    detections, images = load_state()
    frames = image_frame_map(images)
    renderers = {
        "role": render_role,
        "jersey": render_jersey,
        "team": render_team,
    }
    for name, renderer in renderers.items():
        save_new(renderer(detections, frames), OUTPUTS[name], args.overwrite)
        print(f"[OUTPUT] {OUTPUTS[name]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
