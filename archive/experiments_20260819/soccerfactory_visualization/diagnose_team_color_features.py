#!/usr/bin/env python3
"""Compare existing ReID KMeans with CPU-only upper-body color clustering.

This is a diagnostic on cached detections and source JPEGs. It does not modify
the SoccerFactory archive or claim that unsupervised color clusters are truth.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260818_team_color_diagnostic"
VIDEO_ID = "10004"
TOP_CROPS_PER_TRACK = 12
MIN_ROWS_FILTERED = 5
MIN_VALID_CROPS = 3


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"{VIDEO_ID}.pkl") as handle:
            detections = pd.read_pickle(handle)
        with archive.open(f"{VIDEO_ID}_image.pkl") as handle:
            images = pd.read_pickle(handle)
    if len(detections) != 3176 or detections.track_id.nunique() != 49 or len(images) != 255:
        raise AssertionError("Fixed run2 archive identity changed")
    return detections, images


def torso_crop(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    left = max(0, int(math.floor(x + 0.12 * width)))
    right = min(image.width, int(math.ceil(x + 0.88 * width)))
    top = max(0, int(math.floor(y + 0.08 * height)))
    bottom = min(image.height, int(math.ceil(y + 0.58 * height)))
    return image.crop((left, top, right, bottom)).convert("RGB")


def hue_feature(crop: Image.Image) -> tuple[np.ndarray, int]:
    hsv = np.asarray(crop.convert("HSV"), dtype=np.uint8)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    # Exclude low-information pixels and the dominant green pitch range.
    informative = (saturation >= 45) & (value >= 35)
    green = (hue >= 45) & (hue <= 115)
    mask = informative & ~green
    selected = hue[mask]
    if selected.size < 20:
        selected = hue[informative]
    if selected.size == 0:
        return np.zeros(26, dtype=np.float32), 0
    histogram, _ = np.histogram(selected, bins=24, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    pixels = np.asarray(crop, dtype=np.float32)[mask] if mask.any() else np.asarray(crop, dtype=np.float32).reshape(-1, 3)
    chroma = pixels.mean(axis=0)
    chroma /= max(float(chroma.sum()), 1.0)
    feature = np.concatenate([histogram, chroma[:2]]).astype(np.float32)
    feature /= max(float(np.linalg.norm(feature)), 1e-12)
    return feature, int(selected.size)


def extract_track_features(
    detections: pd.DataFrame, images: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    path_by_id = dict(zip(images.id, images.file_path))
    cache: dict[Any, Image.Image] = {}
    records: list[dict[str, Any]] = []
    features: list[np.ndarray] = []

    for track_id, group in detections.groupby("track_id"):
        selected = group.sort_values("bbox_conf", ascending=False).head(TOP_CROPS_PER_TRACK)
        crop_features: list[np.ndarray] = []
        selected_pixels = 0
        for row in selected.itertuples(index=False):
            if row.image_id not in cache:
                cache[row.image_id] = Image.open(Path(str(path_by_id[row.image_id]))).convert("RGB")
            feature, pixels = hue_feature(torso_crop(cache[row.image_id], row.bbox_ltwh))
            if pixels > 0:
                crop_features.append(feature)
                selected_pixels += pixels
        if not crop_features:
            continue
        mean_feature = np.mean(np.vstack(crop_features), axis=0)
        mean_feature /= max(float(np.linalg.norm(mean_feature)), 1e-12)
        role_counts = group.role_detection.value_counts()
        nonplayer_signals = int(role_counts.drop(labels=["player"], errors="ignore").sum())
        records.append(
            {
                "track_id": int(float(track_id)),
                "rows": int(len(group)),
                "valid_color_crops": int(len(crop_features)),
                "selected_color_pixels": int(selected_pixels),
                "saved_reid_cluster": int(float(group.team_cluster.iloc[0])),
                "saved_team": str(group.team.iloc[0]),
                "nonplayer_role_signals": nonplayer_signals,
                "role_suspect": bool(nonplayer_signals > 0),
            }
        )
        features.append(mean_feature)
    return pd.DataFrame(records), np.vstack(features)


def fit_labels(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    labels = np.full(len(features), -1, dtype=int)
    fitted = KMeans(n_clusters=2, random_state=0).fit_predict(features[mask])
    labels[mask] = fitted
    return labels, float(silhouette_score(features[mask], fitted))


def representative_rows(detections: pd.DataFrame) -> pd.DataFrame:
    return detections.loc[detections.groupby("track_id").bbox_conf.idxmax()]


def crop_with_context(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    pad_x, pad_y = max(16, 0.45 * width), max(12, 0.16 * height)
    box = (
        max(0, int(x - pad_x)),
        max(0, int(y - pad_y)),
        min(image.width, int(x + width + pad_x)),
        min(image.height, int(y + height + pad_y)),
    )
    return image.crop(box).convert("RGB")


def label_color(label: int) -> tuple[int, int, int]:
    return {0: (255, 190, 55), 1: (90, 145, 255), -1: (145, 150, 158)}[int(label)]


def draw_contact_sheet(
    detections: pd.DataFrame,
    images: pd.DataFrame,
    tracks: pd.DataFrame,
    label_column: str,
    title: str,
) -> Image.Image:
    reps = representative_rows(detections).copy()
    track_lookup = tracks.set_index("track_id")
    reps["_diagnostic_label"] = reps.track_id.map(
        lambda value: int(track_lookup.loc[int(float(value)), label_column])
    )
    reps = reps.sort_values(["_diagnostic_label", "track_id"], ascending=[False, True], kind="stable")
    path_by_id = dict(zip(images.id, images.file_path))
    cache: dict[Any, Image.Image] = {}
    panel_w, panel_h, columns = 230, 230, 7
    rows_count = math.ceil(len(reps) / columns)
    header_h = 88
    sheet = Image.new("RGB", (panel_w * columns, header_h + panel_h * rows_count), "#11161c")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), title, fill="white", font=font(27))
    draw.text(
        (16, 48),
        "Orange=A, blue=B, gray=excluded. C0/C1 is the saved ReID cluster. No ground truth used.",
        fill="#d9e0e7",
        font=font(17),
    )
    for position, row in enumerate(reps.itertuples(index=False)):
        if row.image_id not in cache:
            cache[row.image_id] = Image.open(Path(str(path_by_id[row.image_id]))).convert("RGB")
        crop = crop_with_context(cache[row.image_id], row.bbox_ltwh)
        crop.thumbnail((panel_w - 14, panel_h - 58), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), "#202832")
        panel.paste(crop, ((panel_w - crop.width) // 2, 4))
        track_id = int(float(row.track_id))
        data = track_lookup.loc[track_id]
        label = int(data[label_column])
        color = label_color(label)
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((1, 1, panel_w - 2, panel_h - 2), outline=color, width=5)
        panel_draw.rectangle((0, panel_h - 53, panel_w, panel_h), fill="black")
        label_text = "excluded" if label < 0 else ("A" if label == 0 else "B")
        panel_draw.text(
            (7, panel_h - 49),
            f"ID {track_id} | color {label_text} | saved C{int(data.saved_reid_cluster)}",
            fill=color,
            font=font(14),
        )
        panel_draw.text(
            (7, panel_h - 27),
            f"rows {int(data.rows)} | role signals {int(data.nonplayer_role_signals)}",
            fill="white",
            font=font(14),
        )
        sheet.paste(panel, ((position % columns) * panel_w, header_h + (position // columns) * panel_h))
    return sheet


def main() -> None:
    outputs = {
        "result": OUTPUT_DIR / "result.json",
        "all_tracks": OUTPUT_DIR / "color_clusters_all_tracks.jpg",
        "filtered": OUTPUT_DIR / "color_clusters_role_length_filtered.jpg",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    detections, images = load_tables()
    tracks, features = extract_track_features(detections, images)
    if len(tracks) != 49 or features.shape != (49, 26):
        raise AssertionError(f"Unexpected color feature contract: tracks={len(tracks)} shape={features.shape}")
    all_mask = np.ones(len(tracks), dtype=bool)
    filtered_mask = (
        (tracks.rows.to_numpy() >= MIN_ROWS_FILTERED)
        & (tracks.valid_color_crops.to_numpy() >= MIN_VALID_CROPS)
        & (~tracks.role_suspect.to_numpy())
    )
    all_labels, all_silhouette = fit_labels(features, all_mask)
    filtered_labels, filtered_silhouette = fit_labels(features, filtered_mask)
    tracks["color_cluster_all"] = all_labels
    tracks["color_cluster_filtered"] = filtered_labels

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_contact_sheet(
        detections,
        images,
        tracks,
        "color_cluster_all",
        "Upper-body HSV color KMeans: all 49 tracks",
    ).save(outputs["all_tracks"], quality=92)
    draw_contact_sheet(
        detections,
        images,
        tracks,
        "color_cluster_filtered",
        "Upper-body HSV color KMeans: filtered tracks",
    ).save(outputs["filtered"], quality=92)

    def counts(labels: np.ndarray) -> dict[str, int]:
        values, frequencies = np.unique(labels, return_counts=True)
        return {str(int(value)): int(frequency) for value, frequency in zip(values, frequencies)}

    eligible = tracks.loc[filtered_mask, "track_id"].astype(int).tolist()
    excluded = tracks.loc[~filtered_mask, "track_id"].astype(int).tolist()
    result = {
        "status": "passed",
        "scope": "CPU-only color-feature diagnostic on fixed cached detections",
        "input_archive": str(ARCHIVE),
        "gpu_used": False,
        "model_or_pipeline_rerun": False,
        "labels_written_back": False,
        "feature_contract": {
            "region": "bbox x=12%-88%, y=8%-58%",
            "maximum_crops_per_track": TOP_CROPS_PER_TRACK,
            "pixels": "HSV saturation>=45 and value>=35; exclude hue 45..115 when possible",
            "feature": "24-bin hue histogram plus normalized red/green chromaticity",
            "clustering": "KMeans(n_clusters=2, random_state=0)",
        },
        "existing_reid": {
            "track_counts": {str(int(key)): int(value) for key, value in tracks.saved_reid_cluster.value_counts().sort_index().items()},
        },
        "color_all_tracks": {
            "track_counts": counts(all_labels),
            "silhouette": all_silhouette,
        },
        "color_role_length_filtered": {
            "track_counts_including_excluded_minus_one": counts(filtered_labels),
            "eligible_track_ids": eligible,
            "excluded_track_ids": excluded,
            "eligibility": f"rows>={MIN_ROWS_FILTERED}, valid_color_crops>={MIN_VALID_CROPS}, no non-player role_detection signal",
            "silhouette": filtered_silhouette,
        },
        "track_assignments": tracks.to_dict("records"),
        "interpretation_boundary": [
            "Balanced clusters do not by themselves prove correct teams.",
            "The color diagnostic uses no team ground truth and does not modify SoccerFactory.",
            "Visual crop review is required before deciding whether the feature is a viable replacement.",
        ],
        "artifacts": {key: str(path) for key, path in outputs.items() if key != "result"},
    }
    outputs["result"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing output: {path}")
    print(
        json.dumps(
            {
                "status": "passed",
                "reid": result["existing_reid"]["track_counts"],
                "color_all": result["color_all_tracks"]["track_counts"],
                "color_filtered": result["color_role_length_filtered"]["track_counts_including_excluded_minus_one"],
            }
        )
    )


if __name__ == "__main__":
    main()
