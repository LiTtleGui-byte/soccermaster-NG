#!/usr/bin/env python3
"""Run the fixed CPU color diagnostic on the predetermined second match."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from diagnose_team_color_features import crop_with_context, font, hue_feature, label_color, torso_crop


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
OUTPUT_DIR = REPO / "reports/g10/20260818_team_color_cross_match_sngs10001"
TOP_CROPS_PER_TRACK = 12
MIN_ROWS_FILTERED = 5
MIN_VALID_CROPS = 3


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    if len(detections) != 11494 or detections.track_id.nunique() != 40 or len(images) != 750:
        raise AssertionError("Predetermined SNGS-10001 archive identity changed")
    images = images.copy()
    images["file_path"] = images.file_path.map(lambda value: str(IMAGE_DIR / Path(str(value)).name))
    missing = [path for path in images.file_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source JPEGs, first={missing[0]}")
    return detections, images


def extract_track_features(
    detections: pd.DataFrame, images: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    path_by_id = dict(zip(images.id, images.file_path))
    cache: dict[Any, Image.Image] = {}
    records: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for track_id, group in detections.groupby("track_id"):
        crop_features = []
        selected_pixels = 0
        for row in group.sort_values("bbox_conf", ascending=False).head(TOP_CROPS_PER_TRACK).itertuples(index=False):
            if row.image_id not in cache:
                cache[row.image_id] = Image.open(Path(path_by_id[row.image_id])).convert("RGB")
            feature, pixels = hue_feature(torso_crop(cache[row.image_id], row.bbox_ltwh))
            if pixels > 0:
                crop_features.append(feature)
                selected_pixels += pixels
        if not crop_features:
            raise AssertionError(f"No valid color crops for track {track_id}")
        feature = np.mean(np.vstack(crop_features), axis=0)
        feature /= max(float(np.linalg.norm(feature)), 1e-12)
        saved_clusters = group.team_cluster.dropna().astype(int).unique().tolist()
        if len(saved_clusters) > 1:
            raise AssertionError(f"Track {track_id} changes saved team cluster")
        role_values = group.role.dropna().astype(str)
        final_role = role_values.iloc[0] if len(role_values) else "unknown"
        role_detection = group.role_detection.dropna().astype(str)
        nonplayer_signals = int((role_detection != "player").sum())
        records.append(
            {
                "track_id": int(float(track_id)),
                "rows": int(len(group)),
                "valid_color_crops": len(crop_features),
                "selected_color_pixels": int(selected_pixels),
                "saved_reid_cluster": saved_clusters[0] if saved_clusters else None,
                "saved_final_role": final_role,
                "nonplayer_role_signals": nonplayer_signals,
                "role_suspect": bool(nonplayer_signals > 0),
            }
        )
        features.append(feature)
    return pd.DataFrame(records), np.vstack(features)


def fit_labels(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float | None]:
    labels = np.full(len(features), -1, dtype=int)
    if int(mask.sum()) < 2:
        return labels, None
    fitted = KMeans(n_clusters=2, random_state=0).fit_predict(features[mask])
    if len(np.unique(fitted)) != 2:
        return labels, None
    labels[mask] = fitted
    return labels, float(silhouette_score(features[mask], fitted))


def draw_contact_sheet(
    detections: pd.DataFrame, images: pd.DataFrame, tracks: pd.DataFrame, label_column: str, title: str
) -> Image.Image:
    reps = detections.loc[detections.groupby("track_id").bbox_conf.idxmax()].copy()
    lookup = tracks.set_index("track_id")
    reps["_label"] = reps.track_id.map(lambda value: int(lookup.loc[int(float(value)), label_column]))
    reps = reps.sort_values(["_label", "track_id"], ascending=[False, True], kind="stable")
    path_by_id = dict(zip(images.id, images.file_path))
    panel_w, panel_h, columns, header_h = 250, 242, 6, 92
    rows_count = math.ceil(len(reps) / columns)
    sheet = Image.new("RGB", (panel_w * columns, header_h + panel_h * rows_count), "#11161c")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), title, fill="white", font=font(27))
    draw.text((16, 50), "Orange=A, blue=B, gray=excluded. Saved C-=no ReID team cluster.", fill="#d9e0e7", font=font(16))
    for position, row in enumerate(reps.itertuples(index=False)):
        image = Image.open(Path(path_by_id[row.image_id])).convert("RGB")
        crop = crop_with_context(image, row.bbox_ltwh)
        crop.thumbnail((panel_w - 14, panel_h - 70), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), "#202832")
        panel.paste(crop, ((panel_w - crop.width) // 2, 4))
        track_id = int(float(row.track_id))
        data = lookup.loc[track_id]
        label = int(data[label_column])
        color = label_color(label)
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((1, 1, panel_w - 2, panel_h - 2), outline=color, width=5)
        panel_draw.rectangle((0, panel_h - 65, panel_w, panel_h), fill="black")
        label_text = "excluded" if label < 0 else ("A" if label == 0 else "B")
        saved = "-" if pd.isna(data.saved_reid_cluster) else str(int(data.saved_reid_cluster))
        panel_draw.text((7, panel_h - 61), f"ID {track_id} | color {label_text} | saved C{saved}", fill=color, font=font(14))
        panel_draw.text((7, panel_h - 40), f"rows {int(data.rows)} | final role {data.saved_final_role}", fill="white", font=font(13))
        panel_draw.text((7, panel_h - 20), f"nonplayer detection signals {int(data.nonplayer_role_signals)}", fill="white", font=font(13))
        sheet.paste(panel, ((position % columns) * panel_w, header_h + (position // columns) * panel_h))
    return sheet


def counts(labels: np.ndarray) -> dict[str, int]:
    values, frequencies = np.unique(labels, return_counts=True)
    return {str(int(value)): int(frequency) for value, frequency in zip(values, frequencies)}


def main() -> None:
    outputs = {
        "result": OUTPUT_DIR / "result.json",
        "all_tracks": OUTPUT_DIR / "color_clusters_all_tracks.jpg",
        "filtered": OUTPUT_DIR / "color_clusters_fixed_filter.jpg",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    detections, images = load_tables()
    tracks, features = extract_track_features(detections, images)
    if len(tracks) != 40 or features.shape != (40, 26):
        raise AssertionError("Unexpected cross-match feature contract")
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
    draw_contact_sheet(detections, images, tracks, "color_cluster_all", "SNGS-10001 upper-body HSV KMeans: all 40 tracks").save(outputs["all_tracks"], quality=92)
    draw_contact_sheet(detections, images, tracks, "color_cluster_filtered", "SNGS-10001 fixed role/length filter").save(outputs["filtered"], quality=92)
    saved_counts = tracks.saved_reid_cluster.map(lambda value: "missing" if pd.isna(value) else str(int(value))).value_counts()
    result = {
        "status": "passed",
        "scope": "predetermined second-match CPU-only diagnostic",
        "selection_policy": "lowest numeric paired video in historical Step-3 archive after excluding SNGS-10004; no result-based reselection",
        "video_id": VIDEO_ID,
        "input_archive_read_only": str(ARCHIVE),
        "source_image_dir_read_only": str(IMAGE_DIR),
        "detections": len(detections),
        "tracks": len(tracks),
        "images": len(images),
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
        "existing_reid": {"track_counts_including_missing": {str(k): int(v) for k, v in saved_counts.items()}},
        "color_all_tracks": {"track_counts": counts(all_labels), "silhouette": all_silhouette},
        "fixed_role_length_filter": {
            "eligible_tracks": int(filtered_mask.sum()),
            "track_counts_including_excluded_minus_one": counts(filtered_labels),
            "silhouette": filtered_silhouette,
            "status": "fit" if filtered_silhouette is not None else "not_fit_insufficient_eligible_tracks",
            "eligibility": f"rows>={MIN_ROWS_FILTERED}, valid_color_crops>={MIN_VALID_CROPS}, no non-player role_detection signal",
        },
        "track_assignments": tracks.to_dict("records"),
        "artifacts": {key: str(path) for key, path in outputs.items() if key != "result"},
        "interpretation_boundary": [
            "No team ground truth is used before manual review.",
            "This historical Step-3 archive has different provenance from the new SNGS-10004 enrichment run.",
            "Missing saved ReID team clusters must be reported as coverage loss, not silently excluded from end-to-end evaluation.",
        ],
    }
    outputs["result"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "reid": result["existing_reid"], "color": result["color_all_tracks"], "filter": result["fixed_role_length_filter"]}))


if __name__ == "__main__":
    main()
