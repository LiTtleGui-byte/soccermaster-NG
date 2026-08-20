#!/usr/bin/env python3
"""Diagnose SoccerFactory team clustering from the existing run2 archive.

CPU-only and read-only with respect to the source archive and source frames.
No model is loaded and no pipeline stage is rerun.
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
OUTPUT_DIR = REPO / "reports/g10/20260818_team_clustering_diagnosis"
VIDEO_ID = "10004"
EXPECTED_ROWS = 3176
EXPECTED_TRACKS = 49


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def cluster_color(cluster: int) -> tuple[int, int, int]:
    return (255, 86, 86) if cluster == 0 else (65, 210, 255)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        with archive.open(f"{VIDEO_ID}.pkl") as handle:
            detections = pd.read_pickle(handle)
        with archive.open(f"{VIDEO_ID}_image.pkl") as handle:
            images = pd.read_pickle(handle)
    if len(detections) != EXPECTED_ROWS or detections["track_id"].nunique() != EXPECTED_TRACKS:
        raise AssertionError("Run2 archive identity changed")
    return detections, images


def track_table(detections: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    for track_id, group in detections[detections["role"] == "player"].groupby("track_id"):
        embedding = np.mean(np.vstack(group["embeddings"].values), axis=0)
        modes = group["jersey_number"].dropna().mode()
        records.append(
            {
                "track_id": int(float(track_id)),
                "rows": int(len(group)),
                "saved_cluster": int(float(group["team_cluster"].iloc[0])),
                "saved_team": str(group["team"].iloc[0]),
                "cluster_nunique": int(group["team_cluster"].nunique()),
                "team_nunique": int(group["team"].nunique()),
                "mean_embedding_norm": float(np.linalg.norm(embedding)),
                "jersey_number": None if modes.empty else int(float(modes.iloc[0])),
            }
        )
        embeddings.append(embedding)
    return pd.DataFrame(records), np.vstack(embeddings)


def representative_rows(detections: pd.DataFrame) -> pd.DataFrame:
    indices = detections.groupby("track_id")["bbox_conf"].idxmax()
    return detections.loc[indices].sort_values(
        ["team_cluster", "track_id"], ascending=[False, True], kind="stable"
    )


def crop_detection(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    pad_x = max(16.0, width * 0.45)
    pad_y = max(12.0, height * 0.16)
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(image.width, int(math.ceil(x + width + pad_x)))
    bottom = min(image.height, int(math.ceil(y + height + pad_y)))
    return image.crop((left, top, right, bottom))


def draw_contact_sheet(detections: pd.DataFrame, images: pd.DataFrame, tracks: pd.DataFrame) -> Image.Image:
    reps = representative_rows(detections)
    path_by_id = dict(zip(images["id"], images["file_path"]))
    rows_by_track = tracks.set_index("track_id")
    panel_w, panel_h = 230, 230
    columns = 7
    rows_count = math.ceil(len(reps) / columns)
    header_h = 86
    sheet = Image.new("RGB", (columns * panel_w, header_h + rows_count * panel_h), "#11161c")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 10), "One highest-confidence crop per track", fill="white", font=font(28))
    draw.text(
        (16, 48),
        "Cluster 1 first. Red=cluster 0, cyan=cluster 1. Labels are candidates, not ground truth.",
        fill="#d9e0e7",
        font=font(18),
    )
    image_cache: dict[str, Image.Image] = {}

    for position, row in enumerate(reps.itertuples(index=False)):
        key = str(row.image_id)
        if key not in image_cache:
            image_cache[key] = Image.open(Path(str(path_by_id[row.image_id]))).convert("RGB")
        crop = crop_detection(image_cache[key], row.bbox_ltwh)
        crop.thumbnail((panel_w - 14, panel_h - 57), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), "#202832")
        panel.paste(crop, ((panel_w - crop.width) // 2, 4))

        cluster = int(float(row.team_cluster))
        track_id = int(float(row.track_id))
        summary = rows_by_track.loc[track_id]
        color = cluster_color(cluster)
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((1, 1, panel_w - 2, panel_h - 2), outline=color, width=5)
        panel_draw.rectangle((0, panel_h - 52, panel_w, panel_h), fill=(0, 0, 0))
        panel_draw.text(
            (7, panel_h - 48),
            f"ID {track_id} | C{cluster} -> {row.team} | rows {int(summary.rows)}",
            fill=color,
            font=font(15),
        )
        jersey = summary.jersey_number
        jersey_text = "?" if pd.isna(jersey) else str(int(jersey))
        panel_draw.text((7, panel_h - 26), f"role {row.role} | jersey #{jersey_text}", fill="white", font=font(14))
        x = (position % columns) * panel_w
        y = header_h + (position // columns) * panel_h
        sheet.paste(panel, (x, y))
    return sheet


def choose_frames(detections: pd.DataFrame, images: pd.DataFrame) -> list[Any]:
    frame_by_id = dict(zip(images["id"], images["frame"].astype(int)))
    counts = detections.groupby("image_id").size()
    choices: list[Any] = []
    for start in range(0, 255, 43):
        stop = min(255, start + 43)
        candidates = [image_id for image_id in counts.index if start <= frame_by_id[image_id] < stop]
        if candidates:
            choices.append(max(candidates, key=lambda image_id: int(counts.loc[image_id])))
    return choices[:6]


def draw_frame_strip(detections: pd.DataFrame, images: pd.DataFrame, image_ids: list[Any]) -> Image.Image:
    path_by_id = dict(zip(images["id"], images["file_path"]))
    frame_by_id = dict(zip(images["id"], images["frame"].astype(int)))
    panel_w, panel_h = 640, 390
    strip = Image.new("RGB", (panel_w * 2, panel_h * 3), "#11161c")
    for position, image_id in enumerate(image_ids):
        image = Image.open(Path(str(path_by_id[image_id]))).convert("RGB")
        rows = detections[detections["image_id"] == image_id]
        scale = min(panel_w / image.width, panel_h / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), "#11161c")
        panel.paste(resized, ((panel_w - resized.width) // 2, (panel_h - resized.height) // 2))
        draw = ImageDraw.Draw(panel)
        offset_x = (panel_w - resized.width) // 2
        offset_y = (panel_h - resized.height) // 2
        for row in rows.itertuples(index=False):
            x, y, width, height = (float(value) * scale for value in row.bbox_ltwh)
            cluster = int(float(row.team_cluster))
            color = cluster_color(cluster)
            draw.rectangle(
                (offset_x + x, offset_y + y, offset_x + x + width, offset_y + y + height),
                outline=color,
                width=3,
            )
            draw.text(
                (offset_x + x, max(22, offset_y + y - 17)),
                f"{int(float(row.track_id))}:C{cluster}",
                fill=color,
                font=font(13),
                stroke_width=2,
                stroke_fill="black",
            )
        draw.rectangle((0, 0, panel_w, 24), fill="black")
        draw.text(
            (8, 3),
            f"file frame {frame_by_id[image_id] + 1:03d} | C0={int((rows.team_cluster == 0).sum())}, C1={int((rows.team_cluster == 1).sum())}",
            fill="white",
            font=font(15),
        )
        strip.paste(panel, ((position % 2) * panel_w, (position // 2) * panel_h))
    return strip


def draw_pca(tracks: pd.DataFrame, embeddings: np.ndarray) -> Image.Image:
    coords = PCA(n_components=2).fit_transform(embeddings)
    width, height, margin = 1100, 760, 90
    canvas = Image.new("RGB", (width, height), "#151b22")
    draw = ImageDraw.Draw(canvas)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-9)
    y_span = max(float(y_max - y_min), 1e-9)
    draw.text((20, 15), "PCA of 49 mean track ReID embeddings", fill="white", font=font(27))
    draw.text((20, 52), "Saved KMeans labels: red=C0 (46 tracks), cyan=C1 (3 tracks)", fill="#d9e0e7", font=font(18))
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#79838e", width=2)
    for row, (x_value, y_value) in zip(tracks.itertuples(index=False), coords):
        px = margin + (float(x_value) - x_min) / x_span * (width - 2 * margin)
        py = height - margin - (float(y_value) - y_min) / y_span * (height - 2 * margin)
        color = cluster_color(int(row.saved_cluster))
        radius = 6 + min(8, math.sqrt(row.rows))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="black", width=2)
        draw.text((px + radius + 2, py - 9), str(row.track_id), fill="white", font=font(14))
    return canvas


def main() -> None:
    outputs = {
        "result": OUTPUT_DIR / "result.json",
        "contact_sheet": OUTPUT_DIR / "track_crop_contact_sheet.jpg",
        "frame_strip": OUTPUT_DIR / "six_frame_cluster_overlay.jpg",
        "pca": OUTPUT_DIR / "track_embedding_pca.jpg",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    detections, images = load_tables()
    tracks, embeddings = track_table(detections)
    saved = tracks["saved_cluster"].to_numpy(dtype=int)
    rerun = KMeans(n_clusters=2, random_state=0).fit_predict(embeddings)
    direct_agreement = float(np.mean(rerun == saved))
    inverse_agreement = float(np.mean((1 - rerun) == saved))
    best_agreement = max(direct_agreement, inverse_agreement)
    if best_agreement != 1.0:
        raise AssertionError("CPU KMeans did not reproduce the saved partition")

    mapping = pd.crosstab(detections["team_cluster"], detections["team"])
    if int(mapping.loc[0.0, "left"]) != 3119 or int(mapping.loc[1.0, "right"]) != 57:
        raise AssertionError("Saved cluster-to-team mapping changed")
    mapping_changed_rows = int(
        (~(((detections.team_cluster == 0) & (detections.team == "left")) |
           ((detections.team_cluster == 1) & (detections.team == "right")))).sum()
    )

    chosen = choose_frames(detections, images)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_contact_sheet(detections, images, tracks).save(outputs["contact_sheet"], quality=92)
    draw_frame_strip(detections, images, chosen).save(outputs["frame_strip"], quality=92)
    draw_pca(tracks, embeddings).save(outputs["pca"], quality=92)

    cluster_track_counts = tracks["saved_cluster"].value_counts().sort_index()
    cluster_row_counts = detections["team_cluster"].value_counts().sort_index()
    norm_summary = tracks.groupby("saved_cluster")["mean_embedding_norm"].agg(["min", "mean", "max"])
    result = {
        "status": "passed",
        "scope": "CPU-only diagnosis of existing pre-Refiner team outputs",
        "input_archive": str(ARCHIVE),
        "model_or_pipeline_rerun": False,
        "gpu_used": False,
        "counts": {
            "frames": int(len(images)),
            "detections": int(len(detections)),
            "tracks": int(len(tracks)),
            "cluster_detection_rows": {str(int(key)): int(value) for key, value in cluster_row_counts.items()},
            "cluster_tracks": {str(int(key)): int(value) for key, value in cluster_track_counts.items()},
            "final_team_rows": {str(key): int(value) for key, value in detections.team.value_counts().items()},
        },
        "cluster_to_final_team": {
            "mapping": {"0": "left", "1": "right"},
            "mapping_changed_rows": mapping_changed_rows,
            "track_cluster_nunique_above_one": int((tracks.cluster_nunique > 1).sum()),
            "track_team_nunique_above_one": int((tracks.team_nunique > 1).sum()),
        },
        "kmeans_reproduction": {
            "track_mean_embedding_shape": list(embeddings.shape),
            "random_state": 0,
            "best_label_permutation_agreement": best_agreement,
            "euclidean_silhouette": float(silhouette_score(embeddings, rerun)),
            "saved_cluster_1_tracks": tracks.loc[tracks.saved_cluster == 1, ["track_id", "rows"]].to_dict("records"),
            "mean_embedding_norm_by_cluster": {
                str(int(index)): {key: float(value) for key, value in row.items()}
                for index, row in norm_summary.iterrows()
            },
        },
        "selected_visual_frames_one_based": [
            int(images.loc[images.id == image_id, "frame"].iloc[0]) + 1 for image_id in chosen
        ],
        "confirmed": [
            "The 3119/57 imbalance already exists in team_cluster.",
            "The final team field is an exact 0->left and 1->right renaming for all rows.",
            "The saved 46/3 track partition is exactly reproduced from mean track ReID embeddings by KMeans(n_clusters=2, random_state=0).",
        ],
        "inference": "The observed collapse originates at or before track-level KMeans clustering, not in the later left/right renaming step.",
        "unknown": [
            "True per-person team labels and formal team accuracy.",
            "Whether ReID embeddings, track fragmentation, crop quality, or KMeans assumptions are the dominant upstream cause.",
        ],
        "artifacts": {key: str(path) for key, path in outputs.items() if key != "result"},
    }
    outputs["result"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing output: {path}")
    print(json.dumps({"status": "passed", "output_dir": str(OUTPUT_DIR), "cluster_tracks": result["counts"]["cluster_tracks"]}))


if __name__ == "__main__":
    main()
