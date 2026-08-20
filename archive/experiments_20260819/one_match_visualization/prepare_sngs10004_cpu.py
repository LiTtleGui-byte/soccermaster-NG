#!/usr/bin/env python3
"""Reproduce SNGS-10004 frames and render reused SoccerFactory evidence.

This is a CPU-only, local-output research stage.  It never imports TrackLab or
model code and never writes to the source video, NAS frame directory, or saved
SoccerFactory archives.
"""

from __future__ import annotations

import json
import argparse
import os
import pickle
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/one_match/20260819_sngs10004_end_to_end"
VISUALS = REPORT / "visuals"
RUNTIME = REPO / ".runtime/one_match/20260819_sngs10004_end_to_end"
RAW_OUT = RUNTIME / "raw_reproduction/SNGS-10004/img1"
CPU_RESULT = RUNTIME / "cpu_result.json"
DATA_VIEW = RUNTIME / "dataloader_view"

RAW_VIDEO = Path(
    "/remote-home/haolinyang/public/sports/SoccerNet/dataset-720p/"
    "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"
)
CAMERA_LABELS = Path(
    "/remote-home/haolinyang/public/sports/SoccerNet/dataset-cameras/"
    "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-cameras.json"
)
MAPPING = Path(
    "/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/sn_2_clip.json"
)
PREPARED = Path(
    "/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/"
    "SoccerNetGS/sn500/SNGS-10004/img1"
)
STEP1 = REPO / ".runtime/g10/sngs10004_step1/run5/states/sn-gamestate.pklz"
ENRICHED = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
REFINED = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
STEP3 = REPO / ".runtime/g10/sngs10004_step3_no_reid/run1/states/sn-gamestate.pklz"
TRAINING_PKL = REPO / ".runtime/g10/sngs10004_current_pipeline_conversion/run1/SNGS-10004.pkl"

START_FRAME = 2278
END_FRAME = 2532
FRAME_COUNT = 255


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def phase(name: str) -> None:
    print(f"[CPU_PHASE] {name}", flush=True)


def write_json_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def load_archive(path: Path) -> tuple[Any, Any]:
    with zipfile.ZipFile(path) as archive:
        detections = pickle.loads(archive.read("10004.pkl"))
        images = pickle.loads(archive.read("10004_image.pkl"))
    return detections, images


def image_row(images: Any, frame_zero: int) -> Any:
    rows = images[images["frame"].astype(int) == frame_zero]
    if len(rows) != 1:
        raise AssertionError(f"frame {frame_zero} matched {len(rows)} image rows")
    return rows.iloc[0]


def detections_for_frame(detections: Any, images: Any, frame_zero: int) -> Any:
    row = image_row(images, frame_zero)
    return detections[detections["image_id"] == row["id"]]


def track_color(track_id: Any) -> tuple[int, int, int]:
    value = int(float(track_id))
    return ((37 * value + 79) % 220 + 25, (83 * value + 41) % 220 + 25, (131 * value + 17) % 220 + 25)


def add_title(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 74), "#101827")
    canvas.paste(image, (0, 74))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), title, fill="white", font=font(24))
    if subtitle:
        draw.text((18, 43), subtitle, fill="#a8c7ff", font=font(15))
    return canvas


def reproduce_frames() -> dict[str, Any]:
    phase("raw_video_to_local_255_frames")
    if RAW_OUT.exists() or RAW_OUT.is_symlink():
        raise FileExistsError(f"Fresh raw output required: {RAW_OUT}")
    RAW_OUT.mkdir(parents=True)
    capture = cv2.VideoCapture(str(RAW_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {RAW_VIDEO}")
    written = []
    try:
        for offset, source_index in enumerate(range(START_FRAME, END_FRAME + 1), start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode source frame {source_index}")
            resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            destination = RAW_OUT / f"{offset:06d}.jpg"
            if not cv2.imwrite(str(destination), resized):
                raise RuntimeError(f"Cannot write {destination}")
            written.append(destination)
            if offset % 64 == 0:
                print(f"[CPU_PROGRESS] extracted={offset}/{FRAME_COUNT}", flush=True)
    finally:
        capture.release()
    if len(written) != FRAME_COUNT:
        raise AssertionError(f"wrote {len(written)} frames")

    phase("all_frame_pixel_correspondence")
    maes: list[float] = []
    exact_fractions: list[float] = []
    byte_equal = 0
    for index in range(1, FRAME_COUNT + 1):
        reproduced_path = RAW_OUT / f"{index:06d}.jpg"
        reference_path = PREPARED / f"{index:06d}.jpg"
        reproduced = cv2.imread(str(reproduced_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reproduced is None or reference is None or reproduced.shape != reference.shape:
            raise AssertionError(f"Unreadable or mismatched frame {index}")
        difference = cv2.absdiff(reproduced, reference)
        maes.append(float(difference.mean()))
        exact_fractions.append(float((difference == 0).mean()))
        if reproduced_path.read_bytes() == reference_path.read_bytes():
            byte_equal += 1
    # The historical JPEGs are not byte-identical, but their decoded pixels
    # should remain within the codec-level difference seen in the lineage check.
    if max(maes) >= 2.0 or float(np.mean(maes)) >= 1.2:
        raise AssertionError(f"all-frame correspondence failed: mean={np.mean(maes)} max={max(maes)}")

    selected = [1, 64, 128, 192, 255]
    tile_w, tile_h = 480, 270
    sheet = Image.new("RGB", (tile_w * len(selected), tile_h * 3 + 105), "#101827")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), "01  Raw video lineage: mapping clip_id=4 -> sequence SNGS-10004", fill="white", font=font(24))
    draw.text((18, 47), "Rows: locally reproduced / prepared reference / amplified pixel difference", fill="#a8c7ff", font=font(16))
    draw.text((18, 75), f"source frames {START_FRAME}-{END_FRAME} inclusive; all {FRAME_COUNT} frames checked", fill="#a8c7ff", font=font(16))
    for column, index in enumerate(selected):
        rep = Image.open(RAW_OUT / f"{index:06d}.jpg").convert("RGB").resize((tile_w, tile_h))
        ref = Image.open(PREPARED / f"{index:06d}.jpg").convert("RGB").resize((tile_w, tile_h))
        diff = np.abs(np.asarray(rep, dtype=np.int16) - np.asarray(ref, dtype=np.int16))
        diff = Image.fromarray(np.clip(diff * 8, 0, 255).astype(np.uint8))
        x = column * tile_w
        sheet.paste(rep, (x, 105))
        sheet.paste(ref, (x, 105 + tile_h))
        sheet.paste(diff, (x, 105 + tile_h * 2))
        draw.text((x + 10, 111), f"prepared #{index} / raw #{START_FRAME + index - 1}", fill="white", font=font(14))
    destination = VISUALS / "01_raw_video_to_prepared_frames.jpg"
    sheet.save(destination, quality=92)
    return {
        "sequence": "SNGS-10004",
        "mapping_clip_id": 4,
        "mapping_output_rule": "SNGS-{clip_id+10000:05d}",
        "source_frames_inclusive": [START_FRAME, END_FRAME],
        "reproduced_frames": FRAME_COUNT,
        "mean_decoded_pixel_mae": float(np.mean(maes)),
        "median_decoded_pixel_mae": float(np.median(maes)),
        "max_frame_decoded_pixel_mae": float(max(maes)),
        "min_exact_channel_fraction": float(min(exact_fractions)),
        "byte_identical_frames": byte_equal,
        "correspondence_rule": "all 255 per-frame decoded-pixel MAE < 2.0 and mean < 1.2",
        "correspondence_passed": True,
        "output_frames": str(RAW_OUT),
        "visual": str(destination),
    }


def step1_stats(step1_det: Any, step1_img: Any, frame_zero: int = 127) -> dict[str, Any]:
    frame_dets = detections_for_frame(step1_det, step1_img, frame_zero)
    embedding_shapes = Counter(str(tuple(np.asarray(v).shape)) for v in step1_det["embeddings"])
    return {
        "detections": int(len(step1_det)),
        "frames": int(len(step1_img)),
        "frames_with_detections": int(step1_det["image_id"].nunique()),
        "unique_track_ids": int(step1_det["track_id"].nunique()),
        "embedding_shapes": dict(embedding_shapes),
        "representative_frame_zero_based": frame_zero,
        "representative_frame_detections": int(len(frame_dets)),
    }


def render_step1(step1_det: Any, step1_img: Any) -> tuple[dict[str, Any], list[str]]:
    phase("render_step1_detection_tracking_reid")
    frame_zero = 127
    frame_dets = detections_for_frame(step1_det, step1_img, frame_zero)
    image = Image.open(PREPARED / f"{frame_zero + 1:06d}.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, row in frame_dets.iterrows():
        x, y, w, h = [float(v) for v in row["bbox_ltwh"]]
        color = track_color(row["track_id"])
        draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
        draw.text((x, max(0, y - 20)), f"track {int(float(row['track_id']))}", fill=color, stroke_fill="black", stroke_width=2, font=font(16))
    image = add_title(image, "02  SoccerFactory Step 1: person detections + StrongSORT track IDs", "Colored boxes are saved Step-1 machine outputs; IDs are not identity ground truth")
    overlay = VISUALS / "02_step1_detection_tracks.jpg"
    image.save(overlay, quality=92)

    counts = step1_det.groupby("track_id").size().sort_values(ascending=False)
    longest = list(counts.head(10).index)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    id_to_frame = {row["id"]: int(row["frame"]) for _, row in step1_img.iterrows()}
    for track_id in longest:
        rows = step1_det[step1_det["track_id"] == track_id].copy()
        frames = np.asarray([id_to_frame[v] for v in rows["image_id"]])
        xs = np.asarray([float(v[0] + v[2] / 2) for v in rows["bbox_ltwh"]])
        order = np.argsort(frames)
        ax.plot(frames[order], xs[order], marker=".", linewidth=1.4, label=f"track {int(float(track_id))} ({len(rows)} dets)")
    ax.set(title="03  Step-1 ReID/track trajectories (10 longest)", xlabel="frame (0-based)", ylabel="bbox center x (pixels)")
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    trajectory = VISUALS / "03_step1_reid_track_trajectories.png"
    fig.savefig(trajectory)
    plt.close(fig)

    stats = step1_stats(step1_det, step1_img, frame_zero)
    return stats, [str(overlay), str(trajectory)]


def render_geometry(enriched_det: Any, enriched_img: Any, reuse_existing_primary: bool = False) -> tuple[dict[str, Any], list[str]]:
    phase("render_pitch_lines_keypoints_camera_coordinates")
    frame_zero = 127
    row = image_row(enriched_img, frame_zero)
    image = Image.open(PREPARED / f"{frame_zero + 1:06d}.jpg").convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    line_point_count = 0
    for name, points in row["lines"].items():
        coords = [(float(p["x"]) * image.width, float(p["y"]) * image.height) for p in points]
        if len(coords) >= 2:
            draw.line(coords, fill=(255, 80, 40, 210), width=5)
            line_point_count += len(coords)
    kp_count = 0
    for key, point in row["keypoints"].items():
        if float(point.get("p", 0.0)) < 0.3:
            continue
        x, y = float(point["x"]), float(point["y"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(20, 255, 230, 230), outline=(0, 0, 0, 255), width=2)
        kp_count += 1
    image = add_title(image, "04  SoccerFactory enrichment: pitch lines + calibration keypoints", "Orange lines and cyan points are saved model outputs, not manual ground truth")
    geometry = VISUALS / "04_factory_pitch_lines_keypoints.jpg"
    if reuse_existing_primary:
        if not geometry.is_file():
            raise FileNotFoundError(f"Expected completed attempt-1 visual {geometry}")
    else:
        image.save(geometry, quality=92)

    frame_dets = detections_for_frame(enriched_det, enriched_img, frame_zero)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    base = cv2.cvtColor(cv2.imread(str(PREPARED / f"{frame_zero + 1:06d}.jpg")), cv2.COLOR_BGR2RGB)
    axes[0].imshow(base)
    for _, det in frame_dets.iterrows():
        x, y, w, h = [float(v) for v in det["bbox_ltwh"]]
        axes[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="yellow", linewidth=1.2))
    axes[0].set_title("camera image and detected people")
    axes[0].axis("off")
    axes[1].set_facecolor("#166534")
    axes[1].plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="white")
    axes[1].axvline(0, color="white", alpha=.7)
    for _, det in frame_dets.iterrows():
        pitch = det["bbox_pitch"]
        team = str(det["team"])
        axes[1].scatter(float(pitch["x_bottom_middle"]), float(pitch["y_bottom_middle"]), c="#60a5fa" if team == "left" else "#f97316", s=45)
    axes[1].set(xlim=(-55, 55), ylim=(-36, 36), aspect="equal", title="camera-derived bottom-middle pitch coordinates")
    fig.suptitle("05  Camera calibration -> player pitch coordinates (structure valid, accuracy unknown)")
    fig.tight_layout()
    camera = VISUALS / "05_factory_camera_pitch_coordinates.png"
    fig.savefig(camera)
    plt.close(fig)
    params = row["parameters"]
    stats = {
        "images_with_keypoints": int(enriched_img["keypoints"].notna().sum()),
        "images_with_lines": int(enriched_img["lines"].notna().sum()),
        "images_with_camera_parameters": int(enriched_img["parameters"].notna().sum()),
        "representative_keypoints_over_threshold": kp_count,
        "representative_line_points": line_point_count,
        "camera_example": {
            "pan_degrees": float(params["pan_degrees"]),
            "tilt_degrees": float(params["tilt_degrees"]),
            "x_focal_length": float(params["x_focal_length"]),
            "position_meters": [float(v) for v in params["position_meters"]],
        },
    }
    return stats, [str(geometry), str(camera)]


def render_attributes(enriched_det: Any, enriched_img: Any) -> tuple[dict[str, Any], list[str]]:
    phase("render_legibility_ocr_role_team")
    per_frame = []
    for frame_zero in range(FRAME_COUNT):
        rows = detections_for_frame(enriched_det, enriched_img, frame_zero)
        readable = int(rows["jersey_number_detection"].notna().sum())
        per_frame.append((readable, len(rows), frame_zero))
    _, _, frame_zero = max(per_frame)
    rows = detections_for_frame(enriched_det, enriched_img, frame_zero)
    image = Image.open(PREPARED / f"{frame_zero + 1:06d}.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, row in rows.iterrows():
        x, y, w, h = [float(v) for v in row["bbox_ltwh"]]
        team = str(row["team"])
        color = (70, 170, 255) if team == "left" else (255, 130, 50)
        draw.rectangle((x, y, x + w, y + h), outline=color, width=4)
        raw_ocr = row["jersey_number_detection"] if row["jersey_number_detection"] is not None else "-"
        final_jn = row["jersey_number"] if row["jersey_number"] is not None else "-"
        label = f"id {int(float(row['track_id']))} {team}/{row['role']} OCR {raw_ocr}->{final_jn} leg {float(row['legibility_score']):.2f}"
        draw.text((x, max(0, y - 20)), label, fill=color, stroke_fill="black", stroke_width=3, font=font(14))
    image = add_title(image, "06  Legibility + OCR + track aggregation + role/team", "Labels are SoccerFactory predictions; non-null fields do not establish semantic correctness")
    destination = VISUALS / "06_factory_ocr_role_team.jpg"
    image.save(destination, quality=92)
    roles = {str(k): int(v) for k, v in enriched_det["role"].value_counts(dropna=False).items()}
    teams = {str(k): int(v) for k, v in enriched_det["team"].value_counts(dropna=False).items()}
    stats = {
        "legibility_score_non_null": int(enriched_det["legibility_score"].notna().sum()),
        "legibility_score_ge_0_5": int((enriched_det["legibility_score"].fillna(-1) >= 0.5).sum()),
        "ocr_detection_non_null": int(enriched_det["jersey_number_detection"].notna().sum()),
        "aggregated_jersey_non_null": int(enriched_det["jersey_number"].notna().sum()),
        "unique_aggregated_jerseys": int(enriched_det["jersey_number"].dropna().astype(str).nunique()),
        "role_distribution": roles,
        "team_distribution": teams,
        "representative_frame_zero_based": frame_zero,
    }
    return stats, [str(destination)]


def xy(row: Any) -> tuple[float, float]:
    value = row["bbox_pitch"]
    return float(value["x_bottom_middle"]), float(value["y_bottom_middle"])


def render_refiner(before: Any, after: Any, images: Any) -> tuple[dict[str, Any], list[str]]:
    phase("render_refiner_before_after")
    if list(before.index) != list(after.index) or len(before) != len(after):
        raise AssertionError("Refiner detection identity changed")
    id_to_frame = {row["id"]: int(row["frame"]) for _, row in images.iterrows()}
    longest = list(before.groupby("track_id").size().sort_values(ascending=False).head(8).index)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharex=True, sharey=True)
    for ax, data, title in [(axes[0], before, "before Refiner"), (axes[1], after, "after coord-only Refiner")]:
        ax.set_facecolor("#166534")
        ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="white")
        ax.axvline(0, color="white", alpha=.6)
        for track_id in longest:
            rows = data[data["track_id"] == track_id].copy()
            frames = np.asarray([id_to_frame[v] for v in rows["image_id"]])
            coords = np.asarray([xy(row) for _, row in rows.iterrows()])
            order = np.argsort(frames)
            ax.plot(coords[order, 0], coords[order, 1], linewidth=1.2, label=f"id {int(float(track_id))}")
        ax.set(xlim=(-55, 55), ylim=(-36, 36), aspect="equal", title=title, xlabel="pitch x (m)", ylabel="pitch y (m)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("07  Refiner before/after: eight longest saved trajectories")
    fig.tight_layout()
    destination = VISUALS / "07_factory_refiner_before_after.png"
    fig.savefig(destination)
    plt.close(fig)
    displacements = []
    for (_, left), (_, right) in zip(before.iterrows(), after.iterrows()):
        a = np.asarray(xy(left))
        b = np.asarray(xy(right))
        displacements.append(float(np.linalg.norm(a - b)))
    stats = {
        "detections_preserved": int(len(after)),
        "changed_coordinates": int(sum(v > 0 for v in displacements)),
        "displacement_median_m": float(np.median(displacements)),
        "displacement_p95_m": float(np.percentile(displacements, 95)),
        "known_temporal_risk": "existing audit found median adjacent step and second-difference jitter increased after Refiner; no 2D coordinate ground truth",
    }
    return stats, [str(destination)]


def render_step3_pkl(step3_det: Any, step3_img: Any) -> tuple[dict[str, Any], list[str]]:
    phase("render_step3_pkl_dataloader")
    conversion = json.loads((REPO / "reports/g10/20260819_current_step3_conversion/result.json").read_text())
    dataloader = json.loads((REPO / "reports/g10/20260819_current_pipeline_dataloader_smoke/result.json").read_text())
    with TRAINING_PKL.open("rb") as handle:
        training = pickle.load(handle)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    axes[0].set_facecolor("#166534")
    axes[0].plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="white")
    for track_id, rows in step3_det.groupby("track_id"):
        coords = np.asarray([xy(row) for _, row in rows.iterrows()])
        team = str(rows.iloc[0]["team"])
        axes[0].plot(coords[:, 0], coords[:, 1], color="#60a5fa" if team == "left" else "#f97316", alpha=.55, linewidth=.9)
    axes[0].set(xlim=(-55, 55), ylim=(-36, 36), aspect="equal", title="no-ReID Step-3 candidate tracks", xlabel="x (m)", ylabel="y (m)")
    axes[1].axis("off")
    lines = [
        "Step 3 -> training PKL -> SoccerMaster DataLoader",
        "",
        f"state: {len(step3_img)} frames / {len(step3_det)} detections / {step3_det['track_id'].nunique()} tracks",
        f"PKL: {len(training)} frames / {sum(len(v['people']) for v in training.values())} people",
        f"valid camera frames: {sum(bool(v['valid_cam_params']) for v in training.values())}",
        f"DataLoader samples: {dataloader['dataset']['samples']}",
        f"first batch: {dataloader['batch']['images_shape']} {dataloader['batch']['images_dtype']}",
        f"first clip people: {dataloader['batch']['people_selected_clip']}",
        "",
        "Confirmed: schema and real loader consumption.",
        "Unknown: semantic correctness of role/team/number/coordinates.",
    ]
    axes[1].text(0.03, .95, "\n".join(lines), va="top", family="monospace", fontsize=12)
    fig.suptitle("08  Final SoccerFactory state and SoccerMaster input contract")
    fig.tight_layout()
    destination = VISUALS / "08_factory_step3_pkl_dataloader.png"
    fig.savefig(destination)
    plt.close(fig)
    stats = {
        "step3_detections": int(len(step3_det)),
        "step3_frames": int(len(step3_img)),
        "step3_tracks": int(step3_det["track_id"].nunique()),
        "step3_roles": {str(k): int(v) for k, v in step3_det["role"].value_counts(dropna=False).items()},
        "step3_teams": {str(k): int(v) for k, v in step3_det["team"].value_counts(dropna=False).items()},
        "training_pkl_frames": int(len(training)),
        "training_pkl_people": int(sum(len(v["people"]) for v in training.values())),
        "valid_camera_frames": int(sum(bool(v["valid_cam_params"]) for v in training.values())),
        "dataloader_result": str(REPO / "reports/g10/20260819_current_pipeline_dataloader_smoke/result.json"),
        "dataloader_samples": int(dataloader["dataset"]["samples"]),
        "dataloader_first_batch_shape": dataloader["batch"]["images_shape"],
    }
    return stats, [str(destination)]


def create_data_view() -> str:
    phase("prepare_local_dataloader_view_for_gpu_worker")
    if DATA_VIEW.exists() or DATA_VIEW.is_symlink():
        raise FileExistsError(f"Fresh data view required: {DATA_VIEW}")
    dataset_root = DATA_VIEW / "SN-GSR-2024"
    (dataset_root / "SoccerNetGS/train").mkdir(parents=True)
    image_link = dataset_root / "SoccerNetGS/sn500/SNGS-10004/img1"
    image_link.parent.mkdir(parents=True)
    image_link.symlink_to(PREPARED, target_is_directory=True)
    extra_dir = dataset_root / "SoccerNetGS/extracted_info"
    extra_dir.mkdir(parents=True)
    (extra_dir / "SNGS-10004.pkl").symlink_to(TRAINING_PKL)
    return str(DATA_VIEW)


def validate_inputs() -> list[str]:
    paths = [RAW_VIDEO, CAMERA_LABELS, MAPPING, PREPARED, STEP1, ENRICHED, REFINED, STEP3, TRAINING_PKL]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    names = sorted(path.name for path in PREPARED.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, FRAME_COUNT + 1)]
    if names != expected:
        raise AssertionError("Prepared frames are not exactly 000001.jpg..000255.jpg")
    mapping = json.loads(MAPPING.read_text())
    matches = [row for row in mapping if int(row[-1]) == 4]
    if matches != [["england_epl", "2014-2015", "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley", 1, START_FRAME, END_FRAME, 4]]:
        raise AssertionError(f"Unexpected clip_id=4 mapping: {matches}")
    return [str(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "continue-after-render-bug"), default="fresh")
    return parser.parse_args()


def continue_after_render_bug() -> int:
    if CPU_RESULT.exists() or DATA_VIEW.exists():
        raise FileExistsError("Continuation requires no CPU result or data view")
    expected_existing = [VISUALS / name for name in (
        "01_raw_video_to_prepared_frames.jpg", "02_step1_detection_tracks.jpg",
        "03_step1_reid_track_trajectories.png", "04_factory_pitch_lines_keypoints.jpg",
    )]
    if not all(path.is_file() for path in expected_existing):
        raise FileNotFoundError("Attempt-1 completed visuals 01-04 are required")
    if len(list(RAW_OUT.glob("*.jpg"))) != FRAME_COUNT:
        raise AssertionError("Attempt-1 local raw reproduction is incomplete")
    if any((VISUALS / name).exists() for name in (
        "05_factory_camera_pitch_coordinates.png", "06_factory_ocr_role_team.jpg",
        "07_factory_refiner_before_after.png", "08_factory_step3_pkl_dataloader.png",
    )):
        raise FileExistsError("Continuation outputs are not fresh")
    started = time.monotonic()
    inputs = validate_inputs()
    step1_det, step1_img = load_archive(STEP1)
    enriched_det, enriched_img = load_archive(ENRICHED)
    refined_det, refined_img = load_archive(REFINED)
    step3_det, step3_img = load_archive(STEP3)
    if not (len(step1_det) == len(enriched_det) == len(refined_det) == len(step3_det) == 3176):
        raise AssertionError("Unexpected detection row count across saved states")
    step1_summary = step1_stats(step1_det, step1_img)
    geometry_stats, geometry_visuals = render_geometry(enriched_det, enriched_img, reuse_existing_primary=True)
    attribute_stats, attribute_visuals = render_attributes(enriched_det, enriched_img)
    refiner_stats, refiner_visuals = render_refiner(enriched_det, refined_det, refined_img)
    step3_stats, step3_visuals = render_step3_pkl(step3_det, step3_img)
    data_view = create_data_view()
    visuals = [str(path) for path in expected_existing[:3]] + geometry_visuals + attribute_visuals + refiner_visuals + step3_visuals
    lineage = {
        "sequence": "SNGS-10004", "mapping_clip_id": 4,
        "mapping_output_rule": "SNGS-{clip_id+10000:05d}",
        "source_frames_inclusive": [START_FRAME, END_FRAME], "reproduced_frames": FRAME_COUNT,
        "correspondence_passed": True,
        "correspondence_evidence": "attempt 1 completed all_frame_pixel_correspondence and entered render_step1 before the later local matplotlib NameError; exact aggregate MAE was not persisted",
        "output_frames": str(RAW_OUT), "visual": str(expected_existing[0]),
    }
    result = {
        "status": "passed", "stage": "cpu_prepare_and_soccerfactory_visualization",
        "sequence": "SNGS-10004", "device": "cpu", "inputs": inputs,
        "continued_after_local_render_bug": True,
        "attempt1_log": str(REPORT / "run.log"),
        "raw_video_lineage": lineage,
        "soccerfactory": {
            "step1_detection_tracking_reid": step1_summary,
            "pitch_lines_keypoints_camera": geometry_stats,
            "legibility_ocr_role_team": attribute_stats,
            "refiner": refiner_stats, "step3_pkl_dataloader": step3_stats,
        },
        "gpu_worker_data_view": data_view, "visuals": visuals,
        "wall_seconds_continuation": round(time.monotonic() - started, 3),
        "training": False,
        "semantic_limit": "saved fields and structure are machine outputs; without independent truth they are not claimed semantically correct",
    }
    write_json_new(CPU_RESULT, result)
    print(f"[CPU_RESULT] passed continuation visuals={len(visuals)} result={CPU_RESULT}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("CPU stage requires CUDA_VISIBLE_DEVICES='' ")
    if args.mode == "continue-after-render-bug":
        return continue_after_render_bug()
    if CPU_RESULT.exists() or VISUALS.exists() or RUNTIME.exists():
        raise FileExistsError("Fresh CPU-stage output paths are required")
    REPORT.mkdir(parents=True, exist_ok=True)
    VISUALS.mkdir(parents=True)
    RUNTIME.mkdir(parents=True)
    started = time.monotonic()
    phase("validate_read_only_inputs")
    inputs = validate_inputs()
    lineage = reproduce_frames()
    step1_det, step1_img = load_archive(STEP1)
    enriched_det, enriched_img = load_archive(ENRICHED)
    refined_det, refined_img = load_archive(REFINED)
    step3_det, step3_img = load_archive(STEP3)
    if not (len(step1_det) == len(enriched_det) == len(refined_det) == len(step3_det) == 3176):
        raise AssertionError("Unexpected detection row count across saved states")
    if not (len(step1_img) == len(enriched_img) == len(refined_img) == len(step3_img) == 255):
        raise AssertionError("Unexpected image row count across saved states")

    step1_stats, step1_visuals = render_step1(step1_det, step1_img)
    geometry_stats, geometry_visuals = render_geometry(enriched_det, enriched_img)
    attribute_stats, attribute_visuals = render_attributes(enriched_det, enriched_img)
    refiner_stats, refiner_visuals = render_refiner(enriched_det, refined_det, refined_img)
    step3_stats, step3_visuals = render_step3_pkl(step3_det, step3_img)
    data_view = create_data_view()
    visuals = [lineage["visual"], *step1_visuals, *geometry_visuals, *attribute_visuals, *refiner_visuals, *step3_visuals]
    result = {
        "status": "passed",
        "stage": "cpu_prepare_and_soccerfactory_visualization",
        "sequence": "SNGS-10004",
        "device": "cpu",
        "new_run": ["raw video extraction", "all-frame comparison", "visual rendering"],
        "reused": [str(STEP1), str(ENRICHED), str(REFINED), str(STEP3), str(TRAINING_PKL)],
        "inputs": inputs,
        "raw_video_lineage": lineage,
        "soccerfactory": {
            "step1_detection_tracking_reid": step1_stats,
            "pitch_lines_keypoints_camera": geometry_stats,
            "legibility_ocr_role_team": attribute_stats,
            "refiner": refiner_stats,
            "step3_pkl_dataloader": step3_stats,
        },
        "gpu_worker_data_view": data_view,
        "visuals": visuals,
        "wall_seconds": round(time.monotonic() - started, 3),
        "training": False,
        "semantic_limit": "saved fields and structure are machine outputs; without independent truth they are not claimed semantically correct",
    }
    write_json_new(CPU_RESULT, result)
    print(f"[CPU_RESULT] passed visuals={len(visuals)} result={CPU_RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
