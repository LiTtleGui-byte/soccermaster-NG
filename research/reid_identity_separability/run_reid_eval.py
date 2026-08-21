#!/usr/bin/env python3
"""Evaluate SoccerFactory PRTReID identity separability on SoccerNetGS GT boxes.

The script has two deliberately separate stages:

1. ``extract`` runs the official PRTReID checkpoint on GT person crops and saves
   one compact embedding file per video. It never invokes StrongSORT.
2. ``analyze`` samples reproducible within-video identity pairs, computes the
   distances used by the experiment, and writes metrics, figures, bad cases,
   and a Markdown report.

No training code is imported or executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


LOGGER = logging.getLogger("reid_identity_separability")
PERSON_ROLES = {"player", "goalkeeper"}
PAIR_TYPES = (
    "same_id",
    "different_id_same_team",
    "different_id_different_team",
)


def configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def list_videos(data_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        videos = sorted(set(requested))
    else:
        videos = sorted(
            path.parent.name for path in data_root.glob("SNGS-*/Labels-GameState.json")
        )
    missing = [
        video
        for video in videos
        if not (data_root / video / "Labels-GameState.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Labels-GameState.json for: {missing}")
    return videos


def frame_number(image: dict[str, Any]) -> int:
    filename = Path(str(image["file_name"]))
    try:
        return int(filename.stem) - 1
    except ValueError:
        return int(image.get("frame", image.get("frame_id", -1)))


def load_gt_records(video_dir: Path) -> list[dict[str, Any]]:
    labels_path = video_dir / "Labels-GameState.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    images = {str(image["image_id"]): image for image in payload["images"]}
    records: list[dict[str, Any]] = []

    for annotation in payload["annotations"]:
        attributes = annotation.get("attributes") or {}
        role = attributes.get("role")
        team = attributes.get("team")
        bbox = annotation.get("bbox_image")
        image_id = str(annotation.get("image_id"))
        if (
            role not in PERSON_ROLES
            or team is None
            or "track_id" not in annotation
            or not bbox
            or image_id not in images
        ):
            continue

        image = images[image_id]
        image_path = video_dir / "img1" / str(image["file_name"])
        records.append(
            {
                "video_id": video_dir.name,
                "annotation_id": str(annotation.get("id", "")),
                "image_id": image_id,
                "frame": frame_number(image),
                "image_path": str(image_path),
                "track_id": int(annotation["track_id"]),
                "team": str(team),
                "role": str(role),
                "bbox": [
                    float(bbox["x"]),
                    float(bbox["y"]),
                    float(bbox["w"]),
                    float(bbox["h"]),
                ],
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
            }
        )

    records.sort(key=lambda row: (row["frame"], row["track_id"], row["annotation_id"]))
    return records


def crop_rgb(image_rgb: np.ndarray, bbox: Iterable[float]) -> np.ndarray:
    x, y, width, height = np.asarray(list(bbox), dtype=np.float64)
    image_height, image_width = image_rgb.shape[:2]
    x = max(0.0, min(x, image_width - 2.0))
    y = max(0.0, min(y, image_height - 2.0))
    width = max(1.0, min(width, image_width - 1.0 - x))
    height = max(1.0, min(height, image_height - 1.0 - y))
    left, top, right, bottom = np.rint(
        [x, y, x + width, y + height]
    ).astype(np.int64)
    crop = image_rgb[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(
            f"Empty crop after clipping: bbox={list(bbox)}, image={image_rgb.shape}"
        )
    return crop


def build_feature_extractor(
    config_path: Path,
    checkpoint: Path,
    device: str,
    runtime_dir: Path,
):
    import yaml
    import prtreid
    from omegaconf import OmegaConf
    from prtreid.scripts.main import build_config
    from prtreid.tools.feature_extractor import FeatureExtractor
    from sn_gamestate.reid.prtreid_dataset_sim import ReidDataset
    from yacs.config import CfgNode as CN

    text = config_path.read_text(encoding="utf-8")
    replacements = {
        "${data_dir}": str(runtime_dir / "unused_data"),
        "${model_dir}": str(checkpoint.parent.parent),
        "${num_cores}": "4",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    module_config = yaml.safe_load(text)
    # In the full Hydra tree this points five levels up to
    # modules.reid.dataset.masks_mode. Resolve the same value explicitly when
    # loading only the ReID module config for this isolated evaluation.
    dataset_config_path = config_path.parent / "dataset" / "prtreid_dataset.yaml"
    dataset_config = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8"))
    module_config["cfg"]["model"]["bpbreid"]["masks"]["dir"] = dataset_config[
        "masks_mode"
    ]
    reid_cfg = CN(OmegaConf.to_container(OmegaConf.create(module_config["cfg"]), resolve=True))
    reid_cfg.model.load_weights = str(checkpoint)
    reid_cfg.model.bpbreid.hrnet_pretrained_path = str(checkpoint.parent)
    reid_cfg.data.save_dir = str(runtime_dir)
    reid_cfg.project.job_id = os.getpid()
    # The production PRTReId wrapper registers this class before build_config,
    # because compute_parts_num_and_names queries its masks configuration.
    prtreid.data.register_image_dataset("SoccerNet", ReidDataset, "sn")
    reid_cfg = build_config(config=reid_cfg)

    extractor = FeatureExtractor(
        reid_cfg,
        model_path=str(checkpoint),
        device=device,
        image_size=(reid_cfg.data.height, reid_cfg.data.width),
        verbose=False,
    )
    extractor.model.eval()
    return extractor, list(reid_cfg.model.bpbreid.test_embeddings)


def extract_batch(extractor, test_embeddings: list[str], crops: list[np.ndarray]):
    import torch
    from prtreid.utils.tools import extract_test_embeddings

    with torch.no_grad():
        model_output = extractor(crops, external_parts_masks=None)
        embeddings, visibility, _, _, _ = extract_test_embeddings(
            model_output, test_embeddings
        )
    embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
    visibility_np = visibility.detach().cpu().numpy().astype(np.float32)
    if embeddings_np.ndim != 3 or embeddings_np.shape[1:] != (1, 256):
        raise AssertionError(f"Expected embeddings [N,1,256], got {embeddings_np.shape}")
    return embeddings_np[:, 0, :], visibility_np[:, 0]


def save_video_embeddings(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
) -> None:
    if len(records) != len(embeddings) or len(records) != len(visibility):
        raise AssertionError("Metadata and embedding row counts do not match")
    if not np.isfinite(embeddings).all():
        raise AssertionError("Embedding contains NaN or infinity")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        visibility=visibility,
        video_ids=np.asarray([row["video_id"] for row in records]),
        annotation_ids=np.asarray([row["annotation_id"] for row in records]),
        image_ids=np.asarray([row["image_id"] for row in records]),
        frames=np.asarray([row["frame"] for row in records], dtype=np.int32),
        image_paths=np.asarray([row["image_path"] for row in records]),
        track_ids=np.asarray([row["track_id"] for row in records], dtype=np.int32),
        teams=np.asarray([row["team"] for row in records]),
        roles=np.asarray([row["role"] for row in records]),
        bboxes=np.asarray([row["bbox"] for row in records], dtype=np.float32),
    )
    os.replace(temporary, output_path)


def run_extract(args: argparse.Namespace) -> None:
    import torch

    if not args.device.startswith("cuda"):
        raise ValueError("The official extraction stage requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this process")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    videos = list_videos(args.data_root, args.videos)
    videos = [
        video
        for index, video in enumerate(videos)
        if index % args.num_shards == args.shard_index
    ]
    if not videos:
        raise ValueError("This shard contains no videos")

    embeddings_dir = args.output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.output_dir / ".prtreid_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    extractor, test_embeddings = build_feature_extractor(
        args.reid_config, args.checkpoint, args.device, runtime_dir
    )
    if test_embeddings != ["globl"]:
        raise AssertionError(f"Expected official test_embeddings=['globl'], got {test_embeddings}")
    if extractor.model.training:
        raise AssertionError("PRTReID model must be in eval mode")

    start = time.time()
    total_crops = 0
    processed_videos: list[dict[str, Any]] = []
    last_heartbeat = start

    for video_index, video in enumerate(videos, start=1):
        output_path = embeddings_dir / f"{video}.npz"
        if output_path.exists():
            if args.skip_existing:
                LOGGER.info("skip existing video=%s path=%s", video, output_path)
                continue
            raise FileExistsError(output_path)

        records = load_gt_records(args.data_root / video)
        if not records:
            raise RuntimeError(f"No team-labeled player/goalkeeper GT boxes in {video}")
        LOGGER.info(
            "extract video=%s (%d/%d), crops=%d", video, video_index, len(videos), len(records)
        )

        video_embeddings: list[np.ndarray] = []
        video_visibility: list[np.ndarray] = []
        batch_crops: list[np.ndarray] = []
        current_image_path: str | None = None
        current_image_rgb: np.ndarray | None = None

        def flush() -> None:
            nonlocal batch_crops, last_heartbeat
            if not batch_crops:
                return
            embeddings, visibility = extract_batch(extractor, test_embeddings, batch_crops)
            video_embeddings.append(embeddings)
            video_visibility.append(visibility)
            batch_crops = []
            now = time.time()
            if now - last_heartbeat >= 30:
                done = sum(len(item) for item in video_embeddings)
                LOGGER.info(
                    "heartbeat video=%s crops_done=%d/%d elapsed_s=%.1f",
                    video,
                    done,
                    len(records),
                    now - start,
                )
                last_heartbeat = now

        for record in records:
            if record["image_path"] != current_image_path:
                current_image_path = record["image_path"]
                image_bgr = cv2.imread(current_image_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(current_image_path)
                current_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            assert current_image_rgb is not None
            batch_crops.append(crop_rgb(current_image_rgb, record["bbox"]))
            if len(batch_crops) >= args.batch_size:
                flush()
        flush()

        embeddings = np.concatenate(video_embeddings, axis=0)
        visibility = np.concatenate(video_visibility, axis=0)
        save_video_embeddings(records, embeddings, visibility, output_path)
        norms = np.linalg.norm(embeddings, axis=1)
        processed_videos.append(
            {
                "video_id": video,
                "crops": len(records),
                "tracks": len({row["track_id"] for row in records}),
                "embedding_shape": list(embeddings.shape),
                "embedding_norm_mean": float(norms.mean()),
                "embedding_norm_min": float(norms.min()),
                "embedding_norm_max": float(norms.max()),
                "visibility_values": sorted(float(value) for value in np.unique(visibility)),
                "output": str(output_path),
            }
        )
        total_crops += len(records)
        LOGGER.info("saved video=%s path=%s", video, output_path)

    elapsed = time.time() - start
    summary = {
        "status": "completed",
        "stage": "extract",
        "data_root": str(args.data_root),
        "checkpoint": str(args.checkpoint),
        "reid_config": str(args.reid_config),
        "device": args.device,
        "physical_gpu_id": args.physical_gpu_id,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "videos": processed_videos,
        "video_count": len(processed_videos),
        "crop_count": total_crops,
        "elapsed_seconds": elapsed,
        "model_eval": True,
        "torch_no_grad": True,
    }
    summary_path = args.output_dir / f"extract_shard_{args.shard_index:02d}.json"
    atomic_json_dump(summary, summary_path)
    LOGGER.info(
        "completed shard=%d videos=%d crops=%d elapsed_s=%.1f summary=%s",
        args.shard_index,
        len(processed_videos),
        total_crops,
        elapsed,
        summary_path,
    )


def load_embedding_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def weighted_choice(rng: random.Random, entries: list[tuple[Any, int]]) -> Any:
    total = sum(weight for _, weight in entries)
    draw = rng.randrange(total)
    cumulative = 0
    for value, weight in entries:
        cumulative += weight
        if draw < cumulative:
            return value
    raise AssertionError("weighted choice fell through")


def sample_unique_pairs(
    data: dict[str, np.ndarray],
    pair_type: str,
    target_count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    tracks: dict[int, list[int]] = defaultdict(list)
    for index, track_id in enumerate(data["track_ids"].tolist()):
        tracks[int(track_id)].append(index)
    track_team = {
        track_id: str(data["teams"][indices[0]]) for track_id, indices in tracks.items()
    }
    frames = data["frames"]

    catalog: list[tuple[Any, int]] = []
    track_ids = sorted(tracks)
    if pair_type == "same_id":
        for track_id in track_ids:
            indices = tracks[track_id]
            frame_counts: dict[int, int] = defaultdict(int)
            for index in indices:
                frame_counts[int(frames[index])] += 1
            count = len(indices) * (len(indices) - 1) // 2
            count -= sum(value * (value - 1) // 2 for value in frame_counts.values())
            if count > 0:
                catalog.append((track_id, count))
    else:
        for left_pos, left_track in enumerate(track_ids):
            for right_track in track_ids[left_pos + 1 :]:
                same_team = track_team[left_track] == track_team[right_track]
                if pair_type == "different_id_same_team" and not same_team:
                    continue
                if pair_type == "different_id_different_team" and same_team:
                    continue
                count = len(tracks[left_track]) * len(tracks[right_track])
                if count > 0:
                    catalog.append(((left_track, right_track), count))

    available = sum(weight for _, weight in catalog)
    goal = min(target_count, available)
    if goal == 0:
        return []
    selected: set[tuple[int, int]] = set()
    max_attempts = max(1000, goal * 100)
    attempts = 0
    while len(selected) < goal and attempts < max_attempts:
        attempts += 1
        choice = weighted_choice(rng, catalog)
        if pair_type == "same_id":
            left, right = rng.sample(tracks[int(choice)], 2)
            if int(frames[left]) == int(frames[right]):
                continue
        else:
            left_track, right_track = choice
            left = rng.choice(tracks[int(left_track)])
            right = rng.choice(tracks[int(right_track)])
        selected.add((min(left, right), max(left, right)))
    if len(selected) < goal:
        LOGGER.warning(
            "sampled only %d/%d unique pairs for %s after %d attempts",
            len(selected),
            goal,
            pair_type,
            attempts,
        )
    return sorted(selected)


def pair_distances(
    embeddings: np.ndarray, pairs: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pairs:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty
    left = embeddings[np.asarray([pair[0] for pair in pairs])]
    right = embeddings[np.asarray([pair[1] for pair in pairs])]
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    cosine = 1.0 - np.sum(left_norm * right_norm, axis=1)
    euclidean = np.linalg.norm(left - right, axis=1)
    strongsort = np.linalg.norm(left_norm - right_norm, axis=1) / 2.0
    return (
        cosine.astype(np.float32),
        euclidean.astype(np.float32),
        strongsort.astype(np.float32),
    )


def pair_record(
    data: dict[str, np.ndarray],
    pair_type: str,
    left: int,
    right: int,
    cosine: float,
    euclidean: float,
    strongsort: float,
) -> dict[str, Any]:
    bbox_a = data["bboxes"][left].tolist()
    bbox_b = data["bboxes"][right].tolist()
    return {
        "pair_type": pair_type,
        "video_id": str(data["video_ids"][left]),
        "frame_a": int(data["frames"][left]),
        "frame_b": int(data["frames"][right]),
        "image_id_a": str(data["image_ids"][left]),
        "image_id_b": str(data["image_ids"][right]),
        "gt_track_id_a": int(data["track_ids"][left]),
        "gt_track_id_b": int(data["track_ids"][right]),
        "team_a": str(data["teams"][left]),
        "team_b": str(data["teams"][right]),
        "role_a": str(data["roles"][left]),
        "role_b": str(data["roles"][right]),
        "bbox_a": ",".join(f"{value:.3f}" for value in bbox_a),
        "bbox_b": ",".join(f"{value:.3f}" for value in bbox_b),
        "image_path_a": str(data["image_paths"][left]),
        "image_path_b": str(data["image_paths"][right]),
        "cosine_distance": float(cosine),
        "euclidean_distance": float(euclidean),
        "strongsort_distance": float(strongsort),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def binary_metrics(same: np.ndarray, different: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.concatenate(
        [np.ones(len(same), dtype=np.int8), np.zeros(len(different), dtype=np.int8)]
    )
    distances = np.concatenate([same, different])
    scores = -distances
    auc = float(roc_auc_score(labels, scores))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    best_index = int(np.argmax(true_positive_rate - false_positive_rate))
    return {
        "roc_auc": auc,
        "eer": float((false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2),
        "eer_distance_threshold": float(-thresholds[eer_index]),
        "best_youden_distance_threshold": float(-thresholds[best_index]),
        "best_youden_false_positive_rate": float(false_positive_rate[best_index]),
        "best_youden_false_negative_rate": float(false_negative_rate[best_index]),
    }


def plot_distributions(
    values: dict[str, dict[str, np.ndarray]],
    negative_groups: list[str],
    title: str,
    output_path: Path,
    strongsort_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("cosine", "Cosine distance"),
        ("euclidean", "Raw Euclidean distance"),
        ("strongsort", "StrongSORT distance"),
    ]
    labels = {
        "same_id": "Same ID",
        "different_id_same_team": "Different ID / same team",
        "different_id_different_team": "Different ID / different team",
    }
    colors = {
        "same_id": "#2a9d8f",
        "different_id_same_team": "#e76f51",
        "different_id_different_team": "#457b9d",
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    groups = ["same_id", *negative_groups]
    for axis, (metric, metric_title) in zip(axes, metrics):
        combined = np.concatenate([values[group][metric] for group in groups])
        upper = float(np.percentile(combined, 99.5))
        lower = float(np.percentile(combined, 0.5))
        bins = np.linspace(lower, upper, 70)
        for group in groups:
            axis.hist(
                values[group][metric],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=labels[group],
                color=colors[group],
            )
        if metric == "strongsort":
            axis.axvline(
                strongsort_threshold,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"StrongSORT threshold={strongsort_threshold:g}",
            )
        axis.set_title(metric_title)
        axis.set_xlabel("Distance (lower means more similar)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_bbox(value: str) -> list[float]:
    return [float(part) for part in value.split(",")]


def save_bad_case_crops(
    cases: list[dict[str, Any]], output_dir: Path, prefix: str
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for rank, case in enumerate(cases, start=1):
        crops: list[np.ndarray] = []
        crop_paths: list[Path] = []
        for side in ("a", "b"):
            image_bgr = cv2.imread(case[f"image_path_{side}"], cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(case[f"image_path_{side}"])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            crop = crop_rgb(image_rgb, parse_bbox(case[f"bbox_{side}"]))
            crop_path = output_dir / f"{prefix}_{rank:02d}_crop_{side}.jpg"
            cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            crops.append(crop)
            crop_paths.append(crop_path)

        target_height = 300
        resized = []
        for crop in crops:
            width = max(1, int(round(crop.shape[1] * target_height / crop.shape[0])))
            resized.append(cv2.resize(crop, (width, target_height)))
        canvas_width = sum(image.shape[1] for image in resized) + 30
        canvas = Image.new("RGB", (canvas_width, target_height + 90), "white")
        draw = ImageDraw.Draw(canvas)
        x_offset = 10
        for image in resized:
            canvas.paste(Image.fromarray(image), (x_offset, 70))
            x_offset += image.shape[1] + 10
        label = (
            f"{case['video_id']} | frames {case['frame_a']} / {case['frame_b']} | "
            f"GT IDs {case['gt_track_id_a']} / {case['gt_track_id_b']}\n"
            f"teams {case['team_a']} / {case['team_b']} | "
            f"StrongSORT distance={case['strongsort_distance']:.5f}"
        )
        draw.multiline_text((10, 8), label, fill="black", spacing=4)
        pair_path = output_dir / f"{prefix}_{rank:02d}_pair.jpg"
        canvas.save(pair_path, quality=92)
        enriched = dict(case)
        enriched["crop_a"] = str(crop_paths[0])
        enriched["crop_b"] = str(crop_paths[1])
        enriched["pair_image"] = str(pair_path)
        saved.append(enriched)
    return saved


def write_case_csv(cases: list[dict[str, Any]], path: Path) -> None:
    if not cases:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)


def report_markdown(
    metrics: dict[str, Any],
    confusing_different: list[dict[str, Any]],
    largest_same: list[dict[str, Any]],
) -> str:
    groups = metrics["groups"]
    auc_all = metrics["discrimination"]["same_vs_all_different"]["strongsort"]
    auc_team = metrics["discrimination"]["same_vs_same_team_different"]["strongsort"]
    auc_team_value = auc_team["roc_auc"]
    if auc_team_value >= 0.90:
        judgment = (
            "同队不同球员的整体可分性较高；PRTReID 不是当前唯一的首要嫌疑，"
            "下一步应优先把相同 GT 检测送入 StrongSORT，检查 gallery、运动门控和长 gap 重连。"
        )
    elif auc_team_value >= 0.80:
        judgment = (
            "同队不同球员具有一定可分性，但仍存在明显重叠；PRTReID 与 StrongSORT 匹配逻辑都需要继续检查。"
        )
    else:
        judgment = (
            "同队不同球员与 Same-ID 大量重叠；当前 PRTReID feature 很可能是 ID switch 的重要来源之一。"
        )

    def group_row(group: str) -> str:
        item = groups[group]["strongsort"]
        return (
            f"| `{group}` | {item['count']:,} | {item['mean']:.4f} | "
            f"{item['median']:.4f} | {item['p10']:.4f} | {item['p90']:.4f} | "
            f"{item['fraction_at_or_below_matching_threshold']:.2%} |"
        )

    gpu_ids = ", ".join(str(value) for value in metrics["gpu_ids"])

    def case_rows(cases: list[dict[str, Any]]) -> str:
        rows = []
        for rank, case in enumerate(cases[:5], start=1):
            preview = Path(case["pair_image"]).name
            rows.append(
                f"| {rank} | {case['video_id']} | {case['frame_a']} / {case['frame_b']} | "
                f"{case['gt_track_id_a']} / {case['gt_track_id_b']} | "
                f"{case['team_a']} / {case['team_b']} | "
                f"{case['strongsort_distance']:.4f} | "
                f"[预览](bad_cases/{preview}) |"
            )
        return "\n".join(rows)

    return f"""# PRTReID 人物身份可分性评估

## 结论

{judgment}

这里的判断只针对固定 checkpoint 的 embedding 可分性，不代表 StrongSORT tracking 的最终性能。

## 数据与模型

- 数据：`{metrics['data_root']}` 的 SoccerNetGS valid，共 {metrics['video_count']} 个视频。
- 真值：`Labels-GameState.json` 的人工 `track_id`、`bbox_image`、`attributes.team`。
- 过滤：仅保留有 team 的 `player` 和 `goalkeeper`；不把球、裁判或模型 track ID 当作 identity GT。
- 有效 crop：{metrics['crop_count']:,}；视频内身份轨迹：{metrics['track_instance_count']:,}。
- checkpoint：`{metrics['checkpoint']}`。
- embedding：官方 `test_embeddings=["globl"]`，每个 crop 为 256 维。
- GPU：物理 GPU {gpu_ids}；所有 forward 均为 `model.eval()` 和 `torch.no_grad()`。

## Pair 构造

Pair 只在同一视频内构造：

- A：同一个人工 track ID、不同帧；
- B：不同人工 track ID、同一人工 team；
- C：不同人工 track ID、不同人工 team。

每个视频、每类最多采样 {metrics['pair_sampling']['pairs_per_category_per_video']:,} 个不重复 pair，seed={metrics['pair_sampling']['seed']}。以下“最难案例”是**已采样 pair 中**的极值，不声称是全部二次方 pair 空间的精确极值。

## StrongSORT 实际距离

当前正式配置使用 `part_based` matching。因为正式 PRTReID 只输出一个 global part，代码会先对 256 维 embedding 做 L2 normalize，计算 normalized Euclidean distance，再除以 2。匹配门槛为 `{metrics['strongsort']['matching_threshold']}`；大于该值的 appearance match 会被 gate。报告同时给出普通 cosine distance 和未归一化 embedding 的 Euclidean distance。

## StrongSORT 距离分布

| Pair | 数量 | mean | median | p10 | p90 | 距离 ≤ {metrics['strongsort']['matching_threshold']} |
|---|---:|---:|---:|---:|---:|---:|
{group_row('same_id')}
{group_row('different_id_same_team')}
{group_row('different_id_different_team')}

在当前 `max_dist={metrics['strongsort']['matching_threshold']}` 下，Same-ID 的 false negative rate 为 {metrics['configured_threshold']['same_id_false_negative_rate']:.2%}；同队不同人的 appearance false positive rate 为 {metrics['configured_threshold']['same_team_different_id_false_positive_rate']:.2%}，所有不同人的 appearance false positive rate 为 {metrics['configured_threshold']['all_different_id_false_positive_rate']:.2%}。这里的 “false positive” 只表示通过 ReID appearance gate，StrongSORT 后面仍有运动和时空门控，不能把它直接等同于最终 ID switch 率。

## 区分指标

| 比较 | ROC-AUC | EER | EER 距离阈值 | 最佳 Youden FPR | 最佳 Youden FNR |
|---|---:|---:|---:|---:|---:|
| Same-ID vs all Different-ID | {auc_all['roc_auc']:.4f} | {auc_all['eer']:.4f} | {auc_all['eer_distance_threshold']:.4f} | {auc_all['best_youden_false_positive_rate']:.4f} | {auc_all['best_youden_false_negative_rate']:.4f} |
| Same-ID vs Same-Team Different-ID | {auc_team['roc_auc']:.4f} | {auc_team['eer']:.4f} | {auc_team['eer_distance_threshold']:.4f} | {auc_team['best_youden_false_positive_rate']:.4f} | {auc_team['best_youden_false_negative_rate']:.4f} |

ROC-AUC 使用 `-distance` 作为相似度分数，正类为 Same-ID。FPR 表示不同人物被误判为同一人物，FNR 表示同一人物被误判为不同人物。

## 可视化与失败案例

- [Same-ID vs all Different-ID](figures/same_vs_all_different.png)
- [Same-ID vs Same-Team Different-ID](figures/same_vs_same_team_different.png)
- [最容易混淆的 20 个 Different-ID pair](bad_cases/most_confusing_different_id.csv)
- [距离最大的 20 个 Same-ID pair](bad_cases/largest_same_id_distance.csv)
- 完整采样 pair：本地生成产物 `pairs.csv`（约 651 MB，未提交 GitHub；可使用本目录脚本和固定 seed 重建）
- 机器可读指标：[metrics.json](metrics.json)

### 最容易混淆的 Different-ID 代表案例

| 排名 | video | frames | GT IDs | teams | StrongSORT距离 | 图片 |
|---:|---|---|---|---|---:|---|
{case_rows(confusing_different)}

这些案例多为同队、相似球衣且人物 crop 较模糊，说明“同队外观相似”确实会形成低距离负样本。

### 距离最大的 Same-ID 代表案例

| 排名 | video | frames | GT IDs | teams | StrongSORT距离 | 图片 |
|---:|---|---|---|---|---:|---|
{case_rows(largest_same)}

这些案例中可见极小 bbox、严重模糊或 crop 几乎只包含草地，因此高 Same-ID 距离不能全部归因于 embedding；GT bbox 可见性与遮挡是明确混杂因素。

## 解释边界

- 这个实验隔离了 GT bbox 上的 ReID embedding；它没有运行 detector 或 StrongSORT。
- team 只用于定义负样本难度，不作为人物 identity 真值。
- Pair 采样避免单个长轨迹产生不可控的二次方文件，但结果仍可能受视频构图、bbox 尺寸和遮挡分布影响。
- 下一步若要定位 tracker，应固定这些 GT detection/embedding，单独评估 StrongSORT 的关联结果。
"""


def run_analyze(args: argparse.Namespace) -> None:
    videos = list_videos(args.data_root, args.videos)
    embedding_paths = [args.embedding_dir / f"{video}.npz" for video in videos]
    missing = [str(path) for path in embedding_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing embedding files ({len(missing)}): {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    bad_cases_dir = args.output_dir / "bad_cases"
    figures_dir.mkdir(parents=True, exist_ok=True)
    bad_cases_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.csv"
    fieldnames = [
        "pair_type", "video_id", "frame_a", "frame_b", "image_id_a", "image_id_b",
        "gt_track_id_a", "gt_track_id_b", "team_a", "team_b", "role_a", "role_b",
        "bbox_a", "bbox_b", "image_path_a", "image_path_b", "cosine_distance",
        "euclidean_distance", "strongsort_distance",
    ]
    values: dict[str, dict[str, list[np.ndarray]]] = {
        group: {metric: [] for metric in ("cosine", "euclidean", "strongsort")}
        for group in PAIR_TYPES
    }
    confusing_different: list[dict[str, Any]] = []
    largest_same: list[dict[str, Any]] = []
    crop_count = 0
    track_instances = 0
    rng = random.Random(args.seed)
    start = time.time()

    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video_index, path in enumerate(embedding_paths, start=1):
            data = load_embedding_file(path)
            embeddings = data["embeddings"].astype(np.float32, copy=False)
            if embeddings.ndim != 2 or embeddings.shape[1] != 256:
                raise AssertionError(f"Unexpected embedding shape in {path}: {embeddings.shape}")
            crop_count += len(embeddings)
            track_instances += len(np.unique(data["track_ids"]))
            for pair_type in PAIR_TYPES:
                pairs = sample_unique_pairs(
                    data, pair_type, args.pairs_per_category_per_video, rng
                )
                cosine, euclidean, strongsort = pair_distances(embeddings, pairs)
                values[pair_type]["cosine"].append(cosine)
                values[pair_type]["euclidean"].append(euclidean)
                values[pair_type]["strongsort"].append(strongsort)
                records = []
                for pair, cos_value, euc_value, ss_value in zip(
                    pairs, cosine, euclidean, strongsort
                ):
                    record = pair_record(
                        data,
                        pair_type,
                        pair[0],
                        pair[1],
                        float(cos_value),
                        float(euc_value),
                        float(ss_value),
                    )
                    writer.writerow(record)
                    records.append(record)
                if pair_type == "same_id":
                    largest_same.extend(
                        sorted(records, key=lambda row: row["strongsort_distance"], reverse=True)[:30]
                    )
                    largest_same = sorted(
                        largest_same,
                        key=lambda row: row["strongsort_distance"],
                        reverse=True,
                    )[:100]
                else:
                    confusing_different.extend(
                        sorted(records, key=lambda row: row["strongsort_distance"])[:30]
                    )
                    confusing_different = sorted(
                        confusing_different,
                        key=lambda row: row["strongsort_distance"],
                    )[:100]
            if video_index % 5 == 0 or video_index == len(embedding_paths):
                LOGGER.info(
                    "analyze heartbeat videos=%d/%d elapsed_s=%.1f",
                    video_index,
                    len(embedding_paths),
                    time.time() - start,
                )

    arrays: dict[str, dict[str, np.ndarray]] = {
        group: {
            metric: np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.float32)
            for metric, chunks in metrics.items()
        }
        for group, metrics in values.items()
    }
    for group in PAIR_TYPES:
        if not len(arrays[group]["strongsort"]):
            raise RuntimeError(f"No pairs generated for {group}")

    group_metrics = {
        group: {
            metric: distribution_summary(metric_values)
            for metric, metric_values in arrays[group].items()
        }
        for group in PAIR_TYPES
    }
    for group in PAIR_TYPES:
        group_metrics[group]["strongsort"][
            "fraction_at_or_below_matching_threshold"
        ] = float(np.mean(arrays[group]["strongsort"] <= args.strongsort_threshold))
    all_different = {
        metric: np.concatenate(
            [arrays["different_id_same_team"][metric], arrays["different_id_different_team"][metric]]
        )
        for metric in ("cosine", "euclidean", "strongsort")
    }
    discrimination = {
        "same_vs_all_different": {
            metric: binary_metrics(arrays["same_id"][metric], all_different[metric])
            for metric in ("cosine", "euclidean", "strongsort")
        },
        "same_vs_same_team_different": {
            metric: binary_metrics(
                arrays["same_id"][metric], arrays["different_id_same_team"][metric]
            )
            for metric in ("cosine", "euclidean", "strongsort")
        },
    }

    extraction_summaries = []
    for summary_path in sorted(args.embedding_dir.parent.glob("extract_shard_*.json")):
        extraction_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    gpu_ids = sorted(
        {
            summary.get("physical_gpu_id")
            for summary in extraction_summaries
            if summary.get("physical_gpu_id") is not None
        }
    )
    checkpoint = (
        extraction_summaries[0]["checkpoint"]
        if extraction_summaries
        else str(args.checkpoint)
    )
    metrics = {
        "status": "completed",
        "data_root": str(args.data_root),
        "checkpoint": checkpoint,
        "video_count": len(videos),
        "videos": videos,
        "crop_count": crop_count,
        "track_instance_count": track_instances,
        "gpu_ids": gpu_ids,
        "pair_sampling": {
            "within_video_only": True,
            "without_replacement": True,
            "pairs_per_category_per_video": args.pairs_per_category_per_video,
            "seed": args.seed,
        },
        "strongsort": {
            "metric": "L2-normalized Euclidean distance / 2 for one global part",
            "matching_threshold": args.strongsort_threshold,
            "config": "baseline/code/soccerfactory/tracklab/tracklab/configs/modules/track/bpbreid_strong_sort.yaml",
        },
        "groups": group_metrics,
        "discrimination": discrimination,
        "configured_threshold": {
            "threshold": args.strongsort_threshold,
            "same_id_false_negative_rate": float(
                np.mean(arrays["same_id"]["strongsort"] > args.strongsort_threshold)
            ),
            "same_team_different_id_false_positive_rate": float(
                np.mean(
                    arrays["different_id_same_team"]["strongsort"]
                    <= args.strongsort_threshold
                )
            ),
            "all_different_id_false_positive_rate": float(
                np.mean(all_different["strongsort"] <= args.strongsort_threshold)
            ),
        },
        "pairs_csv": str(pairs_path),
        "elapsed_seconds": time.time() - start,
    }
    atomic_json_dump(metrics, args.output_dir / "metrics.json")

    plot_distributions(
        arrays,
        ["different_id_same_team", "different_id_different_team"],
        "Same-ID vs all Different-ID",
        figures_dir / "same_vs_all_different.png",
        args.strongsort_threshold,
    )
    plot_distributions(
        arrays,
        ["different_id_same_team"],
        "Same-ID vs Same-Team Different-ID",
        figures_dir / "same_vs_same_team_different.png",
        args.strongsort_threshold,
    )

    confusing_different = sorted(
        confusing_different, key=lambda row: row["strongsort_distance"]
    )[:20]
    largest_same = sorted(
        largest_same, key=lambda row: row["strongsort_distance"], reverse=True
    )[:20]
    confusing_different = save_bad_case_crops(
        confusing_different, bad_cases_dir, "confusing_different"
    )
    largest_same = save_bad_case_crops(largest_same, bad_cases_dir, "largest_same"
    )
    write_case_csv(
        confusing_different, bad_cases_dir / "most_confusing_different_id.csv"
    )
    write_case_csv(largest_same, bad_cases_dir / "largest_same_id_distance.csv")

    report = report_markdown(metrics, confusing_different, largest_same)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    LOGGER.info(
        "completed analysis videos=%d crops=%d pairs=%d output=%s",
        len(videos),
        crop_count,
        sum(group_metrics[group]["strongsort"]["count"] for group in PAIR_TYPES),
        args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    default_data = Path("/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/valid")
    default_checkpoint = project_root / (
        "assets/checkpoints/official/soccerfactory/pretrained_models/reid/"
        "prtreid-soccernet-baseline.pth.tar"
    )
    default_reid_config = project_root / (
        "baseline/code/soccerfactory/sn-gamestate/sn_gamestate/configs/modules/reid/"
        "prtreid.yaml"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Run inference-only embedding extraction")
    extract.add_argument("--data-root", type=Path, default=default_data)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--videos", nargs="*")
    extract.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    extract.add_argument("--reid-config", type=Path, default=default_reid_config)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--physical-gpu-id", type=int, required=True)
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--skip-existing", action="store_true")
    extract.add_argument("--log-file", type=Path)
    extract.set_defaults(func=run_extract)

    analyze = subparsers.add_parser("analyze", help="Sample pairs and generate results")
    analyze.add_argument("--data-root", type=Path, default=default_data)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--embedding-dir", type=Path, required=True)
    analyze.add_argument("--videos", nargs="*")
    analyze.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    analyze.add_argument("--pairs-per-category-per-video", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260820)
    analyze.add_argument("--strongsort-threshold", type=float, default=0.5)
    analyze.add_argument("--log-file", type=Path)
    analyze.set_defaults(func=run_analyze)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "num_shards", 1) <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "num_shards", 1):
        parser.error("--shard-index must be in [0, num-shards)")
    configure_logging(getattr(args, "log_file", None))
    args.func(args)


if __name__ == "__main__":
    main()
