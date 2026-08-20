#!/usr/bin/env python3
"""CPU-only smoke of SoccerMaster consuming the current SoccerFactory PKL."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "research/reproduction/smokes/soccerfactory/manifests/dataloader.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def ensure_new_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite data-view path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


def atomic_json(value: Any, path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    started = time.monotonic()
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage")) != (
        1,
        "G10-C",
        "current_pipeline_dataloader_smoke",
    ):
        raise AssertionError("Unexpected manifest identity")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("This smoke requires CUDA_VISIBLE_DEVICES to be empty")

    training_pkl = Path(manifest["training_pkl"])
    source_images = Path(manifest["source_images"])
    data_view = Path(manifest["data_view"])
    result_path = Path(manifest["result_json"])
    if not training_pkl.is_file() or not source_images.is_dir():
        raise FileNotFoundError(f"Missing PKL or image source: {training_pkl}, {source_images}")
    if data_view.exists() or data_view.is_symlink() or result_path.exists():
        raise FileExistsError("Fresh G10-C output paths are required")
    image_names = sorted(path.name for path in source_images.glob("*.jpg"))
    if image_names != [f"{index:06d}.jpg" for index in range(1, 256)]:
        raise AssertionError("Source image sequence is not the fixed contiguous 255-frame sample")
    with training_pkl.open("rb") as handle:
        raw_sequence = pickle.load(handle)
    if list(raw_sequence) != list(range(1, 256)):
        raise AssertionError("Training PKL does not contain frames 1..255")

    print("[PHASE] prepare_isolated_data_view", flush=True)
    dataset_root = data_view / "SN-GSR-2024"
    (dataset_root / "SoccerNetGS/train").mkdir(parents=True, exist_ok=False)
    ensure_new_symlink(dataset_root / "SoccerNetGS/sn500/SNGS-10004/img1", source_images)
    extra_directory = dataset_root / "SoccerNetGS/extracted_info"
    extra_directory.mkdir(parents=True, exist_ok=False)
    ensure_new_symlink(extra_directory / "SNGS-10004.pkl", training_pkl)

    print("[PHASE] import_and_build_real_dataloader", flush=True)
    import numpy as np
    import torch
    from soccermaster.config import load_super_config, yaml_to_dict
    from soccermaster.data.soccernet_gsr_detection import build_gsr_detection_dataloader

    config_path = Path(manifest["config"])
    experiment_config = yaml_to_dict(str(config_path))
    config = load_super_config(experiment_config, str(REPO / "research/configs/baseline/default.yaml"))
    config.update(
        {
            "DATA_ROOT": str(data_view),
            "USE_EXTRA_DATA": True,
            "EXTRA_DATA_PATH": str(dataset_root / "SoccerNetGS/extracted_info.pkl"),
            "EXTRA_DATA_ONLY": True,
            "USE_EXTRA_DATA_AMOUNT": -1,
            "NUM_WORKERS": 0,
            "BATCH_SIZE": 1,
            "AUG_ENABLE_TRAINING_AUGMENTATION": False,
        }
    )
    torch.manual_seed(42)
    np.random.seed(42)
    dataloader = build_gsr_detection_dataloader(config, split="train")
    dataset = dataloader.dataset
    expected = manifest["expected"]
    if len(dataset) != expected["dataset_samples"]:
        raise AssertionError(f"Unexpected dataset length: {len(dataset)}")

    print("[PHASE] consume_one_real_batch", flush=True)
    batch = next(iter(dataloader))
    images = batch["images"]
    annotations = batch["annotations"]
    metas = batch["metas"]
    selected_meta = metas[0]
    expected_shape = (
        expected["batch_size"],
        expected["frames_per_sample"],
        expected["channels"],
        expected["image_size"],
        expected["image_size"],
    )
    clip_annotations = annotations[0]
    selected_start = int(selected_meta["start_frame"])
    selected_end = int(selected_meta["end_frame"])
    raw_people_first_clip = sum(len(raw_sequence[frame_id]["people"]) for frame_id in range(selected_start + 1, selected_end + 1))
    loaded_people_first_clip = sum(int(annotation["id"].numel()) for annotation in clip_annotations)
    annotation_required = {
        "id", "category", "bbox", "visibility", "role", "jersey", "digit_head",
        "digit_tail", "legibility_score", "boxes", "labels", "roles",
        "lines_target", "keypoints_target", "keypoints_mask", "valid_lines", "valid_keypoints",
    }
    assertions = {
        "batch_keys": set(batch) == {"images", "annotations", "metas"},
        "image_shape": tuple(images.shape) == expected_shape,
        "image_dtype": images.dtype == torch.float32,
        "image_finite": bool(torch.isfinite(images).all()),
        "annotation_frames": len(clip_annotations) == expected["frames_per_sample"],
        "annotation_fields": all(annotation_required.issubset(annotation) for annotation in clip_annotations),
        "annotation_tensor_alignment": all(
            annotation["id"].shape[0] == annotation["bbox"].shape[0] == annotation["role"].shape[0] == annotation["jersey"].shape[0]
            for annotation in clip_annotations
        ),
        "annotation_finite": all(torch.isfinite(annotation["bbox"]).all() and torch.isfinite(annotation["legibility_score"]).all() for annotation in clip_annotations),
        "people_correspondence": loaded_people_first_clip == raw_people_first_clip,
        "meta_sequence": selected_meta["sequence"] == "SNGS-10004",
        "meta_frame_range": selected_start in range(0, 211, 30) and selected_end == selected_start + 30,
        "cpu_tensors": images.device.type == "cpu" and all(annotation["bbox"].device.type == "cpu" for annotation in clip_annotations),
    }
    if not all(assertions.values()):
        raise AssertionError(f"G10-C DataLoader assertions failed: {assertions}")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True).stdout.splitlines()
    result = {
        "schema_version": 1,
        "gate": "G10-C",
        "stage": manifest["stage"],
        "status": "passed",
        "conclusion": "current_soccerfactory_pkl_consumed_by_real_soccerMaster_dataloader",
        "manifest": str(manifest_path),
        "git_commit": commit,
        "git_dirty_files": dirty,
        "python": sys.executable,
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "PYTHONPATH": os.environ.get("PYTHONPATH"), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")},
        "config": {"source": str(config_path), "split": "train", "video_mode": True, "num_frames": config["NUM_FRAMES"], "image_size": config["AUG_MAX_SIZE"], "random_augmentation": False, "batch_size": 1, "num_workers": 0},
        "input": {"training_pkl": str(training_pkl), "source_images": str(source_images), "frames": len(raw_sequence), "people": sum(len(frame["people"]) for frame in raw_sequence.values())},
        "dataset": {"samples": len(dataset), "sample_positions": [[str(sequence), int(frame)] for sequence, frame in dataset.sample_position]},
        "batch": {"images_shape": list(images.shape), "images_dtype": str(images.dtype), "images_min": float(images.min()), "images_max": float(images.max()), "annotation_frames": len(clip_annotations), "people_selected_clip": loaded_people_first_clip, "meta": {"sequence": str(selected_meta["sequence"]), "start_frame": selected_start, "end_frame": selected_end, "actual_num_frames": int(selected_meta["actual_num_frames"]), "total_frames": int(selected_meta["total_frames"]), "image_size": [int(value) for value in selected_meta["image_size"]]}},
        "assertions": assertions,
        "data_view": str(data_view),
        "timing": {"wall_seconds": round(time.monotonic() - started, 3), "max_rss_kib": usage.ru_maxrss},
        "gpu_used": False,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "not_validated": manifest["non_goals"],
    }
    atomic_json(result, result_path)
    print(f"[RESULT] passed shape={tuple(images.shape)} people_first_clip={loaded_people_first_clip} dataset_samples={len(dataset)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
