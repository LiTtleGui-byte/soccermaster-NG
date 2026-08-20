#!/usr/bin/env python3
"""Validate the bounded G8 multi-task protocol.

This is the first, static-only G8 Harness artifact.  It deliberately does not
import SoccerMaster, torch, transformers, accelerate, or any dataset module.
The current authorized scope is manifest/schema/asset validation only; the
multi-GPU training phase must be implemented and approved separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
MANIFEST_PATH = REPO / "reproduction/manifests/g8_multitask.json"
CONFIG_PATH = REPO / (
    "configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
OPS_BUILD = REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
MODEL_PATH = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
CHECKPOINT_DIR = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19"
)
EXPECTED_HEADS = (
    "SoccerNetGSR_Detection",
    "LinesDetection",
    "KeypointsDetection",
    "VideoCaption",
    "CaptionClassification",
)
EXPECTED_CHECKPOINT_FILES = (
    "backbone.pt",
    "text_model/model.safetensors",
    "SoccerNetGSR_Detection.pt",
    "LinesDetection.pt",
    "KeypointsDetection.pt",
    "VideoCaption.pt",
    "CaptionClassification.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("gate") != "G8":
        raise AssertionError("Unexpected G8 manifest identity")
    model = manifest["model"]
    if model.get("ckpt_type") != "soccer_master" or model.get("load_heads") is not True:
        raise AssertionError("G8 must use soccer_master and load_heads=True")
    contract = manifest["training_contract"]
    expected = {
        "world_size": 2,
        "epochs": 2,
        "steps_per_epoch": 4,
        "detection_batch_size": 1,
        "caption_batch_size": 2,
        "num_workers": 0,
        "num_frames": 30,
        "image_size": 512,
        "dtype": "float32",
        "determinism_policy": "enabled_warn_only_for_cuda_ops_without_deterministic_implementation",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise AssertionError(f"Unexpected G8 contract {key}: {contract.get(key)!r}")
    if tuple(contract["heads"]) != EXPECTED_HEADS:
        raise AssertionError("G8 head order changed")
    if tuple(contract["task_order"]) != ("SoccerNetGSR_Detection", "VideoCaption"):
        raise AssertionError("G8 task order changed")
    if "test split" not in manifest["detection"]["validation_role"]:
        raise AssertionError("G8 must explicitly record that no test split is accessed")
    if manifest["detection"]["train_source_split"] != "train":
        raise AssertionError("G8 detection train source changed")
    if manifest["detection"]["valid_source_split"] != "train":
        raise AssertionError("G8 held-out detection source must remain train")
    if "official valid" not in manifest["detection"]["validation_role"]:
        raise AssertionError("G8 must not mislabel held-out train assets as official valid")
    if "test" in json.dumps(manifest["detection"]["train"]).lower():
        raise AssertionError("G8 detection manifest must not mention test")
    expected_train_clips = (
        contract["world_size"]
        * contract["steps_per_epoch"]
        * contract["detection_batch_size"]
    )
    if len(manifest["detection"]["train"]) != expected_train_clips:
        raise AssertionError("G8 requires one unique detection clip per rank and step")
    if len(manifest["detection"]["valid"]) != 4:
        raise AssertionError("G8 requires four held-out detection clips")
    if len(manifest["caption"]["train"]) != 8 or len(manifest["caption"]["valid"]) != 8:
        raise AssertionError("G8 requires eight caption records per split")
    for split in ("train", "valid"):
        detection_ids = [
            (item["sequence"], item["start_frame_zero_based"])
            for item in manifest["detection"][split]
        ]
        if len(detection_ids) != len(set(detection_ids)):
            raise AssertionError(f"Duplicate detection clip in {split}")
        indices = [
            pair[0]
            for item in manifest["caption"][split]
            for pair in item["index_and_bytes"]
        ]
        if len(indices) != len(set(indices)):
            raise AssertionError(f"Duplicate caption JSON index in {split}")
    train_sequences = {item["sequence"] for item in manifest["detection"]["train"]}
    valid_sequences = {item["sequence"] for item in manifest["detection"]["valid"]}
    if train_sequences & valid_sequences:
        raise AssertionError("Detection train/held-out sequence leakage")
    return manifest


def require_paths(manifest: dict[str, Any]) -> None:
    source_gsr = Path(manifest["source_roots"]["gsr"])
    source_caption = Path(manifest["source_roots"]["caption"])
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
        source_gsr,
        source_caption,
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_CHECKPOINT_FILES)
    for split in ("train", "valid"):
        required.append(
            source_caption
            / "video_clip_json"
            / manifest["caption"]["dataset"]
            / f"classification_{split}.json"
        )
    required.append(source_gsr / "legibility_jn" / "train.json")
    for logical_split in ("train", "valid"):
        source_split = manifest["detection"][f"{logical_split}_source_split"]
        for clip in manifest["detection"][logical_split]:
            sequence = clip["sequence"]
            required.extend(
                [
                    source_gsr / "SoccerNetGS" / source_split / sequence / "img1",
                    source_gsr
                    / "SoccerNetGS"
                    / source_split
                    / sequence
                    / "Labels-GameState.json",
                    source_gsr / "camera_params" / source_split / f"{sequence}.json",
                ]
            )
            image_root = source_gsr / "SoccerNetGS" / source_split / sequence / "img1"
            for frame_zero_based in range(
                clip["start_frame_zero_based"], clip["end_frame_exclusive"]
            ):
                required.append(image_root / f"{frame_zero_based + 1:06d}.jpg")
    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        raise FileNotFoundError("Missing G8 assets:\n" + "\n".join(missing))


def validate_caption_assets(manifest: dict[str, Any]) -> None:
    source_caption = Path(manifest["source_roots"]["caption"])
    video_root = source_caption / manifest["caption"]["video_subdirectory"]
    relative_names: dict[str, set[str]] = {}
    for split in ("train", "valid"):
        label_path = (
            source_caption
            / "video_clip_json"
            / manifest["caption"]["dataset"]
            / f"classification_{split}.json"
        )
        if sha256_file(label_path) != manifest["caption"]["label_sha256"][split]:
            raise AssertionError(f"{split} caption label SHA256 changed")
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for class_spec in manifest["caption"][split]:
            for json_index, expected_bytes in class_spec["index_and_bytes"]:
                item = labels[json_index]
                if item.get("caption") != class_spec["label"]:
                    raise AssertionError(f"Caption label changed at {split}[{json_index}]")
                video_path = video_root / item["video"]
                if not video_path.exists() or video_path.stat().st_size != expected_bytes:
                    raise AssertionError(f"Caption video missing or size changed: {video_path}")
                if item["video"] in names:
                    raise AssertionError(f"Duplicate caption video in {split}: {item['video']}")
                names.add(item["video"])
        relative_names[split] = names
    overlap = relative_names["train"] & relative_names["valid"]
    if overlap:
        raise AssertionError(f"Caption train/valid video leakage: {sorted(overlap)[:5]}")


def validate_clip_bounds(manifest: dict[str, Any]) -> None:
    frame_count = manifest["detection"]["frame_count"]
    for split in ("train", "valid"):
        for clip in manifest["detection"][split]:
            if clip["end_frame_exclusive"] - clip["start_frame_zero_based"] != frame_count:
                raise AssertionError(f"Invalid clip length in {split}: {clip}")
            if clip["start_frame_zero_based"] < 0:
                raise AssertionError(f"Negative clip start in {split}: {clip}")


def validate_no_runtime_side_effects() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "import " + "torch",
        "from " + "torch",
        "from " + "models",
        "from " + "data",
        "Data" + "Loader(",
        "optimizer" + ".step(",
        "back" + "ward(",
        "model" + "(",
        "torch" + ".load(",
        "load" + "_checkpoint(",
        "train" + ".py",
    )
    found = [token for token in forbidden if token in source]
    if found:
        raise AssertionError(f"Static-only G8 Harness contains forbidden runtime token(s): {found}")
    write_tokens = (
        "os." + "symlink",
        "shutil." + "copy",
        "Path." + "write_text",
    )
    if any(token in source for token in write_tokens):
        raise AssertionError("Static-only G8 Harness must not copy, link, or write assets")


def main() -> int:
    os.chdir(REPO)
    manifest = load_manifest()
    require_paths(manifest)
    validate_clip_bounds(manifest)
    validate_caption_assets(manifest)
    validate_no_runtime_side_effects()
    identity = git_identity()
    print("[G8_STATIC_PREPARATION] manifest/schema/assets contract valid")
    print(f"[G8_STATIC_PREPARATION] commit={identity['commit']}")
    print(f"[G8_STATIC_PREPARATION] dirty_files={len(identity['dirty_files'])}")
    print("[G8_STATIC_PREPARATION] no project import, GPU, checkpoint load, or training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
