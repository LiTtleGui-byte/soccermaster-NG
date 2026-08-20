#!/usr/bin/env python3
"""Extract three frozen SoccerMaster representations for the 48-clip pilot.

This is inference-only. It never calls loss, backward, optimizer, scheduler, or
checkpoint save code. Full local patch tensors are spatially pooled on device
and are not written to disk.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
OPS_BUILD = REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
CONFIG_PATH = REPO / (
    "configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
MODEL_PATH = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
CHECKPOINT_DIR = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19"
)
PACKET_DIR = REPO / "reports/commentary_attribute_probe_pilot_48_20260818"
MANIFEST_PATH = PACKET_DIR / "blind_manifest.json"
LABELS_PATH = PACKET_DIR / "video_only_labels.json"
OUTPUT_PATH = PACKET_DIR / "frozen_backbone_features.npz"
SUMMARY_PATH = PACKET_DIR / "feature_extraction_summary.json"
PARTIAL_PATH = PACKET_DIR / "frozen_backbone_features.npz.partial"

SEED = 42
NUM_FRAMES = 30
HIDDEN_DIM = 1024
INPUT_SHAPE = (1, NUM_FRAMES, 3, 512, 512)
MIN_FREE_GIB = 20
HEARTBEAT_SECONDS = 30

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OPS_BUILD))
os.chdir(REPO)


class Phase:
    def __init__(self) -> None:
        self.value = "startup"
        self.lock = threading.Lock()

    def set(self, value: str) -> None:
        with self.lock:
            self.value = value
        print(f"[PHASE] {value}", flush=True)

    def get(self) -> str:
        with self.lock:
            return self.value


def heartbeat(stop: threading.Event, phase: Phase) -> None:
    started = time.monotonic()
    while not stop.wait(HEARTBEAT_SECONDS):
        print(
            f"[HEARTBEAT] phase={phase.get()} elapsed={time.monotonic()-started:.1f}s",
            flush=True,
        )


def load_config() -> dict[str, Any]:
    from soccermaster.config import load_super_config, yaml_to_dict

    config = load_super_config(
        yaml_to_dict(str(CONFIG_PATH)), str(DEFAULT_CONFIG_PATH)
    )
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            "DEVICE": "cpu",
            "CKPT_TYPE": "soccer_master",
            "LOAD_CHECKPOINTS": False,
            "LOAD_HEADS": False,
            "CKPT_PATH": str(MODEL_PATH),
            "TEXT_ENCODER_CKPT_PATH": str(MODEL_PATH),
            "STAGE_1_CKPT_DIR": str(CHECKPOINT_DIR),
            # Construct no downstream heads. The experiment calls only the
            # frozen vision backbone after loading epoch_19.
            "DATASETS_TO_HEADS": {"AttributeProbe": []},
        }
    )
    return config


def require_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR / "backbone.pt",
        CHECKPOINT_DIR / "text_model/model.safetensors",
        MANIFEST_PATH,
        LABELS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    occupied = [
        str(path)
        for path in (OUTPUT_PATH, SUMMARY_PATH, PARTIAL_PATH)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise FileExistsError("Refusing to overwrite:\n" + "\n".join(occupied))

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if manifest.get("sample_count") != 48 or len(manifest.get("items", [])) != 48:
        raise AssertionError("Blind manifest is not the locked 48-clip packet")
    if labels.get("reference_seen") is not False or len(labels.get("items", [])) != 48:
        raise AssertionError("Video-only label contract is not locked")
    manifest_ids = [row["annotation_id"] for row in manifest["items"]]
    label_ids = [row["annotation_id"] for row in labels["items"]]
    if manifest_ids != label_ids or len(set(manifest_ids)) != 48:
        raise AssertionError("Manifest/label identity or ordering mismatch")
    for row in manifest["items"]:
        video = Path(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(f"Missing pilot video: {video}")
    return manifest, labels


def main() -> int:
    phase = Phase()
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(stop, phase), daemon=True)
    thread.start()
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "failed",
        "error": None,
        "seed": SEED,
        "device": None,
        "sample_count": 0,
        "assets": {
            "config": str(CONFIG_PATH),
            "checkpoint": str(CHECKPOINT_DIR),
            "manifest": str(MANIFEST_PATH),
            "labels": str(LABELS_PATH),
        },
        "representations": {},
        "explicit_non_goals": [
            "no model or probe training",
            "no loss, backward, optimizer, or scheduler",
            "no task-head inference",
            "no checkpoint write or overwrite",
            "no full local patch tensor persisted",
        ],
    }
    try:
        phase.set("validate_inputs")
        manifest, _ = require_inputs()

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdigit():
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must name exactly one physical GPU index"
            )

        phase.set("import_framework")
        import numpy as np
        import torch
        from soccermaster.data.video_caption import build_transforms, read_frames_decord
        from soccermaster.models.multi_task import MultiTaskingSigLIP

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Expected one visible CUDA device; available={torch.cuda.is_available()} "
                f"count={torch.cuda.device_count()}"
            )
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < MIN_FREE_GIB * 1024**3:
            raise RuntimeError(
                f"Only {free_bytes/1024**3:.2f} GiB free; require {MIN_FREE_GIB} GiB"
            )
        summary["device"] = {
            "physical_index": int(visible),
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "free_bytes_before_load": int(free_bytes),
            "total_bytes": int(total_bytes),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        }

        phase.set("load_config")
        config = load_config()
        expected = {
            "NUM_FRAMES": NUM_FRAMES,
            "BACKBONE_TYPE": "video",
            "SIGLIP_BACKBONE_TYPE": "unisoccer_part_temporal",
            "BACKBONE_HIDDEN_DIM": HIDDEN_DIM,
            "VIDEO_CAPTION_SAMPLE": "middle",
            "AUG_RANDOM_RESIZE": [512],
            "KEEP_ASPECT_RATIO": False,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise AssertionError(f"Config {key}={config.get(key)!r}, expected {value!r}")
        transform = build_transforms(config, split="test")

        phase.set("construct_model_cpu")
        model = MultiTaskingSigLIP(config=config, logger=None)
        phase.set("load_epoch_19_cpu")
        model.load_checkpoint(
            str(CHECKPOINT_DIR),
            ckpt_type="soccer_master",
            logger=None,
            load_heads=False,
        )
        vision = model.backbone.vision_model
        phase.set("move_frozen_vision_to_gpu")
        for parameter in vision.parameters():
            parameter.requires_grad_(False)
        vision.eval().to(device)
        # Keep the text model and absent task heads off GPU.
        del model
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        annotation_ids: list[str] = []
        global_means: list[Any] = []
        global_sequences: list[Any] = []
        local_late_sequences: list[Any] = []
        frame_indices_all: list[list[int]] = []

        phase.set("extract_48_clips")
        for offset, row in enumerate(manifest["items"]):
            video_path = str(row["video_path"])
            frames, frame_indices, _ = read_frames_decord(
                video_path,
                NUM_FRAMES,
                config["VIDEO_CAPTION_SAMPLE"],
                config["VIDEO_CAPTION_FIX_START"],
                config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
                config["VIDEO_CAPTION_TRIMMED30"],
            )
            images_cpu, _, _ = transform(
                frames,
                {},
                {"task": "VideoCaption", "video": video_path},
            )
            if tuple(images_cpu.shape) != INPUT_SHAPE[1:]:
                raise AssertionError(
                    f"Unexpected input shape for {row['annotation_id']}: "
                    f"{tuple(images_cpu.shape)}"
                )
            images = images_cpu.unsqueeze(0).to(device, non_blocking=False)
            with torch.inference_mode():
                _, local_late, _, global_sequence = vision(images, None, None)
                local_late_sequence = local_late.mean(dim=2)
                global_mean = global_sequence.mean(dim=1)
            expected_sequence = (1, NUM_FRAMES, HIDDEN_DIM)
            if tuple(global_sequence.shape) != expected_sequence:
                raise AssertionError(
                    f"global sequence {tuple(global_sequence.shape)} != {expected_sequence}"
                )
            if tuple(local_late_sequence.shape) != expected_sequence:
                raise AssertionError(
                    f"local-late pooled sequence {tuple(local_late_sequence.shape)} "
                    f"!= {expected_sequence}"
                )
            for name, tensor in (
                ("global_mean", global_mean),
                ("global_sequence", global_sequence),
                ("local_late_sequence", local_late_sequence),
            ):
                if not bool(torch.isfinite(tensor).all().item()):
                    raise AssertionError(f"Non-finite {name} for {row['annotation_id']}")
            annotation_ids.append(row["annotation_id"])
            global_means.append(global_mean[0].float().cpu().numpy())
            global_sequences.append(global_sequence[0].float().cpu().numpy())
            local_late_sequences.append(local_late_sequence[0].float().cpu().numpy())
            frame_indices_all.append([int(index) for index in frame_indices])
            del frames, images_cpu, images, local_late, local_late_sequence
            del global_sequence, global_mean
            print(
                f"[SAMPLE] {offset+1}/48 {row['annotation_id']}",
                flush=True,
            )

        phase.set("write_feature_cache")
        arrays = {
            "global_mean": np.stack(global_means).astype(np.float32),
            "global_sequence": np.stack(global_sequences).astype(np.float32),
            "local_late_sequence": np.stack(local_late_sequences).astype(np.float32),
        }
        metadata = {
            "schema_version": "attribute_probe_frozen_features_v1",
            "annotation_ids": annotation_ids,
            "frame_indices": frame_indices_all,
            "checkpoint": str(CHECKPOINT_DIR),
            "sampling": "30 uniform middle frames over each clip",
            "local_pooling": "mean over spatial patch dimension on GPU",
        }
        with PARTIAL_PATH.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(PARTIAL_PATH, OUTPUT_PATH)
        summary["status"] = "completed"
        summary["sample_count"] = len(annotation_ids)
        summary["representations"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        }
        summary["peak_gpu_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        phase.set("completed")
        return 0
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if not SUMMARY_PATH.exists():
            SUMMARY_PATH.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(summary["traceback"], flush=True)
        return 1
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
