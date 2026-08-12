#!/usr/bin/env python3
"""Run one fixed CaptionClassification tiny-overfit on one visible GPU."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable


# Apply offline and deterministic-environment settings before framework imports.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO = Path("/home/tianlin/SoccerMaster")
OPS_BUILD = (
    REPO
    / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
)
CONFIG_PATH = REPO / (
    "configs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
MANIFEST_PATH = REPO / "reproduction/manifests/g6_tiny_overfit.json"
MODEL_PATH = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
CHECKPOINT_DIR = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/"
    "epoch_19"
)
REPORTS = REPO / "reports/g6"
RESULT_PATH = REPORTS / "g6_tiny_overfit_result_20260812.json"

DEVICE = "cuda:0"
CONFIG_DEVICE = "cpu"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
VISIBLE_PHYSICAL_GPU = "7"
SEED = 42
NUM_FRAMES = 30
IMAGE_SIZE = 512
HEARTBEAT_SECONDS = 30
GIB = 1024**3
MIN_FREE_BEFORE_LOAD_GIB = 30
MIN_FREE_BEFORE_TRAIN_GIB = 20

EXPECTED_FILES = (
    "backbone.pt",
    "CaptionClassification.pt",
    "text_model/model.safetensors",
)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OPS_BUILD))
os.chdir(REPO)


class PhaseState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = "startup"

    def set(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
        print(f"[PHASE] {phase}", flush=True)

    def get(self) -> str:
        with self._lock:
            return self._phase


def heartbeat(stop: threading.Event, started: float, state: PhaseState) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        elapsed = round(time.monotonic() - started, 1)
        print(
            f"[HEARTBEAT] phase={state.get()} elapsed={elapsed}s",
            flush=True,
        )


def timed(
    name: str,
    state: PhaseState,
    timings: dict[str, float],
    function: Callable[[], Any],
) -> Any:
    state.set(name)
    started = time.monotonic()
    value = function()
    timings[name] = round(time.monotonic() - started, 3)
    print(f"[TIMING] {name}={timings[name]}s", flush=True)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    if manifest.get("schema_version") != 1 or manifest.get("gate") != "G6":
        raise AssertionError("Unexpected G6 manifest identity")
    if manifest.get("task") != "CaptionClassification":
        raise AssertionError("G6 is scoped to CaptionClassification")
    samples = manifest.get("samples", [])
    if len(samples) != 4:
        raise AssertionError(f"Expected four G6 samples, got {len(samples)}")
    targets = [item["target_class_index"] for item in samples]
    if len(set(targets)) != 4:
        raise AssertionError("G6 target classes must be distinct")
    return manifest


def require_paths(manifest: dict[str, Any]) -> None:
    source_root = Path(manifest["source_root"])
    label_path = (
        source_root
        / "video_clip_json"
        / manifest["dataset"]
        / f"classification_{manifest['split']}.json"
    )
    video_root = (
        source_root
        / "video_clip"
        / f"{manifest['dataset']}-high-resolution"
    )
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
        label_path,
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)
    required.extend(video_root / item["relative_video"] for item in manifest["samples"])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing G6 assets:\n" + "\n".join(missing))
    if sha256_file(label_path) != manifest["label_sha256"]:
        raise AssertionError("G6 label file SHA256 changed")
    for item in manifest["samples"]:
        path = video_root / item["relative_video"]
        if path.stat().st_size != item["video_bytes"]:
            raise AssertionError(f"Video size changed: {path}")


def load_config() -> dict[str, Any]:
    from configs.util import load_super_config, yaml_to_dict

    experiment_config = yaml_to_dict(str(CONFIG_PATH))
    config = load_super_config(experiment_config, str(DEFAULT_CONFIG_PATH))
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            "DEVICE": CONFIG_DEVICE,
            "CKPT_TYPE": CKPT_TYPE,
            "LOAD_CHECKPOINTS": False,
            "LOAD_HEADS": LOAD_HEADS,
            "CKPT_PATH": str(MODEL_PATH),
            "TEXT_ENCODER_CKPT_PATH": str(MODEL_PATH),
            "STAGE_1_CKPT_DIR": str(CHECKPOINT_DIR),
            "DATASETS_TO_HEADS": {
                "VideoCaption": ["CaptionClassification"],
            },
        }
    )
    if config["NUM_FRAMES"] != NUM_FRAMES:
        raise AssertionError(f"NUM_FRAMES is {config['NUM_FRAMES']}")
    if config["AUG_RANDOM_RESIZE"] != [IMAGE_SIZE]:
        raise AssertionError(f"Unexpected resize: {config['AUG_RANDOM_RESIZE']}")
    if config["CAPTION_CLASSIFICATION_USE_MLP"]:
        raise AssertionError("G6 expects the checkpoint's single Linear classifier")
    if config["CAPTION_CLASSIFICATION_DROPOUT_RATE"] != 0.0:
        raise AssertionError("G6 expects zero CaptionClassification dropout")
    return config


def prepare_records(
    manifest: dict[str, Any],
    keywords: list[str],
) -> list[dict[str, Any]]:
    source_root = Path(manifest["source_root"])
    label_path = (
        source_root
        / "video_clip_json"
        / manifest["dataset"]
        / f"classification_{manifest['split']}.json"
    )
    video_root = (
        source_root
        / "video_clip"
        / f"{manifest['dataset']}-high-resolution"
    )
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    records = []
    for manifest_index, spec in enumerate(manifest["samples"]):
        item = labels[spec["label_index"]]
        text = item.get("comments_text_anonymized")
        if item.get("caption") != spec["label"]:
            raise AssertionError(f"Caption changed at {spec['label_index']}")
        if item.get("video") != spec["relative_video"]:
            raise AssertionError(f"Video changed at {spec['label_index']}")
        if not isinstance(text, str) or len(text) != spec["text_length"]:
            raise AssertionError(f"Text changed at {spec['label_index']}")
        if keywords[spec["target_class_index"]] != spec["label"]:
            raise AssertionError(f"Class mapping changed for {spec['label']}")
        records.append(
            {
                "manifest_index": manifest_index,
                "caption": spec["label"],
                "caption_index": spec["target_class_index"],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "video": video_root / spec["relative_video"],
                "relative_video": spec["relative_video"],
                "expected_initial_top1": spec["g5_initial_top1"],
            }
        )
    return records


def memory_snapshot(torch: Any, label: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    item = {
        "label": label,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    print(f"[GPU_MEMORY] {json.dumps(item, sort_keys=True)}", flush=True)
    return item


def require_free_memory(
    torch: Any,
    label: str,
    minimum_gib: int,
    snapshots: list[dict[str, Any]],
) -> None:
    item = memory_snapshot(torch, label)
    snapshots.append(item)
    if item["free_bytes"] < minimum_gib * GIB:
        raise RuntimeError(
            f"Only {item['free_bytes'] / GIB:.2f} GiB free at {label}; "
            f"at least {minimum_gib} GiB is required"
        )


def finite_scalar(torch: Any, value: Any, name: str) -> float:
    scalar = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    if not math.isfinite(scalar):
        raise AssertionError(f"Non-finite {name}: {scalar}")
    return scalar


def evaluate_head(
    torch: Any,
    head: Any,
    loss_fn: Any,
    backbone_outputs: dict[str, Any],
    annotations: list[dict[str, Any]],
    metas: tuple[dict[str, Any], ...],
    keywords: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = head(backbone_outputs, metas)
        raw_losses, weights = loss_fn(outputs, annotations)
        loss = raw_losses["classification_loss"] * weights["classification_loss"]
        logits = outputs["logits"]
        labels = torch.tensor(
            [int(item["caption_index"]) for item in annotations],
            device=logits.device,
            dtype=torch.long,
        )
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        if not bool(torch.isfinite(logits).all().item()):
            raise AssertionError("Non-finite classification logits")
        return {
            "loss": finite_scalar(torch, loss, "evaluation loss"),
            "accuracy": finite_scalar(torch, accuracy, "evaluation accuracy"),
            "prediction_indices": [int(value) for value in predictions.cpu()],
            "prediction_labels": [keywords[int(value)] for value in predictions.cpu()],
            "target_indices": [int(value) for value in labels.cpu()],
            "target_labels": [keywords[int(value)] for value in labels.cpu()],
            "maximum_probability": [
                float(value)
                for value in torch.softmax(logits, dim=1).max(dim=1).values.cpu()
            ],
        }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        print(f"[FAILURE] Output already exists: {RESULT_PATH}", flush=True)
        return 1

    started = time.monotonic()
    stop = threading.Event()
    state = PhaseState()
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        args=(stop, started, state),
        daemon=True,
    )
    heartbeat_thread.start()

    results: dict[str, Any] = {
        "gate": "G6",
        "status": "failed",
        "error": None,
        "code": git_identity(),
        "manifest": str(MANIFEST_PATH),
        "environment": {},
        "gpu": {},
        "assets": {},
        "resolved_config": {},
        "training": {},
        "memory_snapshots": [],
        "timings_seconds": {},
        "peak_cpu_rss_kib": None,
        "total_elapsed_seconds": None,
        "artifacts": {"result_json": str(RESULT_PATH)},
        "explicit_non_goals": [
            "no backbone, text encoder, transformer encoder, or other head update",
            "no scheduler, DataLoader, distributed training, or mixed precision",
            "no model, optimizer, or checkpoint save",
            "no validation-set or population-level performance claim",
            "no G7 execution",
        ],
    }
    exit_code = 1
    try:
        manifest = timed(
            "load_manifest",
            state,
            results["timings_seconds"],
            load_manifest,
        )
        results["manifest_sha256"] = sha256_file(MANIFEST_PATH)
        timed(
            "validate_assets",
            state,
            results["timings_seconds"],
            lambda: require_paths(manifest),
        )

        def import_framework() -> tuple[Any, ...]:
            import torch
            from data.video_caption import (
                build_transforms,
                keywords_list,
                read_frames_decord,
            )
            from models.caption_classification import (
                build_caption_classification_loss,
            )
            from models.multi_task import MultiTaskingSigLIP

            return (
                torch,
                build_transforms,
                keywords_list,
                read_frames_decord,
                build_caption_classification_loss,
                MultiTaskingSigLIP,
            )

        (
            torch,
            build_caption_transforms,
            keywords,
            read_frames,
            build_loss,
            model_class,
        ) = timed(
            "import_framework",
            state,
            results["timings_seconds"],
            import_framework,
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE_PHYSICAL_GPU:
            raise RuntimeError(
                "G6 requires external CUDA_VISIBLE_DEVICES=7; got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one visible CUDA device for G6; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(DEVICE)
        results["environment"] = {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "dtype": "torch.float32",
        }
        results["gpu"] = {
            "visible_physical_index": int(VISIBLE_PHYSICAL_GPU),
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
        require_free_memory(
            torch,
            "after_cuda_init",
            MIN_FREE_BEFORE_LOAD_GIB,
            results["memory_snapshots"],
        )

        config = timed(
            "load_config",
            state,
            results["timings_seconds"],
            load_config,
        )
        results["resolved_config"] = config
        records = timed(
            "validate_records",
            state,
            results["timings_seconds"],
            lambda: prepare_records(manifest, keywords),
        )
        results["assets"] = {
            "config": str(CONFIG_PATH),
            "model": str(MODEL_PATH),
            "checkpoint": str(CHECKPOINT_DIR),
            "checkpoint_type": CKPT_TYPE,
            "load_heads": LOAD_HEADS,
            "sample_count": len(records),
            "total_video_bytes": sum(item["video_bytes"] for item in manifest["samples"]),
            "samples": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in record.items()
                    if key != "text"
                }
                for record in records
            ],
        }

        model = timed(
            "construct_model_cpu",
            state,
            results["timings_seconds"],
            lambda: model_class(config=config, logger=None),
        )
        timed(
            "load_checkpoint_cpu",
            state,
            results["timings_seconds"],
            lambda: model.load_checkpoint(
                str(CHECKPOINT_DIR),
                ckpt_type=CKPT_TYPE,
                logger=None,
                load_heads=LOAD_HEADS,
            ),
        )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        head = model.multi_task_head["CaptionClassification"]
        for parameter in head.classifier.parameters():
            parameter.requires_grad_(True)
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        expected_trainable_names = {
            "multi_task_head.CaptionClassification.classifier.weight",
            "multi_task_head.CaptionClassification.classifier.bias",
        }
        trainable_names = {name for name, _ in trainable}
        if trainable_names != expected_trainable_names:
            raise AssertionError(f"Unexpected trainable parameters: {trainable_names}")

        require_free_memory(
            torch,
            "before_model_to_gpu",
            MIN_FREE_BEFORE_LOAD_GIB,
            results["memory_snapshots"],
        )
        torch.cuda.reset_peak_memory_stats(device)

        def move_model() -> None:
            model.to(device)
            model.eval()
            head.train()
            torch.cuda.synchronize(device)

        timed(
            "move_model_to_gpu",
            state,
            results["timings_seconds"],
            move_model,
        )
        results["memory_snapshots"].append(
            memory_snapshot(torch, "after_model_to_gpu")
        )

        transform = build_caption_transforms(config, split="test")

        def cache_features() -> tuple[dict[str, Any], list[dict[str, Any]], tuple[dict[str, Any], ...], list[list[int]]]:
            global_features = []
            annotations = []
            metas_all = []
            frame_indices_all = []
            model.eval()
            for index, record in enumerate(records):
                state.set(f"cache_real_video_{index + 1}_of_{len(records)}")
                frames, frame_indices, _ = read_frames(
                    str(record["video"]),
                    NUM_FRAMES,
                    config["VIDEO_CAPTION_SAMPLE"],
                    config["VIDEO_CAPTION_FIX_START"],
                    config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
                    config["VIDEO_CAPTION_TRIMMED30"],
                )
                annotation = {
                    "caption": record["caption"],
                    "caption_index": torch.tensor(
                        record["caption_index"], dtype=torch.long
                    ),
                    "text": record["text"],
                }
                metas = {"task": "VideoCaption", "video": str(record["video"])}
                images_cpu, annotation, metas = transform(frames, annotation, metas)
                if tuple(images_cpu.shape) != (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
                    raise AssertionError(f"Unexpected input shape: {images_cpu.shape}")
                if not bool(torch.isfinite(images_cpu).all().item()):
                    raise AssertionError("Non-finite G6 input")
                images = images_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model.backbone(images, text=None)
                feature = output["global_features"].detach()
                if tuple(feature.shape) != (1, NUM_FRAMES, 1024):
                    raise AssertionError(f"Unexpected global feature shape: {feature.shape}")
                if not bool(torch.isfinite(feature).all().item()):
                    raise AssertionError("Non-finite cached global feature")
                global_features.append(feature)
                annotations.append(annotation)
                metas_all.append(metas)
                frame_indices_all.append([int(value) for value in frame_indices])
                del frames, images_cpu, images, output
            head.train()
            return (
                {"global_features": torch.cat(global_features, dim=0)},
                annotations,
                tuple(metas_all),
                frame_indices_all,
            )

        (
            backbone_outputs,
            annotations,
            metas,
            frame_indices,
        ) = timed(
            "cache_fixed_backbone_features",
            state,
            results["timings_seconds"],
            cache_features,
        )
        results["assets"]["frame_indices"] = frame_indices
        require_free_memory(
            torch,
            "before_tiny_overfit",
            MIN_FREE_BEFORE_TRAIN_GIB,
            results["memory_snapshots"],
        )
        torch.cuda.reset_peak_memory_stats(device)

        loss_fn = build_loss(config).to(device)
        contract = manifest["training_contract"]
        learning_rate = float(config["LR_CAPTION_CLASSIFICATION_CLASSIFIER"])
        weight_decay = float(config["WEIGHT_DECAY"])
        max_clip_norm = float(config["MAX_CLIP_NORM"])
        max_steps = int(contract["max_steps"])
        evaluation_interval = int(contract["evaluation_interval_steps"])
        required_consecutive = int(contract["early_stop_consecutive_evaluations"])
        success = contract["success"]
        trainable_parameters = [parameter for _, parameter in trainable]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_parameter_ids != {id(parameter) for parameter in trainable_parameters}:
            raise AssertionError("Optimizer parameter scope is incorrect")

        initial_classifier = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainable
        }
        frozen_versions = {
            name: parameter._version
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        initial = evaluate_head(
            torch,
            head,
            loss_fn,
            backbone_outputs,
            annotations,
            metas,
            keywords,
        )
        expected_initial = [item["expected_initial_top1"] for item in records]
        if initial["prediction_labels"] != expected_initial:
            raise AssertionError(
                "G6 initial predictions no longer match G5: "
                f"{initial['prediction_labels']} != {expected_initial}"
            )
        if initial["accuracy"] >= float(success["initial_accuracy_below"]):
            raise AssertionError("G6 samples are already fully fit at initialization")

        curve = [{"step": 0, **initial}]
        nonzero_gradient_steps = 0
        minimum_gradient_norm = math.inf
        maximum_gradient_norm = 0.0
        consecutive_successes = 0
        completed_steps = 0

        def train_loop() -> None:
            nonlocal consecutive_successes
            nonlocal completed_steps
            nonlocal maximum_gradient_norm
            nonlocal minimum_gradient_norm
            nonlocal nonzero_gradient_steps

            for step in range(1, max_steps + 1):
                state.set(f"tiny_overfit_step_{step}_of_{max_steps}")
                optimizer.zero_grad(set_to_none=True)
                outputs = head(backbone_outputs, metas)
                raw_losses, weights = loss_fn(outputs, annotations)
                total_loss = (
                    raw_losses["classification_loss"]
                    * weights["classification_loss"]
                )
                if not bool(torch.isfinite(total_loss).item()):
                    raise AssertionError(f"Non-finite training loss at step {step}")
                total_loss.backward()
                squared_norm = torch.zeros((), device=device)
                for parameter in trainable_parameters:
                    if parameter.grad is not None:
                        if not bool(torch.isfinite(parameter.grad).all().item()):
                            raise AssertionError(f"Non-finite gradient at step {step}")
                        squared_norm += parameter.grad.detach().float().pow(2).sum()
                gradient_norm = float(torch.sqrt(squared_norm).cpu())
                if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                    raise AssertionError(f"Invalid gradient norm at step {step}: {gradient_norm}")
                nonzero_gradient_steps += 1
                minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm)
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                clipped_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=max_clip_norm,
                )
                finite_scalar(torch, clipped_norm, "clipped gradient norm")
                optimizer.step()
                completed_steps = step

                if step == 1 or step % evaluation_interval == 0:
                    evaluation = evaluate_head(
                        torch,
                        head,
                        loss_fn,
                        backbone_outputs,
                        annotations,
                        metas,
                        keywords,
                    )
                    curve.append({"step": step, **evaluation})
                    loss_ratio = evaluation["loss"] / initial["loss"]
                    print(
                        "[TRAIN] "
                        f"step={step} loss={evaluation['loss']:.8f} "
                        f"accuracy={evaluation['accuracy']:.4f} "
                        f"loss_ratio={loss_ratio:.6f} "
                        f"gradient_norm={gradient_norm:.8f}",
                        flush=True,
                    )
                    meets_success = (
                        evaluation["accuracy"] == float(success["final_accuracy"])
                        and evaluation["loss"] <= float(success["maximum_final_loss"])
                        and loss_ratio
                        <= float(success["maximum_final_to_initial_loss_ratio"])
                    )
                    consecutive_successes = (
                        consecutive_successes + 1 if meets_success else 0
                    )
                    if consecutive_successes >= required_consecutive:
                        print(
                            f"[EARLY_STOP] success held for {required_consecutive} evaluations",
                            flush=True,
                        )
                        break

        timed(
            "tiny_overfit_training",
            state,
            results["timings_seconds"],
            train_loop,
        )
        torch.cuda.synchronize(device)
        final = evaluate_head(
            torch,
            head,
            loss_fn,
            backbone_outputs,
            annotations,
            metas,
            keywords,
        )
        final_loss_ratio = final["loss"] / initial["loss"]

        parameter_deltas = {}
        maximum_parameter_delta = 0.0
        for name, parameter in trainable:
            delta = float(
                (parameter.detach().cpu() - initial_classifier[name]).abs().max()
            )
            parameter_deltas[name] = delta
            maximum_parameter_delta = max(maximum_parameter_delta, delta)
        if maximum_parameter_delta <= 0.0:
            raise AssertionError("Trainable classifier parameters did not change")

        final_frozen_versions = {
            name: parameter._version
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        if final_frozen_versions != frozen_versions:
            changed = [
                name
                for name, version in frozen_versions.items()
                if final_frozen_versions.get(name) != version
            ]
            raise AssertionError(f"Frozen parameter versions changed: {changed[:20]}")
        frozen_with_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_gradients:
            raise AssertionError(f"Frozen parameters received gradients: {frozen_with_gradients[:20]}")

        assertions = {
            "initial_accuracy_below_one": initial["accuracy"]
            < float(success["initial_accuracy_below"]),
            "final_accuracy_reached": final["accuracy"]
            == float(success["final_accuracy"]),
            "final_loss_below_threshold": final["loss"]
            <= float(success["maximum_final_loss"]),
            "loss_ratio_below_threshold": final_loss_ratio
            <= float(success["maximum_final_to_initial_loss_ratio"]),
            "nonzero_finite_gradient_each_step": nonzero_gradient_steps
            == completed_steps,
            "trainable_parameters_changed": maximum_parameter_delta > 0.0,
            "frozen_parameters_unchanged": final_frozen_versions == frozen_versions,
            "frozen_parameters_without_gradients": not frozen_with_gradients,
            "early_stop_stability_reached": consecutive_successes
            >= required_consecutive,
        }
        failed_assertions = [name for name, passed in assertions.items() if not passed]
        if failed_assertions:
            raise AssertionError(f"G6 success assertions failed: {failed_assertions}")

        results["training"] = {
            "scope": "CaptionClassification.classifier weight and bias only",
            "batch_size": len(records),
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_max_norm": max_clip_norm,
            "scheduler": None,
            "max_steps": max_steps,
            "completed_steps": completed_steps,
            "evaluation_interval_steps": evaluation_interval,
            "required_consecutive_successful_evaluations": required_consecutive,
            "trainable_parameter_names": sorted(trainable_names),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
            "initial": initial,
            "final": final,
            "final_to_initial_loss_ratio": final_loss_ratio,
            "curve": curve,
            "nonzero_gradient_steps": nonzero_gradient_steps,
            "minimum_gradient_norm": minimum_gradient_norm,
            "maximum_gradient_norm": maximum_gradient_norm,
            "maximum_parameter_delta": maximum_parameter_delta,
            "parameter_maximum_absolute_deltas": parameter_deltas,
            "frozen_parameter_count": len(frozen_versions),
            "assertions": assertions,
            "checkpoint_saved": False,
        }
        results["memory_snapshots"].append(
            memory_snapshot(torch, "after_tiny_overfit")
        )
        results["status"] = "passed"
        exit_code = 0
    except BaseException as error:
        results["error"] = f"{type(error).__name__}: {error}"
        print(f"[FAILURE] {results['error']}", flush=True)
        traceback.print_exc()
    finally:
        state.set("finished")
        results["peak_cpu_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
        stop.set()
        heartbeat_thread.join(timeout=5)
        print("[G6_RESULT_BEGIN]", flush=True)
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
        print("[G6_RESULT_END]", flush=True)
        try:
            with RESULT_PATH.open("x", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, sort_keys=True)
                handle.write("\n")
            print(f"[ARTIFACT] result_json={RESULT_PATH}", flush=True)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        print(f"[EXIT] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
