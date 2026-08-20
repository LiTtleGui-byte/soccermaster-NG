#!/usr/bin/env python3
"""Run the bounded G7 CaptionClassification training and exact-resume probe.

This file is intentionally independent from train.py. It uses fixed train/valid
records, never touches the test split, freezes the vision/text backbone, writes
only local artifacts, and treats exact resume as a machine-checked assertion.
"""

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


# These settings must be applied before importing framework/project modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO = Path("/home/tianlin/SoccerMaster")
OPS_BUILD = REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
CONFIG_PATH = REPO / (
    "configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
MANIFEST_PATH = REPO / "reproduction/manifests/g7_single_task.json"
MODEL_PATH = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
CHECKPOINT_DIR = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19"
)
REPORTS = REPO / "reports/g7/20260812_retry2_full_2epochs"
OUTPUTS = REPO / "outputs"
RESULT_PATH = REPORTS / "result.json"
CHECKPOINT_PATH = OUTPUTS / "g7/20260812_retry2_full_2epochs/step_000003"
CHECKPOINT_TEMP_PATH = OUTPUTS / "g7/20260812_retry2_full_2epochs/.step_000003.tmp"

DEVICE = "cuda:0"
CONFIG_DEVICE = "cpu"
VISIBLE_PHYSICAL_GPU = "7"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
HEARTBEAT_SECONDS = 30
MIN_FREE_GIB = 25
GIB = 1024**3

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
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


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
    if manifest.get("schema_version") != 1 or manifest.get("gate") != "G7":
        raise AssertionError("Unexpected G7 manifest identity")
    if manifest.get("task") != "CaptionClassification":
        raise AssertionError("G7 is restricted to CaptionClassification")
    expected_counts = {"train": (8, 184), "valid": (4, 92)}
    for split, (per_class, total) in expected_counts.items():
        split_spec = manifest["splits"][split]
        classes = split_spec["classes"]
        if len(classes) != 23:
            raise AssertionError(f"{split} does not cover 23 classes")
        if split_spec["samples_per_class"] != per_class:
            raise AssertionError(f"Unexpected {split} samples_per_class")
        pairs = [pair for item in classes for pair in item["index_and_bytes"]]
        if len(pairs) != total or split_spec["total_samples"] != total:
            raise AssertionError(f"Unexpected {split} total")
        indices = [int(pair[0]) for pair in pairs]
        if len(set(indices)) != total:
            raise AssertionError(f"Duplicate JSON indices in {split}")
        for expected_index, item in enumerate(classes):
            if item["class_index"] != expected_index:
                raise AssertionError(f"Class order changed in {split}")
            if len(item["index_and_bytes"]) != per_class:
                raise AssertionError(f"Unbalanced class in {split}: {item['label']}")
    return manifest


def require_paths(manifest: dict[str, Any]) -> None:
    source_root = Path(manifest["source_root"])
    video_root = source_root / manifest["video_subdirectory"]
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
        source_root,
        video_root,
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)
    for split in ("train", "valid"):
        required.append(
            source_root
            / "video_clip_json"
            / manifest["dataset"]
            / f"classification_{split}.json"
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing G7 assets:\n" + "\n".join(missing))
    forbidden = [RESULT_PATH, CHECKPOINT_PATH, CHECKPOINT_TEMP_PATH]
    occupied = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError("Refusing to overwrite G7 artifacts:\n" + "\n".join(occupied))


def prepare_records(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_root = Path(manifest["source_root"])
    video_root = source_root / manifest["video_subdirectory"]
    records: dict[str, list[dict[str, Any]]] = {}
    relative_names: dict[str, set[str]] = {}
    for split in ("train", "valid"):
        label_path = (
            source_root
            / "video_clip_json"
            / manifest["dataset"]
            / f"classification_{split}.json"
        )
        if sha256_file(label_path) != manifest["splits"][split]["label_sha256"]:
            raise AssertionError(f"{split} label file SHA256 changed")
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        split_records = []
        for class_spec in manifest["splits"][split]["classes"]:
            for json_index, expected_bytes in class_spec["index_and_bytes"]:
                item = labels[json_index]
                if item.get("caption") != class_spec["label"]:
                    raise AssertionError(f"Label changed at {split}[{json_index}]")
                path = video_root / item["video"]
                if not path.exists() or path.stat().st_size != expected_bytes:
                    raise AssertionError(f"Video missing or size changed: {path}")
                split_records.append(
                    {
                        "split": split,
                        "class_index": int(class_spec["class_index"]),
                        "label": class_spec["label"],
                        "json_index": int(json_index),
                        "relative_video": item["video"],
                        "video": path,
                        "video_bytes": int(expected_bytes),
                    }
                )
        if len(split_records) != manifest["splits"][split]["total_samples"]:
            raise AssertionError(f"Prepared {split} record count changed")
        records[split] = split_records
        relative_names[split] = {record["relative_video"] for record in split_records}
    overlap = relative_names["train"] & relative_names["valid"]
    if overlap:
        raise AssertionError(f"Train/valid video leakage: {sorted(overlap)[:5]}")
    return records


def load_config(manifest: dict[str, Any]) -> dict[str, Any]:
    from configs.util import load_super_config, yaml_to_dict

    experiment_config = yaml_to_dict(str(CONFIG_PATH))
    config = load_super_config(experiment_config, str(DEFAULT_CONFIG_PATH))
    contract = manifest["training_contract"]
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
            "VIDEO_CAPTION_DATA_ROOT": manifest["source_root"],
            "VIDEO_CAPTION_DATASETS": [manifest["dataset"]],
            "VIDEO_CAPTION_SAMPLE": contract["frame_sampling"],
            "DATASETS_TO_HEADS": {"VideoCaption": ["CaptionClassification"]},
            "FREEZE_VISION_ENCODER": True,
            "FREEZE_TEXT_ENCODER": True,
        }
    )
    assertions = {
        "NUM_FRAMES": contract["num_frames"],
        "BACKBONE_TYPE": "video",
        "SIGLIP_BACKBONE_TYPE": "unisoccer_part_temporal",
        "CAPTION_CLASSIFICATION_USE_TRANSFORMERS": True,
        "CAPTION_CLASSIFICATION_USE_MLP": False,
        "CAPTION_CLASSIFICATION_DROPOUT_RATE": 0.0,
    }
    for key, expected in assertions.items():
        if config.get(key) != expected:
            raise AssertionError(f"Unexpected {key}: {config.get(key)!r}")
    if config["AUG_RANDOM_RESIZE"] != [contract["image_size"]]:
        raise AssertionError("G7 image size does not match high-resolution config")
    return config


def set_global_seed(torch: Any, numpy: Any, seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_seed(base_seed: int, split: str, epoch: int, json_index: int) -> int:
    payload = f"{base_seed}:{split}:{epoch}:{json_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def epoch_order(records: list[dict[str, Any]], seed: int, epoch: int) -> list[int]:
    order = list(range(len(records)))
    random.Random(seed + 1009 * epoch).shuffle(order)
    return order


def batches(order: list[int], batch_size: int) -> list[list[int]]:
    if len(order) % batch_size:
        raise AssertionError("G7 fixed sample count must divide the batch size")
    return [order[index : index + batch_size] for index in range(0, len(order), batch_size)]


def memory_snapshot(torch: Any, label: str) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    item = {
        "label": label,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    print(f"[GPU_MEMORY] {json.dumps(item, sort_keys=True)}", flush=True)
    return item


def make_batch(
    torch: Any,
    numpy: Any,
    records: list[dict[str, Any]],
    indices: list[int],
    split: str,
    epoch: int,
    base_seed: int,
    config: dict[str, Any],
    transform: Any,
    read_frames: Any,
) -> tuple[Any, list[dict[str, Any]], tuple[dict[str, Any], ...], list[str]]:
    images = []
    annotations = []
    metas = []
    ids = []
    for record_index in indices:
        record = records[record_index]
        seed = sample_seed(base_seed, split, epoch, record["json_index"])
        set_global_seed(torch, numpy, seed)
        frames, frame_indices, _ = read_frames(
            str(record["video"]),
            config["NUM_FRAMES"],
            config["VIDEO_CAPTION_SAMPLE"],
            config["VIDEO_CAPTION_FIX_START"],
            config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
            config["VIDEO_CAPTION_TRIMMED30"],
        )
        annotation = {
            "caption": record["label"],
            "caption_index": torch.tensor(record["class_index"], dtype=torch.long),
            "text": None,
        }
        meta = {
            "task": "VideoCaption",
            "video": str(record["video"]),
            "frame_indices": [int(value) for value in frame_indices],
            "manifest_json_index": record["json_index"],
        }
        image, annotation, meta = transform(frames, annotation, meta)
        expected_shape = (config["NUM_FRAMES"], 3, 512, 512)
        if tuple(image.shape) != expected_shape or not bool(torch.isfinite(image).all()):
            raise AssertionError(f"Invalid transformed input: {record['video']}")
        images.append(image)
        annotations.append(annotation)
        metas.append(meta)
        ids.append(f"{split}:{record['json_index']}")
    return torch.stack(images), annotations, tuple(metas), ids


def forward_features(torch: Any, model: Any, images: Any, device: Any) -> dict[str, Any]:
    images = images.to(device)
    model.backbone.eval()
    with torch.no_grad():
        outputs = model.backbone(images, text=None)
    features = outputs["global_features"].detach()
    if not bool(torch.isfinite(features).all()):
        raise AssertionError("Non-finite frozen backbone features")
    return {"global_features": features}


def loss_and_predictions(
    torch: Any,
    head: Any,
    loss_fn: Any,
    features: dict[str, Any],
    annotations: list[dict[str, Any]],
    metas: tuple[dict[str, Any], ...],
) -> tuple[Any, Any, Any]:
    outputs = head(features, metas)
    raw_losses, weights = loss_fn(outputs, annotations)
    loss = raw_losses["classification_loss"] * weights["classification_loss"]
    logits = outputs["logits"]
    labels = torch.tensor(
        [int(item["caption_index"]) for item in annotations],
        device=logits.device,
        dtype=torch.long,
    )
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
        raise AssertionError("Non-finite G7 loss or logits")
    return loss, logits, labels


def evaluate_split(
    torch: Any,
    numpy: Any,
    model: Any,
    head: Any,
    loss_fn: Any,
    records: list[dict[str, Any]],
    split: str,
    epoch: int,
    config: dict[str, Any],
    transform: Any,
    read_frames: Any,
    device: Any,
    batch_size: int,
    base_seed: int,
    state: PhaseState,
) -> dict[str, Any]:
    was_training = head.training
    head.eval()
    losses = []
    predictions = []
    targets = []
    processed_ids = []
    order = list(range(len(records)))
    for batch_number, batch_indices in enumerate(batches(order, batch_size)):
        state.set(f"evaluate_{split}_epoch_{epoch}_batch_{batch_number + 1}_of_{len(order) // batch_size}")
        images, annotations, metas, ids = make_batch(
            torch, numpy, records, batch_indices, split, epoch, base_seed,
            config, transform, read_frames,
        )
        features = forward_features(torch, model, images, device)
        with torch.no_grad():
            loss, logits, labels = loss_and_predictions(
                torch, head, loss_fn, features, annotations, metas
            )
        losses.append(float(loss.detach().cpu()) * len(ids))
        predictions.extend(int(value) for value in logits.argmax(dim=1).cpu())
        targets.extend(int(value) for value in labels.cpu())
        processed_ids.extend(ids)
    if was_training:
        head.train()
    total = len(targets)
    accuracy = sum(p == t for p, t in zip(predictions, targets)) / total
    class_metrics = []
    for class_index in range(23):
        tp = sum(p == class_index and t == class_index for p, t in zip(predictions, targets))
        fp = sum(p == class_index and t != class_index for p, t in zip(predictions, targets))
        fn = sum(p != class_index and t == class_index for p, t in zip(predictions, targets))
        count = sum(t == class_index for t in targets)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics.append(
            {"class_index": class_index, "count": count, "precision": precision, "recall": recall, "f1": f1}
        )
    return {
        "split": split,
        "epoch": epoch,
        "sample_count": total,
        "processed_ids_sha256": hashlib.sha256("\n".join(processed_ids).encode()).hexdigest(),
        "loss": sum(losses) / total,
        "accuracy": accuracy,
        "macro_precision": sum(item["precision"] for item in class_metrics) / 23,
        "macro_recall": sum(item["recall"] for item in class_metrics) / 23,
        "macro_f1": sum(item["f1"] for item in class_metrics) / 23,
        "classes": class_metrics,
    }


def optimizer_and_scheduler(torch: Any, head: Any, contract: dict[str, Any]) -> tuple[Any, Any]:
    classifier = []
    other = []
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad:
            continue
        (classifier if "classifier" in name.lower() else other).append(parameter)
    if not classifier or not other:
        raise AssertionError("G7 requires classifier and non-classifier head groups")
    optimizer = torch.optim.AdamW(
        [
            {"params": other, "lr": contract["head_other_learning_rate"], "name": "CaptionClassification_other"},
            {"params": classifier, "lr": contract["classifier_learning_rate"], "name": "CaptionClassification_classifier"},
        ],
        weight_decay=contract["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=contract["epochs"],
        eta_min=contract["scheduler_min_lr"],
    )
    return optimizer, scheduler


def rng_state(torch: Any, numpy: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(torch: Any, numpy: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def save_transactional_checkpoint(
    torch: Any,
    head: Any,
    optimizer: Any,
    scheduler: Any,
    numpy: Any,
    progress: dict[str, int],
    manifest_sha256: str,
) -> dict[str, Any]:
    if CHECKPOINT_PATH.exists() or CHECKPOINT_TEMP_PATH.exists():
        raise FileExistsError("G7 checkpoint destination is already occupied")
    CHECKPOINT_TEMP_PATH.mkdir(parents=False, exist_ok=False)
    torch.save(head.state_dict(), CHECKPOINT_TEMP_PATH / "CaptionClassification.pt")
    torch.save(optimizer.state_dict(), CHECKPOINT_TEMP_PATH / "optimizer_state.pt")
    torch.save(scheduler.state_dict(), CHECKPOINT_TEMP_PATH / "scheduler_state.pt")
    torch.save(rng_state(torch, numpy), CHECKPOINT_TEMP_PATH / "rng_state.pt")
    state_document = {
        "schema_version": 1,
        "gate": "G7",
        "scope": "complete CaptionClassification head only; frozen backbone/text referenced externally",
        "progress": progress,
        "source_manifest_sha256": manifest_sha256,
        "base_checkpoint": str(CHECKPOINT_DIR),
        "base_checkpoint_type": CKPT_TYPE,
        "config": str(CONFIG_PATH),
    }
    (CHECKPOINT_TEMP_PATH / "training_state.json").write_text(
        json.dumps(state_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    required = [
        "CaptionClassification.pt",
        "optimizer_state.pt",
        "scheduler_state.pt",
        "rng_state.pt",
        "training_state.json",
    ]
    files = {
        name: {
            "bytes": (CHECKPOINT_TEMP_PATH / name).stat().st_size,
            "sha256": sha256_file(CHECKPOINT_TEMP_PATH / name),
        }
        for name in required
    }
    checkpoint_manifest = {
        "schema_version": 1,
        "complete": True,
        "required_files": required,
        "files": files,
    }
    manifest_path = CHECKPOINT_TEMP_PATH / "manifest.json"
    manifest_path.write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in parsed["required_files"]:
        path = CHECKPOINT_TEMP_PATH / name
        if path.stat().st_size != parsed["files"][name]["bytes"]:
            raise AssertionError(f"Checkpoint size verification failed: {name}")
        if sha256_file(path) != parsed["files"][name]["sha256"]:
            raise AssertionError(f"Checkpoint hash verification failed: {name}")
    (CHECKPOINT_TEMP_PATH / "COMPLETE").write_text("complete\n", encoding="utf-8")
    os.replace(CHECKPOINT_TEMP_PATH, CHECKPOINT_PATH)
    return {
        "path": str(CHECKPOINT_PATH),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "complete_marker": True,
    }


def load_checkpoint_for_resume(
    torch: Any,
    numpy: Any,
    head: Any,
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    manifest = json.loads((CHECKPOINT_PATH / "manifest.json").read_text(encoding="utf-8"))
    if not (CHECKPOINT_PATH / "COMPLETE").is_file() or not manifest.get("complete"):
        raise AssertionError("G7 checkpoint is not complete")
    for name in manifest["required_files"]:
        path = CHECKPOINT_PATH / name
        if not path.is_file() or path.stat().st_size != manifest["files"][name]["bytes"]:
            raise AssertionError(f"Resume checkpoint file invalid: {name}")
        if sha256_file(path) != manifest["files"][name]["sha256"]:
            raise AssertionError(f"Resume checkpoint hash invalid: {name}")
    head.load_state_dict(torch.load(CHECKPOINT_PATH / "CaptionClassification.pt", map_location="cpu"), strict=True)
    optimizer.load_state_dict(torch.load(CHECKPOINT_PATH / "optimizer_state.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(CHECKPOINT_PATH / "scheduler_state.pt", map_location="cpu"))
    saved_rng = torch.load(CHECKPOINT_PATH / "rng_state.pt", map_location="cpu")
    restore_rng_state(torch, numpy, saved_rng)
    return json.loads((CHECKPOINT_PATH / "training_state.json").read_text(encoding="utf-8"))


def train_step(
    torch: Any,
    head: Any,
    loss_fn: Any,
    optimizer: Any,
    features: dict[str, Any],
    annotations: list[dict[str, Any]],
    metas: tuple[dict[str, Any], ...],
    max_norm: float,
) -> dict[str, Any]:
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss, logits, labels = loss_and_predictions(torch, head, loss_fn, features, annotations, metas)
    pre_loss = float(loss.detach().cpu())
    loss.backward()
    squared_norm = torch.zeros((), device=logits.device)
    for parameter in head.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError("Non-finite G7 gradient")
            squared_norm += parameter.grad.detach().float().pow(2).sum()
    gradient_norm = float(torch.sqrt(squared_norm).cpu())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise AssertionError(f"Invalid G7 gradient norm: {gradient_norm}")
    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=max_norm)
    optimizer.step()
    return {
        "loss": pre_loss,
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean().cpu()),
        "gradient_norm": gradient_norm,
    }


def assert_states_close(torch: Any, reference: Any, resumed: Any, rtol: float, atol: float) -> float:
    reference_state = reference.state_dict()
    resumed_state = resumed.state_dict()
    if reference_state.keys() != resumed_state.keys():
        raise AssertionError("Resume head keys differ")
    maximum_delta = 0.0
    for name in reference_state:
        left = reference_state[name].detach().cpu()
        right = resumed_state[name].detach().cpu()
        delta = float((left - right).abs().max()) if left.numel() else 0.0
        maximum_delta = max(maximum_delta, delta)
        if not torch.allclose(left, right, rtol=rtol, atol=atol):
            raise AssertionError(f"Resume parameter mismatch: {name}, max_delta={delta}")
    return maximum_delta


def assert_optimizer_states_close(
    torch: Any,
    reference: Any,
    resumed: Any,
    rtol: float,
    atol: float,
) -> float:
    left = reference.state_dict()
    right = resumed.state_dict()
    if left["param_groups"] != right["param_groups"] or left["state"].keys() != right["state"].keys():
        raise AssertionError("Resume optimizer structure differs")
    maximum_delta = 0.0
    for parameter_id in left["state"]:
        if left["state"][parameter_id].keys() != right["state"][parameter_id].keys():
            raise AssertionError(f"Resume optimizer keys differ for {parameter_id}")
        for name, left_value in left["state"][parameter_id].items():
            right_value = right["state"][parameter_id][name]
            if torch.is_tensor(left_value):
                delta = float((left_value.detach().cpu() - right_value.detach().cpu()).abs().max())
                maximum_delta = max(maximum_delta, delta)
                if not torch.allclose(
                    left_value.detach().cpu(),
                    right_value.detach().cpu(),
                    rtol=rtol,
                    atol=atol,
                ):
                    raise AssertionError(
                        f"Resume optimizer tensor differs: {parameter_id}:{name}"
                    )
            elif left_value != right_value:
                raise AssertionError(
                    f"Resume optimizer scalar differs: {parameter_id}:{name}"
                )
    return maximum_delta


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stop = threading.Event()
    state = PhaseState()
    heartbeat_thread = threading.Thread(
        target=heartbeat, args=(stop, started, state), daemon=True
    )
    heartbeat_thread.start()
    results: dict[str, Any] = {
        "gate": "G7",
        "status": "failed",
        "error": None,
        "code": git_identity(),
        "manifest": str(MANIFEST_PATH),
        "environment": {},
        "gpu": {},
        "assets": {},
        "training": {"epochs": [], "resume_probe": {}},
        "checkpoint": {},
        "assertions": {},
        "memory_snapshots": [],
        "timings_seconds": {},
        "peak_cpu_rss_kib": None,
        "total_elapsed_seconds": None,
        "artifacts": {
            "result_json": str(RESULT_PATH),
            "checkpoint": str(CHECKPOINT_PATH),
        },
        "explicit_non_goals": [
            "no test split access",
            "no backbone or text encoder update",
            "no VideoCaption or detection-head training",
            "no mixed precision, DataLoader, distributed training, or multi-task scheduling",
            "no G8 execution",
        ],
    }
    exit_code = 1
    try:
        manifest = timed("load_manifest", state, results["timings_seconds"], load_manifest)
        manifest_sha = sha256_file(MANIFEST_PATH)
        timed("validate_paths", state, results["timings_seconds"], lambda: require_paths(manifest))
        records = timed(
            "validate_fixed_records",
            state,
            results["timings_seconds"],
            lambda: prepare_records(manifest),
        )

        def import_framework() -> tuple[Any, ...]:
            import numpy
            import torch
            from data.video_caption import build_transforms, keywords_list, read_frames_decord
            from models.caption_classification import (
                build_caption_classification_head,
                build_caption_classification_loss,
            )
            from models.multi_task import MultiTaskingSigLIP
            return (
                numpy,
                torch,
                build_transforms,
                keywords_list,
                read_frames_decord,
                build_caption_classification_head,
                build_caption_classification_loss,
                MultiTaskingSigLIP,
            )

        (
            numpy,
            torch,
            build_transforms,
            keywords,
            read_frames,
            build_head,
            build_loss,
            model_class,
        ) = timed("import_framework", state, results["timings_seconds"], import_framework)
        manifest_labels = [
            item["label"] for item in manifest["splits"]["train"]["classes"]
        ]
        if list(keywords) != manifest_labels:
            raise AssertionError("Manifest class order differs from repository keywords_list")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE_PHYSICAL_GPU:
            raise RuntimeError(
                "G7 requires external CUDA_VISIBLE_DEVICES=7; got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("G7 requires exactly one visible CUDA device")
        contract = manifest["training_contract"]
        seed = int(contract["seed"])
        set_global_seed(torch, numpy, seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        device = torch.device(DEVICE)
        results["environment"] = {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda_build": torch.version.cuda,
            "dtype": "torch.float32",
            "mixed_precision": False,
            "num_workers": 0,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        }
        results["gpu"] = {
            "visible_physical_index": int(VISIBLE_PHYSICAL_GPU),
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        }
        first_memory = memory_snapshot(torch, "after_cuda_init")
        results["memory_snapshots"].append(first_memory)
        if first_memory["free_bytes"] < MIN_FREE_GIB * GIB:
            raise RuntimeError(f"G7 requires at least {MIN_FREE_GIB} GiB free")
        config = timed(
            "load_config", state, results["timings_seconds"], lambda: load_config(manifest)
        )
        results["assets"] = {
            "manifest_sha256": manifest_sha,
            "config": str(CONFIG_PATH),
            "base_model": str(MODEL_PATH),
            "base_checkpoint": str(CHECKPOINT_DIR),
            "checkpoint_type": CKPT_TYPE,
            "dataset": manifest["dataset"],
            "source_root": manifest["source_root"],
            "train_samples": len(records["train"]),
            "valid_samples": len(records["valid"]),
            "train_video_bytes": sum(item["video_bytes"] for item in records["train"]),
            "valid_video_bytes": sum(item["video_bytes"] for item in records["valid"]),
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
                str(CHECKPOINT_DIR), CKPT_TYPE, logger=None, load_heads=LOAD_HEADS
            ),
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        head = model.multi_task_head["CaptionClassification"]
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        trainable_names = [name for name, p in head.named_parameters() if p.requires_grad]
        if not any("transformer_encoder" in name for name in trainable_names):
            raise AssertionError("G7 full-head scope excludes transformer parameters")
        if not any("classifier_ln" in name for name in trainable_names):
            raise AssertionError("G7 full-head scope excludes layer-norm parameters")
        initial_head_state = {
            name: value.detach().cpu().clone() for name, value in head.state_dict().items()
        }
        model.to(device)
        model.eval()
        head.train()
        frozen_versions = {
            name: parameter._version
            for name, parameter in model.named_parameters()
            if not name.startswith("multi_task_head.CaptionClassification.")
        }
        torch.cuda.reset_peak_memory_stats(device)
        results["memory_snapshots"].append(memory_snapshot(torch, "after_model_to_gpu"))
        loss_fn = build_loss(config).to(device)
        optimizer, scheduler = optimizer_and_scheduler(torch, head, contract)
        train_transform = build_transforms(config, split="train")
        valid_transform = build_transforms(config, split="valid")
        batch_size = int(contract["batch_size"])

        initial_train = evaluate_split(
            torch, numpy, model, head, loss_fn, records["train"], "train", -1,
            config, valid_transform, read_frames, device, batch_size, seed, state,
        )
        initial_valid = evaluate_split(
            torch, numpy, model, head, loss_fn, records["valid"], "valid", -1,
            config, valid_transform, read_frames, device, batch_size, seed, state,
        )
        results["training"]["initial"] = {"train": initial_train, "valid": initial_valid}
        global_step = 0
        resume_probe_done = False
        processed_train_ids = []
        gradient_norms = []

        for epoch in range(int(contract["epochs"])):
            order = epoch_order(records["train"], seed, epoch)
            epoch_batches = batches(order, batch_size)
            epoch_losses = []
            for batch_number, batch_indices in enumerate(epoch_batches):
                state.set(f"train_epoch_{epoch}_batch_{batch_number + 1}_of_{len(epoch_batches)}")
                images, annotations, metas, ids = make_batch(
                    torch, numpy, records["train"], batch_indices, "train", epoch,
                    seed, config, train_transform, read_frames,
                )
                input_digest = tensor_sha256(images)
                features = forward_features(torch, model, images, device)
                step_result = train_step(
                    torch, head, loss_fn, optimizer, features, annotations, metas,
                    float(contract["gradient_clip_max_norm"]),
                )
                global_step += 1
                epoch_losses.append(step_result["loss"])
                gradient_norms.append(step_result["gradient_norm"])
                processed_train_ids.extend(ids)
                print(
                    f"[TRAIN] epoch={epoch} step={global_step} "
                    f"loss={step_result['loss']:.8f} accuracy={step_result['accuracy']:.4f}",
                    flush=True,
                )
                if global_step == int(contract["resume_probe_after_optimizer_step"]):
                    progress = {
                        "epoch": epoch,
                        "completed_batch_in_epoch": batch_number + 1,
                        "global_step": global_step,
                    }
                    results["checkpoint"] = save_transactional_checkpoint(
                        torch, head, optimizer, scheduler, numpy, progress, manifest_sha
                    )
                elif (
                    global_step == int(contract["resume_probe_after_optimizer_step"]) + 1
                    and not resume_probe_done
                ):
                    reference_ids = ids
                    reference_input_digest = input_digest
                    reference_step = step_result
                    resumed_head = build_head(config).to(device)
                    for parameter in resumed_head.parameters():
                        parameter.requires_grad_(True)
                    resumed_optimizer, resumed_scheduler = optimizer_and_scheduler(
                        torch, resumed_head, contract
                    )
                    resume_state = load_checkpoint_for_resume(
                        torch, numpy, resumed_head, resumed_optimizer, resumed_scheduler
                    )
                    resumed_images, resumed_annotations, resumed_metas, resumed_ids = make_batch(
                        torch, numpy, records["train"], batch_indices, "train", epoch,
                        seed, config, train_transform, read_frames,
                    )
                    resumed_digest = tensor_sha256(resumed_images)
                    resumed_features = forward_features(torch, model, resumed_images, device)
                    resumed_step = train_step(
                        torch, resumed_head, loss_fn, resumed_optimizer, resumed_features,
                        resumed_annotations, resumed_metas,
                        float(contract["gradient_clip_max_norm"]),
                    )
                    rtol = float(contract["resume_rtol"])
                    atol = float(contract["resume_atol"])
                    max_delta = assert_states_close(
                        torch, head, resumed_head, rtol=rtol, atol=atol
                    )
                    optimizer_max_delta = assert_optimizer_states_close(
                        torch,
                        optimizer,
                        resumed_optimizer,
                        rtol=rtol,
                        atol=atol,
                    )
                    if reference_ids != resumed_ids or reference_input_digest != resumed_digest:
                        raise AssertionError("Exact-resume next batch differs")
                    if not math.isclose(reference_step["loss"], resumed_step["loss"], rel_tol=rtol, abs_tol=atol):
                        raise AssertionError("Exact-resume next loss differs")
                    if scheduler.state_dict() != resumed_scheduler.state_dict():
                        raise AssertionError("Exact-resume scheduler state differs")
                    results["training"]["resume_probe"] = {
                        "passed": True,
                        "saved_progress": resume_state["progress"],
                        "next_batch_ids": reference_ids,
                        "next_input_sha256": reference_input_digest,
                        "reference_loss": reference_step["loss"],
                        "resumed_loss": resumed_step["loss"],
                        "maximum_parameter_delta": max_delta,
                        "maximum_optimizer_state_delta": optimizer_max_delta,
                        "rtol": rtol,
                        "atol": atol,
                    }
                    resume_probe_done = True
                    del resumed_head, resumed_optimizer, resumed_scheduler, resumed_features
            scheduler.step()
            train_metrics = evaluate_split(
                torch, numpy, model, head, loss_fn, records["train"], "train", epoch,
                config, valid_transform, read_frames, device, batch_size, seed, state,
            )
            valid_metrics = evaluate_split(
                torch, numpy, model, head, loss_fn, records["valid"], "valid", epoch,
                config, valid_transform, read_frames, device, batch_size, seed, state,
            )
            results["training"]["epochs"].append(
                {
                    "epoch": epoch,
                    "mean_online_train_loss": sum(epoch_losses) / len(epoch_losses),
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "learning_rates_after_scheduler_step": [group["lr"] for group in optimizer.param_groups],
                }
            )

        final_train = results["training"]["epochs"][-1]["train"]
        final_valid = results["training"]["epochs"][-1]["valid"]
        non_classifier_delta = 0.0
        for name, value in head.state_dict().items():
            if "classifier" not in name.lower():
                non_classifier_delta = max(
                    non_classifier_delta,
                    float((value.detach().cpu() - initial_head_state[name]).abs().max()),
                )
        final_frozen_versions = {
            name: parameter._version
            for name, parameter in model.named_parameters()
            if not name.startswith("multi_task_head.CaptionClassification.")
        }
        frozen_unchanged = frozen_versions == final_frozen_versions
        frozen_with_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not name.startswith("multi_task_head.CaptionClassification.")
            and parameter.grad is not None
        ]
        all_train_ids = {
            f"train:{record['json_index']}" for record in records["train"]
        }
        processed_per_epoch = len(processed_train_ids) == len(records["train"]) * int(contract["epochs"])
        ids_covered = set(processed_train_ids) == all_train_ids
        assertions = {
            "all_train_samples_processed_each_epoch": processed_per_epoch and ids_covered,
            "finite_positive_gradients": bool(gradient_norms)
            and all(math.isfinite(value) and value > 0 for value in gradient_norms),
            "non_classifier_head_parameters_changed": non_classifier_delta > 0,
            "frozen_backbone_and_text_unchanged": frozen_unchanged
            and not frozen_with_gradients,
            "final_train_loss_below_initial": final_train["loss"] < initial_train["loss"],
            "valid_has_all_23_classes": len(final_valid["classes"]) == 23
            and all(item["count"] == 4 for item in final_valid["classes"]),
            "exact_resume_probe_passed": resume_probe_done
            and results["training"]["resume_probe"].get("passed") is True,
            "test_split_not_used": True,
        }
        if not all(assertions.values()):
            raise AssertionError(f"G7 assertions failed: {assertions}")
        results["assertions"] = assertions
        results["training"].update(
            {
                "global_steps": global_step,
                "processed_train_samples": len(processed_train_ids),
                "gradient_norm_minimum": min(gradient_norms),
                "gradient_norm_maximum": max(gradient_norms),
                "non_classifier_maximum_parameter_delta": non_classifier_delta,
            }
        )
        results["memory_snapshots"].append(memory_snapshot(torch, "after_training"))
        results["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        results["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(results["error"]["traceback"], flush=True)
    finally:
        results["peak_cpu_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
        state.set("write_result")
        try:
            if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {RESULT_PATH}")
            RESULT_PATH.write_text(
                json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)
        except BaseException:
            traceback.print_exc()
            exit_code = 1
        stop.set()
        heartbeat_thread.join(timeout=5)
        print(f"[EXIT] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
