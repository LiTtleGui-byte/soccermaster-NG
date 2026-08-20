#!/usr/bin/env python3
"""Run the fixed, two-pass G5 small evaluation on one visible GPU."""

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


# These settings must be applied before importing torch or project modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
MANIFEST_PATH = REPO / "reproduction/manifests/g5_fixed_eval.json"
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
REPORTS = REPO / "reports/g5"
DATA_VIEW_ROOT = REPO / ".runtime/data_views/g5"
RESULT_PATH = REPORTS / "g5_fixed_small_eval_retry1_result_20260812.json"
OVERLAY_PATHS = (
    REPORTS / "g5_detection_clip_start_000_retry1_overlay_20260812.jpg",
    REPORTS / "g5_detection_clip_start_360_retry1_overlay_20260812.jpg",
)

DEVICE = "cuda:0"
CONFIG_DEVICE = "cpu"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
VISIBLE_PHYSICAL_GPU = "7"
SEED = 42
NUM_FRAMES = 30
IMAGE_SIZE = 512
EVALUATION_PASSES = 2
HEARTBEAT_SECONDS = 30
GIB = 1024**3
MIN_FREE_BEFORE_LOAD_GIB = 30
MIN_FREE_BEFORE_FORWARD_GIB = 20
ATOL = 1e-6
RTOL = 1e-5

EXPECTED_FILES = (
    "backbone.pt",
    "SoccerNetGSR_Detection.pt",
    "LinesDetection.pt",
    "KeypointsDetection.pt",
    "VideoCaption.pt",
    "CaptionClassification.pt",
)
EXPECTED_HEADS = {
    "SoccerNetGSR_Detection",
    "LinesDetection",
    "KeypointsDetection",
    "VideoCaption",
    "CaptionClassification",
}

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


class SingleProcessAccelerator:
    """The metrics API only needs this flag in a non-distributed G5 run."""

    is_main_process = True


def heartbeat(stop_event: threading.Event, state: PhaseState) -> None:
    started = time.monotonic()
    while not stop_event.wait(HEARTBEAT_SECONDS):
        print(
            f"[HEARTBEAT] phase={state.get()} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )


def timed(
    phase: str,
    state: PhaseState,
    timings: dict[str, float],
    function: Callable[[], Any],
) -> Any:
    state.set(phase)
    started = time.monotonic()
    try:
        return function()
    finally:
        timings[phase] = round(time.monotonic() - started, 3)
        print(f"[TIMING] {phase}={timings[phase]}s", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
    if manifest.get("schema_version") != 1 or manifest.get("gate") != "G5":
        raise AssertionError("Unexpected G5 manifest schema or gate")
    clips = manifest["detection"]["clips"]
    samples = manifest["caption"]["samples"]
    if len(clips) != 2 or len(samples) != 23:
        raise AssertionError(
            f"Manifest contains {len(clips)} detection clips and "
            f"{len(samples)} caption samples"
        )
    if len({item["label"] for item in samples}) != 23:
        raise AssertionError("Caption labels are not unique")
    return manifest


def require_paths(manifest: dict[str, Any]) -> None:
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
        CHECKPOINT_DIR / "text_model/model.safetensors",
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)

    detection = manifest["detection"]
    detection_root = Path(detection["source_root"])
    sequence_root = (
        detection_root
        / "SoccerNetGS"
        / detection["split"]
        / detection["sequence"]
    )
    required.extend(
        [
            sequence_root,
            sequence_root / "Labels-GameState.json",
            detection_root / "legibility_jn/test.json",
            detection_root
            / "camera_params"
            / detection["split"]
            / f"{detection['sequence']}.json",
        ]
    )
    for clip in detection["clips"]:
        required.extend(
            [
                sequence_root / "img1" / clip["first_image"],
                sequence_root / "img1" / clip["last_image"],
            ]
        )

    caption = manifest["caption"]
    caption_root = Path(caption["source_root"])
    label_path = (
        caption_root
        / "video_clip_json"
        / caption["dataset"]
        / f"classification_{caption['split']}.json"
    )
    required.append(label_path)
    video_root = (
        caption_root
        / "video_clip"
        / f"{caption['dataset']}-high-resolution"
    )
    for sample in caption["samples"]:
        required.append(video_root / sample["relative_video"])

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required G5 paths are missing:\n" + "\n".join(missing)
        )

    detection_label = sequence_root / "Labels-GameState.json"
    if sha256_file(detection_label) != detection["label_sha256"]:
        raise AssertionError("Detection label SHA256 does not match manifest")
    if sha256_file(label_path) != caption["label_sha256"]:
        raise AssertionError("Caption label SHA256 does not match manifest")
    for sample in caption["samples"]:
        path = video_root / sample["relative_video"]
        if path.stat().st_size != sample["video_bytes"]:
            raise AssertionError(f"Video size changed: {path}")


def ensure_unused_output_paths() -> None:
    occupied = [
        str(path)
        for path in (RESULT_PATH, *OVERLAY_PATHS)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise FileExistsError(
            "G5 output paths already exist; refusing to overwrite:\n"
            + "\n".join(occupied)
        )


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Existing symlink {link} points to {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Expected a symlink but found existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_detection_view(manifest: dict[str, Any]) -> Path:
    detection = manifest["detection"]
    source_root = Path(detection["source_root"])
    source_sequence = (
        source_root
        / "SoccerNetGS"
        / detection["split"]
        / detection["sequence"]
    )
    local_dataset_root = DATA_VIEW_ROOT / "SN-GSR-2024"
    sequence_link = (
        local_dataset_root
        / "SoccerNetGS"
        / detection["split"]
        / detection["sequence"]
    )
    legibility_link = local_dataset_root / "legibility_jn/test.json"
    camera_params_link = (
        local_dataset_root
        / "camera_params"
        / detection["split"]
        / f"{detection['sequence']}.json"
    )
    ensure_symlink(sequence_link, source_sequence)
    ensure_symlink(legibility_link, source_root / "legibility_jn/test.json")
    ensure_symlink(
        camera_params_link,
        source_root
        / "camera_params"
        / detection["split"]
        / f"{detection['sequence']}.json",
    )
    return DATA_VIEW_ROOT


def load_config(data_view_root: Path) -> dict[str, Any]:
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
            "DATA_ROOT": str(data_view_root),
            "USE_EXTRA_DATA": False,
            "EXTRA_DATA_ONLY": False,
            "NUM_WORKERS": 0,
            "VIDEO_CAPTION_NUM_WORKERS": 0,
            "BATCH_SIZE": 1,
            "VIDEO_CAPTION_TEST_BATCH_SIZE": 1,
            "DATASETS_TO_HEADS": {
                "SoccerNetGSR_Detection": [
                    "SoccerNetGSR_Detection",
                    "LinesDetection",
                    "KeypointsDetection",
                ],
                "VideoCaption": [
                    "VideoCaption",
                    "CaptionClassification",
                ],
            },
        }
    )
    if config["NUM_FRAMES"] != NUM_FRAMES:
        raise AssertionError(f"NUM_FRAMES is {config['NUM_FRAMES']}")
    if config["AUG_RANDOM_RESIZE"] != [IMAGE_SIZE]:
        raise AssertionError(f"Unexpected resize: {config['AUG_RANDOM_RESIZE']}")
    return config


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


def tensor_fingerprint(torch: Any, tensor: Any) -> dict[str, Any]:
    value = tensor.detach().float()
    if not bool(torch.isfinite(value).all().item()):
        raise AssertionError("Non-finite tensor in G5 output")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "mean": float(value.mean()),
        "sum": float(value.sum()),
    }


def move_nested_to_device(torch: Any, value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: move_nested_to_device(torch, child, device)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [move_nested_to_device(torch, child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(move_nested_to_device(torch, child, device) for child in value)
    return value


def detection_fingerprint(torch: Any, outputs: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "SoccerNetGSR_Detection",
        "LinesDetection",
        "KeypointsDetection",
    }
    if set(outputs) != expected:
        raise AssertionError(f"Detection output heads are {set(outputs)}")
    tensors = {
        "pred_logits": outputs["SoccerNetGSR_Detection"]["pred_logits"],
        "pred_boxes": outputs["SoccerNetGSR_Detection"]["pred_boxes"],
        "pred_roles": outputs["SoccerNetGSR_Detection"]["pred_roles"],
        "pred_jn_holistic": outputs["SoccerNetGSR_Detection"][
            "pred_jn_holistic"
        ],
        "pred_digit_head": outputs["SoccerNetGSR_Detection"][
            "pred_digit_head"
        ],
        "pred_digit_tail": outputs["SoccerNetGSR_Detection"][
            "pred_digit_tail"
        ],
        "pred_lines_heatmap": outputs["LinesDetection"]["pred_lines_heatmap"],
        "pred_keypoints_heatmap": outputs["KeypointsDetection"][
            "pred_keypoints_heatmap"
        ],
    }
    expected_shapes = {
        "pred_logits": (1, 30, 300, 1),
        "pred_boxes": (1, 30, 300, 4),
        "pred_roles": (1, 30, 300, 6),
        "pred_jn_holistic": (1, 30, 300, 101),
        "pred_digit_head": (1, 30, 300, 10),
        "pred_digit_tail": (1, 30, 300, 11),
        "pred_lines_heatmap": (1, 30, 24, 256, 256),
        "pred_keypoints_heatmap": (1, 30, 58, 256, 256),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise AssertionError(
                f"{name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
            )
    return {name: tensor_fingerprint(torch, value) for name, value in tensors.items()}


def prepare_caption_records(
    manifest: dict[str, Any],
    keywords: list[str],
) -> list[dict[str, Any]]:
    caption = manifest["caption"]
    source_root = Path(caption["source_root"])
    label_path = (
        source_root
        / "video_clip_json"
        / caption["dataset"]
        / f"classification_{caption['split']}.json"
    )
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    video_root = (
        source_root
        / "video_clip"
        / f"{caption['dataset']}-high-resolution"
    )
    records = []
    for expected_class_index, spec in enumerate(caption["samples"]):
        item = labels[spec["label_index"]]
        text = item.get("comments_text_anonymized")
        if item.get("caption") != spec["label"]:
            raise AssertionError(f"Caption label changed at {spec['label_index']}")
        if item.get("video") != spec["relative_video"]:
            raise AssertionError(f"Video path changed at {spec['label_index']}")
        if not isinstance(text, str) or len(text) != spec["text_length"]:
            raise AssertionError(f"Text changed at {spec['label_index']}")
        if keywords[expected_class_index] != spec["label"]:
            raise AssertionError(
                "Manifest caption order must exactly match keywords_list"
            )
        records.append(
            {
                "manifest_index": expected_class_index,
                "label_index": spec["label_index"],
                "caption": spec["label"],
                "caption_index": expected_class_index,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "video": video_root / spec["relative_video"],
                "relative_video": spec["relative_video"],
            }
        )
    return records


def select_detection_indices(
    dataset: Any,
    manifest: dict[str, Any],
) -> list[int]:
    sequence = manifest["detection"]["sequence"]
    requested = [
        (sequence, clip["start_frame_zero_based"])
        for clip in manifest["detection"]["clips"]
    ]
    indices = []
    for position in requested:
        matches = [
            index
            for index, candidate in enumerate(dataset.sample_position)
            if candidate == position
        ]
        if len(matches) != 1:
            raise AssertionError(f"Detection sample {position} matched {matches}")
        indices.append(matches[0])
    return indices


def finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"Non-finite metric {name}: {number}")
    return number


def normalize_metric_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: finite_number(value, key) for key, value in sorted(metrics.items())}


def classification_metrics(
    torch: Any,
    logits: Any,
    targets: Any,
) -> dict[str, float | int]:
    predictions = logits.argmax(dim=1)
    probabilities = torch.softmax(logits, dim=1)
    accuracy = float((predictions == targets).float().mean())
    per_class = []
    precision_values = []
    recall_values = []
    f1_values = []
    for class_index in range(logits.shape[1]):
        tp = int(((predictions == class_index) & (targets == class_index)).sum())
        fp = int(((predictions == class_index) & (targets != class_index)).sum())
        fn = int(((predictions != class_index) & (targets == class_index)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class.append(float(predictions[class_index] == targets[class_index]))
    return {
        "classification_accuracy": accuracy,
        "avg_confidence": float(probabilities.max(dim=1).values.mean()),
        "macro_accuracy": sum(per_class) / len(per_class),
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "total_samples": int(targets.numel()),
        "num_classes_with_samples": int(targets.unique().numel()),
    }


def retrieval_metrics(torch: Any, similarities: Any) -> dict[str, float | int]:
    if tuple(similarities.shape) != (23, 23):
        raise AssertionError(f"Similarity matrix shape is {similarities.shape}")
    targets = torch.arange(23)
    ranking = similarities.argsort(dim=1, descending=True)
    output: dict[str, float | int] = {"total_samples": 23}
    for k in (1, 3, 5):
        correct = (ranking[:, :k] == targets[:, None]).any(dim=1)
        output[f"retrieval_top_{k}_accuracy"] = float(correct.float().mean())
    return output


def caption_predictions(
    torch: Any,
    logits: Any,
    similarities: Any,
    records: list[dict[str, Any]],
    keywords: list[str],
) -> list[dict[str, Any]]:
    probabilities = torch.softmax(logits, dim=1)
    class_probabilities, class_indices = probabilities.topk(k=5, dim=1)
    retrieval_indices = similarities.argsort(dim=1, descending=True)[:, :5]
    predictions = []
    for index, record in enumerate(records):
        predictions.append(
            {
                "manifest_index": index,
                "label_index": record["label_index"],
                "relative_video": record["relative_video"],
                "ground_truth": record["caption"],
                "classification_top5": [
                    {
                        "index": int(class_index),
                        "label": keywords[int(class_index)],
                        "probability": float(probability),
                    }
                    for probability, class_index in zip(
                        class_probabilities[index], class_indices[index]
                    )
                ],
                "retrieval_top5_text_indices": [
                    int(value) for value in retrieval_indices[index]
                ],
            }
        )
    return predictions


def save_detection_overlay(
    torch: Any,
    image_class: Any,
    image_draw: Any,
    images_cpu: Any,
    annotations: list[dict[str, Any]],
    detection_output: dict[str, Any],
    detection_metric: Any,
    destination: Path,
) -> None:
    frame = images_cpu[0].float().mul(0.5).add(0.5).clamp(0, 1)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    image = image_class.fromarray(array)
    draw = image_draw.Draw(image)
    target_sizes = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]] * NUM_FRAMES)
    flattened = {}
    for key, value in detection_output.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 3:
            flattened[key] = value.detach().cpu().reshape(
                NUM_FRAMES, *value.shape[2:]
            )
        else:
            flattened[key] = value
    predictions = detection_metric.postprocess(flattened, target_sizes)
    first_prediction = predictions[0]
    for box, score in zip(
        first_prediction["boxes"], first_prediction["scores"]
    ):
        if float(score) <= 0.5:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
    for box in annotations[0]["boxes"]:
        cx, cy, width, height = [float(value) for value in box]
        x1 = (cx - width / 2) * IMAGE_SIZE
        y1 = (cy - height / 2) * IMAGE_SIZE
        x2 = (cx + width / 2) * IMAGE_SIZE
        y2 = (cy + height / 2) * IMAGE_SIZE
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
    image.save(destination, format="JPEG", quality=92)


def run_evaluation_pass(
    pass_index: int,
    torch: Any,
    image_class: Any,
    image_draw: Any,
    model: Any,
    config: dict[str, Any],
    manifest: dict[str, Any],
    detection_dataset: Any,
    detection_indices: list[int],
    caption_records: list[dict[str, Any]],
    caption_transform: Any,
    read_frames: Callable[..., Any],
    keywords: list[str],
    build_metrics_fn: Callable[..., Any],
    state: PhaseState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    metrics = build_metrics_fn(config=config)
    if set(metrics) != EXPECTED_HEADS:
        raise AssertionError(f"Metric heads are {set(metrics)}")
    accelerator = SingleProcessAccelerator()
    device = torch.device(DEVICE)
    pass_result: dict[str, Any] = {
        "pass_index": pass_index,
        "detection_clips": [],
        "caption_predictions": [],
        "metrics": {},
    }
    comparison: dict[str, Any] = {"detection": []}

    for clip_number, dataset_index in enumerate(detection_indices):
        start_frame = manifest["detection"]["clips"][clip_number][
            "start_frame_zero_based"
        ]
        state.set(
            f"pass_{pass_index}_detection_clip_{clip_number + 1}_of_2"
        )
        images_cpu, annotations, metas = detection_dataset[dataset_index]
        if tuple(images_cpu.shape) != (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise AssertionError(f"Detection input shape is {images_cpu.shape}")
        if len(annotations) != NUM_FRAMES:
            raise AssertionError(f"Detection annotation count is {len(annotations)}")
        if metas["start_frame"] != start_frame:
            raise AssertionError(
                f"Dataset start frame {metas['start_frame']} != {start_frame}"
            )
        images = images_cpu.unsqueeze(0).to(device)
        annotations_device = move_nested_to_device(
            torch, annotations, device
        )
        with torch.inference_mode():
            outputs = model(
                images,
                "SoccerNetGSR_Detection",
                metas=(metas,),
                text=None,
            )
        torch.cuda.synchronize(device)
        fingerprint = detection_fingerprint(torch, outputs)
        target_sizes = torch.tensor(
            [[IMAGE_SIZE, IMAGE_SIZE]], device=device, dtype=torch.long
        )
        metrics["SoccerNetGSR_Detection"].update(
            outputs["SoccerNetGSR_Detection"],
            [annotations_device],
            target_sizes,
        )
        metrics["LinesDetection"].update(
            outputs["LinesDetection"], [annotations_device]
        )
        metrics["KeypointsDetection"].update(
            outputs["KeypointsDetection"], [annotations_device]
        )
        valid_lines = sum(bool(item["valid_lines"]) for item in annotations)
        valid_keypoints = sum(bool(item["valid_keypoints"]) for item in annotations)
        if valid_lines != NUM_FRAMES or valid_keypoints != NUM_FRAMES:
            raise AssertionError(
                f"Clip {start_frame}: valid lines={valid_lines}, "
                f"keypoints={valid_keypoints}"
            )
        pass_result["detection_clips"].append(
            {
                "sequence": metas["sequence"],
                "start_frame_zero_based": start_frame,
                "end_frame_exclusive": metas["end_frame"],
                "input_shape": list(images_cpu.shape),
                "valid_lines_frames": valid_lines,
                "valid_keypoints_frames": valid_keypoints,
                "output_fingerprint": fingerprint,
            }
        )
        comparison["detection"].append(fingerprint)
        if pass_index == 1:
            save_detection_overlay(
                torch,
                image_class,
                image_draw,
                images_cpu,
                annotations,
                outputs["SoccerNetGSR_Detection"],
                metrics["SoccerNetGSR_Detection"],
                OVERLAY_PATHS[clip_number],
            )
        del images, outputs, images_cpu, annotations_device
        torch.cuda.empty_cache()

    detection_metrics = normalize_metric_dict(
        metrics["SoccerNetGSR_Detection"].compute_final_metrics(accelerator)
    )
    lines_metrics = normalize_metric_dict(
        metrics["LinesDetection"].compute_final_metrics(accelerator)
    )
    keypoints_metrics = normalize_metric_dict(
        metrics["KeypointsDetection"].compute_final_metrics(accelerator)
    )
    if int(lines_metrics.get("lines_valid_samples", -1)) != 60:
        raise AssertionError(f"Lines valid samples: {lines_metrics}")
    if int(keypoints_metrics.get("keypoints_valid_samples", -1)) != 60:
        raise AssertionError(f"Keypoints valid samples: {keypoints_metrics}")

    vision_features = []
    text_features = []
    classification_logits = []
    frame_indices_all = []
    for sample_number, record in enumerate(caption_records):
        state.set(
            f"pass_{pass_index}_caption_{sample_number + 1}_of_23"
        )
        frames, frame_indices, duration = read_frames(
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
        images_cpu, annotation, metas = caption_transform(frames, annotation, metas)
        if tuple(images_cpu.shape) != (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise AssertionError(f"Caption input shape is {images_cpu.shape}")
        if not bool(torch.isfinite(images_cpu).all().item()):
            raise AssertionError("Caption input contains NaN or Inf")
        images = images_cpu.unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(
                images,
                "VideoCaption",
                metas=(metas,),
                text=[record["text"]],
            )
        torch.cuda.synchronize(device)
        if set(outputs) != {"VideoCaption", "CaptionClassification"}:
            raise AssertionError(f"Caption heads are {set(outputs)}")
        video = outputs["VideoCaption"]
        classification = outputs["CaptionClassification"]
        if tuple(video["vision_features"].shape) != (1, 1024):
            raise AssertionError("Unexpected vision feature shape")
        if tuple(video["text_features"].shape) != (1, 1024):
            raise AssertionError("Unexpected text feature shape")
        if tuple(classification["logits"].shape) != (1, 23):
            raise AssertionError("Unexpected classification logits shape")
        for value in (
            video["vision_features"],
            video["text_features"],
            classification["logits"],
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise AssertionError("Caption output contains NaN or Inf")
        vision_features.append(video["vision_features"].detach().cpu())
        text_features.append(video["text_features"].detach().cpu())
        classification_logits.append(classification["logits"].detach().cpu())
        frame_indices_all.append([int(value) for value in frame_indices])
        del frames, images_cpu, images, outputs
        torch.cuda.empty_cache()

    vision_tensor = torch.cat(vision_features, dim=0)
    text_tensor = torch.cat(text_features, dim=0)
    logits_tensor = torch.cat(classification_logits, dim=0)
    similarity_tensor = vision_tensor @ text_tensor.T
    targets = torch.arange(23, dtype=torch.long)
    caption_classification_metrics = classification_metrics(
        torch, logits_tensor, targets
    )
    caption_retrieval_metrics = retrieval_metrics(torch, similarity_tensor)
    if caption_classification_metrics["total_samples"] != 23:
        raise AssertionError("Caption classification did not process 23 samples")
    if caption_classification_metrics["num_classes_with_samples"] != 23:
        raise AssertionError("Caption classification did not cover 23 classes")

    pass_result["metrics"] = {
        "SoccerNetGSR_Detection": detection_metrics,
        "LinesDetection": lines_metrics,
        "KeypointsDetection": keypoints_metrics,
        "VideoCaption": caption_retrieval_metrics,
        "CaptionClassification": caption_classification_metrics,
    }
    pass_result["caption_predictions"] = caption_predictions(
        torch,
        logits_tensor,
        similarity_tensor,
        caption_records,
        keywords,
    )
    pass_result["caption_similarity_matrix"] = similarity_tensor.tolist()
    comparison.update(
        {
            "similarities": similarity_tensor,
            "classification_logits": logits_tensor,
            "frame_indices": frame_indices_all,
            "metrics": pass_result["metrics"],
        }
    )
    return pass_result, comparison


def numeric_leaves(value: Any, path: str = "root") -> dict[str, float]:
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            output.update(numeric_leaves(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(numeric_leaves(child, f"{path}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[path] = float(value)
    return output


def compare_passes(torch: Any, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    torch.testing.assert_close(
        first["similarities"], second["similarities"], rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        first["classification_logits"],
        second["classification_logits"],
        rtol=RTOL,
        atol=ATOL,
    )
    if first["frame_indices"] != second["frame_indices"]:
        raise AssertionError("Caption frame indices changed between passes")

    first_values = numeric_leaves(
        {"detection": first["detection"], "metrics": first["metrics"]}
    )
    second_values = numeric_leaves(
        {"detection": second["detection"], "metrics": second["metrics"]}
    )
    if set(first_values) != set(second_values):
        raise AssertionError("Pass comparison structures differ")
    maximum_absolute_difference = 0.0
    maximum_relative_difference = 0.0
    for path, first_value in first_values.items():
        second_value = second_values[path]
        absolute = abs(first_value - second_value)
        relative = absolute / max(abs(first_value), abs(second_value), 1.0)
        maximum_absolute_difference = max(maximum_absolute_difference, absolute)
        maximum_relative_difference = max(maximum_relative_difference, relative)
        if absolute > ATOL + RTOL * abs(second_value):
            raise AssertionError(
                f"Pass difference at {path}: {first_value} vs {second_value}"
            )
    similarity_difference = float(
        (first["similarities"] - second["similarities"]).abs().max()
    )
    logits_difference = float(
        (first["classification_logits"] - second["classification_logits"])
        .abs()
        .max()
    )
    return {
        "passed": True,
        "rtol": RTOL,
        "atol": ATOL,
        "maximum_numeric_leaf_absolute_difference": maximum_absolute_difference,
        "maximum_numeric_leaf_relative_difference": maximum_relative_difference,
        "maximum_similarity_absolute_difference": similarity_difference,
        "maximum_classification_logits_absolute_difference": logits_difference,
        "caption_frame_indices_identical": True,
    }


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = PhaseState()
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        args=(stop_event, state),
        name="g5-fixed-small-eval-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    results: dict[str, Any] = {
        "gate": "G5",
        "status": "failed",
        "error": None,
        "seed": SEED,
        "evaluation_passes": EVALUATION_PASSES,
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": None,
        "code": git_identity(),
        "environment": {},
        "resolved_config": None,
        "assets": {},
        "gpu": {},
        "memory_snapshots": [],
        "timings_seconds": {},
        "passes": [],
        "repeatability": None,
        "artifacts": {
            "result_json": str(RESULT_PATH),
            "detection_overlays": [str(path) for path in OVERLAY_PATHS],
        },
        "explicit_non_goals": [
            "no full test-set evaluation",
            "no training, loss backward, optimizer, or scheduler",
            "no multi-GPU or distributed evaluation",
            "no paper-level or population-level metric claim",
            "no G6 execution",
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
        ensure_unused_output_paths()
        timed(
            "validate_assets",
            state,
            results["timings_seconds"],
            lambda: require_paths(manifest),
        )
        data_view_root = timed(
            "prepare_local_detection_view",
            state,
            results["timings_seconds"],
            lambda: prepare_detection_view(manifest),
        )

        def import_framework() -> tuple[Any, ...]:
            import torch
            from PIL import Image, ImageDraw
            from data.soccernet_gsr_detection import build_gsr_detection_dataset
            from data.video_caption import (
                build_transforms as build_caption_transforms,
                keywords_list,
                read_frames_decord,
            )
            from models.build import build_metrics_fn
            from models.multi_task import MultiTaskingSigLIP

            return (
                torch,
                Image,
                ImageDraw,
                build_gsr_detection_dataset,
                build_caption_transforms,
                keywords_list,
                read_frames_decord,
                build_metrics_fn,
                MultiTaskingSigLIP,
            )

        (
            torch,
            image_class,
            image_draw,
            build_detection_dataset,
            build_caption_transforms,
            keywords,
            read_frames,
            build_metrics_fn,
            model_class,
        ) = timed(
            "import_framework",
            state,
            results["timings_seconds"],
            import_framework,
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != VISIBLE_PHYSICAL_GPU:
            raise RuntimeError(
                "G5 requires external CUDA_VISIBLE_DEVICES=7; got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one visible CUDA device for G5; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        device = torch.device(DEVICE)
        results["environment"] = {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
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
            lambda: load_config(data_view_root),
        )
        results["resolved_config"] = config
        caption_records = timed(
            "validate_caption_records",
            state,
            results["timings_seconds"],
            lambda: prepare_caption_records(manifest, keywords),
        )
        detection_dataset = timed(
            "build_fixed_detection_dataset",
            state,
            results["timings_seconds"],
            lambda: build_detection_dataset(config=config, split="test"),
        )
        detection_indices = select_detection_indices(detection_dataset, manifest)
        caption_transform = build_caption_transforms(config, split="test")
        results["assets"] = {
            "config": str(CONFIG_PATH),
            "model": str(MODEL_PATH),
            "checkpoint": str(CHECKPOINT_DIR),
            "checkpoint_type": CKPT_TYPE,
            "load_heads": LOAD_HEADS,
            "detection_data_view": str(data_view_root),
            "detection_dataset_length": len(detection_dataset),
            "detection_dataset_indices": detection_indices,
            "detection_clips": manifest["detection"]["clips"],
            "caption_sample_count": len(caption_records),
            "caption_total_video_bytes": sum(
                item["video_bytes"] for item in manifest["caption"]["samples"]
            ),
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
        require_free_memory(
            torch,
            "before_fixed_evaluation",
            MIN_FREE_BEFORE_FORWARD_GIB,
            results["memory_snapshots"],
        )
        torch.cuda.reset_peak_memory_stats(device)

        comparisons = []
        for pass_index in range(1, EVALUATION_PASSES + 1):
            pass_result, comparison = timed(
                f"evaluation_pass_{pass_index}",
                state,
                results["timings_seconds"],
                lambda current=pass_index: run_evaluation_pass(
                    current,
                    torch,
                    image_class,
                    image_draw,
                    model,
                    config,
                    manifest,
                    detection_dataset,
                    detection_indices,
                    caption_records,
                    caption_transform,
                    read_frames,
                    keywords,
                    build_metrics_fn,
                    state,
                ),
            )
            results["passes"].append(pass_result)
            comparisons.append(comparison)
            results["memory_snapshots"].append(
                memory_snapshot(torch, f"after_evaluation_pass_{pass_index}")
            )
        results["repeatability"] = timed(
            "compare_evaluation_passes",
            state,
            results["timings_seconds"],
            lambda: compare_passes(torch, comparisons[0], comparisons[1]),
        )
        results["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILURE] {results['error']}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        results["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
        results["peak_cpu_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        stop_event.set()
        heartbeat_thread.join(timeout=1)
        state.set("finished")
        print("[G5_RESULT_BEGIN]", flush=True)
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
        print("[G5_RESULT_END]", flush=True)
        try:
            with RESULT_PATH.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(results, indent=2, sort_keys=True) + "\n")
            print(f"[ARTIFACT] result_json={RESULT_PATH}", flush=True)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            exit_code = 1
        print(f"[EXIT] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
