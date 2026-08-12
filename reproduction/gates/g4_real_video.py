#!/usr/bin/env python3
"""Run one controlled G4 inference on a fixed real soccer video."""

from __future__ import annotations

import hashlib
import json
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
DATA_ROOT = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/datasets/VideoCaption"
)
LABEL_PATH = (
    DATA_ROOT
    / "video_clip_json/SoccerReplay-1988/classification_test.json"
)
LABEL_INDEX = 0
EXPECTED_RELATIVE_VIDEO = (
    "italy_serie-a_2018-2019/"
    "2019-03-31_fc-internazionale-ss-lazio-serie-a-2018-2019/"
    "2_49_08.mp4"
)
VIDEO_PATH = (
    DATA_ROOT
    / "video_clip/SoccerReplay-1988-high-resolution"
    / EXPECTED_RELATIVE_VIDEO
)
EXPECTED_VIDEO_SIZE = 6_144_860
EXPECTED_CAPTION = "end of half game"
EXPECTED_TEXT_LENGTH = 67
EXPECTED_TEXT_SHA256 = (
    "632ebad71478a292d9589277be40b1e353052d3b4b2e82e7144d2b9168c09aa8"
)

REPORT_DIR = REPO / "reports/g4"
RESULT_PATH = REPORT_DIR / "g4_real_video_result_20260812.json"
CONTACT_SHEET_PATH = REPORT_DIR / "g4_real_video_contact_sheet_20260812.jpg"

DEVICE = "cuda:0"
CONFIG_DEVICE = "cpu"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
SEED = 42
NUM_FRAMES = 30
INPUT_SHAPE = (1, 30, 3, 512, 512)
HEARTBEAT_SECONDS = 30
GIB = 1024**3
MIN_FREE_BEFORE_LOAD_GIB = 30
MIN_FREE_BEFORE_FORWARD_GIB = 20

EXPECTED_FILES = (
    "backbone.pt",
    "SoccerNetGSR_Detection.pt",
    "LinesDetection.pt",
    "KeypointsDetection.pt",
    "VideoCaption.pt",
    "CaptionClassification.pt",
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


def heartbeat(stop_event: threading.Event, state: PhaseState) -> None:
    started = time.monotonic()
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed = round(time.monotonic() - started, 1)
        print(
            f"[HEARTBEAT] phase={state.get()} elapsed={elapsed}s",
            flush=True,
        )


def require_paths() -> None:
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
        LABEL_PATH,
        VIDEO_PATH,
        CHECKPOINT_DIR / "text_model/model.safetensors",
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required G4 paths are missing:\n" + "\n".join(missing)
        )
    occupied = [
        str(path)
        for path in (RESULT_PATH, CONTACT_SHEET_PATH)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite G4 artifacts:\n" + "\n".join(occupied)
        )
    if VIDEO_PATH.stat().st_size != EXPECTED_VIDEO_SIZE:
        raise AssertionError(
            f"Video size {VIDEO_PATH.stat().st_size} != {EXPECTED_VIDEO_SIZE}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_label() -> dict[str, Any]:
    with LABEL_PATH.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    item = labels[LABEL_INDEX]
    if item.get("video") != EXPECTED_RELATIVE_VIDEO:
        raise AssertionError(
            f"Label video {item.get('video')!r} != {EXPECTED_RELATIVE_VIDEO!r}"
        )
    if item.get("caption") != EXPECTED_CAPTION:
        raise AssertionError(
            f"Caption {item.get('caption')!r} != {EXPECTED_CAPTION!r}"
        )
    text = item.get("comments_text_anonymized")
    if not isinstance(text, str) or len(text) != EXPECTED_TEXT_LENGTH:
        raise AssertionError(
            f"Anonymized text length is {len(text) if isinstance(text, str) else None}"
        )
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash != EXPECTED_TEXT_SHA256:
        raise AssertionError(f"Anonymized text SHA256 is {text_hash}")
    return item


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
    return config


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
            f"at least {minimum_gib} GiB is required."
        )


def tensor_summary(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        finite = True
        if value.is_floating_point() or value.is_complex():
            finite = bool(torch.isfinite(value).all().item())
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": finite,
        }
    if isinstance(value, dict):
        return {key: tensor_summary(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(torch, item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def assert_shape(tensor: Any, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} shape {actual} != {expected}")


def validate_caption_outputs(
    torch: Any,
    outputs: dict[str, Any],
    keywords: list[str],
    expected_caption: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_heads = {"VideoCaption", "CaptionClassification"}
    if set(outputs) != expected_heads:
        raise AssertionError(f"Caption heads {set(outputs)} != {expected_heads}")

    video_caption = outputs["VideoCaption"]
    expected_video_keys = {
        "vision_features",
        "text_features",
        "base_similarity_matrix",
        "processed_similarity_matrix",
        "valid_text_mask",
    }
    if set(video_caption) != expected_video_keys:
        raise AssertionError(
            f"VideoCaption keys {set(video_caption)} != {expected_video_keys}"
        )
    assert_shape(video_caption["vision_features"], (1, 1024), "vision_features")
    assert_shape(video_caption["text_features"], (1, 1024), "text_features")
    assert_shape(
        video_caption["base_similarity_matrix"],
        (1, 1),
        "base_similarity_matrix",
    )
    assert_shape(
        video_caption["processed_similarity_matrix"],
        (1, 1),
        "processed_similarity_matrix",
    )
    assert_shape(video_caption["valid_text_mask"], (1,), "valid_text_mask")
    if not bool(video_caption["valid_text_mask"].all().item()):
        raise AssertionError("Caption text was not marked valid")

    classification = outputs["CaptionClassification"]
    if set(classification) != {"logits", "features"}:
        raise AssertionError(
            f"CaptionClassification keys are {set(classification)}"
        )
    logits = classification["logits"]
    assert_shape(logits, (1, 23), "caption logits")
    assert_shape(classification["features"], (1, 1024), "caption features")

    summary = tensor_summary(torch, outputs)
    for head in summary.values():
        for value in head.values():
            if isinstance(value, dict) and value.get("type") == "tensor":
                if not value["finite"]:
                    raise AssertionError("Caption output contains non-finite values")

    probabilities = torch.softmax(logits[0].float(), dim=0)
    top_probabilities, top_indices = torch.topk(probabilities, k=5)
    top5 = [
        {
            "index": int(index),
            "label": keywords[int(index)],
            "probability": float(probability),
        }
        for probability, index in zip(top_probabilities, top_indices)
    ]
    ground_truth_index = keywords.index(expected_caption)
    ranking = torch.argsort(probabilities, descending=True).tolist()
    prediction = {
        "top5": top5,
        "ground_truth_index": ground_truth_index,
        "ground_truth_label": expected_caption,
        "ground_truth_probability": float(probabilities[ground_truth_index]),
        "ground_truth_rank": ranking.index(ground_truth_index) + 1,
        "base_similarity": float(video_caption["base_similarity_matrix"][0, 0]),
        "processed_similarity": float(
            video_caption["processed_similarity_matrix"][0, 0]
        ),
    }
    return summary, prediction


def save_contact_sheet(frames: Any, image_class: Any) -> None:
    selected = [0, 5, 10, 15, 20, 25]
    tile_size = (256, 144)
    sheet = image_class.new("RGB", (tile_size[0] * 3, tile_size[1] * 2))
    resampling = getattr(image_class, "Resampling", image_class).BILINEAR
    for position, frame_index in enumerate(selected):
        array = frames[frame_index].permute(1, 2, 0).cpu().numpy()
        tile = image_class.fromarray(array).resize(tile_size, resampling)
        sheet.paste(
            tile,
            ((position % 3) * tile_size[0], (position // 3) * tile_size[1]),
        )
    sheet.save(CONTACT_SHEET_PATH, format="JPEG", quality=92)


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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = PhaseState()
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        args=(stop_event, state),
        name="g4-real-video-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    results: dict[str, Any] = {
        "gate": "G4",
        "status": "failed",
        "error": None,
        "seed": SEED,
        "timings_seconds": {},
        "gpu": {},
        "memory_snapshots": [],
        "code": git_identity(),
        "assets": {
            "config": str(CONFIG_PATH),
            "model": str(MODEL_PATH),
            "checkpoint": str(CHECKPOINT_DIR),
            "label_file": str(LABEL_PATH),
            "label_index": LABEL_INDEX,
            "video": str(VIDEO_PATH),
        },
        "sampling": {},
        "preprocessing": {},
        "caption_outputs": None,
        "prediction": None,
        "artifacts": {
            "result_json": str(RESULT_PATH),
            "contact_sheet": str(CONTACT_SHEET_PATH),
        },
        "explicit_non_goals": [
            "no Dataset or DataLoader construction",
            "no loss, backward, optimizer, scheduler, or training",
            "no detection-head inference",
            "no aggregate evaluation metric",
            "no G5 execution",
        ],
    }
    exit_code = 1

    try:
        require_paths()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
            raise RuntimeError(
                "G4 requires external CUDA_VISIBLE_DEVICES=7; got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )

        def import_framework() -> tuple[Any, Any, Any, Any, Any, Any]:
            import torch
            from PIL import Image
            from data.video_caption import (
                build_transforms,
                keywords_list,
                read_frames_decord,
            )
            from models.multi_task import MultiTaskingSigLIP

            return (
                torch,
                Image,
                build_transforms,
                keywords_list,
                read_frames_decord,
                MultiTaskingSigLIP,
            )

        (
            torch,
            image_class,
            build_transforms,
            keywords,
            read_frames,
            model_class,
        ) = timed(
            "import_framework",
            state,
            results["timings_seconds"],
            import_framework,
        )
        random.seed(SEED)
        torch.manual_seed(SEED)

        label = timed(
            "load_and_validate_label",
            state,
            results["timings_seconds"],
            load_fixed_label,
        )
        text = label["comments_text_anonymized"]
        results["assets"].update(
            {
                "video_size_bytes": VIDEO_PATH.stat().st_size,
                "video_mtime_ns": VIDEO_PATH.stat().st_mtime_ns,
                "video_sha256": timed(
                    "hash_video",
                    state,
                    results["timings_seconds"],
                    lambda: sha256_file(VIDEO_PATH),
                ),
                "caption": label["caption"],
                "text_length": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

        config = timed(
            "load_config",
            state,
            results["timings_seconds"],
            load_config,
        )
        if config["NUM_FRAMES"] != NUM_FRAMES:
            raise AssertionError(f"NUM_FRAMES is {config['NUM_FRAMES']}")
        if config["VIDEO_CAPTION_SAMPLE"] != "middle":
            raise AssertionError(
                f"VIDEO_CAPTION_SAMPLE is {config['VIDEO_CAPTION_SAMPLE']!r}"
            )
        results["resolved_config"] = config

        frames, frame_indices, duration = timed(
            "decode_real_video",
            state,
            results["timings_seconds"],
            lambda: read_frames(
                str(VIDEO_PATH),
                NUM_FRAMES,
                config["VIDEO_CAPTION_SAMPLE"],
                config["VIDEO_CAPTION_FIX_START"],
                config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
                config["VIDEO_CAPTION_TRIMMED30"],
            ),
        )
        assert_shape(frames, (NUM_FRAMES, 3, frames.shape[-2], frames.shape[-1]), "decoded frames")
        if frames.dtype != torch.uint8 or frames.device.type != "cpu":
            raise AssertionError(
                f"Decoded frames are {frames.dtype} on {frames.device}"
            )
        if len(frame_indices) != NUM_FRAMES:
            raise AssertionError(f"Decoded {len(frame_indices)} frame indices")
        if frame_indices != sorted(frame_indices):
            raise AssertionError("Frame indices are not monotonically ordered")
        results["sampling"] = {
            "method": config["VIDEO_CAPTION_SAMPLE"],
            "frame_indices": [int(index) for index in frame_indices],
            "duration_seconds": float(duration),
            "decoded_shape": list(frames.shape),
            "decoded_dtype": str(frames.dtype),
        }
        timed(
            "save_contact_sheet",
            state,
            results["timings_seconds"],
            lambda: save_contact_sheet(frames, image_class),
        )

        transform = build_transforms(config, split="test")
        annotation = {
            "caption": label["caption"],
            "caption_index": torch.tensor(
                keywords.index(label["caption"]),
                dtype=torch.long,
            ),
            "text": text,
        }
        metas = {"task": "VideoCaption", "video": str(VIDEO_PATH)}
        images_cpu, annotation, metas = timed(
            "test_preprocessing",
            state,
            results["timings_seconds"],
            lambda: transform(frames, annotation, metas),
        )
        assert_shape(images_cpu, INPUT_SHAPE[1:], "preprocessed frames")
        if images_cpu.dtype != torch.float32 or images_cpu.device.type != "cpu":
            raise AssertionError(
                f"Preprocessed frames are {images_cpu.dtype} on {images_cpu.device}"
            )
        if not bool(torch.isfinite(images_cpu).all().item()):
            raise AssertionError("Preprocessed frames contain NaN or Inf")
        input_min = float(images_cpu.min())
        input_max = float(images_cpu.max())
        if input_min < -1.0001 or input_max > 1.0001:
            raise AssertionError(
                f"Normalized range [{input_min}, {input_max}] is outside [-1, 1]"
            )
        results["preprocessing"] = {
            "shape": list(images_cpu.shape),
            "dtype": str(images_cpu.dtype),
            "device": str(images_cpu.device),
            "finite": True,
            "minimum": input_min,
            "maximum": input_max,
            "mean": float(images_cpu.mean()),
            "original_image_size": list(metas["original_image_size"]),
            "image_size": list(metas["image_size"]),
            "scale_ratio_x": metas["scale_ratio_x"],
            "scale_ratio_y": metas["scale_ratio_y"],
            "annotation_caption_index": int(annotation["caption_index"]),
        }
        del frames

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one visible CUDA device for G4; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        device = torch.device(DEVICE)
        results["gpu"] = {
            "visible_physical_index": 7,
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
        }
        torch.cuda.manual_seed_all(SEED)
        require_free_memory(
            torch,
            "after_cuda_init",
            MIN_FREE_BEFORE_LOAD_GIB,
            results["memory_snapshots"],
        )

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
            "before_real_video_forward",
            MIN_FREE_BEFORE_FORWARD_GIB,
            results["memory_snapshots"],
        )
        images = images_cpu.unsqueeze(0).to(device)
        assert_shape(images, INPUT_SHAPE, "batched real-video input")
        torch.cuda.reset_peak_memory_stats(device)

        def caption_forward() -> Any:
            with torch.inference_mode():
                output = model(
                    images,
                    "VideoCaption",
                    metas=(metas,),
                    text=[text],
                )
            torch.cuda.synchronize(device)
            return output

        outputs = timed(
            "real_video_caption_forward",
            state,
            results["timings_seconds"],
            caption_forward,
        )
        summary, prediction = timed(
            "validate_and_summarize_outputs",
            state,
            results["timings_seconds"],
            lambda: validate_caption_outputs(
                torch,
                outputs,
                keywords,
                EXPECTED_CAPTION,
            ),
        )
        results["caption_outputs"] = summary
        results["prediction"] = prediction
        results["memory_snapshots"].append(
            memory_snapshot(torch, "after_real_video_validation")
        )
        results["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAILURE] {results['error']}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        results["total_elapsed_seconds"] = round(
            time.monotonic() - started,
            3,
        )
        results["peak_cpu_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        stop_event.set()
        heartbeat_thread.join(timeout=1)
        state.set("finished")
        print("[G4_RESULT_BEGIN]", flush=True)
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
        print("[G4_RESULT_END]", flush=True)
        try:
            RESULT_PATH.write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[ARTIFACT] result_json={RESULT_PATH}", flush=True)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            exit_code = 1
        print(f"[EXIT] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
