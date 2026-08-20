#!/usr/bin/env python3
"""Run one controlled G3 random-tensor forward on a single visible GPU."""

from __future__ import annotations

import json
import os
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

DEVICE = "cuda:0"
CONFIG_DEVICE = "cpu"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
SEED = 42
INPUT_SHAPE = (1, 30, 3, 512, 512)
INPUT_DTYPE = "float32"
CAPTION_TEXT = ["a soccer match"]
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
        CHECKPOINT_DIR / "text_model/model.safetensors",
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required G3 paths are missing:\n" + "\n".join(missing)
        )


def load_config() -> dict[str, Any]:
    from configs.util import load_super_config, yaml_to_dict

    experiment_config = yaml_to_dict(str(CONFIG_PATH))
    config = load_super_config(experiment_config, str(DEFAULT_CONFIG_PATH))
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            # Construct and load on CPU, then move the complete model explicitly.
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


def assert_all_finite(summary: Any, path: str = "outputs") -> None:
    if isinstance(summary, dict):
        if summary.get("type") == "tensor":
            if not summary["finite"]:
                raise AssertionError(f"Non-finite tensor at {path}")
            return
        for key, item in summary.items():
            assert_all_finite(item, f"{path}.{key}")
    elif isinstance(summary, list):
        for index, item in enumerate(summary):
            assert_all_finite(item, f"{path}[{index}]")


def assert_shape(tensor: Any, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name} shape {actual} != {expected}")


def validate_detection_outputs(torch: Any, outputs: dict[str, Any]) -> Any:
    expected_heads = {
        "SoccerNetGSR_Detection",
        "LinesDetection",
        "KeypointsDetection",
    }
    if set(outputs) != expected_heads:
        raise AssertionError(f"Detection heads {set(outputs)} != {expected_heads}")

    detection = outputs["SoccerNetGSR_Detection"]
    expected_detection_keys = {
        "pred_logits",
        "pred_boxes",
        "pred_roles",
        "pred_jn_holistic",
        "pred_digit_head",
        "pred_digit_tail",
        "aux_outputs",
        "outputs",
    }
    if set(detection) != expected_detection_keys:
        raise AssertionError(
            f"Detection keys {set(detection)} != {expected_detection_keys}"
        )
    assert_shape(detection["pred_logits"], (1, 30, 300, 1), "pred_logits")
    assert_shape(detection["pred_boxes"], (1, 30, 300, 4), "pred_boxes")
    assert_shape(detection["pred_roles"], (1, 30, 300, 6), "pred_roles")
    assert_shape(
        detection["pred_jn_holistic"],
        (1, 30, 300, 101),
        "pred_jn_holistic",
    )
    assert_shape(
        detection["pred_digit_head"],
        (1, 30, 300, 10),
        "pred_digit_head",
    )
    assert_shape(
        detection["pred_digit_tail"],
        (1, 30, 300, 11),
        "pred_digit_tail",
    )
    assert_shape(detection["outputs"], (1, 30, 300, 256), "outputs")
    if detection["aux_outputs"] != []:
        raise AssertionError("Expected no auxiliary outputs for one decoder layer")

    assert_shape(
        outputs["LinesDetection"]["pred_lines_heatmap"],
        (1, 30, 24, 256, 256),
        "pred_lines_heatmap",
    )
    assert_shape(
        outputs["KeypointsDetection"]["pred_keypoints_heatmap"],
        (1, 30, 58, 256, 256),
        "pred_keypoints_heatmap",
    )
    summary = tensor_summary(torch, outputs)
    assert_all_finite(summary)
    return summary


def validate_caption_outputs(torch: Any, outputs: dict[str, Any]) -> Any:
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
    assert_shape(classification["logits"], (1, 23), "caption logits")
    assert_shape(classification["features"], (1, 1024), "caption features")
    summary = tensor_summary(torch, outputs)
    assert_all_finite(summary)
    return summary


def main() -> int:
    started = time.monotonic()
    state = PhaseState()
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        args=(stop_event, state),
        name="g3-random-forward-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    results: dict[str, Any] = {
        "status": "failed",
        "error": None,
        "seed": SEED,
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": INPUT_DTYPE,
        "caption_text_count": len(CAPTION_TEXT),
        "timings_seconds": {},
        "gpu": {},
        "memory_snapshots": [],
        "detection_outputs": None,
        "caption_outputs": None,
    }
    exit_code = 1

    try:
        require_paths()

        def import_framework() -> tuple[Any, Any]:
            import torch
            from models.multi_task import MultiTaskingSigLIP

            return torch, MultiTaskingSigLIP

        torch, model_class = timed(
            "import_framework",
            state,
            results["timings_seconds"],
            import_framework,
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
            raise RuntimeError(
                "G3 requires external CUDA_VISIBLE_DEVICES=7; got "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one visible CUDA device for G3; "
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
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
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

        model = timed(
            "construct_model_cpu",
            state,
            results["timings_seconds"],
            lambda: model_class(config=config, logger=None),
        )

        def load_checkpoint() -> None:
            model.load_checkpoint(
                str(CHECKPOINT_DIR),
                ckpt_type=CKPT_TYPE,
                logger=None,
                load_heads=LOAD_HEADS,
            )

        timed(
            "load_checkpoint_cpu",
            state,
            results["timings_seconds"],
            load_checkpoint,
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
            "before_detection_forward",
            MIN_FREE_BEFORE_FORWARD_GIB,
            results["memory_snapshots"],
        )

        images = timed(
            "create_random_input",
            state,
            results["timings_seconds"],
            lambda: torch.rand(INPUT_SHAPE, dtype=torch.float32, device=device)
            .mul_(2.0)
            .sub_(1.0),
        )
        assert_shape(images, INPUT_SHAPE, "random input")

        torch.cuda.reset_peak_memory_stats(device)

        def detection_forward() -> Any:
            with torch.inference_mode():
                output = model(
                    images,
                    "SoccerNetGSR_Detection",
                    metas=None,
                    text=None,
                )
            torch.cuda.synchronize(device)
            return output

        detection_outputs = timed(
            "detection_forward",
            state,
            results["timings_seconds"],
            detection_forward,
        )
        results["detection_outputs"] = timed(
            "validate_detection_outputs",
            state,
            results["timings_seconds"],
            lambda: validate_detection_outputs(torch, detection_outputs),
        )
        results["memory_snapshots"].append(
            memory_snapshot(torch, "after_detection_validation")
        )
        del detection_outputs
        torch.cuda.empty_cache()
        require_free_memory(
            torch,
            "before_caption_forward",
            MIN_FREE_BEFORE_FORWARD_GIB,
            results["memory_snapshots"],
        )
        torch.cuda.reset_peak_memory_stats(device)

        def caption_forward() -> Any:
            with torch.inference_mode():
                output = model(
                    images,
                    "VideoCaption",
                    metas=None,
                    text=CAPTION_TEXT,
                )
            torch.cuda.synchronize(device)
            return output

        caption_outputs = timed(
            "caption_forward",
            state,
            results["timings_seconds"],
            caption_forward,
        )
        results["caption_outputs"] = timed(
            "validate_caption_outputs",
            state,
            results["timings_seconds"],
            lambda: validate_caption_outputs(torch, caption_outputs),
        )
        results["memory_snapshots"].append(
            memory_snapshot(torch, "after_caption_validation")
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
        stop_event.set()
        heartbeat_thread.join(timeout=1)
        state.set("finished")
        print("[G3_RESULT_BEGIN]", flush=True)
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
        print("[G3_RESULT_END]", flush=True)
        print(f"[EXIT] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
