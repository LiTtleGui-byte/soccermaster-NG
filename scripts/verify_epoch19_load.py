#!/usr/bin/env python3
"""Safely verify the epoch-19 checkpoint's seven load_state_dict results.

This script only constructs the model and loads its checkpoint.  It does not
create data pipelines or training objects, and it never calls the model.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any


# These settings must be in place before any project module is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cpu"
CKPT_TYPE = "soccer_master"
LOAD_HEADS = True
HEARTBEAT_SECONDS = 30

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

EXPECTED_FILES = (
    "backbone.pt",
    "SoccerNetGSR_Detection.pt",
    "LinesDetection.pt",
    "KeypointsDetection.pt",
    "VideoCaption.pt",
    "CaptionClassification.pt",
)
TEXT_MODEL_FILE = CHECKPOINT_DIR / "text_model/model.safetensors"

HEAD_NAMES = (
    "SoccerNetGSR_Detection",
    "LinesDetection",
    "KeypointsDetection",
    "VideoCaption",
    "CaptionClassification",
)
COMPONENT_NAMES = ("backbone", "text_model", *HEAD_NAMES)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OPS_BUILD))
os.chdir(REPO)


def heartbeat(stop_event: threading.Event) -> None:
    """Print progress every 30 seconds without touching the model."""
    started = time.monotonic()
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed = int(time.monotonic() - started)
        print(f"[heartbeat] still running after {elapsed}s", flush=True)


def require_paths() -> None:
    """Fail before model construction if any required local input is absent."""
    required = [
        REPO,
        OPS_BUILD,
        CONFIG_PATH,
        DEFAULT_CONFIG_PATH,
        MODEL_PATH,
        CHECKPOINT_DIR,
    ]
    required.extend(CHECKPOINT_DIR / name for name in EXPECTED_FILES)
    required.append(TEXT_MODEL_FILE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required offline paths are missing:\n" + "\n".join(missing)
        )


def load_config() -> dict[str, Any]:
    """Load YAML configuration only; no datasets or training objects are built."""
    from configs.util import load_super_config, yaml_to_dict

    experiment_config = yaml_to_dict(str(CONFIG_PATH))
    config = load_super_config(experiment_config, str(DEFAULT_CONFIG_PATH))
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            "DEVICE": DEVICE,
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
                "VideoCaption": ["VideoCaption", "CaptionClassification"],
            },
        }
    )
    return config


def instrument_load_state_dict(
    component_name: str,
    module: Any,
    results: dict[str, dict[str, Any]],
) -> None:
    """Record missing/unexpected keys and make all seven checks non-strict."""
    original = module.load_state_dict
    accepts_assign = "assign" in inspect.signature(original).parameters

    def wrapped(
        self: Any,
        state_dict: Any,
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        del self, strict
        try:
            if accepts_assign:
                outcome = original(state_dict, strict=False, assign=assign)
            else:
                outcome = original(state_dict, strict=False)
        except Exception as exc:
            results[component_name]["error"] = f"{type(exc).__name__}: {exc}"
            raise

        results[component_name].update(
            {
                "loaded": True,
                "missing_keys": list(outcome.missing_keys),
                "unexpected_keys": list(outcome.unexpected_keys),
            }
        )
        return outcome

    module.load_state_dict = types.MethodType(wrapped, module)


def install_instrumentation(
    model: Any,
    results: dict[str, dict[str, Any]],
) -> None:
    """Attach result capture to the backbone, text model, and five heads."""
    components = {
        "backbone": model.backbone.vision_model,
        "text_model": model.backbone.text_model.model,
    }
    components.update({name: model.multi_task_head[name] for name in HEAD_NAMES})
    if tuple(components) != COMPONENT_NAMES:
        raise RuntimeError(f"Unexpected component order: {tuple(components)}")
    for name, module in components.items():
        instrument_load_state_dict(name, module, results)


def main() -> int:
    results = {
        name: {
            "loaded": False,
            "missing_keys": None,
            "unexpected_keys": None,
            "error": None,
        }
        for name in COMPONENT_NAMES
    }
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat,
        args=(stop_event,),
        name="checkpoint-load-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    exit_code = 0
    try:
        require_paths()
        config = load_config()

        # Importing this project class transitively imports its tensor framework.
        from models.multi_task import MultiTaskingSigLIP

        model = MultiTaskingSigLIP(config=config, logger=None)
        install_instrumentation(model, results)
        model.load_checkpoint(
            str(CHECKPOINT_DIR),
            ckpt_type=CKPT_TYPE,
            logger=None,
            load_heads=LOAD_HEADS,
        )
    except BaseException as exc:
        exit_code = 1
        print(f"[failure] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1)
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)

    if exit_code == 0 and not all(item["loaded"] for item in results.values()):
        print("[failure] Not all seven components were loaded.", file=sys.stderr)
        exit_code = 1
    print(f"[exit] exit_code={exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
