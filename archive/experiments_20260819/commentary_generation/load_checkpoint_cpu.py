#!/usr/bin/env python3
"""CPU-only load validation for the historical commentary checkpoint.

This script is intentionally load-only. It must not create a dataset, DataLoader,
optimizer, or scheduler, and it must not call forward, eval, generate, inference,
backward, or train.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from typing import Any


# These must be set before importing torch, transformers, or local model modules.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT = REPO / "experiments/commentary_generation"
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
from runtime.paths import (  # noqa: E402
    BERT_ROOT,
    GENERATION_CHECKPOINT,
    LLAMA_ROOT,
    SIGLIP2_ROOT,
    VISUAL_BACKBONE,
    WORD_WORLD,
)

MIN_AVAILABLE_MEMORY_BYTES = 64 * 1024**3
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_EPOCH = 11

COMPONENT_PREFIXES = {
    "llama_model": ("llama_model.",),
    "visual_encoder": ("visual_encoder.",),
    "video_Qformer": ("video_Qformer.",),
    "video_query_tokens": ("video_query_tokens",),
    "ln_vision": ("ln_vision.",),
    "llama_proj": ("llama_proj.",),
    "video_frame_position_embedding": ("video_frame_position_embedding.",),
}


class Monitor:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.stage = "preflight"
        self.stage_started = self.started
        self.timings: dict[str, float] = {}
        self.memory: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self.sample_memory("start")
        self.thread.start()

    def enter(self, stage: str) -> None:
        now = time.monotonic()
        if self.stage:
            self.timings[self.stage] = round(now - self.stage_started, 3)
        self.stage = stage
        self.stage_started = now
        self.sample_memory(f"enter:{stage}")
        print(f"[STAGE] {stage}", flush=True)

    def finish(self) -> None:
        now = time.monotonic()
        if self.stage:
            self.timings[self.stage] = round(now - self.stage_started, 3)
        self.sample_memory("finish")
        self.stop.set()
        self.thread.join(timeout=2)

    def sample_memory(self, label: str) -> dict[str, Any]:
        status: dict[str, str] = {}
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
        sample = {
            "label": label,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            **status,
        }
        self.memory.append(sample)
        return sample

    def _heartbeat(self) -> None:
        while not self.stop.wait(30):
            sample = self.sample_memory(f"heartbeat:{self.stage}")
            print(
                f"[HEARTBEAT] stage={self.stage} "
                f"elapsed={sample['elapsed_seconds']}s "
                f"rss={sample.get('VmRSS', 'unknown')} "
                f"hwm={sample.get('VmHWM', 'unknown')}",
                flush=True,
            )


def available_memory_bytes() -> int:
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def require_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: expected {expected_size}, "
            f"got {path.stat().st_size}"
        )


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise NotADirectoryError(path)


def require_assets() -> dict[str, Any]:
    require_file(LOCAL_PYTHON)
    require_directory(EXPERIMENT / "runtime")
    require_directory(LLAMA_ROOT)
    require_file(LLAMA_ROOT / "config.json")
    require_file(LLAMA_ROOT / "tokenizer.json")
    require_file(LLAMA_ROOT / "model.safetensors.index.json")
    index = json.loads(
        (LLAMA_ROOT / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 4:
        raise RuntimeError(f"Expected four Llama shards, got {shards}")
    for shard in shards:
        require_file(LLAMA_ROOT / shard)
    require_directory(BERT_ROOT)
    require_file(BERT_ROOT / "config.json")
    require_directory(SIGLIP2_ROOT)
    require_file(SIGLIP2_ROOT / "config.json")
    require_file(SIGLIP2_ROOT / "model.safetensors")
    require_file(VISUAL_BACKBONE, EXPECTED_VISUAL_BACKBONE_SIZE)
    require_file(GENERATION_CHECKPOINT, EXPECTED_CHECKPOINT_SIZE)
    require_file(WORD_WORLD)

    memory_available = available_memory_bytes()
    if memory_available < MIN_AVAILABLE_MEMORY_BYTES:
        raise RuntimeError(
            f"At least {MIN_AVAILABLE_MEMORY_BYTES} available bytes required; "
            f"found {memory_available}"
        )
    return {
        "llama_shards": [str(LLAMA_ROOT / shard) for shard in shards],
        "llama_total_tensor_bytes": index.get("metadata", {}).get("total_size"),
        "visual_backbone_bytes": VISUAL_BACKBONE.stat().st_size,
        "generation_checkpoint_bytes": GENERATION_CHECKPOINT.stat().st_size,
        "memory_available_bytes": memory_available,
    }


def component_coverage(keys: list[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for component, prefixes in COMPONENT_PREFIXES.items():
        matched = [key for key in keys if key.startswith(prefixes)]
        coverage[component] = {
            "checkpoint_present": bool(matched),
            "key_count": len(matched),
            "first_key": matched[0] if matched else None,
            "missing_keys": [],
            "unexpected_keys": [],
            "loaded": False,
        }
    return coverage


def assign_key_to_component(key: str) -> str:
    for component, prefixes in COMPONENT_PREFIXES.items():
        if key.startswith(prefixes):
            return component
    return "unclassified"


def run() -> int:
    monitor = Monitor()
    monitor.start()
    result: dict[str, Any] = {
        "status": "failed",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "offline": True,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "forward_executed": False,
        "eval_executed": False,
        "generate_executed": False,
        "training_executed": False,
    }
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(
                f"Wrong Python: expected {LOCAL_PYTHON}, got {sys.executable}"
            )
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)

        monitor.enter("asset_preflight")
        result["assets"] = require_assets()

        monitor.enter("import_framework")
        import torch

        torch.set_num_threads(1)
        from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
            matchvoice_model_all_blocks,
        )

        result["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise RuntimeError("CUDA must be unavailable in this CPU-only validation")

        monitor.enter("construct_model_and_load_base_assets")
        model = matchvoice_model_all_blocks(
            num_features=1024,
            need_temporal=True,
            need_spatial=False,
            use_local_features=False,
            open_visual_encoder=True,
            open_llm_decoder=True,
            file_path=str(WORD_WORLD),
            tokenizer_ckpt=str(LLAMA_ROOT),
            llm_ckpt=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
            visual_encoder_checkpoint=str(VISUAL_BACKBONE),
            timesformer_type="unisoccer_part_temporal",
            encoder_type="spatial_and_temporal",
            num_video_query_token=32,
            use_mlp=False,
        )
        result["model_constructed"] = True
        result["model_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        result["model_state_key_count"] = len(model.state_dict())
        monitor.sample_memory("model_constructed")

        monitor.enter("load_generation_checkpoint_cpu")
        checkpoint = torch.load(
            GENERATION_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
        if "state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain state_dict")
        epoch = checkpoint.get("epoch")
        if epoch != EXPECTED_EPOCH:
            raise RuntimeError(f"Expected epoch {EXPECTED_EPOCH}, got {epoch}")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint['state_dict'] is not a dict")
        checkpoint_keys = list(state_dict)
        coverage = component_coverage(checkpoint_keys)
        unloaded_components = [
            name
            for name, detail in coverage.items()
            if not detail["checkpoint_present"]
        ]
        if unloaded_components:
            raise RuntimeError(
                f"Checkpoint has no keys for components: {unloaded_components}"
            )
        result.update(
            checkpoint_epoch=epoch,
            checkpoint_state_key_count=len(checkpoint_keys),
            component_coverage=coverage,
        )
        monitor.sample_memory("checkpoint_deserialized")

        monitor.enter("apply_generation_state_dict")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        result["missing_keys"] = missing_keys
        result["unexpected_keys"] = unexpected_keys
        for key in missing_keys:
            component = assign_key_to_component(key)
            coverage.setdefault(
                component,
                {
                    "checkpoint_present": False,
                    "key_count": 0,
                    "first_key": None,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "loaded": False,
                },
            )["missing_keys"].append(key)
        for key in unexpected_keys:
            component = assign_key_to_component(key)
            coverage.setdefault(
                component,
                {
                    "checkpoint_present": False,
                    "key_count": 0,
                    "first_key": None,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "loaded": False,
                },
            )["unexpected_keys"].append(key)
        for detail in coverage.values():
            detail["loaded"] = (
                detail["checkpoint_present"]
                and not detail["missing_keys"]
                and not detail["unexpected_keys"]
            )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"State dict mismatch: missing={missing_keys}, "
                f"unexpected={unexpected_keys}"
            )
        result["checkpoint_loaded"] = True
        monitor.sample_memory("state_dict_applied")

        monitor.enter("release_deserialized_checkpoint")
        del state_dict
        del checkpoint
        gc.collect()
        monitor.sample_memory("checkpoint_released")

        result["status"] = "passed"
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
        return 1
    finally:
        monitor.finish()
        result["timings_seconds"] = monitor.timings
        result["memory_samples"] = monitor.memory
        result["peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        result["elapsed_seconds"] = round(time.monotonic() - monitor.started, 3)
        print("[RESULT]", flush=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
