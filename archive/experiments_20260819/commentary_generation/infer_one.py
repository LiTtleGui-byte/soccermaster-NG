#!/usr/bin/env python3
"""Run one fixed MatchTime commentary-generation sample after explicit GPU approval.

This entry is intentionally fixed to test sample 0 and batch size 1. It does not
create a DataLoader, optimizer, or scheduler, and it never performs training or
backward. The caller must expose exactly one approved physical GPU through
CUDA_VISIBLE_DEVICES; inside this process that device is always addressed as
cuda:0.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import threading
import time
import traceback
from typing import Any


# These must be set before importing torch, transformers, or local runtime modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
from runtime.paths import (  # noqa: E402
    BERT_ROOT,
    GENERATION_CHECKPOINT,
    LLAMA_ROOT,
    SIGLIP2_ROOT,
    TEST_ANNOTATIONS,
    VISUAL_BACKBONE,
    WORD_WORLD,
)
TEST_VIDEO_ROOT = Path(
    "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/"
    "SoccerNetv2/MatchTime/SN-Caption-test-align"
)
EXPECTED_RELATIVE_VIDEO = Path(
    "europe_uefa-champions-league_2016-2017/"
    "2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG/2_43_36.mp4"
)
EXPECTED_REFERENCE = (
    "[PLAYER] ([TEAM]) gets on the end of a pass on the edge of the box but "
    "his shot is blocked."
)
EXPECTED_VIDEO = TEST_VIDEO_ROOT / EXPECTED_RELATIVE_VIDEO
OUTPUT_JSON = REPO / "reports/audits/commentary_infer_one_result.json"

SAMPLE_INDEX = 0
SEED = 42
NUM_FRAMES = 30
SAMPLE_MODE = "middle"
EXPECTED_DATASET_LENGTH = 3_256
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_TEST_ANNOTATIONS_SIZE = 1_541_678
EXPECTED_VIDEO_SIZE = 5_167_597
EXPECTED_EPOCH = 11
EXPECTED_MODEL_PARAMETER_COUNT = 8_418_890_760
EXPECTED_MODEL_STATE_KEYS = 953
MIN_AVAILABLE_CPU_MEMORY_BYTES = 64 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 40 * 1024**3

GENERATION_SETTINGS = {
    "max_new_tokens": 128,
    "num_beams": 5,
    "do_sample": True,
    "min_length": 5,
    "top_p": 0.9,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1,
    "renormalize_logits": True,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def require_single_visible_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set explicitly")
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "Exactly one approved physical GPU must be exposed through "
            f"CUDA_VISIBLE_DEVICES; got {value!r}"
        )
    return devices[0]


def require_assets() -> dict[str, Any]:
    require_file(LOCAL_PYTHON)
    require_directory(REPO / "experiments/commentary_generation/runtime")
    require_directory(LLAMA_ROOT)
    require_file(LLAMA_ROOT / "config.json")
    require_file(LLAMA_ROOT / "tokenizer.json")
    require_file(LLAMA_ROOT / "model.safetensors.index.json")
    require_directory(BERT_ROOT)
    require_file(BERT_ROOT / "config.json")
    require_directory(SIGLIP2_ROOT)
    require_file(SIGLIP2_ROOT / "config.json")
    require_file(SIGLIP2_ROOT / "preprocessor_config.json")
    require_file(SIGLIP2_ROOT / "model.safetensors")
    require_file(VISUAL_BACKBONE, EXPECTED_VISUAL_BACKBONE_SIZE)
    require_file(GENERATION_CHECKPOINT, EXPECTED_CHECKPOINT_SIZE)
    require_file(WORD_WORLD)
    require_file(TEST_ANNOTATIONS, EXPECTED_TEST_ANNOTATIONS_SIZE)
    require_directory(TEST_VIDEO_ROOT)
    require_file(EXPECTED_VIDEO, EXPECTED_VIDEO_SIZE)
    require_directory(OUTPUT_JSON.parent)
    if OUTPUT_JSON.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing inference result: {OUTPUT_JSON}"
        )

    memory_available = available_memory_bytes()
    if memory_available < MIN_AVAILABLE_CPU_MEMORY_BYTES:
        raise RuntimeError(
            f"At least {MIN_AVAILABLE_CPU_MEMORY_BYTES} available CPU bytes required; "
            f"found {memory_available}"
        )
    return {
        "generation_checkpoint_bytes": GENERATION_CHECKPOINT.stat().st_size,
        "visual_backbone_bytes": VISUAL_BACKBONE.stat().st_size,
        "test_annotations_bytes": TEST_ANNOTATIONS.stat().st_size,
        "test_annotations_sha256": sha256(TEST_ANNOTATIONS),
        "video_bytes": EXPECTED_VIDEO.stat().st_size,
        "video_sha256": sha256(EXPECTED_VIDEO),
        "memory_available_bytes": memory_available,
    }


def write_result_exclusive(result: dict[str, Any]) -> None:
    with OUTPUT_JSON.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run() -> int:
    monitor = Monitor()
    monitor.start()
    exit_code = 1
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    result: dict[str, Any] = {
        "status": "failed",
        "sample_index": SAMPLE_INDEX,
        "seed": SEED,
        "batch_size": 1,
        "num_frames": NUM_FRAMES,
        "sample_mode": SAMPLE_MODE,
        "cuda_visible_devices": physical_gpu,
        "logical_device": "cuda:0",
        "offline": True,
        "checkpoint_loaded": False,
        "forward_started": False,
        "forward_completed": False,
        "generate_completed": False,
        "training_executed": False,
        "generation_settings": GENERATION_SETTINGS,
        "output_json": str(OUTPUT_JSON),
    }
    torch = None
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(
                f"Wrong Python: expected {LOCAL_PYTHON}, got {sys.executable}"
            )
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)

        monitor.enter("asset_preflight")
        result["physical_gpu"] = require_single_visible_gpu()
        result["assets"] = require_assets()

        monitor.enter("import_framework")
        import numpy as np
        import torch as torch_module

        torch = torch_module
        torch.set_num_threads(1)
        from research.experiments.commentary_generation.runtime.dataset.commentary import (
            MatchVisionCommentary_new_benchmark_from_npy_Dataset,
        )
        from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
            matchvoice_model_all_blocks,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one visible CUDA device; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
        free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info(device)
        if free_gpu_bytes < MIN_FREE_GPU_MEMORY_BYTES:
            raise RuntimeError(
                f"At least {MIN_FREE_GPU_MEMORY_BYTES} free GPU bytes required; "
                f"found {free_gpu_bytes}"
            )
        result["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_free_bytes_before": free_gpu_bytes,
            "gpu_total_bytes": total_gpu_bytes,
        }

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)

        monitor.enter("construct_dataset_metadata")
        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=[str(TEST_ANNOTATIONS)],
            video_base_dir=[str(TEST_VIDEO_ROOT)],
            num_frames=NUM_FRAMES,
            sample=SAMPLE_MODE,
            tokenizer_name=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
        )
        if len(dataset) != EXPECTED_DATASET_LENGTH:
            raise RuntimeError(
                f"Expected {EXPECTED_DATASET_LENGTH} test records, got {len(dataset)}"
            )
        expected_token_ids = {
            "[PLAYER]": 128256,
            "[TEAM]": 128257,
            "[COACH]": 128258,
            "[REFEREE]": 128259,
            "([TEAM])": 128260,
        }
        actual_token_ids = {
            token: dataset.tokenizer.convert_tokens_to_ids(token)
            for token in expected_token_ids
        }
        if len(dataset.tokenizer) != 128261 or actual_token_ids != expected_token_ids:
            raise RuntimeError(
                "Tokenizer contract mismatch: "
                f"length={len(dataset.tokenizer)} ids={actual_token_ids}"
            )
        if dataset.tokenizer.pad_token_id != 128001:
            raise RuntimeError(
                f"Expected dataset padding token 128001, got "
                f"{dataset.tokenizer.pad_token_id}"
            )

        monitor.enter("decode_and_preprocess_one_video")
        item = dataset[SAMPLE_INDEX]
        if Path(item["video_path"]) != EXPECTED_VIDEO:
            raise RuntimeError(
                f"Unexpected sample path: expected {EXPECTED_VIDEO}, "
                f"got {item['video_path']}"
            )
        if item["caption_text"] != EXPECTED_REFERENCE:
            raise RuntimeError(
                f"Unexpected reference text: {item['caption_text']!r}"
            )
        if tuple(item["frames"].shape) != (3, 30, 512, 512):
            raise RuntimeError(
                f"Expected frame shape (3, 30, 512, 512), "
                f"got {tuple(item['frames'].shape)}"
            )
        if not torch.isfinite(item["frames"]).all().item():
            raise RuntimeError("Preprocessed frames contain non-finite values")
        samples = dataset.collater([item])
        if tuple(samples["frames"].shape) != (1, 3, 30, 512, 512):
            raise RuntimeError(
                f"Expected batch frame shape (1, 3, 30, 512, 512), "
                f"got {tuple(samples['frames'].shape)}"
            )
        result["input"] = {
            "video_path": item["video_path"],
            "reference_commentary": item["caption_text"],
            "frames_shape": list(samples["frames"].shape),
            "frames_dtype": str(samples["frames"].dtype),
            "frames_min": float(samples["frames"].min().item()),
            "frames_max": float(samples["frames"].max().item()),
            "tokenizer_length": len(dataset.tokenizer),
            "added_token_ids": actual_token_ids,
        }

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
        model_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        model_state_keys = len(model.state_dict())
        if model_parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL_PARAMETER_COUNT} model parameters, "
                f"got {model_parameter_count}"
            )
        if model_state_keys != EXPECTED_MODEL_STATE_KEYS:
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL_STATE_KEYS} model state keys, "
                f"got {model_state_keys}"
            )
        result["model_parameter_count"] = model_parameter_count
        result["model_state_key_count"] = model_state_keys

        monitor.enter("load_generation_checkpoint_cpu")
        checkpoint = torch.load(
            GENERATION_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError("Expected checkpoint dict containing state_dict")
        if checkpoint.get("epoch") != EXPECTED_EPOCH:
            raise RuntimeError(
                f"Expected checkpoint epoch {EXPECTED_EPOCH}, "
                f"got {checkpoint.get('epoch')}"
            )
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint['state_dict'] is not a dict")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        result["missing_keys"] = missing_keys
        result["unexpected_keys"] = unexpected_keys
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"State dict mismatch: missing={missing_keys}, "
                f"unexpected={unexpected_keys}"
            )
        result["checkpoint_epoch"] = checkpoint["epoch"]
        result["checkpoint_state_key_count"] = len(state_dict)
        result["checkpoint_loaded"] = True
        del state_dict
        del checkpoint
        gc.collect()

        monitor.enter("move_model_and_sample_to_gpu")
        model = model.to(device)
        model.eval()
        for key, value in samples.items():
            if isinstance(value, torch.Tensor):
                samples[key] = value.to(device)
        torch.cuda.synchronize(device)

        monitor.enter("forward_and_generate_once")
        result["forward_started"] = True
        with torch.no_grad():
            generated_texts, ground_truths, video_paths = model(samples, True)
        torch.cuda.synchronize(device)
        result["forward_completed"] = True
        result["generate_completed"] = True

        if not isinstance(generated_texts, list) or len(generated_texts) != 1:
            raise RuntimeError(f"Unexpected generated output: {generated_texts!r}")
        generated_text = generated_texts[0]
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError(f"Generated commentary is empty: {generated_text!r}")
        if "<|end_of_text|>" in generated_text:
            raise RuntimeError("Decoded output still contains <|end_of_text|>")
        if ground_truths != [EXPECTED_REFERENCE]:
            raise RuntimeError(f"Unexpected ground truths: {ground_truths!r}")
        if [Path(value) for value in video_paths] != [EXPECTED_VIDEO]:
            raise RuntimeError(f"Unexpected returned video paths: {video_paths!r}")

        result["prediction"] = {
            "video_path": str(EXPECTED_VIDEO),
            "reference_commentary": EXPECTED_REFERENCE,
            "generated_commentary": generated_text,
        }
        result["gpu_peak_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )
        result["gpu_peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
            device
        )
        result["status"] = "passed"
        exit_code = 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        monitor.finish()
        result["timings_seconds"] = monitor.timings
        result["memory_samples"] = monitor.memory
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["elapsed_seconds"] = round(
            time.monotonic() - monitor.started,
            3,
        )
        if torch is not None and torch.cuda.is_available():
            result.setdefault(
                "gpu_peak_memory_allocated_bytes",
                torch.cuda.max_memory_allocated(0),
            )
            result.setdefault(
                "gpu_peak_memory_reserved_bytes",
                torch.cuda.max_memory_reserved(0),
            )
        if exit_code == 0:
            try:
                write_result_exclusive(result)
            except BaseException as error:
                exit_code = 1
                result["status"] = "failed"
                result["error_type"] = type(error).__name__
                result["error"] = str(error)
                traceback.print_exc()
        print("[RESULT]", flush=True)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
