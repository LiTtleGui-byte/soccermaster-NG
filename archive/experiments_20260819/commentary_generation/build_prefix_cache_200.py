#!/usr/bin/env python3
"""Build a fixed 200-sample projected visual-prefix cache on one approved GPU.

The cache contains only the projected [32, 4096] float32 visual prefix for each
fixed MatchTime sample plus its dataset index. This script does not generate
text and never creates a DataLoader, optimizer, scheduler, or training step.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


# These are fixed before torch, transformers, or local runtime imports.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
LLAMA_ROOT = Path("/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct")
BERT_ROOT = Path("/remote-home/share/huggingface/bert-base-uncased")
SIGLIP2_ROOT = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
VISUAL_BACKBONE = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000/epoch_19/backbone.pt"
)
GENERATION_CHECKPOINT = Path(
    "/remote-home/haolinyang/sports/dirty_code/UniSoccer/output/"
    "large_512_multitask_w_1_epoch_19_train_matchtime_eval_matchtime_"
    "half_lr_bf16/model_save_11.pth"
)
WORD_WORLD = Path(
    "/remote-home/haolinyang/sports/UniSoccer/words_world/match_time.pkl"
)
TEST_ANNOTATIONS = Path(
    "/remote-home/haolinyang/sports/UniSoccer/train_data/"
    "video_clip_json/MatchTime/classification_test.json"
)
TEST_VIDEO_ROOT = Path(
    "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/"
    "SoccerNetv2/MatchTime/SN-Caption-test-align"
)
SOURCE_RESULT = (
    REPO / "reports/commentary_decode_ablation_200_20260814/result.json"
)
SOURCE_PREDICTIONS = (
    REPO
    / "reports/commentary_decode_ablation_200_20260814/predictions.jsonl"
)
OUTPUT_DIR = REPO / "reports/commentary_prefix_cache_200_20260814_run1"
CACHE_FILE = OUTPUT_DIR / "visual_prefixes.safetensors"
CACHE_TEMP = OUTPUT_DIR / ".visual_prefixes.safetensors.tmp"
MANIFEST_JSON = OUTPUT_DIR / "manifest.json"
RESULT_JSON = OUTPUT_DIR / "result.json"

SAMPLE_COUNT = 200
SAMPLE_SELECTION_SEED = 20_260_814
MODEL_SEED = 42
NUM_FRAMES = 30
SAMPLE_MODE = "middle"
EXPECTED_DATASET_LENGTH = 3_256
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_TEST_ANNOTATIONS_SIZE = 1_541_678
EXPECTED_EPOCH = 11
EXPECTED_MODEL_PARAMETER_COUNT = 8_418_890_760
EXPECTED_MODEL_STATE_KEYS = 953
EXPECTED_SOURCE_RESULT_SHA256 = (
    "dbbcf6c494e8db5517a69704ab9bf2481b79ee2d107461a6d2df9754bb45a7a7"
)
EXPECTED_SOURCE_PREDICTIONS_SHA256 = (
    "00dff75189faf6eb0e9b5121a0e56d176e4996e2fcd297fe023e07d598f79521"
)
EXPECTED_PREFIX_SHAPE = (SAMPLE_COUNT, 32, 4096)
EXPECTED_PREFIX_DTYPE = "torch.float32"
EXPECTED_RAW_PREFIX_BYTES = SAMPLE_COUNT * 32 * 4096 * 4
MIN_AVAILABLE_CPU_MEMORY_BYTES = 64 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 40 * 1024**3


class Monitor:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.stage = "preflight"
        self.progress = "not_started"
        self.memory: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self.sample_memory("start")
        self.thread.start()

    def set(self, stage: str, progress: str = "") -> None:
        self.stage = stage
        self.progress = progress
        self.sample_memory(f"enter:{stage}:{progress}")
        print(f"[STAGE] {stage} {progress}".rstrip(), flush=True)

    def finish(self) -> None:
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
            sample = self.sample_memory(
                f"heartbeat:{self.stage}:{self.progress}"
            )
            print(
                f"[HEARTBEAT] stage={self.stage} progress={self.progress} "
                f"elapsed={sample['elapsed_seconds']}s "
                f"rss={sample.get('VmRSS', 'unknown')} "
                f"hwm={sample.get('VmHWM', 'unknown')}",
                flush=True,
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(contiguous.numpy()).cast("B")).hexdigest()


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


def require_single_visible_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set explicitly")
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) != 1 or devices[0] == "-1" or not devices[0].isdigit():
        raise RuntimeError(
            "Exactly one approved numeric physical GPU must be exposed; "
            f"got CUDA_VISIBLE_DEVICES={value!r}"
        )
    return devices[0]


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def fixed_indices() -> list[int]:
    generator = random.Random(SAMPLE_SELECTION_SEED)
    return sorted(generator.sample(range(EXPECTED_DATASET_LENGTH), SAMPLE_COUNT))


def load_source_rows() -> list[dict[str, Any]]:
    if sha256(SOURCE_RESULT) != EXPECTED_SOURCE_RESULT_SHA256:
        raise RuntimeError("Source result SHA256 changed")
    if sha256(SOURCE_PREDICTIONS) != EXPECTED_SOURCE_PREDICTIONS_SHA256:
        raise RuntimeError("Source predictions SHA256 changed")
    with SOURCE_RESULT.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "passed" or result.get("samples_completed") != 200:
        raise RuntimeError("Source 200-sample experiment is not a completed pass")
    rows = []
    with SOURCE_PREDICTIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != SAMPLE_COUNT:
        raise RuntimeError(f"Expected {SAMPLE_COUNT} source rows, got {len(rows)}")
    if [row["ordinal"] for row in rows] != list(range(1, SAMPLE_COUNT + 1)):
        raise RuntimeError("Source row ordinals are not exactly 1..200")
    return rows


def require_assets(indices: list[int]) -> dict[str, Any]:
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
    require_file(SOURCE_RESULT)
    require_file(SOURCE_PREDICTIONS)
    require_directory(TEST_VIDEO_ROOT)
    require_directory(OUTPUT_DIR.parent)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUTPUT_DIR}")

    source_rows = load_source_rows()
    source_indices = [int(row["dataset_index"]) for row in source_rows]
    if source_indices != indices:
        raise RuntimeError("Fixed indices do not match the passed source experiment")

    with TEST_ANNOTATIONS.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if len(annotations) != EXPECTED_DATASET_LENGTH:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_LENGTH} annotations, got {len(annotations)}"
        )

    manifest = []
    total_video_bytes = 0
    for ordinal, (index, source_row) in enumerate(
        zip(indices, source_rows, strict=True), start=1
    ):
        annotation = annotations[index]
        video = TEST_VIDEO_ROOT / annotation["video"]
        require_file(video)
        reference = annotation["comments_text_anonymized"]
        if str(video) != source_row["video_path"]:
            raise RuntimeError(f"Source video path mismatch at index {index}")
        if reference != source_row["reference_commentary"]:
            raise RuntimeError(f"Source reference mismatch at index {index}")
        total_video_bytes += video.stat().st_size
        manifest.append(
            {
                "ordinal": ordinal,
                "dataset_index": index,
                "video_path": str(video),
                "video_bytes": video.stat().st_size,
                "reference_commentary": reference,
            }
        )

    memory_available = available_memory_bytes()
    if memory_available < MIN_AVAILABLE_CPU_MEMORY_BYTES:
        raise RuntimeError(
            f"At least {MIN_AVAILABLE_CPU_MEMORY_BYTES} available CPU bytes "
            f"required; found {memory_available}"
        )
    return {
        "generation_checkpoint_bytes": GENERATION_CHECKPOINT.stat().st_size,
        "visual_backbone_bytes": VISUAL_BACKBONE.stat().st_size,
        "test_annotations_bytes": TEST_ANNOTATIONS.stat().st_size,
        "test_annotations_sha256": sha256(TEST_ANNOTATIONS),
        "source_result_sha256": EXPECTED_SOURCE_RESULT_SHA256,
        "source_predictions_sha256": EXPECTED_SOURCE_PREDICTIONS_SHA256,
        "memory_available_bytes": memory_available,
        "selected_video_bytes": total_video_bytes,
        "manifest": manifest,
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run() -> int:
    monitor = Monitor()
    monitor.start()
    exit_code = 1
    torch = None
    result: dict[str, Any] = {
        "status": "failed",
        "sample_count": SAMPLE_COUNT,
        "sample_selection_seed": SAMPLE_SELECTION_SEED,
        "model_seed": MODEL_SEED,
        "batch_size": 1,
        "num_frames": NUM_FRAMES,
        "sample_mode": SAMPLE_MODE,
        "cache_shape_expected": list(EXPECTED_PREFIX_SHAPE),
        "cache_dtype_expected": EXPECTED_PREFIX_DTYPE,
        "cache_raw_tensor_bytes_expected": EXPECTED_RAW_PREFIX_BYTES,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "offline": True,
        "checkpoint_loaded": False,
        "model_load_count": 0,
        "dataset_created": False,
        "dataloader_created": False,
        "text_generation_executed": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "backward_executed": False,
        "training_executed": False,
        "samples_completed": 0,
        "output_dir": str(OUTPUT_DIR),
    }
    prefixes = []
    sample_manifest: list[dict[str, Any]] = []
    decode_seconds = 0.0
    visual_forward_seconds = 0.0
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(
                f"Wrong Python: expected {LOCAL_PYTHON}, got {sys.executable}"
            )
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)

        monitor.set("asset_preflight")
        physical_gpu = require_single_visible_gpu()
        indices = fixed_indices()
        assets = require_assets(indices)
        result["physical_gpu"] = physical_gpu
        result["assets"] = {
            key: value for key, value in assets.items() if key != "manifest"
        }
        result["git"] = git_identity()
        OUTPUT_DIR.mkdir()

        monitor.set("import_framework")
        import numpy as np
        from safetensors.torch import load_file, save_file
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
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_free_bytes_before": free_gpu_bytes,
            "gpu_total_bytes": total_gpu_bytes,
        }
        random.seed(MODEL_SEED)
        np.random.seed(MODEL_SEED)
        torch.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
        torch.cuda.reset_peak_memory_stats(device)

        monitor.set("construct_dataset")
        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=[str(TEST_ANNOTATIONS)],
            video_base_dir=[str(TEST_VIDEO_ROOT)],
            num_frames=NUM_FRAMES,
            sample=SAMPLE_MODE,
            tokenizer_name=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
        )
        result["dataset_created"] = True
        if len(dataset) != EXPECTED_DATASET_LENGTH:
            raise RuntimeError(
                f"Expected {EXPECTED_DATASET_LENGTH} records, got {len(dataset)}"
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
                f"Expected padding token 128001, got {dataset.tokenizer.pad_token_id}"
            )
        result["tokenizer_length"] = len(dataset.tokenizer)
        result["added_token_ids"] = actual_token_ids

        monitor.set("construct_model_and_load_base_assets")
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
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        state_key_count = len(model.state_dict())
        if parameter_count != EXPECTED_MODEL_PARAMETER_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL_PARAMETER_COUNT} parameters, "
                f"got {parameter_count}"
            )
        if state_key_count != EXPECTED_MODEL_STATE_KEYS:
            raise RuntimeError(
                f"Expected {EXPECTED_MODEL_STATE_KEYS} state keys, got {state_key_count}"
            )
        result["model_parameter_count"] = parameter_count
        result["model_state_key_count"] = state_key_count
        result["model_load_count"] = 1

        monitor.set("load_generation_checkpoint_cpu")
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
        incompatible = model.load_state_dict(state_dict, strict=False)
        result["missing_keys"] = list(incompatible.missing_keys)
        result["unexpected_keys"] = list(incompatible.unexpected_keys)
        if result["missing_keys"] or result["unexpected_keys"]:
            raise RuntimeError(
                f"State dict mismatch: missing={result['missing_keys']} "
                f"unexpected={result['unexpected_keys']}"
            )
        result["checkpoint_epoch"] = checkpoint["epoch"]
        result["checkpoint_state_key_count"] = len(state_dict)
        result["checkpoint_loaded"] = True
        del state_dict
        del checkpoint
        gc.collect()

        monitor.set("move_model_to_gpu")
        model = model.to(device)
        model.eval()
        torch.cuda.synchronize(device)

        monitor.set("capture_fixed_prefixes", f"0/{SAMPLE_COUNT}")
        for ordinal, dataset_index in enumerate(indices, start=1):
            expected = assets["manifest"][ordinal - 1]
            monitor.progress = f"{ordinal}/{SAMPLE_COUNT} index={dataset_index} decode"
            started = time.monotonic()
            item = dataset[dataset_index]
            decode_seconds += time.monotonic() - started
            if Path(item["video_path"]) != Path(expected["video_path"]):
                raise RuntimeError(f"Manifest path mismatch at index {dataset_index}")
            if item["caption_text"] != expected["reference_commentary"]:
                raise RuntimeError(f"Reference mismatch at index {dataset_index}")
            if tuple(item["frames"].shape) != (3, 30, 512, 512):
                raise RuntimeError(
                    f"Unexpected frame shape at index {dataset_index}: "
                    f"{tuple(item['frames'].shape)}"
                )
            if not torch.isfinite(item["frames"]).all().item():
                raise RuntimeError(f"Non-finite frames at index {dataset_index}")

            samples = dataset.collater([item])
            for key, value in samples.items():
                if isinstance(value, torch.Tensor):
                    samples[key] = value.to(device)

            captured: dict[str, Any] = {}
            original_generate_text = model.generate_text

            def capture_generate_text(inputs_llama: Any) -> list[str]:
                captured["inputs_llama"] = inputs_llama.detach()
                return ["__captured_without_text_generation__"]

            model.generate_text = capture_generate_text
            try:
                monitor.progress = (
                    f"{ordinal}/{SAMPLE_COUNT} index={dataset_index} visual_forward"
                )
                started = time.monotonic()
                with torch.inference_mode():
                    captured_text, ground_truths, video_paths = model(samples, True)
                torch.cuda.synchronize(device)
                visual_forward_seconds += time.monotonic() - started
            finally:
                model.generate_text = original_generate_text

            if captured_text != ["__captured_without_text_generation__"]:
                raise RuntimeError("Visual capture hook did not execute")
            if ground_truths != [item["caption_text"]]:
                raise RuntimeError(f"Ground truth mismatch at index {dataset_index}")
            if video_paths != [item["video_path"]]:
                raise RuntimeError(f"Video path mismatch at index {dataset_index}")
            if "inputs_llama" not in captured:
                raise RuntimeError(f"Missing visual prefix at index {dataset_index}")
            source_prefix = captured["inputs_llama"]
            if tuple(source_prefix.shape) != (1, 32, 4096):
                raise RuntimeError(
                    f"Unexpected prefix shape at index {dataset_index}: "
                    f"{tuple(source_prefix.shape)}"
                )
            if not torch.isfinite(source_prefix).all().item():
                raise RuntimeError(f"Non-finite visual prefix at index {dataset_index}")
            prefix = source_prefix.to(
                device="cpu", dtype=torch.float32
            ).contiguous()
            prefix_hash = tensor_sha256(prefix)
            prefixes.append(prefix)
            sample_manifest.append(
                {
                    **expected,
                    "source_dtype": str(source_prefix.dtype),
                    "stored_dtype": str(prefix.dtype),
                    "stored_shape": list(prefix.shape[1:]),
                    "prefix_sha256": prefix_hash,
                }
            )
            result["samples_completed"] = ordinal
            print(
                f"[PREFIX_OK] {ordinal}/{SAMPLE_COUNT} index={dataset_index} "
                f"sha256={prefix_hash}",
                flush=True,
            )
            del source_prefix
            del prefix
            del captured
            del samples
            del item

        monitor.set("assemble_and_write_cache")
        prefix_tensor = torch.cat(prefixes, dim=0).contiguous()
        index_tensor = torch.tensor(indices, dtype=torch.int64)
        if tuple(prefix_tensor.shape) != EXPECTED_PREFIX_SHAPE:
            raise RuntimeError(f"Unexpected cache shape: {tuple(prefix_tensor.shape)}")
        if str(prefix_tensor.dtype) != EXPECTED_PREFIX_DTYPE:
            raise RuntimeError(f"Unexpected cache dtype: {prefix_tensor.dtype}")
        if prefix_tensor.numel() * prefix_tensor.element_size() != EXPECTED_RAW_PREFIX_BYTES:
            raise RuntimeError("Unexpected raw prefix tensor byte count")
        if CACHE_FILE.exists() or CACHE_TEMP.exists():
            raise FileExistsError("Cache target unexpectedly exists")
        save_file(
            {
                "visual_prefixes": prefix_tensor,
                "dataset_indices": index_tensor,
            },
            str(CACHE_TEMP),
            metadata={
                "schema_version": "1",
                "sample_selection_seed": str(SAMPLE_SELECTION_SEED),
                "checkpoint_epoch": str(EXPECTED_EPOCH),
                "prefix_semantics": "post_qformer_llama_projection_before_bos",
            },
        )
        os.replace(CACHE_TEMP, CACHE_FILE)

        monitor.set("validate_cache")
        loaded = load_file(str(CACHE_FILE), device="cpu")
        if set(loaded) != {"visual_prefixes", "dataset_indices"}:
            raise RuntimeError(f"Unexpected cache tensor keys: {sorted(loaded)}")
        if not torch.equal(loaded["dataset_indices"], index_tensor):
            raise RuntimeError("Reloaded dataset indices differ")
        if not torch.equal(loaded["visual_prefixes"], prefix_tensor):
            raise RuntimeError("Reloaded visual prefixes differ")
        cache_sha = sha256(CACHE_FILE)
        tensor_sha = tensor_sha256(prefix_tensor)
        manifest = {
            "schema_version": 1,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cache_file": str(CACHE_FILE),
            "cache_file_bytes": CACHE_FILE.stat().st_size,
            "cache_file_sha256": cache_sha,
            "visual_prefix_tensor_sha256": tensor_sha,
            "visual_prefix_shape": list(prefix_tensor.shape),
            "visual_prefix_dtype": str(prefix_tensor.dtype),
            "visual_prefix_raw_bytes": EXPECTED_RAW_PREFIX_BYTES,
            "prefix_semantics": "post_qformer_llama_projection_before_bos",
            "dataset_indices": indices,
            "source_result_sha256": EXPECTED_SOURCE_RESULT_SHA256,
            "source_predictions_sha256": EXPECTED_SOURCE_PREDICTIONS_SHA256,
            "samples": sample_manifest,
        }
        write_json_exclusive(MANIFEST_JSON, manifest)
        result["cache"] = {
            "file": str(CACHE_FILE),
            "file_bytes": CACHE_FILE.stat().st_size,
            "file_sha256": cache_sha,
            "tensor_sha256": tensor_sha,
            "shape": list(prefix_tensor.shape),
            "dtype": str(prefix_tensor.dtype),
            "raw_tensor_bytes": EXPECTED_RAW_PREFIX_BYTES,
            "manifest": str(MANIFEST_JSON),
            "manifest_sha256": sha256(MANIFEST_JSON),
            "reload_exact_equal": True,
        }
        result["timing_totals_seconds"] = {
            "decode_and_preprocess": round(decode_seconds, 6),
            "visual_qformer_forward": round(visual_forward_seconds, 6),
        }
        result["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
            device
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
        result["memory_samples"] = monitor.memory
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["elapsed_seconds"] = round(time.monotonic() - monitor.started, 3)
        if torch is not None and torch.cuda.is_available():
            result.setdefault(
                "gpu_peak_memory_allocated_bytes",
                torch.cuda.max_memory_allocated(0),
            )
            result.setdefault(
                "gpu_peak_memory_reserved_bytes",
                torch.cuda.max_memory_reserved(0),
            )
        if OUTPUT_DIR.exists() and not RESULT_JSON.exists():
            try:
                write_json_exclusive(RESULT_JSON, result)
            except BaseException:
                traceback.print_exc()
                exit_code = 1
        print(
            "[RESULT_SUMMARY] "
            + json.dumps(
                {
                    "status": result["status"],
                    "samples_completed": result["samples_completed"],
                    "cache": result.get("cache"),
                    "elapsed_seconds": result["elapsed_seconds"],
                    "peak_rss_kib": result["peak_rss_kib"],
                    "gpu_peak_memory_allocated_bytes": result.get(
                        "gpu_peak_memory_allocated_bytes"
                    ),
                    "gpu_peak_memory_reserved_bytes": result.get(
                        "gpu_peak_memory_reserved_bytes"
                    ),
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
