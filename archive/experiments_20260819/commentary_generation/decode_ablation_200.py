#!/usr/bin/env python3
"""Compare three decoding strategies on a fixed 200-sample MatchTime subset.

This experiment loads the epoch-11 commentary checkpoint once, uses batch size
one, and computes the visual/Q-Former representation once per sample. It never
creates a DataLoader, optimizer, or scheduler and never performs backward or
training. The caller must expose exactly one explicitly approved physical GPU.
"""

from __future__ import annotations

from collections import Counter
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


# These must be set before importing torch, transformers, or local runtime code.
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
OUTPUT_DIR = REPO / "reports/commentary_decode_ablation_200_20260814"
PREDICTIONS_JSONL = OUTPUT_DIR / "predictions.jsonl"
RESULT_JSON = OUTPUT_DIR / "result.json"

SAMPLE_COUNT = 200
SAMPLE_SELECTION_SEED = 20_260_814
GENERATION_SEED = 42
NUM_FRAMES = 30
SAMPLE_MODE = "middle"
EXPECTED_DATASET_LENGTH = 3_256
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_TEST_ANNOTATIONS_SIZE = 1_541_678
EXPECTED_EPOCH = 11
EXPECTED_MODEL_PARAMETER_COUNT = 8_418_890_760
EXPECTED_MODEL_STATE_KEYS = 953
MIN_AVAILABLE_CPU_MEMORY_BYTES = 64 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 40 * 1024**3

COMMON_GENERATION_SETTINGS: dict[str, Any] = {
    "max_new_tokens": 128,
    "min_length": 5,
    "repetition_penalty": 1.0,
    "length_penalty": 1,
    "renormalize_logits": True,
}
STRATEGIES: dict[str, dict[str, Any]] = {
    "historical_beam_sampling": {
        **COMMON_GENERATION_SETTINGS,
        "num_beams": 5,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0,
    },
    "nucleus_sampling": {
        **COMMON_GENERATION_SETTINGS,
        "num_beams": 1,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.0,
    },
    "deterministic_beam": {
        **COMMON_GENERATION_SETTINGS,
        "num_beams": 5,
        "do_sample": False,
    },
}


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
    if len(devices) != 1 or devices[0] == "-1":
        raise RuntimeError(
            "Exactly one approved physical GPU must be exposed through "
            f"CUDA_VISIBLE_DEVICES; got {value!r}"
        )
    return devices[0]


def fixed_indices() -> list[int]:
    generator = random.Random(SAMPLE_SELECTION_SEED)
    return sorted(generator.sample(range(EXPECTED_DATASET_LENGTH), SAMPLE_COUNT))


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
    require_directory(TEST_VIDEO_ROOT)
    require_directory(OUTPUT_DIR.parent)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite output: {OUTPUT_DIR}")

    with TEST_ANNOTATIONS.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)
    if len(annotations) != EXPECTED_DATASET_LENGTH:
        raise RuntimeError(
            f"Expected {EXPECTED_DATASET_LENGTH} annotations, got {len(annotations)}"
        )
    manifest: list[dict[str, Any]] = []
    for index in indices:
        annotation = annotations[index]
        video = TEST_VIDEO_ROOT / annotation["video"]
        require_file(video)
        manifest.append(
            {
                "dataset_index": index,
                "video_path": str(video),
                "video_bytes": video.stat().st_size,
                "reference_commentary": annotation["comments_text_anonymized"],
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
        "memory_available_bytes": memory_available,
        "manifest": manifest,
    }


def generate_from_inputs(model: Any, inputs_llama: Any, settings: dict[str, Any]) -> str:
    import torch

    from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
        process_output_tokens,
    )

    start_token = torch.tensor([128000], device=inputs_llama.device)
    start_embeds = model.llama_model.base_model.model.model.embed_tokens(start_token)
    inputs_with_start = torch.cat(
        [inputs_llama, start_embeds.expand(inputs_llama.size(0), -1, -1)],
        dim=1,
    ).to(dtype=torch.bfloat16)
    output_tokens = model.llama_model.generate(
        logits_processor=model.logits_prosessors,
        inputs_embeds=inputs_with_start,
        **settings,
    )
    texts = process_output_tokens(model, output_tokens)
    if not isinstance(texts, list) or len(texts) != 1:
        raise RuntimeError(f"Unexpected generated output: {texts!r}")
    text = texts[0]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Generated commentary is empty: {text!r}")
    if "<|end_of_text|>" in text:
        raise RuntimeError("Decoded output still contains <|end_of_text|>")
    return text


def diversity_metrics(texts: list[str]) -> dict[str, Any]:
    counts = Counter(texts)
    tokenized = [text.lower().split() for text in texts]
    unigrams = [token for tokens in tokenized for token in tokens]
    bigrams = [
        (tokens[index], tokens[index + 1])
        for tokens in tokenized
        for index in range(len(tokens) - 1)
    ]
    frequencies = sorted(counts.values(), reverse=True)
    return {
        "count": len(texts),
        "unique_count": len(counts),
        "unique_rate": len(counts) / len(texts),
        "top1_share": frequencies[0] / len(texts),
        "top10_share": sum(frequencies[:10]) / len(texts),
        "top50_share": sum(frequencies[:50]) / len(texts),
        "distinct_1": len(set(unigrams)) / len(unigrams) if unigrams else 0.0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "mean_whitespace_tokens": (
            sum(len(tokens) for tokens in tokenized) / len(tokenized)
        ),
        "top10_outputs": counts.most_common(10),
    }


def overlap_metrics(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    references = {
        str(row["dataset_index"]): [row["reference_commentary"]]
        for row in rows
    }
    predictions = {
        str(row["dataset_index"]): [row["predictions"][strategy]]
        for row in rows
    }
    bleu, _ = Bleu(4).compute_score(references, predictions)
    rouge_l, _ = Rouge().compute_score(references, predictions)
    cider, _ = Cider().compute_score(references, predictions)
    return {
        "bleu_1": float(bleu[0]),
        "bleu_2": float(bleu[1]),
        "bleu_3": float(bleu[2]),
        "bleu_4": float(bleu[3]),
        "rouge_l": float(rouge_l),
        "cider": float(cider),
        "meteor_omitted": "Avoids spawning the Java METEOR subprocess in this first ablation.",
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
    predictions_handle = None
    result: dict[str, Any] = {
        "status": "failed",
        "sample_count": SAMPLE_COUNT,
        "sample_selection_seed": SAMPLE_SELECTION_SEED,
        "generation_seed": GENERATION_SEED,
        "batch_size": 1,
        "num_frames": NUM_FRAMES,
        "sample_mode": SAMPLE_MODE,
        "strategies": STRATEGIES,
        "historical_attention_mask_behavior_preserved": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "offline": True,
        "checkpoint_loaded": False,
        "model_load_count": 0,
        "dataloader_created": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "backward_executed": False,
        "training_executed": False,
        "samples_completed": 0,
        "output_dir": str(OUTPUT_DIR),
    }
    rows: list[dict[str, Any]] = []
    generation_seconds = {name: 0.0 for name in STRATEGIES}
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
        result["assets"] = assets

        monitor.set("import_framework")
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
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_free_bytes_before": free_gpu_bytes,
            "gpu_total_bytes": total_gpu_bytes,
        }
        random.seed(GENERATION_SEED)
        np.random.seed(GENERATION_SEED)
        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
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
                f"Expected {EXPECTED_MODEL_STATE_KEYS} state keys, "
                f"got {state_key_count}"
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

        monitor.set("move_model_to_gpu")
        model = model.to(device)
        model.eval()
        torch.cuda.synchronize(device)

        OUTPUT_DIR.mkdir()
        predictions_handle = PREDICTIONS_JSONL.open("x", encoding="utf-8")

        monitor.set("run_fixed_subset", f"0/{SAMPLE_COUNT}")
        for ordinal, dataset_index in enumerate(indices, start=1):
            monitor.progress = f"{ordinal}/{SAMPLE_COUNT} index={dataset_index} decode"
            started = time.monotonic()
            item = dataset[dataset_index]
            decode_seconds += time.monotonic() - started
            expected_manifest = assets["manifest"][ordinal - 1]
            if Path(item["video_path"]) != Path(expected_manifest["video_path"]):
                raise RuntimeError(
                    f"Manifest path mismatch at index {dataset_index}: "
                    f"{item['video_path']}"
                )
            if item["caption_text"] != expected_manifest["reference_commentary"]:
                raise RuntimeError(
                    f"Manifest reference mismatch at index {dataset_index}"
                )
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
                return ["__captured_without_generation__"]

            model.generate_text = capture_generate_text
            try:
                monitor.progress = (
                    f"{ordinal}/{SAMPLE_COUNT} index={dataset_index} visual_forward"
                )
                started = time.monotonic()
                with torch.no_grad():
                    captured_text, ground_truths, video_paths = model(samples, True)
                torch.cuda.synchronize(device)
                visual_forward_seconds += time.monotonic() - started
            finally:
                model.generate_text = original_generate_text
            if captured_text != ["__captured_without_generation__"]:
                raise RuntimeError("Visual capture hook did not execute as expected")
            if ground_truths != [item["caption_text"]]:
                raise RuntimeError(f"Ground truth mismatch at index {dataset_index}")
            if video_paths != [item["video_path"]]:
                raise RuntimeError(f"Video path mismatch at index {dataset_index}")
            if "inputs_llama" not in captured:
                raise RuntimeError(f"Missing visual representation at {dataset_index}")
            inputs_llama = captured["inputs_llama"]
            if tuple(inputs_llama.shape) != (1, 32, 4096):
                raise RuntimeError(
                    f"Unexpected Llama input shape at index {dataset_index}: "
                    f"{tuple(inputs_llama.shape)}"
                )

            predictions: dict[str, str] = {}
            per_strategy_seconds: dict[str, float] = {}
            for strategy, settings in STRATEGIES.items():
                monitor.progress = (
                    f"{ordinal}/{SAMPLE_COUNT} index={dataset_index} {strategy}"
                )
                sample_seed = GENERATION_SEED + dataset_index
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                started = time.monotonic()
                with torch.no_grad():
                    prediction = generate_from_inputs(model, inputs_llama, settings)
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                generation_seconds[strategy] += elapsed
                per_strategy_seconds[strategy] = round(elapsed, 6)
                predictions[strategy] = prediction

            row = {
                "ordinal": ordinal,
                "dataset_index": dataset_index,
                "sample_seed": GENERATION_SEED + dataset_index,
                "video_path": item["video_path"],
                "reference_commentary": item["caption_text"],
                "predictions": predictions,
                "generation_seconds": per_strategy_seconds,
            }
            rows.append(row)
            predictions_handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            predictions_handle.flush()
            result["samples_completed"] = ordinal
            print(
                f"[SAMPLE_OK] {ordinal}/{SAMPLE_COUNT} index={dataset_index}",
                flush=True,
            )
            del inputs_llama
            del captured
            del samples
            del item

        predictions_handle.close()
        predictions_handle = None

        monitor.set("compute_metrics")
        summaries: dict[str, Any] = {}
        for strategy in STRATEGIES:
            texts = [row["predictions"][strategy] for row in rows]
            summaries[strategy] = {
                "generation_seconds": round(generation_seconds[strategy], 6),
                "diversity": diversity_metrics(texts),
                "overlap": overlap_metrics(rows, strategy),
            }
        result["timing_totals_seconds"] = {
            "decode_and_preprocess": round(decode_seconds, 6),
            "visual_qformer_forward": round(visual_forward_seconds, 6),
            "generation": {
                key: round(value, 6) for key, value in generation_seconds.items()
            },
        }
        result["strategy_summaries"] = summaries
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
        if predictions_handle is not None:
            predictions_handle.close()
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
        print("[RESULT]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
