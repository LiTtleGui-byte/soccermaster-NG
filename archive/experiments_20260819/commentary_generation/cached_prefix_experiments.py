#!/usr/bin/env python3
"""Shared runtime for the fixed 200-sample cached-prefix experiments.

This module is imported by three independent GPU entry points.  It never
creates a dataset, DataLoader, optimizer, scheduler, or training step.  The
only model operation after loading the historical checkpoint is decoder-only
text generation (plus first-token diagnostics in E2) from the already cached
post-Q-Former visual prefixes.
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
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable


REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
OPS_BUILD = (
    REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
)
EXTENSION_SO = (
    OPS_BUILD / "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so"
)
TORCH_LIB = REPO / ".local_envs/SoccerMaster-repro/lib/python3.10/site-packages/torch/lib"
ENV_LIB = REPO / ".local_envs/SoccerMaster-repro/lib"
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
CACHE_DIR = REPO / "reports/commentary_prefix_cache_200_20260814_run1"
CACHE_FILE = CACHE_DIR / "visual_prefixes.safetensors"
CACHE_MANIFEST = CACHE_DIR / "manifest.json"
CACHE_RESULT = CACHE_DIR / "result.json"

SAMPLE_COUNT = 200
GENERATION_SEED = 42
EXPECTED_EPOCH = 11
EXPECTED_MODEL_PARAMETER_COUNT = 8_418_890_760
EXPECTED_MODEL_STATE_KEYS = 953
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_CACHE_FILE_SIZE = 104_859_528
EXPECTED_CACHE_FILE_SHA256 = (
    "8b1723926eacfe381ceae2ec5433767574f56028d894a1b28d7c7222c69b6c97"
)
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "f25586238eb1d0138f3510b570eeec92e4d3c7f22606b729f7e593eda1f7d346"
)
EXPECTED_CACHE_RESULT_SHA256 = (
    "99e5e6ec914651c45c81560aef241d48a9b2174742a59bda52e46e50774051ed"
)
EXPECTED_CACHE_TENSOR_SHA256 = (
    "13bac10a3facc1340e2e59a90b7e5ca4d2424d3bf9059e20ecfe58e8419a885e"
)
EXPECTED_PREFIX_SHAPE = (SAMPLE_COUNT, 32, 4096)
EXPECTED_PREFIX_DTYPE = "torch.float32"
EXPECTED_TOKEN_IDS = {
    "[PLAYER]": 128256,
    "[TEAM]": 128257,
    "[COACH]": 128258,
    "[REFEREE]": 128259,
    "([TEAM])": 128260,
}
MIN_AVAILABLE_CPU_MEMORY_BYTES = 64 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 40 * 1024**3

HISTORICAL_GENERATION: dict[str, Any] = {
    "max_new_tokens": 128,
    "num_beams": 5,
    "do_sample": True,
    "min_length": 5,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "length_penalty": 1,
    "temperature": 1.0,
    "renormalize_logits": True,
}
DETERMINISTIC_BEAM: dict[str, Any] = {
    "max_new_tokens": 128,
    "num_beams": 5,
    "do_sample": False,
    "min_length": 5,
    "repetition_penalty": 1.0,
    "length_penalty": 1,
    "renormalize_logits": True,
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
    if len(devices) != 1 or not devices[0].isdigit():
        raise RuntimeError(
            "Exactly one approved numeric physical GPU must be exposed; "
            f"got CUDA_VISIBLE_DEVICES={value!r}"
        )
    return devices[0]


def require_runtime_environment() -> None:
    if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError(
            f"Wrong Python: expected {LOCAL_PYTHON}, got {sys.executable}"
        )
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
    ):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    for required in (str(OPS_BUILD), str(REPO)):
        if required not in python_path:
            raise RuntimeError(f"PYTHONPATH must contain {required}")
    foreign_site_packages = [
        entry
        for entry in python_path
        if "site-packages" in entry
    ]
    if foreign_site_packages:
        raise RuntimeError(
            "PYTHONPATH must not mix Conda site-packages: "
            f"{foreign_site_packages}"
        )
    library_path = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    for required in (str(TORCH_LIB), str(ENV_LIB)):
        if required not in library_path:
            raise RuntimeError(f"LD_LIBRARY_PATH must contain {required}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


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


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


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


def overlap_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    references = {
        str(row["dataset_index"]): [row["reference_commentary"]]
        for row in rows
    }
    predictions = {
        str(row["dataset_index"]): [row["predictions"][condition]["text"]]
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
        "meteor_omitted": "Avoids spawning the Java METEOR subprocess.",
    }


def summarize_conditions(
    rows: list[dict[str, Any]], conditions: list[str]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for condition in conditions:
        texts = [row["predictions"][condition]["text"] for row in rows]
        token_counts = [
            row["predictions"][condition]["generated_token_count"] for row in rows
        ]
        eot_positions = [
            row["predictions"][condition]["first_eot_position"] for row in rows
        ]
        summaries[condition] = {
            "diversity": diversity_metrics(texts),
            "overlap": overlap_metrics(rows, condition),
            "mean_generated_token_count": sum(token_counts) / len(token_counts),
            "eot_observed_count": sum(value is not None for value in eot_positions),
        }
    return summaries


def seed_everything(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(torch: Any, load_file: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    require_file(CACHE_FILE, EXPECTED_CACHE_FILE_SIZE)
    require_file(CACHE_MANIFEST)
    require_file(CACHE_RESULT)
    if sha256(CACHE_FILE) != EXPECTED_CACHE_FILE_SHA256:
        raise RuntimeError("Cached prefix file SHA256 changed")
    if sha256(CACHE_MANIFEST) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise RuntimeError("Cached prefix manifest SHA256 changed")
    if sha256(CACHE_RESULT) != EXPECTED_CACHE_RESULT_SHA256:
        raise RuntimeError("Cached prefix result SHA256 changed")

    manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    source_result = json.loads(CACHE_RESULT.read_text(encoding="utf-8"))
    if source_result.get("status") != "passed":
        raise RuntimeError("Cached prefix source result is not passed")
    if len(manifest.get("samples", [])) != SAMPLE_COUNT:
        raise RuntimeError("Cached prefix manifest does not contain 200 samples")
    tensors = load_file(str(CACHE_FILE), device="cpu")
    if set(tensors) != {"dataset_indices", "visual_prefixes"}:
        raise RuntimeError(f"Unexpected cache keys: {sorted(tensors)}")
    prefixes = tensors["visual_prefixes"]
    indices = tensors["dataset_indices"]
    if tuple(prefixes.shape) != EXPECTED_PREFIX_SHAPE:
        raise RuntimeError(f"Unexpected cached prefix shape: {tuple(prefixes.shape)}")
    if str(prefixes.dtype) != EXPECTED_PREFIX_DTYPE:
        raise RuntimeError(f"Unexpected cached prefix dtype: {prefixes.dtype}")
    if not torch.isfinite(prefixes).all().item():
        raise RuntimeError("Cached prefixes contain non-finite values")
    expected_indices = [sample["dataset_index"] for sample in manifest["samples"]]
    if indices.tolist() != expected_indices:
        raise RuntimeError("Cached indices and manifest indices differ")
    if tensor_sha256(prefixes) != EXPECTED_CACHE_TENSOR_SHA256:
        raise RuntimeError("Cached prefix tensor SHA256 changed")
    for offset, sample in enumerate(manifest["samples"]):
        if tensor_sha256(prefixes[offset]) != sample["prefix_sha256"]:
            raise RuntimeError(
                f"Per-sample prefix SHA256 mismatch at offset {offset}"
            )
    return {
        "prefixes": prefixes,
        "indices": indices,
        "manifest": manifest,
        "source_result": source_result,
    }


def load_decoder_runtime(monitor: Monitor, result: dict[str, Any]) -> dict[str, Any]:
    require_runtime_environment()
    physical_gpu = require_single_visible_gpu()
    require_directory(OPS_BUILD)
    require_file(EXTENSION_SO)
    require_directory(TORCH_LIB)
    require_directory(ENV_LIB)
    require_directory(REPO / "experiments/commentary_generation/runtime")
    require_directory(LLAMA_ROOT)
    require_directory(BERT_ROOT)
    require_directory(SIGLIP2_ROOT)
    require_file(VISUAL_BACKBONE, EXPECTED_VISUAL_BACKBONE_SIZE)
    require_file(GENERATION_CHECKPOINT, EXPECTED_CHECKPOINT_SIZE)
    require_file(WORD_WORLD)
    memory_available = available_memory_bytes()
    if memory_available < MIN_AVAILABLE_CPU_MEMORY_BYTES:
        raise RuntimeError(
            f"At least {MIN_AVAILABLE_CPU_MEMORY_BYTES} available CPU bytes "
            f"required; found {memory_available}"
        )
    result["physical_gpu"] = physical_gpu
    result["git"] = git_identity()
    result["memory_available_bytes_before"] = memory_available

    monitor.set("import_framework")
    import numpy as np
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
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
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }
    seed_everything(GENERATION_SEED, torch, np)
    torch.cuda.reset_peak_memory_stats(device)

    monitor.set("load_and_verify_prefix_cache")
    cache = load_cache(torch, load_file)
    result["cache"] = {
        "file": str(CACHE_FILE),
        "file_sha256": EXPECTED_CACHE_FILE_SHA256,
        "manifest": str(CACHE_MANIFEST),
        "manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "tensor_sha256": EXPECTED_CACHE_TENSOR_SHA256,
        "shape": list(cache["prefixes"].shape),
        "dtype": str(cache["prefixes"].dtype),
    }

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
    if len(model.tokenizer) != 128261:
        raise RuntimeError(f"Unexpected tokenizer length: {len(model.tokenizer)}")
    actual_token_ids = {
        token: model.tokenizer.convert_tokens_to_ids(token)
        for token in EXPECTED_TOKEN_IDS
    }
    if actual_token_ids != EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"Tokenizer contract mismatch: {actual_token_ids}")
    if model.tokenizer.pad_token_id is not None:
        raise RuntimeError(
            "Model tokenizer PAD should remain unset before explicit generation args"
        )
    result["model_parameter_count"] = parameter_count
    result["model_state_key_count"] = state_key_count
    result["model_load_count"] = 1
    result["tokenizer_length"] = len(model.tokenizer)
    result["tokenizer_pad_token_id_before_generation"] = None
    result["tokenizer_eos_token_id"] = model.tokenizer.eos_token_id
    result["generation_config_pad_token_id_before_generation"] = (
        model.llama_model.generation_config.pad_token_id
    )
    result["generation_config_eos_token_id_before_generation"] = (
        model.llama_model.generation_config.eos_token_id
    )
    if model.llama_model.generation_config.pad_token_id is not None:
        raise RuntimeError("Historical generation config PAD must be unset")
    if model.llama_model.generation_config.eos_token_id != [128001, 128009]:
        raise RuntimeError(
            "Historical generation EOS contract changed: "
            f"{model.llama_model.generation_config.eos_token_id}"
        )
    result["historical_effective_auto_pad_token_id"] = 128001

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

    monitor.set("move_decoder_only_to_gpu")
    model.llama_model = model.llama_model.to(device)
    model.llama_model.eval()
    torch.cuda.synchronize(device)
    result["visual_modules_moved_to_gpu"] = False
    result["decoder_moved_to_gpu"] = True
    return {
        "torch": torch,
        "np": np,
        "device": device,
        "model": model,
        **cache,
    }


def build_inputs_with_bos(runtime: dict[str, Any], prefix: Any) -> Any:
    torch = runtime["torch"]
    model = runtime["model"]
    decoder = model.llama_model
    start_token = torch.tensor([128000], device=prefix.device)
    start_embeds = decoder.base_model.model.model.embed_tokens(start_token)
    return torch.cat(
        [prefix, start_embeds.expand(prefix.size(0), -1, -1)],
        dim=1,
    ).to(dtype=torch.bfloat16)


def generate_one(
    runtime: dict[str, Any],
    prefix: Any,
    settings: dict[str, Any],
    *,
    attention_mask: bool = False,
    pad_token_id: int | None = None,
    eos_token_id: int | list[int] | None = None,
) -> dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    inputs_with_bos = build_inputs_with_bos(runtime, prefix)
    kwargs = dict(settings)
    if attention_mask:
        kwargs["attention_mask"] = torch.ones(
            inputs_with_bos.shape[:2],
            dtype=torch.long,
            device=inputs_with_bos.device,
        )
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    output = model.llama_model.generate(
        logits_processor=model.logits_prosessors,
        inputs_embeds=inputs_with_bos,
        **kwargs,
    )
    if output.ndim != 2 or output.shape[0] != 1:
        raise RuntimeError(f"Unexpected generated token shape: {tuple(output.shape)}")
    token_ids = output[0].detach().cpu().tolist()
    decoded = model.tokenizer.decode(output[0])
    eot_marker = "<|end_of_text|>"
    marker_position = decoded.find(eot_marker)
    text = decoded if marker_position == -1 else decoded[:marker_position]
    if not text.strip():
        raise RuntimeError(f"Generated commentary is empty: tokens={token_ids}")
    try:
        first_eot_position = token_ids.index(128001)
    except ValueError:
        first_eot_position = None
    return {
        "text": text,
        "token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "first_eot_position": first_eot_position,
        "contains_eot_128001": 128001 in token_ids,
        "contains_eot_id_128009": 128009 in token_ids,
    }


def first_token_probabilities(runtime: dict[str, Any], prefixes: Any) -> dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    inputs_with_bos = build_inputs_with_bos(runtime, prefixes)
    outputs = model.llama_model(
        inputs_embeds=inputs_with_bos,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :].float()
    allowed_ids = sorted(set(model.processor.allowed_token_id_list))
    if not allowed_ids or allowed_ids[-1] >= logits.shape[-1]:
        raise RuntimeError("Restricted token IDs are empty or out of range")
    allowed_tensor = torch.tensor(allowed_ids, device=logits.device)
    probabilities = torch.softmax(logits.index_select(-1, allowed_tensor), dim=-1)
    if not torch.isfinite(probabilities).all().item():
        raise RuntimeError("First-token probabilities contain non-finite values")
    top_offsets = probabilities.argmax(dim=-1)
    top_token_ids = allowed_tensor[top_offsets]
    return {
        "allowed_token_ids": allowed_ids,
        "probabilities": probabilities,
        "top_token_ids": top_token_ids,
    }


def jensen_shannon(p: Any, q: Any) -> Any:
    midpoint = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1)
        + (q * (q.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1)
    )


def run_experiment(
    *,
    experiment_id: str,
    output_dir: Path,
    execute: Callable[[dict[str, Any], Monitor, dict[str, Any], Any], list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> int:
    monitor = Monitor()
    monitor.start()
    exit_code = 1
    runtime: dict[str, Any] | None = None
    predictions_handle = None
    result: dict[str, Any] = {
        "status": "failed",
        "experiment_id": experiment_id,
        "protocol": protocol,
        "sample_count": SAMPLE_COUNT,
        "generation_seed": GENERATION_SEED,
        "batch_size": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "offline": True,
        "checkpoint_loaded": False,
        "model_load_count": 0,
        "dataset_created": False,
        "dataloader_created": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "backward_executed": False,
        "training_executed": False,
        "samples_completed": 0,
        "output_dir": str(output_dir),
    }
    predictions_path = output_dir / "predictions.jsonl"
    result_path = output_dir / "result.json"
    try:
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
        require_directory(output_dir.parent.parent)
        runtime = load_decoder_runtime(monitor, result)
        output_dir.parent.mkdir(exist_ok=True)
        output_dir.mkdir()
        predictions_handle = predictions_path.open("x", encoding="utf-8")
        rows = execute(runtime, monitor, result, predictions_handle)
        predictions_handle.close()
        predictions_handle = None
        if len(rows) != SAMPLE_COUNT:
            raise RuntimeError(f"Expected {SAMPLE_COUNT} result rows, got {len(rows)}")
        result["predictions_sha256"] = sha256(predictions_path)
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
        if runtime is not None:
            torch = runtime["torch"]
            device = runtime["device"]
            if torch.cuda.is_available():
                result["gpu_peak_memory_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )
                result["gpu_peak_memory_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(device)
                )
        if output_dir.exists() and not result_path.exists():
            try:
                write_json_exclusive(result_path, result)
            except BaseException:
                traceback.print_exc()
                exit_code = 1
        print("[RESULT]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return exit_code
