#!/usr/bin/env python3
"""Capture and decode Q-Former layer-1 versus layer-2 outputs.

This is one fixed, inference-only GPU experiment.  The visual backbone,
projector, Llama checkpoint, prompt, and deterministic beam decoder are shared
by both conditions.  It does not train, backpropagate, create an optimizer, or
write a checkpoint.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT = REPO / "experiments/commentary_generation"
REPORTS = REPO / "reports"
OUTPUT_ROOT = REPORTS / "commentary_qformer_layer_exit_12_20260819"
BLIND_MANIFEST = REPORTS / "commentary_attribute_probe_pilot_48_20260818/blind_manifest.json"
VIDEO_LABELS = REPORTS / "commentary_attribute_probe_pilot_48_20260818/video_only_labels.json"
MIN_FREE_MIB = 60_000
NUM_FRAMES = 30
SEED = 20_260_819
EXPECTED_SAMPLE_IDS = [
    "CE200-135", "CE200-180", "CE200-112", "CE200-020",
    "CE200-195", "CE200-124", "CE200-127", "CE200-142",
    "CE200-190", "CE200-173", "CE200-074", "CE200-010",
]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))


class Heartbeat:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.stage = "preflight"
        self.progress = "not_started"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, stage: str, progress: str = "") -> None:
        self.stage = stage
        self.progress = progress
        print(f"[STAGE] {stage} {progress}".rstrip(), flush=True)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop.wait(30):
            print(
                f"[HEARTBEAT] stage={self.stage} progress={self.progress} "
                f"elapsed_seconds={time.monotonic() - self.started:.1f}",
                flush=True,
            )


def require_gpu6_idle() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != "6":
        raise RuntimeError(f"This one-time run must expose physical GPU 6, got {visible!r}")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    state: dict[str, int] | None = None
    for line in query.stdout.splitlines():
        index, free, utilization = [part.strip() for part in line.split(",")]
        if index == "6":
            state = {"free_mib": int(free), "utilization_percent": int(utilization)}
    if state is None:
        raise RuntimeError("GPU 6 is absent from nvidia-smi")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    uuid_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    uuid_by_index = {
        parts[0].strip(): parts[1].strip()
        for parts in (line.split(",", 1) for line in uuid_rows.stdout.splitlines())
    }
    target_uuid = uuid_by_index.get("6")
    active = [
        line.strip() for line in apps.stdout.splitlines()
        if target_uuid and line.split(",", 1)[0].strip() == target_uuid
    ]
    if state["free_mib"] < MIN_FREE_MIB or state["utilization_percent"] > 5 or active:
        raise RuntimeError(f"GPU 6 is no longer idle: state={state}, active={active}")
    return {**state, "active_compute_apps": active}


def select_samples() -> tuple[list[int], dict[int, dict[str, Any]]]:
    blind = json.loads(BLIND_MANIFEST.read_text(encoding="utf-8"))["items"]
    labels = json.loads(VIDEO_LABELS.read_text(encoding="utf-8"))["items"]
    blind_by_id = {row["annotation_id"]: row for row in blind}
    label_by_id = {row["annotation_id"]: row for row in labels}
    if set(EXPECTED_SAMPLE_IDS) - set(blind_by_id):
        raise RuntimeError("Fixed sample IDs are missing from blind manifest")
    indices: list[int] = []
    metadata: dict[int, dict[str, Any]] = {}
    for annotation_id in EXPECTED_SAMPLE_IDS:
        blind_row = blind_by_id[annotation_id]
        label_row = label_by_id[annotation_id]
        if label_row.get("confidence") != "high" or label_row.get("observability") != "observable":
            raise RuntimeError(f"Fixed sample {annotation_id} is not high-confidence observable")
        for evidence_key in ("event_evidence", "action_evidence", "result_evidence"):
            if label_row.get(evidence_key) != "clear":
                raise RuntimeError(f"Fixed sample {annotation_id} lacks clear {evidence_key}")
        index = int(blind_row["dataset_index"])
        indices.append(index)
        metadata[index] = {
            "annotation_id": annotation_id,
            "match_id": blind_row["match_id"],
            "event_values": label_row["event_values"],
            "action_values": label_row["action_values"],
            "result_value": label_row["result_value"],
        }
    if len(set(indices)) != 12 or len({row["match_id"] for row in metadata.values()}) != 12:
        raise RuntimeError("Fixed depth samples must be 12 distinct matches")
    return indices, metadata


def generate_from_prefix(runtime: dict[str, Any], prefix: Any) -> dict[str, Any]:
    from research.experiments.commentary_generation.cached_prefix_experiments import (
        DETERMINISTIC_BEAM, generate_one,
    )
    return generate_one(runtime, prefix, DETERMINISTIC_BEAM)


def main() -> int:
    heartbeat = Heartbeat()
    heartbeat.start()
    started = time.monotonic()
    temporary: Path | None = None
    hooks: list[Any] = []
    result: dict[str, Any] = {
        "status": "failed",
        "scope": "12 fixed high-confidence development clips; Q-Former layer-1 early exit versus layer-2 output",
        "gpu_used": True,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "checkpoint_written": False,
    }
    try:
        os.chdir(REPO)
        heartbeat.update("resource_and_asset_preflight")
        result["nvidia_smi_before"] = require_gpu6_idle()
        indices, sample_metadata = select_samples()
        result["dataset_indices"] = indices
        result["sample_metadata"] = sample_metadata
        if OUTPUT_ROOT.exists():
            raise FileExistsError(OUTPUT_ROOT)
        temporary = Path(tempfile.mkdtemp(prefix=".qformer_layer_exit_12.", dir=REPORTS))

        heartbeat.update("import_framework")
        import torch
        torch.set_num_threads(1)
        from research.experiments.commentary_generation.runtime.dataset.commentary import (
            MatchVisionCommentary_new_benchmark_from_npy_Dataset,
        )
        from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
            matchvoice_model_all_blocks,
        )
        from research.experiments.commentary_generation.runtime.paths import (
            BERT_ROOT, GENERATION_CHECKPOINT, LLAMA_ROOT, SIGLIP2_ROOT,
            TEST_ANNOTATIONS, VISUAL_BACKBONE, WORD_WORLD,
        )

        heartbeat.update("construct_dataset")
        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=[str(TEST_ANNOTATIONS)],
            video_base_dir=[
                "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/"
                "SoccerNetv2/MatchTime/SN-Caption-test-align"
            ],
            num_frames=NUM_FRAMES,
            sample="middle",
            tokenizer_name=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
        )

        heartbeat.update("construct_and_load_model")
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
        if len(model.video_Qformer.bert.encoder.layer) != 2:
            raise RuntimeError("Expected the current two-layer Q-Former")
        checkpoint = torch.load(
            GENERATION_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True
        )
        if checkpoint.get("epoch") != 11:
            raise RuntimeError(f"Unexpected checkpoint epoch: {checkpoint.get('epoch')}")
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
        del checkpoint
        gc.collect()

        device = torch.device("cuda:0")
        model = model.to(device).eval()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes // 1024**2 < MIN_FREE_MIB:
            raise RuntimeError(f"GPU free memory dropped below {MIN_FREE_MIB} MiB")
        result["gpu"] = {
            "physical_index": "6",
            "name": torch.cuda.get_device_name(device),
            "free_mib_after_load": free_bytes // 1024**2,
            "total_mib": total_bytes // 1024**2,
        }

        captures: dict[str, Any] = {}
        def capture_tensor(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(value, "last_hidden_state"):
                    value = value.last_hidden_state
                captures[name] = value.detach().float().cpu()
            return hook

        def capture_final(_module: Any, _inputs: Any, output: Any) -> None:
            captures["qformer_layer2"] = output.last_hidden_state.detach().float().cpu()

        hooks.append(model.video_Qformer.bert.encoder.layer[0].register_forward_hook(capture_tensor("qformer_layer1")))
        hooks.append(model.video_Qformer.bert.register_forward_hook(capture_final))

        runtime = {"torch": torch, "model": model}
        latest_prediction: dict[str, Any] = {}
        def deterministic_generate(inputs_llama: Any) -> list[str]:
            prediction = generate_from_prefix(runtime, inputs_llama)
            latest_prediction.clear()
            latest_prediction.update(prediction)
            return [prediction["text"]]
        model.generate_text = deterministic_generate

        prediction_rows: list[dict[str, Any]] = []
        layer1_rows: list[Any] = []
        layer2_rows: list[Any] = []
        for ordinal, index in enumerate(indices, start=1):
            heartbeat.update("inference", f"{ordinal}/{len(indices)} index={index}")
            item = dataset[index]
            samples = dataset.collater([item])
            for key, value in samples.items():
                if isinstance(value, torch.Tensor):
                    samples[key] = value.to(device)
            per_sample_seed = SEED + index
            random.seed(per_sample_seed)
            torch.manual_seed(per_sample_seed)
            torch.cuda.manual_seed_all(per_sample_seed)
            captures.clear()
            latest_prediction.clear()
            with torch.inference_mode():
                generated, references, video_paths = model(samples, validating=True)
            if not generated or not generated[0].strip() or not latest_prediction:
                raise RuntimeError(f"Layer-2 generation failed at {index}")
            layer2_prediction = dict(latest_prediction)
            layer1 = captures.get("qformer_layer1")
            layer2 = captures.get("qformer_layer2")
            if layer1 is None or layer2 is None:
                raise RuntimeError(f"Missing Q-Former layer capture at {index}")
            if tuple(layer1.shape) != (1, 32, 768) or tuple(layer2.shape) != (1, 32, 768):
                raise RuntimeError(f"Unexpected Q-Former shape at {index}: {layer1.shape}/{layer2.shape}")
            layer1_rows.append(layer1)
            layer2_rows.append(layer2)
            q1 = layer1.to(device=device, dtype=model.llama_proj.weight.dtype)
            with torch.inference_mode():
                prefix1 = model.llama_proj(q1)
                layer1_prediction = generate_from_prefix(runtime, prefix1)
            metadata = sample_metadata[index]
            prediction_rows.append({
                "dataset_index": index,
                "annotation_id": metadata["annotation_id"],
                "match_id": metadata["match_id"],
                "reference_commentary": item["caption_text"],
                "layer1_text": layer1_prediction["text"],
                "layer2_text": layer2_prediction["text"],
                "layer1_token_ids": layer1_prediction["token_ids"],
                "layer2_token_ids": layer2_prediction["token_ids"],
                "layer1_norm": float(layer1.norm().item()),
                "layer2_norm": float(layer2.norm().item()),
                "layer1_layer2_delta_norm": float((layer2 - layer1).norm().item()),
                "facts": metadata,
            })
            print(f"[SAMPLE_OK] {ordinal}/{len(indices)} index={index}", flush=True)
            del q1, prefix1, samples, item
            torch.cuda.empty_cache()

        heartbeat.update("write_outputs")
        import numpy as np
        from safetensors.numpy import save_file
        layer1_np = torch.cat(layer1_rows, dim=0).numpy().astype(np.float32)
        layer2_np = torch.cat(layer2_rows, dim=0).numpy().astype(np.float32)
        save_file(
            {
                "dataset_indices": np.asarray(indices, dtype=np.int64),
                "qformer_layer1": layer1_np,
                "qformer_layer2": layer2_np,
            },
            str(temporary / "qformer_layers.safetensors"),
            metadata={"scope": "layer1_vs_layer2_inference_only", "physical_gpu": "6"},
        )
        (temporary / "predictions.json").write_text(
            json.dumps(prediction_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result.update({
            "status": "passed",
            "samples_completed": len(indices),
            "layer_shapes": {"layer1": list(layer1_np.shape), "layer2": list(layer2_np.shape)},
            "prediction_file": str(OUTPUT_ROOT / "predictions.json"),
            "tensor_file": str(OUTPUT_ROOT / "qformer_layers.safetensors"),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "interpretation_boundary": [
                "Layer-1 uses the original projector trained after layer 2; this is an early-exit diagnostic, not a trained deeper-model comparison.",
                "Text changes require blind fact review before any depth conclusion.",
            ],
        })
        (temporary / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(OUTPUT_ROOT)
        temporary = None
        print(json.dumps({"status": "passed", "samples": len(indices), "output": str(OUTPUT_ROOT)}), flush=True)
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
        if temporary is not None:
            try:
                (temporary / "failure.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            except BaseException:
                traceback.print_exc()
        return 1
    finally:
        for hook in hooks:
            hook.remove()
        heartbeat.close()


if __name__ == "__main__":
    raise SystemExit(main())
