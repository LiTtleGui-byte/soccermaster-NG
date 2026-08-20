#!/usr/bin/env python3
"""Generate four paired decoder conditions from prepared Q-Former tensors.

The run is inference-only: no visual/Q-Former forward, loss, backward,
optimizer, scheduler, or checkpoint write is permitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
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
DEFAULT_CONFIG = EXPERIMENT / "QFORMER_STAGE3_CONTINUATION_CONFIG_20260818.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {name: Path(value).resolve() for name, value in config["paths"].items()}


def validate_static(config: dict[str, Any], require_artifact: bool) -> dict[str, Path]:
    paths = resolved_paths(config)
    if Path.cwd().resolve() != REPO:
        raise RuntimeError(f"Run from {REPO}")
    if config.get("schema_version") != 1:
        raise RuntimeError("Unsupported config schema")
    for name in ("pilot_design_config", "blind_manifest", "video_only_labels"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    for name, value in config["readonly_assets"].items():
        asset = Path(value).resolve()
        if not asset.exists():
            raise FileNotFoundError(f"Missing read-only asset {name}: {asset}")
    checkpoint = Path(config["readonly_assets"]["generation_checkpoint"])
    backbone = Path(config["readonly_assets"]["visual_backbone"])
    if checkpoint.stat().st_size != 17_615_455_530:
        raise RuntimeError("Historical generation checkpoint size changed")
    if backbone.stat().st_size != 1_435_281_181:
        raise RuntimeError("Historical visual backbone size changed")
    if require_artifact:
        for name in ("steering_artifact", "steering_manifest"):
            if not paths[name].is_file():
                raise FileNotFoundError(paths[name])
    reports = (REPO / "reports").resolve()
    output_names = (
        "predictions", "predictions_partial", "result", "review_packet"
    )
    for name in output_names:
        if not paths[name].is_relative_to(reports):
            raise RuntimeError(f"Output escapes reports: {paths[name]}")
        if paths[name].exists() or paths[name].is_symlink():
            raise FileExistsError(f"Refusing to overwrite {paths[name]}")
    generation = config["generation"]
    expected = {
        "max_new_tokens": 128,
        "num_beams": 5,
        "do_sample": False,
        "min_length": 5,
        "repetition_penalty": 1.0,
        "length_penalty": 1,
        "renormalize_logits": True,
    }
    if generation != expected:
        raise RuntimeError(f"Generation contract changed: {generation}")
    return paths


def write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    paths = validate_static(config, require_artifact=args.run)
    if args.check:
        print(json.dumps({
            "status": "ready",
            "gpu_queried": False,
            "model_loaded": False,
            "outputs_absent": True,
        }))
        return 0

    if config.get("status") != "authorized_once_stage3_continuation":
        raise PermissionError("Stage-3 config is not authorized for GPU execution")
    if config.get("execution_authorized") is not True:
        raise PermissionError("Stage-3 execution is disabled")
    if os.environ.get("QFORMER_STAGE3_GPU_APPROVED") != "YES":
        raise PermissionError("Missing QFORMER_STAGE3_GPU_APPROVED=YES guard")

    from safetensors import safe_open
    from research.experiments.commentary_generation.cached_prefix_experiments import (
        DETERMINISTIC_BEAM,
        Monitor,
        generate_one,
        load_decoder_runtime,
        seed_everything,
    )

    monitor = Monitor()
    monitor.start()
    started = time.monotonic()
    runtime: dict[str, Any] | None = None
    predictions_handle = None
    exit_code = 1
    result: dict[str, Any] = {
        "schema_version": "qformer_oracle_steering_generation_v1",
        "status": "failed",
        "experiment_id": config["experiment_id"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "sample_count": 48,
        "conditions": config["conditions"],
        "checkpoint_loaded": False,
        "model_load_count": 0,
        "visual_forward_executed": False,
        "qformer_forward_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "checkpoint_written": False,
        "samples_completed": 0,
    }
    rows: list[dict[str, Any]] = []
    try:
        manifest = load_json(paths["steering_manifest"])
        if manifest.get("status") != "completed" or manifest.get("sample_count") != 48:
            raise RuntimeError("Steering manifest is incomplete")
        if len(manifest.get("samples", [])) != 48:
            raise RuntimeError("Steering sample manifest is incomplete")
        if sha256(paths["steering_artifact"]) != manifest["artifact_sha256"]:
            raise RuntimeError("Steering artifact identity changed")
        blind = load_json(paths["blind_manifest"])
        labels = load_json(paths["video_only_labels"])
        ids = [row["annotation_id"] for row in blind["items"]]
        if ids != [row["annotation_id"] for row in labels["items"]]:
            raise RuntimeError("Blind/label identity changed")

        monitor.set("load_steering_artifact_cpu")
        with safe_open(paths["steering_artifact"], framework="numpy") as archive:
            keys = set(archive.keys())
            expected_keys = {
                "dataset_indices", "fold_ids", "qformer_baseline",
                "qformer_control", "qformer_oracle_alpha_0_5",
                "qformer_oracle_alpha_1_0",
            }
            if keys != expected_keys:
                raise RuntimeError(f"Unexpected steering keys: {sorted(keys)}")
            arrays = {name: archive.get_tensor(name) for name in expected_keys}
        condition_to_tensor = {
            "cached_qformer_baseline": "qformer_baseline",
            "norm_matched_cyclic_task_direction_control": "qformer_control",
            "oracle_residual_alpha_0.5": "qformer_oracle_alpha_0_5",
            "oracle_residual_alpha_1.0": "qformer_oracle_alpha_1_0",
        }
        if list(condition_to_tensor) != config["conditions"]:
            raise RuntimeError("Condition order changed")
        for tensor_name in condition_to_tensor.values():
            value = arrays[tensor_name]
            if value.shape != (48, 32, 768) or not value.dtype.name == "float32":
                raise RuntimeError(f"Invalid {tensor_name}: {value.shape}/{value.dtype}")

        monitor.set("load_historical_decoder_and_projector")
        runtime = load_decoder_runtime(monitor, result)
        torch = runtime["torch"]
        np = runtime["np"]
        model = runtime["model"]
        device = runtime["device"]
        model.llama_proj.eval()
        for parameter in model.llama_proj.parameters():
            parameter.requires_grad_(False)
        result["checkpoint_loaded"] = True
        result["projector_input_shape"] = [1, 32, 768]
        result["projector_output_shape"] = [1, 32, 4096]

        paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
        predictions_handle = paths["predictions_partial"].open("x", encoding="utf-8")
        monitor.set("paired_generation", "0/48")
        for offset, (blind_row, label_row) in enumerate(zip(blind["items"], labels["items"])):
            dataset_index = int(blind_row["dataset_index"])
            if int(arrays["dataset_indices"][offset]) != dataset_index:
                raise RuntimeError(f"Artifact identity mismatch at offset {offset}")
            steering_sample = manifest["samples"][offset]
            if (
                int(steering_sample["offset"]) != offset
                or steering_sample["annotation_id"] != blind_row["annotation_id"]
            ):
                raise RuntimeError(f"Steering manifest identity mismatch at {offset}")
            predictions: dict[str, Any] = {}
            for condition, tensor_name in condition_to_tensor.items():
                seed_everything(20_260_818 + dataset_index, torch, np)
                q_cpu = torch.from_numpy(arrays[tensor_name][offset:offset + 1]).to(
                    dtype=model.llama_proj.weight.dtype
                )
                with torch.inference_mode():
                    prefix_cpu = model.llama_proj(q_cpu)
                    if tuple(prefix_cpu.shape) != (1, 32, 4096):
                        raise RuntimeError(f"Invalid projector output {tuple(prefix_cpu.shape)}")
                    if not torch.isfinite(prefix_cpu).all().item():
                        raise RuntimeError("Non-finite projected prefix")
                    prediction = generate_one(
                        runtime,
                        prefix_cpu.to(device),
                        DETERMINISTIC_BEAM,
                    )
                predictions[condition] = prediction
                del q_cpu, prefix_cpu
            row = {
                "offset": offset,
                "annotation_id": blind_row["annotation_id"],
                "dataset_index": dataset_index,
                "match_id": blind_row["match_id"],
                "fold": int(arrays["fold_ids"][offset]),
                "steering_applied": bool(steering_sample["steering_applied"]),
                "predictions": predictions,
            }
            predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions_handle.flush()
            rows.append(row)
            result["samples_completed"] = len(rows)
            monitor.set("paired_generation", f"{len(rows)}/48 index={dataset_index}")
        predictions_handle.close()
        predictions_handle = None
        os.replace(paths["predictions_partial"], paths["predictions"])

        condition_changes = {}
        baseline_name = config["conditions"][0]
        for condition in config["conditions"][1:]:
            condition_changes[condition] = sum(
                row["predictions"][condition]["token_ids"]
                != row["predictions"][baseline_name]["token_ids"]
                for row in rows
            )
        review_items = []
        blind_mappings = []
        for row, label_row in zip(rows, labels["items"]):
            condition_order = list(config["conditions"])
            rng = random.Random(20_260_818 + row["dataset_index"])
            rng.shuffle(condition_order)
            code_mapping = {
                chr(ord("A") + position): condition
                for position, condition in enumerate(condition_order)
            }
            review_items.append({
                "annotation_id": row["annotation_id"],
                "dataset_index": row["dataset_index"],
                "match_id": row["match_id"],
                "steering_applied": row["steering_applied"],
                "locked_video_facts": {
                    "event_values": label_row["event_values"],
                    "action_values": label_row["action_values"],
                    "result_value": label_row["result_value"],
                    "phase_values": label_row["phase_values"],
                    "observability": label_row["observability"],
                },
                "candidates": {
                    code: row["predictions"][condition]["text"]
                    for code, condition in code_mapping.items()
                },
                "reviews": {
                    code: {
                        "central_fact_correct": None,
                        "fact_contradiction": None,
                        "unsupported_specificity": None,
                        "commentary_utility_1_to_5": None,
                        "notes": "",
                    }
                    for code in code_mapping
                },
            })
            blind_mappings.append({
                "dataset_index": row["dataset_index"],
                "code_to_condition": code_mapping,
            })
        write_json_exclusive(paths["review_packet"], {
            "schema_version": "qformer_oracle_paired_review_v1",
            "status": "awaiting_review",
            "reference_commentary_included": False,
            "condition_identity_blinded": True,
            "items": review_items,
        })
        result["blind_condition_mappings"] = blind_mappings
        result["condition_token_sequence_changes_vs_baseline"] = condition_changes
        result["predictions_sha256"] = sha256(paths["predictions"])
        result["review_packet_sha256"] = sha256(paths["review_packet"])
        result["status"] = "review_required"
        exit_code = 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        if predictions_handle is not None:
            predictions_handle.close()
        monitor.finish()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if runtime is not None:
            torch = runtime["torch"]
            device = runtime["device"]
            result["gpu_peak_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            result["gpu_peak_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
            del runtime
            gc.collect()
        if not paths["result"].exists():
            try:
                write_json_exclusive(paths["result"], result)
            except BaseException:
                traceback.print_exc()
                exit_code = 1
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
