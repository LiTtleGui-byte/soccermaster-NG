#!/usr/bin/env python3
"""CPU-only static preflight for the Q-Former oracle-steering pilot.

This script reads JSON and safetensors metadata only. It does not import torch,
query CUDA, load a model, generate text, or prepare intervention tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from safetensors import safe_open
from sklearn.model_selection import GroupKFold


REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT_DIR = REPO / "experiments/commentary_generation"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from run_attribute_probe_cpu import build_tasks  # noqa: E402


CONFIG = EXPERIMENT_DIR / "QFORMER_ORACLE_STEERING_PILOT_CONFIG_20260818.json"
PROTOCOL = EXPERIMENT_DIR / "QFORMER_ORACLE_STEERING_PILOT_PROTOCOL_20260818.md"
SOURCE = EXPERIMENT_DIR / "cache_layers_shard_gpu.py"
OUTPUT_DIR = REPO / "reports/commentary_qformer_oracle_steering_pilot_48_20260818"
OUTPUT = OUTPUT_DIR / "preflight.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def stage3_trigger(config: dict[str, Any], stage2: dict[str, Any]) -> dict[str, Any]:
    trigger = config["trigger"]
    attribute_path = Path(config["inputs"]["attribute_probe_result"])
    if not attribute_path.is_file():
        return {
            "ready": False,
            "reason": "attribute_probe_result_absent",
            "path": str(attribute_path),
        }
    attribute = load_json(attribute_path)
    reps = attribute.get("representations", {})
    completed_counts = [
        int(value.get("completed_task_count", 0)) for value in reps.values()
    ]
    macro_values = [
        float(value["macro_roc_auc_over_completed_tasks"])
        for value in reps.values()
        if value.get("macro_roc_auc_over_completed_tasks") is not None
    ]
    adjacent = stage2["screening_summary"]["adjacent_macro_delta"]
    q_drop = float(adjacent["qformer_input->qformer_output"])
    temporal_drop = float(adjacent["layer_normalized->temporal_output"])
    projector_delta = float(adjacent["qformer_output->projector_output"])
    checks = {
        "attribute_status_completed": attribute.get("status") == "completed",
        "attribute_completed_tasks": bool(completed_counts)
        and max(completed_counts) >= trigger["minimum_attribute_completed_tasks"],
        "attribute_macro_auc": bool(macro_values)
        and max(macro_values) >= trigger["minimum_best_attribute_macro_roc_auc"],
        "qformer_drop": q_drop <= trigger["maximum_qformer_input_to_output_delta"],
        "projector_not_materially_worse": projector_delta
        >= trigger["minimum_projector_delta"],
        "qformer_drop_twice_temporal": (
            abs(q_drop) >= 2 * abs(temporal_drop)
            if trigger["qformer_drop_must_be_twice_temporal_drop"]
            else True
        ),
    }
    return {
        "ready": all(checks.values()),
        "reason": "all_checks_passed" if all(checks.values()) else "checks_failed",
        "checks": checks,
        "observed": {
            "maximum_attribute_completed_tasks": max(completed_counts, default=0),
            "best_attribute_macro_roc_auc": max(macro_values, default=None),
            "qformer_input_to_output_delta": q_drop,
            "temporal_delta": temporal_drop,
            "projector_delta": projector_delta,
        },
    }


def main() -> int:
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    config = load_json(CONFIG)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    assert_equal(config["execution_authorized"], False, "Execution must stay disabled")
    assert_equal(config["safety"]["gpu_execution_in_this_config"], False, "GPU scope")
    for forbidden in ("training", "backward", "optimizer", "scheduler"):
        assert_equal(config["safety"][forbidden], False, f"Safety flag {forbidden}")

    stage1 = load_json(Path(config["inputs"]["stage1_manifest"]))
    stage2 = load_json(Path(config["inputs"]["stage2_layer_result"]))
    blind = load_json(Path(config["inputs"]["blind_manifest"]))
    labels = load_json(Path(config["inputs"]["video_only_labels"]))
    oracle_contract = load_json(Path(config["inputs"]["oracle_contract"]))
    assert_equal(stage1.get("status"), "review_required", "Stage-1 status")
    assert_equal(stage1.get("dataset_samples"), 3256, "Stage-1 sample count")
    assert_equal(stage1.get("num_shards"), 8, "Stage-1 shard count")
    assert_equal(stage1.get("index_coverage_exact"), True, "Stage-1 coverage")
    assert_equal(stage2.get("status"), "completed", "Stage-2 status")
    assert_equal(oracle_contract.get("ranking_rule"),
                 "Only interface-matched interventions enter the causal bottleneck ranking.",
                 "Oracle ranking rule")

    blind_items = blind["items"]
    label_items = labels["items"]
    assert_equal(len(blind_items), 48, "Blind packet size")
    assert_equal(len(label_items), 48, "Label packet size")
    blind_ids = [row["annotation_id"] for row in blind_items]
    label_ids = [row["annotation_id"] for row in label_items]
    assert_equal(blind_ids, label_ids, "Blind/label order")
    assert_equal(blind_ids, stage2["identity"]["annotation_ids"], "Stage-2 identity")
    dataset_indices = [int(row["dataset_index"]) for row in blind_items]
    assert_equal(dataset_indices, stage2["identity"]["dataset_indices"], "Dataset indices")
    groups = np.asarray([row["match_id"] for row in blind_items])
    assert_equal(len(np.unique(groups)), 36, "Unique match groups")

    cache_contracts = []
    selected_found: set[int] = set()
    for shard in stage1["shards"]:
        result = load_json(Path(shard["result"]))
        assert_equal(result.get("status"), "passed", "Stage-1 shard status")
        assert_equal(result.get("samples_completed"), 407, "Shard sample count")
        for flag in ("training_executed", "backward_executed", "optimizer_created", "scheduler_created"):
            assert_equal(result.get(flag), False, f"Shard {flag}")
        with safe_open(shard["cache"], framework="numpy") as archive:
            q_slice = archive.get_slice("qformer_output")
            shape = list(q_slice.get_shape())
            dtype = q_slice.get_dtype()
            assert_equal(shape, [407, 32, 768], "Q-Former cache shape")
            assert_equal(dtype, "F32", "Q-Former cache dtype")
            indices = archive.get_tensor("dataset_indices").astype(np.int64).tolist()
            selected_found.update(set(indices).intersection(dataset_indices))
            cache_contracts.append({
                "shard_id": int(shard["shard_id"]),
                "qformer_output_shape": shape,
                "qformer_output_dtype": dtype,
            })
    assert_equal(selected_found, set(dataset_indices), "Selected cache coverage")

    source_text = SOURCE.read_text(encoding="utf-8")
    temporal_alias = (
        'captures["temporal_output"] = captures["qformer_input"].clone()'
        in source_text
    )
    assert_equal(temporal_alias, True, "Temporal/Q-Former-input alias assertion")

    tasks = build_tasks(label_items)
    stable_tasks = config["steering"]["stable_tasks"]
    minimum = int(config["steering"]["minimum_train_count_per_class_per_fold"])
    global_splits = list(
        GroupKFold(n_splits=4).split(np.zeros((48, 1)), np.zeros(48), groups)
    )
    task_support: dict[str, Any] = {}
    for task_name in stable_tasks:
        if task_name not in tasks:
            raise AssertionError(f"Unknown stable task: {task_name}")
        target, eligible = tasks[task_name]
        fold_counts = []
        for fold, (train, test) in enumerate(global_splits):
            train_target = target[train][eligible[train]].astype(np.int64)
            positives = int(train_target.sum())
            negatives = int(len(train_target) - positives)
            if positives < minimum or negatives < minimum:
                raise AssertionError(
                    f"Task {task_name} fold {fold} lacks support: "
                    f"positive={positives} negative={negatives}"
                )
            fold_counts.append({
                "fold": fold,
                "train_positive": positives,
                "train_negative": negatives,
                "test_count": int(len(test)),
            })
        for representation in ("qformer_input", "qformer_output"):
            status = stage2["representations"][representation]["tasks"][task_name]["status"]
            assert_equal(status, "completed", f"Stage-2 task {representation}/{task_name}")
        task_support[task_name] = fold_counts

    trigger = stage3_trigger(config, stage2)
    result = {
        "schema_version": "qformer_oracle_steering_static_preflight_v1",
        "status": "passed",
        "scope": "CPU-only static design preflight; no intervention prepared or executed",
        "config": str(CONFIG),
        "protocol": str(PROTOCOL),
        "execution_authorized": False,
        "torch_imported": False,
        "gpu_queried": False,
        "model_loaded": False,
        "generation_executed": False,
        "training_executed": False,
        "identity": {
            "samples": len(blind_items),
            "unique_matches": int(len(np.unique(groups))),
            "selected_cache_samples_found": len(selected_found),
        },
        "interface": {
            "qformer_output_per_sample": [32, 768],
            "cache_dtype": "F32",
            "all_eight_shards_match": True,
            "shards": cache_contracts,
        },
        "stable_task_count": len(stable_tasks),
        "stable_task_fold_support": task_support,
        "trace_caveat": {
            "temporal_output_is_qformer_input_clone": temporal_alias,
            "interpretation": "The zero Stage-2 delta between these names is by construction and is not independent evidence."
        },
        "rankability": config["rankability"],
        "stage3_trigger": trigger,
        "next_state": (
            "READY_FOR_SEPARATELY_AUTHORIZED_IMPLEMENTATION_AND_GPU_RUN"
            if trigger["ready"]
            else "WAITING_FOR_ATTRIBUTE_PROBE_RESULT"
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": result["status"],
        "stable_task_count": result["stable_task_count"],
        "stage3_trigger": trigger,
        "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
