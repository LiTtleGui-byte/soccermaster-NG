#!/usr/bin/env python3
"""Run match-grouped CPU probes across the six cached commentary interfaces."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from run_attribute_probe_cpu import build_tasks, evaluate_binary


REPO = Path("/home/tianlin/SoccerMaster")
CACHE_ROOT = REPO / "reports/commentary_autorun_3256_20260817_run1"
MANIFEST_PATH = CACHE_ROOT / "manifest.json"
PACKET_DIR = REPO / "reports/commentary_attribute_probe_pilot_48_20260818"
BLIND_PATH = PACKET_DIR / "blind_manifest.json"
LABELS_PATH = PACKET_DIR / "video_only_labels.json"
OUTPUT_DIR = REPO / "reports/commentary_stage2_layer_probe_48_20260818"
OUTPUT_PATH = OUTPUT_DIR / "result.json"

REPRESENTATION_ORDER = [
    "visual_frame_global",
    "layer_normalized",
    "temporal_output",
    "qformer_input",
    "qformer_output",
    "projector_output",
]
CURRENT_PHASE = "startup"


def heartbeat() -> None:
    started = time.monotonic()
    while True:
        time.sleep(30)
        print(
            f"[HEARTBEAT] phase={CURRENT_PHASE} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ordered_token_descriptor(sequence: np.ndarray) -> np.ndarray:
    """Expose coarse order while keeping every probe linear and low capacity."""
    if sequence.ndim != 3 or sequence.shape[1] < 4:
        raise AssertionError(f"Expected [N,L,D] with L>=4, got {sequence.shape}")
    bins = [sequence.mean(axis=1)]
    for parts in (2, 4):
        for chunk in np.array_split(sequence, parts, axis=1):
            bins.append(chunk.mean(axis=1))
    bins.append(sequence[:, -1] - sequence[:, 0])
    bins.append(np.abs(np.diff(sequence, axis=1)).mean(axis=1))
    return np.concatenate(bins, axis=1).astype(np.float32)


def load_inputs() -> tuple[
    list[dict[str, Any]], np.ndarray, dict[str, np.ndarray], dict[str, Any]
]:
    for path in (MANIFEST_PATH, BLIND_PATH, LABELS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    blind = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    labels_doc = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "review_required"
        and manifest.get("dataset_samples") == 3256
        and manifest.get("num_shards") == 8
        and manifest.get("index_coverage_exact") is True
    ):
        raise AssertionError("Stage-1 merged manifest is not complete")

    blind_items = blind.get("items", [])
    label_items = labels_doc.get("items", [])
    if len(blind_items) != 48 or len(label_items) != 48:
        raise AssertionError("Locked stage-2 packet must contain 48 clips")
    blind_ids = [row["annotation_id"] for row in blind_items]
    label_ids = [row["annotation_id"] for row in label_items]
    if blind_ids != label_ids or len(set(blind_ids)) != 48:
        raise AssertionError("Blind manifest and labels are not identically ordered")

    selected_indices = [int(row["dataset_index"]) for row in blind_items]
    if len(set(selected_indices)) != 48:
        raise AssertionError("Selected dataset indices are not unique")
    selected_set = set(selected_indices)
    groups = np.asarray([row["match_id"] for row in blind_items])
    if len(np.unique(groups)) != 36:
        raise AssertionError("Expected 36 match groups in the locked packet")

    predictions: dict[int, dict[str, Any]] = {}
    tensors_by_index: dict[int, dict[str, np.ndarray]] = {}
    all_cache_indices: list[int] = []
    for shard in manifest["shards"]:
        result = json.loads(Path(shard["result"]).read_text(encoding="utf-8"))
        if not (
            result.get("status") == "passed"
            and result.get("samples_completed") == 407
            and result.get("training_executed") is False
            and result.get("backward_executed") is False
            and result.get("optimizer_created") is False
            and result.get("scheduler_created") is False
        ):
            raise AssertionError(f"Shard result is not clean: {shard['shard_id']}")
        for row in read_jsonl(Path(shard["predictions"])):
            index = int(row["dataset_index"])
            if index in predictions:
                raise AssertionError(f"Duplicate prediction index: {index}")
            predictions[index] = row

        with safe_open(shard["cache"], framework="numpy") as archive:
            keys = set(archive.keys())
            expected_keys = {"dataset_indices", *REPRESENTATION_ORDER}
            if keys != expected_keys:
                raise AssertionError(
                    f"Unexpected cache keys for shard {shard['shard_id']}: {keys}"
                )
            cache_indices = archive.get_tensor("dataset_indices").astype(np.int64)
            all_cache_indices.extend(int(value) for value in cache_indices)
            positions = [
                position
                for position, value in enumerate(cache_indices)
                if int(value) in selected_set
            ]
            for name in REPRESENTATION_ORDER:
                selected = archive.get_tensor(name)[positions].astype(np.float32)
                if not np.isfinite(selected).all():
                    raise AssertionError(
                        f"Non-finite values in shard {shard['shard_id']} tensor {name}"
                    )
                for local_position, cache_position in enumerate(positions):
                    index = int(cache_indices[cache_position])
                    tensors_by_index.setdefault(index, {})[name] = selected[local_position]

    if sorted(all_cache_indices) != list(range(3256)):
        raise AssertionError("Full cache index coverage is not exactly 0..3255")
    if set(predictions) != set(range(3256)):
        raise AssertionError("Prediction index coverage is not exactly 0..3255")
    if set(tensors_by_index) != selected_set:
        raise AssertionError("Not every selected clip was found in the layer cache")

    for row in blind_items:
        index = int(row["dataset_index"])
        prediction = predictions[index]
        if prediction["match_id"] != row["match_id"]:
            raise AssertionError(f"Match identity mismatch for dataset index {index}")
        if prediction["video_path"] != row["video_path"]:
            raise AssertionError(f"Video identity mismatch for dataset index {index}")
        if set(tensors_by_index[index]) != set(REPRESENTATION_ORDER):
            raise AssertionError(f"Incomplete representations for dataset index {index}")

    representations = {
        name: np.stack(
            [tensors_by_index[index][name] for index in selected_indices]
        ).astype(np.float32)
        for name in REPRESENTATION_ORDER
    }
    identity = {
        "annotation_ids": blind_ids,
        "dataset_indices": selected_indices,
        "unique_matches": int(len(np.unique(groups))),
        "stage1_samples": len(all_cache_indices),
        "prediction_samples": len(predictions),
    }
    return label_items, groups, representations, {
        "identity": identity,
        "label_source": labels_doc["label_source"],
    }


def main() -> int:
    global CURRENT_PHASE
    threading.Thread(target=heartbeat, daemon=True).start()
    CURRENT_PHASE = "load_and_validate_stage1_cache"
    label_items, groups, sequences, metadata = load_inputs()
    CURRENT_PHASE = "build_descriptors"
    tasks = build_tasks(label_items)
    descriptors = {
        name: ordered_token_descriptor(sequence)
        for name, sequence in sequences.items()
    }

    results: dict[str, Any] = {
        "schema_version": "commentary_stage2_layer_probe_v1",
        "status": "completed",
        "scope": "development information screening only, not causal ranking",
        "label_source": metadata["label_source"],
        "not_independent_human_gold": True,
        "identity": metadata["identity"],
        "split": {
            "type": "GroupKFold",
            "folds": 4,
            "group": "match_id",
            "unique_groups": int(len(np.unique(groups))),
            "same_match_overlap_allowed": False,
        },
        "probe": {
            "implementation": "shared with run_attribute_probe_cpu.py",
            "preprocessing": "fold-local StandardScaler and PCA(<=16)",
            "classifier": "L2 logistic regression, C=0.1, class_weight=balanced",
            "descriptor": (
                "mean + 2-bin means + 4-bin means + last-first + "
                "mean absolute adjacent difference"
            ),
        },
        "representations": {},
        "explicit_non_goals": [
            "not a causal bottleneck ranking",
            "not an independent holdout",
            "no backbone, Q-Former, projector, or Llama training",
            "no GPU, model forward, generation, or checkpoint write",
            "no decoder-logit or Llama-internal probe",
        ],
    }

    for name in REPRESENTATION_ORDER:
        CURRENT_PHASE = f"fit_grouped_probes:{name}"
        features = descriptors[name]
        representation_result: dict[str, Any] = {
            "sequence_shape": list(sequences[name].shape),
            "descriptor_shape": list(features.shape),
            "tasks": {},
        }
        completed_aucs: list[float] = []
        for task_name, (target, eligible) in tasks.items():
            task_result = evaluate_binary(features, target, eligible, groups)
            representation_result["tasks"][task_name] = task_result
            if task_result["status"] == "completed":
                completed_aucs.append(task_result["metrics"]["roc_auc"])
        representation_result["completed_task_count"] = len(completed_aucs)
        representation_result["macro_roc_auc_over_completed_tasks"] = (
            float(np.mean(completed_aucs)) if completed_aucs else None
        )
        results["representations"][name] = representation_result

    macro = {
        name: results["representations"][name][
            "macro_roc_auc_over_completed_tasks"
        ]
        for name in REPRESENTATION_ORDER
    }
    adjacent_deltas = {
        f"{left}->{right}": (
            None
            if macro[left] is None or macro[right] is None
            else float(macro[right] - macro[left])
        )
        for left, right in zip(REPRESENTATION_ORDER, REPRESENTATION_ORDER[1:])
    }
    results["screening_summary"] = {
        "macro_roc_auc": macro,
        "adjacent_macro_delta": adjacent_deltas,
        "best_readout": max(
            (name for name in REPRESENTATION_ORDER if macro[name] is not None),
            key=lambda name: macro[name],
        ),
        "interpretation_guard": (
            "Probe accessibility can nominate information-loss candidates but "
            "cannot establish causal blame without interface-matched intervention."
        ),
    }

    CURRENT_PHASE = "write_result"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    CURRENT_PHASE = "completed"
    print(json.dumps(results["screening_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
