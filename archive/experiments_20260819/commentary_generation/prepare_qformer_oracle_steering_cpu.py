#!/usr/bin/env python3
"""Prepare fold-local Q-Former oracle residuals without importing torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from sklearn.model_selection import GroupKFold


REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT = REPO / "experiments/commentary_generation"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from run_attribute_probe_cpu import build_tasks  # noqa: E402


DEFAULT_CONFIG = EXPERIMENT / "QFORMER_STAGE3_CONTINUATION_CONFIG_20260818.json"


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


def paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        name: Path(value).resolve()
        for name, value in config["paths"].items()
    }


def validate_static(
    config: dict[str, Any],
    require_outputs_absent: bool,
    require_trigger_inputs: bool,
) -> dict[str, Any]:
    resolved = paths(config)
    if Path.cwd().resolve() != REPO:
        raise RuntimeError(f"Run from {REPO}")
    if config.get("schema_version") != 1:
        raise RuntimeError("Unsupported config schema")
    for name in (
        "stage1_manifest", "stage2_layer_result", "blind_manifest",
        "video_only_labels", "pilot_design_config",
    ):
        if not resolved[name].is_file():
            raise FileNotFoundError(resolved[name])
    if require_trigger_inputs and not resolved["attribute_probe_result"].is_file():
        raise FileNotFoundError(resolved["attribute_probe_result"])
    reports = (REPO / "reports").resolve()
    for name in ("steering_artifact", "steering_manifest", "steering_partial"):
        if not resolved[name].is_relative_to(reports):
            raise RuntimeError(f"Output escapes reports: {resolved[name]}")
    if require_outputs_absent:
        occupied = [
            str(resolved[name])
            for name in ("steering_artifact", "steering_manifest", "steering_partial")
            if resolved[name].exists() or resolved[name].is_symlink()
        ]
        if occupied:
            raise FileExistsError(f"Refusing to overwrite: {occupied}")

    pilot = load_json(resolved["pilot_design_config"])
    stable_tasks = config["steering"]["stable_tasks"]
    if stable_tasks != pilot["steering"]["stable_tasks"]:
        raise RuntimeError("Stable-task list differs from the preflighted pilot")
    if config["steering"]["doses"] != [0.5, 1.0]:
        raise RuntimeError("Dose contract changed")
    if config["steering"]["residual_cap_fraction"] != 0.5:
        raise RuntimeError("Residual cap changed")
    if config["steering"]["minimum_train_count_per_class_per_fold"] != 4:
        raise RuntimeError("Fold support threshold changed")
    return {"paths": resolved, "pilot": pilot}


def load_locked_inputs(
    config: dict[str, Any], *, require_attribute: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    resolved = paths(config)
    stage1 = load_json(resolved["stage1_manifest"])
    stage2 = load_json(resolved["stage2_layer_result"])
    attribute = (
        load_json(resolved["attribute_probe_result"])
        if resolved["attribute_probe_result"].is_file()
        else None
    )
    blind = load_json(resolved["blind_manifest"])
    labels = load_json(resolved["video_only_labels"])
    if stage1.get("status") != "review_required" or stage1.get("num_shards") != 8:
        raise RuntimeError("Stage-1 cache is not complete")
    if stage2.get("status") != "completed":
        raise RuntimeError("Stage-2 layer result is not completed")
    if require_attribute and (
        attribute is None or attribute.get("status") != "completed"
    ):
        raise RuntimeError("Attribute trigger input is not completed")
    blind_items = blind["items"]
    label_items = labels["items"]
    ids = [row["annotation_id"] for row in blind_items]
    if len(ids) != 48 or ids != [row["annotation_id"] for row in label_items]:
        raise RuntimeError("Locked 48-item identity differs")
    if ids != stage2["identity"]["annotation_ids"]:
        raise RuntimeError("Stage-2 identity differs")
    indices = np.asarray([int(row["dataset_index"]) for row in blind_items], dtype=np.int64)
    groups = np.asarray([row["match_id"] for row in blind_items])
    selected = set(indices.tolist())
    q_by_index: dict[int, np.ndarray] = {}
    for shard in stage1["shards"]:
        with safe_open(shard["cache"], framework="numpy") as archive:
            shard_indices = archive.get_tensor("dataset_indices").astype(np.int64)
            positions = [i for i, value in enumerate(shard_indices) if int(value) in selected]
            if not positions:
                continue
            q_values = archive.get_tensor("qformer_output")[positions].astype(np.float32)
            for local, position in enumerate(positions):
                index = int(shard_indices[position])
                if index in q_by_index:
                    raise RuntimeError(f"Duplicate Q-Former sample {index}")
                q_by_index[index] = q_values[local]
    if set(q_by_index) != selected:
        raise RuntimeError("Not all 48 Q-Former outputs were found")
    qformer = np.stack([q_by_index[int(index)] for index in indices]).astype(np.float32)
    if qformer.shape != (48, 32, 768) or not np.isfinite(qformer).all():
        raise RuntimeError(f"Invalid Q-Former tensor: {qformer.shape}")
    return blind_items, label_items, indices, groups, {
        "qformer": qformer,
        "stage1": stage1,
        "stage2": stage2,
        "attribute": attribute,
    }


def clip_norm(residual: np.ndarray, cap: float) -> np.ndarray:
    norm = float(np.linalg.norm(residual.reshape(-1)))
    if norm == 0.0:
        raise RuntimeError("Zero oracle residual")
    return residual * min(1.0, cap / norm)


def build_artifact(
    config: dict[str, Any], *, require_attribute: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    blind, labels, indices, groups, loaded = load_locked_inputs(
        config, require_attribute=require_attribute
    )
    qformer = loaded["qformer"]
    tasks = build_tasks(labels)
    task_names = config["steering"]["stable_tasks"]
    minimum = config["steering"]["minimum_train_count_per_class_per_fold"]
    folds = list(GroupKFold(n_splits=4).split(qformer, np.zeros(48), groups))
    fold_ids = np.full(48, -1, dtype=np.int64)
    control = np.empty_like(qformer)
    oracle_half = np.empty_like(qformer)
    oracle_full = np.empty_like(qformer)
    sample_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(folds):
        fold_ids[test] = fold
        train_center = qformer[train].mean(axis=0)
        distances = np.linalg.norm(
            (qformer[train] - train_center).reshape(len(train), -1), axis=1
        )
        natural_scale = float(np.median(distances))
        cap = config["steering"]["residual_cap_fraction"] * natural_scale
        directions: dict[str, np.ndarray] = {}
        support: dict[str, dict[str, int]] = {}
        for task_name in task_names:
            target, eligible = tasks[task_name]
            eligible_train = train[eligible[train]]
            y = target[eligible_train].astype(bool)
            positive = int(y.sum())
            negative = int((~y).sum())
            if positive < minimum or negative < minimum:
                raise RuntimeError(
                    f"Insufficient {task_name} support in fold {fold}: {positive}/{negative}"
                )
            direction = (
                qformer[eligible_train[y]].mean(axis=0)
                - qformer[eligible_train[~y]].mean(axis=0)
            ).astype(np.float32)
            if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
                raise RuntimeError(f"Invalid direction {task_name} fold {fold}")
            directions[task_name] = direction
            support[task_name] = {"positive": positive, "negative": negative}

        cyclic = {
            name: task_names[(position + 1) % len(task_names)]
            for position, name in enumerate(task_names)
        }
        fold_records.append({
            "fold": fold,
            "train_count": int(len(train)),
            "test_count": int(len(test)),
            "train_match_count": int(len(np.unique(groups[train]))),
            "test_match_count": int(len(np.unique(groups[test]))),
            "natural_scale": natural_scale,
            "residual_cap": cap,
            "task_support": support,
        })
        for sample in test:
            components = []
            null_components = []
            eligible_names = []
            for task_name in task_names:
                target, eligible = tasks[task_name]
                if not bool(eligible[sample]):
                    continue
                sign = 1.0 if bool(target[sample]) else -1.0
                components.append(sign * directions[task_name])
                null_components.append(sign * directions[cyclic[task_name]])
                eligible_names.append(task_name)
            if not components:
                oracle_half[sample] = qformer[sample]
                oracle_full[sample] = qformer[sample]
                control[sample] = qformer[sample]
                sample_records.append({
                    "offset": int(sample),
                    "annotation_id": blind[sample]["annotation_id"],
                    "dataset_index": int(indices[sample]),
                    "match_id": blind[sample]["match_id"],
                    "fold": fold,
                    "eligible_tasks": [],
                    "steering_applied": False,
                    "oracle_half_residual_norm": 0.0,
                    "oracle_full_residual_norm": 0.0,
                    "control_residual_norm": 0.0,
                })
                continue
            raw = np.mean(components, axis=0).astype(np.float32)
            full_residual = clip_norm(raw, cap).astype(np.float32)
            half_residual = clip_norm(0.5 * raw, cap).astype(np.float32)
            null_raw = np.mean(null_components, axis=0).astype(np.float32)
            null_norm = float(np.linalg.norm(null_raw.reshape(-1)))
            target_norm = float(np.linalg.norm(full_residual.reshape(-1)))
            if null_norm == 0.0:
                raise RuntimeError(f"Zero null residual for sample {sample}")
            null_residual = (null_raw * (target_norm / null_norm)).astype(np.float32)
            oracle_half[sample] = qformer[sample] + half_residual
            oracle_full[sample] = qformer[sample] + full_residual
            control[sample] = qformer[sample] + null_residual
            sample_records.append({
                "offset": int(sample),
                "annotation_id": blind[sample]["annotation_id"],
                "dataset_index": int(indices[sample]),
                "match_id": blind[sample]["match_id"],
                "fold": fold,
                "eligible_tasks": eligible_names,
                "steering_applied": True,
                "oracle_half_residual_norm": float(np.linalg.norm(half_residual.reshape(-1))),
                "oracle_full_residual_norm": target_norm,
                "control_residual_norm": float(np.linalg.norm(null_residual.reshape(-1))),
            })

    if not (fold_ids >= 0).all():
        raise RuntimeError("Missing fold assignment")
    tensors = {
        "dataset_indices": indices,
        "fold_ids": fold_ids,
        "qformer_baseline": qformer,
        "qformer_control": control,
        "qformer_oracle_alpha_0_5": oracle_half,
        "qformer_oracle_alpha_1_0": oracle_full,
    }
    for name, value in tensors.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite output tensor {name}")
    sample_records.sort(key=lambda row: row["offset"])
    manifest = {
        "schema_version": "qformer_oracle_steering_inputs_v1",
        "status": "completed",
        "scope": "48-clip development oracle rescue; not holdout or final ranking",
        "sample_count": 48,
        "unique_match_count": 36,
        "conditions": [
            "qformer_baseline", "qformer_control",
            "qformer_oracle_alpha_0_5", "qformer_oracle_alpha_1_0",
        ],
        "tensor_shape_per_condition": [48, 32, 768],
        "stable_tasks": task_names,
        "folds": fold_records,
        "samples": sample_records,
        "steering_applied_count": sum(
            bool(row["steering_applied"]) for row in sample_records
        ),
        "no_eligible_task_policy": "leave all four conditions identical to baseline",
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "scheduler_created": False,
        "reference_commentary_used": False,
    }
    return tensors, manifest


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    validated = validate_static(
        config,
        require_outputs_absent=True,
        require_trigger_inputs=args.run,
    )
    if args.check:
        tensors, dry_manifest = build_artifact(
            config, require_attribute=False
        )
        full_delta = tensors["qformer_oracle_alpha_1_0"] - tensors["qformer_baseline"]
        control_delta = tensors["qformer_control"] - tensors["qformer_baseline"]
        full_norms = np.linalg.norm(full_delta.reshape(48, -1), axis=1)
        control_norms = np.linalg.norm(control_delta.reshape(48, -1), axis=1)
        if not np.allclose(full_norms, control_norms, rtol=1e-5, atol=1e-5):
            raise RuntimeError("Norm-matched control dry-run assertion failed")
        print(json.dumps({
            "status": "ready",
            "torch_imported": "torch" in sys.modules,
            "outputs_absent": True,
            "attribute_probe_result_present": validated["paths"]["attribute_probe_result"].is_file(),
            "stable_task_count": len(config["steering"]["stable_tasks"]),
            "residual_dry_run_completed": True,
            "condition_shapes": {
                name: list(value.shape)
                for name, value in tensors.items()
                if name.startswith("qformer_")
            },
            "fold_count": len(dry_manifest["folds"]),
            "steering_applied_count": dry_manifest["steering_applied_count"],
            "no_steering_annotation_ids": [
                row["annotation_id"]
                for row in dry_manifest["samples"]
                if not row["steering_applied"]
            ],
            "norm_matched_control": True,
        }))
        return 0
    tensors, manifest = build_artifact(config, require_attribute=True)
    resolved = validated["paths"]
    resolved["steering_artifact"].parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(resolved["steering_partial"]), metadata={
        "schema_version": "qformer_oracle_steering_inputs_v1",
        "sample_count": "48",
    })
    os.replace(resolved["steering_partial"], resolved["steering_artifact"])
    manifest["artifact"] = str(resolved["steering_artifact"])
    manifest["artifact_sha256"] = sha256(resolved["steering_artifact"])
    with resolved["steering_manifest"].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "status": "completed",
        "artifact": str(resolved["steering_artifact"]),
        "artifact_sha256": manifest["artifact_sha256"],
        "sample_count": 48,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
