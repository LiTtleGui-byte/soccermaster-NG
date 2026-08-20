#!/usr/bin/env python3
"""Run match-grouped CPU probes on the frozen 48-clip feature cache."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path("/home/tianlin/SoccerMaster")
PACKET_DIR = REPO / "reports/commentary_attribute_probe_pilot_48_20260818"
FEATURES_PATH = PACKET_DIR / "frozen_backbone_features.npz"
LABELS_PATH = PACKET_DIR / "video_only_labels.json"
COORDINATOR_PATH = PACKET_DIR / "coordinator_manifest.json"
OUTPUT_PATH = PACKET_DIR / "probe_results_match_grouped.json"

SEED = 42
N_SPLITS = 4
MIN_POSITIVES = 4
MIN_NEGATIVES = 4
PCA_COMPONENTS = 16


def temporal_descriptor(sequence: np.ndarray) -> np.ndarray:
    """Encode order with fixed temporal bins and finite differences.

    This keeps the probe light: temporal structure is exposed to a linear
    classifier without fitting a high-capacity sequence model on 48 clips.
    """

    if sequence.ndim != 3 or sequence.shape[1] != 30:
        raise AssertionError(f"Expected [N,30,D], got {sequence.shape}")
    bins = [sequence.mean(axis=1)]
    for parts in (2, 4):
        for chunk in np.array_split(sequence, parts, axis=1):
            bins.append(chunk.mean(axis=1))
    bins.append(sequence[:, -1] - sequence[:, 0])
    bins.append(np.abs(np.diff(sequence, axis=1)).mean(axis=1))
    return np.concatenate(bins, axis=1).astype(np.float32)


def make_probe(n_train: int) -> Pipeline:
    components = min(PCA_COMPONENTS, n_train - 1)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=components, random_state=SEED)),
            (
                "logistic",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "prevalence": float(y_true.mean()),
    }


def evaluate_binary(
    features: np.ndarray,
    labels: np.ndarray,
    eligible: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    x = features[eligible]
    y = labels[eligible].astype(np.int64)
    g = groups[eligible]
    count = Counter(y.tolist())
    if count.get(1, 0) < MIN_POSITIVES or count.get(0, 0) < MIN_NEGATIVES:
        return {
            "status": "insufficient_support",
            "eligible_count": int(len(y)),
            "negative_count": int(count.get(0, 0)),
            "positive_count": int(count.get(1, 0)),
        }
    unique_groups = np.unique(g)
    splits = min(N_SPLITS, len(unique_groups))
    if splits < 2:
        return {"status": "insufficient_groups", "eligible_count": int(len(y))}

    splitter = GroupKFold(n_splits=splits)
    probability = np.full(len(y), np.nan, dtype=np.float64)
    dummy_probability = np.full(len(y), np.nan, dtype=np.float64)
    fold_records = []
    for fold, (train, test) in enumerate(splitter.split(x, y, g)):
        if len(np.unique(y[train])) != 2:
            return {
                "status": "single_class_training_fold",
                "fold": fold,
                "eligible_count": int(len(y)),
            }
        probe = make_probe(len(train))
        probe.fit(x[train], y[train])
        probability[test] = probe.predict_proba(x[test])[:, 1]
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(x[train], y[train])
        dummy_probability[test] = dummy.predict_proba(x[test])[:, 1]
        fold_records.append(
            {
                "fold": fold,
                "train_count": int(len(train)),
                "test_count": int(len(test)),
                "train_groups": int(len(np.unique(g[train]))),
                "test_groups": int(len(np.unique(g[test]))),
                "train_positive": int(y[train].sum()),
                "test_positive": int(y[test].sum()),
            }
        )
    if not np.isfinite(probability).all():
        raise AssertionError("Missing out-of-fold probe prediction")
    result = {
        "status": "completed",
        "eligible_count": int(len(y)),
        "negative_count": int((y == 0).sum()),
        "positive_count": int((y == 1).sum()),
        "metrics": metrics(y, probability),
        "prior_baseline": metrics(y, dummy_probability),
        "folds": fold_records,
    }
    result["delta_roc_auc_vs_prior"] = (
        result["metrics"]["roc_auc"] - result["prior_baseline"]["roc_auc"]
    )
    return result


def build_tasks(items: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for value in ("shot_attempt", "possession_sequence", "foul", "set_piece"):
        labels = np.asarray([value in row["event_values"] for row in items])
        eligible = np.asarray([row["event_evidence"] == "clear" for row in items])
        tasks[f"event:{value}"] = labels, eligible
    for value in (
        "shoot",
        "header",
        "cross",
        "pass",
        "dribble",
        "tackle",
        "free_kick",
        "corner_kick",
        "save",
    ):
        labels = np.asarray([value in row["action_values"] for row in items])
        eligible = np.asarray([row["action_evidence"] == "clear" for row in items])
        tasks[f"action:{value}"] = labels, eligible
    for value in (
        "goal",
        "saved",
        "missed",
        "blocked_or_cleared",
        "possession_retained",
        "play_stopped",
    ):
        labels = np.asarray([row["result_value"] == value for row in items])
        eligible = np.asarray([row["result_evidence"] == "clear" for row in items])
        tasks[f"result:{value}"] = labels, eligible
    for value in ("live", "stopped", "replay"):
        labels = np.asarray([value in row["phase_values"] for row in items])
        eligible = np.asarray([row["phase_evidence"] == "clear" for row in items])
        tasks[f"phase:{value}"] = labels, eligible
    # The lone invisible clip cannot support a three-way pilot. Preserve the
    # raw label file, but estimate a binary observable-vs-not-fully-observable
    # head here.
    tasks["observability:fully_observable"] = (
        np.asarray([row["observability"] == "observable" for row in items]),
        np.ones(len(items), dtype=bool),
    )
    return tasks


def main() -> int:
    for path in (FEATURES_PATH, LABELS_PATH, COORDINATOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_PATH}")

    labels_doc = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    coordinator = json.loads(COORDINATOR_PATH.read_text(encoding="utf-8"))
    items = labels_doc["items"]
    coordinator_items = coordinator["items"]
    ids = [row["annotation_id"] for row in items]
    if ids != [row["annotation_id"] for row in coordinator_items]:
        raise AssertionError("Label/coordinator ordering mismatch")
    groups = np.asarray([row["match_id"] for row in coordinator_items])

    with np.load(FEATURES_PATH, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata["annotation_ids"] != ids:
            raise AssertionError("Feature/label identity or ordering mismatch")
        representations = {
            "mean_global": archive["global_mean"].astype(np.float32),
            "temporal_global": temporal_descriptor(
                archive["global_sequence"].astype(np.float32)
            ),
            "temporal_pooled_local_late": temporal_descriptor(
                archive["local_late_sequence"].astype(np.float32)
            ),
        }
    if any(value.shape[0] != 48 for value in representations.values()):
        raise AssertionError("Feature cache is not the 48-clip packet")

    tasks = build_tasks(items)
    results: dict[str, Any] = {
        "schema_version": "attribute_probe_match_grouped_v1",
        "status": "completed",
        "scope": "architecture-screening development pilot, not a benchmark",
        "label_source": labels_doc["label_source"],
        "not_independent_human_gold": True,
        "seed": SEED,
        "split": {
            "type": "GroupKFold",
            "n_splits": N_SPLITS,
            "group": "match_id",
            "unique_groups": int(len(np.unique(groups))),
        },
        "probe": {
            "type": "StandardScaler + fold-local PCA(<=16) + L2 logistic regression",
            "C": 0.1,
            "class_weight": "balanced",
            "temporal_descriptor": (
                "full mean + 2-bin means + 4-bin means + last-first + "
                "mean absolute adjacent difference"
            ),
            "minimum_positive_and_negative_support": 4,
        },
        "masking": (
            "event/action/result/phase tasks use only head_evidence=clear; "
            "observability uses all clips"
        ),
        "representations": {},
    }
    for representation_name, features in representations.items():
        representation_result: dict[str, Any] = {
            "input_shape": list(features.shape),
            "tasks": {},
        }
        completed_aucs = []
        for task_name, (target, eligible) in tasks.items():
            task_result = evaluate_binary(features, target, eligible, groups)
            representation_result["tasks"][task_name] = task_result
            if task_result["status"] == "completed":
                completed_aucs.append(task_result["metrics"]["roc_auc"])
        representation_result["completed_task_count"] = len(completed_aucs)
        representation_result["macro_roc_auc_over_completed_tasks"] = (
            float(np.mean(completed_aucs)) if completed_aucs else None
        )
        results["representations"][representation_name] = representation_result

    macro = {
        name: value["macro_roc_auc_over_completed_tasks"]
        for name, value in results["representations"].items()
    }
    results["architecture_screening_summary"] = {
        "macro_roc_auc": macro,
        "best_representation": max(
            (name for name in macro if macro[name] is not None),
            key=lambda name: macro[name],
        ),
        "decision_policy": {
            "backbone_sufficient_signal": (
                "multiple semantically central tasks beat the prior baseline "
                "under match-grouped out-of-fold evaluation"
            ),
            "temporal_readout_needed": (
                "temporal_global materially and consistently exceeds mean_global"
            ),
            "local_readout_needed": (
                "temporal_pooled_local_late materially and consistently exceeds "
                "temporal_global"
            ),
            "representation_change_candidate": (
                "central observable tasks remain near chance for all three readouts"
            ),
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["architecture_screening_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
