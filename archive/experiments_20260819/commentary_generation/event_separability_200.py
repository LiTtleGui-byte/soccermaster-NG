#!/usr/bin/env python3
"""CPU-only event separability probe for the fixed 200 visual prefixes.

This script reads only existing local cache and E1/E2 JSONL artifacts. It
never opens the video paths embedded in those records, imports torch, loads a
model/checkpoint, or trains a model. The classifiers are fold-local linear
probes, not updates to SoccerMaster.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import numpy as np
from safetensors import safe_open
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
CACHE_DIR = REPO / "reports/commentary_prefix_cache_200_20260814_run1"
PREFIX_FILE = CACHE_DIR / "visual_prefixes.safetensors"
PREFIX_MANIFEST = CACHE_DIR / "manifest.json"
E1_PREDICTIONS = (
    REPO
    / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1/"
    "predictions.jsonl"
)
E2_PREDICTIONS = (
    REPO
    / "reports/commentary_parallel_20260814/e2_visual_sensitivity_run1/"
    "predictions.jsonl"
)
OUTPUT_DIR = REPO / "reports/commentary_event_separability_200_20260816"

SEED = 20260816
SAMPLE_COUNT = 200
PREFIX_SHAPE = (200, 32, 4096)
CV_SPLITS = 5
RARE_PRIMARY_THRESHOLD = 10
POOLED_PCA_COMPONENTS = 32
SLOT_PCA_COMPONENTS = 16

# Primary-event selection is the first category in this fixed salience order
# with at least one dictionary hit. Every hit is retained to audit ambiguity.
# The goal veto prevents an explicitly disallowed goal from becoming a goal.
EVENT_DICTIONARY: list[dict[str, Any]] = [
    {
        "category": "goal",
        "patterns": [
            r"\bgoo+a+l\s*!",
            r"\bgoal\s*[!-]",
            r"\bscores?\b",
            r"\bown goal\b",
            r"\bscore is now\b",
            r"\bscore is \d+\s*:\s*\d+\b",
            r"\bmakes? it \d+\s*:\s*\d+\b",
            r"\binside the (?:left|right|middle) post\b",
        ],
        "veto_patterns": [r"\bdisallowed goal\b", r"\bgoal, no wait\b"],
    },
    {
        "category": "penalty",
        "patterns": [
            r"\b(?:take|takes|taking|slams? home|converts?) (?:the )?penalty\b",
            r"\bpenalty\s*!",
        ],
    },
    {
        "category": "yellow_card",
        "patterns": [
            r"\byellow(?:-coloured)? card\b",
            r"\breceives? a yellow card\b",
            r"\bgets? booked\b",
            r"\bis booked\b",
            r"\bis cautioned\b",
        ],
    },
    {
        "category": "substitution",
        "patterns": [
            r"\bsubstitution\b",
            r"\bsubsitution\b",
            r"\bsubstitute\b",
            r"\bis replaced by\b",
            r"\breplacing \[player\]",
            r"\breplaces \[player\]",
            r"\bmakes way for\b",
            r"\bcoming on for\b",
            r"\bcomes on\b",
        ],
    },
    {"category": "offside", "patterns": [r"\boffside\b"]},
    {
        "category": "foul_or_free_kick",
        "patterns": [
            r"\bfree kick\b",
            r"\bfoul\b",
            r"\bhandball\b",
            r"\binfringement\b",
            r"\bpenalis(?:ed|zed)\b",
            r"\bblows? (?:his )?whistle\b",
            r"\bblew (?:his )?whistle\b",
            r"\brough tackle\b",
            r"\breckless foul\b",
            r"\bbrings? an opponent down\b",
            r"\btouches the ball with his hand\b",
            r"\bpenalised for holding\b",
        ],
    },
    {
        "category": "shot_or_save",
        "patterns": [
            r"\bshots?\b",
            r"\bshoots?\b",
            r"\bshot\b",
            r"\bstrike\b",
            r"\bvolley\b",
            r"\b(?:save|saved|saves)\b",
            r"\bgoalkeeping\b",
            r"\bparr(?:y|ies|ied)\b",
            r"\beffort (?:goes|flies|sails|was|towards|from)\b",
            r"\bheader (?:just|well|slightly|inside|towards|from)\b",
        ],
    },
    {"category": "corner", "patterns": [r"\bcorner(?: kick)?\b"]},
    {
        "category": "cross",
        "patterns": [
            r"\bcross(?:es)?\b",
            r"\bwhips?[^.!?]{0,45}\binto the box\b",
            r"\bclips?[^.!?]{0,45}\binto the box\b",
            r"\bsends? the ball into the box from the side\b",
        ],
    },
    {"category": "throw_in", "patterns": [r"\bthrow-in\b", r"\bthrow in\b"]},
    {
        "category": "injury",
        "patterns": [
            r"\binjur(?:y|ed)\b",
            r"\bmedical attention\b",
            r"\bpicking up a knock\b",
            r"\bunable to continue\b",
            r"\bable to continue to play\b",
        ],
    },
    {
        "category": "pass_or_build_up",
        "patterns": [
            r"\bpass(?:es|ed|ing)?\b",
            r"\bthrough ball\b",
            r"\blink-up play\b",
            r"\bcombination play\b",
            r"\bexchange some short passes\b",
        ],
    },
    {
        "category": "restart",
        "patterns": [r"\bhalf-time break\b", r"\bsecond half is about to start\b"],
    },
]


class Heartbeat:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stage = "startup"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def set(self, stage: str) -> None:
        self.stage = stage
        print(f"[STAGE] {stage}", flush=True)

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop.wait(30):
            print(
                f"[HEARTBEAT] elapsed={time.monotonic() - self.started:.1f}s "
                f"stage={self.stage}",
                flush=True,
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def require_runtime() -> dict[str, Any]:
    if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    if os.environ.get("PYTHONPATH") != str(REPO):
        raise RuntimeError("PYTHONPATH must be exactly the local repository")
    expected_ld = ":".join(
        [
            str(
                REPO
                / ".local_envs/SoccerMaster-repro/lib/python3.10/"
                "site-packages/torch/lib"
            ),
            str(REPO / ".local_envs/SoccerMaster-repro/lib"),
        ]
    )
    if os.environ.get("LD_LIBRARY_PATH") != expected_ld:
        raise RuntimeError("LD_LIBRARY_PATH does not match the fixed local env")
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported")
    for path in (PREFIX_FILE, PREFIX_MANIFEST, E1_PREDICTIONS, E2_PREDICTIONS):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not str(path.resolve()).startswith(str(REPO) + os.sep):
            raise RuntimeError(f"Input is not local: {path}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "pythonpath": os.environ["PYTHONPATH"],
        "ld_library_path": os.environ["LD_LIBRARY_PATH"],
        "device": "cpu",
        "torch_imported": False,
        "gpu_used": False,
    }


def extract_primary(reference: str) -> dict[str, Any]:
    lowered = reference.lower()
    matches: dict[str, list[dict[str, Any]]] = {}
    vetoed: dict[str, list[str]] = {}
    for entry in EVENT_DICTIONARY:
        category = entry["category"]
        veto_hits = [
            match.group(0)
            for pattern in entry.get("veto_patterns", [])
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE)
        ]
        if veto_hits:
            vetoed[category] = veto_hits
            continue
        category_hits = []
        for pattern in entry["patterns"]:
            for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
                category_hits.append(
                    {
                        "text": reference[match.start() : match.end()],
                        "span": [match.start(), match.end()],
                        "pattern": pattern,
                    }
                )
        if category_hits:
            matches[category] = category_hits
    primary = next(
        (entry["category"] for entry in EVENT_DICTIONARY if entry["category"] in matches),
        "other",
    )
    return {
        "raw_primary": primary,
        "matched_categories": list(matches),
        "ambiguous": len(matches) > 1,
        "matches": matches,
        "vetoed_categories": vetoed,
    }


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": labels,
    }


def fit_transform_representation(
    prefixes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    representation: str,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = prefixes[train_indices]
    test = prefixes[test_indices]
    if representation == "mean_pooled":
        train = train.mean(axis=1)
        test = test.mean(axis=1)
        input_scaler = StandardScaler().fit(train)
        train_scaled = input_scaler.transform(train)
        test_scaled = input_scaler.transform(test)
        components = min(POOLED_PCA_COMPONENTS, len(train_indices) - 1)
        reducer = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=fold_seed,
        ).fit(train_scaled)
        train_reduced = reducer.transform(train_scaled)
        test_reduced = reducer.transform(test_scaled)
        details = {
            "input_shape": [SAMPLE_COUNT, 4096],
            "pooling": "arithmetic mean over 32 query slots",
            "fold_local_reduction": f"PCA({components}) after StandardScaler",
            "explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
        }
    elif representation == "query_slot_preserving":
        train_rows = train.reshape(-1, train.shape[-1])
        test_rows = test.reshape(-1, test.shape[-1])
        input_scaler = StandardScaler().fit(train_rows)
        train_scaled = input_scaler.transform(train_rows)
        test_scaled = input_scaler.transform(test_rows)
        reducer = PCA(
            n_components=SLOT_PCA_COMPONENTS,
            svd_solver="randomized",
            random_state=fold_seed,
        ).fit(train_scaled)
        train_reduced = reducer.transform(train_scaled).reshape(len(train_indices), -1)
        test_reduced = reducer.transform(test_scaled).reshape(len(test_indices), -1)
        details = {
            "input_shape": [SAMPLE_COUNT, 32, 4096],
            "pooling": "none; 16 fold-local PCs per ordered query slot are concatenated",
            "fold_local_reduction": "shared feature-axis PCA(16) fit on training samples' query rows only",
            "reduced_sample_dimensions": int(32 * SLOT_PCA_COMPONENTS),
            "explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
        }
    else:
        raise ValueError(representation)
    output_scaler = StandardScaler().fit(train_reduced)
    return (
        output_scaler.transform(train_reduced),
        output_scaler.transform(test_reduced),
        details,
    )


def evaluate_representation(
    prefixes: np.ndarray,
    labels_array: np.ndarray,
    label_order: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    representation: str,
) -> dict[str, Any]:
    predictions = np.empty(len(labels_array), dtype=object)
    shuffled_predictions = np.empty(len(labels_array), dtype=object)
    majority_predictions = np.empty(len(labels_array), dtype=object)
    fold_records = []
    representation_offset = 0 if representation == "mean_pooled" else 10_000
    for fold_number, (train_indices, test_indices) in enumerate(folds, 1):
        fold_seed = SEED + representation_offset + fold_number
        train_x, test_x, details = fit_transform_representation(
            prefixes, train_indices, test_indices, representation, fold_seed
        )
        train_y = labels_array[train_indices]
        classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=fold_seed,
        ).fit(train_x, train_y)
        predictions[test_indices] = classifier.predict(test_x)

        shuffled_y = np.random.default_rng(fold_seed + 50_000).permutation(train_y)
        shuffled_classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            class_weight="balanced",
            random_state=fold_seed + 50_000,
        ).fit(train_x, shuffled_y)
        shuffled_predictions[test_indices] = shuffled_classifier.predict(test_x)

        train_counts = Counter(train_y.tolist())
        majority_label = sorted(train_counts, key=lambda x: (-train_counts[x], x))[0]
        majority_predictions[test_indices] = majority_label
        fold_records.append(
            {
                "fold": fold_number,
                "train_count": int(len(train_indices)),
                "test_count": int(len(test_indices)),
                "train_class_counts": dict(sorted(train_counts.items())),
                "test_class_counts": dict(sorted(Counter(labels_array[test_indices]).items())),
                "majority_label_from_training_fold": majority_label,
                "transform": details,
                "classifier_iterations": int(classifier.n_iter_.max()),
                "shuffled_classifier_iterations": int(shuffled_classifier.n_iter_.max()),
            }
        )
    return {
        "out_of_fold": metric_bundle(labels_array, predictions, label_order),
        "fixed_seed_shuffled_training_labels_baseline": metric_bundle(
            labels_array, shuffled_predictions, label_order
        ),
        "training_fold_majority_baseline": metric_bundle(
            labels_array, majority_predictions, label_order
        ),
        "folds": fold_records,
        "out_of_fold_predictions": predictions.tolist(),
        "shuffled_baseline_predictions": shuffled_predictions.tolist(),
        "majority_baseline_predictions": majority_predictions.tolist(),
    }


def build_readme(result: dict[str, Any]) -> str:
    extraction = result["silver_label_extraction"]
    lines = [
        "# Fixed-200 commentary event separability",
        "",
        "This is a CPU-only, reference-relative silver-label diagnostic. It does not establish video-ground-truth event labels or train SoccerMaster.",
        "",
        "## Label extraction",
        "",
        f"- Coverage before rare merging: {extraction['coverage_count']}/{SAMPLE_COUNT} ({extraction['coverage_rate']:.1%})",
        f"- Multi-category ambiguity: {extraction['ambiguous_count']}/{SAMPLE_COUNT} ({extraction['ambiguous_rate']:.1%})",
        f"- Raw `other`: {extraction['raw_other_count']}/{SAMPLE_COUNT}",
        f"- Rare rule: raw primary classes with fewer than {RARE_PRIMARY_THRESHOLD} samples are merged into `other_rare`; no sample is excluded.",
        f"- Final class counts: `{json.dumps(extraction['final_class_counts'], sort_keys=True)}`",
        "",
        "Primary selection uses the fixed salience order and regex dictionary stored verbatim in `result.json`. All category hits, vetoes, and final labels are also stored per sample; there are no manual overrides.",
        "",
        "## Leakage-safe results",
        "",
        "| Representation | Macro-F1 | Balanced accuracy | Shuffled-label macro-F1 | Majority macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, record in result["representations"].items():
        measured = record["out_of_fold"]
        shuffled = record["fixed_seed_shuffled_training_labels_baseline"]
        majority = record["training_fold_majority_baseline"]
        lines.append(
            f"| {name} | {measured['macro_f1']:.4f} | {measured['balanced_accuracy']:.4f} | {shuffled['macro_f1']:.4f} | {majority['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. The five stratified folds and all randomized operations use fixed seed 20260816. Full confusion matrices and fold records are in `result.json`.",
            "",
            "## Interpretation boundary",
            "",
            "Above-baseline out-of-fold performance supports reference-event separability in these cached post-Q-Former/post-projector prefixes for this fixed sample only. It is not evidence of video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    heartbeat = Heartbeat()
    heartbeat.start()
    started = time.monotonic()
    started_at_utc = utc_now()
    try:
        heartbeat.set("preflight")
        environment = require_runtime()
        manifest = load_json(PREFIX_MANIFEST)
        e1_rows = load_jsonl(E1_PREDICTIONS)
        e2_rows = load_jsonl(E2_PREDICTIONS)
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != SAMPLE_COUNT:
            raise RuntimeError("Prefix manifest does not contain 200 samples")
        if len(e1_rows) != SAMPLE_COUNT or len(e2_rows) != SAMPLE_COUNT:
            raise RuntimeError("E1/E2 row count mismatch")

        heartbeat.set("schema_and_identity")
        identities = []
        for offset, (sample, e1, e2) in enumerate(zip(samples, e1_rows, e2_rows)):
            expected = (
                int(sample["ordinal"]),
                int(sample["dataset_index"]),
                str(sample["reference_commentary"]),
            )
            if expected != (
                int(e1["ordinal"]),
                int(e1["dataset_index"]),
                str(e1["reference_commentary"]),
            ) or expected != (
                int(e2["ordinal"]),
                int(e2["dataset_index"]),
                str(e2["reference_commentary"]),
            ):
                raise RuntimeError(f"Manifest/E1/E2 identity mismatch at offset {offset}")
            identities.append(expected[1])

        with safe_open(PREFIX_FILE, framework="np") as handle:
            if set(handle.keys()) != {"dataset_indices", "visual_prefixes"}:
                raise RuntimeError(f"Unexpected safetensors keys: {list(handle.keys())}")
            stored_indices = handle.get_tensor("dataset_indices")
            prefixes = handle.get_tensor("visual_prefixes")
            metadata = handle.metadata()
        if prefixes.shape != PREFIX_SHAPE or prefixes.dtype != np.float32:
            raise RuntimeError(f"Unexpected prefixes: {prefixes.shape} {prefixes.dtype}")
        if stored_indices.shape != (SAMPLE_COUNT,) or stored_indices.tolist() != identities:
            raise RuntimeError("Cached dataset indices do not align with references")
        if not np.isfinite(prefixes).all():
            raise RuntimeError("Prefix cache contains non-finite values")

        heartbeat.set("silver_label_extraction")
        audits = []
        raw_labels = []
        for sample in samples:
            audit = extract_primary(str(sample["reference_commentary"]))
            audit.update(
                {
                    "ordinal": int(sample["ordinal"]),
                    "dataset_index": int(sample["dataset_index"]),
                    "reference_commentary": str(sample["reference_commentary"]),
                    "manual_override": False,
                }
            )
            audits.append(audit)
            raw_labels.append(audit["raw_primary"])
        raw_counts = Counter(raw_labels)
        rare_classes = sorted(
            category
            for category, count in raw_counts.items()
            if count < RARE_PRIMARY_THRESHOLD
        )
        final_labels = [
            "other_rare" if label in rare_classes else label for label in raw_labels
        ]
        for audit, final_label in zip(audits, final_labels):
            audit["final_label"] = final_label
        final_counts = Counter(final_labels)
        if min(final_counts.values()) < CV_SPLITS:
            raise RuntimeError("A final class is too small for five-fold CV")
        labels_array = np.asarray(final_labels, dtype=object)
        label_order = sorted(final_counts)
        splitter = StratifiedKFold(
            n_splits=CV_SPLITS, shuffle=True, random_state=SEED
        )
        folds = list(splitter.split(np.zeros(SAMPLE_COUNT), labels_array))

        representation_results = {}
        for representation in ("mean_pooled", "query_slot_preserving"):
            heartbeat.set(f"cross_validation:{representation}")
            representation_results[representation] = evaluate_representation(
                prefixes, labels_array, label_order, folds, representation
            )

        heartbeat.set("serialize")
        coverage_count = sum(label != "other" for label in raw_labels)
        ambiguous_count = sum(bool(audit["ambiguous"]) for audit in audits)
        result = {
            "experiment": "reference_event_separability_fixed_200",
            "status": "passed",
            "started_at_utc": started_at_utc,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "execution": {
                "timeout_seconds": 1800,
                "heartbeat_seconds": 30,
                "script_exit_code": 0,
                "fallbacks_used": [],
            },
            "scope": {
                "purpose": "Test whether reference-derived primary event categories separate in fixed post-projector prefix space.",
                "non_goals": [
                    "No video access or video-fact event annotation.",
                    "No model/checkpoint access, inference, GPU use, or SoccerMaster training.",
                    "No Gate execution or verdict.",
                ],
                "label_status": "reference-relative silver labels generated without manual overrides",
            },
            "environment": environment,
            "inputs": {
                "prefix_file": str(PREFIX_FILE),
                "prefix_manifest": str(PREFIX_MANIFEST),
                "e1_predictions": str(E1_PREDICTIONS),
                "e2_predictions": str(E2_PREDICTIONS),
                "safetensors_keys": ["dataset_indices", "visual_prefixes"],
                "prefix_shape": list(prefixes.shape),
                "prefix_dtype": str(prefixes.dtype),
                "prefix_semantics": metadata.get("prefix_semantics"),
                "sample_count": SAMPLE_COUNT,
                "identity_alignment": "ordinal, dataset_index, and reference_commentary exact across manifest/E1/E2; cached dataset_indices exact",
                "remote_paths_embedded_in_json_were_not_opened": True,
            },
            "silver_label_extraction": {
                "primary_selection_rule": "first matched category in fixed event salience priority; otherwise other",
                "priority_order": [entry["category"] for entry in EVENT_DICTIONARY],
                "dictionary": EVENT_DICTIONARY,
                "coverage_count": coverage_count,
                "coverage_rate": coverage_count / SAMPLE_COUNT,
                "ambiguous_count": ambiguous_count,
                "ambiguous_rate": ambiguous_count / SAMPLE_COUNT,
                "raw_other_count": raw_counts.get("other", 0),
                "raw_primary_counts": dict(sorted(raw_counts.items())),
                "rare_rule": f"raw primary count < {RARE_PRIMARY_THRESHOLD} merges to other_rare; no exclusion",
                "rare_classes_merged": rare_classes,
                "excluded_sample_count": 0,
                "final_class_counts": dict(sorted(final_counts.items())),
                "per_sample_audit": audits,
            },
            "cross_validation": {
                "scheme": "StratifiedKFold",
                "n_splits": CV_SPLITS,
                "shuffle": True,
                "seed": SEED,
                "leakage_control": "Every StandardScaler, PCA, and classifier fit uses training-fold data only.",
                "classifier": "LogisticRegression(C=1.0, lbfgs, class_weight=balanced)",
                "metrics": ["macro_f1", "balanced_accuracy", "confusion_matrix"],
                "baselines": [
                    "majority label selected from each training fold",
                    "same representation and classifier with fixed-seed permutation of training labels",
                ],
            },
            "representations": representation_results,
            "assertions": {
                "sample_count_200": len(samples) == SAMPLE_COUNT,
                "prefix_shape_200_32_4096": prefixes.shape == PREFIX_SHAPE,
                "all_prefix_values_finite": bool(np.isfinite(prefixes).all()),
                "manifest_e1_e2_cache_identity_aligned": True,
                "no_manual_label_overrides": all(not audit["manual_override"] for audit in audits),
                "all_samples_retained": len(final_labels) == SAMPLE_COUNT,
                "all_final_classes_support_five_folds": min(final_counts.values()) >= CV_SPLITS,
                "no_gpu_used": True,
                "torch_not_imported": "torch" not in sys.modules,
                "no_remote_video_model_checkpoint_access": True,
                "output_path_was_absent": True,
            },
            "limitations": [
                "Labels describe one reference sentence and are not independently verified video-ground-truth events.",
                "The deterministic dictionary can miss paraphrases or choose a debatable primary category in multi-event references.",
                "Rare classes are pooled into a heterogeneous other_rare class.",
                "Only 200 fixed post-Q-Former/post-projector prefixes are tested.",
                "Linear-probe separability is associative, not causal, and does not establish generation faithfulness.",
            ],
        }
        result_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        readme_text = build_readme(result)
        OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
        (OUTPUT_DIR / "result.json").write_text(result_text, encoding="utf-8")
        (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
        print(json.dumps({"status": "passed", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return 0
    finally:
        heartbeat.finish()


if __name__ == "__main__":
    raise SystemExit(main())
