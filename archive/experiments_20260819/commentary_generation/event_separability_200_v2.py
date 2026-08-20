#!/usr/bin/env python3
"""Preregistered CPU-only v2 event separability probe for 200 prefixes."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from safetensors import safe_open
from sklearn.model_selection import StratifiedKFold

from research.experiments.commentary_generation import event_separability_200 as v1


OUTPUT_DIR = v1.REPO / "reports/commentary_event_separability_200_20260816_v2"
PREREGISTRATION = (
    v1.REPO
    / "experiments/commentary_generation/EVENT_DICTIONARY_V2_PREREGISTRATION.md"
)

GOAL_PATTERNS = [
    r"\bgoo+a+l\s*!",
    r"^goal\s*[!-]",
    r"\bscores\b",
    r"\bown goal\b",
    r"\bscore is now\b",
    r"\bscore is \d+\s*:\s*\d+\b",
    r"\bmakes? it \d+\s*:\s*\d+\b",
    r"\binside the (?:left|right|middle) post\b",
]
CORNER_PATTERNS = [
    r"\bcorner kick\b",
    r"\b(?:take|takes|taking|deliver|delivers|will take|will deliver|win|wins|won|get|gets|got|have|has|had|earn|earns|earned|force|forces|forced) (?:a |the )?corner\b",
    r"\bawarded a corner\b",
    r"\b(?:the|this|resulting|another) corner\b",
    r"\bcorner (?:by|from|flag)\b",
    r"\bfrom (?:a |the )?corner\b",
    r"\bfor a corner\b",
]
YELLOW_CARD_PATTERNS = [
    r"\byellow-coloured card\b",
    r"\breceives? a yellow card\b",
    r"\bgets? booked\b",
    r"\bis booked\b",
    r"\bis cautioned\b",
    r"\b(?:is|was) shown a yellow card\b",
    r"\bshows? (?:him |her )?a yellow card\b",
    r"\bthat will be a yellow card\b",
]


def build_v2_dictionary() -> list[dict[str, Any]]:
    dictionary = deepcopy(v1.EVENT_DICTIONARY)
    replacements = {
        "goal": GOAL_PATTERNS,
        "corner": CORNER_PATTERNS,
        "yellow_card": YELLOW_CARD_PATTERNS,
    }
    for entry in dictionary:
        if entry["category"] in replacements:
            entry["patterns"] = replacements[entry["category"]]
    return dictionary


EVENT_DICTIONARY = build_v2_dictionary()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_runtime() -> dict[str, Any]:
    if Path(sys.executable).resolve() != v1.LOCAL_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    if os.environ.get("PYTHONPATH") != str(v1.REPO):
        raise RuntimeError("PYTHONPATH must be exactly the local repository")
    expected_ld = ":".join(
        [
            str(
                v1.REPO
                / ".local_envs/SoccerMaster-repro/lib/python3.10/"
                "site-packages/torch/lib"
            ),
            str(v1.REPO / ".local_envs/SoccerMaster-repro/lib"),
        ]
    )
    if os.environ.get("LD_LIBRARY_PATH") != expected_ld:
        raise RuntimeError("LD_LIBRARY_PATH does not match the fixed local env")
    if "torch" in sys.modules:
        raise RuntimeError("torch must not be imported")
    for path in (
        v1.PREFIX_FILE,
        v1.PREFIX_MANIFEST,
        v1.E1_PREDICTIONS,
        v1.E2_PREDICTIONS,
        PREREGISTRATION,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not str(path.resolve()).startswith(str(v1.REPO) + os.sep):
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


def match_sentence(
    sentence: str, sentence_index: int, reference_offset: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    lowered = sentence.lower()
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
                        "text": sentence[match.start() : match.end()],
                        "sentence_index": sentence_index,
                        "sentence_span": [match.start(), match.end()],
                        "reference_span": [
                            reference_offset + match.start(),
                            reference_offset + match.end(),
                        ],
                        "pattern": pattern,
                    }
                )
        if category_hits:
            matches[category] = category_hits
    return matches, vetoed


def extract_primary(reference: str) -> dict[str, Any]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", reference.strip())
        if sentence.strip()
    ]
    sentence_records = []
    cursor = 0
    for sentence_index, sentence in enumerate(sentences):
        reference_offset = reference.find(sentence, cursor)
        if reference_offset < 0:
            raise RuntimeError("Could not map split sentence back to reference")
        cursor = reference_offset + len(sentence)
        matches, vetoed = match_sentence(sentence, sentence_index, reference_offset)
        sentence_records.append(
            {
                "sentence_index": sentence_index,
                "text": sentence,
                "reference_span": [reference_offset, reference_offset + len(sentence)],
                "matched_categories": list(matches),
                "matches": matches,
                "vetoed_categories": vetoed,
            }
        )

    selected = next(
        (record for record in sentence_records if record["matched_categories"]),
        None,
    )
    if selected is None:
        primary = "other"
        selected_index = None
        selected_categories: list[str] = []
        selected_matches: dict[str, list[dict[str, Any]]] = {}
    else:
        selected_index = int(selected["sentence_index"])
        selected_categories = list(selected["matched_categories"])
        selected_matches = dict(selected["matches"])
        primary = next(
            entry["category"]
            for entry in EVENT_DICTIONARY
            if entry["category"] in selected_matches
        )

    whole_categories = [
        entry["category"]
        for entry in EVENT_DICTIONARY
        if any(
            entry["category"] in record["matched_categories"]
            for record in sentence_records
        )
    ]
    return {
        "raw_primary": primary,
        "selected_sentence_index": selected_index,
        "matched_categories": selected_categories,
        "matches": selected_matches,
        "primary_sentence_ambiguous": len(selected_categories) > 1,
        "whole_reference_matched_categories": whole_categories,
        "whole_reference_multi_event": len(whole_categories) > 1,
        "sentence_audit": sentence_records,
    }


def build_readme(result: dict[str, Any]) -> str:
    extraction = result["silver_label_extraction"]
    lines = [
        "# Fixed-200 commentary event separability v2",
        "",
        "This is the single formal run of the frozen v2 reference-relative silver-label protocol. It is CPU-only and does not establish video-ground-truth events or train SoccerMaster.",
        "",
        "## Label extraction",
        "",
        f"- Coverage: {extraction['coverage_count']}/{v1.SAMPLE_COUNT} ({extraction['coverage_rate']:.1%})",
        f"- Selected-sentence ambiguity: {extraction['primary_sentence_ambiguous_count']}/{v1.SAMPLE_COUNT} ({extraction['primary_sentence_ambiguous_rate']:.1%})",
        f"- Whole-reference multi-event: {extraction['whole_reference_multi_event_count']}/{v1.SAMPLE_COUNT} ({extraction['whole_reference_multi_event_rate']:.1%})",
        f"- Raw `other`: {extraction['raw_other_count']}/{v1.SAMPLE_COUNT}",
        f"- Rare rule: raw primary classes with fewer than {v1.RARE_PRIMARY_THRESHOLD} samples merge into `other_rare`; no sample is excluded.",
        f"- Final class counts: `{json.dumps(extraction['final_class_counts'], sort_keys=True)}`",
        "",
        "The sentence rule, complete dictionary, all hits, vetoes, and final labels are stored in `result.json`. There are no manual overrides.",
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
            "Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. The five stratified folds and all randomized operations use seed 20260816. Full confusion matrices and fold records are in `result.json`.",
            "",
            "## Boundary",
            "",
            "These results apply only to one fixed 200-sample post-Q-Former/post-projector cache and automatically extracted reference labels. They do not establish video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    heartbeat = v1.Heartbeat()
    heartbeat.start()
    started = time.monotonic()
    started_at_utc = utc_now()
    try:
        heartbeat.set("preflight")
        environment = require_runtime()
        manifest = v1.load_json(v1.PREFIX_MANIFEST)
        e1_rows = v1.load_jsonl(v1.E1_PREDICTIONS)
        e2_rows = v1.load_jsonl(v1.E2_PREDICTIONS)
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != v1.SAMPLE_COUNT:
            raise RuntimeError("Prefix manifest does not contain 200 samples")
        if len(e1_rows) != v1.SAMPLE_COUNT or len(e2_rows) != v1.SAMPLE_COUNT:
            raise RuntimeError("E1/E2 row count mismatch")

        heartbeat.set("schema_and_identity")
        identities = []
        for offset, (sample, e1, e2) in enumerate(zip(samples, e1_rows, e2_rows)):
            expected = (
                int(sample["ordinal"]),
                int(sample["dataset_index"]),
                str(sample["reference_commentary"]),
            )
            for source, row in (("E1", e1), ("E2", e2)):
                observed = (
                    int(row["ordinal"]),
                    int(row["dataset_index"]),
                    str(row["reference_commentary"]),
                )
                if observed != expected:
                    raise RuntimeError(
                        f"Manifest/{source} identity mismatch at offset {offset}"
                    )
            identities.append(expected[1])

        with safe_open(v1.PREFIX_FILE, framework="np") as handle:
            if set(handle.keys()) != {"dataset_indices", "visual_prefixes"}:
                raise RuntimeError(f"Unexpected safetensors keys: {list(handle.keys())}")
            stored_indices = handle.get_tensor("dataset_indices")
            prefixes = handle.get_tensor("visual_prefixes")
            metadata = handle.metadata()
        if prefixes.shape != v1.PREFIX_SHAPE or prefixes.dtype != np.float32:
            raise RuntimeError(f"Unexpected prefixes: {prefixes.shape} {prefixes.dtype}")
        if stored_indices.shape != (v1.SAMPLE_COUNT,) or stored_indices.tolist() != identities:
            raise RuntimeError("Cached dataset indices do not align with references")
        if not np.isfinite(prefixes).all():
            raise RuntimeError("Prefix cache contains non-finite values")

        heartbeat.set("silver_label_extraction_v2")
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
            if count < v1.RARE_PRIMARY_THRESHOLD
        )
        final_labels = [
            "other_rare" if label in rare_classes else label for label in raw_labels
        ]
        for audit, final_label in zip(audits, final_labels):
            audit["final_label"] = final_label
        final_counts = Counter(final_labels)
        if min(final_counts.values()) < v1.CV_SPLITS:
            raise RuntimeError("A final class is too small for five-fold CV")
        labels_array = np.asarray(final_labels, dtype=object)
        label_order = sorted(final_counts)
        folds = list(
            StratifiedKFold(
                n_splits=v1.CV_SPLITS, shuffle=True, random_state=v1.SEED
            ).split(np.zeros(v1.SAMPLE_COUNT), labels_array)
        )

        representation_results = {}
        for representation in ("mean_pooled", "query_slot_preserving"):
            heartbeat.set(f"cross_validation:{representation}")
            representation_results[representation] = v1.evaluate_representation(
                prefixes, labels_array, label_order, folds, representation
            )

        heartbeat.set("serialize")
        coverage_count = sum(label != "other" for label in raw_labels)
        primary_ambiguous_count = sum(
            bool(audit["primary_sentence_ambiguous"]) for audit in audits
        )
        whole_multi_count = sum(
            bool(audit["whole_reference_multi_event"]) for audit in audits
        )
        result = {
            "experiment": "reference_event_separability_fixed_200_v2",
            "status": "passed",
            "started_at_utc": started_at_utc,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "execution": {
                "timeout_seconds": 1800,
                "heartbeat_seconds": 30,
                "script_exit_code": 0,
                "fallbacks_used": [],
                "formal_run_number": 1,
            },
            "scope": {
                "purpose": "Run the preregistered v2 reference primary-event separability test on fixed prefixes.",
                "non_goals": [
                    "No video access or video-fact event annotation.",
                    "No model/checkpoint access, inference, GPU use, or SoccerMaster training.",
                    "No Gate execution or verdict.",
                ],
                "label_status": "reference-relative silver labels generated without manual overrides",
                "preregistration": str(PREREGISTRATION),
            },
            "environment": environment,
            "inputs": {
                "prefix_file": str(v1.PREFIX_FILE),
                "prefix_manifest": str(v1.PREFIX_MANIFEST),
                "e1_predictions": str(v1.E1_PREDICTIONS),
                "e2_predictions": str(v1.E2_PREDICTIONS),
                "prefix_shape": list(prefixes.shape),
                "prefix_dtype": str(prefixes.dtype),
                "prefix_semantics": metadata.get("prefix_semantics"),
                "sample_count": v1.SAMPLE_COUNT,
                "identity_alignment": "ordinal, dataset_index, and reference_commentary exact across manifest/E1/E2; cached dataset_indices exact",
                "remote_paths_embedded_in_json_were_not_opened": True,
            },
            "silver_label_extraction": {
                "version": "v2_preregistered_20260816",
                "sentence_split": r"re.split(r'(?<=[.!?])\s+', reference.strip())",
                "primary_selection_rule": "earliest sentence with a non-vetoed hit, then first category in fixed salience order within that sentence",
                "priority_order": [entry["category"] for entry in EVENT_DICTIONARY],
                "dictionary": EVENT_DICTIONARY,
                "coverage_count": coverage_count,
                "coverage_rate": coverage_count / v1.SAMPLE_COUNT,
                "primary_sentence_ambiguous_count": primary_ambiguous_count,
                "primary_sentence_ambiguous_rate": primary_ambiguous_count / v1.SAMPLE_COUNT,
                "whole_reference_multi_event_count": whole_multi_count,
                "whole_reference_multi_event_rate": whole_multi_count / v1.SAMPLE_COUNT,
                "raw_other_count": raw_counts.get("other", 0),
                "raw_primary_counts": dict(sorted(raw_counts.items())),
                "rare_rule": f"raw primary count < {v1.RARE_PRIMARY_THRESHOLD} merges to other_rare; no exclusion",
                "rare_classes_merged": rare_classes,
                "excluded_sample_count": 0,
                "final_class_counts": dict(sorted(final_counts.items())),
                "per_sample_audit": audits,
            },
            "cross_validation": {
                "scheme": "StratifiedKFold",
                "n_splits": v1.CV_SPLITS,
                "shuffle": True,
                "seed": v1.SEED,
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
                "sample_count_200": len(samples) == v1.SAMPLE_COUNT,
                "prefix_shape_200_32_4096": prefixes.shape == v1.PREFIX_SHAPE,
                "all_prefix_values_finite": bool(np.isfinite(prefixes).all()),
                "manifest_e1_e2_cache_identity_aligned": True,
                "preregistered_dictionary_used": True,
                "no_manual_label_overrides": all(
                    not audit["manual_override"] for audit in audits
                ),
                "all_samples_retained": len(final_labels) == v1.SAMPLE_COUNT,
                "all_final_classes_support_five_folds": min(final_counts.values()) >= v1.CV_SPLITS,
                "no_gpu_used": True,
                "torch_not_imported": "torch" not in sys.modules,
                "no_remote_video_model_checkpoint_access": True,
                "output_path_was_absent": True,
            },
            "limitations": [
                "Labels describe one reference and are not independently verified video-ground-truth events.",
                "The deterministic sentence splitter and dictionary can miss paraphrases or choose a debatable primary event within a multi-event sentence.",
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
        print(
            json.dumps(
                {"status": "passed", "elapsed_seconds": result["elapsed_seconds"]}
            ),
            flush=True,
        )
        return 0
    finally:
        heartbeat.finish()


if __name__ == "__main__":
    raise SystemExit(main())
