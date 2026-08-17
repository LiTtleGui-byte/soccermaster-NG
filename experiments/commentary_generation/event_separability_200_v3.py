#!/usr/bin/env python3
"""Preregistered CPU-only v3 event separability probe for 200 prefixes.

V3 reuses the already verified v2 execution engine after replacing only the
dictionary entries frozen in EVENT_DICTIONARY_V3_PREREGISTRATION.md. The v3
protocol identity is finalized before this process exits successfully.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from experiments.commentary_generation import event_separability_200_v2 as v2


OUTPUT_DIR = v2.v1.REPO / "reports/commentary_event_separability_200_20260816_v3"
PREREGISTRATION = (
    v2.v1.REPO
    / "experiments/commentary_generation/EVENT_DICTIONARY_V3_PREREGISTRATION.md"
)

SUBSTITUTION_ADDITIONS = [
    r"\b(?:is being|was|will be) substituted\b",
    r"\bwill be replaced by\b",
]
YELLOW_V2_PATTERN = r"\breceives? a yellow card\b"
YELLOW_V3_PATTERN = r"\breceiv(?:e|es|ed|ing) a yellow card\b"
FOUL_ADDITION = r"\btrips? an opponent\b"
PASS_ADDITION = r"\bsends? a long ball\b"


def build_v3_dictionary() -> list[dict[str, Any]]:
    dictionary = deepcopy(v2.EVENT_DICTIONARY)
    for entry in dictionary:
        category = entry["category"]
        if category == "substitution":
            entry["patterns"].extend(SUBSTITUTION_ADDITIONS)
        elif category == "yellow_card":
            if entry["patterns"].count(YELLOW_V2_PATTERN) != 1:
                raise RuntimeError("Expected exactly one v2 yellow-card pattern")
            entry["patterns"] = [
                YELLOW_V3_PATTERN if pattern == YELLOW_V2_PATTERN else pattern
                for pattern in entry["patterns"]
            ]
        elif category == "foul_or_free_kick":
            entry["patterns"].append(FOUL_ADDITION)
        elif category == "pass_or_build_up":
            entry["patterns"].append(PASS_ADDITION)
    return dictionary


EVENT_DICTIONARY = build_v3_dictionary()


def build_readme(result: dict[str, Any]) -> str:
    extraction = result["silver_label_extraction"]
    lines = [
        "# Fixed-200 commentary event separability v3",
        "",
        "This is the single formal run of the frozen v3 reference-relative silver-label protocol. It is CPU-only and does not establish video-ground-truth events or train SoccerMaster.",
        "",
        "## Label extraction",
        "",
        f"- Coverage: {extraction['coverage_count']}/{v2.v1.SAMPLE_COUNT} ({extraction['coverage_rate']:.1%})",
        f"- Selected-sentence ambiguity: {extraction['primary_sentence_ambiguous_count']}/{v2.v1.SAMPLE_COUNT} ({extraction['primary_sentence_ambiguous_rate']:.1%})",
        f"- Whole-reference multi-event: {extraction['whole_reference_multi_event_count']}/{v2.v1.SAMPLE_COUNT} ({extraction['whole_reference_multi_event_rate']:.1%})",
        f"- Raw `other`: {extraction['raw_other_count']}/{v2.v1.SAMPLE_COUNT}",
        f"- Rare rule: raw primary classes with fewer than {v2.v1.RARE_PRIMARY_THRESHOLD} samples merge into `other_rare`; no sample is excluded.",
        f"- Final class counts: `{json.dumps(extraction['final_class_counts'], sort_keys=True)}`",
        "",
        "The complete frozen dictionary, sentence audit, all hits, vetoes, and final labels are stored in `result.json`. There are no manual overrides.",
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
            "Every StandardScaler, PCA, and logistic classifier is fit only on each training fold. Five-fold stratification and randomized operations use seed 20260816. Full confusion matrices and fold records are in `result.json`.",
            "",
            "## Boundary",
            "",
            "These results apply only to one fixed 200-sample post-Q-Former/post-projector cache and automatically extracted reference labels. They do not establish video factual correctness, causal event encoding, full-test performance, or raw Q-Former-state separability.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_v3_result() -> None:
    result_path = OUTPUT_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["experiment"] = "reference_event_separability_fixed_200_v3"
    result["scope"]["purpose"] = (
        "Run the preregistered v3 minimal dictionary-repair separability test "
        "on fixed prefixes."
    )
    result["scope"]["preregistration"] = str(PREREGISTRATION)
    extraction = result["silver_label_extraction"]
    extraction["version"] = "v3_preregistered_20260816"
    result["implementation"] = {
        "execution_engine": str(
            v2.v1.REPO
            / "experiments/commentary_generation/event_separability_200_v2.py"
        ),
        "v3_changes": {
            "substitution_patterns_appended": SUBSTITUTION_ADDITIONS,
            "yellow_pattern_replaced": {
                "from": YELLOW_V2_PATTERN,
                "to": YELLOW_V3_PATTERN,
            },
            "foul_pattern_appended": FOUL_ADDITION,
            "pass_pattern_appended": PASS_ADDITION,
        },
        "protocol_identity_finalized_before_exit": True,
    }
    result["assertions"].pop("preregistered_dictionary_used", None)
    result["assertions"]["preregistered_v3_dictionary_used"] = True
    result["assertions"]["protocol_identity_finalized_before_exit"] = True
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(build_readme(result), encoding="utf-8")


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    if not PREREGISTRATION.is_file():
        raise FileNotFoundError(PREREGISTRATION)

    # Patch only process-local protocol constants/functions before delegating.
    v2.OUTPUT_DIR = OUTPUT_DIR
    v2.PREREGISTRATION = PREREGISTRATION
    v2.EVENT_DICTIONARY = EVENT_DICTIONARY
    v2.build_readme = build_readme

    exit_code = v2.main()
    if exit_code != 0:
        return exit_code
    finalize_v3_result()
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol": "v3_preregistered_20260816",
                "result_finalized": True,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
