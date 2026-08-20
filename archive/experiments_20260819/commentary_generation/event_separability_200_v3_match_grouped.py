#!/usr/bin/env python3
"""Post-hoc match-grouped correction of the fixed-200 v3 probe.

Reads only existing local prefix/prediction artifacts. The feature pipeline and
v3 silver labels remain unchanged; only the cross-validation grouping changes.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from research.experiments.commentary_generation import event_separability_200_v2 as v2
from research.experiments.commentary_generation import event_separability_200_v3 as v3


REPO = v2.v1.REPO
OUTPUT_DIR = REPO / "reports/commentary_event_separability_200_20260817_v3_match_grouped"
PROTOCOL = (
    REPO
    / "experiments/commentary_generation/"
    "MATCH_GROUPED_V3_DIAGNOSTIC_PROTOCOL_20260817.md"
)
ORIGINAL_V3_RESULT = (
    REPO / "reports/commentary_event_separability_200_20260816_v3/result.json"
)
_MATCH_GROUPS: list[str] = []


def match_group(video_path: str) -> str:
    """Return the match directory, preserving league/season identity."""
    path = Path(video_path)
    if path.name == "" or path.parent == path:
        raise ValueError(f"Cannot derive match group from {video_path!r}")
    return str(path.parent)


class FixedManifestStratifiedGroupKFold:
    """Drop-in splitter for the v2 engine, with groups fixed by the manifest."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.inner = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if len(_MATCH_GROUPS) != len(y):
            raise RuntimeError(
                f"Expected {len(y)} fixed groups, found {len(_MATCH_GROUPS)}"
            )
        yield from self.inner.split(x, y, groups=np.asarray(_MATCH_GROUPS))


def make_folds(labels: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=v2.v1.CV_SPLITS,
        shuffle=True,
        random_state=v2.v1.SEED,
    )
    return list(
        splitter.split(
            np.zeros(len(labels)),
            np.asarray(labels, dtype=object),
            groups=np.asarray(_MATCH_GROUPS, dtype=object),
        )
    )


def fold_audit(
    folds: list[tuple[np.ndarray, np.ndarray]], labels: list[str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_test: list[int] = []
    for number, (train_indices, test_indices) in enumerate(folds, 1):
        train_groups = {_MATCH_GROUPS[index] for index in train_indices}
        test_groups = {_MATCH_GROUPS[index] for index in test_indices}
        overlap = sorted(train_groups & test_groups)
        if overlap:
            raise RuntimeError(f"Fold {number} leaks match groups: {overlap}")
        seen_test.extend(int(index) for index in test_indices)
        records.append(
            {
                "fold": number,
                "train_count": int(len(train_indices)),
                "test_count": int(len(test_indices)),
                "train_match_count": len(train_groups),
                "test_match_count": len(test_groups),
                "match_overlap_count": 0,
                "train_class_counts": dict(
                    sorted(Counter(labels[index] for index in train_indices).items())
                ),
                "test_class_counts": dict(
                    sorted(Counter(labels[index] for index in test_indices).items())
                ),
            }
        )
    if sorted(seen_test) != list(range(v2.v1.SAMPLE_COUNT)):
        raise RuntimeError("Grouped folds do not test every sample exactly once")
    return records


def build_readme(result: dict[str, Any], original: dict[str, Any]) -> str:
    lines = [
        "# Fixed-200 v3 match-grouped event separability",
        "",
        "This is a post-hoc correction for match leakage in the original v3 probe. It keeps the existing reference-relative labels and cached post-Q-Former/post-projector prefixes; it does not establish video-grounded facts or causal module quality.",
        "",
        f"- Samples: {result['inputs']['sample_count']}",
        f"- Match groups: {result['cross_validation']['match_group_count']}",
        "- Split: five-fold StratifiedGroupKFold",
        "- Match overlap in every fold: 0",
        "",
        "## Corrected versus original v3",
        "",
        "| Representation | Original macro-F1 | Grouped macro-F1 | Original balanced accuracy | Grouped balanced accuracy | Grouped shuffled macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in result["representations"].items():
        old = original["representations"][name]["out_of_fold"]
        new = record["out_of_fold"]
        shuffled = record["fixed_seed_shuffled_training_labels_baseline"]
        lines.append(
            f"| {name} | {old['macro_f1']:.4f} | {new['macro_f1']:.4f} | "
            f"{old['balanced_accuracy']:.4f} | {new['balanced_accuracy']:.4f} | "
            f"{shuffled['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The fixed 200 clips cover 48 of the full test set's 49 matches and have already been used for experiment design. This result is a corrected development diagnostic, not a locked holdout or a final generalization estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize() -> None:
    result_path = OUTPUT_DIR / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_V3_RESULT.read_text(encoding="utf-8"))
    labels = [
        str(row["final_label"])
        for row in result["silver_label_extraction"]["per_sample_audit"]
    ]
    folds = make_folds(labels)
    audits = fold_audit(folds, labels)

    result["experiment"] = "reference_event_separability_fixed_200_v3_match_grouped"
    result["scope"]["purpose"] = (
        "Correct the known match leakage in the fixed-200 v3 development probe."
    )
    result["scope"]["protocol"] = str(PROTOCOL)
    result["scope"]["confirmatory"] = False
    result["silver_label_extraction"]["version"] = "v3_preregistered_20260816"
    result["cross_validation"].update(
        {
            "scheme": "StratifiedGroupKFold",
            "group_field": "parent match directory from manifest video_path",
            "match_group_count": len(set(_MATCH_GROUPS)),
            "shuffle": True,
            "seed": v2.v1.SEED,
            "leakage_control": (
                "Match groups are disjoint between train and test in every fold; "
                "scaler, PCA, and classifier remain fold-local."
            ),
            "fold_group_audit": audits,
        }
    )
    result["implementation"] = {
        "execution_engine": str(
            REPO / "experiments/commentary_generation/event_separability_200_v2.py"
        ),
        "label_dictionary": "frozen v3",
        "only_evaluation_change": "StratifiedKFold -> StratifiedGroupKFold by match",
        "original_v3_result": str(ORIGINAL_V3_RESULT),
    }
    result["assertions"].pop("preregistered_dictionary_used", None)
    result["assertions"].update(
        {
            "preregistered_v3_dictionary_used": True,
            "all_folds_match_disjoint": all(
                row["match_overlap_count"] == 0 for row in audits
            ),
            "all_samples_tested_once": True,
            "protocol_frozen_before_corrected_metrics": True,
        }
    )
    result["limitations"].extend(
        [
            "This correction was designed after the original v3 result was known.",
            "The 200 clips are a repeatedly used development set, not a holdout.",
            "Only 48 match groups are available, so grouped fold class balance may be uneven.",
        ]
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        build_readme(result, original), encoding="utf-8"
    )


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    if not ORIGINAL_V3_RESULT.is_file():
        raise FileNotFoundError(ORIGINAL_V3_RESULT)

    manifest = v2.v1.load_json(v2.v1.PREFIX_MANIFEST)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != v2.v1.SAMPLE_COUNT:
        raise RuntimeError("Prefix manifest does not contain 200 samples")
    _MATCH_GROUPS.extend(match_group(str(sample["video_path"])) for sample in samples)
    if len(set(_MATCH_GROUPS)) != 48:
        raise RuntimeError(
            f"Expected 48 fixed-200 match groups, found {len(set(_MATCH_GROUPS))}"
        )

    v2.OUTPUT_DIR = OUTPUT_DIR
    v2.PREREGISTRATION = PROTOCOL
    v2.EVENT_DICTIONARY = v3.EVENT_DICTIONARY
    v2.StratifiedKFold = FixedManifestStratifiedGroupKFold
    exit_code = v2.main()
    if exit_code != 0:
        return exit_code
    finalize()
    print(json.dumps({"status": "passed", "match_groups": 48}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
