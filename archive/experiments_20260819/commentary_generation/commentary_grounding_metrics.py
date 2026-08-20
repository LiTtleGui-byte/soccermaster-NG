#!/usr/bin/env python3
"""Paired, match-clustered metrics for scored commentary grounding records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable


SLOTS = ("action", "result", "actor_role", "target_role")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_rows(rows: list[dict[str, Any]], expected_condition: str) -> None:
    seen: set[int] = set()
    for row in rows:
        index = int(row["dataset_index"])
        if index in seen:
            raise ValueError(f"Duplicate dataset_index {index}")
        seen.add(index)
        if str(row["condition"]) != expected_condition:
            raise ValueError(
                f"Expected condition {expected_condition!r}, got {row['condition']!r}"
            )
        if not str(row["match_id"]):
            raise ValueError(f"Empty match_id at {index}")
        core = row["core_event_error"]
        fluency = row["fluency_degraded"]
        if core not in (0, 1, None) or fluency not in (0, 1, None):
            raise ValueError(f"Invalid binary/null value at {index}")
        if int(row["unsupported_claim_count"]) > int(row["claim_count"]):
            raise ValueError(f"Unsupported claims exceed total claims at {index}")
        if set(row["slots"]) != set(SLOTS):
            raise ValueError(f"Unexpected slots at {index}")
        for slot in SLOTS:
            counts = row["slots"][slot]
            if set(counts) != {"tp", "fp", "fn"}:
                raise ValueError(f"Invalid {slot} counts at {index}")
            if any(int(counts[key]) < 0 for key in ("tp", "fp", "fn")):
                raise ValueError(f"Negative {slot} count at {index}")


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    core_values = [int(row["core_event_error"]) for row in selected if row["core_event_error"] is not None]
    fluency_values = [int(row["fluency_degraded"]) for row in selected if row["fluency_degraded"] is not None]
    unsupported = sum(int(row["unsupported_claim_count"]) for row in selected)
    claims = sum(int(row["claim_count"]) for row in selected)

    slot_metrics: dict[str, Any] = {}
    slot_f1_values: list[float] = []
    for slot in SLOTS:
        tp = sum(int(row["slots"][slot]["tp"]) for row in selected)
        fp = sum(int(row["slots"][slot]["fp"]) for row in selected)
        fn = sum(int(row["slots"][slot]["fn"]) for row in selected)
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp / denominator) if denominator else None
        if f1 is not None:
            slot_f1_values.append(f1)
        slot_metrics[slot] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1}

    return {
        "row_count": len(selected),
        "match_count": len({str(row["match_id"]) for row in selected}),
        "core_event_error_rate": safe_rate(sum(core_values), len(core_values)),
        "core_event_observable_count": len(core_values),
        "slot_macro_f1": (
            sum(slot_f1_values) / len(slot_f1_values) if slot_f1_values else None
        ),
        "slots": slot_metrics,
        "unsupported_claim_rate": safe_rate(unsupported, claims),
        "unsupported_claim_count": unsupported,
        "claim_count": claims,
        "fluency_degraded_rate": safe_rate(sum(fluency_values), len(fluency_values)),
        "fluency_rated_count": len(fluency_values),
    }


def improvement(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    keys = {
        "core_event_error_rate": "lower",
        "slot_macro_f1": "higher",
        "unsupported_claim_rate": "lower",
        "fluency_degraded_rate": "lower",
    }
    result: dict[str, float] = {}
    for key, direction in keys.items():
        left = baseline[key]
        right = candidate[key]
        if left is None or right is None:
            raise ValueError(f"Cannot compare null metric {key}")
        result[key] = (left - right) if direction == "lower" else (right - left)
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def clustered_bootstrap(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_match[str(row["match_id"])].append(row)
    for row in candidate_rows:
        candidate_by_match[str(row["match_id"])].append(row)
    matches = sorted(baseline_by_match)
    if matches != sorted(candidate_by_match):
        raise ValueError("Baseline/candidate match sets differ")
    if len(matches) < 2:
        raise ValueError("At least two match clusters are required")

    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        drawn = [rng.choice(matches) for _ in matches]
        baseline_sample = [row for group in drawn for row in baseline_by_match[group]]
        candidate_sample = [row for group in drawn for row in candidate_by_match[group]]
        delta = improvement(aggregate(baseline_sample), aggregate(candidate_sample))
        for key, value in delta.items():
            samples[key].append(value)

    return {
        key: {
            "improvement": improvement(
                aggregate(baseline_rows), aggregate(candidate_rows)
            )[key],
            "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
        }
        for key, values in samples.items()
    }


def compare(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_condition: str,
    candidate_condition: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    validate_rows(baseline_rows, baseline_condition)
    validate_rows(candidate_rows, candidate_condition)
    baseline_ids = {int(row["dataset_index"]) for row in baseline_rows}
    candidate_ids = {int(row["dataset_index"]) for row in candidate_rows}
    if baseline_ids != candidate_ids:
        raise ValueError("Baseline/candidate dataset_index sets differ")
    baseline_matches = {
        int(row["dataset_index"]): str(row["match_id"]) for row in baseline_rows
    }
    candidate_matches = {
        int(row["dataset_index"]): str(row["match_id"]) for row in candidate_rows
    }
    if baseline_matches != candidate_matches:
        raise ValueError("Baseline/candidate match identity differs")

    base = aggregate(baseline_rows)
    candidate = aggregate(candidate_rows)
    bootstrap = clustered_bootstrap(
        baseline_rows, candidate_rows, resamples=resamples, seed=seed
    )
    core = bootstrap["core_event_error_rate"]
    baseline_core = base["core_event_error_rate"]
    if baseline_core in (None, 0):
        relative_core_reduction = None
    else:
        relative_core_reduction = core["improvement"] / baseline_core
    success = {
        "core_effect_is_practically_meaningful": (
            core["improvement"] >= 0.05
            or (
                relative_core_reduction is not None
                and relative_core_reduction >= 0.20
            )
        ),
        "core_ci_excludes_zero": core["ci95"][0] > 0,
        "slot_macro_f1_not_lower": bootstrap["slot_macro_f1"]["improvement"] >= 0,
        "fluency_degradation_not_material": bootstrap["fluency_degraded_rate"]["improvement"] >= -0.02,
    }
    success["passed"] = all(success.values())
    return {
        "baseline_condition": baseline_condition,
        "candidate_condition": candidate_condition,
        "paired_sample_count": len(baseline_ids),
        "match_cluster_count": base["match_count"],
        "baseline": base,
        "candidate": candidate,
        "paired_match_cluster_bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "positive_means_candidate_is_better": True,
            "metrics": bootstrap,
        },
        "relative_core_event_error_reduction": relative_core_reduction,
        "success_criteria": success,
        "multiple_comparison_policy": "Apply Holm correction when comparing more than one candidate module.",
    }


def synthetic_row(index: int, match: str, condition: str, error: int) -> dict[str, Any]:
    return {
        "dataset_index": index,
        "match_id": match,
        "condition": condition,
        "core_event_error": error,
        "slots": {
            slot: {"tp": 1 if error == 0 else 0, "fp": error, "fn": error}
            for slot in SLOTS
        },
        "unsupported_claim_count": error,
        "claim_count": 2,
        "fluency_degraded": 0,
    }


def self_test() -> int:
    baseline = [
        synthetic_row(0, "m1", "baseline", 1),
        synthetic_row(1, "m1", "baseline", 1),
        synthetic_row(2, "m2", "baseline", 1),
        synthetic_row(3, "m2", "baseline", 0),
    ]
    candidate = [
        synthetic_row(0, "m1", "candidate", 0),
        synthetic_row(1, "m1", "candidate", 0),
        synthetic_row(2, "m2", "candidate", 0),
        synthetic_row(3, "m2", "candidate", 0),
    ]
    result = compare(
        baseline,
        candidate,
        baseline_condition="baseline",
        candidate_condition="candidate",
        resamples=200,
        seed=20260817,
    )
    assert result["baseline"]["core_event_error_rate"] == 0.75
    assert result["candidate"]["core_event_error_rate"] == 0.0
    assert result["paired_match_cluster_bootstrap"]["metrics"]["core_event_error_rate"]["improvement"] == 0.75
    print(json.dumps({"status": "passed", "self_test": True}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-condition")
    parser.add_argument("--candidate-condition")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    required = (
        args.baseline,
        args.candidate,
        args.baseline_condition,
        args.candidate_condition,
        args.output,
    )
    if any(value is None for value in required):
        parser.error("comparison mode requires baseline, candidate, conditions, and output")
    if args.resamples < 1000:
        parser.error("at least 1000 bootstrap resamples are required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = compare(
        load_jsonl(args.baseline),
        load_jsonl(args.candidate),
        baseline_condition=args.baseline_condition,
        candidate_condition=args.candidate_condition,
        resamples=args.resamples,
        seed=args.seed,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
