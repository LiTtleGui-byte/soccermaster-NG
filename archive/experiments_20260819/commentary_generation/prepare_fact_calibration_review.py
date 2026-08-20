#!/usr/bin/env python3
"""Build a small, deliberately balanced fact-grounding calibration packet.

This script is CPU-only. It reads existing fixed-200 reports and writes one
review packet under reports/. It never opens or modifies the source videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/commentary_fact_calibration_24_20260818/items.json"
PREDICTIONS = ROOT / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1/predictions.jsonl"
EVENT_AUDIT = ROOT / "reports/commentary_event_separability_200_20260817_v3_match_grouped/result.json"
REFERENCE_AUDIT = ROOT / "reports/commentary_reference_audit_3256_20260814/reviews.jsonl"
ANNOTATION_MANIFEST = ROOT / "reports/commentary_video_event_annotation_200_20260816/annotation_manifest.json"

BASE_TARGETS = {
    "corner": 2,
    "cross": 2,
    "foul_or_free_kick": 2,
    "pass_or_build_up": 2,
    "shot_or_save": 2,
    "substitution": 2,
    "yellow_card": 2,
    "other": 2,
    "goal": 1,
    "injury": 1,
    "offside": 1,
    "penalty": 1,
    "restart": 1,
    "throw_in": 1,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_order_key(row: dict[str, Any], salt: str) -> str:
    raw = f"{salt}:{row['dataset_index']}:{row['match_id']}".encode()
    return hashlib.sha256(raw).hexdigest()


def candidate_order(dataset_index: int) -> list[str]:
    digest = hashlib.sha256(f"candidate-order:{dataset_index}".encode()).digest()
    return ["historical", "e1_best"] if digest[0] % 2 == 0 else ["e1_best", "historical"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions = {r["dataset_index"]: r for r in read_jsonl(PREDICTIONS)}
    event_rows = json.loads(EVENT_AUDIT.read_text(encoding="utf-8"))["silver_label_extraction"]["per_sample_audit"]
    event_audit = {r["dataset_index"]: r for r in event_rows}
    reference_audit = {r["dataset_index"]: r for r in read_jsonl(REFERENCE_AUDIT)}
    manifest = json.loads(ANNOTATION_MANIFEST.read_text(encoding="utf-8"))
    annotation_ids = {r["dataset_index"]: r["annotation_id"] for r in manifest["linkage"]}

    expected = set(predictions)
    for name, mapping in (
        ("event audit", event_audit),
        ("reference audit", reference_audit),
        ("annotation manifest", annotation_ids),
    ):
        missing = expected - set(mapping)
        if missing:
            raise RuntimeError(f"{name} misses fixed-200 dataset indices: {sorted(missing)[:10]}")

    rows: list[dict[str, Any]] = []
    for dataset_index, pred in predictions.items():
        video_path = Path(pred["video_path"])
        event = event_audit[dataset_index]
        audit = reference_audit[dataset_index]
        rows.append(
            {
                "annotation_id": annotation_ids[dataset_index],
                "dataset_index": dataset_index,
                "fixed200_ordinal": pred["ordinal"],
                "video_path": str(video_path),
                "match_id": f"{video_path.parent.parent.name}/{video_path.parent.name}",
                "reference_commentary": pred["reference_commentary"],
                "historical_prediction": pred["predictions"]["historical_beam_sampling"]["text"],
                "e1_best_prediction": pred["predictions"]["nucleus_t070_p090"]["text"],
                "reference_derived_event": event["raw_primary"],
                "reference_is_multi_event": bool(event["whole_reference_multi_event"]),
                "reference_primary_is_ambiguous": bool(event["primary_sentence_ambiguous"]),
                "existing_reference_audit": {
                    "verdict": audit["verdict"],
                    "confidence": audit["confidence"],
                    "reason": audit["reason"],
                    "source": audit["review_source"],
                },
            }
        )

    selected: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    used_matches: set[str] = set()

    def choose(pool: list[dict[str, Any]], salt: str, preferred_verdict: str | None = None) -> dict[str, Any]:
        options = [r for r in pool if r["dataset_index"] not in used_indices]
        if not options:
            raise RuntimeError(f"no candidate remains for {salt}")

        def rank(r: dict[str, Any]) -> tuple[int, int, str]:
            verdict = r["existing_reference_audit"]["verdict"]
            return (
                0 if r["match_id"] not in used_matches else 1,
                0 if preferred_verdict is not None and verdict == preferred_verdict else 1,
                stable_order_key(r, salt),
            )

        return min(options, key=rank)

    # Common strata get one accurate and one partial example when possible.
    for label, count in BASE_TARGETS.items():
        pool = [r for r in rows if r["reference_derived_event"] == label]
        for occurrence in range(count):
            preferred = ("accurate", "partial")[occurrence % 2] if count > 1 else (
                "accurate" if len(selected) % 2 == 0 else "partial"
            )
            row = choose(pool, f"base:{label}:{occurrence}", preferred)
            row = dict(row, selection_stratum=label, selection_role="event_balance")
            selected.append(row)
            used_indices.add(row["dataset_index"])
            used_matches.add(row["match_id"])

    # Two deliberately difficult examples make bad/ambiguous references visible
    # during rubric calibration. They are not prevalence estimates.
    challenge_pool = [
        r for r in rows
        if r["existing_reference_audit"]["verdict"] == "wrong"
        or r["reference_primary_is_ambiguous"]
        or r["reference_is_multi_event"]
    ]
    for occurrence in range(2):
        row = choose(challenge_pool, f"challenge:{occurrence}", "wrong")
        reason = "reference_audit_wrong" if row["existing_reference_audit"]["verdict"] == "wrong" else "reference_ambiguous_or_multi_event"
        row = dict(row, selection_stratum=row["reference_derived_event"], selection_role=reason)
        selected.append(row)
        used_indices.add(row["dataset_index"])
        used_matches.add(row["match_id"])

    if len(selected) != 24:
        raise RuntimeError(f"expected 24 selected samples, got {len(selected)}")

    output_items = []
    for review_position, row in enumerate(selected, 1):
        order = candidate_order(row["dataset_index"])
        texts = {
            "historical": row.pop("historical_prediction"),
            "e1_best": row.pop("e1_best_prediction"),
        }
        row["review_position"] = review_position
        row["anonymous_candidates"] = [
            {"candidate_id": "candidate_a", "source": order[0], "text": texts[order[0]]},
            {"candidate_id": "candidate_b", "source": order[1], "text": texts[order[1]]},
        ]
        output_items.append(row)

    payload = {
        "schema_version": 1,
        "purpose": "rubric_calibration_only_not_holdout",
        "selection_note": "Balanced by reference-derived event strata, enriched with two difficult references, and selected to prefer distinct matches. Not representative prevalence.",
        "source_population": "fixed200_development_diagnostic_set",
        "sample_count": len(output_items),
        "event_counts": dict(sorted(Counter(r["selection_stratum"] for r in output_items).items())),
        "reference_audit_counts": dict(sorted(Counter(r["existing_reference_audit"]["verdict"] for r in output_items).items())),
        "unique_match_count": len({r["match_id"] for r in output_items}),
        "items": output_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("sample_count", "event_counts", "reference_audit_counts", "unique_match_count")}, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
