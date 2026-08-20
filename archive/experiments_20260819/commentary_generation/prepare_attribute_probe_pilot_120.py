#!/usr/bin/env python3
"""Prepare a blind 120-clip development packet for attribute probes.

CPU-only: reads existing local JSON/JSONL reports, writes a coordinator manifest
and a blank annotation CSV, and never opens source videos.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "reports/commentary_attribute_probe_pilot_120_20260818"
PREDICTIONS = ROOT / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1/predictions.jsonl"
EVENT_AUDIT = ROOT / "reports/commentary_event_separability_200_20260817_v3_match_grouped/result.json"
ANNOTATION_MANIFEST = ROOT / "reports/commentary_video_event_annotation_200_20260816/annotation_manifest.json"
SAMPLE_COUNT = 120

OBSERVABILITY = ("clear", "partial", "not_observable")
PHASES = ("live", "stopped", "replay", "mixed", "indeterminate")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_key(row: dict[str, Any], salt: str) -> str:
    raw = f"{salt}:{row['dataset_index']}:{row['match_id']}".encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_DIR}")

    predictions = {row["dataset_index"]: row for row in read_jsonl(PREDICTIONS)}
    audit_payload = json.loads(EVENT_AUDIT.read_text(encoding="utf-8"))
    audits = {
        row["dataset_index"]: row
        for row in audit_payload["silver_label_extraction"]["per_sample_audit"]
    }
    annotation_payload = json.loads(ANNOTATION_MANIFEST.read_text(encoding="utf-8"))
    annotation_ids = {row["dataset_index"]: row["annotation_id"] for row in annotation_payload["linkage"]}

    if len(predictions) != 200 or set(predictions) != set(audits) or set(predictions) != set(annotation_ids):
        raise RuntimeError("fixed-200 prediction/audit/annotation identities do not match")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dataset_index, prediction in predictions.items():
        video_path = Path(prediction["video_path"])
        audit = audits[dataset_index]
        row = {
            "annotation_id": annotation_ids[dataset_index],
            "dataset_index": dataset_index,
            "fixed200_ordinal": prediction["ordinal"],
            "video_path": str(video_path),
            "match_id": f"{video_path.parent.parent.name}/{video_path.parent.name}",
            "silver_event": audit["raw_primary"],
            "silver_multi_event": bool(audit["whole_reference_multi_event"]),
            "silver_primary_ambiguous": bool(audit["primary_sentence_ambiguous"]),
        }
        by_label[row["silver_event"]].append(row)

    for label, rows in by_label.items():
        rows.sort(key=lambda row: stable_key(row, f"attribute-pilot:{label}"))

    selected: list[dict[str, Any]] = []
    used_matches: Counter[str] = Counter()
    # Balanced round-robin first; later rounds naturally draw more examples
    # only from strata that still contain candidates.
    while len(selected) < SAMPLE_COUNT:
        made_progress = False
        for label in sorted(by_label):
            candidates = [row for row in by_label[label] if row not in selected]
            if not candidates:
                continue
            candidates.sort(key=lambda row: (used_matches[row["match_id"]], stable_key(row, f"pick:{label}")))
            chosen = candidates[0]
            selected.append(chosen)
            used_matches[chosen["match_id"]] += 1
            made_progress = True
            if len(selected) == SAMPLE_COUNT:
                break
        if not made_progress:
            raise RuntimeError("population exhausted before reaching requested sample count")

    selected.sort(key=lambda row: stable_key(row, "blind-order"))
    coordinator = {
        "schema_version": 1,
        "status": "awaiting_video_fact_annotation",
        "purpose": "development_attribute_probe_not_holdout",
        "sample_count": len(selected),
        "unique_match_count": len({row["match_id"] for row in selected}),
        "silver_event_counts": dict(sorted(Counter(row["silver_event"] for row in selected).items())),
        "selection": "deterministic_stratified_round_robin_with_match_reuse_minimization",
        "source_population": "fixed200_development_diagnostic_set",
        "annotation_contract": {
            "multi_value_separator": ";",
            "observability_values": list(OBSERVABILITY),
            "phase_values": list(PHASES),
            "empty_is_negative": False,
            "loss_mask_required_for_unobservable_or_not_applicable": True,
        },
        "items": [dict(row, blind_position=index) for index, row in enumerate(selected, 1)],
    }

    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "coordinator_manifest.json").write_text(
        json.dumps(coordinator, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fields = [
        "blind_position", "annotation_id", "dataset_index", "match_id", "video_path",
        "event_values", "event_observability", "action_values", "action_observability",
        "result_value", "result_observability", "phase", "phase_observability",
        "evidence_note", "annotation_complete",
    ]
    with (OUTPUT_DIR / "video_fact_annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, row in enumerate(selected, 1):
            writer.writerow({
                "blind_position": position,
                "annotation_id": row["annotation_id"],
                "dataset_index": row["dataset_index"],
                "match_id": row["match_id"],
                "video_path": row["video_path"],
                "event_values": "",
                "event_observability": "",
                "action_values": "",
                "action_observability": "",
                "result_value": "",
                "result_observability": "",
                "phase": "",
                "phase_observability": "",
                "evidence_note": "",
                "annotation_complete": "no",
            })

    status = {
        "status": "GPU_APPROVAL_REQUIRED_AFTER_VIDEO_FACT_ANNOTATION",
        "cpu_packet_prepared": True,
        "video_opened": False,
        "torch_imported": False,
        "gpu_used": False,
        "feature_extraction_started": False,
        "probe_training_started": False,
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": coordinator["sample_count"],
        "unique_match_count": coordinator["unique_match_count"],
        "silver_event_counts": coordinator["silver_event_counts"],
        "status": status["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
