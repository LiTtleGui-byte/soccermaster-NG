#!/usr/bin/env python3
"""Freeze track predictions, then evaluate the SNGS-10004 Qwen role swap."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004"
RAW = REPORT / "raw_predictions.json"
TRACKS = REPORT / "track_predictions.json"
EVALUATION = REPORT / "evaluation.json"
ANNOTATIONS = REPO / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json"
ROLES = ("player", "referee", "goalkeeper", "other")
SPECIFIC_NONPLAYER = {"referee", "goalkeeper"}
BROAD_NONPLAYER = {"referee", "goalkeeper", "other"}
KNOWN_MANUAL_ROLES = {"outfield_player", "goalkeeper", "referee_or_staff"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def majority_or_unknown(values: list[str | None]) -> str:
    counts = Counter(value for value in values if value in ROLES)
    if not counts:
        return "unknown"
    best = max(counts.values())
    winners = [role for role in ROLES if counts[role] == best]
    return winners[0] if len(winners) == 1 else "unknown"


def build_track_candidates() -> dict[str, Any]:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    if raw["manual_annotations_read"] is not False or raw["samples"] != 137:
        raise AssertionError("Raw inference contract mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in raw["predictions"]:
        grouped.setdefault(int(row["track_id"]), []).append(row)
    if len(grouped) != 49:
        raise AssertionError(f"Expected 49 tracks, got {len(grouped)}")

    tracks = []
    for track_id, views in sorted(grouped.items()):
        views.sort(key=lambda row: row["view_ordinal"])
        labels = [row["normalized_parsed_role"] for row in views]
        probabilities = {
            role: float(
                np.mean(
                    [
                        row["candidate_role_scores"][f"{role}_normalized_probability"]
                        for row in views
                    ]
                )
            )
            for role in ROLES
        }
        score_order = sorted(ROLES, key=lambda role: probabilities[role], reverse=True)
        tracks.append(
            {
                "track_id": track_id,
                "views": len(views),
                "view_roles": labels,
                "view_role_counts": dict(Counter(labels)),
                "generation_majority_role": majority_or_unknown(labels),
                "mean_candidate_probabilities": probabilities,
                "candidate_score_role": score_order[0],
                "candidate_score_margin": float(
                    probabilities[score_order[0]] - probabilities[score_order[1]]
                ),
                "raw_outputs": [row["raw_output_text"] for row in views],
            }
        )
    value = {
        "status": "candidate_complete_evaluation_required",
        "schema_version": 1,
        "raw_predictions": str(RAW),
        "manual_annotations_read": False,
        "aggregation_fixed_before_manual_evaluation": {
            "primary": "unique majority of normalized generated labels; ties become unknown",
            "diagnostic": "argmax of mean four-candidate normalized sequence likelihood",
            "specific_nonplayer": sorted(SPECIFIC_NONPLAYER),
            "broad_nonplayer": sorted(BROAD_NONPLAYER),
        },
        "tracks": tracks,
    }
    atomic_json(TRACKS, value)
    return value


def metrics(truth: set[int], prediction: set[int], universe: set[int]) -> dict[str, Any]:
    tp, fp = truth & prediction, prediction - truth
    fn, tn = truth - prediction, universe - truth - prediction
    outfield = universe - truth
    return {
        "true_positive": len(tp),
        "false_positive": len(fp),
        "false_negative": len(fn),
        "true_negative": len(tn),
        "precision": len(tp) / (len(tp) + len(fp)) if tp or fp else None,
        "recall": len(tp) / len(truth) if truth else None,
        "outfield_false_rejection_rate": len(fp) / len(outfield) if outfield else None,
        "true_positive_track_ids": sorted(tp),
        "false_positive_track_ids": sorted(fp),
        "false_negative_track_ids": sorted(fn),
    }


def evaluate(candidate: dict[str, Any]) -> dict[str, Any]:
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    labels = {int(row["track_id"]): row["role"] for row in annotations["labels"]}
    rows = {int(row["track_id"]): row for row in candidate["tracks"]}
    if set(labels) != set(rows):
        raise AssertionError("Candidate/annotation track mismatch")
    universe = {
        track_id for track_id, role in labels.items() if role in KNOWN_MANUAL_ROLES
    }
    truth = {
        track_id
        for track_id in universe
        if labels[track_id] in {"goalkeeper", "referee_or_staff"}
    }

    methods = {}
    for method in ("generation_majority_role", "candidate_score_role"):
        specific = {
            track_id for track_id in universe if rows[track_id][method] in SPECIFIC_NONPLAYER
        }
        broad = {
            track_id for track_id in universe if rows[track_id][method] in BROAD_NONPLAYER
        }
        methods[method] = {
            "specific_nonplayer": metrics(truth, specific, universe),
            "broad_nonplayer": metrics(truth, broad, universe),
        }

    primary = methods["generation_majority_role"]["broad_nonplayer"]
    if primary["recall"] >= 0.90 and primary["outfield_false_rejection_rate"] <= 0.15:
        verdict = "qwen_swap_promising_requires_new_match_validation"
    elif primary["recall"] > 0.0:
        verdict = "qwen_swap_improves_recall_but_not_role_gate"
    else:
        verdict = "qwen_swap_does_not_fix_role_failure"
    details = []
    for track_id in sorted(universe):
        row = rows[track_id]
        details.append(
            {
                "track_id": track_id,
                "manual_role": labels[track_id],
                "generation_majority_role": row["generation_majority_role"],
                "candidate_score_role": row["candidate_score_role"],
                "view_roles": row["view_roles"],
                "candidate_score_margin": row["candidate_score_margin"],
            }
        )
    return {
        "status": "passed",
        "verdict": verdict,
        "candidate_file_written_before_label_read": str(TRACKS),
        "manual_annotations": str(ANNOTATIONS),
        "known_role_tracks": len(universe),
        "manual_nonplayer_tracks": len(truth),
        "manual_outfield_tracks": len(universe - truth),
        "prtreid_same_match_baseline": {
            "specific_nonplayer_true_positive": 0,
            "specific_nonplayer_false_positive": 0,
            "specific_nonplayer_false_negative": 8,
            "source": "reports/g10/20260819_role_confidence_audit/result.json",
        },
        "qwen_methods": methods,
        "track_details": details,
        "interpretation_boundary": [
            "The causal comparison fixes match, detections, tracks, and selected views; only role inference changes.",
            "The 49-track match was already manually labeled and is a development diagnostic, not a benchmark.",
            "The candidate score is sequence likelihood over four fixed strings, not a calibrated probability.",
            "No threshold was selected from the manual labels.",
        ],
    }


def main() -> None:
    if TRACKS.exists() or EVALUATION.exists():
        raise FileExistsError("Refusing to overwrite track candidates or evaluation")
    candidate = build_track_candidates()
    result = evaluate(candidate)
    atomic_json(EVALUATION, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "evaluation": str(EVALUATION),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
