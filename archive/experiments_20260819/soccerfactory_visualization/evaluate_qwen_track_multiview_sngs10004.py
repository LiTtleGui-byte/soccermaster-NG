#!/usr/bin/env python3
"""Evaluate frozen track-level multi-view Qwen role predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004"
RAW = REPORT / "track_multiview_raw_predictions.json"
CANDIDATES = REPORT / "track_multiview_candidates.json"
EVALUATION = REPORT / "track_multiview_evaluation.json"
ANNOTATIONS = REPO / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json"
ROLES = ("player", "referee", "goalkeeper", "other")
SPECIFIC = {"referee", "goalkeeper"}
BROAD = {"referee", "goalkeeper", "other"}
KNOWN = {"outfield_player", "goalkeeper", "referee_or_staff"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def freeze_candidates() -> dict[str, Any]:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    if raw["manual_annotations_read"] is not False or raw["tracks"] != 49:
        raise AssertionError("Raw prediction contract mismatch")
    rows = []
    for prediction in raw["predictions"]:
        probabilities = {
            role: prediction["candidate_role_scores"][f"{role}_normalized_probability"]
            for role in ROLES
        }
        score_role = max(ROLES, key=probabilities.get)
        rows.append(
            {
                "track_id": int(prediction["track_id"]),
                "views": int(prediction["views"]),
                "generation_role": prediction["normalized_parsed_role"],
                "candidate_score_role": score_role,
                "candidate_probabilities": probabilities,
                "raw_output_text": prediction["raw_output_text"],
            }
        )
    value = {
        "status": "candidate_complete_evaluation_required",
        "manual_annotations_read": False,
        "policy_fixed_before_label_read": {
            "primary": "exact normalized generated role",
            "specific_nonplayer": sorted(SPECIFIC),
            "broad_nonplayer": sorted(BROAD),
            "production_boundary": "specific recall >=0.90 and outfield false rejection <=0.15",
            "improvement_boundary": "more than 3/8 specific true positives and at most 5/39 outfield false rejects",
        },
        "tracks": rows,
    }
    atomic_json(CANDIDATES, value)
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
        "recall": len(tp) / len(truth),
        "outfield_false_rejection_rate": len(fp) / len(outfield),
        "true_positive_track_ids": sorted(tp),
        "false_positive_track_ids": sorted(fp),
        "false_negative_track_ids": sorted(fn),
    }


def evaluate(candidate: dict[str, Any]) -> dict[str, Any]:
    labels = {
        int(row["track_id"]): row["role"]
        for row in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["labels"]
    }
    rows = {row["track_id"]: row for row in candidate["tracks"]}
    if set(labels) != set(rows):
        raise AssertionError("Candidate/annotation identity mismatch")
    universe = {track_id for track_id, role in labels.items() if role in KNOWN}
    truth = {
        track_id
        for track_id in universe
        if labels[track_id] in {"goalkeeper", "referee_or_staff"}
    }
    results = {}
    for method in ("generation_role", "candidate_score_role"):
        results[method] = {
            "specific_nonplayer": metrics(
                truth, {track_id for track_id in universe if rows[track_id][method] in SPECIFIC}, universe
            ),
            "broad_nonplayer": metrics(
                truth, {track_id for track_id in universe if rows[track_id][method] in BROAD}, universe
            ),
        }
    primary = results["generation_role"]["specific_nonplayer"]
    if primary["recall"] >= 0.90 and primary["outfield_false_rejection_rate"] <= 0.15:
        verdict = "track_multiview_promising_requires_new_match_validation"
    elif primary["true_positive"] > 3 and primary["false_positive"] <= 5:
        verdict = "track_multiview_improves_specific_role_signal"
    else:
        verdict = "track_multiview_does_not_improve_single_view_baseline"
    details = [
        {
            "track_id": track_id,
            "manual_role": labels[track_id],
            "generation_role": rows[track_id]["generation_role"],
            "candidate_score_role": rows[track_id]["candidate_score_role"],
            "views": rows[track_id]["views"],
        }
        for track_id in sorted(universe)
    ]
    return {
        "status": "passed",
        "verdict": verdict,
        "candidate_file_written_before_label_read": str(CANDIDATES),
        "known_role_tracks": len(universe),
        "manual_nonplayer_tracks": len(truth),
        "manual_outfield_tracks": len(universe - truth),
        "baselines_same_match": {
            "prtreid_specific": {"true_positive": 0, "false_positive": 0},
            "qwen_independent_views_majority_specific": {"true_positive": 3, "false_positive": 0},
        },
        "multiview": results,
        "track_details": details,
        "interpretation_boundary": [
            "The frozen frames, boxes, tracks, model, and candidate labels match the single-view experiment.",
            "The changed variable is joint multi-view prompting and its track-level instruction.",
            "No score threshold was selected from manual labels.",
            "This already labeled match is a development diagnostic, not a benchmark.",
        ],
    }


def main() -> None:
    if CANDIDATES.exists() or EVALUATION.exists():
        raise FileExistsError("Refusing to overwrite multiview candidates/evaluation")
    candidate = freeze_candidates()
    result = evaluate(candidate)
    atomic_json(EVALUATION, result)
    print(json.dumps({"status": "passed", "verdict": result["verdict"], "evaluation": str(EVALUATION)}))


if __name__ == "__main__":
    main()
