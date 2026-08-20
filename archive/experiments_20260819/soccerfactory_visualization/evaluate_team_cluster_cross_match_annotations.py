#!/usr/bin/env python3
"""Evaluate SNGS-10001 saved ReID and color clusters against reviewed tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORT_DIR = REPO / "reports/g10/20260818_team_color_cross_match_sngs10001"
ANNOTATIONS = REPORT_DIR / "sngs10001_track_annotations.json"
PREDICTIONS = REPORT_DIR / "result.json"
FIRST_EVALUATION = REPO / "reports/g10/20260818_team_color_diagnostic/annotation_evaluation.json"
OUTPUT = REPORT_DIR / "annotation_evaluation.json"
TEAMS = ("blue", "claret")


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_clusters(
    ids: list[int],
    labels: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    covered = [track_id for track_id in ids if predictions[track_id][field] in (0, 1)]
    missing = [track_id for track_id in ids if predictions[track_id][field] not in (0, 1)]
    candidates = []
    for mapping in ({0: "blue", 1: "claret"}, {0: "claret", 1: "blue"}):
        correct_ids = [
            track_id
            for track_id in covered
            if mapping[int(predictions[track_id][field])] == labels[track_id]["team"]
        ]
        incorrect = [track_id for track_id in covered if track_id not in correct_ids]
        end_to_end_recall = {
            team: ratio(
                sum(track_id in correct_ids and labels[track_id]["team"] == team for track_id in ids),
                sum(labels[track_id]["team"] == team for track_id in ids),
            )
            for team in TEAMS
        }
        candidates.append(
            {
                "cluster_to_team": {str(key): value for key, value in mapping.items()},
                "covered_tracks": len(covered),
                "total_tracks": len(ids),
                "coverage": ratio(len(covered), len(ids)),
                "correct_covered_tracks": len(correct_ids),
                "conditional_accuracy": ratio(len(correct_ids), len(covered)),
                "end_to_end_accuracy_missing_counted_incorrect": ratio(len(correct_ids), len(ids)),
                "end_to_end_balanced_accuracy": sum(end_to_end_recall.values()) / len(TEAMS),
                "end_to_end_recall_by_team": end_to_end_recall,
                "incorrect_covered_track_ids": incorrect,
                "missing_prediction_track_ids": missing,
            }
        )
    return max(
        candidates,
        key=lambda item: (
            item["end_to_end_accuracy_missing_counted_incorrect"],
            item["end_to_end_balanced_accuracy"],
        ),
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    diagnostic = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    first = json.loads(FIRST_EVALUATION.read_text(encoding="utf-8"))
    rows = annotations["labels"]
    labels = {int(row["track_id"]): row for row in rows}
    predictions = {int(row["track_id"]): row for row in diagnostic["track_assignments"]}
    if annotations["video_id"] != "10001" or len(rows) != 40 or len(labels) != 40:
        raise AssertionError("Expected 40 unique reviewed SNGS-10001 tracks")
    if set(labels) != set(predictions) or any(not row["role"] or not row["team"] for row in rows):
        raise AssertionError("Annotation/prediction identity mismatch or incomplete labels")

    outfield = sorted(
        track_id
        for track_id, row in labels.items()
        if row["role"] == "outfield_player" and row["team"] in TEAMS
    )
    if len(outfield) != 34:
        raise AssertionError("Reviewed outfield scope changed")
    support = {team: sum(labels[track_id]["team"] == team for track_id in outfield) for team in TEAMS}
    majority_team = max(TEAMS, key=lambda team: support[team])
    reid = evaluate_clusters(outfield, labels, predictions, "saved_reid_cluster")
    color = evaluate_clusters(outfield, labels, predictions, "color_cluster_all")

    sensitivity_ids = [track_id for track_id in outfield if track_id != 15]
    color_without_low_visibility_15 = evaluate_clusters(
        sensitivity_ids, labels, predictions, "color_cluster_all"
    )

    known_role_ids = sorted(track_id for track_id, row in labels.items() if row["role"] != "uncertain")
    manual_nonplayer = {
        track_id
        for track_id in known_role_ids
        if labels[track_id]["role"] in {"goalkeeper", "referee_or_staff"}
    }
    predicted_nonplayer = {
        track_id
        for track_id in known_role_ids
        if predictions[track_id]["saved_final_role"] != "player"
    }
    tp = len(manual_nonplayer & predicted_nonplayer)
    fp = len(predicted_nonplayer - manual_nonplayer)
    fn = len(manual_nonplayer - predicted_nonplayer)
    tn = len(set(known_role_ids) - manual_nonplayer - predicted_nonplayer)

    first_scope = first["scopes"]["all_reviewed_outfield_tracks"]
    first_n = len(first_scope["track_ids"])
    first_reid_correct = first_scope["existing_reid"]["correct"]
    first_color_correct = first_scope["color_all_tracks"]["correct"]
    two_match_total = first_n + len(outfield)
    two_match_reid_covered = first_n + reid["covered_tracks"]
    two_match_reid_correct = first_reid_correct + reid["correct_covered_tracks"]
    two_match_color_correct = first_color_correct + color["correct_covered_tracks"]

    result = {
        "status": "passed",
        "scope": "predetermined second-match, track-level development diagnosis",
        "video_id": "10001",
        "annotation_file": str(ANNOTATIONS),
        "annotation_summary": {
            "tracks": len(labels),
            "roles": {
                role: sum(row["role"] == role for row in rows)
                for role in sorted({row["role"] for row in rows})
            },
            "team_labeled_outfield_tracks": len(outfield),
            "team_support": support,
        },
        "evaluation_policy": {
            "unit": "track",
            "team_scope": "outfield_player with blue or claret manual label",
            "cluster_mapping": "best binary cluster-to-team permutation within SNGS-10001",
            "missing_reid_policy": "report coverage and conditional accuracy; count missing as incorrect for end-to-end accuracy",
        },
        "majority_baseline": {
            "prediction_for_every_track": majority_team,
            "correct": support[majority_team],
            "accuracy": ratio(support[majority_team], len(outfield)),
            "balanced_accuracy": 0.5,
        },
        "existing_reid": reid,
        "color_all_tracks": color,
        "sensitivity_excluding_low_visibility_track_15": color_without_low_visibility_15,
        "saved_final_role_binary_diagnosis": {
            "known_role_tracks": len(known_role_ids),
            "manual_nonplayer_track_ids": sorted(manual_nonplayer),
            "predicted_nonplayer_track_ids": sorted(predicted_nonplayer),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "nonplayer_precision": ratio(tp, tp + fp),
            "nonplayer_recall": ratio(tp, tp + fn),
            "binary_accuracy": ratio(tp + tn, len(known_role_ids)),
        },
        "fixed_any_nonplayer_signal_filter": {
            "eligible_tracks": diagnostic["fixed_role_length_filter"]["eligible_tracks"],
            "status": diagnostic["fixed_role_length_filter"]["status"],
        },
        "two_match_descriptive_summary": {
            "warning": "Descriptive only: SNGS-10004 new enrichment and SNGS-10001 historical Step-3 have different run provenance.",
            "team_labeled_outfield_tracks": two_match_total,
            "reid_coverage": ratio(two_match_reid_covered, two_match_total),
            "reid_conditional_accuracy": ratio(two_match_reid_correct, two_match_reid_covered),
            "reid_end_to_end_accuracy": ratio(two_match_reid_correct, two_match_total),
            "color_coverage": 1.0,
            "color_accuracy": ratio(two_match_color_correct, two_match_total),
        },
        "interpretation_boundary": [
            "Track 15 is retained as manually labeled blue but its three target crops are very small and visually ambiguous.",
            "The sensitivity result excludes only track 15 and was specified after inspecting the sole color disagreement.",
            "Two matches are development diagnostics, not a benchmark or production-readiness claim.",
            "No labels or predictions were written back to either SoccerFactory archive.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "outfield": len(outfield),
                "reid_coverage": reid["coverage"],
                "reid_conditional_accuracy": reid["conditional_accuracy"],
                "reid_end_to_end_accuracy": reid["end_to_end_accuracy_missing_counted_incorrect"],
                "color_accuracy": color["conditional_accuracy"],
                "color_errors": color["incorrect_covered_track_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
