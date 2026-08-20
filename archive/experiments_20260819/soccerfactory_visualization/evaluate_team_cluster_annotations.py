#!/usr/bin/env python3
"""Evaluate cached team clusters against the manually reviewed track labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORT_DIR = REPO / "reports/g10/20260818_team_color_diagnostic"
ANNOTATIONS = REPORT_DIR / "sngs10004_track_annotations.json"
PREDICTIONS = REPORT_DIR / "result.json"
OUTPUT = REPORT_DIR / "annotation_evaluation.json"
TEAMS = ("blue", "claret")


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate(
    ids: list[int],
    labels: dict[int, dict[str, Any]],
    predictions: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    candidates = []
    for mapping in ({0: "blue", 1: "claret"}, {0: "claret", 1: "blue"}):
        pairs = [(labels[track_id]["team"], mapping[int(predictions[track_id][field])]) for track_id in ids]
        correct = sum(truth == predicted for truth, predicted in pairs)
        recall = {
            team: safe_ratio(
                sum(truth == team and predicted == team for truth, predicted in pairs),
                sum(truth == team for truth, _ in pairs),
            )
            for team in TEAMS
        }
        candidates.append(
            {
                "cluster_to_team": {str(key): value for key, value in mapping.items()},
                "correct": correct,
                "accuracy": safe_ratio(correct, len(ids)),
                "balanced_accuracy": sum(recall.values()) / len(recall),
                "recall_by_team": recall,
                "incorrect_track_ids": [
                    track_id
                    for track_id in ids
                    if labels[track_id]["team"] != mapping[int(predictions[track_id][field])]
                ],
            }
        )
    return max(candidates, key=lambda item: (item["accuracy"], item["balanced_accuracy"]))


def majority_baseline(ids: list[int], labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    support = {team: sum(labels[track_id]["team"] == team for track_id in ids) for team in TEAMS}
    predicted = max(TEAMS, key=lambda team: support[team])
    return {
        "prediction_for_every_track": predicted,
        "correct": support[predicted],
        "accuracy": safe_ratio(support[predicted], len(ids)),
        "balanced_accuracy": 0.5,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    diagnostic = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    label_rows = annotations["labels"]
    labels = {int(row["track_id"]): row for row in label_rows}
    predictions = {int(row["track_id"]): row for row in diagnostic["track_assignments"]}
    if len(label_rows) != 49 or len(labels) != 49 or set(labels) != set(predictions):
        raise AssertionError("Expected exactly the same 49 unique tracks in labels and predictions")
    if any(not row["role"] or not row["team"] for row in label_rows):
        raise AssertionError("Annotation file contains incomplete labels")

    outfield_ids = sorted(
        track_id
        for track_id, row in labels.items()
        if row["role"] == "outfield_player" and row["team"] in TEAMS
    )
    eligible_outfield_ids = [track_id for track_id in outfield_ids if labels[track_id]["eligible"]]
    if len(outfield_ids) != 39 or len(eligible_outfield_ids) != 28:
        raise AssertionError("Reviewed evaluation scope changed")

    role_known_ids = sorted(track_id for track_id, row in labels.items() if row["role"] != "uncertain")
    manual_nonplayer = {
        track_id
        for track_id in role_known_ids
        if labels[track_id]["role"] in {"goalkeeper", "referee_or_staff"}
    }
    detected_nonplayer = {
        track_id
        for track_id in role_known_ids
        if int(predictions[track_id]["nonplayer_role_signals"]) > 0
    }
    tp = len(manual_nonplayer & detected_nonplayer)
    fp = len(detected_nonplayer - manual_nonplayer)
    fn = len(manual_nonplayer - detected_nonplayer)

    scopes = {
        "all_reviewed_outfield_tracks": {
            "track_ids": outfield_ids,
            "support": {team: sum(labels[track_id]["team"] == team for track_id in outfield_ids) for team in TEAMS},
            "majority_baseline": majority_baseline(outfield_ids, labels),
            "existing_reid": evaluate(outfield_ids, labels, predictions, "saved_reid_cluster"),
            "color_all_tracks": evaluate(outfield_ids, labels, predictions, "color_cluster_all"),
        },
        "eligible_reviewed_outfield_tracks": {
            "track_ids": eligible_outfield_ids,
            "support": {
                team: sum(labels[track_id]["team"] == team for track_id in eligible_outfield_ids)
                for team in TEAMS
            },
            "majority_baseline": majority_baseline(eligible_outfield_ids, labels),
            "existing_reid": evaluate(eligible_outfield_ids, labels, predictions, "saved_reid_cluster"),
            "color_all_tracks": evaluate(eligible_outfield_ids, labels, predictions, "color_cluster_all"),
            "color_role_length_filtered": evaluate(
                eligible_outfield_ids, labels, predictions, "color_cluster_filtered"
            ),
        },
    }
    result = {
        "status": "passed",
        "scope": "single-video, track-level development diagnosis",
        "video_id": annotations["video_id"],
        "annotation_file": str(ANNOTATIONS),
        "annotation_summary": {
            "tracks": len(labels),
            "roles": {
                role: sum(row["role"] == role for row in label_rows)
                for role in sorted({row["role"] for row in label_rows})
            },
            "teams": {
                team: sum(row["team"] == team for row in label_rows)
                for team in ("blue", "claret", "unknown_or_na")
            },
            "review_notes": {
                "track_11": labels[11]["note"],
                "track_20": labels[20]["note"],
            },
        },
        "evaluation_policy": {
            "unit": "track",
            "included_roles": ["outfield_player"],
            "included_teams": list(TEAMS),
            "cluster_mapping": "choose the binary cluster-to-team permutation with the highest accuracy within each scope",
            "excluded_from_team_accuracy": "goalkeepers, referee/staff, uncertain roles, or unknown teams",
        },
        "scopes": scopes,
        "role_signal_diagnosis": {
            "known_role_tracks": len(role_known_ids),
            "manual_nonplayer_track_ids": sorted(manual_nonplayer),
            "tracks_with_any_nonplayer_role_detection_signal": sorted(detected_nonplayer),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": safe_ratio(tp, tp + fp),
            "recall": safe_ratio(tp, tp + fn),
        },
        "interpretation_boundary": [
            "This is a single-game development diagnosis, not a cross-game benchmark.",
            "Track 20 remains visually difficult and is labeled claret from the final representative crop.",
            "No annotation or diagnostic cluster was written back to SoccerFactory.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "outfield_tracks": len(outfield_ids),
                "reid_accuracy": scopes["all_reviewed_outfield_tracks"]["existing_reid"]["accuracy"],
                "color_accuracy": scopes["all_reviewed_outfield_tracks"]["color_all_tracks"]["accuracy"],
                "color_errors": scopes["all_reviewed_outfield_tracks"]["color_all_tracks"]["incorrect_track_ids"],
            }
        )
    )


if __name__ == "__main__":
    main()
