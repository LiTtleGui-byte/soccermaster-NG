#!/usr/bin/env python3
"""Build and evaluate a fixed rule-based track role candidate on two matches."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
TEAM_CANDIDATES = REPO / "reports/g10/20260819_team_assignment_candidate_replay/candidate_outputs.json"
OUTPUT_DIR = REPO / "reports/g10/20260819_track_role_candidate"
CANDIDATES = OUTPUT_DIR / "candidate_outputs.json"
EVALUATION = OUTPUT_DIR / "evaluation.json"
PITCH_HALF_LENGTH = 52.5

MATCHES = {
    "10004": {
        "archive": REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz",
        "annotations": REPO / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json",
    },
    "10001": {
        "archive": Path(
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
            "step_3_sn500_1000/states/sn-gamestate.pklz"
        ),
        "annotations": REPO / "reports/g10/20260818_team_color_cross_match_sngs10001/sngs10001_track_annotations.json",
    },
}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output exists: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def pitch_features(group: pd.DataFrame) -> dict[str, float | int | None]:
    points = []
    for value in group.bbox_pitch.dropna():
        if not isinstance(value, dict):
            continue
        x = value.get("x_bottom_middle")
        y = value.get("y_bottom_middle")
        if x is None or y is None or not np.isfinite(float(x)) or not np.isfinite(float(y)):
            continue
        points.append((float(x), float(y)))
    if not points:
        return {
            "valid_pitch_rows": 0,
            "median_pitch_x": None,
            "median_pitch_y": None,
            "goal_line_distance": None,
            "pitch_x_iqr": None,
            "pitch_y_iqr": None,
        }
    array = np.asarray(points, dtype=np.float64)
    median_x, median_y = np.median(array, axis=0)
    return {
        "valid_pitch_rows": len(points),
        "median_pitch_x": float(median_x),
        "median_pitch_y": float(median_y),
        "goal_line_distance": float(max(0.0, PITCH_HALF_LENGTH - abs(median_x))),
        "pitch_x_iqr": float(np.percentile(array[:, 0], 75) - np.percentile(array[:, 0], 25)),
        "pitch_y_iqr": float(np.percentile(array[:, 1], 75) - np.percentile(array[:, 1], 25)),
    }


def build_candidate() -> dict[str, Any]:
    team_result = json.loads(TEAM_CANDIDATES.read_text(encoding="utf-8"))
    team_matches = {match["video_id"]: match for match in team_result["matches"]}
    output_matches = []
    for video_id in ("10004", "10001"):
        team_rows = {int(row["track_id"]): row for row in team_matches[video_id]["tracks_output"]}
        with zipfile.ZipFile(MATCHES[video_id]["archive"]) as archive:
            detections = pd.read_pickle(archive.open(f"{video_id}.pkl"))
        if set(team_rows) != {int(float(value)) for value in detections.track_id.dropna().unique()}:
            raise AssertionError(f"Track identity mismatch for {video_id}")
        rows = []
        for track_id, group in detections.groupby("track_id"):
            numeric_id = int(float(track_id))
            role_values = group.role_detection.dropna().astype(str)
            counts = role_values.value_counts()
            total = int(len(role_values))
            record = {
                "track_id": numeric_id,
                "rows": int(len(group)),
                "saved_final_role": team_rows[numeric_id]["saved_final_role"],
                "valid_color_crops": int(team_rows[numeric_id]["valid_color_crops"]),
                "raw_color_cluster": int(team_rows[numeric_id]["raw_color_cluster"]),
                "color_distance_assigned": float(team_rows[numeric_id]["color_distance_assigned"]),
                "color_margin": float(team_rows[numeric_id]["color_margin"]),
                "role_fractions": {
                    role: fraction(int(counts.get(role, 0)), total)
                    for role in ("player", "goalkeeper", "referee", "other")
                },
                **pitch_features(group),
            }
            rows.append(record)

        for cluster in (0, 1):
            cluster_rows = [row for row in rows if row["raw_color_cluster"] == cluster]
            ordered = sorted(cluster_rows, key=lambda row: row["color_distance_assigned"])
            denominator = max(len(ordered) - 1, 1)
            for rank, row in enumerate(ordered):
                row["color_outlier_percentile"] = float(rank / denominator)

        for row in rows:
            fractions = row["role_fractions"]
            specific = fractions["goalkeeper"] + fractions["referee"]
            reasons = []
            if row["saved_final_role"] in {"goalkeeper", "referee"}:
                decision = "nonplayer"
                reasons.append(f"explicit_final_role={row['saved_final_role']}")
            elif specific >= 0.40:
                decision = "nonplayer"
                reasons.append("specific_goalkeeper_referee_fraction>=0.40")
            elif row["valid_color_crops"] < 3:
                decision = "unknown_low_evidence"
                reasons.append("valid_color_crops<3")
            elif specific >= 0.15:
                decision = "review_required_role"
                reasons.append("specific_goalkeeper_referee_fraction>=0.15")
            elif fractions["other"] >= 0.60 and fractions["player"] < 0.35:
                decision = "review_required_role"
                reasons.append("sustained_other_with_low_player_fraction")
            elif row["color_outlier_percentile"] >= 0.90:
                decision = "review_required_color"
                reasons.append("within_cluster_color_outlier_percentile>=0.90")
            elif (
                row["color_outlier_percentile"] >= 0.75
                and row["goal_line_distance"] is not None
                and row["goal_line_distance"] <= 12.0
            ):
                decision = "review_required_goal_color"
                reasons.append("color_outlier_percentile>=0.75_and_goal_line_distance<=12m")
            else:
                decision = "player"
            row["specific_nonplayer_fraction"] = float(specific)
            row["decision"] = decision
            row["decision_reasons"] = reasons
            row["safe_for_automatic_team_assignment"] = decision == "player"
        rows.sort(key=lambda row: row["track_id"])
        decision_counts = pd.Series([row["decision"] for row in rows]).value_counts()
        output_matches.append(
            {
                "video_id": video_id,
                "input_archive_read_only": str(MATCHES[video_id]["archive"]),
                "tracks": len(rows),
                "decision_counts": {str(key): int(value) for key, value in decision_counts.items()},
                "tracks_output": rows,
            }
        )
    return {
        "status": "passed",
        "schema_version": 1,
        "scope": "fixed rule-based two-match track role candidate",
        "manual_labels_used_for_candidate_generation": False,
        "gpu_used": False,
        "model_training_used": False,
        "labels_written_back": False,
        "fixed_policy": {
            "strict_nonplayer": "explicit final goalkeeper/referee or goalkeeper+referee detection fraction >=0.40",
            "low_evidence": "valid color crops <3",
            "role_review": "specific fraction >=0.15, or other fraction >=0.60 with player fraction <0.35",
            "color_review": "within-team color distance percentile >=0.90",
            "goal_color_review": "color percentile >=0.75 and median goal-line distance <=12m",
        },
        "matches": output_matches,
    }


def metrics(truth_positive: set[int], predicted_positive: set[int], universe: set[int]) -> dict[str, Any]:
    tp = truth_positive & predicted_positive
    fp = predicted_positive - truth_positive
    fn = truth_positive - predicted_positive
    tn = universe - truth_positive - predicted_positive
    return {
        "true_positive": len(tp),
        "false_positive": len(fp),
        "false_negative": len(fn),
        "true_negative": len(tn),
        "precision": fraction(len(tp), len(tp) + len(fp)),
        "recall": fraction(len(tp), len(tp) + len(fn)),
        "accuracy": fraction(len(tp) + len(tn), len(universe)),
        "true_positive_track_ids": sorted(tp),
        "false_positive_track_ids": sorted(fp),
        "false_negative_track_ids": sorted(fn),
    }


def evaluate(candidate: dict[str, Any]) -> dict[str, Any]:
    evaluations = []
    combined_universe: set[int] = set()
    combined_truth: set[int] = set()
    combined_strict: set[int] = set()
    combined_flagged: set[int] = set()
    for match in candidate["matches"]:
        video_id = match["video_id"]
        annotations = json.loads(Path(MATCHES[video_id]["annotations"]).read_text(encoding="utf-8"))
        labels = {int(row["track_id"]): row for row in annotations["labels"]}
        rows = {int(row["track_id"]): row for row in match["tracks_output"]}
        if set(labels) != set(rows):
            raise AssertionError(f"Candidate/label identity mismatch for {video_id}")
        universe = {
            track_id
            for track_id, row in labels.items()
            if row["role"] in {"outfield_player", "goalkeeper", "referee_or_staff"}
        }
        truth = {
            track_id
            for track_id in universe
            if labels[track_id]["role"] in {"goalkeeper", "referee_or_staff"}
        }
        strict = {track_id for track_id in universe if rows[track_id]["decision"] == "nonplayer"}
        flagged = {track_id for track_id in universe if rows[track_id]["decision"] != "player"}
        evaluations.append(
            {
                "video_id": video_id,
                "known_role_tracks": len(universe),
                "manual_nonplayer_tracks": len(truth),
                "strict_nonplayer": metrics(truth, strict, universe),
                "all_nonplayer_or_review_flags": metrics(truth, flagged, universe),
                "automatic_player_coverage": fraction(len(universe - flagged), len(universe)),
                "outfield_false_rejection_rate": fraction(len(flagged - truth), len(universe - truth)),
            }
        )
        prefix = int(video_id) * 1000
        combined_universe |= {prefix + track_id for track_id in universe}
        combined_truth |= {prefix + track_id for track_id in truth}
        combined_strict |= {prefix + track_id for track_id in strict}
        combined_flagged |= {prefix + track_id for track_id in flagged}
    strict_combined = metrics(combined_truth, combined_strict, combined_universe)
    flagged_combined = metrics(combined_truth, combined_flagged, combined_universe)
    outfield_count = len(combined_universe - combined_truth)
    false_flags = len(combined_flagged - combined_truth)
    verdict = (
        "role_candidate_not_sufficient_for_production_gate"
        if flagged_combined["recall"] < 0.90 or fraction(false_flags, outfield_count) > 0.15
        else "role_candidate_promising_requires_new_match_validation"
    )
    return {
        "status": "passed",
        "verdict": verdict,
        "candidate_file": str(CANDIDATES),
        "manual_labels_used_only_after_candidate_write": True,
        "matches": evaluations,
        "two_match_descriptive_summary": {
            "warning": "Two development matches with different run provenance; not a benchmark.",
            "known_role_tracks": len(combined_universe),
            "manual_nonplayer_tracks": len(combined_truth),
            "manual_outfield_tracks": outfield_count,
            "strict_nonplayer": strict_combined,
            "all_nonplayer_or_review_flags": flagged_combined,
            "outfield_false_rejection_rate": fraction(false_flags, outfield_count),
            "automatic_player_coverage": fraction(len(combined_universe - combined_flagged), len(combined_universe)),
        },
        "interpretation_boundary": [
            "Rules and thresholds were fixed before reading manual labels in this run.",
            "Review-required is an abstention, not a claimed nonplayer prediction.",
            "Two matches are development diagnostics and cannot establish production safety.",
            "No source archive or label was modified.",
        ],
    }


def main() -> None:
    if CANDIDATES.exists() or EVALUATION.exists():
        raise FileExistsError(f"Refusing to overwrite outputs in {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = build_candidate()
    atomic_json(CANDIDATES, candidate)
    evaluation = evaluate(candidate)
    atomic_json(EVALUATION, evaluation)
    print(json.dumps({"verdict": evaluation["verdict"], **evaluation["two_match_descriptive_summary"]}))


if __name__ == "__main__":
    main()
