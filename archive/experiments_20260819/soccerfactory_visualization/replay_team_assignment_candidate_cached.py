#!/usr/bin/env python3
"""Cached fallback for the fixed two-match offline team candidate replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from replay_team_assignment_candidate import (
    CANDIDATES,
    COLOR_MARGIN_THRESHOLD,
    EVALUATION,
    MATCHES,
    MIN_VALID_CROPS,
    OUTPUT_DIR,
    best_mapping,
    evaluate_match,
    extract,
    load_tables,
)


REPO = Path("/home/tianlin/SoccerMaster")
CACHE_DIR = REPO / ".runtime/g10/team_assignment_candidate_replay"


def cache_paths(video_id: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"features_{video_id}.npz", CACHE_DIR / f"tracks_{video_id}.json"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_cache(video_id: str) -> None:
    if video_id not in MATCHES:
        raise ValueError(f"Unsupported fixed video: {video_id}")
    feature_path, metadata_path = cache_paths(video_id)
    if feature_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite cache for {video_id}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    detections, images = load_tables(video_id, MATCHES[video_id])
    rows, features = extract(video_id, detections, images)
    if len(rows) != MATCHES[video_id]["expected"]["tracks"] or features.shape != (len(rows), 26):
        raise AssertionError(f"Unexpected feature cache contract for {video_id}")

    feature_tmp = feature_path.with_name(feature_path.stem + ".tmp.npz")
    if feature_tmp.exists():
        raise FileExistsError(f"Temporary feature output already exists: {feature_tmp}")
    np.savez_compressed(feature_tmp, features=features.astype(np.float32))
    os.replace(feature_tmp, feature_path)
    atomic_json(
        metadata_path,
        {
            "status": "passed",
            "video_id": video_id,
            "source_archive_read_only": str(MATCHES[video_id]["archive"]),
            "detections": len(detections),
            "images": len(images),
            "tracks": rows,
            "feature_shape": list(features.shape),
            "manual_labels_read": False,
            "gpu_used": False,
            "source_modified": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "video_id": video_id,
                "features": str(feature_path),
                "metadata": str(metadata_path),
                "shape": list(features.shape),
            }
        )
    )


def load_cache(video_id: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    feature_path, metadata_path = cache_paths(video_id)
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Complete cache pair missing for {video_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(feature_path, allow_pickle=False) as payload:
        features = payload["features"]
    rows = metadata["tracks"]
    expected_tracks = MATCHES[video_id]["expected"]["tracks"]
    if (
        metadata["status"] != "passed"
        or metadata["video_id"] != video_id
        or metadata["manual_labels_read"] is not False
        or len(rows) != expected_tracks
        or features.shape != (expected_tracks, 26)
    ):
        raise AssertionError(f"Invalid feature cache contract for {video_id}")
    return rows, features


def assign_cached(video_id: str) -> dict[str, Any]:
    rows, features = load_cache(video_id)
    model = KMeans(n_clusters=2, random_state=0).fit(features)
    labels = model.labels_.astype(int)
    distances = model.transform(features)
    for index, row in enumerate(rows):
        label = int(labels[index])
        other = 1 - label
        assigned_distance = float(distances[index, label])
        other_distance = float(distances[index, other])
        margin = float((other_distance - assigned_distance) / max(other_distance, 1e-12))
        reasons = []
        if row["saved_final_role"] in {"goalkeeper", "referee"}:
            decision = "unknown_nonplayer"
            reasons.append(f"explicit_final_role={row['saved_final_role']}")
        elif row["valid_color_crops"] < MIN_VALID_CROPS:
            decision = "unknown_low_evidence"
            reasons.append(f"valid_color_crops<{MIN_VALID_CROPS}")
        elif (
            row["saved_final_role"] != "player"
            and row["nonplayer_detection_fraction"] >= 0.60
        ) or row["nonplayer_detection_fraction"] >= 0.80:
            decision = "review_required_role"
            reasons.append("sustained_nonplayer_role_evidence")
        elif margin < COLOR_MARGIN_THRESHOLD:
            decision = "review_required_color"
            reasons.append(f"color_margin<{COLOR_MARGIN_THRESHOLD}")
        else:
            decision = "accepted"
        row.update(
            {
                "raw_color_cluster": label,
                "color_distance_assigned": assigned_distance,
                "color_distance_other": other_distance,
                "color_margin": margin,
                "decision": decision,
                "auto_team_cluster": label if decision == "accepted" else None,
                "decision_reasons": reasons,
            }
        )

    diagnostic = json.loads(Path(MATCHES[video_id]["diagnostic"]).read_text(encoding="utf-8"))
    saved = {int(row["track_id"]): int(row["color_cluster_all"]) for row in diagnostic["track_assignments"]}
    direct = sum(saved[row["track_id"]] == row["raw_color_cluster"] for row in rows)
    flipped = sum((1 - saved[row["track_id"]]) == row["raw_color_cluster"] for row in rows)
    if max(direct, flipped) != len(rows):
        raise AssertionError(f"Cached candidate differs from fixed diagnostic for {video_id}")
    counts = pd.Series([row["decision"] for row in rows]).value_counts()
    return {
        "video_id": video_id,
        "provenance": MATCHES[video_id]["provenance"],
        "input_archive_read_only": str(MATCHES[video_id]["archive"]),
        "tracks": len(rows),
        "decision_counts": {str(key): int(value) for key, value in counts.items()},
        "diagnostic_replay_agreement_up_to_cluster_permutation": 1.0,
        "tracks_output": rows,
    }


def combine() -> None:
    if CANDIDATES.exists() or EVALUATION.exists():
        raise FileExistsError(f"Refusing to overwrite outputs in {OUTPUT_DIR}")
    matches = [assign_cached(video_id) for video_id in ("10004", "10001")]
    candidate_result = {
        "status": "passed",
        "schema_version": 1,
        "scope": "offline, non-writing two-match cached candidate replay",
        "manual_labels_used_for_candidate_generation": False,
        "gpu_used": False,
        "model_or_pipeline_rerun": False,
        "labels_written_back": False,
        "fixed_policy": {
            "color_feature": "same 26-D upper-body HSV/chromaticity feature as both prior diagnostics",
            "color_clustering": "KMeans(n_clusters=2, random_state=0) over all tracks in each match",
            "explicit_nonplayer": "saved final role goalkeeper/referee -> unknown_nonplayer",
            "low_evidence": f"valid color crops < {MIN_VALID_CROPS} -> unknown_low_evidence",
            "soft_role_review": "final role not player plus nonplayer detection fraction >=0.60, or fraction >=0.80",
            "color_review": f"normalized two-centroid distance margin < {COLOR_MARGIN_THRESHOLD}",
            "raw_and_safe_outputs": "raw_color_cluster always retained; auto_team_cluster only for accepted tracks",
        },
        "cache_contract": {
            "directory": str(CACHE_DIR),
            "per_match_atomic_pairs": True,
            "manual_labels_read_during_extraction": False,
        },
        "matches": matches,
    }
    atomic_json(CANDIDATES, candidate_result)

    evaluations = [
        evaluate_match(match, Path(MATCHES[match["video_id"]]["annotations"])) for match in matches
    ]
    total = sum(item["manual_team_labeled_outfield_tracks"] for item in evaluations)
    raw_correct = sum(item["raw_color"]["correct"] for item in evaluations)
    auto_covered = sum(item["automatic_accept_only"]["covered"] for item in evaluations)
    auto_correct = sum(item["automatic_accept_only"]["correct"] for item in evaluations)
    nonplayer_total = sum(item["known_nonplayer_abstention"]["total"] for item in evaluations)
    nonplayer_abstained = sum(item["known_nonplayer_abstention"]["abstained"] for item in evaluations)
    evaluation_result = {
        "status": "passed",
        "verdict": "raw_color_supported_auto_acceptance_not_production_safe",
        "candidate_file": str(CANDIDATES),
        "manual_labels_used_only_after_candidate_write": True,
        "matches": evaluations,
        "two_match_descriptive_summary": {
            "warning": "Two development matches with different run provenance; not a benchmark.",
            "outfield_tracks": total,
            "raw_color_coverage": 1.0,
            "raw_color_accuracy": raw_correct / total,
            "automatic_coverage": auto_covered / total,
            "automatic_conditional_accuracy": auto_correct / max(auto_covered, 1),
            "automatic_end_to_end_accuracy": auto_correct / total,
            "known_nonplayer_abstention_recall": nonplayer_abstained / max(nonplayer_total, 1),
        },
        "interpretation_boundary": [
            "Abstention trades automatic coverage for conditional precision.",
            "The candidate emits anonymous clusters; blue/claret mappings are evaluation-only.",
            "No source archive or SoccerFactory label was modified.",
        ],
    }
    atomic_json(EVALUATION, evaluation_result)
    print(json.dumps(evaluation_result["two_match_descriptive_summary"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("extract", "combine"))
    parser.add_argument("--video-id", choices=tuple(MATCHES))
    args = parser.parse_args()
    if args.stage == "extract" and args.video_id is None:
        parser.error("--video-id is required for extract")
    if args.stage == "combine" and args.video_id is not None:
        parser.error("--video-id is not allowed for combine")
    return args


def main() -> None:
    args = parse_args()
    if args.stage == "extract":
        extract_cache(args.video_id)
    else:
        combine()


if __name__ == "__main__":
    main()
