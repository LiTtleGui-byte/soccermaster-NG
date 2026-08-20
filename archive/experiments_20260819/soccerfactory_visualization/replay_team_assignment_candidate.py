#!/usr/bin/env python3
"""Replay a non-writing color-team candidate on two fixed development matches."""

from __future__ import annotations

import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

from diagnose_team_color_features import hue_feature, torso_crop


REPO = Path("/home/tianlin/SoccerMaster")
OUTPUT_DIR = REPO / "reports/g10/20260819_team_assignment_candidate_replay"
CANDIDATES = OUTPUT_DIR / "candidate_outputs.json"
EVALUATION = OUTPUT_DIR / "evaluation.json"
COLOR_MARGIN_THRESHOLD = 0.15
MIN_VALID_CROPS = 3
TOP_CROPS_PER_TRACK = 12

MATCHES = {
    "10004": {
        "archive": REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz",
        "image_dir": None,
        "expected": {"detections": 3176, "tracks": 49, "images": 255},
        "annotations": REPO / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json",
        "diagnostic": REPO / "reports/g10/20260818_team_color_diagnostic/result.json",
        "provenance": "new local prerefiner enrichment run2",
    },
    "10001": {
        "archive": Path(
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
            "step_3_sn500_1000/states/sn-gamestate.pklz"
        ),
        "image_dir": Path(
            "/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/"
            "SoccerNetGS/sn500/SNGS-10001/img1"
        ),
        "expected": {"detections": 11494, "tracks": 40, "images": 750},
        "annotations": REPO / "reports/g10/20260818_team_color_cross_match_sngs10001/sngs10001_track_annotations.json",
        "diagnostic": REPO / "reports/g10/20260818_team_color_cross_match_sngs10001/result.json",
        "provenance": "read-only historical Step-3",
    },
}


def load_tables(video_id: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(spec["archive"]) as archive:
        detections = pd.read_pickle(archive.open(f"{video_id}.pkl"))
        images = pd.read_pickle(archive.open(f"{video_id}_image.pkl"))
    expected = spec["expected"]
    if (
        len(detections) != expected["detections"]
        or detections.track_id.nunique() != expected["tracks"]
        or len(images) != expected["images"]
    ):
        raise AssertionError(f"Fixed input identity changed for {video_id}")
    images = images.copy()
    if spec["image_dir"] is not None:
        images["file_path"] = images.file_path.map(
            lambda value: str(spec["image_dir"] / Path(str(value)).name)
        )
    missing = [path for path in images.file_path if not Path(str(path)).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source JPEG for {video_id}: {missing[0]}")
    return detections, images


def extract(video_id: str, detections: pd.DataFrame, images: pd.DataFrame) -> tuple[list[dict[str, Any]], np.ndarray]:
    path_by_id = {row.id: str(row.file_path) for row in images.itertuples(index=False)}

    @lru_cache(maxsize=64)
    def load_image(image_id: Any) -> Image.Image:
        return Image.open(Path(path_by_id[image_id])).convert("RGB")

    rows = []
    features = []
    for track_id, group in detections.groupby("track_id"):
        crop_features = []
        selected_pixels = 0
        for row in group.sort_values("bbox_conf", ascending=False).head(TOP_CROPS_PER_TRACK).itertuples(index=False):
            feature, pixels = hue_feature(torso_crop(load_image(row.image_id), row.bbox_ltwh))
            if pixels > 0:
                crop_features.append(feature)
                selected_pixels += pixels
        if not crop_features:
            raise AssertionError(f"No valid color crop for {video_id} track {track_id}")
        feature = np.mean(np.vstack(crop_features), axis=0)
        feature /= max(float(np.linalg.norm(feature)), 1e-12)
        role_detection = group.role_detection.dropna().astype(str)
        nonplayer_count = int((role_detection != "player").sum())
        final_roles = group.role.dropna().astype(str)
        final_role = final_roles.iloc[0] if len(final_roles) else "unknown"
        rows.append(
            {
                "track_id": int(float(track_id)),
                "rows": int(len(group)),
                "valid_color_crops": len(crop_features),
                "selected_color_pixels": int(selected_pixels),
                "saved_final_role": final_role,
                "nonplayer_detection_count": nonplayer_count,
                "role_detection_count": int(len(role_detection)),
                "nonplayer_detection_fraction": float(nonplayer_count / max(len(role_detection), 1)),
            }
        )
        features.append(feature)
    return rows, np.vstack(features)


def assign(video_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    detections, images = load_tables(video_id, spec)
    rows, features = extract(video_id, detections, images)
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

    diagnostic = json.loads(Path(spec["diagnostic"]).read_text(encoding="utf-8"))
    saved = {int(row["track_id"]): int(row["color_cluster_all"]) for row in diagnostic["track_assignments"]}
    direct = sum(saved[row["track_id"]] == row["raw_color_cluster"] for row in rows)
    flipped = sum((1 - saved[row["track_id"]]) == row["raw_color_cluster"] for row in rows)
    if max(direct, flipped) != len(rows):
        raise AssertionError(f"Candidate color replay differs from fixed diagnostic for {video_id}")
    counts = pd.Series([row["decision"] for row in rows]).value_counts()
    return {
        "video_id": video_id,
        "provenance": spec["provenance"],
        "input_archive_read_only": str(spec["archive"]),
        "detections": len(detections),
        "tracks": len(rows),
        "images": len(images),
        "decision_counts": {str(key): int(value) for key, value in counts.items()},
        "diagnostic_replay_agreement_up_to_cluster_permutation": 1.0,
        "tracks_output": rows,
    }


def best_mapping(
    ids: list[int], labels: dict[int, dict[str, Any]], rows: dict[int, dict[str, Any]], field: str
) -> dict[str, Any]:
    covered = [track_id for track_id in ids if rows[track_id][field] in (0, 1)]
    candidates = []
    for mapping in ({0: "blue", 1: "claret"}, {0: "claret", 1: "blue"}):
        correct = [
            track_id
            for track_id in covered
            if mapping[int(rows[track_id][field])] == labels[track_id]["team"]
        ]
        candidates.append(
            {
                "cluster_to_team": {str(key): value for key, value in mapping.items()},
                "covered": len(covered),
                "total": len(ids),
                "coverage": len(covered) / len(ids),
                "correct": len(correct),
                "conditional_accuracy": len(correct) / max(len(covered), 1),
                "end_to_end_accuracy": len(correct) / len(ids),
                "incorrect_covered_track_ids": [track_id for track_id in covered if track_id not in correct],
                "abstained_track_ids": [track_id for track_id in ids if track_id not in covered],
            }
        )
    return max(candidates, key=lambda item: (item["end_to_end_accuracy"], item["conditional_accuracy"]))


def evaluate_match(candidate: dict[str, Any], annotation_path: Path) -> dict[str, Any]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    labels = {int(row["track_id"]): row for row in annotation["labels"]}
    rows = {int(row["track_id"]): row for row in candidate["tracks_output"]}
    if set(labels) != set(rows):
        raise AssertionError(f"Candidate/annotation identity mismatch for {candidate['video_id']}")
    outfield = sorted(
        track_id
        for track_id, row in labels.items()
        if row["role"] == "outfield_player" and row["team"] in {"blue", "claret"}
    )
    known_nonplayers = sorted(
        track_id
        for track_id, row in labels.items()
        if row["role"] in {"goalkeeper", "referee_or_staff"}
    )
    raw = best_mapping(outfield, labels, rows, "raw_color_cluster")
    automatic = best_mapping(outfield, labels, rows, "auto_team_cluster")
    abstained_nonplayers = [track_id for track_id in known_nonplayers if rows[track_id]["auto_team_cluster"] is None]
    return {
        "video_id": candidate["video_id"],
        "manual_team_labeled_outfield_tracks": len(outfield),
        "manual_known_nonplayer_tracks": len(known_nonplayers),
        "raw_color": raw,
        "automatic_accept_only": automatic,
        "known_nonplayer_abstention": {
            "abstained": len(abstained_nonplayers),
            "total": len(known_nonplayers),
            "recall": len(abstained_nonplayers) / max(len(known_nonplayers), 1),
            "missed_nonplayer_track_ids": [
                track_id for track_id in known_nonplayers if track_id not in abstained_nonplayers
            ],
        },
    }


def main() -> None:
    if CANDIDATES.exists() or EVALUATION.exists():
        raise FileExistsError(f"Refusing to overwrite outputs in {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = [assign(video_id, MATCHES[video_id]) for video_id in ("10004", "10001")]
    candidate_result = {
        "status": "passed",
        "schema_version": 1,
        "scope": "offline, non-writing two-match candidate replay",
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
        "matches": matches,
    }
    CANDIDATES.write_text(json.dumps(candidate_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
    EVALUATION.write_text(json.dumps(evaluation_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation_result["two_match_descriptive_summary"]))


if __name__ == "__main__":
    main()
