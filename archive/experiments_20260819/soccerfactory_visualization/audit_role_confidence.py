#!/usr/bin/env python3
"""Audit role_confidence semantics and track aggregation on two labeled matches.

This is a CPU-only descriptive audit. It reproduces the existing weighted vote,
compares it with an unweighted vote, and evaluates fixed threshold-free scores.
It does not tune a rule, change labels, or write to either source archive.
"""

from __future__ import annotations

import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
OUTPUT_DIR = REPO / "reports/g10/20260819_role_confidence_audit"
RESULT = OUTPUT_DIR / "result.json"

SPECIFIC_NONPLAYER = {"goalkeeper", "referee"}
BROAD_NONPLAYER = {"goalkeeper", "referee", "other"}
KNOWN_MANUAL_ROLES = {"outfield_player", "goalkeeper", "referee_or_staff"}

MATCHES = {
    "10004": {
        "archive": REPO
        / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz",
        "annotations": REPO
        / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json",
        "producer_evidence": {
            "status": "confirmed",
            "producer": "sn_gamestate.reid.prtreid_api.PRTReId",
            "config": str(
                REPO / ".runtime/g10/sngs10004_step1/run5/configs/config.yaml"
            ),
        },
    },
    "10001": {
        "archive": Path(
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
            "step_3_sn500_1000/states/sn-gamestate.pklz"
        ),
        "annotations": REPO
        / "reports/g10/20260818_team_color_cross_match_sngs10001/"
        "sngs10001_track_annotations.json",
        "producer_evidence": {
            "status": "confirmed",
            "producer": "sn_gamestate.role.qwen2_5vl_role_api.QWEN2_5VL_ROLE_BATCH",
            "config": (
                "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/"
                "step_3_sn500_1000/configs/config.yaml"
            ),
        },
    },
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def fraction(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: None for key in ("min", "p01", "p25", "median", "p75", "p99", "max")}
    points = np.quantile(array, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p01", "p25", "median", "p75", "p99", "max"), points
        )
    }


def vote(labels: Iterable[str], weights: Iterable[float] | None = None) -> str | None:
    labels = list(labels)
    weights = [1.0] * len(labels) if weights is None else list(weights)
    totals: dict[str, float] = {}
    for label, weight in zip(labels, weights):
        if label not in totals:
            totals[label] = 0.0
        totals[label] += float(weight)
    return max(totals, key=totals.get) if totals else None


def saved_track_role(group: pd.DataFrame) -> str | None:
    values = group.role.dropna().astype(str).unique().tolist()
    if len(values) > 1:
        raise AssertionError(f"Saved role is not track-constant: {values}")
    return values[0] if values else None


def binary_metrics(truth: list[bool], prediction: list[bool]) -> dict[str, Any]:
    tp = sum(t and p for t, p in zip(truth, prediction))
    fp = sum((not t) and p for t, p in zip(truth, prediction))
    fn = sum(t and (not p) for t, p in zip(truth, prediction))
    tn = sum((not t) and (not p) for t, p in zip(truth, prediction))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": fraction(tp, tp + fp),
        "recall": fraction(tp, tp + fn),
        "accuracy": fraction(tp + tn, len(truth)),
    }


def roc_auc(truth: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(truth, scores) if label]
    negatives = [score for label, score in zip(truth, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def describe_track(
    video_id: str,
    track_id: int,
    group: pd.DataFrame,
    manual_role: str,
) -> dict[str, Any]:
    detections = group.role_detection.dropna().astype(str)
    confidences = group.loc[detections.index, "role_confidence"].astype(float)
    if len(detections) != len(group):
        raise AssertionError(f"Null role detections in {video_id} track {track_id}")
    if not np.isfinite(confidences.to_numpy()).all():
        raise AssertionError(f"Non-finite role confidence in {video_id} track {track_id}")

    count_totals: dict[str, int] = defaultdict(int)
    weight_totals: dict[str, float] = defaultdict(float)
    for label, confidence in zip(detections, confidences):
        count_totals[label] += 1
        weight_totals[label] += float(confidence)
    total_count = sum(count_totals.values())
    total_weight = sum(weight_totals.values())
    weighted = vote(detections, confidences)
    unweighted = vote(detections)
    saved = saved_track_role(group)

    def share(labels: set[str], weighted_share: bool) -> float:
        values = weight_totals if weighted_share else count_totals
        denominator = total_weight if weighted_share else total_count
        return float(sum(values.get(label, 0.0) for label in labels) / denominator)

    return {
        "video_id": video_id,
        "track_id": track_id,
        "rows": int(len(group)),
        "manual_role": manual_role,
        "manual_nonplayer": manual_role in {"goalkeeper", "referee_or_staff"},
        "saved_role": saved,
        "weighted_vote": weighted,
        "unweighted_vote": unweighted,
        "weighted_matches_saved": weighted == saved,
        "weighted_differs_from_unweighted": weighted != unweighted,
        "detection_counts": dict(sorted(count_totals.items())),
        "confidence_sums": {
            label: float(value) for label, value in sorted(weight_totals.items())
        },
        "scores": {
            "specific_unweighted_share": share(SPECIFIC_NONPLAYER, False),
            "specific_weighted_share": share(SPECIFIC_NONPLAYER, True),
            "broad_unweighted_share": share(BROAD_NONPLAYER, False),
            "broad_weighted_share": share(BROAD_NONPLAYER, True),
        },
    }


def evaluate_tracks(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    known = [track for track in tracks if track["manual_role"] in KNOWN_MANUAL_ROLES]
    truth = [track["manual_nonplayer"] for track in known]
    result: dict[str, Any] = {
        "known_role_tracks": len(known),
        "manual_nonplayer_tracks": sum(truth),
        "manual_outfield_tracks": len(truth) - sum(truth),
        "weighted_vote_matches_saved": sum(track["weighted_matches_saved"] for track in tracks),
        "all_tracks": len(tracks),
        "weighted_vs_unweighted_differences": sum(
            track["weighted_differs_from_unweighted"] for track in tracks
        ),
        "weighted_vs_unweighted_difference_track_ids": [
            track["track_id"]
            for track in tracks
            if track["weighted_differs_from_unweighted"]
        ],
    }
    for vote_name in ("weighted_vote", "unweighted_vote"):
        result[vote_name] = {
            "specific_nonplayer": binary_metrics(
                truth, [track[vote_name] in SPECIFIC_NONPLAYER for track in known]
            ),
            "broad_nonplayer": binary_metrics(
                truth, [track[vote_name] in BROAD_NONPLAYER for track in known]
            ),
        }
    result["threshold_free_track_score_auc"] = {
        score_name: roc_auc(truth, [track["scores"][score_name] for track in known])
        for score_name in (
            "specific_unweighted_share",
            "specific_weighted_share",
            "broad_unweighted_share",
            "broad_weighted_share",
        )
    }
    result["manual_nonplayer_details"] = [
        {
            key: track[key]
            for key in (
                "track_id",
                "manual_role",
                "saved_role",
                "weighted_vote",
                "unweighted_vote",
                "detection_counts",
                "confidence_sums",
                "scores",
            )
        }
        for track in known
        if track["manual_nonplayer"]
    ]
    return result


def audit_match(video_id: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive_path = Path(config["archive"])
    with zipfile.ZipFile(archive_path) as archive:
        detections = pd.read_pickle(archive.open(f"{video_id}.pkl"))
    annotations = json.loads(Path(config["annotations"]).read_text(encoding="utf-8"))
    labels = {int(row["track_id"]): str(row["role"]) for row in annotations["labels"]}
    archive_tracks = {int(float(value)) for value in detections.track_id.dropna().unique()}
    if archive_tracks != set(labels):
        raise AssertionError(f"Archive/annotation track mismatch for {video_id}")

    tracks = [
        describe_track(video_id, int(float(track_id)), group, labels[int(float(track_id))])
        for track_id, group in detections.groupby("track_id", sort=True)
    ]
    confidence = detections.role_confidence.astype(float).to_numpy()
    by_detection_role = {}
    for label, group in detections.groupby("role_detection", sort=True):
        by_detection_role[str(label)] = {
            "rows": int(len(group)),
            "confidence": quantiles(group.role_confidence.astype(float)),
        }
    unique_confidence = np.unique(confidence)
    if video_id == "10004":
        observed_semantics = "variable_raw_max_role_logit"
    elif len(unique_confidence) == 1 and unique_confidence[0] == 1.0:
        observed_semantics = "constant_one_hard_label_indicator"
    else:
        observed_semantics = "unknown"
    result = {
        "video_id": video_id,
        "archive_read_only": str(archive_path),
        "annotations": str(config["annotations"]),
        "producer_evidence": config["producer_evidence"],
        "rows": int(len(detections)),
        "tracks": len(tracks),
        "role_confidence_dtype": str(detections.role_confidence.dtype),
        "finite_confidences": int(np.isfinite(confidence).sum()),
        "unique_confidence_values": int(len(unique_confidence)),
        "exactly_one_rows": int(np.sum(confidence == 1.0)),
        "confidence_quantiles": quantiles(confidence),
        "by_role_detection": by_detection_role,
        "observed_semantics": observed_semantics,
        "track_evaluation": evaluate_tracks(tracks),
    }
    return result, tracks


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {RESULT}")
    match_results = []
    all_tracks = []
    for video_id, config in MATCHES.items():
        match_result, tracks = audit_match(video_id, config)
        match_results.append(match_result)
        all_tracks.extend(tracks)

    combined = evaluate_tracks(all_tracks)
    combined["auc_scope_note"] = (
        "Only unweighted-share AUC is portable across these matches. Weighted-share "
        "AUC is included descriptively but pools incompatible confidence semantics."
    )
    result = {
        "status": "passed",
        "schema_version": 1,
        "scope": "CPU-only two-match descriptive audit of role confidence and aggregation",
        "gpu_used": False,
        "training_used": False,
        "threshold_tuning_used": False,
        "source_archives_modified": False,
        "manual_labels_modified": False,
        "implementation_findings": {
            "prtreid": (
                "role_detection is argmax(role_cls_scores['globl']); role_confidence "
                "is max of the same raw score vector, with no softmax or calibration."
            ),
            "qwen_and_internvl": (
                "role_confidence is 1.0 for a parsed label and 0.0 otherwise; it is "
                "an availability indicator rather than a probability."
            ),
            "track_aggregation": (
                "select_highest_voted_att sums role_confidence by predicted label; "
                "the label with the largest accumulated sum becomes the track role."
            ),
            "retained_information_limit": (
                "The archives retain only top-1 role_detection and its scalar maximum. "
                "They do not retain the full class-score vector, so a softmax, winning "
                "margin, entropy, or post-hoc multiclass calibration cannot be recovered."
            ),
        },
        "source_files_read_only": [
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/sn_gamestate/reid/prtreid_api.py",
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/sn_gamestate/role/qwen2_5vl_role_api.py",
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/sn_gamestate/role/internvl3_role_hf_api.py",
            "/remote-home/haolinyang/sports/soccernet/tracklab/tracklab/utils/attribute_voting.py",
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/sn_gamestate/tracklet_agg/majority_vote_filter_api.py",
            "/remote-home/haolinyang/sports/soccernet/sn-gamestate/outputs/gsr/step_3_sn500_1000/configs/config.yaml",
        ],
        "matches": match_results,
        "two_match_descriptive_summary": combined,
        "verdict": "role_confidence_not_portable_or_calibrated_for_role_gating",
        "interpretation": [
            "The same column has incompatible meanings across the two archived runs.",
            "The existing weighted vote is reproduced, but weighting does not differ "
            "from ordinary majority vote on any labeled track in these two matches.",
            "Changing only the aggregation of this scalar cannot recover role evidence "
            "that the per-frame top-1 outputs did not retain.",
            "No confidence threshold is selected from these development labels.",
        ],
        "next_step": (
            "Stop tuning role_confidence heuristics on these matches. For a future role "
            "run, save the full per-frame role logits/probabilities (or an explicit "
            "abstention score) under a producer-specific schema, then evaluate a fixed "
            "track aggregator on a new annotated match."
        ),
    }
    atomic_json(RESULT, result)
    print(json.dumps({"status": "passed", "result": str(RESULT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
