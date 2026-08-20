#!/usr/bin/env python3
"""Find likely ID-switch moments in a preserved SoccerFactory archive.

This is a CPU-only diagnostic.  It does not relabel the archive or claim that
an anomaly is a confirmed switch; it ranks moments for human review using the
saved ReID embedding, image-space motion, and field conflicts.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
REPORT = REPO / "reports/g10/20260818_track_switch_candidates_v2"
VIDEO_ID = "10004"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_archive() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    if len(detections) != 3176 or detections.track_id.nunique() != 49 or len(images) != 255:
        raise AssertionError("The fixed SNGS-10004 archive identity changed")
    return detections, images


def embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros_like(vector)


def bbox_center(value: Any) -> tuple[float, float, float]:
    x, y, width, height = (float(item) for item in value)
    return x + width / 2.0, y + height / 2.0, max(height, 1.0)


def transition_rows(detections: pd.DataFrame, images: pd.DataFrame) -> list[dict[str, Any]]:
    frame_by_image = {row.id: int(row.frame) for row in images.itertuples(index=False)}
    work = detections.copy()
    work["frame"] = work.image_id.map(frame_by_image)
    rows_by_frame = {int(frame): group for frame, group in work.groupby("frame", sort=True)}
    transitions: list[dict[str, Any]] = []
    for frame in range(int(images.frame.min()), int(images.frame.max())):
        left = rows_by_frame.get(frame)
        right = rows_by_frame.get(frame + 1)
        if left is None or right is None:
            continue
        right_by_track = {int(float(row.track_id)): row for row in right.itertuples(index=False)}
        right_vectors = {int(float(row.track_id)): embedding(row.embeddings) for row in right.itertuples(index=False)}
        for previous in left.itertuples(index=False):
            track_id = int(float(previous.track_id))
            current = right_by_track.get(track_id)
            if current is None:
                continue
            previous_vector = embedding(previous.embeddings)
            assigned_similarity = float(np.dot(previous_vector, right_vectors[track_id]))
            alternatives = [
                (other_id, float(np.dot(previous_vector, vector)))
                for other_id, vector in right_vectors.items()
                if other_id != track_id
            ]
            best_other_id, best_other_similarity = max(alternatives, key=lambda item: item[1]) if alternatives else (-1, -1.0)
            px, py, ph = bbox_center(previous.bbox_ltwh)
            cx, cy, ch = bbox_center(current.bbox_ltwh)
            normalized_jump = math.hypot(cx - px, cy - py) / max(ph, ch)
            appearance_distance = 1.0 - assigned_similarity
            cross_gain = max(0.0, best_other_similarity - assigned_similarity)
            team_conflict = int(str(previous.team) != str(current.team))
            jersey_conflict = int(
                previous.jersey_number is not None
                and current.jersey_number is not None
                and str(previous.jersey_number) != str(current.jersey_number)
            )
            score = (
                0.45 * min(max(appearance_distance, 0.0), 2.0) / 2.0
                + 0.35 * min(normalized_jump, 3.0) / 3.0
                + 0.20 * min(cross_gain, 2.0) / 2.0
                + 0.20 * team_conflict
                + 0.15 * jersey_conflict
            )
            transitions.append(
                {
                    "frame": frame,
                    "next_frame": frame + 1,
                    "track_id": track_id,
                    "best_other_track_id": best_other_id,
                    "appearance_distance": appearance_distance,
                    "assigned_similarity": assigned_similarity,
                    "best_other_similarity": best_other_similarity,
                    "cross_assignment_gain": cross_gain,
                    "normalized_bbox_jump": normalized_jump,
                    "team_conflict": bool(team_conflict),
                    "jersey_conflict": bool(jersey_conflict),
                    "score": score,
                }
            )
    return sorted(transitions, key=lambda item: item["score"], reverse=True)


def reappearance_rows(detections: pd.DataFrame, images: pd.DataFrame) -> list[dict[str, Any]]:
    """Score a track's first row after a multi-frame disappearance."""
    frame_by_image = {row.id: int(row.frame) for row in images.itertuples(index=False)}
    work = detections.copy()
    work["frame"] = work.image_id.map(frame_by_image)
    rows_by_frame = {int(frame): group for frame, group in work.groupby("frame", sort=True)}
    candidates: list[dict[str, Any]] = []
    for track_id, group in work.groupby("track_id"):
        group = group.sort_values("frame", kind="stable")
        for previous, current in zip(group.iloc[:-1].itertuples(index=False), group.iloc[1:].itertuples(index=False)):
            gap = int(current.frame - previous.frame)
            if gap <= 1:
                continue
            current_rows = rows_by_frame[int(current.frame)]
            current_vectors = {
                int(float(row.track_id)): embedding(row.embeddings)
                for row in current_rows.itertuples(index=False)
            }
            track_number = int(float(track_id))
            previous_vector = embedding(previous.embeddings)
            assigned_similarity = float(np.dot(previous_vector, current_vectors[track_number]))
            alternatives = [
                (other_id, float(np.dot(previous_vector, vector)))
                for other_id, vector in current_vectors.items()
                if other_id != track_number
            ]
            best_other_id, best_other_similarity = max(alternatives, key=lambda item: item[1]) if alternatives else (-1, -1.0)
            px, py, ph = bbox_center(previous.bbox_ltwh)
            cx, cy, ch = bbox_center(current.bbox_ltwh)
            normalized_jump = math.hypot(cx - px, cy - py) / max(ph, ch)
            appearance_distance = 1.0 - assigned_similarity
            cross_gain = max(0.0, best_other_similarity - assigned_similarity)
            score = (
                0.50 * min(max(appearance_distance, 0.0), 2.0) / 2.0
                + 0.30 * min(normalized_jump, 8.0) / 8.0
                + 0.20 * min(cross_gain, 2.0) / 2.0
            )
            candidates.append(
                {
                    "last_frame": int(previous.frame),
                    "reappear_frame": int(current.frame),
                    "gap": gap,
                    "track_id": track_number,
                    "best_other_track_id": best_other_id,
                    "appearance_distance": appearance_distance,
                    "assigned_similarity": assigned_similarity,
                    "best_other_similarity": best_other_similarity,
                    "cross_assignment_gain": cross_gain,
                    "normalized_bbox_jump": normalized_jump,
                    "score": score,
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def gap_summary(detections: pd.DataFrame, images: pd.DataFrame) -> dict[str, Any]:
    frame_by_image = {row.id: int(row.frame) for row in images.itertuples(index=False)}
    gaps: list[dict[str, Any]] = []
    for track_id, group in detections.groupby("track_id"):
        frames = sorted(int(frame_by_image[value]) for value in group.image_id)
        differences = np.diff(frames)
        missing = differences[differences > 1]
        gaps.append(
            {
                "track_id": int(float(track_id)),
                "frames": len(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "max_gap": int(missing.max()) if len(missing) else 1,
                "missing_frame_count": int(np.maximum(missing - 1, 0).sum()) if len(missing) else 0,
            }
        )
    return {
        "tracks_with_gaps": sum(item["max_gap"] > 1 for item in gaps),
        "max_gap": max(item["max_gap"] for item in gaps),
        "tracks": sorted(gaps, key=lambda item: (item["max_gap"], item["missing_frame_count"]), reverse=True),
    }


def render(transitions: list[dict[str, Any]], reappearances: list[dict[str, Any]]) -> Path:
    figure = REPORT / "track_switch_candidate_timeline.png"
    if figure.exists():
        raise FileExistsError(figure)
    REPORT.mkdir(parents=True, exist_ok=True)
    top = transitions[:30]
    top_reappearances = reappearances[:30]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    scores = np.asarray([item["score"] for item in transitions], dtype=float)
    axes[0].hist(scores, bins=30, color="#5b8ff9", alpha=0.85)
    axes[0].set(title="All consecutive-track transition scores", xlabel="heuristic suspicion score", ylabel="transitions")
    axes[1].scatter(
        [item["frame"] for item in top],
        [item["score"] for item in top],
        c=[item["track_id"] for item in top],
        cmap="tab20",
        s=55,
    )
    for item in top[:12]:
        axes[1].annotate(
            f"{item['track_id']}→{item['best_other_track_id']}",
            (item["frame"], item["score"]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set(title="Top candidate moments (review, not confirmed switches)", xlabel="frame", ylabel="score")
    if top_reappearances:
        axes[1].scatter(
            [item["reappear_frame"] for item in top_reappearances],
            [item["score"] for item in top_reappearances],
            marker="x",
            color="#d94c5c",
            s=55,
            label="after gap",
        )
        axes[1].legend()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    return figure


def main() -> None:
    detections, images = load_archive()
    transitions = transition_rows(detections, images)
    if not transitions:
        raise AssertionError("No consecutive transitions found")
    reappearances = reappearance_rows(detections, images)
    gaps = gap_summary(detections, images)
    figure = render(transitions, reappearances)
    result = {
        "status": "passed",
        "gpu_used": False,
        "manual_labels_used": False,
        "scope": "CPU-only ranking of possible ID-switch moments in one fixed archive",
        "input": str(ARCHIVE),
        "detections": len(detections),
        "tracks": int(detections.track_id.nunique()),
        "frames": len(images),
        "consecutive_transitions": len(transitions),
        "top_candidates": transitions[:20],
        "reappearance_candidates": reappearances[:20],
        "gap_summary": gaps,
        "figure": str(figure),
        "interpretation": [
            "A high score is a review candidate, not a confirmed ID switch.",
            "The archive has no person-identity ground truth, so precision and recall cannot be reported.",
            "A confirmed switch requires checking the adjacent frames and splitting the track only after visual review.",
        ],
        "next_step": "Manually inspect the top five adjacent-frame candidates before changing track IDs.",
    }
    atomic_json(REPORT / "result.json", result)
    print(json.dumps({"status": "passed", "top_candidates": len(result["top_candidates"]), "result": str(REPORT / 'result.json'), "figure": str(figure)}))


if __name__ == "__main__":
    main()
