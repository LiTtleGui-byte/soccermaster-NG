#!/usr/bin/env python3
"""Compare Refiner input/output pitch coordinates and render a compact visual."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle


REPO = Path("/home/tianlin/SoccerMaster")
BEFORE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
AFTER = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
REPORT = REPO / "reports/g10/20260819_refiner_coordinate_quality"
RESULT = REPORT / "result.json"
FIGURE = REPORT / "refiner_quality_comparison.png"
VIDEO_ID = "10004"
PITCH_X = (-52.5, 52.5)
PITCH_Y = (-34.0, 34.0)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        return (
            pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl")),
            pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl")),
        )


def coordinates(detections: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            [float(value["x_bottom_middle"]), float(value["y_bottom_middle"])]
            for value in detections.bbox_pitch
        ],
        dtype=np.float64,
    )


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "min": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    q = np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 1])
    return {
        key: float(value)
        for key, value in zip(("min", "median", "p90", "p95", "p99", "max"), q)
    } | {"count": int(len(values))}


def out_of_bounds(points: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] < PITCH_X[0])
        | (points[:, 0] > PITCH_X[1])
        | (points[:, 1] < PITCH_Y[0])
        | (points[:, 1] > PITCH_Y[1])
    )


def temporal_metrics(
    detections: pd.DataFrame, points: np.ndarray, frame_by_image: dict[Any, int]
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    work = detections[["image_id", "track_id"]].copy()
    work["frame"] = work.image_id.map(frame_by_image)
    work["x"] = points[:, 0]
    work["y"] = points[:, 1]
    steps = []
    accelerations = []
    track_acceleration_medians = {}
    for track_id, group in work.groupby("track_id", sort=True):
        group = group.sort_values("frame", kind="stable")
        frames = group.frame.to_numpy(dtype=np.int64)
        xy = group[["x", "y"]].to_numpy(dtype=np.float64)
        delta_frames = np.diff(frames)
        delta_xy = np.diff(xy, axis=0)
        consecutive = delta_frames == 1
        steps.extend(np.linalg.norm(delta_xy[consecutive], axis=1).tolist())
        track_acc = []
        for index in range(1, len(frames) - 1):
            if frames[index] - frames[index - 1] == 1 and frames[index + 1] - frames[index] == 1:
                value = float(np.linalg.norm(xy[index + 1] - 2 * xy[index] + xy[index - 1]))
                accelerations.append(value)
                track_acc.append(value)
        if track_acc:
            track_acceleration_medians[int(float(track_id))] = float(np.median(track_acc))
    return np.asarray(steps), np.asarray(accelerations), track_acceleration_medians


def draw_pitch(axis: Any) -> None:
    axis.add_patch(Rectangle((-52.5, -34), 105, 68, fill=False, edgecolor="#bac7d3", linewidth=1.2))
    axis.plot([0, 0], [-34, 34], color="#bac7d3", linewidth=0.8)
    axis.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor="#bac7d3", linewidth=0.8))
    for side in (-1, 1):
        x = side * 52.5
        left = -52.5 if side < 0 else 36.0
        axis.add_patch(Rectangle((left, -20.16), 16.5, 40.32, fill=False, edgecolor="#bac7d3", linewidth=0.8))
        axis.add_patch(Arc((side * 41.5, 0), 18.3, 18.3, theta1=90 if side < 0 else -90, theta2=270 if side < 0 else 90, edgecolor="#bac7d3", linewidth=0.8))
    axis.set_xlim(-54, 54)
    axis.set_ylim(-36, 36)
    axis.set_aspect("equal")
    axis.set_xlabel("pitch x (m)")
    axis.set_ylabel("pitch y (m)")
    axis.grid(alpha=0.12)


def plot_trajectories(
    axis: Any,
    detections: pd.DataFrame,
    points: np.ndarray,
    frame_by_image: dict[Any, int],
    track_ids: list[int],
    title: str,
) -> None:
    draw_pitch(axis)
    colors = plt.cm.tab10(np.linspace(0, 1, len(track_ids)))
    for color, track_id in zip(colors, track_ids):
        mask = detections.track_id.astype(float) == float(track_id)
        subset = detections.loc[mask, ["image_id"]].copy()
        subset["frame"] = subset.image_id.map(frame_by_image)
        xy = points[mask.to_numpy()]
        order = np.argsort(subset.frame.to_numpy())
        axis.plot(xy[order, 0], xy[order, 1], color=color, linewidth=1.3, alpha=0.85, label=str(track_id))
    axis.set_title(title)
    axis.legend(title="track", ncol=2, fontsize=7, title_fontsize=8, loc="upper right")


def main() -> None:
    if RESULT.exists() or FIGURE.exists():
        raise FileExistsError("Refusing to overwrite Refiner quality outputs")
    before, before_images = load(BEFORE)
    after, after_images = load(AFTER)
    identity_columns = ["image_id", "track_id"]
    if len(before) != len(after) or not before[identity_columns].equals(after[identity_columns]):
        raise AssertionError("Detection identity/order changed across Refiner")
    if not before_images[["id", "frame"]].equals(after_images[["id", "frame"]]):
        raise AssertionError("Image identity/frame mapping changed across Refiner")
    before_xy, after_xy = coordinates(before), coordinates(after)
    before_finite = np.isfinite(before_xy).all(axis=1)
    after_finite = np.isfinite(after_xy).all(axis=1)
    displacement = np.linalg.norm(after_xy - before_xy, axis=1)
    frame_by_image = {row.id: int(row.frame) for row in before_images.itertuples(index=False)}
    before_steps, before_acc, before_track_acc = temporal_metrics(before, before_xy, frame_by_image)
    after_steps, after_acc, after_track_acc = temporal_metrics(after, after_xy, frame_by_image)
    common_tracks = sorted(set(before_track_acc) & set(after_track_acc))
    improved_tracks = [track for track in common_tracks if after_track_acc[track] < before_track_acc[track]]
    worsened_tracks = [track for track in common_tracks if after_track_acc[track] > before_track_acc[track]]
    unchanged_tracks = [track for track in common_tracks if after_track_acc[track] == before_track_acc[track]]
    before_span = np.ptp(before_xy, axis=0)
    after_span = np.ptp(after_xy, axis=0)
    span_ratio = after_span / before_span
    structural_safe = bool(
        after_finite.all()
        and out_of_bounds(after_xy).sum() <= out_of_bounds(before_xy).sum()
        and np.all(span_ratio >= 0.5)
    )
    smoother = bool(
        len(before_acc)
        and len(after_acc)
        and np.median(after_acc) < np.median(before_acc)
        and np.quantile(after_acc, 0.95) < np.quantile(before_acc, 0.95)
    )
    if structural_safe and smoother:
        verdict = "refiner_structurally_safe_and_temporally_smoother"
    elif structural_safe:
        verdict = "refiner_structurally_safe_but_not_temporally_smoother"
    else:
        verdict = "refiner_output_has_structural_or_spatial_risk"

    lengths = before.groupby("track_id").size().sort_values(ascending=False)
    longest_tracks = [int(float(value)) for value in lengths.head(8).index]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plot_trajectories(axes[0, 0], before, before_xy, frame_by_image, longest_tracks, "Before Refiner: 8 longest tracks")
    plot_trajectories(axes[0, 1], after, after_xy, frame_by_image, longest_tracks, "After Refiner: same tracks")
    axes[1, 0].hist(displacement, bins=40, color="#4da6ff", alpha=0.85)
    axes[1, 0].axvline(np.median(displacement), color="#ef8f3c", linestyle="--", label=f"median {np.median(displacement):.3f} m")
    axes[1, 0].axvline(np.quantile(displacement, 0.95), color="#d94c5c", linestyle="--", label=f"p95 {np.quantile(displacement, 0.95):.3f} m")
    axes[1, 0].set_title("Per-detection coordinate change")
    axes[1, 0].set_xlabel("Euclidean displacement (m)")
    axes[1, 0].set_ylabel("detections")
    axes[1, 0].legend()
    labels = ["median", "p95"]
    before_values = [np.median(before_acc), np.quantile(before_acc, 0.95)]
    after_values = [np.median(after_acc), np.quantile(after_acc, 0.95)]
    x = np.arange(2)
    width = 0.36
    axes[1, 1].bar(x - width / 2, before_values, width, label="before", color="#7b8a99")
    axes[1, 1].bar(x + width / 2, after_values, width, label="after", color="#50b77d")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("consecutive-frame second difference (m)")
    axes[1, 1].set_title("Temporal jitter proxy (lower is smoother)")
    axes[1, 1].legend()
    fig.suptitle("SNGS-10004 Refiner coordinate quality audit", fontsize=16)
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)

    result = {
        "status": "passed",
        "verdict": verdict,
        "gpu_used": False,
        "manual_labels_used": False,
        "before_archive_read_only": str(BEFORE),
        "after_archive_read_only": str(AFTER),
        "rows": len(before),
        "tracks": int(before.track_id.nunique()),
        "frames": len(before_images),
        "identity_and_order_preserved": True,
        "finite_coordinates": {"before": int(before_finite.sum()), "after": int(after_finite.sum())},
        "out_of_pitch_rows": {"before": int(out_of_bounds(before_xy).sum()), "after": int(out_of_bounds(after_xy).sum())},
        "coordinate_range": {
            "before_x": [float(before_xy[:, 0].min()), float(before_xy[:, 0].max())],
            "before_y": [float(before_xy[:, 1].min()), float(before_xy[:, 1].max())],
            "after_x": [float(after_xy[:, 0].min()), float(after_xy[:, 0].max())],
            "after_y": [float(after_xy[:, 1].min()), float(after_xy[:, 1].max())],
            "span_ratio_after_over_before": [float(value) for value in span_ratio],
        },
        "changed_rows_above_1e_6_m": int((displacement > 1e-6).sum()),
        "coordinate_displacement_m": stats(displacement),
        "consecutive_frame_step_m": {"before": stats(before_steps), "after": stats(after_steps)},
        "consecutive_frame_second_difference_m": {"before": stats(before_acc), "after": stats(after_acc)},
        "track_median_jitter": {
            "comparable_tracks": len(common_tracks),
            "improved_tracks": len(improved_tracks),
            "worsened_tracks": len(worsened_tracks),
            "unchanged_tracks": len(unchanged_tracks),
            "improved_track_ids": improved_tracks,
            "worsened_track_ids": worsened_tracks,
        },
        "structural_safe": structural_safe,
        "temporally_smoother": smoother,
        "figure": str(FIGURE),
        "interpretation_boundary": [
            "Lower second difference is a smoothness proxy, not ground-truth coordinate accuracy.",
            "This is one 255-frame development sequence with no coordinate ground truth.",
            "A structurally safe result supports testing Step 3 but does not validate Refiner on other matches.",
        ],
        "next_step": (
            "Proceed to a fixed SNGS-10004 Step 3 smoke if structural safety and smoothing hold; "
            "otherwise inspect the largest-displacement/worsened tracks first."
        ),
    }
    atomic_json(RESULT, result)
    print(json.dumps({"status": "passed", "verdict": verdict, "result": str(RESULT), "figure": str(FIGURE)}))


if __name__ == "__main__":
    main()
