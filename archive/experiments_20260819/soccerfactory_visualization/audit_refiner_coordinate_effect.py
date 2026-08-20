#!/usr/bin/env python3
"""Compare pre/post Refiner pitch coordinates without claiming ground-truth accuracy."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BEFORE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
DEFAULT_AFTER = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
DEFAULT_REPORT = REPO / "reports/g10/20260819_refiner_coordinate_effect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_archive(path: Path):
    with zipfile.ZipFile(path) as archive:
        detections = pickle.loads(archive.read("10004.pkl"))
        images = pickle.loads(archive.read("10004_image.pkl"))
    return detections, images


def pitch_xy(value: object) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise AssertionError("bbox_pitch must be a dictionary")
    xy = (float(value["x_bottom_middle"]), float(value["y_bottom_middle"]))
    if not all(math.isfinite(v) for v in xy):
        raise AssertionError("bbox_pitch bottom-middle must be finite")
    return xy


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def temporal_metrics(frames: np.ndarray, tracks: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, float | int]]]:
    steps: list[float] = []
    accelerations: list[float] = []
    per_track: dict[int, dict[str, float | int]] = {}
    for track_value in sorted(set(int(v) for v in tracks)):
        mask = tracks.astype(int) == track_value
        order = np.argsort(frames[mask])
        track_frames = frames[mask][order]
        track_xy = xy[mask][order]
        track_steps: list[float] = []
        track_accels: list[float] = []
        for idx in range(1, len(track_frames)):
            if track_frames[idx] - track_frames[idx - 1] == 1:
                distance = float(np.linalg.norm(track_xy[idx] - track_xy[idx - 1]))
                steps.append(distance)
                track_steps.append(distance)
        for idx in range(2, len(track_frames)):
            if track_frames[idx] - track_frames[idx - 1] == 1 and track_frames[idx - 1] - track_frames[idx - 2] == 1:
                acceleration = float(np.linalg.norm(track_xy[idx] - 2 * track_xy[idx - 1] + track_xy[idx - 2]))
                accelerations.append(acceleration)
                track_accels.append(acceleration)
        per_track[track_value] = {
            "rows": int(len(track_frames)),
            "consecutive_steps": int(len(track_steps)),
            "step_median": float(np.median(track_steps)) if track_steps else 0.0,
            "acceleration_median": float(np.median(track_accels)) if track_accels else 0.0,
        }
    return np.asarray(steps), np.asarray(accelerations), per_track


def ratio(after: float | None, before: float | None) -> float | None:
    if after is None or before is None or before == 0:
        return None
    return float(after / before)


def plot_summary(
    output: Path,
    before_xy: np.ndarray,
    after_xy: np.ndarray,
    displacement: np.ndarray,
    before_steps: np.ndarray,
    after_steps: np.ndarray,
    before_accels: np.ndarray,
    after_accels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(before_xy[:, 0], before_xy[:, 1], s=5, alpha=0.25, label="Before")
    ax.scatter(after_xy[:, 0], after_xy[:, 1], s=5, alpha=0.25, label="After")
    ax.set_xlim(-54, 54)
    ax.set_ylim(-36, 36)
    ax.set_aspect("equal")
    ax.set_title("All bottom-middle pitch coordinates")
    ax.set_xlabel("Pitch x (m)")
    ax.set_ylabel("Pitch y (m)")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    upper = max(float(np.quantile(displacement, 0.99)), 0.01)
    ax.hist(np.clip(displacement, 0, upper), bins=40, color="#4472C4", alpha=0.85)
    ax.axvline(float(np.median(displacement)), color="black", linestyle="--", label="Median")
    ax.set_title("Refiner displacement (clipped at p99 for display)")
    ax.set_xlabel("Before-to-after distance (m)")
    ax.set_ylabel("Detections")
    ax.legend()

    ax = axes[1, 0]
    step_data = [before_steps, after_steps]
    ax.boxplot(step_data, labels=["Before", "After"], showfliers=False)
    ax.set_title("Consecutive-frame step distance")
    ax.set_ylabel("Distance (m/frame)")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    accel_data = [before_accels, after_accels]
    ax.boxplot(accel_data, labels=["Before", "After"], showfliers=False)
    ax.set_title("Consecutive-frame second difference (jitter proxy)")
    ax.set_ylabel("Magnitude (m/frame²)")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("SNGS-10004 Refiner coordinate effect — no ground-truth accuracy claim", fontsize=16)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    result_path = args.report_dir / "result.json"
    figure_path = args.report_dir / "coordinate_effect_summary.png"
    readme_path = args.report_dir / "README.md"
    for target in (result_path, figure_path, readme_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    if not args.before.is_file() or not args.after.is_file():
        raise FileNotFoundError("Both fixed Refiner archives must exist")

    before, before_images = load_archive(args.before)
    after, after_images = load_archive(args.after)
    if len(before) != len(after) or len(before) == 0:
        raise AssertionError("Detection row counts must match and be nonzero")
    identity_columns = ["video_id", "image_id", "track_id"]
    if not before[identity_columns].reset_index(drop=True).equals(after[identity_columns].reset_index(drop=True)):
        raise AssertionError("Detection identity/order changed across Refiner")
    if not before_images[["id", "frame"]].reset_index(drop=True).equals(after_images[["id", "frame"]].reset_index(drop=True)):
        raise AssertionError("Image identity/frame contract changed across Refiner")

    frame_by_id = {int(row.id): int(row.frame) for row in before_images.itertuples()}
    frames = np.asarray([frame_by_id[int(v)] for v in before["image_id"]], dtype=np.int64)
    tracks = before["track_id"].to_numpy(dtype=np.int64)
    before_xy = np.asarray([pitch_xy(v) for v in before["bbox_pitch"]], dtype=np.float64)
    after_xy = np.asarray([pitch_xy(v) for v in after["bbox_pitch"]], dtype=np.float64)
    displacement = np.linalg.norm(after_xy - before_xy, axis=1)

    before_steps, before_accels, before_tracks = temporal_metrics(frames, tracks, before_xy)
    after_steps, after_accels, after_tracks = temporal_metrics(frames, tracks, after_xy)
    if before_steps.size != after_steps.size or before_accels.size != after_accels.size:
        raise AssertionError("Temporal comparison support changed")

    before_step_desc = describe(before_steps)
    after_step_desc = describe(after_steps)
    before_accel_desc = describe(before_accels)
    after_accel_desc = describe(after_accels)
    in_bounds_before = (np.abs(before_xy[:, 0]) <= 52.5) & (np.abs(before_xy[:, 1]) <= 34.0)
    in_bounds_after = (np.abs(after_xy[:, 0]) <= 52.5) & (np.abs(after_xy[:, 1]) <= 34.0)

    per_track = []
    for track_id in sorted(before_tracks):
        mask = tracks == track_id
        per_track.append(
            {
                "track_id": track_id,
                "rows": int(np.sum(mask)),
                "median_displacement": float(np.median(displacement[mask])),
                "before_step_median": before_tracks[track_id]["step_median"],
                "after_step_median": after_tracks[track_id]["step_median"],
                "before_acceleration_median": before_tracks[track_id]["acceleration_median"],
                "after_acceleration_median": after_tracks[track_id]["acceleration_median"],
            }
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(
        figure_path,
        before_xy,
        after_xy,
        displacement,
        before_steps,
        after_steps,
        before_accels,
        after_accels,
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "scope": "single-match descriptive Refiner coordinate-effect audit without pitch-coordinate ground truth",
        "inputs": {"before": str(args.before.resolve()), "after": str(args.after.resolve())},
        "rows": int(len(before)),
        "tracks": int(len(set(tracks.tolist()))),
        "frames": int(len(set(frames.tolist()))),
        "identity_contract_preserved": True,
        "coordinates": {
            "changed_rows": int(np.sum(displacement > 1e-9)),
            "displacement_meters": describe(displacement),
            "before_in_pitch_bounds": int(np.sum(in_bounds_before)),
            "after_in_pitch_bounds": int(np.sum(in_bounds_after)),
            "new_out_of_bounds_rows": int(np.sum(in_bounds_before & ~in_bounds_after)),
        },
        "temporal_continuity": {
            "consecutive_step_before": before_step_desc,
            "consecutive_step_after": after_step_desc,
            "step_median_ratio_after_over_before": ratio(after_step_desc["median"], before_step_desc["median"]),
            "second_difference_before": before_accel_desc,
            "second_difference_after": after_accel_desc,
            "second_difference_median_ratio_after_over_before": ratio(after_accel_desc["median"], before_accel_desc["median"]),
        },
        "per_track": per_track,
        "interpretation_boundary": [
            "Lower temporal step or second difference is evidence of smoother coordinates, not ground-truth accuracy.",
            "This single 255-frame match cannot establish cross-match quality.",
        ],
        "verdict": "effect_observed_without_ground_truth_accuracy_claim" if np.any(displacement > 1e-9) else "no_coordinate_effect_observed",
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {"result": str(result_path.resolve()), "figure": str(figure_path.resolve())},
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    readme_path.write_text(
        "# SNGS-10004 Refiner坐标效果对比\n\n"
        "本报告只比较Refiner前后的坐标变化、球场边界和时间连续性。没有二维坐标真值，因此不能把更平滑写成更准确。\n\n"
        "- 机器结果：`result.json`\n"
        "- 可视摘要：`coordinate_effect_summary.png`\n"
    )
    print(json.dumps({key: result[key] for key in ("status", "rows", "tracks", "frames", "coordinates", "temporal_continuity", "verdict")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
