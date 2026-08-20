#!/usr/bin/env python3
"""Separate Refiner temporal changes at 100-frame clip boundaries from clip interiors."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
BEFORE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
AFTER = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
REPORT = REPO / "reports/g10/20260819_refiner_coordinate_quality"
RESULT = REPORT / "clip_boundary_diagnostic.json"
FIGURE = REPORT / "jitter_by_frame.png"
VIDEO_ID = "10004"
CLIP_BOUNDARIES = (100, 200)


def atomic_json(path: Path, value: Any) -> None:
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


def summarize(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "p95": None, "mean": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    if RESULT.exists() or FIGURE.exists():
        raise FileExistsError("Refusing to overwrite clip-boundary diagnostic outputs")

    before, images = load(BEFORE)
    after, after_images = load(AFTER)
    identity = ["image_id", "track_id"]
    if not before[identity].equals(after[identity]):
        raise AssertionError("Detection identity/order changed")
    if not images[["id", "frame"]].equals(after_images[["id", "frame"]]):
        raise AssertionError("Image identity/frame mapping changed")

    frame_by_image = {row.id: int(row.frame) for row in images.itertuples(index=False)}
    before_xy = coordinates(before)
    after_xy = coordinates(after)
    work = before[identity].copy()
    work["frame"] = work.image_id.map(frame_by_image)
    work[["before_x", "before_y"]] = before_xy
    work[["after_x", "after_y"]] = after_xy

    step_rows: list[dict[str, Any]] = []
    jitter_rows: list[dict[str, Any]] = []
    for track_id, group in work.groupby("track_id", sort=True):
        group = group.sort_values("frame", kind="stable")
        frames = group.frame.to_numpy(dtype=np.int64)
        xy_before = group[["before_x", "before_y"]].to_numpy(dtype=np.float64)
        xy_after = group[["after_x", "after_y"]].to_numpy(dtype=np.float64)
        for index in range(1, len(frames)):
            if frames[index] - frames[index - 1] != 1:
                continue
            end_frame = int(frames[index])
            step_rows.append(
                {
                    "track_id": int(float(track_id)),
                    "end_frame": end_frame,
                    "crosses_boundary": end_frame in CLIP_BOUNDARIES,
                    "before": float(np.linalg.norm(xy_before[index] - xy_before[index - 1])),
                    "after": float(np.linalg.norm(xy_after[index] - xy_after[index - 1])),
                }
            )
        for index in range(1, len(frames) - 1):
            if frames[index] - frames[index - 1] != 1 or frames[index + 1] - frames[index] != 1:
                continue
            center_frame = int(frames[index])
            crosses_boundary = any(frames[index - 1] < boundary <= frames[index + 1] for boundary in CLIP_BOUNDARIES)
            jitter_rows.append(
                {
                    "track_id": int(float(track_id)),
                    "center_frame": center_frame,
                    "touches_boundary": bool(crosses_boundary),
                    "before": float(np.linalg.norm(xy_before[index + 1] - 2 * xy_before[index] + xy_before[index - 1])),
                    "after": float(np.linalg.norm(xy_after[index + 1] - 2 * xy_after[index] + xy_after[index - 1])),
                }
            )

    steps = pd.DataFrame(step_rows)
    jitter = pd.DataFrame(jitter_rows)
    steps["increase"] = steps.after - steps.before
    jitter["increase"] = jitter.after - jitter.before

    def split_summary(table: pd.DataFrame, flag: str) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for label, mask in (("boundary", table[flag]), ("interior", ~table[flag])):
            subset = table.loc[mask]
            output[label] = {
                "before_m": summarize(subset.before.tolist()),
                "after_m": summarize(subset.after.tolist()),
                "after_minus_before_m": summarize(subset.increase.tolist()),
                "fraction_increased": float((subset.increase > 0).mean()) if len(subset) else None,
            }
        return output

    step_summary = split_summary(steps, "crosses_boundary")
    jitter_summary = split_summary(jitter, "touches_boundary")
    boundary_jitter_fraction = float(jitter.touches_boundary.mean())
    interior_jitter_increase = jitter_summary["interior"]["after_m"]["median"] - jitter_summary["interior"]["before_m"]["median"]
    boundary_only_explanation_rejected = bool(interior_jitter_increase > 0 and boundary_jitter_fraction < 0.1)

    by_frame = jitter.groupby("center_frame")[["before", "after", "increase"]].median().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(by_frame.center_frame, by_frame.before, label="before", color="#657786", linewidth=1.2)
    axes[0].plot(by_frame.center_frame, by_frame.after, label="after", color="#35a76f", linewidth=1.2)
    axes[0].set_ylabel("median second difference (m)")
    axes[0].set_title("Refiner temporal jitter by frame")
    axes[0].legend()
    axes[1].plot(by_frame.center_frame, by_frame.increase, color="#d35f5f", linewidth=1.1)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("after − before (m)")
    axes[1].set_xlabel("center frame")
    for axis in axes:
        for boundary in CLIP_BOUNDARIES:
            axis.axvline(boundary, color="#e49b38", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.18)
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)

    result = {
        "status": "passed",
        "gpu_used": False,
        "clip_ranges": [[0, 99], [100, 199], [200, 254]],
        "clip_boundaries": list(CLIP_BOUNDARIES),
        "consecutive_step": step_summary,
        "second_difference_jitter": jitter_summary,
        "boundary_jitter_sample_fraction": boundary_jitter_fraction,
        "boundary_only_explanation_rejected": boundary_only_explanation_rejected,
        "interpretation": (
            "Clip boundaries are not the primary explanation if interior jitter also rises and "
            "boundary-adjacent samples are a small fraction of all samples."
        ),
        "figure": str(FIGURE),
    }
    atomic_json(RESULT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
