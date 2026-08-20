#!/usr/bin/env python3
"""Run and visualize the CPU-only Refiner-preserving Step-3 postprocessing chain."""

from __future__ import annotations

import copy
import json
import os
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_original_makedirs = os.makedirs
_blocked_mim_cache = Path("/home/tianlin/.cache/mim").resolve(strict=False)


def _guarded_makedirs(name: Any, *args: Any, **kwargs: Any) -> Any:
    if Path(name).resolve(strict=False) == _blocked_mim_cache:
        return None
    return _original_makedirs(name, *args, **kwargs)


os.makedirs = _guarded_makedirs
try:
    from sn_gamestate.concat_tracklets_by_jn.concat_tracklets_by_jn_api import ConcatTrackletsByJN
    from sn_gamestate.concat_tracklets_by_reid.concat_tracklets_by_reid_api import ConcatTrackletsByReid
    from sn_gamestate.remove_outside.remove_outside_api import RemoveOutside
    from sn_gamestate.team.tracklet_team_clustering_api import TrackletTeamClustering
    from sn_gamestate.team.tracklet_team_side_labeling_api import TrackletTeamSideLabeling
    from sn_gamestate.tracklet_agg.majority_vote_filter_api import MajorityVoteTrackletFilter2
finally:
    os.makedirs = _original_makedirs


REPO = Path("/home/tianlin/SoccerMaster")
INPUT = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
OUTPUT = REPO / ".runtime/g10/sngs10004_step3_refiner_preserving/run1/states/sn-gamestate.pklz"
REPORT = REPO / "reports/g10/20260819_step3_refiner_preserving_cpu"
GENERATED_COLUMNS = ["jersey_number", "role", "team_cluster", "team"]


def load_state(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("10004.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open("10004_image.pkl") as handle:
            images = pickle.load(handle)
    return detections, images


def counts(series: pd.Series) -> dict[str, int]:
    values = series.fillna("<null>").astype(str).value_counts()
    return {key: int(value) for key, value in values.items()}


def summarize(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "stage": name,
        "rows": len(frame),
        "tracks": int(frame.track_id.nunique()),
        "role": counts(frame["role"]) if "role" in frame else {},
        "team": counts(frame["team"]) if "team" in frame else {},
        "jersey_non_null": int(frame["jersey_number"].notna().sum()) if "jersey_number" in frame else 0,
        "jersey_unique": int(frame["jersey_number"].dropna().nunique()) if "jersey_number" in frame else 0,
    }


def transition(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    before_ids = set(before["__source_row_id"])
    after_ids = set(after["__source_row_id"])
    common = sorted(before_ids & after_ids)
    before_map = before.set_index("__source_row_id").loc[common, "track_id"]
    after_map = after.set_index("__source_row_id").loc[common, "track_id"]
    groups = (
        after.groupby("track_id")["__original_track_id"]
        .agg(lambda values: sorted({int(value) for value in values}))
    )
    merged = [
        {"new_track_id": int(track_id), "source_track_ids": source_ids}
        for track_id, source_ids in groups.items() if len(source_ids) > 1
    ]
    return {
        "removed_rows": len(before_ids - after_ids),
        "changed_track_id_rows": int((before_map.to_numpy() != after_map.to_numpy()).sum()),
        "merged_groups_from_original_tracks": merged,
    }


def pitch_xy(value: Any) -> tuple[float, float]:
    if not isinstance(value, dict):
        return np.nan, np.nan
    return float(value["x_bottom_middle"]), float(value["y_bottom_middle"])


def bbox_pitch_preserved(original: pd.DataFrame, final: pd.DataFrame) -> bool:
    original_map = original.assign(__source_row_id=original.index).set_index("__source_row_id")["bbox_pitch"]
    final_map = final.set_index("__source_row_id")["bbox_pitch"]
    for source_row_id, right in final_map.items():
        left = original_map.loc[source_row_id]
        if not isinstance(left, dict) or not isinstance(right, dict):
            if not (pd.isna(left) and pd.isna(right)):
                return False
            continue
        if left.keys() != right.keys():
            return False
        if any(float(left[key]) != float(right[key]) for key in left):
            return False
    return True


def frame_collision_groups(final: pd.DataFrame) -> list[dict[str, Any]]:
    collisions = []
    for track_id, group in final.groupby("track_id"):
        duplicate_frames = group.image_id[group.image_id.duplicated()].unique().tolist()
        if duplicate_frames:
            collisions.append({"track_id": int(track_id), "duplicate_image_ids": duplicate_frames})
    return collisions


def save_state(detections: pd.DataFrame, images: pd.DataFrame) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    clean = detections.drop(columns=["__source_row_id", "__original_track_id"])
    summary = {"columns": {"detection": clean.columns.tolist(), "image": images.columns.tolist()}}
    with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        archive.writestr("10004.pkl", pickle.dumps(clean, protocol=pickle.DEFAULT_PROTOCOL))
        archive.writestr("10004_image.pkl", pickle.dumps(images, protocol=pickle.DEFAULT_PROTOCOL))


def plot_stage_summary(stages: list[dict[str, Any]]) -> None:
    names = [stage["stage"] for stage in stages]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(x, [stage["rows"] for stage in stages], marker="o", color="#2878b5", label="detections")
    track_axis = axes[0].twinx()
    track_axis.plot(x, [stage["tracks"] for stage in stages], marker="o", color="#e57c23", label="tracks")
    axes[0].set_ylabel("detections", color="#2878b5")
    track_axis.set_ylabel("tracks", color="#e57c23")
    track_axis.set_ylim(0, max(stage["tracks"] for stage in stages) * 1.15)
    axes[0].legend(loc="upper left")
    track_axis.legend(loc="upper right")
    axes[0].grid(alpha=0.25)
    role_names = sorted({key for stage in stages for key in stage["role"]})
    for role in role_names:
        axes[1].plot(x, [stage["role"].get(role, 0) for stage in stages], marker="o", label=role)
    axes[1].set_ylabel("role rows")
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=0.25)
    team_names = sorted({key for stage in stages for key in stage["team"]})
    for team in team_names:
        axes[2].plot(x, [stage["team"].get(team, 0) for stage in stages], marker="o", label=team)
    axes[2].set_ylabel("team rows")
    axes[2].legend(ncol=4)
    axes[2].grid(alpha=0.25)
    axes[2].set_xticks(x, names, rotation=30, ha="right")
    fig.suptitle("Refiner-preserving Step-3 module trace")
    fig.savefig(REPORT / "pipeline_changes.png", dpi=170)
    plt.close(fig)


def plot_pitch(final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.set_facecolor("#eff8ed")
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="black")
    ax.axvline(0, color="black", lw=1)
    colors = {"left": "#2878b5", "right": "#d64f4f"}
    for track_id, group in final.sort_values("image_id").groupby("track_id"):
        xy = np.asarray([pitch_xy(value) for value in group.bbox_pitch], dtype=float)
        team = str(group.team.dropna().iloc[0]) if group.team.notna().any() else "<null>"
        color = colors.get(team, "#777777")
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.28, lw=1)
        ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.32, s=8)
    ax.set_xlim(-62.5, 62.5)
    ax.set_ylim(-39, 39)
    ax.set_aspect("equal")
    ax.set_xlabel("pitch x (m)")
    ax.set_ylabel("pitch y (m)")
    ax.set_title("Final tracks and assigned team (coordinates preserved from Refiner)")
    fig.savefig(REPORT / "final_pitch_tracks.png", dpi=170)
    plt.close(fig)


def main() -> int:
    if REPORT.exists() or OUTPUT.exists():
        raise FileExistsError("Experiment output already exists")
    REPORT.mkdir(parents=True)
    original, images = load_state(INPUT)
    work = copy.deepcopy(original)
    work["__source_row_id"] = work.index.astype(int)
    work["__original_track_id"] = work.track_id.astype(int)
    work = work.drop(columns=GENERATED_COLUMNS)

    modules = [
        ("remove_outside", RemoveOutside()),
        ("tracklet_agg_1", MajorityVoteTrackletFilter2(
            cfg=SimpleNamespace(attributes=["jersey_number", "role"]), device="cpu"
        )),
        ("concat_by_jersey", ConcatTrackletsByJN()),
        ("concat_by_reid", ConcatTrackletsByReid(threshold=0.1)),
        ("tracklet_agg_2", MajorityVoteTrackletFilter2(
            cfg=SimpleNamespace(attributes=["jersey_number", "role"]), device="cpu"
        )),
        ("team_cluster", TrackletTeamClustering()),
        ("team_side", TrackletTeamSideLabeling()),
    ]
    summaries = [summarize("refiner_input", original), summarize("pipeline_load", work)]
    transitions: dict[str, Any] = {}
    for name, module in modules:
        before = work.copy(deep=True)
        work = module.process(work, images)
        if work["__source_row_id"].duplicated().any():
            raise AssertionError(f"{name} duplicated source rows")
        transitions[name] = transition(before, work)
        summaries.append(summarize(name, work))

    preserved = bbox_pitch_preserved(original, work)
    collisions = frame_collision_groups(work)
    final_team_tracks = {
        str(team): int(group.track_id.nunique())
        for team, group in work.assign(team=work.team.fillna("<null>")).groupby("team")
    }
    final_role_tracks = {
        str(role): int(group.track_id.nunique())
        for role, group in work.assign(role=work.role.fillna("<null>")).groupby("role")
    }
    warnings = []
    if not preserved:
        warnings.append("bbox_pitch_changed")
    if collisions:
        warnings.append("merged_track_has_same_frame_collision")
    merged_after_jersey = transitions["concat_by_jersey"]["merged_groups_from_original_tracks"]
    if merged_after_jersey:
        warnings.append("jersey_merge_combines_original_tracks_before_team_assignment")
    if set(final_role_tracks) == {"player"}:
        warnings.append("role_output_collapsed_to_player")
    nonempty_team_counts = [value for key, value in final_team_tracks.items() if key != "<null>" and value]
    if len(nonempty_team_counts) == 2 and max(nonempty_team_counts) > 4 * min(nonempty_team_counts):
        warnings.append("team_track_distribution_severely_imbalanced")

    save_state(work, images)
    plot_stage_summary(summaries)
    plot_pitch(work)
    decision = "reject" if (not preserved or collisions) else "diagnostic_only_not_production"
    result = {
        "status": "passed",
        "experiment": "refiner_preserving_step3_cpu_run1",
        "input": str(INPUT),
        "output": str(OUTPUT),
        "pipeline": [name for name, _ in modules],
        "stages": summaries,
        "transitions": transitions,
        "bbox_pitch_exactly_preserved": preserved,
        "same_frame_track_collisions": collisions,
        "final_team_tracks": final_team_tracks,
        "final_role_tracks": final_role_tracks,
        "warnings": warnings,
        "decision": decision,
        "gpu_used": False,
        "training": False,
    }
    (REPORT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
