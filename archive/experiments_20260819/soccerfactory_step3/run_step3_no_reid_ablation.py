#!/usr/bin/env python3
"""Run the Refiner-preserving Step-3 CPU ablation without ReID concatenation."""

from __future__ import annotations

import copy
import json
import pickle
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_refiner_preserving_step3 as base


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/g10/20260819_step3_no_reid_ablation"
OUTPUT = REPO / ".runtime/g10/sngs10004_step3_no_reid/run1/states/sn-gamestate.pklz"
WITH_REID_RESULT = REPO / "reports/g10/20260819_step3_refiner_preserving_cpu/result.json"


def plot_comparison(with_reid: dict, without_reid: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ["with ReID merge", "without ReID merge"]
    tracks = [with_reid["stages"][-1]["tracks"], without_reid["stages"][-1]["tracks"]]
    left_tracks = [with_reid["final_team_tracks"].get("left", 0), without_reid["final_team_tracks"].get("left", 0)]
    right_tracks = [with_reid["final_team_tracks"].get("right", 0), without_reid["final_team_tracks"].get("right", 0)]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(x, tracks, color=["#d65f5f", "#4c956c"])
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel("tracks")
    axes[0].set_title("Track count")
    axes[1].bar(x, left_tracks, label="left", color="#2878b5")
    axes[1].bar(x, right_tracks, bottom=left_tracks, label="right", color="#d64f4f")
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].set_ylabel("team-assigned tracks")
    axes[1].set_title("Final team distribution")
    axes[1].legend()
    fig.savefig(REPORT / "with_vs_without_reid.png", dpi=170)
    plt.close(fig)


def main() -> int:
    if REPORT.exists() or OUTPUT.exists():
        raise FileExistsError("No-ReID ablation output already exists")
    REPORT.mkdir(parents=True)
    base.REPORT = REPORT
    base.OUTPUT = OUTPUT
    original, images = base.load_state(base.INPUT)
    work = copy.deepcopy(original)
    work["__source_row_id"] = work.index.astype(int)
    work["__original_track_id"] = work.track_id.astype(int)
    work = work.drop(columns=base.GENERATED_COLUMNS)

    modules = [
        ("remove_outside", base.RemoveOutside()),
        ("tracklet_agg_1", base.MajorityVoteTrackletFilter2(
            cfg=SimpleNamespace(attributes=["jersey_number", "role"]), device="cpu"
        )),
        ("concat_by_jersey", base.ConcatTrackletsByJN()),
        ("tracklet_agg_2", base.MajorityVoteTrackletFilter2(
            cfg=SimpleNamespace(attributes=["jersey_number", "role"]), device="cpu"
        )),
        ("team_cluster", base.TrackletTeamClustering()),
        ("team_side", base.TrackletTeamSideLabeling()),
    ]
    summaries = [base.summarize("refiner_input", original), base.summarize("pipeline_load", work)]
    transitions = {}
    for name, module in modules:
        before = work.copy(deep=True)
        work = module.process(work, images)
        transitions[name] = base.transition(before, work)
        summaries.append(base.summarize(name, work))

    preserved = base.bbox_pitch_preserved(original, work)
    collisions = base.frame_collision_groups(work)
    final_team_tracks = {
        str(team): int(group.track_id.nunique())
        for team, group in work.assign(team=work.team.fillna("<null>")).groupby("team")
    }
    final_role_tracks = {
        str(role): int(group.track_id.nunique())
        for role, group in work.assign(role=work.role.fillna("<null>")).groupby("role")
    }
    xy = [base.pitch_xy(value) for value in work.bbox_pitch]
    base.save_state(work, images)
    base.plot_stage_summary(summaries)
    base.plot_pitch(work)
    with_reid = json.loads(WITH_REID_RESULT.read_text(encoding="utf-8"))
    result = {
        "status": "passed",
        "experiment": "refiner_preserving_step3_without_reid_merge",
        "input": str(base.INPUT),
        "output": str(OUTPUT),
        "pipeline": [name for name, _ in modules],
        "stages": summaries,
        "transitions": transitions,
        "bbox_pitch_exactly_preserved": preserved,
        "same_frame_track_collisions": collisions,
        "final_team_tracks": final_team_tracks,
        "final_role_tracks": final_role_tracks,
        "pitch_range": {
            "x": [min(value[0] for value in xy), max(value[0] for value in xy)],
            "y": [min(value[1] for value in xy), max(value[1] for value in xy)],
        },
        "comparison": {
            "with_reid_tracks": with_reid["stages"][-1]["tracks"],
            "without_reid_tracks": summaries[-1]["tracks"],
            "reid_merge_removed_from_candidate": True,
        },
        "warnings": ["role_output_collapsed_to_player", "pitch_coordinates_cover_only_left_half"],
        "decision": "retain_no_reid_variant_as_safer_structural_candidate_not_quality_validated",
        "gpu_used": False,
        "training": False,
    }
    (REPORT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    plot_comparison(with_reid, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
