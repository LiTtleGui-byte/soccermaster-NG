#!/usr/bin/env python3
"""Run the retained CPU-only Step 3 variant on the local Refiner output."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


REPO = Path("/home/tianlin/SoccerMaster")
INPUT = REPO / ".runtime/path_smoke_20260819/soccerfactory/sngs10004_refiner/output/refined_sn-gamestate.pklz"
OUTPUT = REPO / ".runtime/path_smoke_20260819/soccerfactory/sngs10004_step3/states/sn-gamestate.pklz"
REPORT = REPO / "runs/path_smoke_20260819/soccerfactory/step3_sngs10004"
BASE_SCRIPT = REPO / "archive/experiments_20260819/soccerfactory_step3/run_refiner_preserving_step3.py"


def load_historical_step3_module():
    spec = importlib.util.spec_from_file_location("soccerfactory_step3_historical", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_once() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Step 3 is CPU-only and requires CUDA_VISIBLE_DEVICES='' ")
    if REPORT.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite local Step 3 output")

    base = load_historical_step3_module()

    base.INPUT = INPUT
    base.OUTPUT = OUTPUT
    base.REPORT = REPORT
    original, images = base.load_state(INPUT)
    if len(original) != 3390 or len(images) != 255 or original.track_id.nunique() != 48:
        raise AssertionError("Local Refiner state identity changed")
    REPORT.mkdir(parents=True)
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
        if work["__source_row_id"].duplicated().any():
            raise AssertionError(f"{name} duplicated source rows")
        transitions[name] = base.transition(before, work)
        summaries.append(base.summarize(name, work))
    preserved = base.bbox_pitch_preserved(original, work)
    collisions = base.frame_collision_groups(work)
    if not preserved or collisions:
        raise AssertionError("Retained Step 3 variant changed coordinates or created frame collisions")
    final_team_tracks = {
        str(team): int(group.track_id.nunique())
        for team, group in work.assign(team=work.team.fillna("<null>")).groupby("team")
    }
    final_role_tracks = {
        str(role): int(group.track_id.nunique())
        for role, group in work.assign(role=work.role.fillna("<null>")).groupby("role")
    }
    base.save_state(work, images)
    base.plot_stage_summary(summaries)
    base.plot_pitch(work)
    result = {
        "status": "passed", "input": str(INPUT), "output": str(OUTPUT),
        "pipeline": [name for name, _ in modules], "stages": summaries,
        "transitions": transitions, "bbox_pitch_exactly_preserved": preserved,
        "same_frame_track_collisions": collisions, "final_team_tracks": final_team_tracks,
        "final_role_tracks": final_role_tracks, "gpu_used": False, "training": False,
        "quality_validated": False,
        "visuals": [str(REPORT / "pipeline_changes.png"), str(REPORT / "final_pitch_tracks.png")],
    }
    (REPORT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "passed", "result": str(REPORT / "result.json")}, ensure_ascii=False))
    return 0


def main() -> int:
    stopped = threading.Event()

    def heartbeat() -> None:
        started = time.monotonic()
        while not stopped.wait(30):
            print(f"heartbeat phase=local_step3_cpu elapsed_seconds={time.monotonic()-started:.1f}", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return run_once()
    finally:
        stopped.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
