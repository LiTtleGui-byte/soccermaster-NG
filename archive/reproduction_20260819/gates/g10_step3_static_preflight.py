#!/usr/bin/env python3
"""Statically review historical Step 3 and preflight a load/save interface smoke."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO = Path("/home/tianlin/SoccerMaster")
MANIFEST = REPO / "reproduction/manifests/g10_step3_interface_smoke_run1_sngs10004.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, text: str) -> None:
    if not path.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError(f"Output escapes repository: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("Static preflight requires CUDA_VISIBLE_DEVICES='' ")
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["schema_version"], manifest["gate"], manifest["stage"]) != (
        1, "G10-B", "step3_interface_smoke_run1_prepared"
    ):
        raise AssertionError("Unexpected manifest identity")

    output = manifest["outputs"]
    result_path = Path(output["result"])
    readme_path = Path(output["readme"])
    if result_path.exists() or readme_path.exists():
        raise FileExistsError("Preflight outputs already exist")
    for key in ("future_hydra_dir", "future_state"):
        target = Path(output[key])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Future output is already occupied: {target}")

    historical = yaml.safe_load(
        Path(manifest["historical_step3_config"]).read_text(encoding="utf-8")
    )
    if historical["pipeline"] != manifest["historical_pipeline"]:
        raise AssertionError("Historical Step-3 pipeline changed")
    if historical["state"]["load_file"] != "states/refined_sn-gamestate.pklz":
        raise AssertionError("Historical Step-3 input contract changed")
    if historical["state"]["save_file"] != "states/sn-gamestate.pklz":
        raise AssertionError("Historical Step-3 output contract changed")

    apply_source = Path(manifest["apply_camera_params_source"]).read_text(encoding="utf-8")
    if 'output_columns = dict(detection=["bbox_pitch"]' not in apply_source:
        raise AssertionError("Could not prove apply_camera_params writes bbox_pitch")

    prepared = yaml.safe_load(
        Path(manifest["prepared_config"]["path"]).read_text(encoding="utf-8")
    )
    if prepared["pipeline"] != [] or prepared["eval_tracking"] is not False:
        raise AssertionError("Prepared interface smoke must have no modules or evaluation")
    if prepared["visualization"] is not None or prepared["use_wandb"] is not False:
        raise AssertionError("Prepared interface smoke enables a forbidden side effect")
    if prepared["state"]["load_file"] != manifest["input_state"]["path"]:
        raise AssertionError("Prepared input state changed")
    future_state = Path(output["future_state"])
    expected_state = Path(prepared["hydra"]["run"]["dir"]) / prepared["state"]["save_file"]
    if expected_state != future_state:
        raise AssertionError("Prepared output state does not match manifest")

    state_spec = manifest["input_state"]
    state_path = Path(state_spec["path"])
    if state_path.stat().st_size != state_spec["bytes"] or sha256(state_path) != state_spec["sha256"]:
        raise AssertionError("Fixed Refiner state identity changed")
    with zipfile.ZipFile(state_path) as archive:
        if archive.namelist() != state_spec["members_exact"]:
            raise AssertionError("Fixed Refiner ZIP members changed")
        with archive.open("10004.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open("10004_image.pkl") as handle:
            images = pickle.load(handle)
    if not isinstance(detections, pd.DataFrame) or not isinstance(images, pd.DataFrame):
        raise TypeError("Refiner state members are not DataFrames")
    if len(detections) != manifest["detection_rows"] or len(images) != manifest["frame_count"]:
        raise AssertionError("Refiner state row counts changed")
    if detections["track_id"].nunique() != manifest["unique_track_ids"]:
        raise AssertionError("Refiner state track count changed")

    candidate_requirements = {
        "remove_outside": ["track_id", "bbox_pitch"],
        "tracklet_agg": [
            "track_id", "jersey_number_detection", "jersey_number_confidence",
            "role_detection", "role_confidence"
        ],
        "concat_tracklets_by_jn": ["track_id", "jersey_number"],
        "concat_tracklets_by_reid": ["track_id", "embeddings"],
        "team": ["track_id", "embeddings", "role"],
        "team_side": ["track_id", "team_cluster", "bbox_pitch", "role"]
    }
    missing = {
        module: sorted(set(columns) - set(detections.columns))
        for module, columns in candidate_requirements.items()
    }
    if any(missing.values()):
        raise AssertionError(f"Post-Refiner candidate inputs are missing: {missing}")

    result = {
        "status": "passed",
        "stage": manifest["stage"],
        "gpu_used": False,
        "step3_executed": False,
        "input": {
            "detections": len(detections),
            "images": len(images),
            "tracks": int(detections["track_id"].nunique()),
            "columns": list(detections.columns),
        },
        "historical_step3": {
            "pipeline": historical["pipeline"],
            "evaluation_enabled": bool(historical["eval_tracking"]),
            "models_reexecuted": [
                "reid", "pitch", "calibration", "legibility",
                "jersey_number_detect", "role"
            ],
            "refiner_bbox_pitch_overwritten": True,
        },
        "prepared_interface_smoke": {
            "pipeline": [],
            "input": prepared["state"]["load_file"],
            "future_output": str(future_state),
            "gpu_required": False,
            "authorized_or_run": False,
        },
        "post_refiner_candidate": {
            "pipeline": manifest["post_refiner_candidate_pipeline"],
            "all_declared_inputs_present": True,
            "executed": False,
            "semantic_status": "candidate_not_historical_exact_step3",
        },
        "decision_required_after_interface_smoke": (
            "Choose between historical GPU-heavy Step 3, which recomputes bbox_pitch, "
            "and a Refiner-preserving CPU postprocessing variant."
        ),
    }
    write_new(result_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    readme = (
        "# G10 Step 3 static preflight\n\n"
        "The static review passed. No Step 3 module, GPU inference, evaluation, or training ran.\n\n"
        "The saved historical Step 3 is not a light finalization stage: it reruns ReID, "
        "pitch/calibration, legibility, jersey OCR, and role recognition before postprocessing. "
        "In particular, `apply_camera_params` writes `bbox_pitch`, so running it unchanged "
        "after the coord-only Refiner would recompute and overwrite the Refiner output.\n\n"
        "A separate empty-pipeline TrackLab load/save smoke is prepared. It only tests whether "
        "the current 255-frame Refiner archive crosses the Step-3 state interface, and it has "
        "not been executed. The current archive also contains every declared input needed by "
        "the candidate Refiner-preserving postprocessing chain. That candidate is a local "
        "semantic variant, not an exact reproduction of the historical Step 3.\n"
    )
    write_new(readme_path, readme)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
