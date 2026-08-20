#!/usr/bin/env python3
"""Guarded local Refiner inference on the local enrichment archive."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import signal
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/tianlin/SoccerMaster")
PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
SOURCE = REPO / "vendor/soccerfactory/refiner/inference.py"
CONFIG = REPO / "reproduction/configs/local_takeover/g10_refiner_sngs10004.json"
CHECKPOINT = REPO / ".local_assets/models/soccerfactory/refiner/best_model.pth"
INPUT = REPO / ".runtime/local_takeover/g10/sngs10004_enrichment/states/sn-gamestate.pklz"
METADATA = REPO / "reproduction/manifests/g10_sngs10004_refiner_metadata.json"
WORK = REPO / ".runtime/local_takeover/g10/sngs10004_refiner/work"
OUTPUT_DIR = REPO / ".runtime/local_takeover/g10/sngs10004_refiner/output"
OUTPUT = OUTPUT_DIR / "refined_sn-gamestate.pklz"
REPORT = REPO / "reports/local_takeover/20260819_soccerfactory_refiner_sngs10004"
RESULT = REPORT / "result.json"
LOG = REPORT / "run.log"
APPROVAL = "G10_LOCAL_REFINER_GPU_APPROVED"
TIMEOUT_SECONDS = 14400


def fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def read_state(path: Path) -> tuple[Any, Any, list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if archive.testzip() is not None:
            raise AssertionError("State ZIP CRC failed")
        detections = pickle.load(archive.open("10004.pkl"))
        images = pickle.load(archive.open("10004_image.pkl"))
    return detections, images, members


def preflight() -> dict[str, Any]:
    for path in (SOURCE, CONFIG, CHECKPOINT, INPUT, METADATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    if CHECKPOINT.stat().st_size != 323985486:
        raise AssertionError("Local Refiner checkpoint size changed")
    config_text = CONFIG.read_text(encoding="utf-8")
    if "/remote-home" in config_text:
        raise AssertionError("Local Refiner config contains /remote-home")
    config = json.loads(config_text)
    if config["data"]["max_frames"] != 255 or config["data"]["max_clip_frames"] != 100:
        raise AssertionError("Refiner 255-frame contract changed")
    tasks = config["training"]["tasks"]
    if tasks != {"track": False, "role": False, "team": False, "jersey": False, "coord": True, "missing": False}:
        raise AssertionError("Refiner must remain coord-only")
    base = Path(config["imports"][0])
    if not base.is_file() or not base.is_relative_to(REPO / "vendor/soccerfactory/refiner"):
        raise AssertionError("Refiner base config is not vendored")
    metadata = yaml.safe_load(METADATA.read_text(encoding="utf-8"))
    if metadata.get("sn500") != [{"id": 3, "name": "SNGS-10004", "n_frames": 255}]:
        raise AssertionError("Refiner metadata changed")
    detections, images, members = read_state(INPUT)
    required = {"bbox_ltwh", "bbox_pitch", "role", "team", "jersey_number", "track_id"}
    if not required.issubset(detections.columns) or "parameters" not in images.columns:
        raise AssertionError("Enrichment archive lacks Refiner inputs")
    if len(detections) != 3390 or len(images) != 255 or detections["track_id"].nunique() != 48:
        raise AssertionError("Enrichment archive identity changed")
    for path in (WORK, OUTPUT_DIR, REPORT, OUTPUT):
        fresh(path)
    return {
        "source": str(SOURCE), "config": str(CONFIG), "checkpoint": str(CHECKPOINT),
        "input": str(INPUT), "input_members": members, "image_rows": len(images),
        "detection_rows": len(detections), "unique_track_ids": int(detections["track_id"].nunique()),
        "max_frames": 255, "max_clip_frames": 100, "enabled_tasks": ["coord"],
        "remote_runtime_paths": False, "future_outputs_fresh": True,
    }


def terminate(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def inspect_output() -> dict[str, Any]:
    before, before_images, _ = read_state(INPUT)
    after, after_images, members = read_state(OUTPUT)
    if sorted(members) != ["10004.pkl", "10004_image.pkl"]:
        raise AssertionError(f"Unexpected Refiner members: {members}")
    if not before.index.equals(after.index) or not before_images.index.equals(after_images.index):
        raise AssertionError("Refiner changed row identity")
    if len(after) != 3390 or after["track_id"].nunique() != 48:
        raise AssertionError("Refiner changed row/track counts")
    if not before["track_id"].equals(after["track_id"]):
        raise AssertionError("Refiner changed track_id")
    coordinate_keys = [
        "x_bottom_left", "y_bottom_left", "x_bottom_right", "y_bottom_right",
        "x_bottom_middle", "y_bottom_middle",
    ]
    finite = 0
    changed = 0
    for old, new in zip(before["bbox_pitch"], after["bbox_pitch"]):
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise AssertionError("bbox_pitch is not the expected coordinate dictionary")
        if set(old) != set(coordinate_keys) or set(new) != set(coordinate_keys):
            raise AssertionError("bbox_pitch coordinate keys changed")
        values = [new[key] for key in coordinate_keys]
        if not all(math.isfinite(float(value)) for value in values):
            raise AssertionError("Refiner produced a non-finite bbox_pitch")
        finite += 1
        changed += int(any(float(old[key]) != float(new[key]) for key in coordinate_keys))
    return {
        "path": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "members": members,
        "image_rows": len(after_images), "detection_rows": len(after),
        "unique_track_ids": int(after["track_id"].nunique()),
        "finite_bbox_pitch_rows": finite, "changed_bbox_pitch_rows": changed,
        "track_ids_preserved": True,
    }


def run(checked: dict[str, Any]) -> int:
    if os.environ.get(APPROVAL) != "YES":
        raise PermissionError(f"Missing approval guard: {APPROVAL}=YES")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise PermissionError("Exactly one CUDA_VISIBLE_DEVICES index is required")
    WORK.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    command = [
        str(PYTHON), str(SOURCE), "--config", str(CONFIG), "--checkpoint", str(CHECKPOINT),
        "--input_pklz", str(INPUT), "--output_dir", str(OUTPUT_DIR),
        "--metadata_path", str(METADATA), "--split", "sn500",
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(REPO / "vendor/soccerfactory/refiner"),
        "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(WORK / "matplotlib"),
    })
    started = time.monotonic()
    timed_out = False
    with LOG.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_visible_devices": visible}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        next_heartbeat = started + 30
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= TIMEOUT_SECONDS:
                timed_out = True
                terminate(process)
                break
            if now >= next_heartbeat:
                print(f"heartbeat phase=refiner_inference elapsed_seconds={now-started:.1f}", flush=True)
                next_heartbeat += 30
            time.sleep(1)
        exit_code = process.wait()
    result: dict[str, Any] = {
        "status": "failed", "process_exit_code": exit_code, "timed_out": timed_out,
        "wall_seconds": round(time.monotonic() - started, 3), "cuda_visible_devices": visible,
        "training": False, "evaluation": False, "visualization": False,
        "preflight": checked, "log": str(LOG),
    }
    final_exit = exit_code or (1 if timed_out else 0)
    if final_exit == 0:
        try:
            result["output"] = inspect_output()
            result["status"] = "passed"
        except Exception as error:
            result["artifact_error"] = f"{type(error).__name__}: {error}"
            final_exit = 1
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "result": str(RESULT)}), flush=True)
    return final_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--inspect-existing", action="store_true")
    args = parser.parse_args()
    if args.run and args.inspect_existing:
        raise ValueError("Choose only one of --run and --inspect-existing")
    if args.inspect_existing:
        previous = json.loads(RESULT.read_text(encoding="utf-8"))
        if previous.get("process_exit_code") != 0 or not OUTPUT.is_file():
            raise AssertionError("No successful GPU process output is available for CPU reinspection")
        previous_error = previous.pop("artifact_error", None)
        previous["output"] = inspect_output()
        previous["status"] = "passed"
        previous["artifact_check_correction"] = {
            "gpu_rerun": False,
            "previous_error": previous_error,
            "reason": "The initial checker treated the six-key bbox_pitch dictionary as a four-value array.",
        }
        RESULT.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "passed", "result": str(RESULT), "gpu_rerun": False}))
        return 0
    checked = preflight()
    if not args.run:
        print(json.dumps({"status": "preflight_passed", **checked}, indent=2))
        return 0
    return run(checked)


if __name__ == "__main__":
    raise SystemExit(main())
