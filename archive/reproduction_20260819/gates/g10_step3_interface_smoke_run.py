#!/usr/bin/env python3
"""Launch, monitor, and validate the CPU-only Step-3 state interface smoke."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
MANIFEST = REPO / "reproduction/manifests/g10_step3_interface_smoke_run1_execute_sngs10004.json"
WORKER = REPO / "reproduction/gates/g10_step3_interface_smoke_worker.py"


def load_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        with archive.open("10004.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open("10004_image.pkl") as handle:
            images = pickle.load(handle)
    return detections, images, names


def write_new(path: Path, value: Any) -> None:
    if not path.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError("Output escapes repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        if isinstance(value, str):
            handle.write(value)
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if os.environ.get("G10_STEP3_INTERFACE_RUN1_CPU_APPROVED") != "YES":
        raise PermissionError("Missing CPU run approval guard")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CUDA must be hidden")
    for key in ("output_dir", "output_state", "report_dir", "log", "result"):
        path = Path(manifest[key])
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Run target is already occupied: {path}")

    report_dir = Path(manifest["report_dir"])
    report_dir.mkdir(parents=True)
    log_path = Path(manifest["log"])
    command = [
        manifest["python"], str(WORKER), str(MANIFEST),
        f"--config-path={manifest['config_dir']}",
        f"--config-name={manifest['config_name']}",
    ]
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > manifest["timeout_seconds"]:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TimeoutError("CPU interface smoke timed out")
            print(json.dumps({"heartbeat": "running", "pid": process.pid, "elapsed_seconds": round(elapsed, 1)}), flush=True)
            time.sleep(manifest["heartbeat_seconds"])
        exit_code = process.returncode
    if exit_code != 0:
        raise RuntimeError(f"Worker failed with exit code {exit_code}; see {log_path}")

    source_det, source_img, source_names = load_tables(Path(manifest["input_state"]))
    output_det, output_img, output_names = load_tables(Path(manifest["output_state"]))
    if set(output_names) != {"summary.json", "10004.pkl", "10004_image.pkl"}:
        raise AssertionError(f"Unexpected output ZIP members: {output_names}")
    if source_names != ["10004.pkl", "10004_image.pkl"]:
        raise AssertionError("Source ZIP members changed")
    pd.testing.assert_frame_equal(
        source_det.sort_index(axis=1), output_det.sort_index(axis=1),
        check_like=True, check_dtype=True, check_exact=True,
    )
    pd.testing.assert_frame_equal(
        source_img.sort_index(axis=1), output_img.sort_index(axis=1),
        check_like=True, check_dtype=True, check_exact=True,
    )
    if len(output_det) != manifest["expected_detection_rows"] or len(output_img) != manifest["expected_image_rows"]:
        raise AssertionError("Output row count changed")
    if output_det.track_id.nunique() != manifest["expected_tracks"]:
        raise AssertionError("Output track count changed")
    result = {
        "status": "passed",
        "stage": manifest["stage"],
        "exit_code": exit_code,
        "wall_seconds": time.monotonic() - started,
        "gpu_used": False,
        "step3_modules_executed": [],
        "evaluation": False,
        "training": False,
        "input_state": manifest["input_state"],
        "output_state": manifest["output_state"],
        "output_zip_members": output_names,
        "detections": len(output_det),
        "images": len(output_img),
        "tracks": int(output_det.track_id.nunique()),
        "input_output_tables_exact": True,
        "fallbacks": [],
    }
    write_new(Path(manifest["result"]), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
