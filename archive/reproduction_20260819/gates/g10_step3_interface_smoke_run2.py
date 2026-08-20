#!/usr/bin/env python3
"""Static preflight and future launcher for Step-3 interface smoke run2."""

from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO = Path("/home/tianlin/SoccerMaster")
MANIFEST = REPO / "reproduction/manifests/g10_step3_interface_smoke_run2_sngs10004.json"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args()


def write_new(path: Path, value: Any) -> None:
    if not path.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError("Output escapes repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        with archive.open("10004.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open("10004_image.pkl") as handle:
            images = pickle.load(handle)
    return detections, images, names


def preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("Preflight must hide CUDA")
    if manifest["execution_authorized"] is not False:
        raise AssertionError("Prepared run2 unexpectedly authorized")
    config = yaml.safe_load(Path(manifest["config_path"]).read_text(encoding="utf-8"))
    if config["pipeline"] != [] or config["eval_tracking"] is not False:
        raise AssertionError("Run2 is not an empty-pipeline no-eval smoke")
    if config["visualization"] is not None or config["use_wandb"] is not False:
        raise AssertionError("Run2 enables a forbidden side effect")
    if config["state"]["load_file"] != manifest["input_state"]:
        raise AssertionError("Run2 input changed")
    expected_output = Path(config["hydra"]["run"]["dir"]) / config["state"]["save_file"]
    if expected_output != Path(manifest["output_state"]):
        raise AssertionError("Run2 output contract changed")
    for key in ("output_dir", "output_state", "cache_dir", "report_dir"):
        path = Path(manifest[key])
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Future run2 path occupied: {path}")
    worker_text = Path(manifest["worker"]).read_text(encoding="utf-8")
    ast.parse(worker_text)
    launcher_text = Path(__file__).read_text(encoding="utf-8")
    if 'f"--config-dir={manifest[\'config_dir\']}"' not in launcher_text:
        raise AssertionError("Run2 launcher does not use --config-dir")
    forbidden_launcher_fragment = 'f"--config' + '-path='
    if forbidden_launcher_fragment in launcher_text:
        raise AssertionError("Run2 launcher still contains --config-path")
    detections, images, members = load_tables(Path(manifest["input_state"]))
    if members != ["10004.pkl", "10004_image.pkl"]:
        raise AssertionError("Input ZIP members changed")
    if len(detections) != manifest["expected_detection_rows"] or len(images) != manifest["expected_image_rows"]:
        raise AssertionError("Input row counts changed")
    return {
        "status": "passed",
        "stage": manifest["stage"],
        "gpu_used": False,
        "run2_executed": False,
        "hydra_local_config_option": "--config-dir",
        "only_delta_from_run1_cause": "replace --config-path with --config-dir and isolate run2 paths",
        "pipeline": [],
        "detections": len(detections),
        "images": len(images),
        "tracks": int(detections.track_id.nunique()),
        "future_paths_unused": True,
    }


def validate_output(manifest: dict[str, Any]) -> dict[str, Any]:
    source_det, source_img, _ = load_tables(Path(manifest["input_state"]))
    output_det, output_img, names = load_tables(Path(manifest["output_state"]))
    if set(names) != {"summary.json", "10004.pkl", "10004_image.pkl"}:
        raise AssertionError("Unexpected output ZIP members")
    pd.testing.assert_frame_equal(source_det.sort_index(axis=1), output_det.sort_index(axis=1), check_exact=True)
    pd.testing.assert_frame_equal(source_img.sort_index(axis=1), output_img.sort_index(axis=1), check_exact=True)
    return {"members": names, "input_output_tables_exact": True}


def run(manifest: dict[str, Any]) -> int:
    if not manifest["execution_authorized"] or os.environ.get("G10_STEP3_INTERFACE_RUN2_CPU_APPROVED") != "YES":
        raise PermissionError("Run2 is not authorized")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CUDA must remain hidden")
    for key in ("output_dir", "cache_dir", "report_dir"):
        if Path(manifest[key]).exists():
            raise FileExistsError(f"Run2 target occupied: {manifest[key]}")
    report = Path(manifest["report_dir"])
    report.mkdir(parents=True)
    cache = Path(manifest["cache_dir"])
    cache.mkdir(parents=True)
    command = [
        manifest["python"], manifest["worker"], str(MANIFEST),
        f"--config-dir={manifest['config_dir']}",
        f"--config-name={manifest['config_name']}",
    ]
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": str(cache / "hf"), "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "transformers"), "XDG_CACHE_HOME": str(cache / "xdg"),
        "MPLCONFIGDIR": str(cache / "mpl"), "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
    })
    started = time.monotonic()
    with Path(manifest["log"]).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > manifest["timeout_seconds"]:
                process.terminate()
                process.wait(timeout=30)
                raise TimeoutError("Run2 timed out")
            print(json.dumps({"heartbeat": "running", "elapsed_seconds": round(elapsed, 1)}), flush=True)
            time.sleep(manifest["heartbeat_seconds"])
    if process.returncode != 0:
        raise RuntimeError(f"Run2 worker exited {process.returncode}")
    output = validate_output(manifest)
    result = {
        "status": "passed", "stage": "step3_interface_smoke_run2_execute",
        "exit_code": 0, "wall_seconds": time.monotonic() - started,
        "gpu_used": False, "step3_modules_executed": [], "evaluation": False,
        "training": False, **output,
    }
    write_new(Path(manifest["result"]), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = arguments()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.mode == "preflight":
        result = preflight(manifest)
        write_new(Path(manifest["preflight_result"]), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return run(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
