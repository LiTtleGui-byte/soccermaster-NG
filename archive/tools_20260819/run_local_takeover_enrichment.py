#!/usr/bin/env python3
"""Guarded local enrichment smoke using the existing phase-observable worker."""

from __future__ import annotations

import argparse
import copy
import json
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
BASE = REPO / "reproduction/manifests/g10_prerefiner_enrichment_run2_sngs10004.json"
CONFIG = REPO / "reproduction/configs/local_takeover/g10_enrichment_sngs10004.yaml"
INPUT = REPO / ".runtime/local_takeover/g10/sngs10004_step1/states/sn-gamestate.pklz"
RUN_DIR = REPO / ".runtime/local_takeover/g10/sngs10004_enrichment"
CACHE = REPO / ".runtime/local_takeover/g10/sngs10004_enrichment_cache"
RUNTIME_MANIFEST = REPO / ".runtime/local_takeover/g10/sngs10004_enrichment_manifest.json"
REPORT = REPO / "reports/local_takeover/20260819_soccerfactory_enrichment_sngs10004"
RESULT = REPORT / "result.json"
EVENTS = REPORT / "events.jsonl"
LOG = REPORT / "run.log"
PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
WORKER = REPO / "reproduction/gates/g10_prerefiner_enrichment_run2_worker.py"
QWEN = REPO / ".local_assets/models/soccerfactory/pretrained_models/jn/Qwen2.5-VL-7B-Instruct"
VENDOR = [REPO / "vendor/soccerfactory/tracklab", REPO / "vendor/soccerfactory/sn-gamestate", REPO]
APPROVAL = "G10_LOCAL_ENRICHMENT_GPU_APPROVED"
TIMEOUT_SECONDS = 28800


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


def build_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    text = CONFIG.read_text(encoding="utf-8")
    if "/remote-home" in text:
        raise AssertionError("Local enrichment config contains /remote-home")
    config = yaml.safe_load(text)
    if config["state"]["load_file"] != str(INPUT) or config["state"]["save_file"] != "states/sn-gamestate.pklz":
        raise AssertionError("Local enrichment state paths changed")
    if config["eval_tracking"] is not False or config["visualization"] is not None:
        raise AssertionError("Enrichment must remain inference-only")
    detections, images, members = read_state(INPUT)
    if len(images) != 255 or len(detections) != 3390 or detections["track_id"].nunique() != 48:
        raise AssertionError("Local Step 1 state identity changed")

    manifest = copy.deepcopy(json.loads(BASE.read_text(encoding="utf-8")))
    manifest["python"] = str(PYTHON)
    manifest["source_files"]["tracklab_main"] = str(
        REPO / "vendor/soccerfactory/tracklab/tracklab/main.py"
    )
    manifest["source_files"]["worker"] = str(WORKER)
    manifest["config"].update({
        "directory": str(CONFIG.parent), "name": CONFIG.stem, "path": str(CONFIG),
    })
    manifest["sample"].update({
        "detection_rows": 3390, "unique_track_ids": 48,
        "frame_root": str(REPO / ".local_assets/data/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1"),
    })
    manifest["input_state"].update({"path": str(INPUT), "bytes": INPUT.stat().st_size})
    local_models = REPO / ".local_assets/models/soccerfactory/pretrained_models"
    manifest["modules"]["pitch"]["weight_reads"] = [
        {"path": str(local_models / "calibration/SV_kp"), "bytes": 264964645},
        {"path": str(local_models / "calibration/SV_lines"), "bytes": 264857893},
    ]
    manifest["modules"]["legibility"]["weight_reads"] = [{
        "path": str(local_models / "legibility/legibility_resnet34_soccer_20240215.pth"),
        "bytes": 85289629,
    }]
    qwen = manifest["qwen_assets"]["qwen2_5vl_7b"]
    qwen.update({"configured_path": str(QWEN), "configured_path_is_symlink": False})
    qwen.pop("link_target", None)
    manifest["adapters"]["local_cache_dir"] = str(CACHE)
    manifest["approval"]["required_environment"] = f"{APPROVAL}=YES"
    manifest["outputs"].update({
        "hydra_run_dir": str(RUN_DIR),
        "state_archive": str(RUN_DIR / "states/sn-gamestate.pklz"),
        "report_dir": str(REPORT), "events": str(EVENTS), "log": str(LOG), "result": str(RESULT),
    })
    manifest["success"].update({"required_detection_rows": 3390, "required_unique_track_ids": 48})
    manifest["execution_semantics"]["sampling"] = "all 255 ordered frames and all 3390 local Step-1 detections"

    for module in manifest["modules"].values():
        for spec in module["weight_reads"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Local weight mismatch: {path}")
    total = 0
    for shard in qwen["shards"]:
        path = QWEN / shard["name"]
        if not path.is_file() or path.stat().st_size != shard["bytes"]:
            raise AssertionError(f"Qwen shard mismatch: {path}")
        total += path.stat().st_size
    if total != qwen["logical_shard_bytes"]:
        raise AssertionError("Qwen shard total changed")
    for path in (REPORT, RUN_DIR, CACHE, RUNTIME_MANIFEST):
        fresh(path)
    checked = {
        "input_state": str(INPUT), "image_rows": len(images), "detection_rows": len(detections),
        "unique_track_ids": int(detections["track_id"].nunique()), "input_members": members,
        "pipeline": config["pipeline"], "local_weight_files": 8,
        "qwen_shard_bytes": total, "remote_runtime_paths": False, "future_outputs_fresh": True,
    }
    return manifest, checked


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
    detections, images, members = read_state(RUN_DIR / "states/sn-gamestate.pklz")
    fields = ["bbox_pitch", "role", "team", "jersey_number"]
    required = {"bbox_ltwh", "track_id", *fields}
    if not required.issubset(detections.columns) or "parameters" not in images.columns:
        raise AssertionError("Enrichment output fields are incomplete")
    if len(images) != 255 or len(detections) != 3390 or detections["track_id"].nunique() != 48:
        raise AssertionError("Enrichment changed the row/track identity")
    return {
        "path": str(RUN_DIR / "states/sn-gamestate.pklz"),
        "bytes": (RUN_DIR / "states/sn-gamestate.pklz").stat().st_size,
        "members": members, "image_rows": len(images), "detection_rows": len(detections),
        "unique_track_ids": int(detections["track_id"].nunique()),
        "non_null_fields": {field: int(detections[field].notna().sum()) for field in fields},
        "valid_camera_rows": int(images["parameters"].notna().sum()),
    }


def run(manifest: dict[str, Any], checked: dict[str, Any]) -> int:
    if os.environ.get(APPROVAL) != "YES":
        raise PermissionError(f"Missing approval guard: {APPROVAL}=YES")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise PermissionError("Exactly one CUDA_VISIBLE_DEVICES index is required")
    REPORT.mkdir(parents=True)
    CACHE.mkdir(parents=True)
    EVENTS.touch()
    RUNTIME_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    command = [
        str(PYTHON), str(WORKER), "--manifest", str(RUNTIME_MANIFEST), "--events", str(EVENTS),
        "--config-dir", str(CONFIG.parent), "--config-name", CONFIG.stem,
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": ":".join(str(path) for path in VENDOR),
        "HF_HOME": str(CACHE / "hf"), "HUGGINGFACE_HUB_CACHE": str(CACHE / "hf/hub"),
        "TRANSFORMERS_CACHE": str(CACHE / "hf/transformers"), "XDG_CACHE_HOME": str(CACHE / "xdg"),
        "MPLCONFIGDIR": str(CACHE / "matplotlib"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "WANDB_DISABLED": "true", "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false", "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    started = time.monotonic()
    timed_out = False
    with LOG.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_visible_devices": visible}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
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
                lines = EVENTS.read_text(encoding="utf-8").splitlines()
                phase = json.loads(lines[-1])["phase"] if lines else "worker_boot"
                print(f"heartbeat phase={phase} elapsed_seconds={now-started:.1f}", flush=True)
                next_heartbeat += 30
            time.sleep(1)
        exit_code = process.wait()
    result: dict[str, Any] = {
        "status": "failed", "process_exit_code": exit_code, "timed_out": timed_out,
        "wall_seconds": round(time.monotonic() - started, 3), "cuda_visible_devices": visible,
        "training": False, "evaluation": False, "preflight": checked,
        "log": str(LOG), "events": str(EVENTS),
    }
    final_exit = exit_code or (1 if timed_out else 0)
    if final_exit == 0:
        try:
            result["state"] = inspect_output()
            records = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines()]
            worker = [r for r in records if r["phase"] == "worker" and r["status"] == "passed"]
            if len(worker) != 1:
                raise AssertionError("Worker completion event missing")
            result["gpu_memory"] = worker[0]["output"]
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
    args = parser.parse_args()
    manifest, checked = build_manifest()
    if not args.run:
        print(json.dumps({"status": "preflight_passed", **checked}, indent=2))
        return 0
    return run(manifest, checked)


if __name__ == "__main__":
    raise SystemExit(main())
