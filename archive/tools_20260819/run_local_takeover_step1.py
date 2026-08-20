#!/usr/bin/env python3
"""Guarded local-source/local-asset Step 1 smoke with one log and result."""

from __future__ import annotations

import argparse
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
MANIFEST_PATH = REPO / "reproduction/manifests/g10_local_takeover_step1_sngs10004.json"
VENDOR_PATHS = [
    REPO / "vendor/soccerfactory/tracklab",
    REPO / "vendor/soccerfactory/sn-gamestate",
    REPO,
]
OVERALL_TIMEOUT = 7200
HEARTBEAT_SECONDS = 30


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def require_fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(manifest["config"]["path"])
    config_text = config_path.read_text(encoding="utf-8")
    if "/remote-home" in config_text:
        raise AssertionError("Local Step 1 config still contains /remote-home")
    config = yaml.safe_load(config_text)
    if config["pipeline"] != manifest["config"]["pipeline"]:
        raise AssertionError("Pipeline mismatch")
    if config["dataset"]["nframes"] != 255 or config["eval_tracking"] is not False:
        raise AssertionError("Fixed inference-only sample contract changed")
    if config["hydra"].get("searchpath") != [
        "pkg://tracklab.configs", "pkg://sn_gamestate.configs"
    ]:
        raise AssertionError("Vendored Hydra search path is missing")

    for source in manifest["source_files"].values():
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    for model in manifest["models"].values():
        for spec in model["weight_reads"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Weight asset mismatch: {path}")

    frame_root = Path(manifest["sample"]["frame_root"])
    names = sorted(path.name for path in frame_root.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, 256)]
    if names != expected:
        raise AssertionError("Fixed frame set is not exactly 000001.jpg..000255.jpg")
    frame_bytes = sum((frame_root / name).stat().st_size for name in names)
    if frame_bytes != manifest["sample"]["frame_total_bytes"]:
        raise AssertionError("Fixed frame byte total changed")

    outputs = manifest["outputs"]
    for path in (
        Path(outputs["report_dir"]),
        Path(outputs["hydra_run_dir"]),
        Path(manifest["adapters"]["local_cache_dir"]),
    ):
        require_fresh(path)
    return {
        "config": str(config_path),
        "pipeline": config["pipeline"],
        "frames": len(names),
        "frame_bytes": frame_bytes,
        "model_weight_files": sum(len(m["weight_reads"]) for m in manifest["models"].values()),
        "remote_runtime_paths": False,
        "future_outputs_fresh": True,
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


def inspect_state(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["outputs"]["state_archive"])
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if sorted(members) != ["10004.pkl", "10004_image.pkl", "summary.json"]:
            raise AssertionError(f"Unexpected state members: {members}")
        if archive.testzip() is not None:
            raise AssertionError("State ZIP CRC failed")
        detections = pickle.load(archive.open("10004.pkl"))
        images = pickle.load(archive.open("10004_image.pkl"))
    required = {"image_id", "video_id", "bbox_ltwh", "bbox_conf", "track_id"}
    if not required.issubset(detections.columns) or len(detections) == 0:
        raise AssertionError("Detection output contract failed")
    if len(images) != 255 or images["frame"].astype(int).tolist() != list(range(255)):
        raise AssertionError("Image output contract failed")
    if detections["track_id"].isna().any():
        raise AssertionError("Some detections have no track_id")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "members": members,
        "image_rows": int(len(images)),
        "detection_rows": int(len(detections)),
        "unique_track_ids": int(detections["track_id"].nunique()),
    }


def run(manifest: dict[str, Any], checked: dict[str, Any]) -> int:
    approval_name, _, approval_value = manifest["approval"]["required_environment"].partition("=")
    if os.environ.get(approval_name) != approval_value:
        raise PermissionError(f"Missing approval guard: {approval_name}={approval_value}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise PermissionError("Exactly one CUDA_VISIBLE_DEVICES index is required")

    outputs = manifest["outputs"]
    report = Path(outputs["report_dir"])
    report.mkdir(parents=True)
    cache = Path(manifest["adapters"]["local_cache_dir"])
    cache.mkdir(parents=True)
    events = Path(outputs["events"])
    events.touch()
    command = [
        manifest["python"], manifest["source_files"]["worker"],
        "--manifest", str(MANIFEST_PATH), "--events", str(events),
        "--config-dir", manifest["config"]["directory"],
        "--config-name", manifest["config"]["name"],
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": ":".join(str(path) for path in VENDOR_PATHS),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false", "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    started = time.monotonic()
    timed_out = False
    with Path(outputs["log"]).open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_visible_devices": visible}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        next_heartbeat = started + HEARTBEAT_SECONDS
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= OVERALL_TIMEOUT:
                timed_out = True
                terminate(process)
                break
            if now >= next_heartbeat:
                phase = "worker_boot"
                lines = events.read_text(encoding="utf-8").splitlines()
                if lines:
                    phase = json.loads(lines[-1])["phase"]
                print(f"heartbeat phase={phase} elapsed_seconds={now-started:.1f}", flush=True)
                next_heartbeat += HEARTBEAT_SECONDS
            time.sleep(1)
        exit_code = process.wait()

    result: dict[str, Any] = {
        "status": "failed", "process_exit_code": exit_code, "timed_out": timed_out,
        "wall_seconds": round(time.monotonic() - started, 3),
        "cuda_visible_devices": visible, "training": False, "evaluation": False,
        "preflight": checked, "log": outputs["log"], "events": outputs["events"],
    }
    final_exit = exit_code or (1 if timed_out else 0)
    if final_exit == 0:
        try:
            result["state"] = inspect_state(manifest)
            records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            worker = [r for r in records if r["phase"] == "worker" and r["status"] == "passed"]
            if len(worker) != 1:
                raise AssertionError("Worker completion event missing")
            result["gpu_memory"] = worker[0]["output"]
            result["status"] = "passed"
        except Exception as error:
            result["artifact_error"] = f"{type(error).__name__}: {error}"
            final_exit = 1
    Path(outputs["result"]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "result": outputs["result"]}), flush=True)
    return final_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    checked = preflight(manifest)
    if not args.run:
        print(json.dumps({"status": "preflight_passed", **checked}, indent=2))
        return 0
    return run(manifest, checked)


if __name__ == "__main__":
    raise SystemExit(main())
