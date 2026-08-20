#!/usr/bin/env python3
"""Preflight and guarded launcher for G10-B TrackLab Step 1.

The default mode is static preflight. It parses local YAML and validates paths
without importing TrackLab, torch, or model code. ``--mode run`` is deliberately
guarded by explicit GPU approval and a single CUDA_VISIBLE_DEVICES value.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import resource
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_soccerfactory_step1_sngs10004.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Local output escapes repository: {path}")
    return resolved


def require_absent(path: Path, label: str) -> None:
    require_inside_repo(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} must be unused; refusing overwrite: {path}")


def atomic_json_dump(value: Any, path: Path) -> None:
    require_inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_new_text(text: str, path: Path) -> None:
    require_inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = (
        manifest.get("schema_version"),
        manifest.get("gate"),
        manifest.get("stage"),
    )
    if identity != (1, "G10-B", "tracklab_step1_preflight"):
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    return manifest


def validate_file_size(spec: dict[str, Any], label: str) -> None:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(spec["bytes"]):
        raise AssertionError(f"{label} size changed: {path}")


def validate_frames(sample: dict[str, Any]) -> dict[str, Any]:
    root = Path(sample["frame_root"])
    count = int(sample["frame_count"])
    names = sorted(path.name for path in root.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, count + 1)]
    if names != expected:
        raise AssertionError("Prepared frames are not exactly 000001.jpg..000255.jpg")
    paths = [root / name for name in names]
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes != int(sample["frame_total_bytes"]):
        raise AssertionError("Prepared-frame total byte size changed")
    for endpoint_name in ("first_frame", "last_frame"):
        spec = sample[endpoint_name]
        path = root / spec["name"]
        if path.stat().st_size != int(spec["bytes"]):
            raise AssertionError(f"Prepared-frame endpoint size changed: {path}")
        if sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"Prepared-frame endpoint SHA256 changed: {path}")
    return {"count": len(paths), "total_bytes": total_bytes}


def validate_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(manifest["config"]["path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Local config is not a YAML mapping")
    if config.get("pipeline") != manifest["config"]["pipeline"]:
        raise AssertionError("Pipeline changed")
    if config.get("eval_tracking") is not False or config.get("test_tracking") is not True:
        raise AssertionError("Step 1 must track but must not evaluate")
    if config.get("visualization") is not None:
        raise AssertionError("Step 1 visualization must be disabled")
    dataset = config["dataset"]
    expected_sample = manifest["sample"]
    if dataset.get("nframes") != expected_sample["frame_count"]:
        raise AssertionError("nframes changed")
    if dataset.get("eval_set") != expected_sample["split"]:
        raise AssertionError("eval_set changed")
    if dataset.get("vids_dict") != {"sn500": [expected_sample["sequence"]]}:
        raise AssertionError("vids_dict changed")
    if Path(dataset["dataset_path"]) / "sn500" / expected_sample["sequence"] / "img1" != Path(expected_sample["frame_root"]):
        raise AssertionError("Dataset path no longer resolves to fixed frames")
    if config["state"] != {
        "save_file": "states/sn-gamestate.pklz",
        "load_file": None,
        "compression": 0,
    }:
        raise AssertionError("Tracker-state policy changed")
    if Path(config["hydra"]["run"]["dir"]) != Path(manifest["local_outputs"]["hydra_run_dir"]):
        raise AssertionError("Hydra run directory changed")
    detector_checkpoint = config["modules"]["bbox_detector"]["cfg"]["path_to_checkpoint"]
    if detector_checkpoint != manifest["models"]["yolo_person"]["path"]:
        raise AssertionError("Detector checkpoint changed")
    if config.get("use_wandb") is not False:
        raise AssertionError("W&B must remain disabled")
    if config.get("num_cores") != 4:
        raise AssertionError("num_cores changed")
    return {
        "path": str(config_path),
        "pipeline": config["pipeline"],
        "nframes": dataset["nframes"],
        "eval_tracking": config["eval_tracking"],
        "visualization": config["visualization"],
        "hydra_run_dir": config["hydra"]["run"]["dir"],
    }


def validate_import_locations() -> dict[str, str]:
    expected_roots = {
        "tracklab": "/remote-home/haolinyang/sports/soccernet/tracklab",
        "sn_gamestate": "/remote-home/haolinyang/sports/soccernet/sn-gamestate",
        "torch": "/remote-home/haolinyang/anaconda3/envs/tracklab2",
        "ultralytics": "/remote-home/haolinyang/anaconda3/envs/tracklab2",
        "hydra": "/remote-home/haolinyang/anaconda3/envs/tracklab2",
        "pandas": "/remote-home/haolinyang/anaconda3/envs/tracklab2",
    }
    origins: dict[str, str] = {}
    for package, expected_root in expected_roots.items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(package)
        origins[package] = spec.origin
        if not str(Path(spec.origin).resolve()).startswith(expected_root):
            raise AssertionError(f"Unexpected {package} origin: {spec.origin}")
    return origins


def static_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    python = Path(manifest["python"])
    if Path(sys.executable).resolve() != python.resolve():
        raise AssertionError(f"Wrong Python: {sys.executable}")

    base = manifest["base_manifest"]
    base_path = Path(base["path"])
    if sha256_file(base_path) != base["sha256"]:
        raise AssertionError("Base G10-B manifest changed")

    source_reference = Path(manifest["config"]["source_reference"])
    if not source_reference.is_file() or source_reference.stat().st_size != manifest["config"]["source_reference_bytes"]:
        raise AssertionError("Source debug Step 1 config changed or is missing")
    for path_text in manifest["source_contract_files"]:
        if not Path(path_text).is_file():
            raise FileNotFoundError(path_text)
    for label, spec in manifest["models"].items():
        validate_file_size(spec, label)

    frames = validate_frames(manifest["sample"])
    config = validate_config(manifest)
    imports = validate_import_locations()

    command = manifest["launcher"]["command"]
    expected_command = [
        manifest["python"],
        "-m",
        "tracklab.main",
        "--config-dir",
        str(Path(manifest["config"]["path"]).parent),
        "--config-name",
        manifest["config"]["name"],
    ]
    if command != expected_command:
        raise AssertionError("Launcher command changed")
    if Path(manifest["launcher"]["working_directory"]) != REPO:
        raise AssertionError("Launcher working directory changed")
    if manifest["launcher"]["timeout_seconds"] <= 0 or manifest["launcher"]["heartbeat_seconds"] <= 0:
        raise AssertionError("Invalid timeout or heartbeat")

    outputs = manifest["local_outputs"]
    require_absent(Path(outputs["hydra_run_dir"]), "Hydra output directory")
    require_absent(Path(outputs["run_report_dir"]), "future run report directory")
    if Path(outputs["state_archive"]) != Path(outputs["hydra_run_dir"]) / "states/sn-gamestate.pklz":
        raise AssertionError("State archive path changed")
    expected_members = sorted([
        f"{manifest['sample']['video_id']}.pkl",
        f"{manifest['sample']['video_id']}_image.pkl",
        "summary.json",
    ])
    if sorted(manifest["success"]["zip_members_exact"]) != expected_members:
        raise AssertionError("Expected ZIP member contract changed")

    return {
        "python": str(python),
        "base_manifest_sha256": base["sha256"],
        "source_contract_file_count": len(manifest["source_contract_files"]),
        "models": {
            label: {"path": spec["path"], "bytes": spec["bytes"]}
            for label, spec in manifest["models"].items()
        },
        "frames": frames,
        "config": config,
        "package_origins_without_import": imports,
        "command": command,
        "timeout_seconds": manifest["launcher"]["timeout_seconds"],
        "heartbeat_seconds": manifest["launcher"]["heartbeat_seconds"],
        "output_paths_unused": True,
    }


def require_columns(frame: Any, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing {label} columns: {missing}")


def validate_state_archive(manifest: dict[str, Any]) -> dict[str, Any]:
    archive_path = Path(manifest["local_outputs"]["state_archive"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing TrackLab state: {archive_path}")
    video_id = manifest["sample"]["video_id"]
    success = manifest["success"]
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise AssertionError("State ZIP contains duplicate members")
        if sorted(members) != sorted(success["zip_members_exact"]):
            raise AssertionError(f"Unexpected state ZIP members: {members}")
        if archive.testzip() is not None:
            raise AssertionError("State ZIP CRC validation failed")
        with archive.open("summary.json") as handle:
            summary = json.load(handle)
        with archive.open(f"{video_id}.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open(f"{video_id}_image.pkl") as handle:
            images = pickle.load(handle)

    require_columns(detections, success["required_detection_columns"], "detection")
    require_columns(images, success["required_image_columns"], "image")
    if len(images) != success["required_image_rows"]:
        raise AssertionError(f"Expected 255 image rows, found {len(images)}")
    if images["frame"].astype(int).tolist() != list(range(success["required_image_rows"])):
        raise AssertionError("Image frames are not exactly 0..254")
    if set(images["video_id"].astype(str)) != {video_id}:
        raise AssertionError("Image video_id mismatch")
    expected_names = [f"{index:06d}.jpg" for index in range(1, 256)]
    actual_names = [Path(path).name for path in images["file_path"].tolist()]
    if actual_names != expected_names:
        raise AssertionError("Image file paths are not exactly 000001.jpg..000255.jpg")
    if len(detections) < success["minimum_detection_rows"]:
        raise AssertionError("No person detections were saved")
    if set(detections["video_id"].astype(str)) != {video_id}:
        raise AssertionError("Detection video_id mismatch")
    if not set(detections["image_id"]).issubset(set(images["id"])):
        raise AssertionError("Detection image_id is outside image table")
    if detections["track_id"].isna().any():
        raise AssertionError("Some detections have no track_id")
    summary_columns = summary.get("columns", {})
    for level, required in (
        ("detection", success["required_detection_columns"]),
        ("image", success["required_image_columns"]),
    ):
        if not set(required).issubset(set(summary_columns.get(level, []))):
            raise AssertionError(f"summary.json is missing {level} columns")
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "members": members,
        "image_rows": len(images),
        "detection_rows": len(detections),
        "unique_track_ids": int(detections["track_id"].nunique()),
    }


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_tracklab(manifest: dict[str, Any], preflight: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if os.environ.get("G10_STEP1_GPU_APPROVED") != "YES":
        raise PermissionError("Run mode requires G10_STEP1_GPU_APPROVED=YES")
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", cuda_devices) is None:
        raise PermissionError("CUDA_VISIBLE_DEVICES must name exactly one numeric device")

    outputs = manifest["local_outputs"]
    run_report_dir = Path(outputs["run_report_dir"])
    require_absent(run_report_dir, "run report directory")
    require_absent(Path(outputs["hydra_run_dir"]), "Hydra output directory")
    run_report_dir.mkdir(parents=True, exist_ok=False)
    log_path = Path(outputs["run_log"])
    result_path = Path(outputs["run_result"])
    command = manifest["launcher"]["command"]
    timeout_seconds = int(manifest["launcher"]["timeout_seconds"])
    heartbeat_seconds = int(manifest["launcher"]["heartbeat_seconds"])

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "HYDRA_FULL_ERROR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    started_wall = time.time()
    started_monotonic = time.monotonic()
    timed_out = False
    with log_path.open("x", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps({"command": command, "cuda_visible_devices": cuda_devices}) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=manifest["launcher"]["working_directory"],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        next_heartbeat = started_monotonic + heartbeat_seconds
        while process.poll() is None:
            now = time.monotonic()
            if now - started_monotonic >= timeout_seconds:
                timed_out = True
                print(f"heartbeat phase=timeout elapsed_seconds={now - started_monotonic:.1f}", flush=True)
                terminate_process_group(process)
                break
            if now >= next_heartbeat:
                print(
                    f"heartbeat phase=tracklab_step1 elapsed_seconds={now - started_monotonic:.1f} pid={process.pid}",
                    flush=True,
                )
                next_heartbeat += heartbeat_seconds
            time.sleep(1)
        exit_code = process.wait()

    ended_wall = time.time()
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": "tracklab_step1_run",
        "started_unix": started_wall,
        "ended_unix": ended_wall,
        "wall_seconds": ended_wall - started_wall,
        "command": command,
        "cuda_visible_devices": cuda_devices,
        "timeout_seconds": timeout_seconds,
        "heartbeat_seconds": heartbeat_seconds,
        "timed_out": timed_out,
        "process_exit_code": exit_code,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "preflight": preflight,
        "git": git_identity(),
        "log": str(log_path),
        "fallbacks_used": [],
        "not_validated": ["tracking quality", "evaluation", "training"],
    }
    final_exit = exit_code
    if not timed_out and exit_code == 0:
        try:
            result["state_archive"] = validate_state_archive(manifest)
            result["assertions_passed"] = True
            result["outcome"] = "passed"
            final_exit = 0
        except Exception as error:  # Preserve a machine-readable failed run.
            result["assertions_passed"] = False
            result["outcome"] = "failed"
            result["failure_category"] = "artifact_write_or_integrity"
            result["error"] = f"{type(error).__name__}: {error}"
            final_exit = 1
    else:
        result["assertions_passed"] = False
        result["outcome"] = "failed"
        result["failure_category"] = "distributed_order_or_timeout" if timed_out else "forward_or_numerical"
        final_exit = exit_code if exit_code != 0 else 1
    atomic_json_dump(result, result_path)
    return result, final_exit


def main() -> int:
    args = parse_args()
    started = time.time()
    manifest = load_manifest(args.manifest)
    print("heartbeat phase=static_preflight status=started", flush=True)
    preflight = static_preflight(manifest)
    print("heartbeat phase=static_preflight status=passed", flush=True)

    if args.mode == "run":
        result, exit_code = run_tracklab(manifest, preflight)
        print(json.dumps({"outcome": result["outcome"], "result": manifest["local_outputs"]["run_result"]}), flush=True)
        return exit_code

    result_path = Path(manifest["local_outputs"]["preflight_result"])
    if result_path.exists() or result_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite preflight result: {result_path}")
    log_path = Path(manifest["local_outputs"]["preflight_log"])
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite preflight log: {log_path}")
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": "tracklab_step1_preflight",
        "outcome": "passed",
        "assertions_passed": True,
        "started_unix": started,
        "ended_unix": time.time(),
        "wall_seconds": time.time() - started,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "git": git_identity(),
        "preflight": preflight,
        "gpu_operations": [],
        "tracklab_imported_or_launched": False,
        "inference_started": False,
        "evaluation_started": False,
        "training_started": False,
        "not_validated": manifest["not_validated_by_preflight"],
    }
    atomic_json_dump(result, result_path)
    write_new_text(
        "heartbeat phase=static_preflight status=started\n"
        "heartbeat phase=static_preflight status=passed\n"
        "exit_code=0\n",
        log_path,
    )
    print(json.dumps({"outcome": "passed", "result": str(result_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
