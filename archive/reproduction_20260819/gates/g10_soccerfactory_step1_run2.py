#!/usr/bin/env python3
"""Static preflight and guarded launcher for G10-B Step 1 isolated runs.

The default mode only parses local text and stats declared assets. It never
imports TrackLab/torch, opens model weights, or touches CUDA. Run mode is kept
behind a fresh explicit approval environment guard.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
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
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_soccerfactory_step1_run2_sngs10004.json"
RUN1_CONFIG = REPO / "reproduction/configs/g10/g10_step1_sngs10004.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Local path escapes repository: {path}")
    return resolved


def require_absent(path: Path, label: str) -> None:
    inside_repo(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} is already used: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_new(text: str, path: Path) -> None:
    inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage"))
    allowed = {
        (1, "G10-B", "tracklab_step1_run2_prepared"),
        (1, "G10-B", "tracklab_step1_run3_prepared"),
        (1, "G10-B", "tracklab_step1_run4_prepared"),
        (1, "G10-B", "tracklab_step1_run5_prepared"),
    }
    if identity not in allowed:
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    return manifest


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["experiment_subname"] = "RUN_ID"
    value["hydra"]["run"]["dir"] = "RUN_DIR"
    value["hydra"]["sweep"]["dir"] = "SWEEP_DIR"
    return value


def validate_config(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["config"]["path"])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    run1 = yaml.safe_load(RUN1_CONFIG.read_text(encoding="utf-8"))
    if normalize_config(config) != normalize_config(run1):
        raise AssertionError("isolated run differs from run1 beyond immutable run identity")
    sample = manifest["sample"]
    if config["pipeline"] != manifest["config"]["pipeline"]:
        raise AssertionError("Pipeline changed")
    if config["dataset"]["nframes"] != sample["frame_count"]:
        raise AssertionError("nframes changed")
    if config["dataset"]["eval_set"] != sample["split"]:
        raise AssertionError("eval_set changed")
    if config["dataset"]["vids_dict"] != {sample["split"]: [sample["sequence"]]}:
        raise AssertionError("Sequence selection changed")
    if config["eval_tracking"] is not False or config["visualization"] is not None:
        raise AssertionError("Evaluation/visualization must remain disabled")
    if config["num_cores"] != manifest["adapters"]["dataset_pool_workers"]:
        raise AssertionError("Declared pool adaptation changed")
    if Path(config["hydra"]["run"]["dir"]) != Path(manifest["outputs"]["hydra_run_dir"]):
        raise AssertionError("Hydra output changed")
    detector = config["modules"]["bbox_detector"]["cfg"]["path_to_checkpoint"]
    declared = manifest["models"]["bbox_detector"]["weight_reads"][0]["path"]
    if detector != declared:
        raise AssertionError("Detector weight changed")
    return {
        "path": str(path),
        "semantically_equal_to_run1": True,
        "allowed_identity_differences": ["experiment_subname", "hydra.run.dir", "hydra.sweep.dir"],
        "pipeline": config["pipeline"],
        "nframes": config["dataset"]["nframes"],
        "eval_tracking": False,
        "visualization": None,
    }


def validate_frames(sample: dict[str, Any]) -> dict[str, Any]:
    root = Path(sample["frame_root"])
    names = sorted(path.name for path in root.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, sample["frame_count"] + 1)]
    if names != expected:
        raise AssertionError("Frame names/count changed")
    total = sum((root / name).stat().st_size for name in names)
    if total != sample["frame_total_bytes"]:
        raise AssertionError("Frame byte total changed")
    for key in ("first_frame", "last_frame"):
        spec = sample[key]
        if (root / spec["name"]).stat().st_size != spec["bytes"]:
            raise AssertionError(f"{key} size changed")
    return {"count": len(names), "total_bytes": total, "contents_read": False}


def validate_sources_and_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, text in manifest["source_files"].items():
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(path)
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sources[name] = {"path": str(path), "python_syntax": "valid"}
    assets: dict[str, Any] = {}
    for component, model in manifest["models"].items():
        reads = []
        for spec in model["weight_reads"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Weight stat changed: {path}")
            reads.append({"path": str(path), "bytes": path.stat().st_size, "contents_read": False})
        assets[component] = {"target": model["target"], "weight_reads": reads}
    return {"sources": sources, "declared_model_reads": assets}


def static_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise AssertionError(f"Wrong Python: {sys.executable}")
    base = manifest["base_step1_manifest"]
    if sha256_file(Path(base["path"])) != base["sha256"]:
        raise AssertionError("Base Step 1 manifest changed")
    expected_phases = [
        "import_tracklab_main", "hydra_cli_compose", "init_environment",
        "instantiate_dataset", "instantiate_evaluator", "instantiate_bbox_detector",
        "instantiate_reid", "instantiate_track", "build_pipeline",
        "build_tracker_state", "instantiate_engine", "track_dataset",
        "evaluation_skipped",
    ]
    if manifest["phases"] != expected_phases:
        raise AssertionError("Phase order changed")
    for phase in expected_phases:
        if phase not in manifest["timeouts_seconds"]:
            raise AssertionError(f"Missing phase timeout: {phase}")
    adapters = manifest["adapters"]
    if adapters["dataset_pool_workers"] != 4:
        raise AssertionError("Dataset worker cap changed")
    if adapters["suppressed_path_exact"] != "/home/tianlin/.cache/mim":
        raise AssertionError("Cache suppression scope widened")
    inside_repo(Path(adapters["local_cache_dir"]))
    outputs = manifest["outputs"]
    require_absent(Path(outputs["hydra_run_dir"]), "future Hydra run")
    require_absent(Path(outputs["report_dir"]), "future run report")
    require_absent(Path(adapters["local_cache_dir"]), "future run cache")
    if Path(outputs["state_archive"]) != Path(outputs["hydra_run_dir"]) / "states/sn-gamestate.pklz":
        raise AssertionError("State archive contract changed")
    cuda_guard = manifest.get("cuda_guard")
    if manifest["stage"] in {
        "tracklab_step1_run3_prepared",
        "tracklab_step1_run4_prepared",
        "tracklab_step1_run5_prepared",
    }:
        expected_guard = {
            "cuda_available": True,
            "visible_device_count": 1,
            "logical_device_index": 0,
            "expected_device_name": "NVIDIA H800",
            "minimum_total_memory_bytes": 80000000000,
            "minimum_free_memory_bytes": 70000000000,
        }
        if cuda_guard != expected_guard:
            raise AssertionError(f"CUDA guard changed: {cuda_guard}")
        run_number = manifest["stage"].removeprefix("tracklab_step1_run").removesuffix("_prepared")
        expected_approval = f"G10_STEP1_RUN{run_number}_GPU_APPROVED=YES"
        if manifest["approval"]["required_environment"] != expected_approval:
            raise AssertionError(f"run{run_number} approval guard changed")
    null_filter = adapters.get("engine_null_callback_filter")
    if manifest["stage"] == "tracklab_step1_run4_prepared":
        expected_filter = {
            "enabled": True,
            "copy_before_filter": True,
            "expected_original_keys": ["ignored_regions", "progress", "vis"],
            "expected_null_keys": ["vis"],
            "expected_remaining_keys": ["ignored_regions", "progress"],
            "scope": "instantiate_engine_only",
        }
        if null_filter != expected_filter:
            raise AssertionError(f"run4 null callback filter changed: {null_filter}")
    if manifest["stage"] == "tracklab_step1_run5_prepared":
        expected_filter = {
            "enabled": True,
            "copy_before_filter": True,
            "deletion_mode": "omegaconf_open_dict",
            "require_struct_before": True,
            "require_struct_restored": True,
            "expected_original_keys": ["ignored_regions", "progress", "vis"],
            "expected_null_keys": ["vis"],
            "expected_remaining_keys": ["ignored_regions", "progress"],
            "scope": "instantiate_engine_only",
        }
        if null_filter != expected_filter:
            raise AssertionError(f"run5 null callback filter changed: {null_filter}")
    return {
        "python": manifest["python"],
        "base_step1_manifest_sha256": base["sha256"],
        "config": validate_config(manifest),
        "sample": validate_frames(manifest["sample"]),
        **validate_sources_and_assets(manifest),
        "phases": expected_phases,
        "phase_timeouts_seconds": manifest["timeouts_seconds"],
        "future_outputs_unused": True,
        "tracklab_or_model_imports": False,
        "weight_contents_read": False,
        "gpu_operations": [],
        "future_cuda_guard": cuda_guard,
        "future_engine_null_callback_filter": null_filter,
    }


def require_columns(frame: Any, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing {label} columns: {missing}")


def validate_state(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["outputs"]["state_archive"])
    success = manifest["success"]
    video_id = manifest["sample"]["video_id"]
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if sorted(members) != sorted(success["zip_members_exact"]) or archive.testzip() is not None:
            raise AssertionError("State ZIP member/CRC contract failed")
        with archive.open(f"{video_id}.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open(f"{video_id}_image.pkl") as handle:
            images = pickle.load(handle)
    require_columns(detections, success["required_detection_columns"], "detection")
    require_columns(images, success["required_image_columns"], "image")
    if len(images) != success["required_image_rows"] or len(detections) < success["minimum_detection_rows"]:
        raise AssertionError("State row-count contract failed")
    if images["frame"].astype(int).tolist() != list(range(success["required_image_rows"])):
        raise AssertionError("State frames are not 0..254")
    if detections["track_id"].isna().any():
        raise AssertionError("Some detections have no track id")
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "members": members, "image_rows": len(images), "detection_rows": len(detections),
        "unique_track_ids": int(detections["track_id"].nunique()),
    }


def terminate(process: subprocess.Popen[Any], grace: int) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def latest_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else None


def validate_events(path: Path, phases: list[str]) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(record["status"] == "failed" for record in records):
        raise AssertionError("Worker event stream contains a failed phase")
    observed = [(record["phase"], record["status"]) for record in records if record["phase"] in phases]
    expected = [(phase, status) for phase in phases for status in ("started", "passed")]
    if observed != expected:
        raise AssertionError(f"Worker phase event order changed: {observed}")
    return {
        "path": str(path),
        "record_count": len(records),
        "phase_order": phases,
        "all_required_phases_started_and_passed": True,
    }


def run_worker(
    manifest: dict[str, Any], manifest_path: Path, preflight: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    approval_name, separator, approval_value = manifest["approval"]["required_environment"].partition("=")
    if separator != "=" or not approval_name or os.environ.get(approval_name) != approval_value:
        raise PermissionError(f"Run mode requires {manifest['approval']['required_environment']}")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", cuda) is None:
        raise PermissionError("CUDA_VISIBLE_DEVICES must name exactly one numeric device")
    outputs = manifest["outputs"]
    report_dir = Path(outputs["report_dir"])
    require_absent(report_dir, "run report")
    require_absent(Path(outputs["hydra_run_dir"]), "Hydra run")
    require_absent(Path(manifest["adapters"]["local_cache_dir"]), "run cache")
    report_dir.mkdir(parents=True, exist_ok=False)
    cache = Path(manifest["adapters"]["local_cache_dir"])
    cache.mkdir(parents=True, exist_ok=False)
    events = Path(outputs["events"])
    events.touch(exist_ok=False)
    command = [
        manifest["python"], manifest["source_files"]["worker"],
        "--manifest", str(manifest_path), "--events", str(events),
        "--config-dir", manifest["config"]["directory"],
        "--config-name", manifest["config"]["name"],
    ]
    environment = os.environ.copy()
    environment.update({
        "MPLCONFIGDIR": str(cache / "matplotlib"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled", "HYDRA_FULL_ERROR": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    timeouts = manifest["timeouts_seconds"]
    started_wall = time.time()
    started = time.monotonic()
    timed_out = False
    timeout_phase = None
    phase_seen_at: dict[str, float] = {}
    heartbeat = int(manifest["heartbeat_seconds"])
    with Path(outputs["log"]).open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_visible_devices": cuda}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        next_heartbeat = started + heartbeat
        while process.poll() is None:
            now = time.monotonic()
            event = latest_event(events)
            active = "worker_boot" if event is None else event["phase"]
            if event is not None and event["status"] == "passed":
                active = "worker_transition"
            phase_seen_at.setdefault(active, now)
            allowed = int(timeouts.get(active, timeouts["worker_transition"]))
            if now - started >= timeouts["overall"] or now - phase_seen_at[active] >= allowed:
                timed_out = True
                timeout_phase = active
                terminate(process, int(timeouts["termination_grace"]))
                break
            if now >= next_heartbeat:
                print(f"heartbeat phase={active} elapsed_seconds={now-started:.1f} pid={process.pid}", flush=True)
                next_heartbeat += heartbeat
            time.sleep(1)
        exit_code = process.wait()
    result: dict[str, Any] = {
        "schema_version": 1, "gate": "G10-B",
        "stage": manifest["stage"].removesuffix("_prepared"),
        "started_unix": started_wall, "ended_unix": time.time(),
        "wall_seconds": time.monotonic() - started, "process_exit_code": exit_code,
        "timed_out": timed_out, "timeout_phase": timeout_phase, "command": command,
        "cuda_visible_devices": cuda, "events": str(events), "log": outputs["log"],
        "preflight": preflight, "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "fallbacks_used": [], "evaluation_started": False, "training_started": False,
    }
    final_exit = exit_code if exit_code else 0
    if not timed_out and exit_code == 0:
        try:
            result["event_contract"] = validate_events(events, manifest["phases"])
            result["state_archive"] = validate_state(manifest)
            result.update({"outcome": "passed", "assertions_passed": True})
        except Exception as error:
            result.update({"outcome": "failed", "assertions_passed": False,
                           "failure_category": "artifact_integrity", "error": f"{type(error).__name__}: {error}"})
            final_exit = 1
    else:
        result.update({"outcome": "failed", "assertions_passed": False,
                       "failure_category": "phase_timeout" if timed_out else "worker_failure"})
        final_exit = final_exit or 1
    atomic_json(result, Path(outputs["result"]))
    return result, final_exit


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    started = time.time()
    print("heartbeat phase=static_preflight status=started", flush=True)
    preflight = static_preflight(manifest)
    print("heartbeat phase=static_preflight status=passed", flush=True)
    if args.mode == "run":
        result, exit_code = run_worker(manifest, args.manifest, preflight)
        print(json.dumps({"outcome": result["outcome"], "result": manifest["outputs"]["result"]}), flush=True)
        return exit_code
    result_path = Path(manifest["outputs"]["preflight_result"])
    log_path = Path(manifest["outputs"]["preflight_log"])
    require_absent(result_path, "preflight result")
    require_absent(log_path, "preflight log")
    result = {
        "schema_version": 1, "gate": "G10-B",
        "stage": f"{manifest['stage'].removesuffix('_prepared')}_preflight",
        "outcome": "passed", "assertions_passed": True, "started_unix": started,
        "ended_unix": time.time(), "wall_seconds": time.time() - started,
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "git": git_identity(), "preflight": preflight, "gpu_operations": [],
        "tracklab_or_model_imports": False, "weight_contents_read": False,
        "inference_started": False, "evaluation_started": False, "training_started": False,
        "next_step_requires_fresh_gpu_approval": True,
    }
    atomic_json(result, result_path)
    write_new(
        "heartbeat phase=static_preflight status=started\n"
        "heartbeat phase=static_preflight status=passed\nexit_code=0\n",
        log_path,
    )
    print(json.dumps({"outcome": "passed", "result": str(result_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
