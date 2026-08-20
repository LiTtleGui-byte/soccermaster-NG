#!/usr/bin/env python3
"""CPU-only preflight and approval-guarded launcher for the G10 Refiner probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_refiner_probe_run1_sngs10004.json"


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


def inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Path escapes workspace: {path}")
    return resolved


def require_absent(path: Path, label: str) -> None:
    inside_repo(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} is already used: {path}")


def validate_file(spec: dict[str, Any], *, hash_contents: bool = True) -> Path:
    path = Path(spec["path"])
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise AssertionError(f"Pinned file stat changed: {path}")
    if hash_contents and sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"Pinned file hash changed: {path}")
    return path


def leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(child)
            else:
                differences.update(leaf_differences(left[key], right[key], child))
        return differences
    return set() if left == right else {prefix}


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


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def validate_environment(mode: str) -> dict[str, str | None]:
    actual = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }
    for key, expected in {
        "PYTHONPATH": "", "LD_LIBRARY_PATH": "", "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        if actual[key] != expected:
            raise AssertionError(f"Environment changed for {key}: {actual[key]}")
    cuda = actual["CUDA_VISIBLE_DEVICES"] or ""
    if mode == "preflight" and cuda != "":
        raise AssertionError("CPU-only preflight must hide all CUDA devices")
    if mode == "run" and re.fullmatch(r"[0-9]+", cuda) is None:
        raise AssertionError("Run mode requires exactly one numeric CUDA_VISIBLE_DEVICES")
    return actual


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage"))
    if identity != (1, "G10-B", "refiner_255_probe_run1_prepared"):
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    return manifest


def validate_local_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    references = {
        name: str(validate_file(spec)) for name, spec in manifest["references"].items()
    }
    input_path = validate_file(manifest["input_archive"])
    metadata_path = validate_file(manifest["metadata"])
    with zipfile.ZipFile(input_path) as archive:
        members = archive.namelist()
        if sorted(members) != sorted(manifest["input_archive"]["members_exact"]):
            raise AssertionError("Input archive members changed")
        if len(members) != len(set(members)) or archive.testzip() is not None:
            raise AssertionError("Input archive ZIP/CRC contract failed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta_spec = manifest["metadata"]
    if metadata.get(meta_spec["split"]) != [meta_spec["record"]]:
        raise AssertionError("Metadata record changed")

    audit = json.loads(Path(manifest["references"]["contract_audit_result"]["path"]).read_text())
    if not (
        audit.get("audit_outcome") == "passed"
        and audit.get("audit_assertions_passed") is True
        and audit.get("refiner_255_frame_static_contract_ready") is True
        and audit.get("compatibility_verdict") == "incompatible_default_max_frames_only"
    ):
        raise AssertionError("Pinned Refiner input-contract audit changed")
    expected_side_effects = {
        "creates_output_directory": True,
        "deletes_existing_output_archive": True,
        "writes_temporary_pickles_in_current_working_directory": True,
        "loads_model_and_runs_inference": True,
        "audit_executed_any_of_these": False,
    }
    if audit.get("source_side_effects_if_executed") != expected_side_effects:
        raise AssertionError("Pinned upstream side-effect audit changed")
    contract = audit["refiner_contract"]
    if not (
        contract["configured_max_frames"] == 750
        and contract["actual_frames"] == 255
        and contract["ready_with_max_frames_255_override"] is True
        and contract["would_truncate_detections"] is False
    ):
        raise AssertionError("Pinned 255-frame compatibility evidence changed")

    config_spec = manifest["config"]
    baseline_path = Path(config_spec["baseline"])
    target_path = Path(config_spec["target"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    differences = leaf_differences(baseline, target)
    if differences != {config_spec["allowed_leaf_difference"]}:
        raise AssertionError(f"Unexpected config delta: {sorted(differences)}")
    if "max_frames" in baseline["data"] or target["data"]["max_frames"] != 255:
        raise AssertionError("max_frames override contract changed")
    base_path = manifest["refiner"]["sources"]["base_config"]["path"]
    if baseline["imports"] != [base_path] or target["imports"] != [base_path]:
        raise AssertionError("Pinned base-config import changed")
    expected_override = {
        "experiment_name": "train_timesformer_100clip_coord_only_not_0init_l2_xyflip",
        "model": {
            "type": "SoccerTrackerTransformerTimeSformer",
            "zero_init_for_residual": False,
            "coord_loss_type": "l2",
        },
        "data": {
            "dataset_type": "SoccerSequenceCachedDataset",
            "max_clip_frames": 100,
            "flip_x_prob": 0.5,
            "flip_y_prob": 0.5,
        },
        "training": {
            "num_workers": 8,
            "tasks": {
                "track": False, "role": False, "team": False,
                "jersey": False, "coord": True, "missing": False,
            },
        },
    }
    for key, value in expected_override.items():
        if baseline[key] != value:
            raise AssertionError(f"Pinned upstream override changed: {key}")

    runtime = manifest["runtime"]
    future_paths = [
        Path(runtime["work_dir"]), Path(runtime["output_dir"]),
        Path(runtime["report_dir"]), Path(runtime["expected_output"]),
    ]
    for path in future_paths:
        require_absent(path, "future probe target")
    if runtime["heartbeat_seconds"] != 30 or runtime["timeout_seconds"] != 14400:
        raise AssertionError("Runtime timeout/heartbeat contract changed")
    if runtime["fallbacks"] != [] or not runtime["retry"].startswith("forbidden"):
        raise AssertionError("Failure policy changed")
    if any((runtime["visualization"], runtime["evaluation"], runtime["training"])):
        raise AssertionError("Forbidden runtime activity was enabled")

    arguments = manifest["future_command_arguments"]
    expected_arguments = [
        "--config", config_spec["target"],
        "--checkpoint", manifest["refiner"]["checkpoint"]["path"],
        "--input_pklz", manifest["input_archive"]["path"],
        "--output_dir", runtime["output_dir"],
        "--metadata_path", manifest["metadata"]["path"],
        "--split", manifest["metadata"]["split"],
    ]
    if arguments != expected_arguments:
        raise AssertionError("Future Refiner arguments changed")
    forbidden_flags = {"--visualize", "--visualize_video", "--img_dir", "--gt_dir"}
    if forbidden_flags.intersection(arguments):
        raise AssertionError("Visualization argument leaked into future command")

    approval = manifest["cuda_guard"]
    if approval["current_gpu_authorization"] is not False:
        raise AssertionError("Static manifest must not claim GPU authorization")
    if approval["approval_environment"] != "G10_REFINER_PROBE_RUN1_GPU_APPROVED=YES":
        raise AssertionError("GPU approval guard changed")
    attestation = manifest["refiner"]["read_only_attestation"]
    if not all((
        attestation["revision_and_all_source_config_hashes_matched"],
        attestation["checkpoint_stat_and_sha256_matched"],
        not attestation["torch_or_refiner_imported"],
        not attestation["checkpoint_deserialized"],
        not attestation["gpu_used"],
    )):
        raise AssertionError("GPU202 read-only asset attestation is incomplete")
    return {
        "references": references,
        "input_archive": {
            "path": str(input_path), "bytes": input_path.stat().st_size,
            "sha256": manifest["input_archive"]["sha256"], "members": members,
            "pickle_members_deserialized": False,
        },
        "metadata": manifest["metadata"],
        "config": {
            "baseline": str(baseline_path), "baseline_sha256": sha256_file(baseline_path),
            "target": str(target_path), "target_sha256": sha256_file(target_path),
            "leaf_differences": sorted(differences),
            "resolved_max_frames_before": 750, "target_max_frames": 255,
        },
        "source_side_effects": expected_side_effects,
        "mitigation": manifest["side_effect_contract"]["mitigation"],
        "future_paths_unused": True,
        "future_command": [manifest["python"], manifest["refiner"]["entry"], *arguments],
        "cuda_guard": approval,
        "remote_asset_attestation": attestation,
    }


def validate_runtime_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["refiner"]["root"])
    revision = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != manifest["refiner"]["revision"]:
        raise AssertionError("Refiner revision changed")
    sources = {name: str(validate_file(spec)) for name, spec in manifest["refiner"]["sources"].items()}
    checkpoint = validate_file(manifest["refiner"]["checkpoint"])
    python = Path(manifest["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    return {"revision": revision, "sources": sources, "checkpoint": str(checkpoint)}


def cuda_guard(manifest: dict[str, Any]) -> dict[str, Any]:
    guard = manifest["cuda_guard"]
    snippet = (
        "import json,torch; "
        "assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; "
        "free,total=torch.cuda.mem_get_info(0); p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'available':True,'count':1,'name':p.name,'free':free,'total':total}))"
    )
    completed = subprocess.run(
        [manifest["python"], "-c", snippet], check=True,
        capture_output=True, text=True, env=os.environ.copy(),
    )
    record = json.loads(completed.stdout.strip().splitlines()[-1])
    if record["name"] != guard["expected_device_name"]:
        raise AssertionError(f"Unexpected GPU: {record['name']}")
    if record["total"] < guard["minimum_total_memory_bytes"]:
        raise AssertionError("GPU total memory below guard")
    if record["free"] < guard["minimum_free_memory_bytes"]:
        raise AssertionError("GPU free memory below guard")
    return record


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


def validate_output(manifest: dict[str, Any]) -> dict[str, Any]:
    output = Path(manifest["runtime"]["expected_output"])
    if not output.is_file():
        raise FileNotFoundError(output)
    with zipfile.ZipFile(output) as archive:
        members = archive.namelist()
        if sorted(members) != ["10004.pkl", "10004_image.pkl"]:
            raise AssertionError(f"Unexpected Refiner output members: {members}")
        if len(members) != len(set(members)) or archive.testzip() is not None:
            raise AssertionError("Refiner output ZIP/CRC contract failed")
    work_dir = Path(manifest["runtime"]["work_dir"])
    leftovers = [name for name in manifest["side_effect_contract"]["expected_temp_files"] if (work_dir / name).exists()]
    if leftovers:
        raise AssertionError(f"Refiner temporary files remain: {leftovers}")
    if sha256_file(Path(manifest["input_archive"]["path"])) != manifest["input_archive"]["sha256"]:
        raise AssertionError("Immutable input archive changed")
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output), "members": members}


def run_probe(manifest: dict[str, Any], preflight: dict[str, Any]) -> int:
    approval_name, separator, approval_value = manifest["cuda_guard"]["approval_environment"].partition("=")
    if separator != "=" or os.environ.get(approval_name) != approval_value:
        raise PermissionError(f"Run mode requires {manifest['cuda_guard']['approval_environment']}")
    runtime_assets = validate_runtime_assets(manifest)
    runtime = manifest["runtime"]
    work_dir = Path(runtime["work_dir"])
    output_dir = Path(runtime["output_dir"])
    report_dir = Path(runtime["report_dir"])
    for path in (work_dir, output_dir, report_dir):
        require_absent(path, "future run target")
    work_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    log_path = Path(runtime["log"])
    command = [manifest["python"], manifest["refiner"]["entry"], *manifest["future_command_arguments"]]
    device = cuda_guard(manifest)
    started_wall = time.time()
    started = time.monotonic()
    timed_out = False
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_guard": device}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=work_dir, env=os.environ.copy(), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        next_heartbeat = started + runtime["heartbeat_seconds"]
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= runtime["timeout_seconds"]:
                timed_out = True
                terminate(process, runtime["termination_grace_seconds"])
                break
            if now >= next_heartbeat:
                print(f"heartbeat phase=refiner_forward elapsed_seconds={now-started:.1f} pid={process.pid}", flush=True)
                next_heartbeat += runtime["heartbeat_seconds"]
            time.sleep(1)
        process_exit = process.wait()
    result: dict[str, Any] = {
        "schema_version": 1, "gate": "G10-B", "stage": "refiner_255_probe_run1",
        "started_unix": started_wall, "ended_unix": time.time(),
        "wall_seconds": time.monotonic() - started, "command": command,
        "device": device, "runtime_assets": runtime_assets,
        "process_exit_code": process_exit, "timed_out": timed_out,
        "preflight": preflight, "fallbacks_used": [], "retry_used": False,
        "evaluation_started": False, "visualization_started": False, "training_started": False,
    }
    exit_code = process_exit or 0
    if not timed_out and process_exit == 0:
        try:
            result["output"] = validate_output(manifest)
            result.update({"outcome": "passed", "assertions_passed": True})
        except Exception as error:
            result.update({"outcome": "failed", "assertions_passed": False, "error": f"{type(error).__name__}: {error}"})
            exit_code = 1
    else:
        result.update({"outcome": "failed", "assertions_passed": False, "failure_category": "timeout" if timed_out else "worker_failure"})
        exit_code = exit_code or 1
    atomic_json(result, Path(runtime["result"]))
    return exit_code


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    started = time.time()
    print("heartbeat phase=static_preflight status=started", flush=True)
    environment = validate_environment(args.mode)
    local = validate_local_contract(manifest)
    if "torch" in sys.modules:
        raise AssertionError("Torch was unexpectedly imported during static preflight")
    preflight = {
        "environment": environment, **local,
        "torch_imported": False, "refiner_imported": False,
        "checkpoint_deserialized": False, "gpu_operations": [],
    }
    print("heartbeat phase=static_preflight status=passed", flush=True)
    if args.mode == "run":
        return run_probe(manifest, preflight)

    outputs = manifest["preflight_outputs"]
    report_dir = Path(outputs["report_dir"])
    require_absent(report_dir, "preflight report")
    report_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1, "gate": "G10-B", "stage": "refiner_255_probe_run1_preflight",
        "outcome": "passed", "assertions_passed": True, "process_exit_code": 0,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.time() - started,
        "command": [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest.resolve()), "--mode", "preflight"],
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256_file(args.manifest),
        "git": git_identity(), "preflight": preflight,
        "resources": {"peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "gpu_used": False},
        "forbidden_actions_executed": [], "fallbacks_used": [], "retry_used": False,
        "not_run": ["Torch import", "Refiner import", "checkpoint deserialization", "nvidia-smi", "GPU", "model load", "forward", "Step 3", "conversion", "evaluation", "visualization", "training"],
        "next_step_requires_fresh_nvidia_smi_and_explicit_gpu_approval": True,
    }
    atomic_json(result, Path(outputs["result"]))
    write_new(
        "heartbeat phase=static_preflight status=started\n"
        "heartbeat phase=static_preflight status=passed\nexit_code=0\n",
        Path(outputs["log"]),
    )
    print(json.dumps({"outcome": "passed", "result": outputs["result"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
