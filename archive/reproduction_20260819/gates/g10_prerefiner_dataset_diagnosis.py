#!/usr/bin/env python3
"""Static preflight and future launcher for CPU-only pre-Refiner dataset diagnosis run2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_prerefiner_dataset_diagnosis_run2_sngs10004.json"
EXPECTED_IDENTITY = (1, "G10-B", "prerefiner_dataset_diagnosis_run2_prepared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("preflight", "diagnose"), default="preflight")
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Local path escapes repository: {path}")
    return resolved


def require_absent(path: Path, label: str) -> None:
    inside_repo(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} is already occupied: {path}")


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
    if identity != EXPECTED_IDENTITY:
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


def worker_ast_contract(worker: Path, required_phases: list[str]) -> dict[str, Any]:
    source = worker.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(worker))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    allowed_top_level = {
        "__future__", "argparse", "json", "multiprocessing", "os", "time", "pathlib", "typing"
    }
    if not set(top_level_imports).issubset(allowed_top_level):
        raise AssertionError(f"Unexpected worker top-level imports: {top_level_imports}")

    phase_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "run_phase" and node.args and isinstance(node.args[0], ast.Constant):
            phase_calls.append((node.lineno, node.args[0].value))
    phase_calls.sort()
    phase_names = [name for _, name in phase_calls]
    if phase_names != required_phases:
        raise AssertionError(f"Worker phase order changed: {phase_names}")

    banned_fragments = [
        "tracklab_main.main(",
        "tracklab_main.evaluate(",
        "TrackerState(",
        "Pipeline(",
        "cfg.modules",
        "nvidia-smi",
        "torch.",
        "cuda.",
    ]
    present = [fragment for fragment in banned_fragments if fragment in source]
    if present:
        raise AssertionError(f"Worker contains prohibited execution fragments: {present}")
    if "instantiate(composed[\"cfg\"].dataset)" not in source:
        raise AssertionError("Worker no longer limits Hydra instantiate to cfg.dataset")
    if "video_dir_to_dfs" not in source or "ReplayPool" not in source:
        raise AssertionError("Worker lost required single-video/replay diagnostic boundaries")
    return {
        "top_level_imports": top_level_imports,
        "phase_order": phase_names,
        "phase_source_lines": {name: line for line, name in phase_calls},
        "prohibited_fragments_present": [],
        "dataset_only_hydra_instantiate": True,
        "single_video_worker_present": True,
        "replay_pool_present": True,
    }


def static_preflight(manifest: dict[str, Any], manifest_path: Path, mode: str) -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise AssertionError(f"Wrong Python: {sys.executable}")
    expected_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "",
        "LD_LIBRARY_PATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    actual_env = {name: os.environ.get(name) for name in expected_env}
    if actual_env != expected_env:
        raise AssertionError(f"Static environment mismatch: {actual_env}")

    reference_results = {}
    for label, spec in manifest["references"].items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise AssertionError(f"Reference changed: {label}")
        reference_results[label] = {"path": str(path), "sha256": actual}
    run1_result = json.loads(Path(manifest["references"]["run1_result"]["path"]).read_text(encoding="utf-8"))
    expected_failure = {
        "outcome": "failed",
        "failure_category": "phase_timeout",
        "timeout_phase": "instantiate_dataset",
        "timed_out": True,
        "evaluation_started": False,
        "training_started": False,
        "fallbacks_used": [],
    }
    for key, expected in expected_failure.items():
        if run1_result.get(key) != expected:
            raise AssertionError(f"Run1 failure contract changed: {key}")

    source_results = {}
    for label in ("tracklab_main", "soccernet_dataset"):
        spec = manifest["source_files"][label]
        path = Path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise AssertionError(f"Read-only source changed: {label}")
        source_results[label] = {"path": str(path), "sha256": actual}
    launcher = Path(manifest["source_files"]["launcher"])
    worker = Path(manifest["source_files"]["worker"])
    if launcher.resolve() != Path(__file__).resolve() or not worker.is_file():
        raise AssertionError("Local diagnosis source paths changed")
    ast.parse(launcher.read_text(encoding="utf-8"), filename=str(launcher))
    worker_contract = worker_ast_contract(worker, manifest["phases"])

    config_path = Path(manifest["config"]["path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["pipeline"] != manifest["config"]["pipeline"]:
        raise AssertionError("Pipeline changed")
    dataset = config["dataset"]
    expected_dataset = {
        "nframes": manifest["sample"]["frame_count"],
        "eval_set": manifest["sample"]["split"],
        "dataset_path": manifest["sample"]["dataset_root"],
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise AssertionError(f"Dataset config changed: {key}")
    if dataset["vids_dict"][manifest["sample"]["split"]] != [manifest["sample"]["sequence"]]:
        raise AssertionError("Fixed sequence changed")
    if config.get("eval_tracking") is not False or config.get("visualization") is not None or config.get("use_wandb") is not False:
        raise AssertionError("Disabled runtime semantics changed")

    sample = manifest["sample"]
    for key in ("dataset_root", "split_root", "video_root", "frame_root"):
        if not Path(sample[key]).is_dir():
            raise FileNotFoundError(sample[key])
    if (Path(sample["video_root"]) / "Labels-GameState.json").exists():
        raise AssertionError("Fixed sample unexpectedly gained Labels-GameState.json")
    frames = sorted(Path(sample["frame_root"]).glob("*.jpg"))
    total_bytes = sum(path.stat().st_size for path in frames)
    if len(frames) != sample["frame_count"] or total_bytes != sample["frame_total_bytes"]:
        raise AssertionError("Fixed frame inventory changed")
    if (frames[0].name, frames[0].stat().st_size) != (
        sample["first_frame"]["name"], sample["first_frame"]["bytes"]
    ):
        raise AssertionError("First frame metadata changed")
    if (frames[-1].name, frames[-1].stat().st_size) != (
        sample["last_frame"]["name"], sample["last_frame"]["bytes"]
    ):
        raise AssertionError("Last frame metadata changed")

    if manifest["diagnostic_adapter"]["dataset_pool_workers"] != 4:
        raise AssertionError("Pool worker count changed")
    if manifest["diagnostic_adapter"]["pool_jobs"] != 1:
        raise AssertionError("Pool job count changed")
    if manifest["diagnostic_adapter"]["suppressed_path_exact"] != "/home/tianlin/.cache/mim":
        raise AssertionError("OpenMIM suppression is not exact")
    if manifest["diagnostic_adapter"]["cache_environment"] != [
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "MPLCONFIGDIR"
    ]:
        raise AssertionError("Cache environment contract changed")
    if not manifest["diagnostic_adapter"]["single_video_result_replay"]["enabled"]:
        raise AssertionError("Single-video replay disabled")
    if not manifest["diagnostic_adapter"]["load_set_replay"]["enabled"]:
        raise AssertionError("TrackingSet replay disabled")

    timeouts = manifest["timeouts_seconds"]
    required_timeout_keys = {"worker_boot", "worker_transition", "termination_grace", "overall", *manifest["phases"]}
    if set(timeouts) != required_timeout_keys or any(int(value) <= 0 for value in timeouts.values()):
        raise AssertionError("Timeout contract changed")
    if timeouts["single_video_worker"] != 1200 or manifest["heartbeat_seconds"] != 30:
        raise AssertionError("Failed-stage timeout or heartbeat changed")
    if manifest["success"]["required_phases"] != manifest["phases"]:
        raise AssertionError("Success phase list changed")

    outputs = {key: Path(value) for key, value in manifest["outputs"].items()}
    for path in outputs.values():
        inside_repo(path)
    future_roots = (outputs["report_dir"], outputs["cache_dir"], outputs["hydra_run_dir"])
    for path in future_roots:
        require_absent(path, "future diagnosis output")
    preflight_dir = outputs["preflight_report_dir"]
    if mode == "preflight":
        require_absent(preflight_dir, "preflight report")
    elif not preflight_dir.is_dir():
        raise FileNotFoundError("Formal preflight report is missing")
    for key in ("preflight_result", "preflight_log"):
        if outputs[key].parent != preflight_dir:
            raise AssertionError(f"Preflight output escapes report directory: {key}")
    for key in ("events", "log", "result"):
        if outputs[key].parent != outputs["report_dir"]:
            raise AssertionError(f"Future report output escapes report directory: {key}")
    if outputs["resolved_config"].parent != outputs["hydra_run_dir"]:
        raise AssertionError("Resolved config escapes future Hydra run directory")

    approval_name, separator, approval_value = manifest["approval"]["required_environment"].partition("=")
    if separator != "=" or not approval_name or approval_value != "YES":
        raise AssertionError("Future approval guard changed")
    future_command = [
        manifest["python"], str(launcher), "--manifest", str(manifest_path.resolve()), "--mode", "diagnose"
    ]
    return {
        "python": str(Path(sys.executable).resolve()),
        "environment": actual_env,
        "git": git_identity(),
        "references": reference_results,
        "run1_failure_contract": expected_failure,
        "read_only_sources": source_results,
        "local_sources": {
            "launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
            "worker": {"path": str(worker), "sha256": sha256_file(worker)},
        },
        "worker_ast_contract": worker_contract,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "pipeline": config["pipeline"],
            "dataset": expected_dataset,
            "sequence": manifest["sample"]["sequence"],
            "eval_tracking": False,
            "visualization": None,
            "use_wandb": False,
        },
        "sample": {
            "frame_root": sample["frame_root"],
            "frame_count": len(frames),
            "frame_total_bytes": total_bytes,
            "labels_gamestate_present": False,
            "frame_contents_read": False,
        },
        "diagnostic_boundaries": manifest["phases"],
        "timeouts_seconds": timeouts,
        "heartbeat_seconds": manifest["heartbeat_seconds"],
        "future_outputs_absent": [str(path) for path in future_roots],
        "future_command": future_command,
        "future_approval_guard": manifest["approval"]["required_environment"],
        "tracklab_or_model_imports": False,
        "torch_imported": "torch" in sys.modules,
        "video_or_weight_contents_read": False,
        "gpu_operations": [],
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def run_diagnosis(manifest: dict[str, Any], manifest_path: Path, preflight: dict[str, Any]) -> int:
    approval_name, _, approval_value = manifest["approval"]["required_environment"].partition("=")
    if os.environ.get(approval_name) != approval_value:
        raise PermissionError("Future CPU diagnosis has not been approved")
    outputs = {key: Path(value) for key, value in manifest["outputs"].items()}
    outputs["report_dir"].mkdir(parents=True, exist_ok=False)
    outputs["cache_dir"].mkdir(parents=True, exist_ok=False)
    command = [
        manifest["python"],
        str(Path(manifest["source_files"]["worker"])),
        "--manifest", str(manifest_path.resolve()),
        "--events", str(outputs["events"]),
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "",
        "LD_LIBRARY_PATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HOME": str(outputs["cache_dir"] / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(outputs["cache_dir"] / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(outputs["cache_dir"] / "huggingface" / "transformers"),
        "XDG_CACHE_HOME": str(outputs["cache_dir"] / "xdg"),
        "MPLCONFIGDIR": str(outputs["cache_dir"] / "matplotlib"),
    })
    started_wall = time.time()
    started = time.monotonic()
    active_phase = "worker_boot"
    active_started = started
    seen = 0
    last_heartbeat = started
    timed_out = False
    timeout_phase = None
    with outputs["log"].open("x", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps({"command": command, "cuda_visible_devices": ""}) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        while process.poll() is None:
            now = time.monotonic()
            events = read_events(outputs["events"])
            for event in events[seen:]:
                if event["status"] == "started":
                    active_phase = event["phase"]
                    active_started = now
            seen = len(events)
            allowed = int(manifest["timeouts_seconds"].get(active_phase, manifest["timeouts_seconds"]["worker_transition"]))
            if now - started >= manifest["timeouts_seconds"]["overall"] or now - active_started >= allowed:
                timed_out = True
                timeout_phase = active_phase
                terminate(process, int(manifest["timeouts_seconds"]["termination_grace"]))
                break
            if now - last_heartbeat >= manifest["heartbeat_seconds"]:
                print(
                    f"heartbeat phase={active_phase} elapsed_seconds={now - started:.1f} pid={process.pid}",
                    flush=True,
                )
                last_heartbeat = now
            time.sleep(1.0)
    exit_code = process.wait()
    events = read_events(outputs["events"])
    passed = [event["phase"] for event in events if event["status"] == "passed"]
    required = manifest["success"]["required_phases"]
    assertions_passed = (
        not timed_out
        and exit_code == 0
        and all(passed.count(phase) == 1 for phase in required)
        and any(event["phase"] == "worker" and event["status"] == "passed" for event in events)
    )
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": "prerefiner_dataset_diagnosis_run2",
        "outcome": "passed" if assertions_passed else "failed",
        "started_unix": started_wall,
        "ended_unix": time.time(),
        "wall_seconds": time.monotonic() - started,
        "process_exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_phase": timeout_phase,
        "assertions_passed": assertions_passed,
        "evaluation_started": False,
        "training_started": False,
        "gpu_operations": [],
        "fallbacks_used": [],
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "command": command,
        "preflight": preflight,
        "events": events,
    }
    atomic_json(result, outputs["result"])
    print(json.dumps({"outcome": result["outcome"], "result": str(outputs["result"])}), flush=True)
    return 0 if assertions_passed else 1


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    started = time.monotonic()
    preflight = static_preflight(manifest, manifest_path, args.mode)
    if preflight["torch_imported"] or preflight["tracklab_or_model_imports"]:
        raise AssertionError("Static preflight imported forbidden runtime modules")
    if args.mode == "diagnose":
        return run_diagnosis(manifest, manifest_path, preflight)

    outputs = {key: Path(value) for key, value in manifest["outputs"].items()}
    outputs["preflight_report_dir"].mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": "prerefiner_dataset_diagnosis_run2_preflight",
        "outcome": "passed",
        "assertions_passed": True,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "fallbacks_used": [],
        "preflight": preflight,
    }
    atomic_json(result, outputs["preflight_result"])
    write_new("static_preflight passed\n", outputs["preflight_log"])
    print(json.dumps({"outcome": "passed", "result": str(outputs["preflight_result"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
