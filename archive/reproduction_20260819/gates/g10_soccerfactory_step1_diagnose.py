#!/usr/bin/env python3
"""CPU-only staged startup diagnosis for G10-B TrackLab Step 1.

The parent supervises one worker and enforces a separate timeout for importing
TrackLab, composing the Hydra config, and instantiating only the fixed dataset.
It never calls TrackLab main and never instantiates evaluator or model modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_soccerfactory_step1_diagnosis_sngs10004.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--events", type=Path)
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
        raise AssertionError(f"Output escapes repository: {path}")
    return resolved


def atomic_json_dump(value: Any, path: Path) -> None:
    require_inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    if identity != (1, "G10-B", "tracklab_step1_startup_diagnosis"):
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    return manifest


def static_preflight(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise AssertionError(f"Wrong Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CPU diagnosis requires CUDA_VISIBLE_DEVICES to be explicitly empty")

    base = manifest["base_step1_manifest"]
    if sha256_file(Path(base["path"])) != base["sha256"]:
        raise AssertionError("Base Step 1 manifest changed")
    for label, path_text in manifest["source_files"].items():
        if not Path(path_text).is_file():
            raise FileNotFoundError(f"{label}: {path_text}")
    if not Path(manifest["config"]["path"]).is_file():
        raise FileNotFoundError(manifest["config"]["path"])
    if not Path(manifest["sample"]["dataset_root"]).is_dir():
        raise FileNotFoundError(manifest["sample"]["dataset_root"])

    report_dir = Path(manifest["outputs"]["report_dir"])
    require_inside_repo(report_dir)
    if report_dir.exists() or report_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite diagnosis report: {report_dir}")
    for output in manifest["outputs"].values():
        if not Path(output).resolve(strict=False).is_relative_to(report_dir.resolve(strict=False)):
            raise AssertionError(f"Output is outside diagnosis report directory: {output}")

    timeouts = manifest["timeouts_seconds"]
    required_timeout_keys = {
        "worker_boot",
        "import_tracklab_main",
        "hydra_compose",
        "instantiate_dataset",
        "termination_grace",
        "overall",
    }
    if set(timeouts) != required_timeout_keys or any(int(value) <= 0 for value in timeouts.values()):
        raise AssertionError("Invalid staged timeout policy")
    if manifest["diagnostic_adapter"]["dataset_pool_workers"] != 4:
        raise AssertionError("Diagnostic Pool cap changed")
    diagnostic_adapter = manifest["diagnostic_adapter"]
    if diagnostic_adapter.get("suppress_mim_home_cache_creation", False):
        expected_mim_cache = str((Path.home() / ".cache/mim").resolve(strict=False))
        if diagnostic_adapter.get("suppressed_path_exact") != expected_mim_cache:
            raise AssertionError("OpenMIM suppression path is not the exact home cache path")
        local_cache = diagnostic_adapter.get("local_cache_dir")
        if local_cache is None:
            raise AssertionError("Cache-adapted diagnosis requires local_cache_dir")
        local_cache_path = require_inside_repo(Path(local_cache))
        if local_cache_path.exists() or local_cache_path.is_symlink():
            raise FileExistsError(f"Diagnostic cache path is already occupied: {local_cache_path}")
    if manifest["heartbeat_seconds"] != 30:
        raise AssertionError("Heartbeat interval changed")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "base_manifest_sha256": base["sha256"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_root": manifest["sample"]["dataset_root"],
        "host_logical_cpus": os.cpu_count(),
        "dataset_pool_workers": manifest["diagnostic_adapter"]["dataset_pool_workers"],
        "home_unchanged": str(Path.home()),
        "suppressed_path_exact": diagnostic_adapter.get("suppressed_path_exact"),
        "local_cache_dir": diagnostic_adapter.get("local_cache_dir"),
        "timeouts_seconds": timeouts,
    }


def emit_event(events_path: Path, phase: str, status: str, started: float, **extra: Any) -> None:
    event = {
        "phase": phase,
        "status": status,
        "unix_time": time.time(),
        "worker_elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        **extra,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(event, ensure_ascii=False), flush=True)


def run_worker(manifest: dict[str, Any], events_path: Path) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("Worker requires CUDA_VISIBLE_DEVICES to be empty")
    require_inside_repo(events_path)
    started = time.monotonic()

    def run_phase(name: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        phase_started = time.monotonic()
        emit_event(events_path, name, "started", started)
        try:
            output = action()
        except BaseException as error:
            emit_event(
                events_path,
                name,
                "failed",
                started,
                phase_seconds=time.monotonic() - phase_started,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        emit_event(
            events_path,
            name,
            "passed",
            started,
            phase_seconds=time.monotonic() - phase_started,
            output=output,
        )
        return output

    imported: dict[str, Any] = {}

    def import_tracklab() -> dict[str, Any]:
        import importlib

        module = importlib.import_module("tracklab.main")
        imported["tracklab_main"] = module
        return {"origin": str(Path(module.__file__).resolve())}

    run_phase("import_tracklab_main", import_tracklab)

    composed: dict[str, Any] = {}

    def compose_config() -> dict[str, Any]:
        from hydra._internal.hydra import Hydra
        from hydra._internal.utils import create_config_search_path
        from hydra.core.global_hydra import GlobalHydra
        from hydra.types import RunMode

        # Match the real CLI's two primary search roots: the local config is
        # selected first, while TrackLab's package still supplies its groups.
        config_search_path = create_config_search_path(
            search_path_dir=manifest["config"]["directory"]
        )
        config_search_path.append(
            provider="tracklab_core",
            path="pkg://tracklab.configs",
            anchor="main",
        )
        hydra = Hydra.create_main_hydra2(
            task_name="g10_step1_startup_diagnosis",
            config_search_path=config_search_path,
        )
        try:
            cfg = hydra.compose_config(
                config_name=manifest["config"]["name"],
                overrides=[],
                run_mode=RunMode.RUN,
                with_log_configuration=False,
            )
        finally:
            GlobalHydra.instance().clear()
        composed["cfg"] = cfg
        return {
            "pipeline": list(cfg.pipeline),
            "eval_set": str(cfg.dataset.eval_set),
            "nframes": int(cfg.dataset.nframes),
            "sequence": str(cfg.dataset.vids_dict.sn500[0]),
            "dataset_target": str(cfg.dataset._target_),
        }

    compose_output = run_phase("hydra_compose", compose_config)
    if compose_output != {
        "pipeline": ["bbox_detector", "reid", "track"],
        "eval_set": "sn500",
        "nframes": 255,
        "sequence": "SNGS-10004",
        "dataset_target": "tracklab.wrappers.SoccerNetGameState",
    }:
        raise AssertionError(f"Composed dataset contract changed: {compose_output}")

    def instantiate_dataset() -> dict[str, Any]:
        from hydra.utils import instantiate

        suppress_mim_cache = bool(
            manifest["diagnostic_adapter"].get("suppress_mim_home_cache_creation", False)
        )
        original_makedirs = os.makedirs
        blocked_mim_cache = (Path.home() / ".cache/mim").resolve(strict=False)

        def diagnostic_makedirs(name: Any, *args: Any, **kwargs: Any) -> None:
            requested = Path(name).resolve(strict=False)
            if suppress_mim_cache and requested == blocked_mim_cache:
                return
            original_makedirs(name, *args, **kwargs)

        if suppress_mim_cache:
            os.makedirs = diagnostic_makedirs
        try:
            from tracklab.wrappers.datasets.soccernet import soccernet_game_state
        finally:
            os.makedirs = original_makedirs

        original_pool = soccernet_game_state.Pool
        worker_count = int(manifest["diagnostic_adapter"]["dataset_pool_workers"])

        def limited_pool(*args: Any, **kwargs: Any) -> Any:
            if not args and "processes" not in kwargs:
                kwargs["processes"] = worker_count
            return original_pool(*args, **kwargs)

        soccernet_game_state.Pool = limited_pool
        try:
            dataset = instantiate(composed["cfg"].dataset)
        finally:
            soccernet_game_state.Pool = original_pool

        expected = manifest["sample"]
        if sorted(dataset.sets) != [expected["split"]]:
            raise AssertionError(f"Unexpected dataset sets: {sorted(dataset.sets)}")
        tracking_set = dataset.sets[expected["split"]]
        video_rows = len(tracking_set.video_metadatas)
        image_rows = len(tracking_set.image_metadatas)
        if video_rows != 1 or image_rows != expected["required_image_rows"]:
            raise AssertionError(f"Unexpected dataset rows: videos={video_rows}, images={image_rows}")
        sequence_names = tracking_set.video_metadatas["name"].astype(str).tolist()
        if sequence_names != [expected["sequence"]]:
            raise AssertionError(f"Unexpected sequence: {sequence_names}")
        frame_values = tracking_set.image_metadatas["frame"].astype(int).tolist()
        if frame_values != list(range(expected["required_image_rows"])):
            raise AssertionError("Dataset frames are not exactly 0..254")
        detections = tracking_set.detections_gt
        return {
            "sets": sorted(dataset.sets),
            "video_rows": video_rows,
            "image_rows": image_rows,
            "sequence_names": sequence_names,
            "ground_truth_detection_rows": None if detections is None else len(detections),
            "pool_workers": worker_count,
            "mim_home_cache_creation_suppressed": suppress_mim_cache,
        }

    run_phase("instantiate_dataset", instantiate_dataset)
    emit_event(events_path, "worker", "passed", started)
    return 0


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: int) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(manifest: dict[str, Any], manifest_path: Path, preflight: dict[str, Any]) -> int:
    outputs = {key: Path(value) for key, value in manifest["outputs"].items()}
    report_dir = outputs["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=False)
    events_path = outputs["events"]
    log_path = outputs["log"]
    result_path = outputs["result"]
    command = [
        manifest["python"],
        str(Path(__file__).resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--worker",
        "--events",
        str(events_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    diagnostic_cache = manifest["diagnostic_adapter"].get("local_cache_dir")
    if diagnostic_cache is not None:
        diagnostic_cache_path = require_inside_repo(Path(diagnostic_cache))
        if diagnostic_cache_path.exists() or diagnostic_cache_path.is_symlink():
            raise FileExistsError(
                f"Refusing to reuse diagnostic cache directory: {diagnostic_cache_path}"
            )
        matplotlib_cache = diagnostic_cache_path / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=False)
        environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    started_wall = time.time()
    started_monotonic = time.monotonic()
    timeouts = manifest["timeouts_seconds"]
    heartbeat_seconds = int(manifest["heartbeat_seconds"])
    active_phase = "worker_boot"
    phase_started = started_monotonic
    seen_event_count = 0
    timed_out_phase: str | None = None

    with log_path.open("x", encoding="utf-8") as log_handle:
        log_handle.write(json.dumps({"command": command, "environment": {"CUDA_VISIBLE_DEVICES": ""}}) + "\n")
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
        next_heartbeat = started_monotonic + heartbeat_seconds
        while process.poll() is None:
            now = time.monotonic()
            events = read_events(events_path)
            for event in events[seen_event_count:]:
                if event["status"] == "started":
                    active_phase = event["phase"]
                    phase_started = now
                elif event["phase"] == active_phase and event["status"] in {"passed", "failed"}:
                    active_phase = "worker_transition"
                    phase_started = now
            seen_event_count = len(events)

            overall_elapsed = now - started_monotonic
            phase_elapsed = now - phase_started
            phase_limit = int(timeouts.get(active_phase, timeouts["worker_boot"]))
            if overall_elapsed >= int(timeouts["overall"]):
                timed_out_phase = "overall"
            elif active_phase != "worker_transition" and phase_elapsed >= phase_limit:
                timed_out_phase = active_phase
            if timed_out_phase is not None:
                print(
                    f"heartbeat phase={timed_out_phase} status=timeout overall_seconds={overall_elapsed:.1f} phase_seconds={phase_elapsed:.1f}",
                    flush=True,
                )
                terminate_process_group(process, int(timeouts["termination_grace"]))
                break
            if now >= next_heartbeat:
                print(
                    f"heartbeat phase={active_phase} status=running overall_seconds={overall_elapsed:.1f} phase_seconds={phase_elapsed:.1f} pid={process.pid}",
                    flush=True,
                )
                next_heartbeat += heartbeat_seconds
            time.sleep(1)
        worker_exit_code = process.wait()

    ended_wall = time.time()
    events = read_events(events_path)
    required_phases = manifest["success"]["required_phases"]
    passed_phases = [event["phase"] for event in events if event["status"] == "passed"]
    assertions_passed = (
        timed_out_phase is None
        and worker_exit_code == 0
        and all(phase in passed_phases for phase in required_phases)
        and any(event["phase"] == "worker" and event["status"] == "passed" for event in events)
    )
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": "tracklab_step1_startup_diagnosis",
        "outcome": "passed" if assertions_passed else "failed",
        "assertions_passed": assertions_passed,
        "started_unix": started_wall,
        "ended_unix": ended_wall,
        "wall_seconds": ended_wall - started_wall,
        "command": command,
        "worker_exit_code": worker_exit_code,
        "timed_out_phase": timed_out_phase,
        "heartbeat_seconds": heartbeat_seconds,
        "events": events,
        "preflight": preflight,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "git": git_identity(),
        "gpu_operations": [],
        "model_weight_reads": [],
        "evaluator_or_model_instantiation": [],
        "inference_evaluation_training": [],
        "fallbacks_used": [
            {
                "kind": "diagnostic_concurrency_adapter",
                "upstream_default_pool_workers": manifest["diagnostic_adapter"]["host_logical_cpus_observed_before_run"],
                "diagnostic_pool_workers": manifest["diagnostic_adapter"]["dataset_pool_workers"],
                "declared_before_run": True
            },
            {
                "kind": "diagnostic_cache_adapter",
                "mim_home_cache_creation_suppressed": bool(
                    manifest["diagnostic_adapter"].get(
                        "suppress_mim_home_cache_creation", False
                    )
                ),
                "local_cache_dir": manifest["diagnostic_adapter"].get(
                    "local_cache_dir"
                ),
                "declared_before_run": True,
            },
        ],
    }
    if not assertions_passed:
        result["failure_category"] = "environment_or_abi" if timed_out_phase == "import_tracklab_main" else "data_contract"
    atomic_json_dump(result, result_path)
    print(json.dumps({"outcome": result["outcome"], "result": str(result_path)}), flush=True)
    return 0 if assertions_passed else 1


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.worker:
        if args.events is None:
            raise ValueError("--worker requires --events")
        try:
            return run_worker(manifest, args.events)
        except BaseException:
            traceback.print_exc()
            return 1
    preflight = static_preflight(manifest, args.manifest)
    print("heartbeat phase=static_preflight status=passed", flush=True)
    return supervise(manifest, args.manifest, preflight)


if __name__ == "__main__":
    raise SystemExit(main())
