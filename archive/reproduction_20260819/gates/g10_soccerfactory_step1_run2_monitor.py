#!/usr/bin/env python3
"""Wait for one genuinely idle H800, then launch exactly one guarded run2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
PYTHON = "/remote-home/haolinyang/anaconda3/envs/tracklab2/bin/python"
LAUNCHER = REPO / "reproduction/gates/g10_soccerfactory_step1_run2.py"
REPORT_DIR = REPO / "reports/g10/20260813_step1_run2_monitor"
EVENTS = REPORT_DIR / "events.jsonl"
RESULT = REPORT_DIR / "result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--monitor-timeout-seconds", type=int, default=172800)
    parser.add_argument("--maximum-used-mib", type=int, default=1024)
    parser.add_argument("--maximum-utilization-percent", type=int, default=10)
    return parser.parse_args()


def emit(phase: str, status: str, **extra: Any) -> None:
    record = {"unix_time": time.time(), "phase": phase, "status": status, **extra}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


def query() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_output = subprocess.run(gpu_command, check=True, capture_output=True, text=True, timeout=30).stdout
    app_output = subprocess.run(app_command, check=True, capture_output=True, text=True, timeout=30).stdout
    gpus = []
    for row in csv.reader(StringIO(gpu_output)):
        values = [value.strip() for value in row]
        gpus.append(
            {
                "index": int(values[0]),
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": int(values[3]),
                "memory_used_mib": int(values[4]),
                "memory_free_mib": int(values[5]),
                "utilization_percent": int(values[6]),
                "temperature_c": int(values[7]),
            }
        )
    apps = []
    for row in csv.reader(StringIO(app_output)):
        values = [value.strip() for value in row]
        if values:
            apps.append(
                {
                    "gpu_uuid": values[0],
                    "pid": int(values[1]),
                    "process_name": values[2],
                    "used_memory_mib": int(values[3]),
                }
            )
    return gpus, apps


def candidates(
    gpus: list[dict[str, Any]],
    apps: list[dict[str, Any]],
    maximum_used_mib: int,
    maximum_utilization_percent: int,
) -> list[int]:
    active_uuids = {app["gpu_uuid"] for app in apps}
    return sorted(
        gpu["index"]
        for gpu in gpus
        if gpu["name"] == "NVIDIA H800"
        and gpu["uuid"] not in active_uuids
        and gpu["memory_used_mib"] <= maximum_used_mib
        and gpu["utilization_percent"] <= maximum_utilization_percent
    )


def write_result(value: dict[str, Any]) -> None:
    temporary = RESULT.with_name(f".{RESULT.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, RESULT)


def main() -> int:
    args = parse_args()
    if args.poll_seconds != 1800 or args.heartbeat_seconds != 60:
        raise AssertionError("The approved 30-minute poll / 60-second heartbeat contract changed")
    if REPORT_DIR.exists() or REPORT_DIR.is_symlink():
        raise FileExistsError(f"Monitor report path is already used: {REPORT_DIR}")
    future_paths = [
        REPO / ".runtime/g10/sngs10004_step1/run2",
        REPO / ".runtime/g10/sngs10004_step1/run2_cache",
        REPO / "reports/g10/20260813_step1_run2",
    ]
    if any(path.exists() or path.is_symlink() for path in future_paths):
        raise FileExistsError(f"A future run2 path is already occupied: {future_paths}")
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    EVENTS.touch(exist_ok=False)
    started_wall = time.time()
    started = time.monotonic()
    deadline = started + args.monitor_timeout_seconds
    next_check = started
    next_heartbeat = started
    checks = 0
    selected: int | None = None

    emit(
        "monitor",
        "started",
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        monitor_timeout_seconds=args.monitor_timeout_seconds,
        idle_contract={
            "no_compute_process": True,
            "maximum_used_mib": args.maximum_used_mib,
            "maximum_utilization_percent": args.maximum_utilization_percent,
            "stable_checks": 2,
        },
    )
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_check:
                checks += 1
                gpus, apps = query()
                first = candidates(gpus, apps, args.maximum_used_mib, args.maximum_utilization_percent)
                emit("nvidia_smi", "checked", check=checks, gpus=gpus, compute_apps=apps, candidates=first)
                if first:
                    time.sleep(5)
                    confirm_gpus, confirm_apps = query()
                    second = candidates(
                        confirm_gpus,
                        confirm_apps,
                        args.maximum_used_mib,
                        args.maximum_utilization_percent,
                    )
                    stable = sorted(set(first) & set(second))
                    emit(
                        "nvidia_smi_confirmation",
                        "checked",
                        check=checks,
                        gpus=confirm_gpus,
                        compute_apps=confirm_apps,
                        candidates=second,
                        stable_candidates=stable,
                    )
                    if stable:
                        selected = stable[0]
                        break
                next_check = time.monotonic() + args.poll_seconds
            if now >= next_heartbeat:
                emit(
                    "monitor",
                    "heartbeat",
                    elapsed_seconds=time.monotonic() - started,
                    checks=checks,
                    next_check_in_seconds=max(0, next_check - time.monotonic()),
                )
                next_heartbeat = time.monotonic() + args.heartbeat_seconds
            time.sleep(1)

        if selected is None:
            result = {
                "schema_version": 1,
                "gate": "G10-B",
                "stage": "tracklab_step1_run2_monitor",
                "outcome": "timed_out_waiting_for_idle_gpu",
                "started_unix": started_wall,
                "ended_unix": time.time(),
                "wall_seconds": time.monotonic() - started,
                "checks": checks,
                "run2_started": False,
            }
            write_result(result)
            emit("monitor", "failed", reason="idle_gpu_timeout", checks=checks)
            return 124

        emit("run2", "launching", physical_gpu=selected, checks=checks)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(selected)
        environment["G10_STEP1_RUN2_GPU_APPROVED"] = "YES"
        command = [PYTHON, str(LAUNCHER), "--mode", "run"]
        completed = subprocess.run(command, cwd=REPO, env=environment, check=False)
        run_result_path = REPO / "reports/g10/20260813_step1_run2/result.json"
        result = {
            "schema_version": 1,
            "gate": "G10-B",
            "stage": "tracklab_step1_run2_monitor",
            "outcome": "run2_completed" if completed.returncode == 0 else "run2_failed",
            "started_unix": started_wall,
            "ended_unix": time.time(),
            "wall_seconds": time.monotonic() - started,
            "checks": checks,
            "physical_gpu": selected,
            "run2_started": True,
            "run2_exit_code": completed.returncode,
            "run2_result": str(run_result_path),
            "run3_started": False,
        }
        write_result(result)
        emit("run2", "passed" if completed.returncode == 0 else "failed", physical_gpu=selected, exit_code=completed.returncode)
        return completed.returncode
    except BaseException as error:
        if not RESULT.exists():
            write_result(
                {
                    "schema_version": 1,
                    "gate": "G10-B",
                    "stage": "tracklab_step1_run2_monitor",
                    "outcome": "monitor_failed",
                    "started_unix": started_wall,
                    "ended_unix": time.time(),
                    "wall_seconds": time.monotonic() - started,
                    "checks": checks,
                    "run2_started": selected is not None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        emit("monitor", "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
