#!/usr/bin/env python3
"""Conditionally run a bounded commentary diagnostic and stop at review."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORTS = REPO / "reports"
RUNTIME = REPO / ".runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--arm", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--authorization-note")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise TypeError("Config must be a JSON object")
    return config, hashlib.sha256(raw).hexdigest()


class Supervisor:
    def __init__(self, config: dict[str, Any], config_hash: str) -> None:
        self.config = config
        self.config_hash = config_hash
        self.state_dir = Path(config["state_dir"]).resolve()
        self.output_root = Path(config["output_root"]).resolve()
        self.python = Path(config["python"]).resolve()
        self.worker = Path(config["worker"]).resolve()
        self.merger = Path(config["merger"]).resolve()
        self.state_path = self.state_dir / "state.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.log_path = self.state_dir / "supervisor.log"
        self.arm_path = self.state_dir / "ARMED.json"

    def validate(self, require_outputs_absent: bool = True) -> None:
        if Path.cwd().resolve() != REPO:
            raise RuntimeError(f"Run from {REPO}")
        if self.config.get("schema_version") != 1:
            raise RuntimeError("Unsupported config schema")
        if self.config.get("host") != socket.gethostname():
            raise RuntimeError(
                f"Config is for {self.config.get('host')}, current host is "
                f"{socket.gethostname()}"
            )
        if not self.output_root.is_relative_to(REPORTS.resolve()):
            raise RuntimeError("output_root must be inside reports/")
        if not self.state_dir.is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("state_dir must be inside .runtime/")
        for path in (self.python, self.worker, self.merger):
            if not path.exists():
                raise FileNotFoundError(path)
        smoke_policy = self.config["availability"]["smoke"]
        full_policy = self.config["availability"]["full_eight_shards"]
        if smoke_policy["required_gpu_indices"] != [1]:
            raise RuntimeError("The approved smoke GPU must be physical GPU 1")
        if smoke_policy["minimum_free_mib"] != 30000:
            raise RuntimeError("The approved smoke threshold is 30000 MiB")
        if smoke_policy["allowed_compute_pids"] != [3375814]:
            raise RuntimeError("The approved smoke coexistence PID changed")
        if full_policy["required_gpu_indices"] != list(range(8)):
            raise RuntimeError("The full stage requires physical GPU indices 0..7")
        if full_policy["minimum_free_mib"] != 60000:
            raise RuntimeError("The full-stage threshold must remain 60000 MiB")
        if self.config["poll_seconds"] != 1800:
            raise RuntimeError("The approved polling interval must be 1800 seconds")
        if full_policy.get("allowed_compute_pids") != []:
            raise RuntimeError("The eight-card stage permits no compute process")
        if self.config["automatic_scope"] != [
            "one_sample_smoke_inference",
            "eight_shard_full_development_cache_and_baseline_generation",
            "cpu_merge_and_review_gate",
        ]:
            raise RuntimeError("Automatic scope changed")
        if require_outputs_absent:
            occupied = [
                self.output_root / "smoke",
                *(self.output_root / f"shard_{i:02d}_of_08" for i in range(8)),
                self.output_root / "manifest.json",
                self.output_root / "REVIEW_REQUIRED.md",
            ]
            existing = [str(path) for path in occupied if path.exists()]
            if existing:
                raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            **payload,
        }
        message = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(message, flush=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def save_state(self, phase: str, **payload: Any) -> None:
        value = {
            "schema_version": 1,
            "config_sha256": self.config_hash,
            "phase": phase,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def environment(self, gpu: int) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config["environment"])
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def snapshot(self, policy_name: str) -> dict[str, Any] | None:
        gpu_command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        process_command = [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            gpu_result = subprocess.run(
                gpu_command, check=True, capture_output=True, text=True, timeout=30
            )
            process_result = subprocess.run(
                process_command, check=True, capture_output=True, text=True, timeout=30
            )
        except (subprocess.SubprocessError, OSError) as error:
            self.emit("nvidia_smi_failed", error=repr(error))
            return None

        gpus = []
        for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            gpus.append(
                {
                    "index": int(row[0]),
                    "uuid": row[1],
                    "name": row[2],
                    "total_mib": int(row[3]),
                    "used_mib": int(row[4]),
                    "free_mib": int(row[5]),
                    "utilization_percent": int(row[6]),
                }
            )
        processes = []
        for row in csv.reader(process_result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            processes.append(
                {
                    "gpu_uuid": row[0],
                    "pid": int(row[1]),
                    "process_name": row[2],
                    "used_memory_mib": int(row[3]),
                }
            )
        policy = self.config["availability"][policy_name]
        allowed_pids = set(policy["allowed_compute_pids"])
        for gpu in gpus:
            active_pids = {
                process["pid"]
                for process in processes
                if process["gpu_uuid"] == gpu["uuid"]
            }
            gpu["active_compute_pids"] = sorted(active_pids)
            gpu["unexpected_compute_pids"] = sorted(active_pids - allowed_pids)
            gpu["idle"] = (
                gpu["free_mib"] >= policy["minimum_free_mib"]
                and gpu["utilization_percent"] <= policy["maximum_utilization_percent"]
                and not gpu["unexpected_compute_pids"]
            )
        snapshot = {"policy": policy_name, "gpus": gpus, "processes": processes}
        self.emit("nvidia_smi_snapshot", **snapshot)
        return snapshot

    def confirmed_idle(self, policy_name: str, required_count: int) -> list[int] | None:
        policy = self.config["availability"][policy_name]
        required_indices = set(policy["required_gpu_indices"])
        first = self.snapshot(policy_name)
        if first is None:
            return None
        first_idle = {
            gpu["index"] for gpu in first["gpus"] if gpu["idle"]
        } & required_indices
        if len(first_idle) < required_count:
            return None
        delay = policy["confirmation_delay_seconds"]
        self.emit("availability_confirmation_wait", seconds=delay)
        time.sleep(delay)
        second = self.snapshot(policy_name)
        if second is None:
            return None
        second_idle = {
            gpu["index"] for gpu in second["gpus"] if gpu["idle"]
        } & required_indices
        stable = sorted(first_idle & second_idle)
        if len(stable) < required_count:
            self.emit(
                "availability_not_stable",
                required_count=required_count,
                stable_idle=stable,
            )
            return None
        return stable

    def run_child(
        self,
        name: str,
        command: list[str],
        env: dict[str, str],
        log_path: Path,
        timeout_seconds: int,
    ) -> int:
        self.emit("child_start", name=name, command=command, log=str(log_path))
        started = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process: subprocess.Popen[str] | None = None
        with log_path.open("x", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > timeout_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        self.emit("child_timeout", name=name, elapsed_seconds=elapsed)
                        return 124
                    self.emit("child_heartbeat", name=name, elapsed_seconds=round(elapsed, 1))
                    time.sleep(30)
            except BaseException:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.emit("child_interrupted", name=name)
                raise
        if process is None:
            raise RuntimeError(f"Child {name} was not created")
        code = process.returncode
        self.emit(
            "child_exit",
            name=name,
            exit_code=code,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        return code

    def run_smoke(self, gpu: int) -> bool:
        policy = self.config["availability"]["smoke"]
        command = [
            str(self.python),
            str(self.worker),
            "--mode",
            "smoke",
            "--output-root",
            str(self.output_root),
            "--minimum-free-mib",
            str(policy["minimum_free_mib"]),
        ]
        for pid in policy["allowed_compute_pids"]:
            command.extend(("--allow-existing-pid", str(pid)))
        code = self.run_child(
            "smoke",
            command,
            self.environment(gpu),
            self.state_dir / "smoke.log",
            self.config["smoke_timeout_seconds"],
        )
        if code != 0:
            self.save_state("failed", failed_stage="smoke", exit_code=code)
            return False
        result = json.loads(
            (self.output_root / "smoke/result.json").read_text(encoding="utf-8")
        )
        passed = result.get("status") == "passed" and result.get("samples_completed") == 1
        if not passed:
            self.save_state("failed", failed_stage="smoke_result_validation")
            return False
        self.save_state("waiting_for_all_eight", smoke_gpu=gpu)
        self.emit("smoke_passed", gpu=gpu)
        return True

    def run_eight_shards(self) -> bool:
        processes: list[tuple[int, subprocess.Popen[str], Any, Path]] = []
        started = time.monotonic()

        def stop_own_workers() -> None:
            for _, process, _, _ in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process, _, _ in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        try:
            for gpu in self.config["availability"]["full_eight_shards"]["required_gpu_indices"]:
                shard_log = self.state_dir / f"shard_{gpu:02d}.log"
                handle = shard_log.open("x", encoding="utf-8")
                command = [
                    str(self.python),
                    str(self.worker),
                    "--mode",
                    "shard",
                    "--shard-id",
                    str(gpu),
                    "--num-shards",
                    "8",
                    "--output-root",
                    str(self.output_root),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=self.environment(gpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((gpu, process, handle, shard_log))
                self.emit(
                    "shard_start",
                    shard_id=gpu,
                    gpu=gpu,
                    pid=process.pid,
                    log=str(shard_log),
                )
                time.sleep(self.config["shard_start_stagger_seconds"])
                if process.poll() not in (None, 0):
                    self.emit(
                        "shard_failed_during_stagger",
                        shard_id=gpu,
                        exit_code=process.returncode,
                    )
                    stop_own_workers()
                    self.save_state(
                        "failed",
                        failed_stage="eight_shards_startup",
                        shard_id=gpu,
                        exit_code=process.returncode,
                    )
                    return False

            while True:
                statuses = {gpu: process.poll() for gpu, process, _, _ in processes}
                failed = {gpu: code for gpu, code in statuses.items() if code not in (None, 0)}
                if failed:
                    self.emit("shard_failure_detected", failures=failed)
                    stop_own_workers()
                    self.save_state("failed", failed_stage="eight_shards", failures=failed)
                    return False
                if all(code == 0 for code in statuses.values()):
                    break
                elapsed = time.monotonic() - started
                if elapsed > self.config["shard_group_timeout_seconds"]:
                    self.emit("shard_group_timeout", elapsed_seconds=elapsed)
                    stop_own_workers()
                    self.save_state("failed", failed_stage="eight_shards_timeout")
                    return False
                self.emit("shard_group_heartbeat", elapsed_seconds=round(elapsed, 1), statuses=statuses)
                time.sleep(30)
        except BaseException:
            stop_own_workers()
            self.emit("shard_group_interrupted")
            raise
        finally:
            for _, _, handle, _ in processes:
                handle.close()
        self.emit("all_shards_passed")
        return True

    def merge_and_stop(self) -> bool:
        command = [
            str(self.python),
            str(self.merger),
            "--output-root",
            str(self.output_root),
            "--num-shards",
            "8",
        ]
        env = os.environ.copy()
        env.update(self.config["environment"])
        env["CUDA_VISIBLE_DEVICES"] = ""
        code = self.run_child(
            "cpu_merge",
            command,
            env,
            self.state_dir / "merge.log",
            self.config["merge_timeout_seconds"],
        )
        if code != 0:
            self.save_state("failed", failed_stage="cpu_merge", exit_code=code)
            return False
        self.save_state(
            "review_required",
            review_file=str(self.output_root / "REVIEW_REQUIRED.md"),
        )
        self.emit(
            "review_required",
            review_file=str(self.output_root / "REVIEW_REQUIRED.md"),
        )
        return True

    def run(self) -> int:
        self.validate(require_outputs_absent=True)
        if not self.config.get("execution_authorized", False):
            raise RuntimeError("Config is prepared but execution_authorized is false")
        arm = json.loads(self.arm_path.read_text(encoding="utf-8"))
        if arm.get("config_sha256") != self.config_hash:
            raise RuntimeError("ARMED.json does not match the current config")
        lock_path = self.state_dir / "LOCK"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another supervisor already holds the lock") from error
            self.save_state("waiting_for_one_gpu")
            self.emit("supervisor_started", config_sha256=self.config_hash)

            while True:
                idle = self.confirmed_idle(policy_name="smoke", required_count=1)
                if idle:
                    if not self.run_smoke(idle[0]):
                        return 1
                    break
                self.emit("waiting_for_one_gpu", next_check_seconds=self.config["poll_seconds"])
                time.sleep(self.config["poll_seconds"])

            while True:
                idle = self.confirmed_idle(
                    policy_name="full_eight_shards", required_count=8
                )
                if idle and idle == list(range(8)):
                    self.save_state("running_eight_shards")
                    if not self.run_eight_shards():
                        return 1
                    break
                self.emit("waiting_for_all_eight", next_check_seconds=self.config["poll_seconds"])
                time.sleep(self.config["poll_seconds"])

            return 0 if self.merge_and_stop() else 1


def main() -> int:
    args = parse_args()
    config, config_hash = load_config(args.config)
    supervisor = Supervisor(config, config_hash)
    if args.check:
        supervisor.validate(require_outputs_absent=True)
        print(
            json.dumps(
                {
                    "status": (
                        "authorized_not_armed"
                        if config.get("execution_authorized")
                        else "prepared_not_authorized"
                    ),
                    "config_sha256": config_hash,
                    "execution_authorized": config.get("execution_authorized"),
                    "automatic_scope": config.get("automatic_scope"),
                    "output_root": str(supervisor.output_root),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.arm:
        supervisor.validate(require_outputs_absent=True)
        if not config.get("execution_authorized", False):
            raise RuntimeError("Refusing to arm while execution_authorized is false")
        if not args.authorization_note:
            raise RuntimeError("--authorization-note is required")
        supervisor.state_dir.mkdir(parents=True, exist_ok=True)
        if supervisor.arm_path.exists():
            raise FileExistsError(f"Refusing to overwrite {supervisor.arm_path}")
        supervisor.arm_path.write_text(
            json.dumps(
                {
                    "config_sha256": config_hash,
                    "authorization_note": args.authorization_note,
                    "armed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"armed={supervisor.arm_path}")
        return 0
    try:
        return supervisor.run()
    except KeyboardInterrupt:
        supervisor.state_dir.mkdir(parents=True, exist_ok=True)
        supervisor.save_state("interrupted")
        supervisor.emit("supervisor_interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
