#!/usr/bin/env python3
"""Wait for one idle H800, run the 48-clip attribute probe once, and stop."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORTS = REPO / "reports"
RUNTIME = REPO / ".runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--arm", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--authorization-note")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve().read_bytes()
    value = json.loads(raw)
    return value, hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Supervisor:
    def __init__(self, config: dict[str, Any], config_hash: str) -> None:
        self.config = config
        self.config_hash = config_hash
        self.state_dir = Path(config["state_dir"]).resolve()
        self.state_path = self.state_dir / "state.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.log_path = self.state_dir / "supervisor.log"
        self.arm_path = self.state_dir / "ARMED.json"
        self.consumed_path = self.state_dir / "CONSUMED.json"
        self.lock_path = self.state_dir / "supervisor.lock"
        self.python = Path(config["python"]).resolve()
        self.gpu_worker = Path(config["gpu_worker"]).resolve()
        self.cpu_probe = Path(config["cpu_probe"]).resolve()
        self.outputs = {
            name: Path(path).resolve() for name, path in config["outputs"].items()
        }

    def validate(self, require_outputs_absent: bool) -> None:
        if Path.cwd().resolve() != REPO:
            raise RuntimeError(f"Run from {REPO}")
        if self.config.get("schema_version") != 1:
            raise RuntimeError("Unsupported config schema")
        if self.config.get("status") != "authorized_once_any_idle_h800":
            raise RuntimeError("Unexpected authorization status")
        if self.config.get("execution_authorized") is not True:
            raise RuntimeError("Execution is not authorized")
        if self.config.get("host") != socket.gethostname():
            raise RuntimeError("Host identity changed")
        if not self.state_dir.is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("state_dir must be inside .runtime")
        for path in (self.python, self.gpu_worker, self.cpu_probe):
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in self.outputs.values():
            if not path.is_relative_to(REPORTS.resolve()):
                raise RuntimeError(f"Output escapes reports: {path}")
        policy = self.config["availability"]
        if policy != {
            "candidate_gpu_indices": list(range(8)),
            "minimum_free_mib": 70000,
            "maximum_utilization_percent": 5,
            "allowed_compute_pids": [],
            "confirmation_delay_seconds": 60,
            "selection": "lowest_stable_eligible_physical_index",
        }:
            raise RuntimeError("GPU availability policy changed")
        if self.config["poll_seconds"] != 1800:
            raise RuntimeError("Polling interval changed")
        if self.config["automatic_scope"] != [
            "poll_all_eight_gpus_every_30_minutes",
            "one_48_clip_frozen_attribute_feature_extraction_on_one_idle_h800",
            "one_cpu_match_grouped_attribute_probe",
            "stage3_qformer_readiness_evaluation_and_stop",
        ]:
            raise RuntimeError("Automatic scope changed")
        if self.config["stage3_readiness"]["stage3_execution_available"] is not False:
            raise RuntimeError("Stage-3 execution must remain unavailable")
        required_forbidden = {
            "main_model_training", "backward", "checkpoint_write_or_overwrite",
            "automatic_retry", "gpu_coexistence_with_any_compute_pid",
            "stage3_gpu_execution", "oracle_intervention", "holdout_evaluation",
        }
        if not required_forbidden.issubset(self.config["explicitly_forbidden"]):
            raise RuntimeError("Forbidden-action contract changed")
        if not self.outputs["stage2_layer_result"].is_file():
            raise FileNotFoundError(self.outputs["stage2_layer_result"])
        if require_outputs_absent:
            occupied = [
                path for name, path in self.outputs.items()
                if name != "stage2_layer_result" and (path.exists() or path.is_symlink())
            ]
            if occupied:
                raise FileExistsError(f"Refusing to overwrite outputs: {occupied}")

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            **payload,
        }
        message = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(message, flush=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.events_path, self.log_path):
            with path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def save_state(self, phase: str, **payload: Any) -> None:
        atomic_json(
            self.state_path,
            {
                "schema_version": 1,
                "config_sha256": self.config_hash,
                "phase": phase,
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                **payload,
            },
        )

    def snapshot(self) -> dict[str, Any] | None:
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
        except (OSError, subprocess.SubprocessError) as error:
            self.emit("nvidia_smi_failed", error=repr(error))
            return None
        processes = []
        for row in csv.reader(process_result.stdout.splitlines(), skipinitialspace=True):
            if row:
                processes.append(
                    {
                        "gpu_uuid": row[0], "pid": int(row[1]),
                        "process_name": row[2], "used_memory_mib": int(row[3]),
                    }
                )
        policy = self.config["availability"]
        gpus = []
        for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            active = [p for p in processes if p["gpu_uuid"] == row[1]]
            gpu = {
                "index": int(row[0]), "uuid": row[1], "name": row[2],
                "total_mib": int(row[3]), "used_mib": int(row[4]),
                "free_mib": int(row[5]), "utilization_percent": int(row[6]),
                "active_compute_pids": sorted(p["pid"] for p in active),
            }
            gpu["eligible"] = (
                gpu["index"] in policy["candidate_gpu_indices"]
                and gpu["name"] == "NVIDIA H800"
                and gpu["free_mib"] >= policy["minimum_free_mib"]
                and gpu["utilization_percent"] <= policy["maximum_utilization_percent"]
                and not active
            )
            gpus.append(gpu)
        snapshot = {"gpus": gpus, "processes": processes}
        self.emit("nvidia_smi_snapshot", **snapshot)
        return snapshot

    def confirmed_gpu(self) -> int | None:
        first = self.snapshot()
        if first is None:
            return None
        first_indices = {gpu["index"] for gpu in first["gpus"] if gpu["eligible"]}
        if not first_indices:
            return None
        delay = self.config["availability"]["confirmation_delay_seconds"]
        self.emit("availability_confirmation_wait", seconds=delay)
        time.sleep(delay)
        second = self.snapshot()
        if second is None:
            return None
        second_indices = {gpu["index"] for gpu in second["gpus"] if gpu["eligible"]}
        stable = sorted(first_indices & second_indices)
        if not stable:
            self.emit("availability_not_stable")
            return None
        return stable[0]

    def environment(self, gpu: int | None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config["environment"])
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        return env

    def run_child(
        self, name: str, command: list[str], env: dict[str, str],
        log_path: Path, timeout_seconds: int,
    ) -> int:
        self.emit("child_start", name=name, command=command, log=str(log_path))
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    self.emit("child_timeout", name=name, elapsed_seconds=elapsed)
                    return 124
                self.emit(
                    "child_heartbeat", name=name,
                    elapsed_seconds=round(elapsed, 1),
                )
                time.sleep(30)
        code = process.wait()
        self.emit("child_exit", name=name, exit_code=code)
        return code

    def stage3_readiness(self) -> tuple[bool, dict[str, Any]]:
        attribute = json.loads(self.outputs["cpu_result"].read_text(encoding="utf-8"))
        layer = json.loads(
            self.outputs["stage2_layer_result"].read_text(encoding="utf-8")
        )
        attribute_representations = attribute["representations"]
        completed = max(
            value["completed_task_count"]
            for value in attribute_representations.values()
        )
        attribute_macro = attribute["architecture_screening_summary"]["macro_roc_auc"]
        best_attribute = max(
            value for value in attribute_macro.values() if value is not None
        )
        deltas = layer["screening_summary"]["adjacent_macro_delta"]
        q_delta = deltas["qformer_input->qformer_output"]
        projector_delta = deltas["qformer_output->projector_output"]
        temporal_delta = deltas["layer_normalized->temporal_output"]
        rule = self.config["stage3_readiness"]
        checks = {
            "attribute_completed_tasks": completed
            >= rule["minimum_attribute_completed_tasks"],
            "attribute_best_macro": best_attribute
            >= rule["minimum_best_attribute_macro_roc_auc"],
            "qformer_drop": q_delta
            <= rule["maximum_qformer_input_to_output_delta"],
            "projector_noncompeting": projector_delta
            >= rule["minimum_projector_delta"],
            "qformer_drop_twice_temporal": abs(q_delta)
            >= 2 * abs(min(temporal_delta, 0.0)),
        }
        evidence = {
            "checks": checks,
            "attribute_completed_tasks": completed,
            "attribute_best_macro_roc_auc": best_attribute,
            "qformer_delta": q_delta,
            "projector_delta": projector_delta,
            "temporal_delta": temporal_delta,
        }
        return all(checks.values()), evidence

    def consume_arm(self, outcome: str) -> None:
        if self.arm_path.is_file():
            record = json.loads(self.arm_path.read_text(encoding="utf-8"))
            record["consumed_at_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            record["outcome"] = outcome
            atomic_json(self.consumed_path, record)
            self.arm_path.unlink()

    def run(self) -> int:
        self.validate(require_outputs_absent=True)
        if not self.arm_path.is_file():
            raise PermissionError("Missing ARMED.json")
        arm = json.loads(self.arm_path.read_text(encoding="utf-8"))
        if arm.get("config_sha256") != self.config_hash:
            raise PermissionError("Arm/config SHA-256 mismatch")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.emit("supervisor_started", config_sha256=self.config_hash)
            while True:
                self.save_state("waiting_for_one_idle_h800")
                gpu = self.confirmed_gpu()
                if gpu is None:
                    self.emit(
                        "waiting_for_one_idle_h800",
                        next_check_seconds=self.config["poll_seconds"],
                    )
                    time.sleep(self.config["poll_seconds"])
                    continue
                self.validate(require_outputs_absent=True)
                self.save_state("running_attribute_extraction", physical_gpu=gpu)
                gpu_log = self.state_dir / "attribute_gpu.log"
                code = self.run_child(
                    "attribute_gpu",
                    [str(self.python), str(self.gpu_worker)],
                    self.environment(gpu), gpu_log,
                    self.config["gpu_timeout_seconds"],
                )
                if code != 0:
                    self.save_state("failed", stage="attribute_gpu", exit_code=code)
                    self.consume_arm("failed")
                    return code or 1
                summary = json.loads(
                    self.outputs["gpu_summary"].read_text(encoding="utf-8")
                )
                if not (
                    summary.get("status") == "completed"
                    and summary.get("sample_count") == 48
                    and self.outputs["features"].is_file()
                ):
                    self.save_state("failed", stage="attribute_output_contract")
                    self.consume_arm("failed")
                    return 1
                self.save_state("running_cpu_probe", physical_gpu=gpu)
                cpu_log = self.state_dir / "attribute_cpu.log"
                code = self.run_child(
                    "attribute_cpu",
                    [str(self.python), str(self.cpu_probe)],
                    self.environment(None), cpu_log,
                    self.config["cpu_timeout_seconds"],
                )
                if code != 0:
                    self.save_state("failed", stage="attribute_cpu", exit_code=code)
                    self.consume_arm("failed")
                    return code or 1
                ready, evidence = self.stage3_readiness()
                if ready:
                    self.save_state(
                        "stage3_ready",
                        physical_gpu=gpu,
                        readiness=evidence,
                        stage3_executed=False,
                        stop_reason=self.config["stage3_readiness"]["stop_reason"],
                    )
                    self.emit("stage3_ready_stop", evidence=evidence)
                    self.consume_arm("stage3_ready")
                else:
                    self.save_state(
                        "stage2_review_required",
                        physical_gpu=gpu,
                        readiness=evidence,
                        stage3_executed=False,
                    )
                    self.emit("stage2_review_required", evidence=evidence)
                    self.consume_arm("stage2_review_required")
                return 0


def main() -> int:
    args = parse_args()
    config, config_hash = load_config(args.config)
    supervisor = Supervisor(config, config_hash)
    if args.check:
        supervisor.validate(require_outputs_absent=True)
        print(json.dumps({"status": "ready", "config_sha256": config_hash}))
        return 0
    if args.arm:
        if not args.authorization_note:
            raise ValueError("--authorization-note is required when arming")
        supervisor.validate(require_outputs_absent=True)
        supervisor.state_dir.mkdir(parents=True, exist_ok=True)
        if supervisor.arm_path.exists() or supervisor.consumed_path.exists():
            raise FileExistsError("Autorun identity is already armed or consumed")
        atomic_json(
            supervisor.arm_path,
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "authorized_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "authorization_note": args.authorization_note,
            },
        )
        print(json.dumps({"status": "armed", "config_sha256": config_hash}))
        return 0
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
