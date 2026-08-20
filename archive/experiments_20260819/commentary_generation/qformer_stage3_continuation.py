#!/usr/bin/env python3
"""One-shot Stage-3 continuation gated by the attribute watcher result."""

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
REPORTS = (REPO / "reports").resolve()
RUNTIME = (REPO / ".runtime").resolve()


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
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Continuation:
    def __init__(self, config_path: Path, config: dict[str, Any], config_hash: str) -> None:
        self.config_path = config_path.resolve()
        self.config = config
        self.config_hash = config_hash
        self.paths = {
            name: Path(value).resolve() for name, value in config["paths"].items()
        }
        self.python = Path(config["python"]).resolve()
        self.state_dir = Path(config["state_dir"]).resolve()
        self.state_path = self.state_dir / "state.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.log_path = self.state_dir / "supervisor.log"
        self.arm_path = self.state_dir / "ARMED.json"
        self.consumed_path = self.state_dir / "CONSUMED.json"
        self.lock_path = self.state_dir / "supervisor.lock"
        source = config["source_watcher"]
        self.source_config = Path(source["config"]).resolve()
        self.source_state = Path(source["state"]).resolve()

    def validate(self, require_authority: bool) -> None:
        if Path.cwd().resolve() != REPO:
            raise RuntimeError(f"Run from {REPO}")
        if self.config.get("schema_version") != 1:
            raise RuntimeError("Unsupported config schema")
        if self.config.get("host") != socket.gethostname():
            raise RuntimeError("Host identity changed")
        if require_authority:
            if self.config.get("status") != "authorized_once_stage3_continuation":
                raise PermissionError("Continuation config is not authorized")
            if self.config.get("execution_authorized") is not True:
                raise PermissionError("Continuation execution is disabled")
        else:
            if self.config.get("status") not in {
                "static_unarmed", "authorized_once_stage3_continuation"
            }:
                raise RuntimeError("Unexpected static status")
        if not self.python.is_file():
            raise FileNotFoundError(self.python)
        if not self.state_dir.is_relative_to(RUNTIME):
            raise RuntimeError("state_dir escapes .runtime")
        for name in ("cpu_preparer", "gpu_worker", "pilot_design_config"):
            if not self.paths[name].is_file():
                raise FileNotFoundError(self.paths[name])
        identity_paths = {
            "cpu_preparer": self.paths["cpu_preparer"],
            "gpu_worker": self.paths["gpu_worker"],
            "continuation": Path(__file__).resolve(),
            "pilot_design_config": self.paths["pilot_design_config"],
            "decoder_runtime": Path(
                self.config["readonly_assets"]["decoder_runtime"]
            ).resolve(),
        }
        expected_hashes = self.config.get("source_sha256", {})
        if set(expected_hashes) != set(identity_paths):
            raise RuntimeError("Source identity set changed")
        for name, path in identity_paths.items():
            if sha256(path) != expected_hashes[name]:
                raise RuntimeError(f"Source identity changed: {name}")
        for name in ("stage1_manifest", "stage2_layer_result", "blind_manifest", "video_only_labels"):
            if not self.paths[name].is_file():
                raise FileNotFoundError(self.paths[name])
        if not self.source_config.is_file():
            raise FileNotFoundError(self.source_config)
        if sha256(self.source_config) != self.config["source_watcher"]["config_sha256"]:
            raise RuntimeError("Source watcher config identity changed")
        output_names = (
            "steering_artifact", "steering_partial", "steering_manifest",
            "predictions", "predictions_partial", "result", "review_packet",
        )
        for name in output_names:
            path = self.paths[name]
            if not path.is_relative_to(REPORTS):
                raise RuntimeError(f"Output escapes reports: {path}")
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {path}")
        expected_availability = {
            "candidate_gpu_indices": list(range(8)),
            "minimum_free_mib": 70000,
            "maximum_utilization_percent": 5,
            "allowed_compute_pids": [],
            "confirmation_delay_seconds": 60,
            "poll_seconds": 1800,
            "selection": "lowest_stable_eligible_physical_index",
        }
        if self.config["availability"] != expected_availability:
            raise RuntimeError("GPU availability policy changed")
        expected_scope = [
            "wait_for_exact_attribute_watcher_stage3_ready_state",
            "run_one_cpu_fold_local_residual_preparation",
            "wait_for_one_twice_confirmed_idle_h800",
            "run_one_48_clip_four_condition_inference_only_generation",
            "stop_at_review_required",
        ]
        if self.config["automatic_scope_if_later_armed"] != expected_scope:
            raise RuntimeError("Automatic scope changed")
        forbidden = set(self.config["explicitly_forbidden"])
        required = {
            "visual_encoder_forward", "qformer_forward", "main_model_training",
            "backward", "optimizer_or_scheduler", "checkpoint_write_or_overwrite",
            "automatic_retry", "gpu_coexistence_with_any_compute_pid",
            "holdout_evaluation", "automatic_result_claim_or_next_stage",
        }
        if not required.issubset(forbidden):
            raise RuntimeError("Forbidden-action contract changed")

    def environment(self, gpu: int | None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config["environment"])
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
        if gpu is not None:
            env["QFORMER_STAGE3_GPU_APPROVED"] = "YES"
        else:
            env.pop("QFORMER_STAGE3_GPU_APPROVED", None)
        return env

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
        atomic_json(self.state_path, {
            "schema_version": 1,
            "config_sha256": self.config_hash,
            "phase": phase,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        })

    def run_check_child(self, script: Path) -> dict[str, Any]:
        result = subprocess.run(
            [str(self.python), str(script), "--config", str(self.config_path), "--check"],
            cwd=REPO,
            env=self.environment(None),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

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
                self.emit("child_heartbeat", name=name, elapsed_seconds=round(elapsed, 1))
                time.sleep(30)
        code = process.wait()
        self.emit("child_exit", name=name, exit_code=code)
        return code

    def source_phase(self) -> tuple[str, dict[str, Any]]:
        if not self.source_state.is_file():
            return "waiting", {"reason": "source_state_absent"}
        state = json.loads(self.source_state.read_text(encoding="utf-8"))
        if state.get("config_sha256") != self.config["source_watcher"]["config_sha256"]:
            raise RuntimeError("Source watcher state/config identity mismatch")
        phase = state.get("phase")
        if phase == self.config["source_watcher"]["required_phase"]:
            checks = state.get("readiness", {}).get("checks", {})
            if not checks or not all(checks.values()):
                raise RuntimeError("Source stage3_ready lacks passing checks")
            if state.get("stage3_executed") is not False:
                raise RuntimeError("Source watcher reports Stage-3 execution")
            if not self.paths["attribute_probe_result"].is_file():
                raise FileNotFoundError(self.paths["attribute_probe_result"])
            attribute = json.loads(
                self.paths["attribute_probe_result"].read_text(encoding="utf-8")
            )
            if attribute.get("status") != "completed":
                raise RuntimeError("Attribute result is not completed")
            return "ready", state
        if phase in {"failed", "stage2_review_required"}:
            return "terminal_not_ready", state
        return "waiting", state

    def snapshot(self) -> dict[str, Any] | None:
        gpu_command = [
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        process_command = [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            gpu_result = subprocess.run(gpu_command, check=True, capture_output=True, text=True, timeout=30)
            process_result = subprocess.run(process_command, check=True, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            self.emit("nvidia_smi_failed", error=repr(error))
            return None
        processes = []
        for row in csv.reader(process_result.stdout.splitlines(), skipinitialspace=True):
            if row:
                processes.append({
                    "gpu_uuid": row[0], "pid": int(row[1]),
                    "process_name": row[2], "used_memory_mib": int(row[3]),
                })
        policy = self.config["availability"]
        gpus = []
        for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
            if not row:
                continue
            active = [process for process in processes if process["gpu_uuid"] == row[1]]
            gpu = {
                "index": int(row[0]), "uuid": row[1], "name": row[2],
                "total_mib": int(row[3]), "used_mib": int(row[4]),
                "free_mib": int(row[5]), "utilization_percent": int(row[6]),
                "active_compute_pids": sorted(process["pid"] for process in active),
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
        stable = sorted(first_indices & {
            gpu["index"] for gpu in second["gpus"] if gpu["eligible"]
        })
        return stable[0] if stable else None

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
        self.validate(require_authority=True)
        if not self.arm_path.is_file():
            raise PermissionError("Missing ARMED.json")
        arm = json.loads(self.arm_path.read_text(encoding="utf-8"))
        if arm.get("config_sha256") != self.config_hash:
            raise PermissionError("Arm/config SHA-256 mismatch")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.emit("continuation_started", config_sha256=self.config_hash)
            while True:
                source_status, source = self.source_phase()
                if source_status == "terminal_not_ready":
                    self.save_state("not_triggered", source_phase=source.get("phase"))
                    self.consume_arm("not_triggered")
                    return 0
                if source_status == "waiting":
                    self.save_state("waiting_for_attribute_stage3_ready")
                    time.sleep(self.config["source_watcher"]["poll_seconds"])
                    continue
                break

            self.save_state("preparing_residuals_cpu")
            code = self.run_child(
                "prepare_residuals_cpu",
                [str(self.python), str(self.paths["cpu_preparer"]),
                 "--config", str(self.config_path), "--run"],
                self.environment(None), self.state_dir / "prepare_cpu.log",
                self.config["timeouts_seconds"]["cpu_prepare"],
            )
            if code != 0:
                self.save_state("failed", stage="prepare_residuals_cpu", exit_code=code)
                self.consume_arm("failed")
                return code or 1

            while True:
                self.save_state("waiting_for_one_idle_h800")
                gpu = self.confirmed_gpu()
                if gpu is not None:
                    break
                time.sleep(self.config["availability"]["poll_seconds"])
            self.save_state("running_stage3_generation", physical_gpu=gpu)
            code = self.run_child(
                "stage3_generation",
                [str(self.python), str(self.paths["gpu_worker"]),
                 "--config", str(self.config_path), "--run"],
                self.environment(gpu), self.state_dir / "generation_gpu.log",
                self.config["timeouts_seconds"]["gpu_generation"],
            )
            if code != 0:
                self.save_state("failed", stage="stage3_generation", exit_code=code)
                self.consume_arm("failed")
                return code or 1
            result = json.loads(self.paths["result"].read_text(encoding="utf-8"))
            required_false = (
                "visual_forward_executed", "qformer_forward_executed",
                "training_executed", "backward_executed", "optimizer_created",
                "scheduler_created", "checkpoint_written",
            )
            if not (
                result.get("status") == "review_required"
                and result.get("samples_completed") == 48
                and all(result.get(name) is False for name in required_false)
                and self.paths["predictions"].is_file()
                and self.paths["review_packet"].is_file()
            ):
                self.save_state("failed", stage="stage3_output_contract")
                self.consume_arm("failed")
                return 1
            self.save_state(
                "review_required", physical_gpu=gpu,
                result=str(self.paths["result"]),
                review_packet=str(self.paths["review_packet"]),
            )
            self.consume_arm("review_required")
            return 0


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config, config_hash = load_config(config_path)
    continuation = Continuation(config_path, config, config_hash)
    if args.check:
        continuation.validate(require_authority=False)
        cpu = continuation.run_check_child(continuation.paths["cpu_preparer"])
        gpu = continuation.run_check_child(continuation.paths["gpu_worker"])
        print(json.dumps({
            "status": "static_ready_unarmed",
            "config_sha256": config_hash,
            "execution_authorized": config.get("execution_authorized"),
            "cpu_preparer": cpu,
            "gpu_worker": gpu,
            "nvidia_smi_queried": False,
        }))
        return 0
    if args.arm:
        continuation.validate(require_authority=True)
        if not args.authorization_note:
            raise ValueError("--authorization-note is required")
        continuation.state_dir.mkdir(parents=True, exist_ok=True)
        if continuation.arm_path.exists() or continuation.consumed_path.exists():
            raise FileExistsError("Continuation is already armed or consumed")
        atomic_json(continuation.arm_path, {
            "schema_version": 1,
            "config_sha256": config_hash,
            "authorized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "authorization_note": args.authorization_note,
        })
        print(json.dumps({"status": "armed", "config_sha256": config_hash}))
        return 0
    return continuation.run()


if __name__ == "__main__":
    raise SystemExit(main())
