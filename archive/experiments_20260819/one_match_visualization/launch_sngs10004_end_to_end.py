#!/usr/bin/env python3
"""Run the fresh CPU stage, wait for one idle H800, then run one GPU worker."""

from __future__ import annotations

import json
import argparse
import os
import selectors
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
LOCAL_LIB = REPO / ".local_envs/SoccerMaster-repro/lib"
TORCH_LIB = LOCAL_LIB / "python3.10/site-packages/torch/lib"
CPU_SCRIPT = REPO / "experiments/one_match_visualization/prepare_sngs10004_cpu.py"
GPU_SCRIPT = REPO / "experiments/one_match_visualization/run_soccermaster_five_heads_gpu.py"
REPORT = REPO / "reports/one_match/20260819_sngs10004_end_to_end"
RUNTIME = REPO / ".runtime/one_match/20260819_sngs10004_end_to_end"
RUN_LOG = REPORT / "run.log"
CPU_RESULT = RUNTIME / "cpu_result.json"
GPU_RESULT = REPORT / "soccermaster_gpu_result.json"
FINAL_RESULT = REPORT / "result.json"
SUMMARY = REPORT / "summary.md"
RESEARCH_LOG = REPO / "research_log.md"
LAB_NOTE = REPO / "lab_notes/2026-08/2026-08-19.md"

POLL_SECONDS = 60
HEARTBEAT_SECONDS = 30
CPU_TIMEOUT_SECONDS = 3600
GPU_TIMEOUT_SECONDS = 1800
MIN_FREE_MIB = 70000
MAX_UTIL_PERCENT = 5
EXPECTED_GPU_INDICES = set(range(8))
EXPECTED_HEADS = {
    "SoccerNetGSR_Detection",
    "LinesDetection",
    "KeypointsDetection",
    "VideoCaption",
    "CaptionClassification",
}


class Log:
    def __init__(self, path: Path, mode: str = "x") -> None:
        self.handle = path.open(mode, encoding="utf-8", buffering=1)

    def emit(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def child(self, line: str) -> None:
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


def base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPO),
            "LD_LIBRARY_PATH": f"{TORCH_LIB}:{LOCAL_LIB}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def stop_own_process_group(process: subprocess.Popen[str], log: Log) -> None:
    if process.poll() is not None:
        return
    log.emit(f"stopping owned process group pid={process.pid}")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.emit(f"owned process group pid={process.pid} did not exit; sending SIGKILL")
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_once(
    name: str,
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: int,
    log: Log,
) -> int:
    log.emit(f"starting {name}: {' '.join(command)} timeout={timeout_seconds}s")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("Child stdout pipe was not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    next_heartbeat = started + HEARTBEAT_SECONDS
    timed_out = False
    while process.poll() is None:
        for key, _ in selector.select(timeout=1):
            line = key.fileobj.readline()
            if line:
                log.child(line.rstrip("\n"))
        now = time.monotonic()
        if now >= next_heartbeat:
            log.emit(f"heartbeat stage={name} elapsed={now - started:.1f}s pid={process.pid}")
            next_heartbeat = now + HEARTBEAT_SECONDS
        if now - started >= timeout_seconds:
            timed_out = True
            stop_own_process_group(process, log)
            break
    remainder = process.stdout.read()
    if remainder:
        for line in remainder.splitlines():
            log.child(line)
    selector.close()
    return_code = process.wait()
    log.emit(
        f"finished {name} exit_code={return_code} elapsed={time.monotonic() - started:.1f}s timed_out={timed_out}"
    )
    return return_code


def csv_rows(command: list[str]) -> list[list[str]]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def gpu_snapshot(log: Log) -> tuple[dict[int, dict[str, Any]], float]:
    observed_at = time.monotonic()
    rows = csv_rows(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    cards: dict[int, dict[str, Any]] = {}
    for row in rows:
        if len(row) != 7:
            raise RuntimeError(f"Unexpected GPU row: {row}")
        index = int(row[0])
        cards[index] = {
            "index": index,
            "uuid": row[1],
            "name": row[2],
            "total_mib": int(row[3]),
            "used_mib": int(row[4]),
            "free_mib": int(row[5]),
            "util_percent": int(row[6]),
            "processes": [],
        }
    if set(cards) != EXPECTED_GPU_INDICES:
        raise RuntimeError(f"Expected GPU indices 0-7, got {sorted(cards)}")
    by_uuid = {card["uuid"]: card for card in cards.values()}
    process_rows = csv_rows(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in process_rows:
        if len(row) != 4:
            raise RuntimeError(f"Unexpected compute-process row: {row}")
        if row[0] not in by_uuid:
            raise RuntimeError(f"Compute process references unknown GPU UUID {row[0]}")
        by_uuid[row[0]]["processes"].append(
            {"pid": int(row[1]), "name": row[2], "used_mib": int(row[3])}
        )
    log.emit("GPU snapshot begin")
    for index in range(8):
        card = cards[index]
        processes = card["processes"] or "none"
        log.emit(
            f"GPU {index}: {card['name']} total={card['total_mib']} MiB used={card['used_mib']} MiB "
            f"free={card['free_mib']} MiB util={card['util_percent']}% compute_processes={processes}"
        )
    log.emit("GPU snapshot end")
    return cards, observed_at


def eligible(card: dict[str, Any]) -> bool:
    return (
        "H800" in card["name"]
        and card["free_mib"] >= MIN_FREE_MIB
        and card["util_percent"] <= MAX_UTIL_PERCENT
        and not card["processes"]
    )


def wait_for_gpu(log: Log) -> tuple[int, list[dict[str, Any]]]:
    previous: tuple[dict[int, dict[str, Any]], float] | None = None
    while True:
        cards, observed_at = gpu_snapshot(log)
        if previous is not None:
            previous_cards, previous_at = previous
            gap = observed_at - previous_at
            qualified = [
                index
                for index in range(8)
                if gap >= 60 and eligible(previous_cards[index]) and eligible(cards[index])
            ]
            if qualified:
                selected = min(qualified)
                log.emit(f"selected physical GPU {selected} after two eligible snapshots gap={gap:.1f}s")
                return selected, [previous_cards[selected], cards[selected]]
        previous = (cards, observed_at)
        wait_started = time.monotonic()
        while time.monotonic() - wait_started < POLL_SECONDS:
            remaining = POLL_SECONDS - (time.monotonic() - wait_started)
            sleep_for = min(HEARTBEAT_SECONDS, max(0.0, remaining))
            if sleep_for:
                time.sleep(sleep_for)
            if time.monotonic() - wait_started < POLL_SECONDS:
                log.emit("heartbeat stage=waiting_for_idle_h800")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def verify_images(paths: list[str]) -> None:
    from PIL import Image

    for value in paths:
        path = Path(value)
        with Image.open(path) as image:
            image.verify()


def append_actual_result(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n{heading}\n\n{body.rstrip()}\n")


def aggregate(selected_gpu: int, confirmations: list[dict[str, Any]], log: Log) -> None:
    if FINAL_RESULT.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite final result or summary")
    cpu = load_json(CPU_RESULT)
    gpu = load_json(GPU_RESULT)
    if cpu.get("status") != "passed" or gpu.get("status") != "passed":
        raise AssertionError("CPU or GPU stage did not pass")
    if set(gpu.get("heads", {})) != EXPECTED_HEADS:
        raise AssertionError(f"Five-head output incomplete: {set(gpu.get('heads', {}))}")
    cpu_visuals = list(cpu.get("visuals", []))
    gpu_visuals = list(gpu.get("visuals", {}).values())
    verify_images(cpu_visuals + gpu_visuals)
    result = {
        "status": "completed",
        "sequence": "SNGS-10004",
        "cpu_result": str(CPU_RESULT),
        "gpu_result": str(GPU_RESULT),
        "selected_physical_gpu": selected_gpu,
        "gpu_idle_confirmations": confirmations,
        "heads": sorted(EXPECTED_HEADS),
        "visuals": cpu_visuals + gpu_visuals,
        "images_opened_successfully": True,
        "training": False,
        "automatic_retries": 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_new(FINAL_RESULT, result)
    classification = gpu["heads"]["CaptionClassification"]["top5_probability"]
    retrieval = gpu["heads"]["VideoCaption"]["top5_cosine_similarity"]
    summary_text = (
        "# SNGS-10004 end-to-end result\n\n"
        "Status: completed. The fresh CPU stage and the single inference-only GPU worker both exited 0.\n\n"
        f"- Physical GPU: {selected_gpu}, selected after two qualifying H800 snapshots at least 60 seconds apart.\n"
        f"- Input: `{gpu['input_shape']}` `{gpu['dtype']}` from the first legal 30-frame SNGS-10004 clip.\n"
        f"- Five heads: {', '.join(sorted(EXPECTED_HEADS))}.\n"
        f"- CaptionClassification top-1: `{classification[0]['phrase']}` ({classification[0]['score']:.5f}).\n"
        f"- VideoCaption retrieval top-1: `{retrieval[0]['phrase']}` ({retrieval[0]['score']:.5f}); this is retrieval over 23 fixed event phrases, not generated commentary.\n"
        f"- Peak GPU allocated/reserved: {gpu['peak_gpu_memory_bytes']['allocated']} / {gpu['peak_gpu_memory_bytes']['reserved']} bytes.\n"
        "- All listed images were opened successfully. No retry occurred and no external process was altered.\n"
    )
    with SUMMARY.open("x", encoding="utf-8") as handle:
        handle.write(summary_text)
    timestamp = result["completed_at_utc"]
    research_body = (
        f"- 实际执行：SNGS-10004 CPU 准备与一次五头 inference-only GPU worker 均成功，物理 GPU {selected_gpu}。\n"
        f"- 结果变化：五个头齐全；分类 top-1 为 `{classification[0]['phrase']}`，VideoCaption 固定短语 retrieval top-1 为 `{retrieval[0]['phrase']}`。\n"
        f"- 去留决定：保留结果；VideoCaption 仅解释为 23 类短语检索，不解释为生成解说。证据：`{FINAL_RESULT.relative_to(REPO)}`。"
    )
    lab_body = (
        f"- 已确认（{timestamp}）：新鲜 CPU 阶段和单次 GPU worker 均退出 0；使用物理 GPU {selected_gpu}，五个 SoccerMaster 头齐全，全部完成图片可打开。\n"
        f"- 已确认：CaptionClassification top-1 为 `{classification[0]['phrase']}`；VideoCaption top-1 为 `{retrieval[0]['phrase']}`，其含义仅为固定 23 事件短语 retrieval。\n"
        f"- 风险：单个 30 帧 clip 不代表总体指标；未运行训练、自动重试或 holdout evaluation。\n"
        f"- 原始证据：`{FINAL_RESULT.relative_to(REPO)}`、`{RUN_LOG.relative_to(REPO)}`。"
    )
    append_actual_result(RESEARCH_LOG, "## SNGS-10004 单场五头可视化", research_body)
    append_actual_result(LAB_NOTE, "## SNGS-10004 单场五头端到端可视化", lab_body)
    log.emit(f"aggregation complete result={FINAL_RESULT} summary={SUMMARY}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "cpu-only", "cpu-continuation", "gpu-continuation"),
        default="full",
        help="cpu-only is the safe sandbox stage; gpu-continuation consumes its passed result without rerunning it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not PYTHON.is_file() or not CPU_SCRIPT.is_file() or not GPU_SCRIPT.is_file():
        raise FileNotFoundError("Required local Python or stage script is missing")
    if args.mode in {"full", "cpu-only"}:
        if REPORT.exists() or REPORT.is_symlink() or RUNTIME.exists() or RUNTIME.is_symlink():
            raise FileExistsError("Fresh report and runtime paths are required")
        REPORT.mkdir(parents=True)
        log = Log(RUN_LOG)
    elif args.mode == "cpu-continuation":
        if not RUN_LOG.is_file() or not REPORT.is_dir() or not RUNTIME.is_dir():
            raise FileNotFoundError("cpu-continuation requires preserved attempt-1 report/runtime")
        if CPU_RESULT.exists() or GPU_RESULT.exists() or FINAL_RESULT.exists() or SUMMARY.exists():
            raise FileExistsError("Continuation targets are not fresh")
        log = Log(RUN_LOG, "a")
    else:
        if not CPU_RESULT.is_file() or not RUN_LOG.is_file():
            raise FileNotFoundError("gpu-continuation requires the passed CPU result and existing run.log")
        if GPU_RESULT.exists() or FINAL_RESULT.exists() or SUMMARY.exists():
            raise FileExistsError("GPU/final output already exists; continuation never retries or overwrites")
        cpu_existing = load_json(CPU_RESULT)
        if cpu_existing.get("status") != "passed":
            raise AssertionError("Existing CPU result is not passed")
        log = Log(RUN_LOG, "a")
    try:
        if args.mode in {"full", "cpu-only", "cpu-continuation"}:
            cpu_environment = base_environment()
            cpu_environment["CUDA_VISIBLE_DEVICES"] = ""
            cpu_command = [str(PYTHON), str(CPU_SCRIPT)]
            if args.mode == "cpu-continuation":
                cpu_command.extend(["--mode", "continue-after-render-bug"])
            cpu_exit = run_once(
                "cpu_prepare_continuation" if args.mode == "cpu-continuation" else "cpu_prepare",
                cpu_command,
                cpu_environment,
                CPU_TIMEOUT_SECONDS,
                log,
            )
            if cpu_exit != 0 or not CPU_RESULT.is_file():
                log.emit("CPU stage failed; stopping without GPU polling")
                return 1
            if load_json(CPU_RESULT).get("status") != "passed":
                raise AssertionError("CPU worker exited zero without a passed result")
        if args.mode in {"cpu-only", "cpu-continuation"}:
            log.emit("CPU stage complete; paused before host GPU access, no GPU query or worker was attempted")
            return 0
        if args.mode == "gpu-continuation":
            log.emit("continuing from passed CPU result; CPU worker will not be rerun")
        selected_gpu, confirmations = wait_for_gpu(log)
        gpu_environment = base_environment()
        gpu_environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        gpu_exit = run_once(
            "gpu_worker",
            [str(PYTHON), str(GPU_SCRIPT)],
            gpu_environment,
            GPU_TIMEOUT_SECONDS,
            log,
        )
        if gpu_exit != 0 or not GPU_RESULT.is_file():
            log.emit("GPU worker failed; stopping without card change, repair, or retry")
            return 1
        aggregate(selected_gpu, confirmations, log)
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
