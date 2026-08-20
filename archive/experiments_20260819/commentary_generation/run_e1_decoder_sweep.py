#!/usr/bin/env python3
"""E1: decoder Pareto sweep on the fixed 200 cached visual prefixes."""

from __future__ import annotations

import json
import os
import time
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from research.experiments.commentary_generation.cached_prefix_experiments import (  # noqa: E402
    GENERATION_SEED,
    HISTORICAL_GENERATION,
    REPO,
    SAMPLE_COUNT,
    generate_one,
    run_experiment,
    seed_everything,
    summarize_conditions,
)


OUTPUT_DIR = REPO / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1"
COMMON_NUCLEUS: dict[str, Any] = {
    "max_new_tokens": 128,
    "num_beams": 1,
    "do_sample": True,
    "min_length": 5,
    "repetition_penalty": 1.0,
    "length_penalty": 1,
    "renormalize_logits": True,
}
STRATEGIES: dict[str, dict[str, Any]] = {
    "historical_beam_sampling": dict(HISTORICAL_GENERATION),
    "nucleus_t070_p090": {
        **COMMON_NUCLEUS,
        "temperature": 0.70,
        "top_p": 0.90,
    },
    "nucleus_t085_p090": {
        **COMMON_NUCLEUS,
        "temperature": 0.85,
        "top_p": 0.90,
    },
    "nucleus_t085_p095": {
        **COMMON_NUCLEUS,
        "temperature": 0.85,
        "top_p": 0.95,
    },
    "nucleus_t085_p090_rep105": {
        **COMMON_NUCLEUS,
        "temperature": 0.85,
        "top_p": 0.90,
        "repetition_penalty": 1.05,
    },
}


def execute(
    runtime: dict[str, Any], monitor: Any, result: dict[str, Any], handle: Any
) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    np = runtime["np"]
    device = runtime["device"]
    prefixes = runtime["prefixes"]
    samples = runtime["manifest"]["samples"]
    rows: list[dict[str, Any]] = []
    timings = {name: 0.0 for name in STRATEGIES}
    monitor.set("e1_decoder_sweep", f"0/{SAMPLE_COUNT}")
    for offset, sample in enumerate(samples):
        dataset_index = sample["dataset_index"]
        prefix = prefixes[offset : offset + 1].to(device)
        predictions: dict[str, Any] = {}
        per_strategy_seconds: dict[str, float] = {}
        for name, settings in STRATEGIES.items():
            monitor.progress = (
                f"{offset + 1}/{SAMPLE_COUNT} index={dataset_index} {name}"
            )
            sample_seed = GENERATION_SEED + dataset_index
            seed_everything(sample_seed, torch, np)
            started = time.monotonic()
            with torch.inference_mode():
                prediction = generate_one(runtime, prefix, settings)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            predictions[name] = prediction
            timings[name] += elapsed
            per_strategy_seconds[name] = round(elapsed, 6)
        row = {
            "ordinal": offset + 1,
            "dataset_index": dataset_index,
            "sample_seed": GENERATION_SEED + dataset_index,
            "video_path": sample["video_path"],
            "reference_commentary": sample["reference_commentary"],
            "predictions": predictions,
            "generation_seconds": per_strategy_seconds,
        }
        rows.append(row)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        result["samples_completed"] = offset + 1
        print(
            f"[SAMPLE_OK] {offset + 1}/{SAMPLE_COUNT} index={dataset_index}",
            flush=True,
        )
        del prefix
    monitor.set("compute_metrics")
    result["condition_summaries"] = summarize_conditions(rows, list(STRATEGIES))
    result["generation_seconds"] = {
        key: round(value, 6) for key, value in timings.items()
    }
    return rows


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            experiment_id="E1_decoder_pareto_sweep",
            output_dir=OUTPUT_DIR,
            execute=execute,
            protocol={
                "purpose": "Find a diversity/overlap decoding trade-off.",
                "conditions": STRATEGIES,
                "same_process_historical_baseline": True,
                "visual_prefix_forward_executed": False,
                "model_forward_scope": "decoder generation only",
            },
        )
    )
