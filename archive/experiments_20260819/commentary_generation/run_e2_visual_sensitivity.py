#!/usr/bin/env python3
"""E2: correct/shuffled/zero visual-prefix sensitivity on 200 samples."""

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
    DETERMINISTIC_BEAM,
    REPO,
    SAMPLE_COUNT,
    first_token_probabilities,
    generate_one,
    jensen_shannon,
    run_experiment,
    summarize_conditions,
)


OUTPUT_DIR = REPO / "reports/commentary_parallel_20260814/e2_visual_sensitivity_run1"
CONDITIONS = ["correct_prefix", "cyclic_shift_prefix", "zero_prefix"]


def execute(
    runtime: dict[str, Any], monitor: Any, result: dict[str, Any], handle: Any
) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    device = runtime["device"]
    prefixes = runtime["prefixes"]
    samples = runtime["manifest"]["samples"]
    rows: list[dict[str, Any]] = []
    timings = {name: 0.0 for name in CONDITIONS}
    js_shift: list[float] = []
    js_zero: list[float] = []
    first_token_shift_changes = 0
    first_token_zero_changes = 0
    monitor.set("e2_visual_sensitivity", f"0/{SAMPLE_COUNT}")
    for offset, sample in enumerate(samples):
        dataset_index = sample["dataset_index"]
        shifted_offset = (offset + 1) % SAMPLE_COUNT
        condition_prefixes = {
            "correct_prefix": prefixes[offset : offset + 1],
            "cyclic_shift_prefix": prefixes[shifted_offset : shifted_offset + 1],
            "zero_prefix": torch.zeros_like(prefixes[offset : offset + 1]),
        }
        batched_prefixes = torch.cat(
            [condition_prefixes[name] for name in CONDITIONS], dim=0
        ).to(device)
        monitor.progress = (
            f"{offset + 1}/{SAMPLE_COUNT} index={dataset_index} first_token"
        )
        with torch.inference_mode():
            diagnostic = first_token_probabilities(runtime, batched_prefixes)
        if "restricted_vocabulary_size" not in result:
            result["restricted_vocabulary_size"] = len(
                diagnostic["allowed_token_ids"]
            )
        probabilities = diagnostic["probabilities"]
        top_ids = diagnostic["top_token_ids"].detach().cpu().tolist()
        sample_js_shift = float(
            jensen_shannon(probabilities[0], probabilities[1]).item()
        )
        sample_js_zero = float(
            jensen_shannon(probabilities[0], probabilities[2]).item()
        )
        js_shift.append(sample_js_shift)
        js_zero.append(sample_js_zero)
        first_token_shift_changes += int(top_ids[0] != top_ids[1])
        first_token_zero_changes += int(top_ids[0] != top_ids[2])

        predictions: dict[str, Any] = {}
        per_condition_seconds: dict[str, float] = {}
        for name in CONDITIONS:
            monitor.progress = (
                f"{offset + 1}/{SAMPLE_COUNT} index={dataset_index} {name}"
            )
            prefix = condition_prefixes[name].to(device)
            started = time.monotonic()
            with torch.inference_mode():
                prediction = generate_one(runtime, prefix, DETERMINISTIC_BEAM)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            predictions[name] = prediction
            timings[name] += elapsed
            per_condition_seconds[name] = round(elapsed, 6)
            del prefix
        row = {
            "ordinal": offset + 1,
            "dataset_index": dataset_index,
            "video_path": sample["video_path"],
            "reference_commentary": sample["reference_commentary"],
            "prefix_sources": {
                "correct_prefix_dataset_index": dataset_index,
                "cyclic_shift_prefix_dataset_index": samples[shifted_offset][
                    "dataset_index"
                ],
                "zero_prefix": True,
            },
            "first_token_diagnostic": {
                "top_token_ids": dict(zip(CONDITIONS, top_ids)),
                "js_correct_vs_shifted": sample_js_shift,
                "js_correct_vs_zero": sample_js_zero,
            },
            "predictions": predictions,
            "generation_seconds": per_condition_seconds,
        }
        rows.append(row)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        result["samples_completed"] = offset + 1
        print(
            f"[SAMPLE_OK] {offset + 1}/{SAMPLE_COUNT} index={dataset_index}",
            flush=True,
        )
        del batched_prefixes
    monitor.set("compute_metrics")
    result["condition_summaries"] = summarize_conditions(rows, CONDITIONS)
    result["generation_seconds"] = {
        key: round(value, 6) for key, value in timings.items()
    }
    result["visual_sensitivity"] = {
        "mean_js_correct_vs_shifted": sum(js_shift) / len(js_shift),
        "mean_js_correct_vs_zero": sum(js_zero) / len(js_zero),
        "first_token_top1_change_correct_vs_shifted": first_token_shift_changes,
        "first_token_top1_change_correct_vs_zero": first_token_zero_changes,
        "full_text_change_correct_vs_shifted": sum(
            row["predictions"]["correct_prefix"]["text"]
            != row["predictions"]["cyclic_shift_prefix"]["text"]
            for row in rows
        ),
        "full_text_change_correct_vs_zero": sum(
            row["predictions"]["correct_prefix"]["text"]
            != row["predictions"]["zero_prefix"]["text"]
            for row in rows
        ),
    }
    return rows


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            experiment_id="E2_visual_prefix_sensitivity",
            output_dir=OUTPUT_DIR,
            execute=execute,
            protocol={
                "purpose": "Measure whether generation changes with visual conditioning.",
                "conditions": CONDITIONS,
                "baseline": "correct cached prefix in the same process",
                "negative_controls": "cyclic shift by one and all-zero prefix",
                "decode": DETERMINISTIC_BEAM,
                "first_token_metric": "Jensen-Shannon divergence over restricted vocabulary",
                "visual_prefix_forward_executed": False,
                "model_forward_scope": "decoder first-token diagnostic and generation",
            },
        )
    )
