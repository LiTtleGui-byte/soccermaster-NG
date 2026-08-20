#!/usr/bin/env python3
"""E3: isolate attention-mask and PAD/EOS behavior on cached prefixes."""

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
    generate_one,
    run_experiment,
    summarize_conditions,
)


OUTPUT_DIR = REPO / "reports/commentary_parallel_20260814/e3_mask_pad_run1"
ARMS: dict[str, dict[str, Any]] = {
    "historical_implicit_mask_pad": {
        "attention_mask": False,
        "pad_token_id": None,
        "eos_token_id": None,
    },
    "explicit_all_ones_mask": {
        "attention_mask": True,
        "pad_token_id": None,
        "eos_token_id": None,
    },
    "explicit_mask_eot_pad_eos": {
        "attention_mask": True,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009],
    },
}


def execute(
    runtime: dict[str, Any], monitor: Any, result: dict[str, Any], handle: Any
) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    device = runtime["device"]
    prefixes = runtime["prefixes"]
    samples = runtime["manifest"]["samples"]
    rows: list[dict[str, Any]] = []
    timings = {name: 0.0 for name in ARMS}
    monitor.set("e3_mask_pad", f"0/{SAMPLE_COUNT}")
    for offset, sample in enumerate(samples):
        dataset_index = sample["dataset_index"]
        prefix = prefixes[offset : offset + 1].to(device)
        predictions: dict[str, Any] = {}
        per_arm_seconds: dict[str, float] = {}
        for name, arm in ARMS.items():
            monitor.progress = (
                f"{offset + 1}/{SAMPLE_COUNT} index={dataset_index} {name}"
            )
            started = time.monotonic()
            with torch.inference_mode():
                prediction = generate_one(
                    runtime,
                    prefix,
                    DETERMINISTIC_BEAM,
                    **arm,
                )
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            predictions[name] = prediction
            timings[name] += elapsed
            per_arm_seconds[name] = round(elapsed, 6)
        row = {
            "ordinal": offset + 1,
            "dataset_index": dataset_index,
            "video_path": sample["video_path"],
            "reference_commentary": sample["reference_commentary"],
            "predictions": predictions,
            "generation_seconds": per_arm_seconds,
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
    result["condition_summaries"] = summarize_conditions(rows, list(ARMS))
    result["generation_seconds"] = {
        key: round(value, 6) for key, value in timings.items()
    }
    historical = "historical_implicit_mask_pad"
    mask_only = "explicit_all_ones_mask"
    aligned = "explicit_mask_eot_pad_eos"
    result["pairwise_differences"] = {
        "text_changed_mask_only_vs_historical": sum(
            row["predictions"][mask_only]["text"]
            != row["predictions"][historical]["text"]
            for row in rows
        ),
        "tokens_changed_mask_only_vs_historical": sum(
            row["predictions"][mask_only]["token_ids"]
            != row["predictions"][historical]["token_ids"]
            for row in rows
        ),
        "text_changed_aligned_vs_historical": sum(
            row["predictions"][aligned]["text"]
            != row["predictions"][historical]["text"]
            for row in rows
        ),
        "tokens_changed_aligned_vs_historical": sum(
            row["predictions"][aligned]["token_ids"]
            != row["predictions"][historical]["token_ids"]
            for row in rows
        ),
    }
    return rows


if __name__ == "__main__":
    raise SystemExit(
        run_experiment(
            experiment_id="E3_attention_mask_pad_eos",
            output_dir=OUTPUT_DIR,
            execute=execute,
            protocol={
                "purpose": "Separate mask inference from PAD/EOS alignment.",
                "arms": ARMS,
                "decode": DETERMINISTIC_BEAM,
                "historical_model_config_eos_token_id": 128009,
                "historical_generation_eos_token_ids": [128001, 128009],
                "historical_generation_pad_token_id": None,
                "historical_effective_auto_pad_token_id": 128001,
                "training_dataset_padding_token_id": 128001,
                "post_decode_truncation_token_id": 128001,
                "visual_prefix_forward_executed": False,
                "model_forward_scope": "decoder generation only",
            },
        )
    )
