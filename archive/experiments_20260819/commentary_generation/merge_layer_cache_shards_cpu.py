#!/usr/bin/env python3
"""Validate eight completed layer-cache shards and publish a review gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORTS = REPO / "reports"
DATASET_SIZE = 3_256
EXPECTED_LAYERS = {
    "visual_frame_global": [None, 30, 1024],
    "layer_normalized": [None, 30, 1024],
    "temporal_output": [None, 30, 1024],
    "qformer_input": [None, 30, 1024],
    "qformer_output": [None, 32, 768],
    "projector_output": [None, 32, 4096],
    "dataset_indices": [None],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(REPORTS.resolve()):
        raise RuntimeError(f"Output root must be inside {REPORTS}: {root}")
    manifest_path = root / "manifest.json"
    review_path = root / "REVIEW_REQUIRED.md"
    if manifest_path.exists() or review_path.exists():
        raise FileExistsError("Refusing to overwrite the merged review gate")

    from safetensors import safe_open

    smoke = read_json(root / "smoke/result.json")
    if smoke.get("status") != "passed" or smoke.get("samples_completed") != 1:
        raise RuntimeError("Smoke test is not a one-sample pass")

    all_indices: list[int] = []
    shard_summaries: list[dict[str, Any]] = []
    prediction_count = 0
    for shard_id in range(args.num_shards):
        shard = root / f"shard_{shard_id:02d}_of_{args.num_shards:02d}"
        result = read_json(shard / "result.json")
        if result.get("status") != "passed":
            raise RuntimeError(f"Shard {shard_id} did not pass")
        cache = shard / "layers.safetensors"
        predictions = shard / "predictions.jsonl"
        with safe_open(cache, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != set(EXPECTED_LAYERS):
                raise RuntimeError(f"Shard {shard_id} tensor keys changed: {keys}")
            observed_shapes = {
                name: list(handle.get_slice(name).get_shape()) for name in keys
            }
            count = observed_shapes["dataset_indices"][0]
            for name, expected in EXPECTED_LAYERS.items():
                target = [count if value is None else value for value in expected]
                if observed_shapes[name] != target:
                    raise RuntimeError(
                        f"Shard {shard_id} shape mismatch for {name}: "
                        f"{observed_shapes[name]} != {target}"
                    )
            indices = handle.get_tensor("dataset_indices").tolist()
        expected_indices = list(range(shard_id, DATASET_SIZE, args.num_shards))
        if indices != expected_indices:
            raise RuntimeError(f"Shard {shard_id} index assignment changed")

        rows = [
            json.loads(line)
            for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row.get("dataset_index") for row in rows] != expected_indices:
            raise RuntimeError(f"Shard {shard_id} prediction identities changed")
        if any(not row.get("generated_commentary", "").strip() for row in rows):
            raise RuntimeError(f"Shard {shard_id} contains an empty generation")
        all_indices.extend(indices)
        prediction_count += len(rows)
        shard_summaries.append(
            {
                "shard_id": shard_id,
                "samples": count,
                "cache": str(cache),
                "cache_bytes": cache.stat().st_size,
                "predictions": str(predictions),
                "result": str(shard / "result.json"),
                "physical_gpu": result.get("physical_gpu"),
                "elapsed_seconds": result.get("elapsed_seconds"),
            }
        )

    if sorted(all_indices) != list(range(DATASET_SIZE)):
        raise RuntimeError("Merged shards do not cover each dataset index exactly once")
    if len(all_indices) != len(set(all_indices)) or prediction_count != DATASET_SIZE:
        raise RuntimeError("Merged shards contain duplicate or missing samples")

    manifest = {
        "schema_version": 1,
        "status": "review_required",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "full-development layer cache and baseline generation",
        "dataset_samples": DATASET_SIZE,
        "num_shards": args.num_shards,
        "index_coverage_exact": True,
        "smoke_result": str(root / "smoke/result.json"),
        "shards": shard_summaries,
        "training_executed": False,
        "backward_executed": False,
        "review_gate": {
            "reason": (
                "Choose the video-fact judging batch and the layer-probe/oracle "
                "candidates before any further GPU experiment."
            ),
            "automatic_continuation_allowed": False,
            "decoder_diagnostics_deferred": ["first_token_logits", "fact_token_nll"],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    review_path.write_text(
        "# 人工审核闸门\n\n"
        "自动阶段已经完成单视频 smoke test，以及 3,256 条开发数据的八卡逐层缓存和基线生成。\n\n"
        "此处必须停止。下一步需要人工确认：\n\n"
        "1. 抽查各分片的视频、reference、生成文本和张量身份；\n"
        "2. 冻结 Video Fact Record 与 Candidate-Grounding Review 的评审批次；\n"
        "3. 根据视频事实标签决定哪些层进入 match-grouped probe；\n"
        "4. 决定 first-token logits 和 fact-token NLL 只跑哪个诊断子集；\n"
        "5. 明确首轮 interface-matched oracle 条件后再申请新的 GPU 授权。\n\n"
        "本自动任务没有训练、backward、optimizer 或 checkpoint 写入。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "review_required",
                "samples": DATASET_SIZE,
                "manifest": str(manifest_path),
                "review": str(review_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
