#!/usr/bin/env python3
"""Classify each SNGS-10004 track from all frozen views in one Qwen prompt."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_qwen_role_swap_sngs10004_gpu import (
    DEFAULT_MODEL,
    ROLES,
    atomic_json,
    candidate_role_scores,
    crop_exact,
    generated_token_scores,
    heartbeat,
)


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004/input_manifest.json"
DEFAULT_OUTPUT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004/track_multiview_raw_predictions.json"
TRACK_PROMPT = (
    "These image pairs show the same tracked person at different times in one soccer match. "
    "For every time point, the first image is the full frame and the second image is the crop "
    "of the target person. Judge the person's role from the whole track, using recurring "
    "uniform, action, and field context; an unclear view should not override clearer views. "
    "Respond ONLY with one word in ['player', 'referee', 'goalkeeper', 'other']. "
    "Use 'other' only when the track is not a soccer player, referee, or goalkeeper, or when "
    "the target is not a person."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def build_track_message(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    content: list[dict[str, Any]] = []
    opened = []
    for view, sample in enumerate(samples, start=1):
        full = Image.open(sample["image_path_read_only"]).convert("RGB")
        crop = crop_exact(full, sample["bbox_ltwh"])
        opened.extend((full, crop))
        content.extend(
            [
                {"type": "text", "text": f"Time point {view}, full frame:"},
                {"type": "image", "image": full},
                {"type": "text", "text": f"Time point {view}, target crop:"},
                {"type": "image", "image": crop},
            ]
        )
    content.append({"type": "text", "text": TRACK_PROMPT})
    return [{"role": "user", "content": content}], opened


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one physical GPU must be exposed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["manual_annotations_read"] is not False or manifest["tracks"] != 49:
        raise AssertionError("Manifest isolation/track contract mismatch")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in manifest["sample_manifest"]:
        grouped.setdefault(int(sample["track_id"]), []).append(sample)
    for samples in grouped.values():
        samples.sort(key=lambda row: row["view_ordinal"])
    if len(grouped) != 49:
        raise AssertionError("Expected 49 grouped tracks")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.time()
    progress = {"phase": "model_load", "completed": 0, "total": 49}
    stop = Event()
    Thread(target=heartbeat, args=(stop, progress, started), daemon=True).start()
    properties = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "event": "start",
                "visible_physical_gpu": visible,
                "device_name": properties.name,
                "tracks": 49,
                "frozen_views": manifest["samples"],
            }
        ),
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        use_fast=False,
        max_pixels=(1920 * 1080 // 16) // (28 * 28) * (28 * 28),
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    progress["phase"] = "track_multiview_inference"
    predictions = []
    with torch.inference_mode():
        for completed, (track_id, samples) in enumerate(sorted(grouped.items()), start=1):
            message, opened = build_track_message(samples)
            messages = [message]
            text = processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            outputs = model.generate(
                **inputs,
                num_beams=1,
                temperature=None,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs.sequences[:, input_length:]
            raw_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            normalized = raw_text.strip().lower()
            predictions.append(
                {
                    "track_id": track_id,
                    "sample_ids": [sample["sample_id"] for sample in samples],
                    "views": len(samples),
                    "image_inputs": 2 * len(samples),
                    "raw_output_text": raw_text,
                    "normalized_parsed_role": normalized if normalized in ROLES else None,
                    "candidate_role_scores": candidate_role_scores(
                        model, processor, inputs, outputs.scores[0]
                    )[0],
                    "generation_score": generated_token_scores(
                        outputs, input_length, processor
                    )[0],
                }
            )
            for image in opened:
                image.close()
            progress["completed"] = completed
            print(
                json.dumps(
                    {
                        "event": "track_complete",
                        "completed": completed,
                        "total": 49,
                        "track_id": track_id,
                        "elapsed_seconds": round(time.time() - started, 1),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                ),
                flush=True,
            )

    if len(predictions) != 49 or any(row["normalized_parsed_role"] is None for row in predictions):
        raise AssertionError("Prediction count or parse contract failed")
    progress["phase"] = "artifact_write"
    result = {
        "status": "inference_complete_evaluation_required",
        "schema_version": 1,
        "experiment": "SNGS-10004 same-track multi-view Qwen role classification",
        "manifest": str(args.manifest),
        "model_read_only": str(args.model),
        "manual_annotations_read": False,
        "training_used": False,
        "prompt": TRACK_PROMPT,
        "visible_physical_gpu": visible,
        "logical_device": "cuda:0",
        "device_name": properties.name,
        "tracks": 49,
        "frozen_views": manifest["samples"],
        "elapsed_seconds": float(time.time() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "predictions": predictions,
    }
    atomic_json(args.output, result)
    stop.set()
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
