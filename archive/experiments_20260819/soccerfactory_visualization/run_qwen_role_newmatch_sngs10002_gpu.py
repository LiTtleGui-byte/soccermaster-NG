#!/usr/bin/env python3
"""Run label-blind single-view Qwen role inference on SNGS-10002."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Event, Thread

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_qwen_role_swap_sngs10004_gpu import (
    ROLES,
    atomic_json,
    build_messages,
    candidate_role_scores,
    generated_token_scores,
    heartbeat,
)


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reports/g10/20260819_qwen_role_newmatch_sngs10002/input_manifest.json"
DEFAULT_OUTPUT = REPO / "reports/g10/20260819_qwen_role_newmatch_sngs10002/raw_predictions.json"
DEFAULT_MODEL = Path(
    "/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/"
    "jn/Qwen2.5-VL-7B-Instruct"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one physical GPU must be exposed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["sample_manifest"]
    if manifest.get("video_id") != "10002" or manifest.get("tracks") != 33 or len(samples) != 89:
        raise AssertionError("Unexpected SNGS-10002 manifest contract")
    if manifest.get("manual_annotations_read") is not False or manifest.get("historical_role_fields_read_by_gpu") is not False:
        raise AssertionError("Label-isolation contract is missing")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.time()
    progress = {"phase": "model_load", "completed": 0, "total": len(samples)}
    stop = Event()
    Thread(target=heartbeat, args=(stop, progress, started), daemon=True).start()
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"event": "start", "visible_physical_gpu": visible, "device_name": props.name, "samples": 89, "batch_size": args.batch_size}), flush=True)

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

    predictions = []
    progress["phase"] = "inference"
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            messages, opened = build_messages(batch, manifest["prompt"])
            texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
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
            raw_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            role_scores = candidate_role_scores(model, processor, inputs, outputs.scores[0])
            generation_scores = generated_token_scores(outputs, input_length, processor)
            for sample, raw_text, scores, generation_score in zip(batch, raw_texts, role_scores, generation_scores):
                normalized = raw_text.strip().lower()
                predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "video_id": sample["video_id"],
                        "track_id": sample["track_id"],
                        "view_ordinal": sample["view_ordinal"],
                        "raw_output_text": raw_text,
                        "normalized_parsed_role": normalized if normalized in ROLES else None,
                        "candidate_role_scores": scores,
                        "generation_score": generation_score,
                    }
                )
            for image in opened:
                image.close()
            progress["completed"] = min(start + len(batch), len(samples))
            print(json.dumps({"event": "heartbeat", "completed": progress["completed"], "total": len(samples), "elapsed_seconds": round(time.time() - started, 1), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}), flush=True)

    if len(predictions) != 89 or [row["sample_id"] for row in predictions] != [row["sample_id"] for row in samples]:
        raise AssertionError("Prediction identity contract failed")
    result = {
        "status": "inference_complete_independent_labels_required",
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "manifest": str(args.manifest),
        "model_read_only": str(args.model),
        "manual_annotations_read": False,
        "historical_role_fields_read": False,
        "training_used": False,
        "dtype": "bfloat16",
        "visible_physical_gpu": visible,
        "device_name": props.name,
        "batch_size": args.batch_size,
        "samples": len(samples),
        "elapsed_seconds": time.time() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "predictions": predictions,
    }
    progress["phase"] = "artifact_write"
    atomic_json(args.output, result)
    stop.set()
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
