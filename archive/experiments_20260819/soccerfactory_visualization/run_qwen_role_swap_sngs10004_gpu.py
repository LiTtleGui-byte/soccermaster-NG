#!/usr/bin/env python3
"""Run the fixed SNGS-10004 Qwen role swap on one visible GPU.

The script is inference-only. It never reads manual annotations and writes one
atomic prediction JSON in the local report directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004/input_manifest.json"
DEFAULT_OUTPUT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004/raw_predictions.json"
DEFAULT_MODEL = Path(
    "/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/"
    "jn/Qwen2.5-VL-7B-Instruct"
)
ROLES = ("player", "referee", "goalkeeper", "other")


def heartbeat(stop: Event, progress: dict[str, Any], started: float) -> None:
    while not stop.wait(30):
        print(
            json.dumps(
                {
                    "event": "heartbeat",
                    "phase": progress["phase"],
                    "completed": progress["completed"],
                    "total": progress["total"],
                    "elapsed_seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def crop_exact(image: Image.Image, bbox_ltwh: list[float]) -> Image.Image:
    left, top, width, height = bbox_ltwh
    right, bottom = left + width, top + height
    left = max(0, min(round(left), image.width - 2))
    top = max(0, min(round(top), image.height - 2))
    right = max(left + 1, min(round(right), image.width - 1))
    bottom = max(top + 1, min(round(bottom), image.height - 1))
    return image.crop((left, top, right, bottom)).convert("RGB")


def build_messages(samples: list[dict[str, Any]], prompt: str) -> tuple[list[list[dict[str, Any]]], list[Image.Image]]:
    messages = []
    opened_images = []
    for sample in samples:
        full_image = Image.open(sample["image_path_read_only"]).convert("RGB")
        crop = crop_exact(full_image, sample["bbox_ltwh"])
        opened_images.extend((full_image, crop))
        messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_image},
                        {"type": "image", "image": crop},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
    return messages, opened_images


def candidate_role_scores(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    inputs: Any,
    first_step_scores: torch.Tensor,
) -> list[dict[str, float]]:
    """Score the exact candidate strings using autoregressive token likelihood."""
    token_ids = {
        role: processor.tokenizer.encode(role, add_special_tokens=False) for role in ROLES
    }
    if any(not ids for ids in token_ids.values()):
        raise AssertionError(f"Empty candidate tokenization: {token_ids}")
    first_log_probs = torch.log_softmax(first_step_scores.float(), dim=-1)
    batch_size = inputs.input_ids.shape[0]
    role_log_likelihoods: dict[str, torch.Tensor] = {}
    passthrough = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    for role, ids in token_ids.items():
        score = first_log_probs[:, ids[0]].clone()
        prefix: list[int] = [ids[0]]
        for next_token in ids[1:]:
            prefix_tensor = torch.tensor(prefix, device="cuda", dtype=inputs.input_ids.dtype)
            prefix_tensor = prefix_tensor.unsqueeze(0).expand(batch_size, -1)
            extended_ids = torch.cat((inputs.input_ids, prefix_tensor), dim=1)
            extended_mask = torch.cat(
                (
                    inputs.attention_mask,
                    torch.ones_like(prefix_tensor, dtype=inputs.attention_mask.dtype),
                ),
                dim=1,
            )
            output = model(
                input_ids=extended_ids,
                attention_mask=extended_mask,
                use_cache=False,
                return_dict=True,
                **passthrough,
            )
            next_log_probs = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
            score = score + next_log_probs[:, next_token]
            prefix.append(next_token)
        role_log_likelihoods[role] = score
    stacked = torch.stack([role_log_likelihoods[role] for role in ROLES], dim=1)
    probabilities = torch.softmax(stacked, dim=1).cpu()
    stacked = stacked.cpu()
    results = []
    for row in range(batch_size):
        record = {}
        for column, role in enumerate(ROLES):
            record[f"{role}_log_likelihood"] = float(stacked[row, column])
            record[f"{role}_normalized_probability"] = float(probabilities[row, column])
        results.append(record)
    return results


def generated_token_scores(outputs: Any, input_length: int, processor: AutoProcessor) -> list[dict[str, Any]]:
    generated_ids = outputs.sequences[:, input_length:]
    batch_size = generated_ids.shape[0]
    token_log_probs = [[] for _ in range(batch_size)]
    for step, step_scores in enumerate(outputs.scores):
        if step >= generated_ids.shape[1]:
            break
        ids = generated_ids[:, step]
        log_probs = torch.log_softmax(step_scores.float(), dim=-1)
        selected = log_probs.gather(1, ids.unsqueeze(1)).squeeze(1).cpu()
        for row in range(batch_size):
            token_id = int(ids[row].item())
            if token_id in processor.tokenizer.all_special_ids:
                continue
            token_log_probs[row].append(float(selected[row]))
    records = []
    for values in token_log_probs:
        records.append(
            {
                "non_special_generated_tokens": len(values),
                "sum_log_probability": float(sum(values)),
                "mean_log_probability": float(sum(values) / len(values)) if values else None,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Exactly one physical GPU must be exposed with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("manual_annotations_read") is not False:
        raise AssertionError("Manifest label-isolation contract is missing")
    samples = manifest["sample_manifest"]
    if manifest["tracks"] != 49 or len(samples) != manifest["samples"]:
        raise AssertionError("Unexpected manifest contract")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started = time.time()
    progress = {"phase": "model_load", "completed": 0, "total": len(samples)}
    heartbeat_stop = Event()
    Thread(target=heartbeat, args=(heartbeat_stop, progress, started), daemon=True).start()
    device_properties = torch.cuda.get_device_properties(0)
    print(
        json.dumps(
            {
                "event": "start",
                "visible_physical_gpu": visible,
                "logical_device": "cuda:0",
                "device_name": device_properties.name,
                "total_memory_bytes": device_properties.total_memory,
                "samples": len(samples),
                "batch_size": args.batch_size,
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

    predictions = []
    progress["phase"] = "inference"
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            messages, opened_images = build_messages(batch_samples, manifest["prompt"])
            texts = [
                processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=texts,
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
            raw_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            role_scores = candidate_role_scores(
                model, processor, inputs, outputs.scores[0]
            )
            generation_scores = generated_token_scores(outputs, input_length, processor)
            for sample, raw_text, candidate_scores, generation_score in zip(
                batch_samples, raw_texts, role_scores, generation_scores
            ):
                strict_text = raw_text.lower()
                normalized_text = raw_text.strip().lower()
                predictions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "video_id": sample["video_id"],
                        "track_id": sample["track_id"],
                        "view_ordinal": sample["view_ordinal"],
                        "raw_output_text": raw_text,
                        "strict_parsed_role": strict_text if strict_text in ROLES else None,
                        "normalized_parsed_role": (
                            normalized_text if normalized_text in ROLES else None
                        ),
                        "candidate_role_scores": candidate_scores,
                        "generation_score": generation_score,
                    }
                )
            for image in opened_images:
                image.close()
            completed = min(start + len(batch_samples), len(samples))
            progress["completed"] = completed
            print(
                json.dumps(
                    {
                        "event": "heartbeat",
                        "completed": completed,
                        "total": len(samples),
                        "elapsed_seconds": round(time.time() - started, 1),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                ),
                flush=True,
            )

    if len(predictions) != len(samples):
        raise AssertionError("Prediction count mismatch")
    if [row["sample_id"] for row in predictions] != [row["sample_id"] for row in samples]:
        raise AssertionError("Prediction identity/order mismatch")
    result = {
        "status": "inference_complete_evaluation_required",
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "manifest": str(args.manifest),
        "model_read_only": str(args.model),
        "manual_annotations_read": False,
        "training_used": False,
        "dtype": "bfloat16",
        "visible_physical_gpu": visible,
        "logical_device": "cuda:0",
        "device_name": device_properties.name,
        "batch_size": args.batch_size,
        "samples": len(samples),
        "elapsed_seconds": float(time.time() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "prediction_contract": {
            "primary_equivalent_output": "normalized_parsed_role from deterministic generation",
            "diagnostic_scores": (
                "Autoregressive log likelihood of each exact candidate role string, "
                "normalized only across the four fixed candidates."
            ),
            "score_warning": (
                "Candidate strings have different token lengths; these values are "
                "model sequence likelihoods, not calibrated role probabilities."
            ),
        },
        "predictions": predictions,
    }
    progress["phase"] = "artifact_write"
    atomic_json(args.output, result)
    heartbeat_stop.set()
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
