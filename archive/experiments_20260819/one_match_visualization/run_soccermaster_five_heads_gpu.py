#!/usr/bin/env python3
"""Run the five SoccerMaster heads on one prepared SNGS-10004 clip."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
OPS_BUILD = REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
CONFIG_PATH = REPO / (
    "configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
MODEL_PATH = Path(os.environ.get(
    "SOCCERMASTER_SIGLIP2_PATH",
    "/remote-home/haolinyang/sports/Soccer-Backbone/pretrained_models/"
    "google/siglip2-large-patch16-512",
))
CHECKPOINT_DIR = Path(os.environ.get(
    "SOCCERMASTER_CHECKPOINT_DIR",
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19",
))

REPORT = Path(os.environ.get(
    "SOCCERMASTER_FIVE_HEAD_REPORT_DIR",
    str(REPO / "reports/one_match/20260819_sngs10004_end_to_end"),
))
VISUALS = REPORT / "visuals"
RESULT = REPORT / "soccermaster_gpu_result.json"
IMAGE_PATHS = {
    "SoccerNetGSR_Detection": VISUALS / "06_soccermaster_detection_overlay.jpg",
    "LinesDetection": VISUALS / "07_soccermaster_lines_heatmap_overlay.jpg",
    "KeypointsDetection": VISUALS / "08_soccermaster_keypoints_heatmap_overlay.jpg",
    "CaptionClassification": VISUALS / "09_soccermaster_caption_classification_top5.png",
    "VideoCaption": VISUALS / "10_soccermaster_video_caption_retrieval_top5.png",
}

NUM_FRAMES = 30
IMAGE_SIZE = 512
EVENT_PHRASES = [
    "var",
    "end of half game",
    "clearance",
    "second yellow card",
    "injury",
    "ball possession",
    "throw in",
    "show added time",
    "shot off target",
    "start of half game",
    "substitution",
    "saved by goal-keeper",
    "red card",
    "lead to corner",
    "ball out of play",
    "off side",
    "goal",
    "penalty",
    "yellow card",
    "foul lead to penalty",
    "corner",
    "free kick",
    "foul with no card",
]
EXPECTED_HEADS = set(IMAGE_PATHS)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OPS_BUILD))
os.chdir(REPO)


def write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_image_new(image: Any, path: Path, image_format: str) -> None:
    with path.open("xb") as handle:
        if image_format == "JPEG":
            image.save(handle, format=image_format, quality=92)
        else:
            image.save(handle, format=image_format)


def load_config(data_view: Path) -> dict[str, Any]:
    from soccermaster.config import load_super_config, yaml_to_dict

    config = load_super_config(
        yaml_to_dict(str(CONFIG_PATH)), str(DEFAULT_CONFIG_PATH)
    )
    extra_stub = data_view / "SN-GSR-2024/SoccerNetGS/extracted_info.pkl"
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            "DEVICE": "cpu",
            "CKPT_TYPE": "soccer_master",
            "LOAD_CHECKPOINTS": False,
            "LOAD_HEADS": True,
            "CKPT_PATH": str(MODEL_PATH),
            "TEXT_ENCODER_CKPT_PATH": str(MODEL_PATH),
            "STAGE_1_CKPT_DIR": str(CHECKPOINT_DIR),
            "DATA_ROOT": str(data_view),
            "USE_EXTRA_DATA": True,
            "EXTRA_DATA_ONLY": True,
            "EXTRA_DATA_PATH": str(extra_stub),
            "USE_EXTRA_DATA_AMOUNT": -1,
            "AUG_ENABLE_TRAINING_AUGMENTATION": False,
            "NUM_WORKERS": 0,
            "BATCH_SIZE": 1,
            "DATASETS_TO_HEADS": {
                "SoccerNetGSR_Detection": [
                    "SoccerNetGSR_Detection",
                    "LinesDetection",
                    "KeypointsDetection",
                ],
                "VideoCaption": ["VideoCaption", "CaptionClassification"],
            },
        }
    )
    if config["NUM_FRAMES"] != NUM_FRAMES:
        raise AssertionError(f"Unexpected NUM_FRAMES={config['NUM_FRAMES']}")
    if config["AUG_RANDOM_RESIZE"] != [IMAGE_SIZE]:
        raise AssertionError(f"Unexpected resize={config['AUG_RANDOM_RESIZE']}")
    return config


def denormalized_frame(torch: Any, images: Any, frame_index: int) -> Any:
    frame = images[0, frame_index].detach().cpu().float()
    return frame.mul(0.5).add(0.5).clamp(0, 1)


def pil_frame(torch: Any, image_class: Any, images: Any, frame_index: int) -> Any:
    frame = denormalized_frame(torch, images, frame_index)
    array = frame.mul(255).byte().permute(1, 2, 0).numpy()
    return image_class.fromarray(array)


def flatten_detection(torch: Any, output: dict[str, Any]) -> dict[str, Any]:
    flattened = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 3:
            flattened[key] = value.detach().reshape(NUM_FRAMES, *value.shape[2:])
        else:
            flattened[key] = value
    return flattened


def detection_visual(
    torch: Any,
    image_class: Any,
    image_draw: Any,
    images: Any,
    output: dict[str, Any],
    postprocessor: Any,
    frame_index: int,
) -> tuple[Any, dict[str, Any]]:
    target_sizes = torch.tensor(
        [[IMAGE_SIZE, IMAGE_SIZE]] * NUM_FRAMES,
        device=output["pred_logits"].device,
        dtype=torch.long,
    )
    predictions = postprocessor.postprocess(flatten_detection(torch, output), target_sizes)
    selected = predictions[frame_index]
    image = pil_frame(torch, image_class, images, frame_index)
    draw = image_draw.Draw(image)
    kept = []
    for box, score in zip(selected["boxes"], selected["scores"]):
        score_value = float(score.detach().cpu())
        if score_value <= 0.5:
            continue
        coords = [float(value) for value in box.detach().cpu()]
        draw.rectangle(coords, outline=(255, 50, 50), width=3)
        draw.text((coords[0] + 2, max(0, coords[1] - 14)), f"{score_value:.2f}", fill=(255, 255, 0))
        kept.append({"score": score_value, "box_xyxy_512": coords})
    return image, {
        "representative_frame_zero_based": frame_index,
        "score_threshold": 0.5,
        "boxes_over_threshold": len(kept),
        "top_boxes": kept[:10],
        "pred_logits_shape": list(output["pred_logits"].shape),
        "pred_boxes_shape": list(output["pred_boxes"].shape),
    }


def color_heatmap(np: Any, heatmap: Any) -> Any:
    values = heatmap.detach().cpu().float().numpy()
    values = values - values.min()
    maximum = float(values.max())
    if maximum > 0:
        values = values / maximum
    red = np.clip(2.0 * values, 0, 1)
    green = np.clip(2.0 * (1.0 - np.abs(values - 0.5)), 0, 1)
    blue = np.clip(2.0 * (1.0 - values), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype("uint8")


def heatmap_visual(
    torch: Any,
    np: Any,
    image_class: Any,
    base: Any,
    tensor: Any,
    frame_index: int,
    title: str,
) -> tuple[Any, dict[str, Any]]:
    activation = torch.sigmoid(tensor[0, frame_index]).amax(dim=0)
    colored = image_class.fromarray(color_heatmap(np, activation)).resize(
        base.size, image_class.Resampling.BILINEAR
    )
    overlay = image_class.blend(base, colored, 0.42)
    canvas = image_class.new("RGB", (base.width * 2, base.height + 42), "#101827")
    canvas.paste(colored, (0, 42))
    canvas.paste(overlay, (base.width, 42))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"{title}: max-channel heatmap | overlay", fill="white")
    return canvas, {
        "shape": list(tensor.shape),
        "representative_frame_zero_based": frame_index,
        "activation_min": float(activation.min().detach().cpu()),
        "activation_mean": float(activation.mean().detach().cpu()),
        "activation_max": float(activation.max().detach().cpu()),
    }


def bar_chart(image_class: Any, image_draw: Any, title: str, rows: list[dict[str, Any]]) -> Any:
    width, height = 1120, 520
    image = image_class.new("RGB", (width, height), "#101827")
    draw = image_draw.Draw(image)
    draw.text((24, 20), title, fill="white")
    maximum = max(abs(float(row["score"])) for row in rows) or 1.0
    for rank, row in enumerate(rows, start=1):
        y = 72 + (rank - 1) * 82
        score = float(row["score"])
        bar_width = int(620 * abs(score) / maximum)
        draw.text((24, y), f"{rank}. {row['phrase']}", fill="#dbeafe")
        draw.rectangle((420, y, 420 + bar_width, y + 25), fill="#3b82f6")
        draw.text((1050, y), f"{score:.5f}", fill="white", anchor="ra")
    return image


def top_five(torch: Any, values: Any, probability: bool) -> list[dict[str, Any]]:
    scores = torch.softmax(values, dim=-1) if probability else values
    top_values, top_indices = torch.topk(scores, 5)
    return [
        {"rank": rank, "index": int(index), "phrase": EVENT_PHRASES[int(index)], "score": float(value)}
        for rank, (value, index) in enumerate(
            zip(top_values.detach().cpu(), top_indices.detach().cpu()), start=1
        )
    ]


def main() -> int:
    started = time.monotonic()
    occupied = [path for path in (RESULT, *IMAGE_PATHS.values()) if path.exists() or path.is_symlink()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite: {occupied}")
    if not VISUALS.is_dir():
        raise FileNotFoundError(f"CPU stage did not create {VISUALS}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit() or "," in visible:
        raise RuntimeError("Exactly one physical GPU must be selected externally")

    from research.experiments.one_match_visualization.prepare_sngs10004_cpu import DATA_VIEW

    data_view = Path(os.environ.get("SOCCERMASTER_DATA_VIEW", str(DATA_VIEW)))
    source_images = data_view / "SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1"
    source_pkl = data_view / "SN-GSR-2024/SoccerNetGS/extracted_info/SNGS-10004.pkl"
    if not data_view.is_dir() or not source_images.is_dir() or not source_pkl.is_file():
        raise FileNotFoundError("Prepared SNGS-10004 data view is incomplete")

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    from soccermaster.data.soccernet_gsr_detection import build_gsr_detection_dataset, collate_fn
    from soccermaster.data.video_caption import keywords_list
    from soccermaster.models.build import build_metrics_fn
    from soccermaster.models.multi_task import MultiTaskingSigLIP

    if list(keywords_list) != EVENT_PHRASES:
        raise AssertionError("The model event vocabulary is not the fixed 23-phrase list")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    config = load_config(data_view)
    dataset = build_gsr_detection_dataset(config=config, split="train")
    if not dataset.sample_position or dataset.sample_position[0] != ("SNGS-10004", 0):
        raise AssertionError(f"Unexpected first legal clip: {dataset.sample_position[:1]}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    images_cpu = batch["images"]
    metas = batch["metas"]
    if tuple(images_cpu.shape) != (1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise AssertionError(f"Unexpected input shape {tuple(images_cpu.shape)}")
    if metas[0]["sequence"] != "SNGS-10004" or metas[0]["start_frame"] != 0:
        raise AssertionError(f"Unexpected clip metadata {metas[0]}")

    model = MultiTaskingSigLIP(config=config, logger=None)
    model.load_checkpoint(
        str(CHECKPOINT_DIR),
        ckpt_type="soccer_master",
        logger=None,
        load_heads=True,
    )
    model.to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    images = images_cpu.to(device)
    with torch.inference_mode():
        detection_outputs = model(
            images,
            "SoccerNetGSR_Detection",
            metas=metas,
            text=None,
        )
        caption_outputs = model(
            images,
            "VideoCaption",
            metas=metas,
            text=EVENT_PHRASES,
        )
    torch.cuda.synchronize(device)
    if set(detection_outputs) != {"SoccerNetGSR_Detection", "LinesDetection", "KeypointsDetection"}:
        raise AssertionError(f"Unexpected detection branch heads {set(detection_outputs)}")
    if set(caption_outputs) != {"VideoCaption", "CaptionClassification"}:
        raise AssertionError(f"Unexpected caption branch heads {set(caption_outputs)}")

    frame_index = 15
    metrics = build_metrics_fn(config=config)
    detection_image, detection_result = detection_visual(
        torch,
        Image,
        ImageDraw,
        images_cpu,
        detection_outputs["SoccerNetGSR_Detection"],
        metrics["SoccerNetGSR_Detection"],
        frame_index,
    )
    save_image_new(detection_image, IMAGE_PATHS["SoccerNetGSR_Detection"], "JPEG")

    base = pil_frame(torch, Image, images_cpu, frame_index)
    lines_image, lines_result = heatmap_visual(
        torch,
        np,
        Image,
        base,
        detection_outputs["LinesDetection"]["pred_lines_heatmap"],
        frame_index,
        "SoccerMaster LinesDetection",
    )
    save_image_new(lines_image, IMAGE_PATHS["LinesDetection"], "JPEG")
    keypoints_image, keypoints_result = heatmap_visual(
        torch,
        np,
        Image,
        base,
        detection_outputs["KeypointsDetection"]["pred_keypoints_heatmap"],
        frame_index,
        "SoccerMaster KeypointsDetection",
    )
    save_image_new(keypoints_image, IMAGE_PATHS["KeypointsDetection"], "JPEG")

    logits = caption_outputs["CaptionClassification"]["logits"][0]
    classification_top5 = top_five(torch, logits, probability=True)
    classification_image = bar_chart(
        Image,
        ImageDraw,
        "CaptionClassification: 23-class probability top-5",
        classification_top5,
    )
    save_image_new(classification_image, IMAGE_PATHS["CaptionClassification"], "PNG")

    video_caption = caption_outputs["VideoCaption"]
    similarities = video_caption["base_similarity_matrix"]
    if tuple(similarities.shape) != (1, len(EVENT_PHRASES)):
        raise AssertionError(f"Unexpected retrieval matrix shape {tuple(similarities.shape)}")
    retrieval_top5 = top_five(torch, similarities[0], probability=False)
    retrieval_image = bar_chart(
        Image,
        ImageDraw,
        "VideoCaption retrieval (not generated commentary): cosine-similarity top-5",
        retrieval_top5,
    )
    save_image_new(retrieval_image, IMAGE_PATHS["VideoCaption"], "PNG")

    result = {
        "status": "passed",
        "stage": "soccermaster_five_head_inference",
        "sequence": "SNGS-10004",
        "data_view": str(data_view),
        "clip": {
            "selection": "first legal 30-frame clip from the prepared DataLoader view",
            "start_frame_zero_based": int(metas[0]["start_frame"]),
            "end_frame_exclusive": int(metas[0]["end_frame"]),
        },
        "config": str(CONFIG_PATH),
        "checkpoint": str(CHECKPOINT_DIR),
        "checkpoint_type": "soccer_master",
        "dtype": str(images.dtype),
        "input_shape": list(images.shape),
        "device": {
            "visible_physical_index": int(visible),
            "logical": str(device),
            "name": torch.cuda.get_device_name(device),
        },
        "heads": {
            "SoccerNetGSR_Detection": detection_result,
            "LinesDetection": lines_result,
            "KeypointsDetection": keypoints_result,
            "CaptionClassification": {
                "class_count": len(EVENT_PHRASES),
                "top5_probability": classification_top5,
                "logits_shape": list(caption_outputs["CaptionClassification"]["logits"].shape),
            },
            "VideoCaption": {
                "mode": "retrieval_over_fixed_23_event_phrases_not_generated_commentary",
                "top5_cosine_similarity": retrieval_top5,
                "similarity_shape": list(similarities.shape),
            },
        },
        "visuals": {name: str(path) for name, path in IMAGE_PATHS.items()},
        "peak_gpu_memory_bytes": {
            "allocated": int(torch.cuda.max_memory_allocated(device)),
            "reserved": int(torch.cuda.max_memory_reserved(device)),
        },
        "wall_seconds": round(time.monotonic() - started, 3),
        "training": False,
    }
    if set(result["heads"]) != EXPECTED_HEADS:
        raise AssertionError("Five-head result is incomplete")
    write_json_new(RESULT, result)
    print(f"[GPU_RESULT] passed result={RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
