#!/usr/bin/env python3
"""Create a CPU-only visual report for the fixed commentary sample.

This script decodes and preprocesses the same MatchTime sample used by
``infer_one.py``. It does not import the commentary model, load a checkpoint,
run forward/generate, create a DataLoader, or use a GPU.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


# Set safety and offline controls before importing ML/image-processing packages.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
SIGLIP2_ROOT = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
TEST_ANNOTATIONS = Path(
    "/remote-home/haolinyang/sports/UniSoccer/train_data/"
    "video_clip_json/MatchTime/classification_test.json"
)
TEST_VIDEO_ROOT = Path(
    "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/"
    "SoccerNetv2/MatchTime/SN-Caption-test-align"
)
EXPECTED_RELATIVE_VIDEO = Path(
    "europe_uefa-champions-league_2016-2017/"
    "2016-11-23 - 22-45 Arsenal 2 - 2 Paris SG/2_43_36.mp4"
)
EXPECTED_VIDEO = TEST_VIDEO_ROOT / EXPECTED_RELATIVE_VIDEO
INFERENCE_RESULT = REPO / "reports/audits/commentary_infer_one_result.json"
OUTPUT_PARENT = REPO / "reports/commentary_trace"
OUTPUT_DIR = OUTPUT_PARENT / "sample_000"

EXPECTED_REFERENCE = (
    "[PLAYER] ([TEAM]) gets on the end of a pass on the edge of the box but "
    "his shot is blocked."
)
EXPECTED_GENERATED = (
    "[PLAYER] ([TEAM]) sends a pass into the box, but the opposition's "
    "defence is alert to the danger and intercepts the ball."
)
EXPECTED_ANNOTATIONS_SIZE = 1_541_678
EXPECTED_VIDEO_SIZE = 5_167_597
EXPECTED_INFERENCE_RESULT_SIZE = 6_076
EXPECTED_VIDEO_FRAMES = 750
EXPECTED_FPS = 25.0
NUM_FRAMES = 30
SAMPLE_INDEX = 0
SAMPLE_MODE = "middle"


class Monitor:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stage = "preflight"
        self.stage_started = self.started
        self.stop = threading.Event()
        self.timings: dict[str, float] = {}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def enter(self, stage: str) -> None:
        now = time.monotonic()
        if self.stage:
            self.timings[self.stage] = round(now - self.stage_started, 3)
        self.stage = stage
        self.stage_started = now
        print(f"[STAGE] {stage}", flush=True)

    def finish(self) -> None:
        now = time.monotonic()
        if self.stage:
            self.timings[self.stage] = round(now - self.stage_started, 3)
        self.stop.set()
        self.thread.join(timeout=2)

    def _heartbeat(self) -> None:
        while not self.stop.wait(30):
            print(
                f"[HEARTBEAT] stage={self.stage} "
                f"elapsed={time.monotonic() - self.started:.1f}s",
                flush=True,
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Unexpected size for {path}: expected {expected_size}, "
            f"got {path.stat().st_size}"
        )


def frame_indices_middle(num_frames: int, video_length: int) -> list[int]:
    import numpy as np

    accepted = min(num_frames, video_length)
    intervals = np.linspace(0, video_length, num=accepted + 1).astype(int)
    ranges = [
        (start, intervals[index + 1] - 1)
        for index, start in enumerate(intervals[:-1])
    ]
    indices = [(start + end) // 2 for start, end in ranges]
    if len(indices) < num_frames:
        indices.extend([indices[-1]] * (num_frames - len(indices)))
    return indices


def fit_with_letterbox(image: Any, size: tuple[int, int], Image: Any) -> Any:
    target_width, target_height = size
    copy = image.copy()
    copy.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (32, 35, 42))
    left = (target_width - copy.width) // 2
    top = (target_height - copy.height) // 2
    canvas.paste(copy, (left, top))
    return canvas


def make_contact_sheet(
    frames_rgb: Any,
    indices: list[int],
    fps: float,
    output: Path,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
) -> None:
    columns = 5
    rows = math.ceil(len(indices) / columns)
    width, height = 318, 179
    label_height = 27
    gap = 8
    margin = 14
    canvas_width = margin * 2 + columns * width + (columns - 1) * gap
    canvas_height = margin * 2 + rows * (height + label_height + gap) - gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for position, (frame, frame_index) in enumerate(zip(frames_rgb, indices)):
        row, column = divmod(position, columns)
        left = margin + column * (width + gap)
        top = margin + row * (height + label_height + gap)
        image = Image.fromarray(frame).resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(image, (left, top))
        timestamp = frame_index / fps
        draw.rectangle(
            (left, top + height, left + width, top + height + label_height),
            fill=(245, 247, 250),
        )
        draw.text(
            (left + 5, top + height + 7),
            f"sample {position + 1:02d} | frame {frame_index:03d} | {timestamp:05.2f}s",
            fill=(20, 28, 40),
            font=font,
        )
    canvas.save(output, format="JPEG", quality=92, subsampling=0)


def make_preprocess_comparison(
    frames_rgb: Any,
    processed_rgb: Any,
    indices: list[int],
    fps: float,
    output: Path,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
) -> None:
    selected = [0, 7, 14, 22, 29]
    cell = 256
    columns = len(selected)
    title_height = 38
    label_height = 27
    gap = 9
    margin = 14
    canvas_width = margin * 2 + columns * cell + (columns - 1) * gap
    canvas_height = (
        margin * 2 + 2 * (title_height + cell + label_height) + gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    row_titles = [
        "Raw RGB: original 398x224 aspect ratio (letterboxed only for display)",
        "SigLIP2 input: resized to 512x512, then denormalized for display",
    ]
    for row, title in enumerate(row_titles):
        row_top = margin + row * (title_height + cell + label_height + gap)
        draw.text((margin, row_top + 10), title, fill=(18, 27, 44), font=font)
        image_top = row_top + title_height
        for column, position in enumerate(selected):
            left = margin + column * (cell + gap)
            if row == 0:
                image = fit_with_letterbox(
                    Image.fromarray(frames_rgb[position]),
                    (cell, cell),
                    Image,
                )
            else:
                image = Image.fromarray(processed_rgb[position]).resize(
                    (cell, cell),
                    Image.Resampling.LANCZOS,
                )
            canvas.paste(image, (left, image_top))
            timestamp = indices[position] / fps
            draw.rectangle(
                (left, image_top + cell, left + cell, image_top + cell + label_height),
                fill=(245, 247, 250),
            )
            draw.text(
                (left + 5, image_top + cell + 7),
                f"frame {indices[position]} | {timestamp:.2f}s",
                fill=(20, 28, 40),
                font=font,
            )
    canvas.save(output, format="JPEG", quality=92, subsampling=0)


def build_html(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    preprocess = manifest["preprocessing"]
    prior = manifest["prior_inference_result"]
    indices = manifest["sampling"]["frame_indices"]
    index_preview = ", ".join(str(value) for value in indices)

    stages = [
        ("1", "原始视频", "complete", "30秒真实比赛片段，路径与哈希已锁定。"),
        ("2", "均匀采样", "complete", "Decord middle 模式采样30帧。"),
        ("3", "SigLIP2预处理", "complete", "逐帧变为512×512并归一化到[-1, 1]。"),
        ("4", "视觉编码器", "pending", "下一阶段采集真实输出shape、范数和空间摘要。"),
        ("5", "时间编码与Q-Former", "pending", "下一阶段采集帧间关系和32个query摘要。"),
        ("6", "Llama投影", "pending", "下一阶段采集32×4096视觉embedding统计。"),
        ("7", "受限词表生成", "complete", "既有固定单样本运行已成功生成文本。"),
    ]
    stage_html = "\n".join(
        f'<div class="stage {css}"><div class="number">{number}</div>'
        f'<h3>{html.escape(title)}</h3><p>{html.escape(description)}</p></div>'
        for number, title, css, description in stages
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SoccerMaster 解说链路 · Sample 000</title>
  <style>
    :root {{ --ink:#152033; --muted:#637083; --line:#dce2ea; --ok:#147d64; --wait:#8a6d1d; }}
    body {{ margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#f4f7fa; }}
    main {{ max-width:1500px; margin:0 auto; padding:34px 26px 70px; }}
    h1 {{ margin:0 0 8px; font-size:32px; }} h2 {{ margin-top:38px; }}
    .subtitle {{ color:var(--muted); margin-bottom:26px; }}
    .pipeline {{ display:grid; grid-template-columns:repeat(7,minmax(145px,1fr)); gap:10px; }}
    .stage {{ background:white; border:1px solid var(--line); border-top:5px solid var(--wait); border-radius:10px; padding:14px; min-height:145px; }}
    .stage.complete {{ border-top-color:var(--ok); }} .stage.pending {{ background:#fbfaf5; }}
    .number {{ width:27px; height:27px; display:grid; place-items:center; border-radius:50%; background:#e9eef5; font-weight:700; }}
    .stage h3 {{ font-size:16px; margin:12px 0 7px; }} .stage p {{ color:var(--muted); font-size:13px; line-height:1.5; }}
    .card {{ background:white; border:1px solid var(--line); border-radius:12px; padding:20px; margin-top:16px; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:9px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ width:220px; color:var(--muted); }}
    code {{ overflow-wrap:anywhere; }} img {{ display:block; max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }}
    .caption-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .caption {{ background:#f7f9fb; border-radius:8px; padding:16px; line-height:1.6; }}
    .tag {{ display:inline-block; font-size:12px; border-radius:999px; padding:4px 9px; background:#e7f4ef; color:#10644f; }}
    .warning {{ border-left:5px solid #d49b27; background:#fff9e8; padding:14px 18px; border-radius:7px; }}
    @media (max-width:1100px) {{ .pipeline {{ grid-template-columns:repeat(2,1fr); }} .caption-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <span class="tag">CPU-only input trace · model not loaded</span>
  <h1>SoccerMaster 解说生成链路</h1>
  <p class="subtitle">固定 MatchTime 测试样本 000：先看清模型真正接收了什么。</p>

  <section class="pipeline">{stage_html}</section>

  <h2>1. 输入契约</h2>
  <div class="card"><table>
    <tr><th>源视频</th><td><code>{html.escape(source['video_path'])}</code></td></tr>
    <tr><th>视频属性</th><td>{source['frame_count']}帧 · {source['fps']:.2f} FPS · {source['duration_seconds']:.2f}秒 · {source['width']}×{source['height']}</td></tr>
    <tr><th>采样索引</th><td><code>{html.escape(index_preview)}</code></td></tr>
    <tr><th>预处理输出</th><td>{html.escape(str(preprocess['output_shape']))} · {html.escape(preprocess['dtype'])} · 范围 [{preprocess['min']:.3f}, {preprocess['max']:.3f}]</td></tr>
    <tr><th>本阶段设备</th><td>CPU；未加载模型/checkpoint，未执行forward/generate</td></tr>
  </table></div>

  <h2>2. 30帧时间轴</h2>
  <p>每个时间区间取中点。标签同时显示采样序号、原视频帧号和时间。</p>
  <a href="01_sampled_frames.jpg"><img src="01_sampled_frames.jpg" alt="30 sampled video frames"></a>

  <h2>3. SigLIP2预处理前后</h2>
  <p>上排保留398×224宽高比并以黑边展示；下排是实际送入视觉编码器的512×512方形输入，显示前已从[-1,1]反归一化。</p>
  <a href="02_preprocess_comparison.jpg"><img src="02_preprocess_comparison.jpg" alt="Raw and SigLIP2-preprocessed frame comparison"></a>

  <h2>4. 已有最终文本结果</h2>
  <div class="caption-grid">
    <div class="caption"><strong>参考解说</strong><br>{html.escape(prior['reference_commentary'])}</div>
    <div class="caption"><strong>模型生成</strong><br>{html.escape(prior['generated_commentary'])}</div>
  </div>

  <h2>5. 尚未采集的内部阶段</h2>
  <div class="warning">视觉编码器输出、时间特征、Q-Former query、Llama投影和逐token概率尚未在本报告中采集。它们需要一次独立、重新授权的GPU trace，不能由最终文本反推。</div>

  <p class="subtitle">机器可读证据见 <a href="manifest.json">manifest.json</a>。</p>
</main></body></html>
"""


def run() -> int:
    monitor = Monitor()
    monitor.start()
    result: dict[str, Any] = {
        "status": "failed",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_imported": False,
        "checkpoint_loaded": False,
        "forward_executed": False,
        "generate_executed": False,
        "training_executed": False,
        "output_dir": str(OUTPUT_DIR),
    }
    temporary_dir: Path | None = None
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(
                f"Wrong Python: expected {LOCAL_PYTHON}, got {sys.executable}"
            )
        os.chdir(REPO)
        if OUTPUT_DIR.exists():
            raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")

        monitor.enter("asset_preflight")
        require_file(TEST_ANNOTATIONS, EXPECTED_ANNOTATIONS_SIZE)
        require_file(EXPECTED_VIDEO, EXPECTED_VIDEO_SIZE)
        require_file(INFERENCE_RESULT, EXPECTED_INFERENCE_RESULT_SIZE)
        require_file(SIGLIP2_ROOT / "preprocessor_config.json")
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".sample_000.", dir=str(OUTPUT_PARENT))
        )

        annotations = json.loads(TEST_ANNOTATIONS.read_text(encoding="utf-8"))
        annotation = annotations[SAMPLE_INDEX]
        if Path(annotation["video"]) != EXPECTED_RELATIVE_VIDEO:
            raise RuntimeError(f"Unexpected video field: {annotation['video']!r}")
        if annotation["comments_text_anonymized"] != EXPECTED_REFERENCE:
            raise RuntimeError("Reference commentary changed")
        prior_result = json.loads(INFERENCE_RESULT.read_text(encoding="utf-8"))
        if prior_result["status"] != "passed":
            raise RuntimeError("Prior fixed-sample inference did not pass")
        prediction = prior_result["prediction"]
        if prediction["generated_commentary"] != EXPECTED_GENERATED:
            raise RuntimeError("Generated commentary changed")

        monitor.enter("import_cpu_video_stack")
        import decord
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import torch
        from transformers import AutoProcessor

        if torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise RuntimeError("CUDA must be unavailable in the CPU-only trace")
        torch.set_num_threads(1)
        decord.bridge.set_bridge("torch")

        monitor.enter("decode_30_middle_frames")
        reader = decord.VideoReader(str(EXPECTED_VIDEO), num_threads=1)
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count != EXPECTED_VIDEO_FRAMES or abs(fps - EXPECTED_FPS) > 1e-6:
            raise RuntimeError(f"Unexpected video metadata: frames={frame_count} fps={fps}")
        indices = [
            int(value)
            for value in frame_indices_middle(NUM_FRAMES, frame_count)
        ]
        expected_indices = list(range(12, 738, 25))
        if indices != expected_indices:
            raise RuntimeError(f"Unexpected frame indices: {indices}")
        frames_thwc = reader.get_batch(indices)
        if tuple(frames_thwc.shape) != (30, 224, 398, 3):
            raise RuntimeError(f"Unexpected decoded shape: {tuple(frames_thwc.shape)}")
        frames_tchw = frames_thwc.permute(0, 3, 1, 2)

        monitor.enter("siglip2_preprocess_cpu")
        processor = AutoProcessor.from_pretrained(
            str(SIGLIP2_ROOT),
            local_files_only=True,
            use_fast=False,
        )
        processed = torch.cat(
            [
                processor(images=frame, return_tensors="pt")["pixel_values"]
                for frame in frames_tchw
            ],
            dim=0,
        )
        if tuple(processed.shape) != (30, 3, 512, 512):
            raise RuntimeError(f"Unexpected processed shape: {tuple(processed.shape)}")
        if processed.dtype != torch.float32 or not torch.isfinite(processed).all().item():
            raise RuntimeError("Processed frames have invalid dtype or values")
        processed_min = float(processed.min().item())
        processed_max = float(processed.max().item())
        if processed_min < -1.000001 or processed_max > 1.000001:
            raise RuntimeError(
                f"Processed range outside [-1, 1]: {processed_min}, {processed_max}"
            )

        frames_rgb = frames_thwc.cpu().numpy()
        processed_rgb = (
            ((processed * 0.5 + 0.5).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        if frames_rgb.dtype != np.uint8 or processed_rgb.dtype != np.uint8:
            raise RuntimeError("Visualization frames must be uint8 RGB")

        monitor.enter("render_visual_artifacts")
        contact_sheet = temporary_dir / "01_sampled_frames.jpg"
        comparison = temporary_dir / "02_preprocess_comparison.jpg"
        make_contact_sheet(
            frames_rgb,
            indices,
            fps,
            contact_sheet,
            Image,
            ImageDraw,
            ImageFont,
        )
        make_preprocess_comparison(
            frames_rgb,
            processed_rgb,
            indices,
            fps,
            comparison,
            Image,
            ImageDraw,
            ImageFont,
        )

        preprocessor_config = json.loads(
            (SIGLIP2_ROOT / "preprocessor_config.json").read_text(encoding="utf-8")
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "scope": "cpu_only_input_sampling_and_preprocessing_visualization",
            "sample_index": SAMPLE_INDEX,
            "device": "cpu",
            "cuda_visible_devices": "",
            "model_imported": False,
            "checkpoint_loaded": False,
            "forward_executed": False,
            "generate_executed": False,
            "training_executed": False,
            "source": {
                "annotations_path": str(TEST_ANNOTATIONS),
                "annotations_sha256": sha256(TEST_ANNOTATIONS),
                "video_path": str(EXPECTED_VIDEO),
                "video_sha256": sha256(EXPECTED_VIDEO),
                "video_bytes": EXPECTED_VIDEO.stat().st_size,
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": frame_count / fps,
                "width": int(frames_thwc.shape[2]),
                "height": int(frames_thwc.shape[1]),
            },
            "sampling": {
                "method": SAMPLE_MODE,
                "requested_frames": NUM_FRAMES,
                "frame_indices": indices,
                "timestamps_seconds": [round(index / fps, 3) for index in indices],
                "decoded_shape_thwc": list(frames_thwc.shape),
                "decoded_dtype": str(frames_thwc.dtype),
            },
            "preprocessing": {
                "processor_root": str(SIGLIP2_ROOT),
                "processor_type": preprocessor_config["image_processor_type"],
                "use_fast": False,
                "resize": preprocessor_config["size"],
                "resample": preprocessor_config["resample"],
                "rescale_factor": preprocessor_config["rescale_factor"],
                "image_mean": preprocessor_config["image_mean"],
                "image_std": preprocessor_config["image_std"],
                "output_shape": list(processed.shape),
                "dtype": str(processed.dtype),
                "min": processed_min,
                "max": processed_max,
            },
            "prior_inference_result": {
                "result_path": str(INFERENCE_RESULT),
                "result_sha256": sha256(INFERENCE_RESULT),
                "reference_commentary": prediction["reference_commentary"],
                "generated_commentary": prediction["generated_commentary"],
                "note": "Displayed as prior evidence; no model was run by this trace.",
            },
            "artifacts": {
                "contact_sheet": {
                    "path": "01_sampled_frames.jpg",
                    "bytes": contact_sheet.stat().st_size,
                    "sha256": sha256(contact_sheet),
                },
                "preprocess_comparison": {
                    "path": "02_preprocess_comparison.jpg",
                    "bytes": comparison.stat().st_size,
                    "sha256": sha256(comparison),
                },
                "html": {"path": "index.html"},
            },
            "unverified_internal_stages": [
                "visual_encoder_features",
                "temporal_position_features",
                "qformer_queries",
                "llama_projection",
                "per_token_generation_scores",
            ],
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_dir / "index.html").write_text(
            build_html(manifest),
            encoding="utf-8",
        )

        monitor.enter("publish_report_atomically")
        temporary_dir.rename(OUTPUT_DIR)
        temporary_dir = None
        result.update(
            status="passed",
            manifest=str(OUTPUT_DIR / "manifest.json"),
            html=str(OUTPUT_DIR / "index.html"),
            contact_sheet=str(OUTPUT_DIR / "01_sampled_frames.jpg"),
            preprocess_comparison=str(
                OUTPUT_DIR / "02_preprocess_comparison.jpg"
            ),
        )
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        if temporary_dir is not None:
            result["incomplete_temporary_dir"] = str(temporary_dir)
        traceback.print_exc()
        return 1
    finally:
        monitor.finish()
        result["timings_seconds"] = monitor.timings
        result["elapsed_seconds"] = round(time.monotonic() - monitor.started, 3)
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("[RESULT]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
