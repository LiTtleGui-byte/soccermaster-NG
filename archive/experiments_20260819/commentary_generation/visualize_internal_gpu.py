#!/usr/bin/env python3
"""Trace and visualize one fixed commentary sample on one approved GPU.

This is an inference-only audit entry.  It captures compact CPU copies of the
real intermediate tensors without changing the model implementation.  It does
not create an optimizer, scheduler, DataLoader, backward pass, or training run.
"""

from __future__ import annotations

import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
LLAMA_ROOT = Path("/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct")
BERT_ROOT = Path("/remote-home/share/huggingface/bert-base-uncased")
SIGLIP2_ROOT = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
VISUAL_BACKBONE = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000/epoch_19/backbone.pt"
)
GENERATION_CHECKPOINT = Path(
    "/remote-home/haolinyang/sports/dirty_code/UniSoccer/output/"
    "large_512_multitask_w_1_epoch_19_train_matchtime_eval_matchtime_"
    "half_lr_bf16/model_save_11.pth"
)
WORD_WORLD = Path(
    "/remote-home/haolinyang/sports/UniSoccer/words_world/match_time.pkl"
)
TEST_ANNOTATIONS = Path(
    "/remote-home/haolinyang/sports/UniSoccer/train_data/video_clip_json/"
    "MatchTime/classification_test.json"
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
EXPECTED_REFERENCE = (
    "[PLAYER] ([TEAM]) gets on the end of a pass on the edge of the box but "
    "his shot is blocked."
)
INPUT_REPORT = REPO / "reports/commentary_trace/sample_000"
PRIOR_INFERENCE = REPO / "reports/audits/commentary_infer_one_result.json"
OUTPUT_PARENT = REPO / "reports/commentary_trace"
OUTPUT_DIR = OUTPUT_PARENT / "sample_000_internal"

SAMPLE_INDEX = 0
SEED = 42
NUM_FRAMES = 30
SAMPLE_MODE = "middle"
EXPECTED_DATASET_LENGTH = 3_256
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_VISUAL_BACKBONE_SIZE = 1_435_281_181
EXPECTED_VIDEO_SIZE = 5_167_597
EXPECTED_EPOCH = 11
EXPECTED_MODEL_PARAMETER_COUNT = 8_418_890_760
EXPECTED_MODEL_STATE_KEYS = 953
MIN_AVAILABLE_CPU_MEMORY_BYTES = 64 * 1024**3
MIN_FREE_GPU_MEMORY_BYTES = 40 * 1024**3


class Monitor:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stage = "preflight"
        self.stage_started = self.started
        self.stop = threading.Event()
        self.timings: dict[str, float] = {}
        self.memory: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self.sample_memory("start")
        self.thread.start()

    def enter(self, stage: str) -> None:
        now = time.monotonic()
        self.timings[self.stage] = round(now - self.stage_started, 3)
        self.stage = stage
        self.stage_started = now
        self.sample_memory(f"enter:{stage}")
        print(f"[STAGE] {stage}", flush=True)

    def finish(self) -> None:
        now = time.monotonic()
        self.timings[self.stage] = round(now - self.stage_started, 3)
        self.sample_memory("finish")
        self.stop.set()
        self.thread.join(timeout=2)

    def sample_memory(self, label: str) -> None:
        status: dict[str, str] = {}
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
        self.memory.append(
            {
                "label": label,
                "elapsed_seconds": round(time.monotonic() - self.started, 3),
                **status,
            }
        )

    def _heartbeat(self) -> None:
        while not self.stop.wait(30):
            self.sample_memory(f"heartbeat:{self.stage}")
            sample = self.memory[-1]
            print(
                f"[HEARTBEAT] stage={self.stage} "
                f"elapsed={sample['elapsed_seconds']}s "
                f"rss={sample.get('VmRSS', 'unknown')}",
                flush=True,
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def require_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected size for {path}: {path.stat().st_size}")


def require_single_visible_gpu() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [] if value is None else [x.strip() for x in value.split(",") if x.strip()]
    if len(devices) != 1 or not devices[0].isdigit():
        raise RuntimeError(f"Exactly one numeric CUDA device is required; got {value!r}")
    return devices[0]


def repository_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def tensor_summary(tensor: Any) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "source_dtype": str(tensor.dtype),
        "finite": bool(value.isfinite().all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "l2_norm": float(value.norm().item()),
    }


def cosine_matrix(array: Any, np: Any) -> list[list[float]]:
    value = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    normalized = value / np.maximum(norms, 1e-12)
    return np.round(normalized @ normalized.T, 6).tolist()


def vector_norms(array: Any, np: Any) -> list[float]:
    return np.round(np.linalg.norm(array.astype(np.float64), axis=-1), 6).tolist()


def projection_bins(array: Any, np: Any, bins: int = 64) -> list[list[float]]:
    value = array.astype(np.float64)
    width = value.shape[-1]
    if width % bins:
        raise RuntimeError(f"Projection width {width} is not divisible by {bins}")
    return np.round(value.reshape(value.shape[0], bins, width // bins).mean(-1), 6).tolist()


def make_position_visualization(
    arrays: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    content = arrays["layer_normalized"][0, :, 0]
    position = arrays["position_embedding"][0]
    fused = arrays["temporal_fused"][0]
    content_bins = np.asarray(projection_bins(content, np))
    position_bins = np.asarray(projection_bins(position, np))
    fused_bins = np.asarray(projection_bins(fused, np))
    common_abs_limit = float(
        max(
            np.abs(content_bins).max(),
            np.abs(position_bins).max(),
            np.abs(fused_bins).max(),
        )
    )
    addition_preview = []
    for dimension in range(12):
        calculated = float(content[0, dimension] + position[0, dimension])
        captured = float(fused[0, dimension])
        addition_preview.append(
            {
                "dimension": dimension,
                "visual": float(content[0, dimension]),
                "position": float(position[0, dimension]),
                "calculated_sum": calculated,
                "captured_fused": captured,
                "absolute_error": abs(calculated - captured),
            }
        )
    return {
        "content_bins": content_bins.tolist(),
        "position_bins": position_bins.tolist(),
        "fused_bins": fused_bins.tolist(),
        "common_abs_limit": common_abs_limit,
        "max_abs_addition_error": float(
            np.abs(fused - (content + position)).max()
        ),
        "addition_preview": addition_preview,
    }


def build_html(
    manifest: dict[str, Any],
    tensor_previews: dict[str, list[float]],
    position_visualization: dict[str, Any],
) -> str:
    payload = json.dumps(manifest["visualization"], ensure_ascii=False).replace("</", "<\\/")
    position_payload = json.dumps(
        position_visualization,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    generated = html.escape(manifest["prediction"]["generated_commentary"])
    reference = html.escape(manifest["prediction"]["reference_commentary"])
    matched = "是" if manifest["prediction"]["matches_prior_run"] else "否（随机采样允许变化）"
    tensor_order = (
        "visual_encoder",
        "layer_normalized",
        "position_embedding",
        "temporal_fused",
        "qformer",
        "llama_projection",
    )
    rows = []
    for name in tensor_order:
        item = manifest["tensors"][name]
        elements = math.prod(item["shape"])
        preview = ", ".join(f"{value:.5f}" for value in tensor_previews[name])
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f"<td><code>{item['shape']}</code><br><small>{elements:,}个元素</small></td>"
            f"<td>{item['source_dtype']}</td><td>{item['min']:.5f} ～ {item['max']:.5f}</td>"
            f"<td>{item['mean']:.5f}</td><td>{item['std']:.5f}</td>"
            f"<td>{item['l2_norm']:.3f}</td>"
            f"<td class='preview'><code>{preview}</code></td></tr>"
        )
    token_rows = []
    for token in manifest["generation_trace"]["tokens"]:
        probability = token.get("probability")
        probability_text = "—" if probability is None else f"{100 * probability:.2f}%"
        bar = 0 if probability is None else max(0, min(100, 100 * probability))
        token_rows.append(
            f"<tr><td>{token['step']}</td><td>{token['token_id']}</td>"
            f"<td><code>{html.escape(token['token_text'])}</code></td>"
            f"<td>{probability_text}</td><td><div class='bar'><i style='width:{bar:.3f}%'></i></div></td></tr>"
        )
    addition_rows = []
    for item in position_visualization["addition_preview"]:
        addition_rows.append(
            f"<tr><td>{item['dimension']}</td>"
            f"<td>{item['visual']:.6f}</td><td>{item['position']:.6f}</td>"
            f"<td>{item['calculated_sum']:.6f}</td>"
            f"<td>{item['captured_fused']:.6f}</td>"
            f"<td>{item['absolute_error']:.3g}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoccerMaster 内部张量链路 · Sample 000</title><style>
:root{{--ink:#152033;--muted:#637083;--line:#dce2ea;--ok:#147d64;--blue:#3566b8;--soft:#f7f9fb;--purple:#7556a8}}
body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:#f4f7fa}}
main{{max-width:1500px;margin:auto;padding:32px 25px 70px}} h1{{margin:7px 0}} h2{{margin-top:38px}} h3{{margin:0 0 10px}}
.tag{{display:inline-block;padding:5px 10px;border-radius:999px;background:#e7f4ef;color:#10644f;font-size:12px}}
.pipeline{{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin:24px 0}} .stage{{background:#fff;border:1px solid var(--line);border-top:5px solid var(--ok);border-radius:9px;padding:12px;min-height:110px}}
.stage b{{display:block;margin:8px 0}} .stage small,.muted{{color:var(--muted)}} .card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:7px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}
.caption{{padding:15px;background:#f7f9fb;border-radius:8px;line-height:1.6}}.bar{{width:180px;height:9px;background:#e8edf4;border-radius:9px}}.bar i{{display:block;height:100%;background:var(--blue);border-radius:9px}}
.module{{background:#fff;border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:12px;padding:20px;margin:18px 0}}
.module.purple{{border-left-color:var(--purple)}} .io{{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:12px;align-items:stretch}}
.io-box{{background:var(--soft);border-radius:9px;padding:15px;line-height:1.55}} .io-box.action{{background:#eef4ff}} .io-label{{font-size:12px;font-weight:700;color:var(--muted);letter-spacing:.05em}}
.shape{{font-size:18px;font-weight:700;color:#244f91;display:block;margin:6px 0}} .arrow{{color:var(--blue);font-weight:700}} .axis{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
.axis span{{background:#e8edf4;border-radius:6px;padding:4px 7px;font-size:12px}} .actual{{border-top:1px solid var(--line);margin-top:14px;padding-top:12px}}
.preview{{max-width:330px;line-height:1.45;overflow-wrap:anywhere}} .warning{{background:#fff7df;border-left:5px solid #d49b27;padding:13px 16px;border-radius:7px}}
.legend{{height:12px;border-radius:8px;background:linear-gradient(90deg,rgb(245,248,252),rgb(75,153,217));margin:9px 0 4px}} .legend-labels{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}}
.legend.diverging{{background:linear-gradient(90deg,rgb(45,92,175),rgb(248,249,251),rgb(190,60,55))}} .three{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
code{{overflow-wrap:anywhere}} small{{color:var(--muted)}} @media(max-width:950px){{.pipeline{{grid-template-columns:repeat(2,1fr)}}.grid,.io{{grid-template-columns:1fr}}}}
</style></head><body><main><span class="tag">真实单样本 GPU trace · inference only</span>
<h1>SoccerMaster 解说生成：数据怎样穿过每个模块</h1><p class="muted">固定 MatchTime sample 000。主线先回答“模块吃进去什么、做什么、吐出来什么”；后半部分保留真实张量和统计图。</p>
<div class="pipeline">
<div class="stage"><b>1 原始视频</b><small>30秒比赛片段</small></div><div class="stage"><b>2 采样</b><small>均匀取30帧</small></div>
<div class="stage"><b>3 预处理</b><small>1×3×30×512×512</small></div><div class="stage"><b>4 视觉编码</b><small>每帧压成1024维</small></div>
<div class="stage"><b>5 Q-Former</b><small>整段视频压成32个query</small></div><div class="stage"><b>6 Llama投影</b><small>32×4096视觉前缀</small></div>
<div class="stage"><b>7 逐词生成</b><small>受限词表自回归生成</small></div></div>
<p><a href="../sample_000/index.html">查看上一份：原视频、30帧时间轴和预处理对比</a></p>

<h2>逐模块输入与输出</h2>
<div class="module"><h3>0. 视频采样与 SigLIP2 预处理</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">30帧 RGB 图像</span><div class="axis"><span>每帧 398×224</span><span>uint8</span><span>0～255</span></div></div>
<div class="io-box action"><div class="io-label">做什么</div>从30秒视频均匀取30帧；逐帧缩放为512×512，并按SigLIP2规则归一化。<br><span class="arrow">图像 → 模型可读取的像素张量</span></div>
<div class="io-box"><div class="io-label">输出 / 下一模块输入</div><span class="shape">[1, 3, 30, 512, 512]</span><div class="axis"><span>1个视频</span><span>RGB 3通道</span><span>30帧</span><span>512×512</span></div></div></div></div>

<div class="module"><h3>1. Visual Encoder（SigLIP2 + 时空视觉编码）</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">[1, 3, 30, 512, 512]</span>一批中的一个视频，共30张已经归一化的画面。</div>
<div class="io-box action"><div class="io-label">做什么</div>把每帧切成视觉patch，提取人物、球场、动作和场景等抽象特征；后部时序层还允许不同帧交换信息。最后对每帧的空间信息做汇总。<br><span class="arrow">像素 → 每帧一个视觉向量</span></div>
<div class="io-box"><div class="io-label">输出</div><span class="shape">[1, 30, 1024]</span><div class="axis"><span>1个视频</span><span>30帧</span><span>每帧1024维</span></div></div></div>
<div class="actual">本次真实输出：均值 {manifest['tensors']['visual_encoder']['mean']:.5f}，标准差 {manifest['tensors']['visual_encoder']['std']:.5f}，范围 {manifest['tensors']['visual_encoder']['min']:.3f}～{manifest['tensors']['visual_encoder']['max']:.3f}。</div></div>

<div class="module"><h3>2. LayerNorm + 帧位置编码</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">视觉 [1,30,1,1024]</span>上一阶段每帧的1024维视觉特征；额外的1是统一的token轴。</div>
<div class="io-box action"><div class="io-label">做什么</div>LayerNorm先稳定每帧特征尺度；再查询30个可学习的位置向量，并与对应帧逐元素相加。这样内容相似的画面也能保留“发生在第几帧”的区别。<br><span class="arrow">画面内容 + 时间位置</span></div>
<div class="io-box"><div class="io-label">输出</div><span class="shape">[1, 30, 1024]</span>去掉长度为1的token轴后，得到送给Q-Former的30个时序视觉token。</div></div>
<div class="actual">真实中间量：归一化特征 <code>[1,30,1,1024]</code>；位置向量 <code>[1,30,1024]</code>；融合结果标准差 {manifest['tensors']['temporal_fused']['std']:.5f}。</div></div>

<h2>位置编码究竟怎样加进去</h2>
<div class="card"><p>模型先为帧序号 <code>0...29</code> 分别查出一个训练得到的1024维向量，然后逐帧、逐维相加。下面三图使用完全相同的蓝—白—红色标：蓝色是负值，白色接近0，红色是正值。</p>
<div class="three"><div><h3>A. 归一化视觉内容</h3><canvas id="contentBins" width="560" height="420"></canvas><p class="muted">30帧 × 64个显示列</p></div><div><h3>B. 学到的位置向量</h3><canvas id="positionBins" width="560" height="420"></canvas><p class="muted">第1行只属于第1帧，第30行只属于第30帧</p></div><div><h3>C. A + B 的融合结果</h3><canvas id="fusedBins" width="560" height="420"></canvas><p class="muted">这就是Q-Former实际接收的30个token</p></div></div>
<div class="legend diverging"></div><div class="legend-labels"><span id="positionScaleLow">负值</span><span>0</span><span id="positionScaleHigh">正值</span></div>
<p class="muted">为了放进屏幕，每16个相邻维度取均值，将1024维压成64列。均值是线性运算，所以显示图仍严格满足 C=A+B；完整数据没有压缩地保存在NPZ。</p></div>
<div class="card"><h3>第1帧前12维：真实逐元素加法</h3><table><thead><tr><th>维度</th><th>视觉 A</th><th>位置 B</th><th>计算 A+B</th><th>捕获的 C</th><th>误差</th></tr></thead><tbody>{''.join(addition_rows)}</tbody></table><p>全张量最大绝对加法误差：<code>{position_visualization['max_abs_addition_error']:.9g}</code>。</p></div>
<div class="note"><b>位置向量里编码的不是文字或秒数。</b>每个位置只是checkpoint中训练得到的一组1024个浮点参数。训练过程让“第几帧”对应的向量变得有用；单独某一维通常没有可直接命名的人类含义。</div>

<div class="module purple"><h3>3. Video Q-Former</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">视频 [1,30,1024]</span>另外还有32个可学习query，每个query内部宽度为768。</div>
<div class="io-box action"><div class="io-label">做什么</div>32个query通过cross-attention共同阅读30帧：某些query可能收集动作，另一些收集人物或全局事件。它把长度随视频变化的信息整理成固定数量的摘要。<br><span class="arrow">30个帧token → 32个视频摘要</span></div>
<div class="io-box"><div class="io-label">输出</div><span class="shape">[1, 32, 768]</span><div class="axis"><span>1个视频</span><span>32个query</span><span>每个768维</span></div></div></div>
<div class="actual">本次真实输出均值 {manifest['tensors']['qformer']['mean']:.5f}、标准差 {manifest['tensors']['qformer']['std']:.5f}。单个query没有预先指定的人类语义，不能直接命名为“足球query”或“球员query”。</div></div>

<div class="module purple"><h3>4. Llama Projection（线性投影层）</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">[1, 32, 768]</span>Q-Former生成的32个视频摘要。</div>
<div class="io-box action"><div class="io-label">做什么</div>用一个学习到的线性层把每个query从768维映射到Llama-3-8B的4096维隐藏空间。它不是生成文字，而是在做“接口翻译”。<br><span class="arrow">Q-Former语言 → Llama内部语言</span></div>
<div class="io-box"><div class="io-label">输出</div><span class="shape">[1, 32, 4096]</span><div class="axis"><span>32个视觉前缀token</span><span>每个4096维</span></div></div></div>
<div class="actual">这32个向量就是Llama真正收到的视频信息；本次标准差 {manifest['tensors']['llama_projection']['std']:.5f}，所有元素均为有限值。</div></div>

<div class="module"><h3>5. Llama-3-8B 自回归解码器</h3><div class="io">
<div class="io-box"><div class="io-label">输入</div><span class="shape">视觉 [1,32,4096] + BOS</span>32个视觉token后拼接一个开始生成token，形成视觉条件前缀。</div>
<div class="io-box action"><div class="io-label">做什么</div>每一步根据视觉前缀和此前已经生成的token，预测下一个token；历史实现还用 <code>match_time.pkl</code> 限制可选词表，并采用5-beam sampling。<br><span class="arrow">视觉语义 → token ID → 解说文字</span></div>
<div class="io-box"><div class="io-label">输出</div><span class="shape">27个实际token</span>解码后组成下方英文解说；逐步token ID和选中概率见页面末尾。</div></div></div>
<div class="warning">已知风险：历史生成路径没有显式传入attention mask，而且padding token与EOS有关联。它不妨碍本次链路运行，但可能影响生成可靠性，不能把单样本结果当成总体质量结论。</div>

<h2>内部张量数据浏览器</h2><p>“真实值预览”显示每个张量按内存顺序展开后的前12个float32元素，方便确认里面确实是数值数据，而不是模块名称。完整数组可下载：<a href="internal_tensors.npz">internal_tensors.npz</a>。</p>
<div class="card"><table><thead><tr><th>张量</th><th>shape / 数量</th><th>dtype</th><th>最小～最大</th><th>均值</th><th>标准差</th><th>L2</th><th>前12个真实值</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>

<h2>辅助图：内部数据的结构</h2>
<div class="grid"><div class="card"><h3>Visual Encoder：绝对相似度</h3><canvas id="visualCos" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span>-1（方向相反）</span><span>1（完全同向）</span></div><p class="muted">固定色标[-1,1]，适合跨视频比较。这里几乎同色本身就是结论：30帧总体非常相似。悬停色块查看帧号和精确值。</p><p id="visualCosInfo"><code>等待悬停</code></p></div>
<div class="card"><h3>Visual Encoder：本视频局部增强</h3><canvas id="visualCosAdaptive" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span id="adaptiveMin">本视频最小值</span><span id="adaptiveMax">本视频最大值</span></div><p class="muted">忽略恒为1的对角线，并把非对角线的最小～最大值拉满整个色带。它只用于观察本视频内部的细微差异，不能直接与其他视频颜色比较。</p><p id="visualCosAdaptiveInfo"><code>等待悬停</code></p></div></div>
<div class="card"><h3>每帧视觉向量强度（L2范数）</h3><canvas id="visualNorm" width="1100" height="420"></canvas><p class="muted">每个点把该帧1024个特征数平方、求和、再开根号。它反映特征的整体数值尺度，用于发现异常突增、骤降或接近零；它不是置信度、重要性或对最终解说的贡献度。</p></div>
<div class="grid"><div class="card"><h3>时间融合后：绝对相似度</h3><canvas id="temporalCos" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span>-1</span><span>1</span></div><p>这里比较的是“归一化视觉特征 + 帧位置向量”之后的30个token。它回答帧token加入时间身份后还剩多少相似性，不是attention权重。</p></div><div class="card"><h3>时间融合后：局部增强</h3><canvas id="temporalCosAdaptive" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span id="temporalMin">最小</span><span id="temporalMax">最大</span></div><p class="muted">拉伸本矩阵非对角线范围，突出哪些时间位置相对更接近。</p></div></div>
<div class="grid"><div class="card"><h3>Q-Former：32个query绝对相似度</h3><canvas id="queryCos" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span>-1</span><span>1</span></div><p>每个格子比较两个768维query输出。高表示两个摘要方向接近，低表示它们提取的信息更不同；它不是query对某一帧的注意力。</p></div><div class="card"><h3>Q-Former：局部增强</h3><canvas id="queryCosAdaptive" width="620" height="620"></canvas><div class="legend"></div><div class="legend-labels"><span id="queryMin">最小</span><span id="queryMax">最大</span></div><p class="muted">用于观察32个摘要内部的相对分工。要回答“第几个query看了第几帧”，仍需另行采集cross-attention。</p></div></div>
<div class="card"><h3>Llama视觉前缀：32×4096</h3><canvas id="projection" width="1100" height="520"></canvas><p class="muted">显示时把4096维按每64维求均值，得到32×64色块；NPZ中保存的是完整数据，没有降维。</p></div>
<h2>最终生成</h2><div class="grid"><div class="caption"><b>参考解说</b><br>{reference}</div><div class="caption"><b>本次模型生成</b><br>{generated}<br><small>与此前固定运行一致：{matched}</small></div></div>
<h2>逐步生成 token</h2><div class="card"><table><thead><tr><th>步</th><th>token ID</th><th>token文本</th><th>选中token概率</th><th></th></tr></thead><tbody>{''.join(token_rows)}</tbody></table><p class="muted">概率由生成时经过限制词表处理后的分数归一化计算；beam sampling 下它描述最终返回序列的transition score，不是所有候选beam的完整搜索树。</p></div>
<p class="muted">完整机器证据：<a href="manifest.json">manifest.json</a>。本报告不证明总体指标，也没有执行训练。</p>
<script>const D={payload};const P={position_payload};
function heat(id,matrix,min=-1,max=1,maskDiagonal=false){{const c=document.getElementById(id),x=c.getContext('2d'),rows=matrix.length,cols=matrix[0].length,w=c.width/cols,h=c.height/rows;x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);matrix.forEach((row,i)=>row.forEach((v,j)=>{{if(maskDiagonal&&i===j){{x.fillStyle='#e4e8ee'}}else{{let t=Math.max(0,Math.min(1,(v-min)/(max-min||1)));let r=Math.round(245-(170*t)),g=Math.round(248-(95*t)),b=Math.round(252-(35*t));x.fillStyle=`rgb(${{r}},${{g}},${{b}})`}}x.fillRect(j*w,i*h,w+1,h+1)}}));const info=document.getElementById(id+'Info');if(info)c.onmousemove=(event)=>{{const rect=c.getBoundingClientRect(),col=Math.min(cols-1,Math.floor((event.clientX-rect.left)*cols/rect.width)),row=Math.min(rows-1,Math.floor((event.clientY-rect.top)*rows/rect.height));info.innerHTML=`帧 ${{row+1}} 与帧 ${{col+1}}：<code>${{matrix[row][col].toFixed(6)}}</code>${{row===col?'（自身）':''}}`}};}}
function line(id,values){{const c=document.getElementById(id),x=c.getContext('2d'),p=45,W=c.width-2*p,H=c.height-2*p,lo=Math.min(...values),hi=Math.max(...values);x.strokeStyle='#ccd5e1';x.beginPath();x.moveTo(p,p);x.lineTo(p,p+H);x.lineTo(p+W,p+H);x.stroke();x.strokeStyle='#3566b8';x.lineWidth=3;x.beginPath();values.forEach((v,i)=>{{let px=p+i*W/(values.length-1),py=p+H-(v-lo)*H/(hi-lo||1);i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke();x.fillStyle='#637083';x.fillText(`min ${{lo.toFixed(2)}} / max ${{hi.toFixed(2)}}`,p,25)}}
function diverging(id,matrix,limit){{const c=document.getElementById(id),x=c.getContext('2d'),rows=matrix.length,cols=matrix[0].length,w=c.width/cols,h=c.height/rows;matrix.forEach((row,i)=>row.forEach((v,j)=>{{let t=Math.max(-1,Math.min(1,v/(limit||1))),r,g,b;if(t<0){{let u=t+1;r=Math.round(45+203*u);g=Math.round(92+157*u);b=Math.round(175+76*u)}}else{{r=Math.round(248-58*t);g=Math.round(249-189*t);b=Math.round(251-196*t)}}x.fillStyle=`rgb(${{r}},${{g}},${{b}})`;x.fillRect(j*w,i*h,w+1,h+1)}}));}}
diverging('contentBins',P.content_bins,P.common_abs_limit);diverging('positionBins',P.position_bins,P.common_abs_limit);diverging('fusedBins',P.fused_bins,P.common_abs_limit);document.getElementById('positionScaleLow').textContent=(-P.common_abs_limit).toFixed(4);document.getElementById('positionScaleHigh').textContent=P.common_abs_limit.toFixed(4);
const offDiagonal=D.visual_cosine.flatMap((row,i)=>row.filter((_,j)=>i!==j)),adaptiveLow=Math.min(...offDiagonal),adaptiveHigh=Math.max(...offDiagonal);
document.getElementById('adaptiveMin').textContent=adaptiveLow.toFixed(6)+'（非对角最小）';document.getElementById('adaptiveMax').textContent=adaptiveHigh.toFixed(6)+'（非对角最大）';
const temporalOff=D.temporal_cosine.flatMap((row,i)=>row.filter((_,j)=>i!==j)),temporalLow=Math.min(...temporalOff),temporalHigh=Math.max(...temporalOff);
const queryOff=D.query_cosine.flatMap((row,i)=>row.filter((_,j)=>i!==j)),queryLow=Math.min(...queryOff),queryHigh=Math.max(...queryOff);
document.getElementById('temporalMin').textContent=temporalLow.toFixed(6);document.getElementById('temporalMax').textContent=temporalHigh.toFixed(6);
document.getElementById('queryMin').textContent=queryLow.toFixed(6);document.getElementById('queryMax').textContent=queryHigh.toFixed(6);
heat('visualCos',D.visual_cosine);heat('visualCosAdaptive',D.visual_cosine,adaptiveLow,adaptiveHigh,true);line('visualNorm',D.visual_norms);
heat('temporalCos',D.temporal_cosine);heat('temporalCosAdaptive',D.temporal_cosine,temporalLow,temporalHigh,true);
heat('queryCos',D.query_cosine);heat('queryCosAdaptive',D.query_cosine,queryLow,queryHigh,true);
let vals=D.projection_bins.flat(),lo=Math.min(...vals),hi=Math.max(...vals);heat('projection',D.projection_bins,lo,hi);
</script></main></body></html>"""


def run() -> int:
    monitor = Monitor()
    monitor.start()
    temporary_dir: Path | None = None
    torch = None
    hooks: list[Any] = []
    result: dict[str, Any] = {
        "status": "failed",
        "scope": "single_sample_gpu_internal_tensor_trace",
        "sample_index": SAMPLE_INDEX,
        "seed": SEED,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "dataloader_created": False,
        "output_dir": str(OUTPUT_DIR),
    }
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(f"Wrong Python: {sys.executable}")
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)

        monitor.enter("asset_preflight")
        physical_gpu = require_single_visible_gpu()
        for path, size in (
            (VISUAL_BACKBONE, EXPECTED_VISUAL_BACKBONE_SIZE),
            (GENERATION_CHECKPOINT, EXPECTED_CHECKPOINT_SIZE),
            (EXPECTED_VIDEO, EXPECTED_VIDEO_SIZE),
        ):
            require_file(path, size)
        for path in (LLAMA_ROOT, BERT_ROOT, SIGLIP2_ROOT, TEST_VIDEO_ROOT, INPUT_REPORT):
            if not path.is_dir():
                raise NotADirectoryError(path)
        require_file(TEST_ANNOTATIONS)
        require_file(WORD_WORLD)
        require_file(PRIOR_INFERENCE)
        if OUTPUT_DIR.exists():
            raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
        if available_memory_bytes() < MIN_AVAILABLE_CPU_MEMORY_BYTES:
            raise RuntimeError("Insufficient available CPU memory")
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=".sample_000_internal.", dir=OUTPUT_PARENT))

        monitor.enter("import_framework")
        import numpy as np
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(1)
        from research.experiments.commentary_generation.runtime.dataset.commentary import (
            MatchVisionCommentary_new_benchmark_from_npy_Dataset,
        )
        from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
            matchvoice_model_all_blocks,
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA GPU")
        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < MIN_FREE_GPU_MEMORY_BYTES:
            raise RuntimeError(f"Only {free_bytes} free GPU bytes")
        torch.cuda.reset_peak_memory_stats(device)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

        monitor.enter("decode_and_preprocess_sample")
        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=[str(TEST_ANNOTATIONS)], video_base_dir=[str(TEST_VIDEO_ROOT)],
            num_frames=NUM_FRAMES, sample=SAMPLE_MODE, tokenizer_name=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
        )
        if len(dataset) != EXPECTED_DATASET_LENGTH:
            raise RuntimeError(f"Unexpected dataset length {len(dataset)}")
        item = dataset[SAMPLE_INDEX]
        if Path(item["video_path"]) != EXPECTED_VIDEO or item["caption_text"] != EXPECTED_REFERENCE:
            raise RuntimeError("Fixed sample contract changed")
        samples = dataset.collater([item])
        if tuple(samples["frames"].shape) != (1, 3, 30, 512, 512):
            raise RuntimeError(f"Unexpected frame shape {samples['frames'].shape}")

        monitor.enter("construct_model_and_load_checkpoint")
        model = matchvoice_model_all_blocks(
            num_features=1024, need_temporal=True, need_spatial=False,
            use_local_features=False, open_visual_encoder=True, open_llm_decoder=True,
            file_path=str(WORD_WORLD), tokenizer_ckpt=str(LLAMA_ROOT), llm_ckpt=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT), visual_encoder_checkpoint=str(VISUAL_BACKBONE),
            timesformer_type="unisoccer_part_temporal", encoder_type="spatial_and_temporal",
            num_video_query_token=32, use_mlp=False,
        )
        if sum(p.numel() for p in model.parameters()) != EXPECTED_MODEL_PARAMETER_COUNT:
            raise RuntimeError("Model parameter count changed")
        if len(model.state_dict()) != EXPECTED_MODEL_STATE_KEYS:
            raise RuntimeError("Model state key count changed")
        checkpoint = torch.load(GENERATION_CHECKPOINT, map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != EXPECTED_EPOCH:
            raise RuntimeError(f"Unexpected checkpoint epoch {checkpoint.get('epoch')}")
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint keys mismatch: {incompatible}")
        del checkpoint
        gc.collect()

        captures: dict[str, Any] = {}
        generation: dict[str, Any] = {}
        def capture_tensor(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
                captures[name] = value.detach().float().cpu()
            return hook
        def capture_qformer_input(_module: Any, _args: Any, kwargs: Any) -> None:
            captures["temporal_fused"] = kwargs["encoder_hidden_states"].detach().float().cpu()
        hooks.append(model.visual_encoder.register_forward_hook(capture_tensor("visual_encoder")))
        hooks.append(model.ln_vision.register_forward_hook(capture_tensor("layer_normalized")))
        hooks.append(model.video_frame_position_embedding.register_forward_hook(capture_tensor("position_embedding")))
        hooks.append(model.video_Qformer.bert.register_forward_pre_hook(capture_qformer_input, with_kwargs=True))
        hooks.append(model.video_Qformer.bert.register_forward_hook(capture_tensor("qformer")))
        hooks.append(model.llama_proj.register_forward_hook(capture_tensor("llama_projection")))

        original_generate = model.llama_model.generate
        def traced_generate(*args: Any, **kwargs: Any) -> Any:
            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
            output = original_generate(*args, **kwargs)
            generation["sequences"] = output.sequences.detach().cpu()
            if output.scores:
                scores = model.llama_model.compute_transition_scores(
                    output.sequences, output.scores,
                    getattr(output, "beam_indices", None),
                    normalize_logits=True,
                )
                generation["transition_scores"] = scores.detach().float().cpu()
            return output.sequences
        model.llama_model.generate = traced_generate

        monitor.enter("move_to_gpu")
        model = model.to(device).eval()
        for key, value in samples.items():
            if isinstance(value, torch.Tensor):
                samples[key] = value.to(device)
        torch.cuda.synchronize(device)

        monitor.enter("forward_generate_and_trace")
        with torch.no_grad():
            generated_texts, references, video_paths = model(samples, True)
        torch.cuda.synchronize(device)
        if references != [EXPECTED_REFERENCE] or [Path(x) for x in video_paths] != [EXPECTED_VIDEO]:
            raise RuntimeError("Returned sample metadata changed")
        if len(generated_texts) != 1 or not generated_texts[0].strip():
            raise RuntimeError("Generated text is empty")
        required = {"visual_encoder", "layer_normalized", "position_embedding", "temporal_fused", "qformer", "llama_projection"}
        if set(captures) != required:
            raise RuntimeError(f"Missing captures: {sorted(required - set(captures))}")
        expected_shapes = {
            "visual_encoder": (1, 30, 1024), "layer_normalized": (1, 30, 1, 1024),
            "position_embedding": (1, 30, 1024), "temporal_fused": (1, 30, 1024),
            "qformer": (1, 32, 768), "llama_projection": (1, 32, 4096),
        }
        for name, shape in expected_shapes.items():
            if tuple(captures[name].shape) != shape or not captures[name].isfinite().all().item():
                raise RuntimeError(f"Invalid {name}: {captures[name].shape}")

        monitor.enter("render_report")
        arrays = {name: value.numpy() for name, value in captures.items()}
        np.savez_compressed(temporary_dir / "internal_tensors.npz", **arrays)
        visual = arrays["visual_encoder"][0]
        temporal = arrays["temporal_fused"][0]
        queries = arrays["qformer"][0]
        projection = arrays["llama_projection"][0]
        sequences = generation["sequences"][0].tolist()
        scores = generation.get("transition_scores")
        score_values = [] if scores is None else scores[0].tolist()
        generated_ids = sequences[len(sequences) - len(score_values):] if score_values else sequences
        tokens = []
        for index, token_id in enumerate(generated_ids):
            probability = None if not score_values else math.exp(score_values[index])
            tokens.append({
                "step": index + 1, "token_id": int(token_id),
                "token_text": model.tokenizer.decode([token_id]),
                "log_probability": None if not score_values else float(score_values[index]),
                "probability": probability,
            })
        prior = json.loads(PRIOR_INFERENCE.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1, "status": "passed", "scope": result["scope"],
            "sample_index": SAMPLE_INDEX, "seed": SEED, "device": "cuda:0",
            "physical_gpu": physical_gpu, "cuda_visible_devices": physical_gpu,
            "python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device), "gpu_free_bytes_before": free_bytes,
            "gpu_total_bytes": total_bytes, "checkpoint_epoch": EXPECTED_EPOCH,
            "environment": {
                "python_executable": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
                "offline": True,
                "model_dtype": "mixed: trainable modules float32, Llama base bfloat16",
            },
            "repository": repository_state(),
            "assets": {
                "llama": str(LLAMA_ROOT), "bert": str(BERT_ROOT),
                "siglip2": str(SIGLIP2_ROOT), "visual_backbone": str(VISUAL_BACKBONE),
                "generation_checkpoint": str(GENERATION_CHECKPOINT),
                "word_world": str(WORD_WORLD), "annotations": str(TEST_ANNOTATIONS),
            },
            "missing_keys": [], "unexpected_keys": [], "forward_executed": True,
            "generate_executed": True, "training_executed": False, "backward_executed": False,
            "optimizer_created": False, "dataloader_created": False,
            "input": {"video_path": str(EXPECTED_VIDEO), "frames_shape": list(samples["frames"].shape), "sample_mode": SAMPLE_MODE},
            "tensors": {name: tensor_summary(value) for name, value in captures.items()},
            "prediction": {"reference_commentary": EXPECTED_REFERENCE, "generated_commentary": generated_texts[0],
                "matches_prior_run": generated_texts[0] == prior["prediction"]["generated_commentary"]},
            "generation_trace": {"sequence_ids": sequences, "score_count": len(score_values), "tokens": tokens,
                "note": "Transition probabilities are normalized post-processor scores for the returned beam-sampled sequence."},
            "visualization": {"visual_cosine": cosine_matrix(visual, np), "visual_norms": vector_norms(visual, np),
                "temporal_cosine": cosine_matrix(temporal, np), "query_cosine": cosine_matrix(queries, np),
                "projection_bins": projection_bins(projection, np)},
            "artifacts": {"html": "index.html", "manifest": "manifest.json", "raw_tensors": "internal_tensors.npz",
                "input_report": "../sample_000/index.html"},
            "unverified": ["batch_level_metrics", "beam_search_tree", "attention_maps", "training_behavior"],
        }
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tensor_previews = {
            name: array.reshape(-1)[:12].round(6).tolist()
            for name, array in arrays.items()
        }
        (temporary_dir / "index.html").write_text(
            build_html(
                manifest,
                tensor_previews,
                make_position_visualization(arrays, np),
            ),
            encoding="utf-8",
        )
        manifest["artifacts"]["raw_tensors_bytes"] = (temporary_dir / "internal_tensors.npz").stat().st_size
        manifest["artifacts"]["raw_tensors_sha256"] = sha256(temporary_dir / "internal_tensors.npz")
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        monitor.enter("publish_report_atomically")
        temporary_dir.rename(OUTPUT_DIR)
        temporary_dir = None
        result.update(status="passed", html=str(OUTPUT_DIR / "index.html"), manifest=str(OUTPUT_DIR / "manifest.json"),
            generated_commentary=generated_texts[0], gpu_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
            gpu_peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        if temporary_dir is not None:
            result["incomplete_temporary_dir"] = str(temporary_dir)
        traceback.print_exc()
        return 1
    finally:
        for hook in hooks:
            hook.remove()
        monitor.finish()
        result["timings_seconds"] = monitor.timings
        result["elapsed_seconds"] = round(time.monotonic() - monitor.started, 3)
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if torch is not None and torch.cuda.is_available():
            result.setdefault("gpu_peak_memory_allocated_bytes", torch.cuda.max_memory_allocated(0))
            result.setdefault("gpu_peak_memory_reserved_bytes", torch.cuda.max_memory_reserved(0))
        print("[RESULT]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(run())
