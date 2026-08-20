#!/usr/bin/env python3
"""Replay only Q-Former on a captured tensor and visualize cross-attention.

CPU-only and offline.  This does not construct the visual encoder or Llama,
decode video, generate text, create a DataLoader/optimizer, or train anything.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
BERT_ROOT = Path("/remote-home/share/huggingface/bert-base-uncased")
CHECKPOINT = Path(
    "/remote-home/haolinyang/sports/dirty_code/UniSoccer/output/"
    "large_512_multitask_w_1_epoch_19_train_matchtime_eval_matchtime_"
    "half_lr_bf16/model_save_11.pth"
)
INTERNAL_ROOT = REPO / "reports/commentary_trace/sample_000_internal"
INPUT_NPZ = INTERNAL_ROOT / "internal_tensors.npz"
INPUT_MANIFEST = INTERNAL_ROOT / "manifest.json"
FRAME_MANIFEST = REPO / "reports/commentary_trace/sample_000/manifest.json"
OUTPUT_PARENT = REPO / "reports/commentary_trace"
OUTPUT_DIR = OUTPUT_PARENT / "sample_000_qformer_attention"
EXPECTED_CHECKPOINT_SIZE = 17_615_455_530
EXPECTED_EPOCH = 11
EXPECTED_ALL_KEYS = 953
EXPECTED_QFORMER_KEYS = 55
EXPECTED_LAYERS = 2
EXPECTED_HEADS = 12
EXPECTED_QUERIES = 32
EXPECTED_FRAMES = 30


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
        self.timings[self.stage] = round(now - self.stage_started, 3)
        self.stage = stage
        self.stage_started = now
        print(f"[STAGE] {stage}", flush=True)

    def finish(self) -> None:
        now = time.monotonic()
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


def build_html(manifest: dict[str, Any]) -> str:
    data = json.dumps(manifest["visualization"], ensure_ascii=False).replace("</", "<\\/")
    query_rows = []
    for item in manifest["query_top_frames"]:
        top = "；".join(
            f"帧{frame['frame_position']}（{frame['timestamp_seconds']:.2f}s，{100*frame['weight']:.2f}%）"
            for frame in item["top_frames"]
        )
        query_rows.append(
            f"<tr><td>Query {item['query']}</td><td>{top}</td>"
            f"<td>{item['entropy']:.4f}</td></tr>"
        )
    frame_rows = []
    for item in manifest["top_global_frames"]:
        frame_rows.append(
            f"<tr><td>{item['rank']}</td><td>帧{item['frame_position']}</td>"
            f"<td>{item['timestamp_seconds']:.2f}s</td>"
            f"<td>{100*item['weight']:.3f}%</td></tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q-Former Query × Frame Attention</title><style>
:root{{--ink:#162033;--muted:#657286;--line:#dce3eb;--blue:#3566b8;--green:#147d64}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f9;color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1500px;margin:auto;padding:32px 25px 75px}}h1{{margin:7px 0}}h2{{margin-top:38px}}h3{{margin:0 0 10px}}a{{color:#245bac}}.tag{{display:inline-block;background:#e7f4ef;color:#10644f;border-radius:999px;padding:5px 10px;font-size:12px}}.muted,small{{color:var(--muted)}}.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.heads{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}canvas{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}.flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:10px;align-items:center}}.flow div{{background:#f7f9fc;border-radius:9px;padding:15px;text-align:center}}.arrow{{font-size:25px;color:var(--blue)}}.shape{{display:block;color:#24539a;font-size:18px;font-weight:750;margin:5px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}.legend{{height:12px;border-radius:8px;background:linear-gradient(90deg,rgb(246,248,252),rgb(64,139,207));margin-top:9px}}.legend-labels{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}}.note{{background:#fff8e7;border-left:5px solid #d49b27;border-radius:8px;padding:14px 17px}}code{{overflow-wrap:anywhere}}@media(max-width:1000px){{.grid,.heads,.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg)}}}}
</style></head><body><main><span class="tag">CPU-only Q-Former replay · exact captured input</span><h1>32个 Q-Former Query 到底读取了哪些帧？</h1><p class="muted">使用此前GPU trace保存的真实 <code>temporal_fused [1,30,1024]</code>，只在CPU重放checkpoint中的两层Q-Former。</p>
<p><a href="frames.html">Attention对齐真实30帧</a> · <a href="../sample_000/index.html">原始30帧</a> · <a href="../sample_000_internal/index.html">完整内部链路</a> · <a href="cross_attention.npz">下载原始attention</a></p>
<div class="card flow"><div><b>30个帧token</b><span class="shape">[1,30,1024]</span></div><span class="arrow">+</span><div><b>32个可学习Query</b><span class="shape">[1,32,768]</span></div><span class="arrow">→</span><div><b>Cross-Attention</b><span class="shape">[2,12,32,30]</span><small>2层 × 12头 × 32query × 30帧</small></div></div>
<div class="note"><b>读图规则：</b>纵轴是Query 1～32，横轴是帧1～30；同一行30个权重之和为1。颜色越深，表示该query在这一层/头里从该帧读取的权重越高。这才是query→frame的attention，不是此前query输出之间的余弦相似度。</div>
<h2>两层 Q-Former</h2><div class="grid"><div class="card"><h3>Layer 1（12头平均）</h3><canvas id="layer0" width="700" height="700"></canvas><div class="legend"></div><div class="legend-labels"><span>低权重</span><span>高权重</span></div><p id="layer0Info" class="muted">悬停查看精确值</p></div><div class="card"><h3>Layer 2（12头平均）</h3><canvas id="layer1" width="700" height="700"></canvas><div class="legend"></div><div class="legend-labels"><span>低权重</span><span>高权重</span></div><p id="layer1Info" class="muted">悬停查看精确值</p></div></div>
<h2>两层、12头全部平均</h2><div class="grid"><div class="card"><canvas id="average" width="760" height="760"></canvas><p id="averageInfo" class="muted">悬停查看精确值</p></div><div class="card"><h3>整体最受关注的帧</h3><canvas id="frameWeights" width="700" height="390"></canvas><p>这是对两层、12头和32个query取平均后的30帧权重；全部帧权重之和为1。</p><table><thead><tr><th>排名</th><th>采样帧</th><th>时间</th><th>权重</th></tr></thead><tbody>{''.join(frame_rows)}</tbody></table></div></div>
<h2>每个 Query 最关注的3帧</h2><div class="card"><table><thead><tr><th>Query</th><th>Top-3帧（两层和所有头平均）</th><th>熵</th></tr></thead><tbody>{''.join(query_rows)}</tbody></table><p class="muted">熵越高表示注意力越平均；熵越低表示更集中。最大可能值为 ln(30)≈3.401。</p></div>
<h2>分开看12个 Attention Head</h2><p>下面每张图已经对两层取平均，但保留不同head。它们可能学习不同的时间读取方式。</p><div class="heads">{''.join(f'<div class="card"><h3>Head {i+1}</h3><canvas id="head{i}" width="520" height="480"></canvas></div>' for i in range(12))}</div>
<div class="note"><b>解释边界：</b>attention权重描述Q-Former读取帧的路径，但不等同于因果重要性。最终文本还经过query输出混合、768→4096投影和Llama解码；要证明某帧对某个词的因果贡献，需要遮挡或干预实验。</div>
<p class="muted">机器证据：<a href="manifest.json">manifest.json</a>。本报告没有运行视觉编码器、Llama、文本生成或训练。</p>
<script>const D={data};
function heat(id,matrix,min,max){{const c=document.getElementById(id),x=c.getContext('2d'),rows=matrix.length,cols=matrix[0].length,w=c.width/cols,h=c.height/rows;x.fillStyle='#fff';x.fillRect(0,0,c.width,c.height);matrix.forEach((row,i)=>row.forEach((v,j)=>{{let t=Math.max(0,Math.min(1,(v-min)/(max-min||1))),r=Math.round(246-182*t),g=Math.round(248-109*t),b=Math.round(252-45*t);x.fillStyle=`rgb(${{r}},${{g}},${{b}})`;x.fillRect(j*w,i*h,w+1,h+1)}}));const info=document.getElementById(id+'Info');if(info)c.onmousemove=e=>{{const z=c.getBoundingClientRect(),col=Math.min(cols-1,Math.floor((e.clientX-z.left)*cols/z.width)),row=Math.min(rows-1,Math.floor((e.clientY-z.top)*rows/z.height));info.innerHTML=`Query ${{row+1}} → 帧 ${{col+1}}：<code>${{(100*matrix[row][col]).toFixed(3)}}%</code>`}}}}
function line(id,v){{const c=document.getElementById(id),x=c.getContext('2d'),p=42,W=c.width-2*p,H=c.height-2*p,lo=Math.min(...v),hi=Math.max(...v);x.strokeStyle='#ccd5e1';x.beginPath();x.moveTo(p,p);x.lineTo(p,p+H);x.lineTo(p+W,p+H);x.stroke();x.strokeStyle='#3566b8';x.lineWidth=3;x.beginPath();v.forEach((n,i)=>{{let px=p+i*W/(v.length-1),py=p+H-(n-lo)*H/(hi-lo||1);i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke();x.fillStyle='#657286';x.fillText(`min ${{(100*lo).toFixed(3)}}% / max ${{(100*hi).toFixed(3)}}%`,p,24)}}
const all=[...D.layer_means.flat(),...D.overall.flat()],lo=Math.min(...all),hi=Math.max(...all);heat('layer0',D.layer_means[0],lo,hi);heat('layer1',D.layer_means[1],lo,hi);heat('average',D.overall,lo,hi);line('frameWeights',D.frame_weights);D.head_means.forEach((m,i)=>{{let f=m.flat();heat('head'+i,m,Math.min(...f),Math.max(...f))}});
</script></main></body></html>"""


def run() -> int:
    monitor = Monitor()
    monitor.start()
    temporary: Path | None = None
    result: dict[str, Any] = {
        "status": "failed",
        "device": "cpu",
        "cuda_visible_devices": "",
        "visual_encoder_loaded": False,
        "llama_loaded": False,
        "text_generated": False,
        "training_executed": False,
    }
    try:
        if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
            raise RuntimeError(f"Wrong Python: {sys.executable}")
        os.chdir(REPO)
        monitor.enter("asset_preflight")
        if CHECKPOINT.stat().st_size != EXPECTED_CHECKPOINT_SIZE:
            raise RuntimeError("Checkpoint size changed")
        for path in (INPUT_NPZ, INPUT_MANIFEST, FRAME_MANIFEST, BERT_ROOT / "config.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        if OUTPUT_DIR.exists():
            raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
        temporary = Path(tempfile.mkdtemp(prefix=".sample_000_qformer_attention.", dir=OUTPUT_PARENT))

        monitor.enter("import_cpu_framework")
        import numpy as np
        import torch
        from research.experiments.commentary_generation.runtime.model.matchvoice_Qformer import (
            BertConfig,
            BertLMHeadModel,
        )
        if torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise RuntimeError("CUDA must be invisible")
        torch.set_num_threads(1)
        with np.load(INPUT_NPZ) as captured:
            temporal = torch.from_numpy(captured["temporal_fused"].copy())
        if tuple(temporal.shape) != (1, EXPECTED_FRAMES, 1024):
            raise RuntimeError(f"Unexpected captured input {temporal.shape}")

        monitor.enter("load_qformer_checkpoint_subset")
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = checkpoint["state_dict"]
        if checkpoint.get("epoch") != EXPECTED_EPOCH or len(state) != EXPECTED_ALL_KEYS:
            raise RuntimeError("Checkpoint identity changed")
        query_tokens = state["video_query_tokens"].clone()
        qformer_state = {
            key.removeprefix("video_Qformer."): value.clone()
            for key, value in state.items()
            if key.startswith("video_Qformer.")
        }
        if len(qformer_state) != EXPECTED_QFORMER_KEYS:
            raise RuntimeError(f"Expected {EXPECTED_QFORMER_KEYS} Q-Former keys, got {len(qformer_state)}")
        del state
        del checkpoint

        monitor.enter("construct_and_load_qformer")
        config = BertConfig.from_pretrained(str(BERT_ROOT), local_files_only=True)
        config.num_hidden_layers = EXPECTED_LAYERS
        config.encoder_width = 1024
        config.add_cross_attention = True
        config.cross_attention_freq = 1
        config.query_length = EXPECTED_QUERIES
        qformer = BertLMHeadModel(config=config)
        qformer.cls = None
        qformer.bert.embeddings.word_embeddings = None
        qformer.bert.embeddings.position_embeddings = None
        for layer in qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        incompatible = qformer.load_state_dict(qformer_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(str(incompatible))
        del qformer_state
        qformer.eval()

        monitor.enter("qformer_cross_attention_forward")
        with torch.no_grad():
            output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=temporal,
                encoder_attention_mask=torch.ones((1, EXPECTED_FRAMES), dtype=torch.long),
                output_attentions=True,
                return_dict=True,
            )
        if output.cross_attentions is None or len(output.cross_attentions) != EXPECTED_LAYERS:
            raise RuntimeError("Cross-attention was not returned")
        attention = torch.stack(output.cross_attentions).float().cpu()
        expected_shape = (EXPECTED_LAYERS, 1, EXPECTED_HEADS, EXPECTED_QUERIES, EXPECTED_FRAMES)
        if tuple(attention.shape) != expected_shape:
            raise RuntimeError(f"Unexpected attention shape {attention.shape}")
        if not torch.isfinite(attention).all().item():
            raise RuntimeError("Attention has non-finite values")
        row_sums = attention.sum(dim=-1)
        max_row_sum_error = float((row_sums - 1).abs().max().item())
        if max_row_sum_error > 1e-5:
            raise RuntimeError(f"Attention rows do not sum to one: {max_row_sum_error}")

        monitor.enter("render_report")
        array = attention[:, 0].numpy()
        layer_means = array.mean(axis=1)
        head_means = array.mean(axis=0)
        overall = array.mean(axis=(0, 1))
        frame_weights = overall.mean(axis=0)
        frame_meta = json.loads(FRAME_MANIFEST.read_text(encoding="utf-8"))
        timestamps = frame_meta["sampling"]["timestamps_seconds"]
        frame_indices = frame_meta["sampling"]["frame_indices"]
        query_top_frames = []
        for query_index, weights in enumerate(overall):
            order = np.argsort(weights)[::-1][:3]
            entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum())
            query_top_frames.append(
                {
                    "query": query_index + 1,
                    "entropy": entropy,
                    "top_frames": [
                        {
                            "frame_position": int(index + 1),
                            "source_frame_index": int(frame_indices[index]),
                            "timestamp_seconds": float(timestamps[index]),
                            "weight": float(weights[index]),
                        }
                        for index in order
                    ],
                }
            )
        global_order = np.argsort(frame_weights)[::-1][:10]
        top_global_frames = [
            {
                "rank": rank + 1,
                "frame_position": int(index + 1),
                "source_frame_index": int(frame_indices[index]),
                "timestamp_seconds": float(timestamps[index]),
                "weight": float(frame_weights[index]),
            }
            for rank, index in enumerate(global_order)
        ]
        np.savez_compressed(
            temporary / "cross_attention.npz",
            cross_attention=array,
            layer_means=layer_means,
            head_means=head_means,
            overall=overall,
            frame_weights=frame_weights,
            query_output=output.last_hidden_state.float().cpu().numpy(),
        )
        manifest = {
            "schema_version": 1,
            "status": "passed",
            "scope": "cpu_only_qformer_cross_attention_replay_from_captured_input",
            "device": "cpu",
            "cuda_visible_devices": "",
            "checkpoint_epoch": EXPECTED_EPOCH,
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_sha256_not_recomputed": True,
            "captured_input_path": str(INPUT_NPZ),
            "captured_input_sha256": sha256(INPUT_NPZ),
            "input_shape": list(temporal.shape),
            "query_tokens_shape": list(query_tokens.shape),
            "attention_shape": list(attention.shape),
            "max_attention_row_sum_error": max_row_sum_error,
            "cross_attention_layers": EXPECTED_LAYERS,
            "attention_heads_per_layer": EXPECTED_HEADS,
            "query_count": EXPECTED_QUERIES,
            "frame_count": EXPECTED_FRAMES,
            "visual_encoder_loaded": False,
            "llama_loaded": False,
            "text_generated": False,
            "training_executed": False,
            "query_top_frames": query_top_frames,
            "top_global_frames": top_global_frames,
            "visualization": {
                "layer_means": layer_means.round(8).tolist(),
                "head_means": head_means.round(8).tolist(),
                "overall": overall.round(8).tolist(),
                "frame_weights": frame_weights.round(8).tolist(),
            },
            "artifacts": {
                "html": "index.html",
                "raw_attention": "cross_attention.npz",
            },
            "limitations": [
                "Attention weights are not causal attribution.",
                "Layer/head averages hide per-head variation; raw values are preserved in NPZ.",
                "This replay starts at the exact captured Q-Former input and does not rerun the visual encoder.",
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "index.html").write_text(build_html(manifest), encoding="utf-8")
        manifest["artifacts"]["raw_attention_bytes"] = (temporary / "cross_attention.npz").stat().st_size
        manifest["artifacts"]["raw_attention_sha256"] = sha256(temporary / "cross_attention.npz")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        monitor.enter("publish_atomically")
        temporary.rename(OUTPUT_DIR)
        temporary = None
        result.update(
            status="passed",
            html=str(OUTPUT_DIR / "index.html"),
            manifest=str(OUTPUT_DIR / "manifest.json"),
            attention_shape=list(attention.shape),
            max_attention_row_sum_error=max_row_sum_error,
        )
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        if temporary is not None:
            result["incomplete_temporary_dir"] = str(temporary)
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
