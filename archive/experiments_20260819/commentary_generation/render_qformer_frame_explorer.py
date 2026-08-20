#!/usr/bin/env python3
"""Render an interactive frame explorer from existing Q-Former evidence.

No model, Torch, checkpoint, inference, or GPU is used.  The 30 thumbnails are
cropped from the already verified contact sheet.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path("/home/tianlin/SoccerMaster")
ATTENTION_ROOT = REPO / "reports/commentary_trace/sample_000_qformer_attention"
ATTENTION_MANIFEST = ATTENTION_ROOT / "manifest.json"
FRAME_MANIFEST = REPO / "reports/commentary_trace/sample_000/manifest.json"
CONTACT_SHEET = REPO / "reports/commentary_trace/sample_000/01_sampled_frames.jpg"
OUTPUT_HTML = ATTENTION_ROOT / "frames.html"
OUTPUT_MANIFEST = ATTENTION_ROOT / "frames_manifest.json"
OUTPUT_FRAMES = ATTENTION_ROOT / "frames"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_html(attention: dict, frames: dict) -> str:
    overall = attention["visualization"]["overall"]
    global_weights = attention["visualization"]["frame_weights"]
    timestamps = frames["sampling"]["timestamps_seconds"]
    source_indices = frames["sampling"]["frame_indices"]
    cards = []
    for index in range(30):
        cards.append(
            f"<article class='frame' id='frame{index}'><div class='rank'></div>"
            f"<img src='frames/frame_{index+1:02d}.jpg' alt='sampled frame {index+1}'>"
            f"<div class='meta'><b>采样帧 {index+1}</b><span>原帧 {source_indices[index]} · {timestamps[index]:.2f}s</span>"
            f"<div class='bar'><i></i></div><strong class='weight'></strong></div></article>"
        )
    payload = json.dumps(
        {
            "overall": overall,
            "global": global_weights,
            "timestamps": timestamps,
            "source_indices": source_indices,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q-Former Attention 对齐真实30帧</title><style>
:root{{--ink:#162033;--muted:#657286;--line:#dce3eb;--blue:#3566b8;--green:#147d64;--gold:#c58a14}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f9;color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1500px;margin:auto;padding:32px 24px 75px}}h1{{margin:7px 0}}a{{color:#245bac}}.tag{{display:inline-block;background:#e7f4ef;color:#10644f;border-radius:999px;padding:5px 10px;font-size:12px}}.muted{{color:var(--muted)}}.control{{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:12px;padding:15px;margin:20px 0;box-shadow:0 5px 18px #26364a16;display:flex;flex-wrap:wrap;align-items:center;gap:14px}}select{{font:inherit;padding:8px 12px;border:1px solid var(--line);border-radius:7px}}.summary{{font-weight:650}}.frames{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}.frame{{position:relative;background:white;border:2px solid transparent;border-radius:11px;overflow:hidden;box-shadow:0 2px 8px #26364a13;transition:.15s}}.frame.top{{border-color:var(--gold);transform:translateY(-2px)}}.frame img{{display:block;width:100%;height:auto}}.meta{{padding:10px}}.meta b,.meta span{{display:block}}.meta span{{font-size:12px;color:var(--muted);margin:3px 0 8px}}.bar{{height:10px;background:#e7ecf3;border-radius:9px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--blue);border-radius:9px}}.weight{{display:block;color:#245bac;margin-top:5px}}.rank{{display:none;position:absolute;right:7px;top:7px;background:var(--gold);color:white;border-radius:999px;padding:5px 8px;font-weight:750;z-index:2}}.frame.top .rank{{display:block}}.note{{background:#fff8e7;border-left:5px solid #d49b27;border-radius:8px;padding:14px 17px;margin:20px 0}}@media(max-width:1050px){{.frames{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:650px){{.frames{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><span class="tag">Existing attention + verified frame contact sheet · no model rerun</span><h1>把 Q-Former Attention 放回真实比赛画面</h1><p class="muted">保持30帧时间顺序；切换Query后，权重条和金色Top-3标记会同步变化。</p><p><a href="index.html">返回完整attention图</a> · <a href="../sample_000/index.html">返回输入报告</a> · <a href="cross_attention.npz">下载原始attention</a></p>
<div class="control"><label for="query"><b>查看：</b></label><select id="query"><option value="global">全部Query综合</option>{''.join(f'<option value="{i}">Query {i+1}</option>' for i in range(32))}</select><span class="summary" id="summary"></span><span class="muted">均匀基线：3.333%</span></div>
<section class="frames">{''.join(cards)}</section><div class="note"><b>怎样理解：</b>金色只表示当前选择下权重最高的3帧。单个Query的30个权重和为100%；综合模式则对32个Query、12个head和2层全部平均。Attention是读取路径，不等同于该帧对最终文本的因果贡献。</div>
<p class="muted">证据清单：<a href="frames_manifest.json">frames_manifest.json</a>。</p><script>const D={payload},select=document.getElementById('query');function show(){{const global=select.value==='global',weights=global?D.global:D.overall[Number(select.value)],order=weights.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]),top=new Map(order.slice(0,3).map((x,i)=>[x[1],i+1])),max=Math.max(...weights),sum=weights.reduce((a,b)=>a+b,0);weights.forEach((v,i)=>{{const card=document.getElementById('frame'+i),rank=card.querySelector('.rank');card.querySelector('.bar i').style.width=(100*v/max)+'%';card.querySelector('.weight').textContent=(100*v).toFixed(3)+'%';card.classList.toggle('top',top.has(i));rank.textContent=top.has(i)?'Top '+top.get(i):''}});document.getElementById('summary').textContent=`权重和 ${{(100*sum).toFixed(4)}}% · 最高帧 ${{order[0][1]+1}}（${{D.timestamps[order[0][1]].toFixed(2)}}s） ${{(100*order[0][0]).toFixed(3)}}%`}}select.addEventListener('change',show);show();</script></main></body></html>"""


def main() -> int:
    from PIL import Image

    for path in (ATTENTION_MANIFEST, FRAME_MANIFEST, CONTACT_SHEET):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (OUTPUT_HTML, OUTPUT_MANIFEST, OUTPUT_FRAMES):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    attention = json.loads(ATTENTION_MANIFEST.read_text(encoding="utf-8"))
    frames = json.loads(FRAME_MANIFEST.read_text(encoding="utf-8"))
    if attention.get("status") != "passed" or frames.get("status") != "passed":
        raise RuntimeError("Source evidence must be passed")
    temporary = Path(tempfile.mkdtemp(prefix=".qformer_frames.", dir=ATTENTION_ROOT))
    temp_frames = temporary / "frames"
    temp_frames.mkdir()
    with Image.open(CONTACT_SHEET) as sheet:
        if sheet.size != (1650, 1304):
            raise RuntimeError(f"Unexpected contact sheet size {sheet.size}")
        for index in range(30):
            row, column = divmod(index, 5)
            left = 14 + column * (318 + 8)
            top = 14 + row * (179 + 27 + 8)
            crop = sheet.crop((left, top, left + 318, top + 179 + 27))
            crop.save(
                temp_frames / f"frame_{index+1:02d}.jpg",
                format="JPEG",
                quality=92,
                subsampling=0,
            )
    temp_html = temporary / "frames.html"
    temp_html.write_text(build_html(attention, frames), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "scope": "interactive_frame_explorer_from_existing_attention_and_contact_sheet",
        "model_imported": False,
        "gpu_used": False,
        "inference_executed": False,
        "sources": {
            str(ATTENTION_MANIFEST): {"bytes": ATTENTION_MANIFEST.stat().st_size, "sha256": sha256(ATTENTION_MANIFEST)},
            str(FRAME_MANIFEST): {"bytes": FRAME_MANIFEST.stat().st_size, "sha256": sha256(FRAME_MANIFEST)},
            str(CONTACT_SHEET): {"bytes": CONTACT_SHEET.stat().st_size, "sha256": sha256(CONTACT_SHEET)},
        },
        "html": {"path": "frames.html", "bytes": temp_html.stat().st_size, "sha256": sha256(temp_html)},
        "thumbnails": [
            {"path": f"frames/frame_{i+1:02d}.jpg", "bytes": (temp_frames / f"frame_{i+1:02d}.jpg").stat().st_size, "sha256": sha256(temp_frames / f"frame_{i+1:02d}.jpg")}
            for i in range(30)
        ],
    }
    temp_manifest = temporary / "frames_manifest.json"
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_frames.rename(OUTPUT_FRAMES)
    temp_html.rename(OUTPUT_HTML)
    temp_manifest.rename(OUTPUT_MANIFEST)
    temporary.rmdir()
    print(json.dumps({"status": "passed", "html": str(OUTPUT_HTML), "thumbnails": 30}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
