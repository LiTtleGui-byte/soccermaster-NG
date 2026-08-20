#!/usr/bin/env python3
"""Build a browser-only, prediction-blind role annotation page for SNGS-10002."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/g10/20260819_qwen_role_newmatch_sngs10002"
INPUT_MANIFEST = REPORT / "input_manifest.json"
PAGE = REPORT / "blind_role_annotations.html"
BLIND_MANIFEST = REPORT / "blind_annotation_manifest.json"
CROP_DIR = REPORT / "blind_annotation_crops"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def boxed_full_frame(image: Image.Image, bbox: list[float]) -> Image.Image:
    result = image.copy()
    x, y, width, height = bbox
    draw = ImageDraw.Draw(result)
    line_width = max(4, round(min(image.size) / 220))
    draw.rectangle((x, y, x + width, y + height), outline="#ff3bd4", width=line_width)
    return result


def context_crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    x, y, width, height = bbox
    pad_x = max(24.0, width * 0.65)
    pad_y = max(18.0, height * 0.25)
    left = max(0, math.floor(x - pad_x))
    top = max(0, math.floor(y - pad_y))
    right = min(image.width, math.ceil(x + width + pad_x))
    bottom = min(image.height, math.ceil(y + height + pad_y))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (x - left, y - top, x + width - left, y + height - top),
        outline="#ff3bd4",
        width=max(3, round(min(crop.size) / 60)),
    )
    return crop


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#101820")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_track_strip(samples: list[dict[str, Any]]) -> Image.Image:
    cell_w, top_h, crop_h, footer_h = 510, 185, 205, 35
    strip = Image.new("RGB", (cell_w * len(samples), top_h + crop_h + footer_h), "#101820")
    draw = ImageDraw.Draw(strip)
    for position, sample in enumerate(samples):
        image = Image.open(sample["image_path_read_only"]).convert("RGB")
        full = fit(boxed_full_frame(image, sample["bbox_ltwh"]), (cell_w, top_h))
        crop = fit(context_crop(image, sample["bbox_ltwh"]), (cell_w, crop_h))
        x = position * cell_w
        strip.paste(full, (x, 0))
        strip.paste(crop, (x, top_h))
        draw.line((x, 0, x, top_h + crop_h), fill="#344150", width=2)
        draw.text(
            (x + 12, top_h + crop_h + 7),
            f"view {sample['view_ordinal']}  ·  frame {sample['frame'] + 1}",
            fill="#dce7f1",
            font=font(16),
        )
        image.close()
    return strip


def build_html(tracks: list[dict[str, Any]]) -> str:
    tracks_json = json.dumps(tracks, ensure_ascii=False)
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SNGS-10002 角色盲标</title>
  <style>
    :root { color-scheme:dark; --bg:#0e141a; --panel:#19222c; --line:#344454; --text:#edf4fa; --muted:#adbdca; --accent:#6eb8ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }
    header { position:sticky; top:0; z-index:3; padding:14px 20px; background:#0e141af2; border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 5px; font-size:22px; }
    .sub { color:var(--muted); max-width:1050px; }
    .toolbar { display:flex; gap:9px; flex-wrap:wrap; align-items:center; margin-top:10px; }
    button, select { border:1px solid #536477; background:#202c38; color:var(--text); border-radius:7px; padding:7px 10px; cursor:pointer; }
    button:hover { border-color:var(--accent); }
    #progress { font-weight:700; color:#9ed0ff; }
    main { display:grid; grid-template-columns:repeat(auto-fill,minmax(620px,1fr)); gap:14px; padding:18px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .card.complete { border-color:#4c9a6d; }
    .card img { display:block; width:100%; height:360px; object-fit:contain; background:#101820; }
    .body { padding:11px; }
    .title { display:flex; justify-content:space-between; margin-bottom:9px; font-size:17px; font-weight:700; }
    .meta { color:var(--muted); font-size:13px; }
    .choices { display:flex; gap:7px; flex-wrap:wrap; }
    .choice.selected { background:#185a8c; border-color:#77c0ff; }
    .note { width:100%; margin-top:10px; padding:7px; border:1px solid #46586a; border-radius:6px; background:#101820; color:var(--text); }
    .hidden { display:none; }
    #preview { display:none; width:100%; height:150px; margin-top:8px; background:#0a1015; color:#cfe2f3; border:1px solid var(--line); }
  </style>
</head>
<body>
<header>
  <h1>SNGS-10002 轨迹角色盲标</h1>
  <div class="sub">洋红框是目标人物。每列上方是整帧位置，下方是放大裁剪。页面没有读取或显示Qwen预测；不确定时请选择“看不清”，不要猜。</div>
  <div class="toolbar">
    <span id="progress">0 / 33 已完成</span>
    <select id="scope"><option value="all">显示全部</option><option value="incomplete">只显示未完成</option></select>
    <button id="export">导出 JSON</button>
    <button id="copy">复制 JSON</button>
    <button id="toggle">显示/隐藏 JSON</button>
  </div>
  <textarea id="preview" readonly></textarea>
</header>
<main id="cards"></main>
<script>
const tracks = __TRACKS__;
const choices = [['outfield_player','普通外场球员'],['goalkeeper','门将'],['referee_or_staff','裁判/工作人员'],['uncertain','看不清']];
const storageKey = 'sngs10002-role-blind-v1';
let labels = {};
try { labels = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) { labels = {}; }
function current(id) { return labels[id] || {role:'',note:''}; }
function save() { localStorage.setItem(storageKey, JSON.stringify(labels)); render(); }
function setRole(id, role) { labels[id] = {...current(id),role}; save(); }
function payload() { return {schema_version:1,video_id:'10002',unit:'track',blind_to_qwen_predictions:true,exported_at:new Date().toISOString(),labels:tracks.map(t=>({track_id:t.track_id,...current(t.track_id)}))}; }
function card(track) {
  const value=current(track.track_id);
  const buttons=choices.map(([v,t])=>`<button class="choice ${value.role===v?'selected':''}" data-id="${track.track_id}" data-role="${v}">${t}</button>`).join('');
  const note=(value.note||'').replaceAll('&','&amp;').replaceAll('"','&quot;');
  return `<article class="card ${value.role?'complete':''}" data-id="${track.track_id}"><img src="${track.image}" alt="轨迹${track.track_id}"><div class="body"><div class="title"><span>轨迹 ${track.track_id}</span><span class="meta">${track.rows}个检测 · ${track.views}个视图</span></div><div class="choices">${buttons}</div><input class="note" data-note="${track.track_id}" value="${note}" placeholder="可选备注"></div></article>`;
}
function applyScope() { const scope=document.querySelector('#scope').value; document.querySelectorAll('.card').forEach(el=>el.classList.toggle('hidden',scope==='incomplete'&&Boolean(current(Number(el.dataset.id)).role))); }
function render() {
  document.querySelector('#cards').innerHTML=tracks.map(card).join('');
  document.querySelectorAll('[data-role]').forEach(el=>el.onclick=()=>setRole(Number(el.dataset.id),el.dataset.role));
  document.querySelectorAll('[data-note]').forEach(el=>el.onchange=()=>{const id=Number(el.dataset.note);labels[id]={...current(id),note:el.value};save();});
  document.querySelector('#progress').textContent=`${tracks.filter(t=>current(t.track_id).role).length} / ${tracks.length} 已完成`;
  document.querySelector('#preview').value=JSON.stringify(payload(),null,2); applyScope();
}
document.querySelector('#scope').onchange=applyScope;
document.querySelector('#export').onclick=()=>{const blob=new Blob([JSON.stringify(payload(),null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='sngs10002_blind_role_annotations.json';a.click();URL.revokeObjectURL(url);};
document.querySelector('#copy').onclick=()=>navigator.clipboard.writeText(JSON.stringify(payload(),null,2));
document.querySelector('#toggle').onclick=()=>{const p=document.querySelector('#preview');p.style.display=p.style.display==='block'?'none':'block';};
render();
</script>
</body>
</html>'''
    return template.replace("__TRACKS__", tracks_json)


def main() -> None:
    if PAGE.exists() or BLIND_MANIFEST.exists() or CROP_DIR.exists():
        raise FileExistsError("Refusing to overwrite blind annotation outputs")
    manifest = json.loads(INPUT_MANIFEST.read_text())
    if manifest.get("video_id") != "10002" or manifest.get("tracks") != 33 or manifest.get("samples") != 89:
        raise AssertionError("Input manifest identity changed")
    if manifest.get("manual_annotations_read") is not False or manifest.get("historical_role_fields_read_by_gpu") is not False:
        raise AssertionError("Input manifest is not label-blind")

    by_track: dict[int, list[dict[str, Any]]] = {}
    rows_by_track = {int(row["track_id"]): int(row["archive_rows"]) for row in manifest["track_summary"]}
    for sample in manifest["sample_manifest"]:
        by_track.setdefault(int(sample["track_id"]), []).append(sample)
    if len(by_track) != 33:
        raise AssertionError("Expected exactly 33 tracks")

    CROP_DIR.mkdir(parents=True)
    tracks = []
    blind_records = []
    for track_id in sorted(by_track):
        samples = sorted(by_track[track_id], key=lambda row: row["view_ordinal"])
        filename = f"track_{track_id:03d}.jpg"
        make_track_strip(samples).save(CROP_DIR / filename, quality=92)
        tracks.append(
            {
                "track_id": track_id,
                "rows": rows_by_track[track_id],
                "views": len(samples),
                "image": f"blind_annotation_crops/{filename}",
            }
        )
        blind_records.append(
            {
                "track_id": track_id,
                "sample_ids": [sample["sample_id"] for sample in samples],
                "frames_one_based": [sample["frame"] + 1 for sample in samples],
            }
        )
    PAGE.write_text(build_html(tracks), encoding="utf-8")
    BLIND_MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "prepared",
                "video_id": "10002",
                "tracks": 33,
                "views": 89,
                "qwen_predictions_read": False,
                "historical_role_fields_read": False,
                "source_input_manifest": str(INPUT_MANIFEST),
                "role_choices": ["outfield_player", "goalkeeper", "referee_or_staff", "uncertain"],
                "records": blind_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "page": str(PAGE), "tracks": 33, "views": 89, "qwen_predictions_read": False}))


if __name__ == "__main__":
    main()
