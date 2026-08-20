#!/usr/bin/env python3
"""Build a static, browser-only track annotation page for SNGS-10004."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
ARCHIVE = REPO / ".runtime/g10/sngs10004_prerefiner_enrichment/run2/states/sn-gamestate.pklz"
REPORT_DIR = REPO / "reports/g10/20260818_team_color_diagnostic"
RESULT = REPORT_DIR / "result.json"
CROP_DIR = REPORT_DIR / "annotation_crops"
PAGE = REPORT_DIR / "annotate_tracks.html"
VIDEO_ID = "10004"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def crop_with_context(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    pad_x, pad_y = max(16, 0.45 * width), max(12, 0.16 * height)
    return image.crop(
        (
            max(0, int(math.floor(x - pad_x))),
            max(0, int(math.floor(y - pad_y))),
            min(image.width, int(math.ceil(x + width + pad_x))),
            min(image.height, int(math.ceil(y + height + pad_y))),
        )
    ).convert("RGB")


def crop_with_target_box(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    pad_x, pad_y = max(16, 0.45 * width), max(12, 0.16 * height)
    left = max(0, int(math.floor(x - pad_x)))
    top = max(0, int(math.floor(y - pad_y)))
    right = min(image.width, int(math.ceil(x + width + pad_x)))
    bottom = min(image.height, int(math.ceil(y + height + pad_y)))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        (
            max(0, int(round(x - left))),
            max(0, int(round(y - top))),
            min(crop.width - 1, int(round(x + width - left))),
            min(crop.height - 1, int(round(y + height - top))),
        ),
        outline="#ff3bd4",
        width=max(2, int(round(min(crop.width, crop.height) / 45))),
    )
    return crop


def select_representatives(group: pd.DataFrame) -> pd.DataFrame:
    """Choose up to three high-confidence observations spread across a track."""
    ordered = group.sort_values("image_id", kind="stable")
    if len(ordered) <= 3:
        return ordered
    selections = []
    for positions in np.array_split(np.arange(len(ordered)), 3):
        part = ordered.iloc[positions]
        selections.append(part.loc[part.bbox_conf.astype(float).idxmax()])
    return pd.DataFrame(selections).drop_duplicates(subset=["image_id"])


def make_strip(rows: pd.DataFrame, path_by_id: dict[Any, str]) -> Image.Image:
    tile_w, tile_h = 210, 250
    strip = Image.new("RGB", (tile_w * len(rows), tile_h), "#161d25")
    for position, row in enumerate(rows.itertuples(index=False)):
        image = Image.open(Path(path_by_id[row.image_id])).convert("RGB")
        crop = crop_with_target_box(image, row.bbox_ltwh)
        crop.thumbnail((tile_w - 12, tile_h - 34), Image.Resampling.LANCZOS)
        x = position * tile_w + (tile_w - crop.width) // 2
        strip.paste(crop, (x, 4))
        draw = ImageDraw.Draw(strip)
        draw.text(
            (position * tile_w + 8, tile_h - 25),
            f"frame {int(str(row.image_id)[-6:])}",
            fill="#dce6ef",
            font=font(14),
        )
    return strip


def build_html(
    tracks: list[dict[str, Any]], video_id: str = "10004", priority_count: int = 33
) -> str:
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SNGS-__VIDEO_ID__ 轨迹标注</title>
  <style>
    :root { color-scheme: dark; --bg:#10151b; --panel:#1b232d; --line:#344150; --text:#eef4fa; --muted:#aebdcb; --accent:#70b7ff; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }
    header { position:sticky; top:0; z-index:3; padding:14px 20px; background:#10151bf2; border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 5px; font-size:22px; }
    .sub, .meta { color:var(--muted); }
    .toolbar { display:flex; gap:9px; flex-wrap:wrap; align-items:center; margin-top:10px; }
    button, select { border:1px solid #536274; background:#202a35; color:var(--text); border-radius:7px; padding:7px 10px; cursor:pointer; }
    button:hover { border-color:var(--accent); }
    #progress { font-weight:700; color:#9ed0ff; }
    main { display:grid; grid-template-columns:repeat(auto-fill,minmax(440px,1fr)); gap:14px; padding:18px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .card.complete { border-color:#4f936d; }
    .card img { width:100%; height:235px; object-fit:contain; display:block; background:#111820; }
    .body { padding:11px; }
    .title { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:17px; font-weight:700; }
    .badge { padding:2px 7px; border-radius:999px; font-size:12px; background:#273443; color:#bdd0e0; }
    .badge.eligible { background:#1f4937; color:#b9f0d2; }
    .field { margin-top:9px; }
    .field-label { display:block; color:var(--muted); margin-bottom:5px; }
    .choices { display:flex; flex-wrap:wrap; gap:6px; }
    .choice.selected { background:#185a8c; border-color:#70b7ff; }
    .note { width:100%; margin-top:10px; padding:7px; border:1px solid #465466; border-radius:6px; background:#111820; color:var(--text); }
    .hidden { display:none !important; }
    #jsonPreview { width:100%; height:150px; margin-top:8px; background:#0d1217; color:#cfe2f3; border:1px solid var(--line); display:none; }
  </style>
</head>
<body>
<header>
  <h1>SNGS-__VIDEO_ID__ 轨迹级角色/球队标注</h1>
  <div class="sub">每条轨迹看三张跨时间裁剪，洋红框表示目标。不要参考模型聚类；看不清就选“无法判断”。浏览器自动暂存，完成后导出 JSON。</div>
  <div class="toolbar">
    <span id="progress">0 / __TRACK_COUNT__ 已完成</span>
    <select id="scope">__SCOPE_OPTIONS__<option value="incomplete">只显示未完成</option></select>
    <button id="export">导出 JSON</button>
    <button id="copy">复制 JSON</button>
    <button id="togglePreview">显示/隐藏 JSON</button>
  </div>
  <textarea id="jsonPreview" readonly></textarea>
</header>
<main id="cards"></main>
<script>
const tracks = __TRACKS_JSON__;
const storageKey = 'sngs__VIDEO_ID__-track-labels-v1';
let labels = {};
try { labels = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) { labels = {}; }

const roleChoices = [
  ['outfield_player','外场球员'], ['goalkeeper','门将'],
  ['referee_or_staff','裁判/工作人员'], ['uncertain','无法判断']
];
const teamChoices = [['blue','蓝队'], ['claret','酒红队'], ['unknown_or_na','未知/不适用']];

function valueFor(id) { return labels[id] || {role:'', team:'', note:''}; }
function isComplete(value) { return Boolean(value.role && value.team); }
function save() {
  try { localStorage.setItem(storageKey, JSON.stringify(labels)); } catch (_) {}
  refresh();
}
function setField(id, field, value) {
  const current = valueFor(id);
  labels[id] = {...current, [field]:value};
  if (field === 'role' && (value === 'referee_or_staff' || value === 'uncertain')) labels[id].team = 'unknown_or_na';
  save();
}
function button(id, field, value, text) {
  const selected = valueFor(id)[field] === value ? ' selected' : '';
  return `<button class="choice${selected}" data-id="${id}" data-field="${field}" data-value="${value}">${text}</button>`;
}
function renderCard(track) {
  const value = valueFor(track.track_id);
  return `<article class="card ${isComplete(value)?'complete':''}" data-id="${track.track_id}" data-eligible="${track.eligible}">
    <img src="${track.image}" alt="轨迹 ${track.track_id} 的代表性裁剪">
    <div class="body">
      <div class="title"><span>轨迹 ${track.track_id}</span><span class="badge ${track.eligible?'eligible':''}">${track.eligible?'优先标注':'补充轨迹'} · ${track.rows}帧</span></div>
      <div class="field"><span class="field-label">角色</span><div class="choices">${roleChoices.map(x=>button(track.track_id,'role',x[0],x[1])).join('')}</div></div>
      <div class="field"><span class="field-label">球队</span><div class="choices">${teamChoices.map(x=>button(track.track_id,'team',x[0],x[1])).join('')}</div></div>
      <input class="note" data-note="${track.track_id}" value="${(value.note||'').replaceAll('&','&amp;').replaceAll('"','&quot;')}" placeholder="可选备注">
    </div>
  </article>`;
}
function payload() {
  return {
    schema_version: 1,
    video_id: '__VIDEO_ID__',
    unit: 'track',
    exported_at: new Date().toISOString(),
    labels: tracks.map(t => ({track_id:t.track_id, eligible:t.eligible, ...valueFor(t.track_id)}))
  };
}
function refresh() {
  const done = tracks.filter(t=>isComplete(valueFor(t.track_id))).length;
  document.querySelector('#progress').textContent = `${done} / ${tracks.length} 已完成（优先轨迹 ${tracks.filter(t=>t.eligible&&isComplete(valueFor(t.track_id))).length} / __PRIORITY_COUNT__）`;
  document.querySelector('#cards').innerHTML = tracks.map(renderCard).join('');
  document.querySelectorAll('[data-field]').forEach(el => el.onclick = () => setField(Number(el.dataset.id), el.dataset.field, el.dataset.value));
  document.querySelectorAll('[data-note]').forEach(el => el.onchange = () => { const id=Number(el.dataset.note); labels[id]={...valueFor(id),note:el.value}; save(); });
  applyScope();
  document.querySelector('#jsonPreview').value = JSON.stringify(payload(), null, 2);
}
function applyScope() {
  const scope = document.querySelector('#scope').value;
  document.querySelectorAll('.card').forEach(card => {
    const id = Number(card.dataset.id);
    const show = scope === 'all' || (scope === 'eligible' && card.dataset.eligible === 'true') || (scope === 'incomplete' && !isComplete(valueFor(id)));
    card.classList.toggle('hidden', !show);
  });
}
document.querySelector('#scope').onchange = applyScope;
document.querySelector('#export').onclick = () => {
  const blob = new Blob([JSON.stringify(payload(), null, 2)+'\n'], {type:'application/json'});
  const url = URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='sngs__VIDEO_ID___track_annotations.json'; a.click(); URL.revokeObjectURL(url);
};
document.querySelector('#copy').onclick = async () => { await navigator.clipboard.writeText(JSON.stringify(payload(), null, 2)); };
document.querySelector('#togglePreview').onclick = () => { const p=document.querySelector('#jsonPreview'); p.style.display=p.style.display==='block'?'none':'block'; };
refresh();
</script>
</body>
</html>
'''
    if priority_count < len(tracks):
        scope_options = (
            f'<option value="eligible">先标{priority_count}条诊断轨迹</option>'
            f'<option value="all">显示全部{len(tracks)}条</option>'
        )
    else:
        scope_options = f'<option value="all">显示全部{len(tracks)}条</option>'
    return (
        template.replace("__TRACKS_JSON__", json.dumps(tracks, ensure_ascii=False))
        .replace("__VIDEO_ID__", video_id)
        .replace("__TRACK_COUNT__", str(len(tracks)))
        .replace("__PRIORITY_COUNT__", str(priority_count))
        .replace("__SCOPE_OPTIONS__", scope_options)
    )


def main() -> None:
    if PAGE.exists() or CROP_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite annotation outputs: {PAGE}, {CROP_DIR}")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    eligible = set(result["color_role_length_filtered"]["eligible_track_ids"])
    with zipfile.ZipFile(ARCHIVE) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    path_by_id = {row.id: str(row.file_path) for row in images.itertuples(index=False)}
    CROP_DIR.mkdir(parents=True)
    tracks = []
    for track_id, group in detections.groupby("track_id"):
        numeric_id = int(float(track_id))
        selected = select_representatives(group)
        relative = f"annotation_crops/track_{numeric_id:03d}.jpg"
        make_strip(selected, path_by_id).save(REPORT_DIR / relative, quality=92)
        tracks.append(
            {
                "track_id": numeric_id,
                "rows": int(len(group)),
                "eligible": numeric_id in eligible,
                "image": relative,
            }
        )
    tracks.sort(key=lambda item: (not item["eligible"], item["track_id"]))
    if len(tracks) != 49 or sum(item["eligible"] for item in tracks) != 33:
        raise AssertionError("Unexpected track contract")
    PAGE.write_text(build_html(tracks), encoding="utf-8")
    print(json.dumps({"status": "passed", "page": str(PAGE), "tracks": 49, "eligible": 33}))


if __name__ == "__main__":
    main()
