#!/usr/bin/env python3
"""Render the existing 200-sample decoding ablation as a self-contained HTML report.

This script is intentionally standard-library-only. It reads existing JSON evidence,
does not import project or machine-learning modules, and never opens video/model files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO / "reports/commentary_decode_ablation_200_20260814"
DEFAULT_OUTPUT_HTML = DEFAULT_INPUT_DIR / "index.html"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_INPUT_DIR / "report_manifest.json"

STRATEGIES = (
    "historical_beam_sampling",
    "deterministic_beam",
    "nucleus_sampling",
)
STRATEGY_LABELS = {
    "historical_beam_sampling": "历史 Beam + Sampling",
    "deterministic_beam": "确定性 Beam",
    "nucleus_sampling": "Nucleus Sampling",
}

TOKEN_RE = re.compile(r"\[\w+\]|[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
EVENT_CUES = {
    "进球": ("goal", "scores", "scored", "net"),
    "射门": ("shot", "shoot", "fires", "effort", "volley", "strike"),
    "扑救": ("save", "keeper", "goalkeeper"),
    "犯规": ("foul", "whistle", "free kick", "handball"),
    "黄牌": ("yellow card", "booked"),
    "红牌": ("red card", "sent off"),
    "越位": ("offside",),
    "角球": ("corner",),
    "换人": ("substitution", "substitute", "replaces", "replaced"),
    "传中": ("cross", "whipped in"),
    "传球": ("pass", "through ball"),
    "界外球": ("throw in", "throw-in"),
    "点球": ("penalty",),
    "受伤": ("injury", "injured", "medical attention"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the two report files owned by this renderer.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = Counter(tokenize(reference))
    prediction_tokens = Counter(tokenize(prediction))
    overlap = sum((reference_tokens & prediction_tokens).values())
    if not reference_tokens or not prediction_tokens or overlap == 0:
        return 0.0
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def extract_cues(text: str) -> list[str]:
    lowered = text.lower()
    return [
        label
        for label, needles in EVENT_CUES.items()
        if any(needle in lowered for needle in needles)
    ]


def disagreement(predictions: dict[str, str]) -> float:
    values = [predictions[name].lower().strip() for name in STRATEGIES]
    distances = [
        1.0 - SequenceMatcher(None, values[i], values[j]).ratio()
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return sum(distances) / len(distances)


def load_rows(predictions_path: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    raw_rows = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL line {line_number}: {exc}") from exc

    expected_count = int(result["samples_completed"])
    if len(raw_rows) != expected_count or expected_count != 200:
        raise ValueError(
            f"expected exactly 200 completed rows, got {len(raw_rows)} "
            f"(result says {expected_count})"
        )

    duplicate_counts = {
        strategy: Counter(
            row["predictions"][strategy].strip() for row in raw_rows
        )
        for strategy in STRATEGIES
    }

    rows = []
    seen_ordinals: set[int] = set()
    seen_indices: set[int] = set()
    for row in raw_rows:
        ordinal = int(row["ordinal"])
        dataset_index = int(row["dataset_index"])
        if ordinal in seen_ordinals or dataset_index in seen_indices:
            raise ValueError(f"duplicate row identity: ordinal={ordinal}, index={dataset_index}")
        seen_ordinals.add(ordinal)
        seen_indices.add(dataset_index)

        reference = str(row["reference_commentary"]).strip()
        predictions = {
            strategy: str(row["predictions"][strategy]).strip()
            for strategy in STRATEGIES
        }
        if not reference or any(not value for value in predictions.values()):
            raise ValueError(f"empty reference/prediction at dataset index {dataset_index}")

        reference_cues = extract_cues(reference)
        strategy_details = {}
        for strategy, prediction in predictions.items():
            prediction_cues = extract_cues(prediction)
            shared_cues = sorted(set(reference_cues) & set(prediction_cues))
            strategy_details[strategy] = {
                "text": prediction,
                "token_f1": token_f1(reference, prediction),
                "duplicate_count": duplicate_counts[strategy][prediction],
                "cues": prediction_cues,
                "shared_cues": shared_cues,
                "cue_match": bool(shared_cues) if reference_cues else None,
                "generation_seconds": float(row["generation_seconds"][strategy]),
            }

        normalized = {name: text.lower() for name, text in predictions.items()}
        historical_score = strategy_details["historical_beam_sampling"]["token_f1"]
        deterministic_score = strategy_details["deterministic_beam"]["token_f1"]
        nucleus_score = strategy_details["nucleus_sampling"]["token_f1"]
        rows.append(
            {
                "ordinal": ordinal,
                "dataset_index": dataset_index,
                "sample_seed": int(row["sample_seed"]),
                "video_path": str(row["video_path"]),
                "video_name": Path(str(row["video_path"])).name,
                "reference": reference,
                "reference_cues": reference_cues,
                "strategies": strategy_details,
                "disagreement": disagreement(predictions),
                "all_same": len(set(normalized.values())) == 1,
                "historical_equals_deterministic": (
                    normalized["historical_beam_sampling"]
                    == normalized["deterministic_beam"]
                ),
                "nucleus_changed": (
                    normalized["nucleus_sampling"]
                    != normalized["historical_beam_sampling"]
                ),
                "max_duplicate_count": max(
                    detail["duplicate_count"] for detail in strategy_details.values()
                ),
                "nucleus_overlap_delta": nucleus_score
                - max(historical_score, deterministic_score),
            }
        )

    if seen_ordinals != set(range(1, 201)):
        raise ValueError("ordinals are not exactly 1..200")
    return rows


def strategy_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for strategy in STRATEGIES:
        source = result["strategy_summaries"][strategy]
        diversity = source["diversity"]
        overlap = source["overlap"]
        summaries.append(
            {
                "key": strategy,
                "label": STRATEGY_LABELS[strategy],
                "unique_count": diversity["unique_count"],
                "unique_rate": diversity["unique_rate"],
                "top1_share": diversity["top1_share"],
                "top10_share": diversity["top10_share"],
                "distinct_1": diversity["distinct_1"],
                "distinct_2": diversity["distinct_2"],
                "bleu_4": overlap["bleu_4"],
                "rouge_l": overlap["rouge_l"],
                "cider": overlap["cider"],
                "generation_seconds": source["generation_seconds"],
                "top_templates": [
                    {"text": text, "count": count}
                    for text, count in diversity["top10_outputs"]
                ],
            }
        )
    return summaries


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def render_html(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    result: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    hist_det_same = sum(row["historical_equals_deterministic"] for row in rows)
    all_same = sum(row["all_same"] for row in rows)
    nucleus_changed = sum(row["nucleus_changed"] for row in rows)
    payload = json_for_script(rows)
    summary_payload = json_for_script(summaries)
    source_payload = json_for_script(source_hashes)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoccerMaster 200 样本解码策略对比</title>
<style>
:root{{--ink:#172033;--muted:#687386;--line:#dce3ec;--panel:#fff;--bg:#f3f6fa;--blue:#315fa8;--green:#14765f;--amber:#a96808;--red:#b53b45;--purple:#6f4ba0;--soft:#f7f9fc}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}}
main{{max-width:1600px;margin:auto;padding:30px 24px 80px}}h1{{margin:6px 0 8px;font-size:clamp(27px,4vw,42px)}}h2{{margin:38px 0 14px}}h3{{margin:0 0 8px}}p{{margin:8px 0}}
.tag,.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:650}}.tag{{background:#e6f3ef;color:#0c654f}}.badge{{background:#e9eef6;color:#45536a}}.badge.good{{background:#e7f4ee;color:#12664f}}.badge.warn{{background:#fff0d5;color:#82560a}}.badge.bad{{background:#fde8e9;color:#92343b}}.badge.purple{{background:#eee8f7;color:#5d3f87}}
.lead{{max-width:1050px;color:var(--muted);font-size:17px}}.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}}.metric small{{display:block;color:var(--muted)}}.metric b{{display:block;font-size:25px;color:var(--blue);margin-top:3px}}
.note{{background:#fff8e7;border-left:5px solid #d39220;border-radius:9px;padding:14px 17px;margin:16px 0}}.ok{{background:#eaf6f1;border-left:5px solid var(--green);border-radius:9px;padding:14px 17px;margin:16px 0}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:880px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--muted);font-size:13px;position:sticky;top:0;background:white}}td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
.controls{{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:2fr 1fr 1fr auto;gap:10px;padding:12px;background:rgba(243,246,250,.96);backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:12px;margin:18px 0}}input,select,button{{font:inherit;border:1px solid #cbd4df;border-radius:8px;padding:10px;background:white;color:var(--ink)}}button{{cursor:pointer}}button:hover{{border-color:var(--blue)}}
.sample{{background:white;border:1px solid var(--line);border-radius:14px;margin:14px 0;overflow:hidden}}.sample-head{{padding:13px 16px;background:#f8fafc;border-bottom:1px solid var(--line);display:flex;gap:9px;align-items:center;flex-wrap:wrap}}.sample-head strong{{font-size:17px}}.sample-grid{{display:grid;grid-template-columns:1.05fr repeat(3,1fr)}}.text-box{{padding:16px;border-right:1px solid var(--line);min-height:190px}}.text-box:last-child{{border-right:0}}.text-box.reference{{background:#f6f9ff}}.box-title{{font-weight:750;margin-bottom:9px;color:#34465f}}.text{{font-size:15px;white-space:pre-wrap}}.detail{{display:flex;gap:6px;flex-wrap:wrap;margin-top:13px}}.path{{padding:10px 16px;border-top:1px solid var(--line);color:var(--muted);font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}
.templates{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}}.template-column{{background:white;border:1px solid var(--line);border-radius:12px;padding:15px}}.template{{padding:10px 0;border-bottom:1px solid var(--line)}}.template:last-child{{border-bottom:0}}.template-count{{float:right;color:var(--red);font-weight:750;margin-left:8px}}.muted{{color:var(--muted)}}code{{overflow-wrap:anywhere}}#empty{{display:none;text-align:center;padding:35px;color:var(--muted)}}
@media(max-width:1150px){{.cards{{grid-template-columns:repeat(3,1fr)}}.sample-grid{{grid-template-columns:1fr 1fr}}.text-box{{border-bottom:1px solid var(--line)}}.templates{{grid-template-columns:1fr}}}}
@media(max-width:720px){{main{{padding:20px 12px 60px}}.cards{{grid-template-columns:1fr 1fr}}.controls{{grid-template-columns:1fr}}.sample-grid{{grid-template-columns:1fr}}.text-box{{min-height:0;border-right:0}}}}
</style>
</head>
<body><main>
<span class="tag">已有结果离线可视化 · 未重新运行模型</span>
<h1>200 个真实视频：三种解码方式到底说了什么？</h1>
<p class="lead">同一个视觉表示分别交给历史 Beam+Sampling、确定性 Beam 和 Nucleus Sampling。这个页面把总体指标、高频模板和每个样本的参考答案放在一起，帮助判断“更丰富”是否也“更贴合画面”。</p>

<div class="cards">
  <div class="metric"><small>固定真实样本</small><b>200</b></div>
  <div class="metric"><small>历史法唯一输出</small><b>{summaries[0]['unique_count']}</b></div>
  <div class="metric"><small>Nucleus 唯一输出</small><b>{summaries[2]['unique_count']}</b></div>
  <div class="metric"><small>历史法 = 确定性 Beam</small><b>{hist_det_same}</b></div>
  <div class="metric"><small>Nucleus 改变文本</small><b>{nucleus_changed}</b></div>
</div>

<div class="ok"><b>最直接的观察：</b>Nucleus Sampling 把唯一输出率从 {summaries[0]['unique_rate']:.1%} 提高到 {summaries[2]['unique_rate']:.1%}，Top-10 模板占比从 {summaries[0]['top10_share']:.1%} 降到 {summaries[2]['top10_share']:.1%}。它明显缓解模板重复，但不自动代表事件判断更准确。</div>
<div class="note"><b>阅读边界：</b>BLEU/ROUGE/CIDEr 是 200 条固定子集结果，不是论文完整测试集指标。页面里的“Token F1”和“事件词线索”只是用于排序和人工检查的词面辅助量，不是新的正式评估指标。</div>

<h2>总体对比</h2>
<div class="panel"><table><thead><tr><th>策略</th><th>唯一输出</th><th>Top-1 模板</th><th>Top-10 模板</th><th>Distinct-1 / 2</th><th>BLEU-4</th><th>ROUGE-L</th><th>CIDEr</th><th>生成耗时</th></tr></thead><tbody id="summaryBody"></tbody></table></div>

<h2>最常重复的模板</h2>
<p class="muted">右侧数字表示该句在 200 个样本中完整重复的次数。</p>
<div class="templates" id="templates"></div>

<h2>逐样本对照</h2>
<div class="controls">
  <input id="search" type="search" placeholder="搜索参考答案、生成文本、视频名或 dataset index">
  <select id="filter">
    <option value="all">全部 200 条</option>
    <option value="hist-det-same">历史法 = 确定性 Beam</option>
    <option value="all-same">三种输出完全相同</option>
    <option value="nucleus-changed">Nucleus 改变了输出</option>
    <option value="template">至少一种输出重复 ≥ 4 次</option>
    <option value="cue-mismatch">Nucleus 与参考事件词无交集</option>
  </select>
  <select id="sort">
    <option value="ordinal">按固定样本顺序</option>
    <option value="disagreement">策略分歧最大优先</option>
    <option value="template">高频模板优先</option>
    <option value="nucleus-gain">Nucleus 词面改善优先</option>
    <option value="nucleus-loss">Nucleus 词面下降优先</option>
  </select>
  <button id="reset" type="button">重置</button>
</div>
<p id="count" class="muted"></p><div id="samples"></div><div id="empty">没有符合条件的样本。</div>

<h2>证据与限制</h2>
<div class="panel">
  <ul>
    <li>实验状态：<code>{html.escape(str(result['status']))}</code>；完成 {result['samples_completed']} 条；总墙钟 {result['elapsed_seconds']:.3f} 秒。</li>
    <li>没有训练、backward、optimizer、scheduler 或 DataLoader；本页面也没有导入 Torch 或项目模块。</li>
    <li>历史 attention-mask/PAD 行为被原样保留，因此本次是公平的解码器对照，不是已修正的生成基线。</li>
    <li>逐样本事件词只检查显式词面。近义表达、否定关系和角色关系仍需人读，不能由徽标自动判定。</li>
    <li>来源 SHA256 可在同目录 <a href="report_manifest.json">report_manifest.json</a> 中复核。</li>
  </ul>
</div>

<script>
const ROWS={payload};
const SUMMARIES={summary_payload};
const SOURCE_HASHES={source_payload};
const LABELS={json_for_script(STRATEGY_LABELS)};
const ORDER={json_for_script(list(STRATEGIES))};
const escapeHtml=(value)=>String(value).replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
const pct=(value)=>(100*value).toFixed(1)+'%';
const fixed=(value,digits=3)=>Number(value).toFixed(digits);

document.getElementById('summaryBody').innerHTML=SUMMARIES.map(s=>`<tr><td><b>${{escapeHtml(s.label)}}</b></td><td class="num">${{s.unique_count}} / 200 (${{pct(s.unique_rate)}})</td><td class="num">${{pct(s.top1_share)}}</td><td class="num">${{pct(s.top10_share)}}</td><td class="num">${{fixed(s.distinct_1)}} / ${{fixed(s.distinct_2)}}</td><td class="num">${{fixed(s.bleu_4,4)}}</td><td class="num">${{fixed(s.rouge_l,4)}}</td><td class="num">${{fixed(s.cider,4)}}</td><td class="num">${{fixed(s.generation_seconds,1)}} s</td></tr>`).join('');

document.getElementById('templates').innerHTML=SUMMARIES.map(s=>`<section class="template-column"><h3>${{escapeHtml(s.label)}}</h3>${{s.top_templates.map((item,i)=>`<div class="template"><span class="template-count">×${{item.count}}</span><small class="muted">#${{i+1}}</small><br>${{escapeHtml(item.text)}}</div>`).join('')}}</section>`).join('');

function badgesFor(detail, referenceCues){{
  const badges=[`<span class="badge">Token F1 ${{fixed(detail.token_f1)}}</span>`,`<span class="badge ${{detail.duplicate_count>=4?'bad':''}}">完整重复 ×${{detail.duplicate_count}}</span>`];
  if(referenceCues.length){{
    badges.push(detail.cue_match?`<span class="badge good">共享线索：${{escapeHtml(detail.shared_cues.join('、'))}}</span>`:`<span class="badge warn">事件词无交集</span>`);
  }} else {{ badges.push('<span class="badge">参考无预设事件词</span>'); }}
  if(detail.cues.length) badges.push(`<span class="badge purple">输出线索：${{escapeHtml(detail.cues.join('、'))}}</span>`);
  return badges.join('');
}}

function sampleHtml(row){{
  const header=[`<strong>#${{row.ordinal}} · dataset index ${{row.dataset_index}}</strong>`,`<span class="badge">seed ${{row.sample_seed}}</span>`,`<span class="badge">策略分歧 ${{fixed(row.disagreement)}}</span>`];
  if(row.historical_equals_deterministic) header.push('<span class="badge warn">历史法 = 确定性 Beam</span>');
  if(row.all_same) header.push('<span class="badge bad">三种完全相同</span>');
  const boxes=[`<div class="text-box reference"><div class="box-title">参考答案（GT）</div><div class="text">${{escapeHtml(row.reference)}}</div><div class="detail">${{row.reference_cues.length?`<span class="badge good">参考线索：${{escapeHtml(row.reference_cues.join('、'))}}</span>`:'<span class="badge">未命中预设事件词</span>'}}</div></div>`];
  ORDER.forEach(key=>{{const detail=row.strategies[key];boxes.push(`<div class="text-box"><div class="box-title">${{escapeHtml(LABELS[key])}}</div><div class="text">${{escapeHtml(detail.text)}}</div><div class="detail">${{badgesFor(detail,row.reference_cues)}}</div></div>`);}});
  return `<article class="sample"><div class="sample-head">${{header.join('')}}</div><div class="sample-grid">${{boxes.join('')}}</div><div class="path">${{escapeHtml(row.video_name)}}<details><summary>显示完整视频路径</summary>${{escapeHtml(row.video_path)}}</details></div></article>`;
}}

const search=document.getElementById('search'),filter=document.getElementById('filter'),sort=document.getElementById('sort');
function render(){{
  const query=search.value.trim().toLowerCase();
  let rows=ROWS.filter(row=>{{
    const text=[row.dataset_index,row.video_name,row.reference,...ORDER.map(key=>row.strategies[key].text)].join(' ').toLowerCase();
    if(query && !text.includes(query)) return false;
    if(filter.value==='hist-det-same' && !row.historical_equals_deterministic) return false;
    if(filter.value==='all-same' && !row.all_same) return false;
    if(filter.value==='nucleus-changed' && !row.nucleus_changed) return false;
    if(filter.value==='template' && row.max_duplicate_count<4) return false;
    if(filter.value==='cue-mismatch' && !(row.reference_cues.length && row.strategies.nucleus_sampling.cue_match===false)) return false;
    return true;
  }});
  rows=[...rows].sort((a,b)=>{{
    if(sort.value==='disagreement') return b.disagreement-a.disagreement || a.ordinal-b.ordinal;
    if(sort.value==='template') return b.max_duplicate_count-a.max_duplicate_count || a.ordinal-b.ordinal;
    if(sort.value==='nucleus-gain') return b.nucleus_overlap_delta-a.nucleus_overlap_delta || a.ordinal-b.ordinal;
    if(sort.value==='nucleus-loss') return a.nucleus_overlap_delta-b.nucleus_overlap_delta || a.ordinal-b.ordinal;
    return a.ordinal-b.ordinal;
  }});
  document.getElementById('samples').innerHTML=rows.map(sampleHtml).join('');
  document.getElementById('count').textContent=`当前显示 ${{rows.length}} / ${{ROWS.length}} 条`;
  document.getElementById('empty').style.display=rows.length?'none':'block';
}}
[search,filter,sort].forEach(element=>element.addEventListener(element===search?'input':'change',render));
document.getElementById('reset').addEventListener('click',()=>{{search.value='';filter.value='all';sort.value='ordinal';render();}});
render();
</script>
</main></body></html>
"""


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    predictions_path = input_dir / "predictions.jsonl"
    result_path = input_dir / "result.json"
    output_html = args.output_html.resolve()
    output_manifest = args.output_manifest.resolve()

    for path in (predictions_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_html, output_manifest):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing report: {path}")
        if path.parent != input_dir:
            raise ValueError(f"report output must stay in input directory: {path}")

    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "passed":
        raise ValueError(f"source experiment did not pass: {result.get('status')!r}")

    source_hashes = {
        "predictions.jsonl": sha256(predictions_path),
        "result.json": sha256(result_path),
    }
    rows = load_rows(predictions_path, result)
    summaries = strategy_summary(result)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        render_html(rows, summaries, result, source_hashes), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "model_executed": False,
        "torch_imported": False,
        "gpu_used": False,
        "training_executed": False,
        "source_files": source_hashes,
        "output_html": {
            "path": str(output_html),
            "bytes": output_html.stat().st_size,
            "sha256": sha256(output_html),
        },
        "validated_rows": len(rows),
        "strategies": list(STRATEGIES),
        "derived_counts": {
            "historical_equals_deterministic": sum(
                row["historical_equals_deterministic"] for row in rows
            ),
            "all_three_equal": sum(row["all_same"] for row in rows),
            "nucleus_changed": sum(row["nucleus_changed"] for row in rows),
            "rows_with_repeat_count_at_least_4": sum(
                row["max_duplicate_count"] >= 4 for row in rows
            ),
        },
        "limitations": [
            "The 200 fixed samples are a subset, not the full paper evaluation.",
            "Token F1 and lexical event cues are inspection aids, not official metrics.",
            "The report does not inspect video pixels or rerun the model.",
            "Historical attention-mask and PAD behavior remains unchanged in source results.",
        ],
    }
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] validated_rows={len(rows)}")
    print(f"[OK] html={output_html} bytes={output_html.stat().st_size}")
    print(
        f"[OK] manifest={output_manifest} bytes={output_manifest.stat().st_size}"
    )
    print("[SAFETY] model_executed=0 torch_imported=0 gpu_used=0 training_executed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
