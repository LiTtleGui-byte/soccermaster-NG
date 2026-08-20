#!/usr/bin/env python3
"""Render a self-contained semantic review page for the best E1 candidate.

The renderer is standard-library-only. It reads completed E1/E2/E3 JSON
evidence, never imports project or machine-learning modules, and never opens
video, model, cache, or checkpoint files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
REPORT_ROOT = REPO / "reports/commentary_parallel_20260814"
E1_DIR = REPORT_ROOT / "e1_decoder_sweep_run1"
E2_DIR = REPORT_ROOT / "e2_visual_sensitivity_run1"
E3_DIR = REPORT_ROOT / "e3_mask_pad_run1"
OUTPUT_HTML = REPORT_ROOT / "semantic_review.html"
OUTPUT_MANIFEST = REPORT_ROOT / "semantic_review_manifest.json"

EXPECTED_HASHES = {
    "e1_result": "f07f53b96b8f7e72935147a02f131805fe6d7014245f338a480f8d4aa03e1b33",
    "e1_predictions": "14c8c999fa7a7ff3a598db9b482746f4956d284772ecb16b4811bedb3f684d02",
    "e2_result": "94178ed3558b452bda185a7fa6f669f10d3346adf761ef874640ab38012b0ebe",
    "e2_predictions": "2be5bd010fa7c905f4656d88918caa562e7d2e39f7e33ea474af382b32056921",
    "e3_result": "81dcea39ed4a5e9805b82abf71220b4237365b4deaae9b7d62a24a5d0fe0fec4",
}
SOURCES = {
    "e1_result": E1_DIR / "result.json",
    "e1_predictions": E1_DIR / "predictions.jsonl",
    "e2_result": E2_DIR / "result.json",
    "e2_predictions": E2_DIR / "predictions.jsonl",
    "e3_result": E3_DIR / "result.json",
}

HISTORICAL = "historical_beam_sampling"
CANDIDATE = "nucleus_t070_p090"
TOKEN_RE = re.compile(r"\[\w+\]|[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
EVENT_CUES = {
    "进球": ("goal", "scores", "scored", "net"),
    "射门": ("shot", "shoot", "fires", "effort", "volley", "strike"),
    "扑救": ("save", "keeper", "goalkeeper"),
    "犯规/任意球": ("foul", "whistle", "free kick", "handball"),
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
    "解围/拦截": ("clear", "intercept", "blocks", "blocked"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {line_number}: {path}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected row object at line {line_number}: {path}")
            rows.append(row)
    return rows


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


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the HTML and manifest owned by this renderer.",
    )
    return parser.parse_args()


def validate_sources() -> dict[str, str]:
    actual = {}
    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if actual[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(
                f"Source SHA256 changed for {name}: expected "
                f"{EXPECTED_HASHES[name]}, got {actual[name]}"
            )
    return actual


def prepare_rows(
    e1_rows: list[dict[str, Any]], e2_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(e1_rows) != 200 or len(e2_rows) != 200:
        raise RuntimeError(
            f"Expected 200 E1/E2 rows, got {len(e1_rows)}/{len(e2_rows)}"
        )
    e2_by_index = {int(row["dataset_index"]): row for row in e2_rows}
    if len(e2_by_index) != 200:
        raise RuntimeError("E2 dataset indices are not unique")

    historical_counts = Counter(
        str(row["predictions"][HISTORICAL]["text"]).strip() for row in e1_rows
    )
    candidate_counts = Counter(
        str(row["predictions"][CANDIDATE]["text"]).strip() for row in e1_rows
    )
    output_rows = []
    seen_indices: set[int] = set()
    for expected_ordinal, row in enumerate(e1_rows, 1):
        ordinal = int(row["ordinal"])
        dataset_index = int(row["dataset_index"])
        if ordinal != expected_ordinal or dataset_index in seen_indices:
            raise RuntimeError(
                f"E1 identity mismatch: ordinal={ordinal}, index={dataset_index}"
            )
        seen_indices.add(dataset_index)
        e2_row = e2_by_index.get(dataset_index)
        if e2_row is None:
            raise RuntimeError(f"E2 row missing for dataset index {dataset_index}")

        reference = str(row["reference_commentary"]).strip()
        historical = str(row["predictions"][HISTORICAL]["text"]).strip()
        candidate = str(row["predictions"][CANDIDATE]["text"]).strip()
        if not reference or not historical or not candidate:
            raise RuntimeError(f"Empty text at dataset index {dataset_index}")
        historical_f1 = token_f1(reference, historical)
        candidate_f1 = token_f1(reference, candidate)
        reference_cues = extract_cues(reference)
        historical_cues = extract_cues(historical)
        candidate_cues = extract_cues(candidate)
        diagnostic = e2_row["first_token_diagnostic"]
        e2_correct = e2_row["predictions"]["correct_prefix"]["text"]
        e2_shifted = e2_row["predictions"]["cyclic_shift_prefix"]["text"]
        e2_zero = e2_row["predictions"]["zero_prefix"]["text"]
        output_rows.append(
            {
                "ordinal": ordinal,
                "dataset_index": dataset_index,
                "video_name": Path(str(row["video_path"])).name,
                "video_path": str(row["video_path"]),
                "reference": reference,
                "historical": {
                    "text": historical,
                    "token_f1": historical_f1,
                    "duplicate_count": historical_counts[historical],
                    "cues": historical_cues,
                    "shared_cues": sorted(set(reference_cues) & set(historical_cues)),
                },
                "candidate": {
                    "text": candidate,
                    "token_f1": candidate_f1,
                    "duplicate_count": candidate_counts[candidate],
                    "cues": candidate_cues,
                    "shared_cues": sorted(set(reference_cues) & set(candidate_cues)),
                },
                "reference_cues": reference_cues,
                "candidate_changed": candidate != historical,
                "token_f1_delta": candidate_f1 - historical_f1,
                "visual_diagnostic": {
                    "js_correct_vs_shifted": float(
                        diagnostic["js_correct_vs_shifted"]
                    ),
                    "js_correct_vs_zero": float(diagnostic["js_correct_vs_zero"]),
                    "correct_vs_shifted_text_changed": e2_correct != e2_shifted,
                    "correct_vs_zero_text_changed": e2_correct != e2_zero,
                },
            }
        )

    counts = {
        "candidate_changed": sum(row["candidate_changed"] for row in output_rows),
        "candidate_token_f1_higher": sum(
            row["token_f1_delta"] > 0 for row in output_rows
        ),
        "candidate_token_f1_equal": sum(
            row["token_f1_delta"] == 0 for row in output_rows
        ),
        "candidate_token_f1_lower": sum(
            row["token_f1_delta"] < 0 for row in output_rows
        ),
        "historical_repeated_at_least_4": sum(
            row["historical"]["duplicate_count"] >= 4 for row in output_rows
        ),
        "candidate_repeated_at_least_4": sum(
            row["candidate"]["duplicate_count"] >= 4 for row in output_rows
        ),
    }
    return output_rows, counts


def render_html(
    rows: list[dict[str, Any]],
    counts: dict[str, int],
    e1_result: dict[str, Any],
    e2_result: dict[str, Any],
    e3_result: dict[str, Any],
    source_hashes: dict[str, str],
) -> str:
    historical = e1_result["condition_summaries"][HISTORICAL]
    candidate = e1_result["condition_summaries"][CANDIDATE]
    payload = json_for_script(rows)
    hashes_payload = json_for_script(source_hashes)
    storage_key = "soccermaster-semantic-review-" + source_hashes["e1_predictions"][:16]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoccerMaster 解说语义人工审查</title>
<style>
:root{{--ink:#182235;--muted:#697588;--line:#d9e1eb;--bg:#f3f6fa;--panel:#fff;--blue:#285fa8;--green:#16745c;--amber:#a96808;--red:#ad3945;--purple:#684596}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}}
main{{max-width:1540px;margin:auto;padding:28px 22px 80px}}h1{{font-size:clamp(28px,4vw,43px);margin:8px 0}}h2{{margin:34px 0 12px}}p{{margin:8px 0}}.lead{{max-width:1100px;color:var(--muted);font-size:17px}}
.tag,.badge{{display:inline-block;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:650;background:#e9eef6;color:#425169}}.tag{{background:#e5f3ee;color:#11664f}}.badge.good{{background:#e4f4ec;color:#11634c}}.badge.warn{{background:#fff0d4;color:#805307}}.badge.bad{{background:#fde7e9;color:#91343c}}.badge.purple{{background:#eee7f7;color:#5b3a84}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;margin:20px 0}}.metric{{background:white;border:1px solid var(--line);border-radius:12px;padding:14px}}.metric small{{display:block;color:var(--muted)}}.metric b{{display:block;font-size:24px;color:var(--blue)}}
.ok,.note{{border-radius:9px;padding:13px 16px;margin:14px 0;border-left:5px solid}}.ok{{background:#e9f6f0;border-color:var(--green)}}.note{{background:#fff8e5;border-color:#d39220}}
.panel{{background:white;border:1px solid var(--line);border-radius:12px;padding:17px;margin:13px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:850px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}
.controls{{position:sticky;top:0;z-index:10;display:grid;grid-template-columns:2fr 1fr 1fr auto auto;gap:9px;padding:11px;background:rgba(243,246,250,.97);border:1px solid var(--line);border-radius:12px;backdrop-filter:blur(8px)}}input,select,button{{font:inherit;padding:9px;border:1px solid #c8d2df;border-radius:8px;background:white;color:var(--ink)}}button{{cursor:pointer}}button:hover{{border-color:var(--blue)}}
.sample{{background:white;border:1px solid var(--line);border-radius:13px;margin:13px 0;overflow:hidden}}.sample-head{{padding:12px 15px;background:#f8fafc;border-bottom:1px solid var(--line);display:flex;gap:7px;align-items:center;flex-wrap:wrap}}.sample-grid{{display:grid;grid-template-columns:1fr 1fr 1fr}}.box{{padding:15px;border-right:1px solid var(--line);min-height:185px}}.box:last-child{{border-right:0}}.box.ref{{background:#f5f8ff}}.box.candidate{{background:#f4fbf7}}.title{{font-weight:750;margin-bottom:8px}}.text{{white-space:pre-wrap}}.details{{display:flex;gap:6px;flex-wrap:wrap;margin-top:11px}}
.review{{border-top:1px solid var(--line);padding:12px 15px;background:#fffdf8;display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:12px}}.review label{{margin-right:10px;white-space:nowrap}}.review textarea{{width:100%;min-height:65px;border:1px solid #c8d2df;border-radius:8px;padding:8px;resize:vertical}}.diagnostic{{border-top:1px solid var(--line);padding:9px 15px;color:var(--muted);font-size:13px}}.path{{overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}
.empty{{display:none;text-align:center;color:var(--muted);padding:35px}}.muted{{color:var(--muted)}}
@media(max-width:1050px){{.cards{{grid-template-columns:repeat(3,1fr)}}.sample-grid{{grid-template-columns:1fr}}.box{{border-right:0;border-bottom:1px solid var(--line)}}.review{{grid-template-columns:1fr}}}}
@media(max-width:720px){{main{{padding:18px 10px 60px}}.cards{{grid-template-columns:1fr 1fr}}.controls{{grid-template-columns:1fr}}}}
</style></head><body><main>
<span class="tag">CPU-only 离线报告 · 没有重新运行模型</span>
<h1>历史解码 vs 当前最佳候选：逐条人工判断</h1>
<p class="lead">固定 200 条真实 MatchTime 样本。左侧是参考答案，中间是历史 Beam+Sampling，右侧是 E1 当前最佳候选 Nucleus T=0.70/P=0.90。请重点判断事件、动作、结果和角色关系，而不是只看句子是否更丰富。</p>

<div class="cards">
 <div class="metric"><small>固定样本</small><b>200</b></div>
 <div class="metric"><small>候选改变文本</small><b>{counts['candidate_changed']}</b></div>
 <div class="metric"><small>候选 Token F1 更高</small><b>{counts['candidate_token_f1_higher']}</b></div>
 <div class="metric"><small>历史唯一率</small><b>{historical['diversity']['unique_rate']:.1%}</b></div>
 <div class="metric"><small>候选唯一率</small><b>{candidate['diversity']['unique_rate']:.1%}</b></div>
</div>
<div class="ok"><b>机器指标：</b>候选把 Top-10 模板占比从 {historical['diversity']['top10_share']:.1%} 降到 {candidate['diversity']['top10_share']:.1%}，CIDEr 从 {historical['overlap']['cider']:.5f} 提到 {candidate['overlap']['cider']:.5f}，ROUGE-L 从 {historical['overlap']['rouge_l']:.5f} 提到 {candidate['overlap']['rouge_l']:.5f}。</div>
<div class="note"><b>不要直接据此判定候选更好：</b>Token F1 和事件词徽标只用于排序。真正的结论应来自下方人工判断，尤其是动作方向、否定关系、角色归属和结果是否正确。</div>

<h2>整体证据</h2><div class="panel"><table><thead><tr><th>检查</th><th>结果</th><th>含义</th></tr></thead><tbody>
<tr><td>E1 历史基线复现</td><td>200/200 与此前完全一致</td><td>候选比较没有换基线</td></tr>
<tr><td>E2 正确 vs 错配视觉</td><td>{e2_result['visual_sensitivity']['full_text_change_correct_vs_shifted']}/200 改变；JS={e2_result['visual_sensitivity']['mean_js_correct_vs_shifted']:.5f}</td><td>decoder 会使用视觉前缀</td></tr>
<tr><td>E2 正确 vs 全零视觉</td><td>{e2_result['visual_sensitivity']['full_text_change_correct_vs_zero']}/200 改变；JS={e2_result['visual_sensitivity']['mean_js_correct_vs_zero']:.5f}</td><td>全零输入不是正常生成条件</td></tr>
<tr><td>E3 Mask/PAD 三臂</td><td>{e3_result['pairwise_differences']['tokens_changed_aligned_vs_historical']}/200 token 改变</td><td>当前单样本警告不改变输出</td></tr>
</tbody></table></div>

<h2>逐样本人工审查</h2>
<div class="controls">
 <input id="search" type="search" placeholder="搜索文本、视频名或 dataset index">
 <select id="filter"><option value="all">全部样本</option><option value="changed">候选改变文本</option><option value="f1gain">候选 Token F1 更高</option><option value="f1loss">候选 Token F1 更低</option><option value="hist-template">历史高频模板 ≥4</option><option value="candidate-template">候选高频模板 ≥4</option><option value="visual-change">错配视觉后文本改变</option><option value="unreviewed">尚未人工判断</option></select>
 <select id="sort"><option value="ordinal">固定样本顺序</option><option value="gain">Token F1 改善优先</option><option value="loss">Token F1 下降优先</option><option value="hist-template">历史模板重复优先</option><option value="candidate-template">候选模板重复优先</option><option value="visual-js">视觉 JS 最大优先</option></select>
 <button id="export" type="button">导出标注 JSON</button><button id="clear" type="button">清空本地标注</button>
</div>
<p id="count" class="muted"></p><div id="samples"></div><div id="empty" class="empty">没有符合条件的样本。</div>

<h2>使用边界</h2><div class="panel"><ul>
<li>人工标注只保存在当前浏览器 localStorage；点击“导出标注 JSON”才能得到可迁移文件。</li>
<li>页面不加载视频画面，因此本轮是“输出相对参考答案”的文本审查；需要看画面时可使用每条记录里的视频路径。</li>
<li>固定 200 条不是完整 3,256 条测试集，不能直接作为论文总体结论。</li>
<li>来源哈希已经嵌入页面和同目录 manifest；本页面没有导入 Torch、读取权重或使用 GPU。</li>
</ul><details><summary>显示来源 SHA256</summary><pre id="hashes"></pre></details></div>

<script>
const ROWS={payload};const SOURCE_HASHES={hashes_payload};const STORAGE_KEY={json_for_script(storage_key)};
const escapeHtml=v=>String(v).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fixed=(v,n=3)=>Number(v).toFixed(n);let annotations=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{{}}');
const getAnnotation=i=>annotations[String(i)]||{{verdict:'',dimensions:{{event:false,action:false,result:false,roles:false}},note:''}};
function saveAnnotation(index,patch){{const old=getAnnotation(index);annotations[String(index)]={{...old,...patch,dimensions:{{...old.dimensions,...(patch.dimensions||{{}})}}}};localStorage.setItem(STORAGE_KEY,JSON.stringify(annotations));renderCount();}}
function cueBadges(detail,referenceCues){{const shared=detail.shared_cues||[];let out=`<span class="badge">Token F1 ${{fixed(detail.token_f1)}}</span><span class="badge ${{detail.duplicate_count>=4?'bad':''}}">完整重复 ×${{detail.duplicate_count}}</span>`;if(referenceCues.length)out+=shared.length?`<span class="badge good">共享线索：${{escapeHtml(shared.join('、'))}}</span>`:'<span class="badge warn">事件词无交集</span>';if(detail.cues.length)out+=`<span class="badge purple">输出线索：${{escapeHtml(detail.cues.join('、'))}}</span>`;return out;}}
function reviewHtml(row){{const a=getAnnotation(row.dataset_index);const verdicts=[['candidate_better','候选更好'],['tie','大致相同'],['historical_better','历史更好'],['both_wrong','两者都错']];const dims=[['event','事件'],['action','动作'],['result','结果'],['roles','角色关系']];return `<div><b>总体判断</b><div>${{verdicts.map(([v,l])=>`<label><input type="radio" name="verdict-${{row.dataset_index}}" data-index="${{row.dataset_index}}" data-verdict="${{v}}" ${{a.verdict===v?'checked':''}}> ${{l}}</label>`).join('')}}</div></div><div><b>候选在哪些维度更准确？</b><div>${{dims.map(([v,l])=>`<label><input type="checkbox" data-index="${{row.dataset_index}}" data-dimension="${{v}}" ${{a.dimensions[v]?'checked':''}}> ${{l}}</label>`).join('')}}</div></div><div><b>备注</b><textarea data-note="${{row.dataset_index}}" placeholder="记录具体错误或改进">${{escapeHtml(a.note)}}</textarea></div>`;}}
function sampleHtml(row){{const delta=row.token_f1_delta;return `<article class="sample"><div class="sample-head"><strong>#${{row.ordinal}} · dataset index ${{row.dataset_index}}</strong><span class="badge ${{delta>0?'good':delta<0?'bad':''}}">候选 F1 Δ ${{delta>=0?'+':''}}${{fixed(delta)}}</span><span class="badge">视觉错配 JS ${{fixed(row.visual_diagnostic.js_correct_vs_shifted)}}</span>${{row.candidate_changed?'':'<span class="badge warn">候选与历史相同</span>'}}</div><div class="sample-grid"><div class="box ref"><div class="title">参考答案（GT）</div><div class="text">${{escapeHtml(row.reference)}}</div><div class="details">${{row.reference_cues.length?`<span class="badge good">参考线索：${{escapeHtml(row.reference_cues.join('、'))}}</span>`:'<span class="badge">未命中预设事件词</span>'}}</div></div><div class="box"><div class="title">历史 Beam + Sampling</div><div class="text">${{escapeHtml(row.historical.text)}}</div><div class="details">${{cueBadges(row.historical,row.reference_cues)}}</div></div><div class="box candidate"><div class="title">候选 Nucleus T=0.70 / P=0.90</div><div class="text">${{escapeHtml(row.candidate.text)}}</div><div class="details">${{cueBadges(row.candidate,row.reference_cues)}}</div></div></div><div class="diagnostic">错配视觉后文本：${{row.visual_diagnostic.correct_vs_shifted_text_changed?'改变':'未改变'}}；全零视觉后文本：${{row.visual_diagnostic.correct_vs_zero_text_changed?'改变':'未改变'}}；视频：<span class="path">${{escapeHtml(row.video_path)}}</span></div><div class="review">${{reviewHtml(row)}}</div></article>`;}}
const search=document.getElementById('search'),filter=document.getElementById('filter'),sort=document.getElementById('sort');
function selectedRows(){{const q=search.value.trim().toLowerCase();let rows=ROWS.filter(r=>{{const hay=[r.dataset_index,r.video_name,r.reference,r.historical.text,r.candidate.text].join(' ').toLowerCase();if(q&&!hay.includes(q))return false;if(filter.value==='changed'&&!r.candidate_changed)return false;if(filter.value==='f1gain'&&r.token_f1_delta<=0)return false;if(filter.value==='f1loss'&&r.token_f1_delta>=0)return false;if(filter.value==='hist-template'&&r.historical.duplicate_count<4)return false;if(filter.value==='candidate-template'&&r.candidate.duplicate_count<4)return false;if(filter.value==='visual-change'&&!r.visual_diagnostic.correct_vs_shifted_text_changed)return false;if(filter.value==='unreviewed'&&getAnnotation(r.dataset_index).verdict)return false;return true;}});return [...rows].sort((a,b)=>{{if(sort.value==='gain')return b.token_f1_delta-a.token_f1_delta||a.ordinal-b.ordinal;if(sort.value==='loss')return a.token_f1_delta-b.token_f1_delta||a.ordinal-b.ordinal;if(sort.value==='hist-template')return b.historical.duplicate_count-a.historical.duplicate_count||a.ordinal-b.ordinal;if(sort.value==='candidate-template')return b.candidate.duplicate_count-a.candidate.duplicate_count||a.ordinal-b.ordinal;if(sort.value==='visual-js')return b.visual_diagnostic.js_correct_vs_shifted-a.visual_diagnostic.js_correct_vs_shifted||a.ordinal-b.ordinal;return a.ordinal-b.ordinal;}});}}
function renderCount(){{const reviewed=Object.values(annotations).filter(a=>a.verdict).length;document.getElementById('count').textContent=`当前显示 ${{selectedRows().length}} / 200 条；已完成总体判断 ${{reviewed}} / 200 条。`;}}
function render(){{const rows=selectedRows();document.getElementById('samples').innerHTML=rows.map(sampleHtml).join('');document.getElementById('empty').style.display=rows.length?'none':'block';renderCount();}}
document.addEventListener('change',e=>{{if(e.target.dataset.verdict)saveAnnotation(e.target.dataset.index,{{verdict:e.target.dataset.verdict}});if(e.target.dataset.dimension)saveAnnotation(e.target.dataset.index,{{dimensions:{{[e.target.dataset.dimension]:e.target.checked}}}});}});document.addEventListener('input',e=>{{if(e.target.dataset.note)saveAnnotation(e.target.dataset.note,{{note:e.target.value}});}});[search,filter,sort].forEach(el=>el.addEventListener(el===search?'input':'change',render));
document.getElementById('export').addEventListener('click',()=>{{const payload={{schema_version:1,source_e1_predictions_sha256:SOURCE_HASHES.e1_predictions,exported_at:new Date().toISOString(),annotations}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='soccermaster_semantic_review_annotations.json';a.click();URL.revokeObjectURL(a.href);}});
document.getElementById('clear').addEventListener('click',()=>{{if(confirm('确定清空当前浏览器中的全部人工标注？')){{annotations={{}};localStorage.removeItem(STORAGE_KEY);render();}}}});document.getElementById('hashes').textContent=JSON.stringify(SOURCE_HASHES,null,2);render();
</script></main></body></html>"""


def main() -> int:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be an empty string")
    for path in (OUTPUT_HTML, OUTPUT_MANIFEST):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite report output: {path}")
    source_hashes = validate_sources()
    e1_result = load_json(SOURCES["e1_result"])
    e2_result = load_json(SOURCES["e2_result"])
    e3_result = load_json(SOURCES["e3_result"])
    for name, result in (
        ("E1", e1_result),
        ("E2", e2_result),
        ("E3", e3_result),
    ):
        if result.get("status") != "passed" or result.get("samples_completed") != 200:
            raise RuntimeError(f"{name} source result is not a 200-sample pass")
    rows, counts = prepare_rows(
        load_jsonl(SOURCES["e1_predictions"]),
        load_jsonl(SOURCES["e2_predictions"]),
    )
    html_text = render_html(
        rows,
        counts,
        e1_result,
        e2_result,
        e3_result,
        source_hashes,
    )
    output_mode = "w" if args.overwrite else "x"
    with OUTPUT_HTML.open(output_mode, encoding="utf-8") as handle:
        handle.write(html_text)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "source_files": {
            name: {"path": str(SOURCES[name]), "sha256": digest}
            for name, digest in source_hashes.items()
        },
        "output_html": {
            "path": str(OUTPUT_HTML),
            "bytes": OUTPUT_HTML.stat().st_size,
            "sha256": sha256(OUTPUT_HTML),
        },
        "validated_rows": len(rows),
        "comparison": {
            "historical": HISTORICAL,
            "candidate": CANDIDATE,
        },
        "derived_counts": counts,
        "manual_review_storage": "Browser localStorage; export button downloads JSON.",
        "model_executed": False,
        "torch_imported": False,
        "project_modules_imported": False,
        "gpu_used": False,
        "training_executed": False,
        "limitations": [
            "The 200 fixed samples are not the full 3,256-sample test set.",
            "Token F1 and lexical event cues are review aids, not official metrics.",
            "The page compares text against references and does not display video pixels.",
            "Human annotations remain local to the browser until explicitly exported.",
        ],
    }
    with OUTPUT_MANIFEST.open(output_mode, encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[OK] validated_rows={len(rows)}")
    print(f"[OK] derived_counts={json.dumps(counts, sort_keys=True)}")
    print(f"[OK] html={OUTPUT_HTML} bytes={OUTPUT_HTML.stat().st_size}")
    print(
        f"[OK] manifest={OUTPUT_MANIFEST} "
        f"bytes={OUTPUT_MANIFEST.stat().st_size}"
    )
    print("[SAFETY] model=0 torch=0 project_imports=0 gpu=0 training=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
