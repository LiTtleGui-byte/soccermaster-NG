#!/usr/bin/env python3
"""Generate five task-head HTML explainers from existing G4/G5 evidence.

Standard-library only: this entry does not import the project, Torch, or any
model package.  It never runs inference or training.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import tempfile
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
G4 = REPO / "reports/g4/g4_real_video_result_20260812.json"
G5 = REPO / "reports/g5/g5_fixed_small_eval_retry1_result_20260812.json"
OVERLAY_0 = REPO / "reports/g5/g5_detection_clip_start_000_retry1_overlay_20260812.jpg"
OVERLAY_1 = REPO / "reports/g5/g5_detection_clip_start_360_retry1_overlay_20260812.jpg"
OUTPUT_PARENT = REPO / "reports"
OUTPUT_DIR = OUTPUT_PARENT / "task_head_explanations"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


STYLE = """
:root{--ink:#162033;--muted:#667286;--line:#dce3eb;--blue:#3566b8;--green:#147d64;--purple:#7453a6;--amber:#b57916;--soft:#f7f9fc}
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1450px;margin:auto;padding:32px 25px 75px}a{color:#245bac}h1{margin:6px 0 8px}h2{margin-top:38px}h3{margin:0 0 10px}
.tag,.badge{display:inline-block;border-radius:999px;padding:5px 10px;font-size:12px}.tag{background:#e7f4ef;color:#10644f}.badge.code{background:#eee9f7;color:#5f4289}.badge.run{background:#e7f4ef;color:#10644f}.badge.limit{background:#fff1d5;color:#895b0b}
.nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}.nav a{background:white;border:1px solid var(--line);padding:8px 11px;border-radius:8px;text-decoration:none}
.module{background:white;border:1px solid var(--line);border-left:6px solid var(--blue);border-radius:12px;padding:20px;margin:16px 0}.module.green{border-left-color:var(--green)}.module.purple{border-left-color:var(--purple)}
.io{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:12px}.box{background:var(--soft);padding:15px;border-radius:9px;line-height:1.55}.box.action{background:#edf3ff}.label{font-size:12px;font-weight:700;letter-spacing:.06em;color:var(--muted)}
.shape{display:block;margin:6px 0;color:#244f91;font-weight:750;font-size:18px}.axis{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.axis span{background:#e7edf4;border-radius:6px;padding:4px 7px;font-size:12px}
.card{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:white;border:1px solid var(--line);border-radius:10px;padding:14px}.metric b{display:block;font-size:22px;color:#245bac;margin-top:5px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted)}code{overflow-wrap:anywhere}.preview{max-width:430px}
.note{background:#fff8e7;border-left:5px solid #d49b27;border-radius:8px;padding:14px 17px}.ok{background:#eaf6f1;border-left:5px solid var(--green);border-radius:8px;padding:14px 17px}
img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:9px}.bar{height:10px;background:#e6ebf2;border-radius:10px;overflow:hidden}.bar i{height:100%;display:block;background:var(--blue)}canvas{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}
.head-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.head-link{background:white;border:1px solid var(--line);border-top:5px solid var(--blue);border-radius:11px;padding:16px;text-decoration:none;color:var(--ink)}.head-link b{display:block;margin-bottom:7px}.muted,small{color:var(--muted)}
@media(max-width:1000px){.io,.grid{grid-template-columns:1fr}.cards,.head-grid{grid-template-columns:repeat(2,1fr)}}
"""


NAV = """<nav class="nav"><a href="index.html">总览</a><a href="detection.html">人物检测</a><a href="lines.html">球场线</a><a href="keypoints.html">球场关键点</a><a href="video_caption.html">视频-文本对齐</a><a href="caption_classification.html">事件分类</a></nav>"""


def document(title: str, body: str, script: str = "") -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main><span class="tag">SoccerMaster epoch_19 · existing verified evidence</span><h1>{html.escape(title)}</h1>{NAV}{body}<p class="muted">机器来源与页面哈希见 <a href="manifest.json">manifest.json</a>。本次只生成HTML，没有重新运行模型。</p>{script}</main></body></html>"""


def io(title: str, incoming: str, action: str, outgoing: str, note: str, color: str = "") -> str:
    return f"""<section class="module {color}"><h3>{title}</h3><div class="io"><div class="box"><div class="label">输入</div>{incoming}</div><div class="box action"><div class="label">模块做什么</div>{action}</div><div class="box"><div class="label">输出</div>{outgoing}</div></div><p><span class="badge run">运行证据</span> {note}</p></section>"""


def fingerprint_table(values: dict[str, Any], keys: list[str]) -> str:
    rows = []
    for key in keys:
        value = values[key]
        rows.append(
            f"<tr><td><code>{key}</code></td><td><code>{value['shape']}</code></td>"
            f"<td>{value['dtype']}</td><td>{value['minimum']:.6g}</td>"
            f"<td>{value['maximum']:.6g}</td><td>{value['mean']:.6g}</td>"
            f"<td>{value['sum']:.6g}</td></tr>"
        )
    return "<div class='card'><table><thead><tr><th>真实输出</th><th>shape</th><th>dtype</th><th>最小</th><th>最大</th><th>均值</th><th>总和</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def metrics(values: list[tuple[str, float, str]]) -> str:
    cards = []
    for label, value, suffix in values:
        cards.append(f"<div class='metric'><small>{html.escape(label)}</small><b>{value:.4f}{suffix}</b></div>")
    return "<div class='cards'>" + "".join(cards) + "</div>"


def index_page(g5: dict[str, Any]) -> str:
    m = g5["passes"][0]["metrics"]
    body = f"""<p>五个任务头共享同一个视频视觉骨干，但读取的骨干特征不同、回答的问题也不同。点击任意卡片查看逐模块输入、作用、输出和真实数据。</p>
<div class="head-grid"><a class="head-link" href="detection.html"><b>SoccerNetGSR Detection</b><small>每帧找人、框位置、角色和球衣号码</small></a><a class="head-link" href="lines.html"><b>Lines Detection</b><small>输出24类球场线热图</small></a><a class="head-link" href="keypoints.html"><b>Keypoints Detection</b><small>输出58类球场关键点热图</small></a><a class="head-link" href="video_caption.html"><b>Video Caption</b><small>视频和文字是否语义匹配</small></a><a class="head-link" href="caption_classification.html"><b>Caption Classification</b><small>视频属于哪一种比赛事件</small></a></div>
<h2>它们怎样共享骨干</h2><div class="module"><div class="io"><div class="box"><div class="label">共同输入</div><span class="shape">[B,30,3,512,512]</span>每个样本30帧比赛视频。</div><div class="box action"><div class="label">共享骨干</div>SigLIP2前16层产生早期patch特征，后8层加入时间交互；同时产生每帧全局向量。<br><b>几何头</b>取早期局部特征；<b>语义头</b>取后期全局特征。</div><div class="box"><div class="label">分流</div><span class="shape">局部 [B,30,1024,1024]</span><span class="shape">全局 [B,30,1024]</span></div></div></div>
<h2>已有固定小规模证据</h2>{metrics([('人物检测 mAP@0.5',m['SoccerNetGSR_Detection']['mAP@0.5'],''),('球场线 F1',m['LinesDetection']['lines_f1'],''),('关键点 F1',m['KeypointsDetection']['keypoints_f1'],''),('视频文本 Top-1',m['VideoCaption']['retrieval_top_1_accuracy'],''),('事件分类准确率',m['CaptionClassification']['classification_accuracy'],'')])}
<div class="note"><b>结论边界：</b>这些是G5固定小规模评估证据，不是论文全测试集指标。页面中的模块接口来自当前代码，真实数值来自已经通过的G4/G5结果文件。</div>"""
    return document("SoccerMaster 五个任务头：输入与输出总览", body)


def detection_page(g5: dict[str, Any]) -> str:
    clip = g5["passes"][0]["detection_clips"][0]
    fp = clip["output_fingerprint"]
    m = g5["passes"][0]["metrics"]["SoccerNetGSR_Detection"]
    body = io(
        "1. 早期局部特征 → Deformable DETR",
        "<span class='shape'>[1,30,1024,1024]</span><div class='axis'><span>1段视频</span><span>30帧</span><span>1024个patch</span><span>每patch 1024维</span></div>",
        "每帧把1024个patch恢复成 <code>1024×32×32</code> 特征图，投影到256维；300个object query通过Deformable DETR读取特征，分别尝试代表一个人物。",
        "<span class='shape'>300个候选/帧</span><div class='axis'><span>人物置信度</span><span>cx,cy,w,h</span><span>6种角色</span><span>球衣号码</span></div>",
        f"固定序列 {clip['sequence']}、帧0–29实际输出均为有限float32。",
    ) + io(
        "2. 每个 query 的多路预测器",
        "<span class='shape'>[1,30,300,256]</span>DETR最后一层的query表示（shape由当前代码路径确定）。",
        "不同线性层读取同一个query：人物分类、归一化框、6类角色、101类整体号码、十位数字和个位/空位。",
        "<span class='shape'>7组输出张量</span><code>pred_logits / pred_boxes / pred_roles / pred_jn_holistic / pred_digit_head / pred_digit_tail</code>",
        "下表是G5保存的真实输出指纹；不是根据指标反推。",
        "green",
    )
    body += "<h2>真实输出张量</h2>" + fingerprint_table(fp, ["pred_logits","pred_boxes","pred_roles","pred_jn_holistic","pred_digit_head","pred_digit_tail"])
    body += f"<h2>一帧实际可视结果</h2><div class='grid'><div><a href='../g5/g5_detection_clip_start_000_retry1_overlay_20260812.jpg'><img src='../g5/g5_detection_clip_start_000_retry1_overlay_20260812.jpg' alt='detection overlay'></a></div><div class='card'><p>绿色框是Ground Truth，红色框是模型预测。原图来自固定序列 <code>{clip['sequence']}</code>。</p>{metrics([('mAP@0.5',m['mAP@0.5'],''),('Precision',m['precision'],''),('Recall',m['recall'],''),('角色准确率',m['role_accuracy'],'')])}</div></div>"
    body += "<div class='note'><b>未保存：</b>G5只保存完整输出的shape与统计指纹，没有把约数百万个query数值逐元素落盘；本页不会伪造所谓“前12个值”。</div>"
    return document("任务头 1：SoccerNetGSR 人物与球衣检测", body)


def lines_page(g5: dict[str, Any]) -> str:
    clip = g5["passes"][0]["detection_clips"][0]
    fp = clip["output_fingerprint"]
    m = g5["passes"][0]["metrics"]["LinesDetection"]
    body = io(
        "1. Patch特征恢复为空间网格",
        "<span class='shape'>[1,30,1024,1024]</span>早期局部patch特征。",
        "合并batch和时间维，再把1024个patch按32×32排列：<code>[30,1024,32,32]</code>。这里保留了球场几何位置。",
        "<span class='shape'>[30,1024,32,32]</span>每一帧一张低分辨率特征图。",
        "输入shape由512像素、16像素patch和运行配置共同确定。",
    ) + io(
        "2. PixelShuffle逐级上采样",
        "<span class='shape'>[30,1024,32,32]</span>",
        "三阶段卷积和PixelShuffle：<code>1024×32² → 192×64² → 96×128² → 24×256²</code>，最后Sigmoid把每个像素变成0～1响应。",
        "<span class='shape'>[1,30,24,256,256]</span><div class='axis'><span>30帧</span><span>24种球场线</span><span>256×256热图</span></div>",
        f"真实均值 {fp['pred_lines_heatmap']['mean']:.7f}，范围 {fp['pred_lines_heatmap']['minimum']:.3g}～{fp['pred_lines_heatmap']['maximum']:.6f}。",
        "green",
    )
    body += "<h2>真实输出数据指纹</h2>" + fingerprint_table(fp,["pred_lines_heatmap"])
    body += metrics([('F1',m['lines_f1'],''),('Precision',m['lines_precision'],''),('Recall',m['lines_recall'],''),('有效帧',m['lines_valid_samples'],'')])
    body += "<div class='note'><b>为什么均值很小：</b>绝大多数像素不是球场线，只有细线附近应当有高响应。完整热图约180 MiB（约189 MB），历史G5没有保存原始数组，因此本页展示真实统计与指标，但不假装能够逐像素浏览。</div>"
    return document("任务头 2：LinesDetection 球场线热图", body)


def keypoints_page(g5: dict[str, Any]) -> str:
    clip = g5["passes"][0]["detection_clips"][0]
    fp = clip["output_fingerprint"]
    out = fp["pred_keypoints_heatmap"]
    m = g5["passes"][0]["metrics"]["KeypointsDetection"]
    body = io(
        "1. 早期局部特征",
        "<span class='shape'>[1,30,1024,1024]</span>与Lines头相同的早期patch特征。",
        "把patch重新排列为每帧32×32空间网格；使用早期层是因为几何定位比高层语义更重要。",
        "<span class='shape'>[30,1024,32,32]</span>",
        "该头和Lines头共享输入，但参数完全独立。",
    ) + io(
        "2. 上采样并在58类关键点间竞争",
        "<span class='shape'>[30,1024,32,32]</span>",
        "卷积+PixelShuffle逐级放大到256×256，最后得到58个通道；Softmax沿58类通道执行，所以每个像素的58类概率和约为1。",
        "<span class='shape'>[1,30,58,256,256]</span><div class='axis'><span>58种球场关键点</span><span>每类一张热图</span></div>",
        f"真实均值 {out['mean']:.9f}，而 1/58 = {1/58:.9f}，与通道Softmax契约吻合。",
        "green",
    )
    body += "<h2>真实输出数据指纹</h2>" + fingerprint_table(fp,["pred_keypoints_heatmap"])
    body += metrics([('F1',m['keypoints_f1'],''),('Precision',m['keypoints_precision'],''),('Recall',m['keypoints_recall'],''),('有效帧',m['keypoints_valid_samples'],'')])
    body += "<div class='note'><b>数据量：</b>完整float32关键点热图约435 MiB。G5只保存了shape、最小/最大/均值/总和和最终指标，没有保存全部像素，因此本页明确保留这一边界。</div>"
    return document("任务头 3：KeypointsDetection 球场关键点热图", body)


def video_caption_page(g5: dict[str, Any]) -> str:
    p = g5["passes"][0]
    matrix = p["caption_similarity_matrix"]
    m = p["metrics"]["VideoCaption"]
    payload = json.dumps(matrix)
    body = io(
        "1. 视频与文字分别变成同维向量",
        "<span class='shape'>视频 [23,30,1024]</span><span class='shape'>文字 [23,1024]</span>本次固定评估共有23条视频-文字配对。",
        "视频对30帧全局特征求平均；文字由冻结的SigLIP2 text encoder编码；二者都做L2归一化。",
        "<span class='shape'>vision [23,1024]</span><span class='shape'>text [23,1024]</span>每个向量长度约为1。",
        "单样本G4已实际记录两者shape均为[1,1024]且全部有限。",
    ) + io(
        "2. 全体两两相似度",
        "<span class='shape'>[23,1024] @ [1024,23]</span>",
        "做点积得到23×23余弦相似度矩阵。第i行表示第i个视频与23条文字分别有多像；训练时再用可学习scale和bias变换后计算SigLIP loss。",
        "<span class='shape'>[23,23]</span>对角线是原始配对，非对角线是其他候选文字。",
        f"G5保存了完整529个真实相似度数值；Top-1={m['retrieval_top_1_accuracy']:.4f}。",
        "purple",
    )
    body += "<h2>真实 23×23 相似度矩阵</h2><div class='card'><canvas id='sim' width='760' height='760'></canvas><p class='muted'>越亮表示越相似；对角线不一定总是最亮，这正是检索错误的来源。</p></div>"
    body += metrics([('Top-1',m['retrieval_top_1_accuracy'],''),('Top-3',m['retrieval_top_3_accuracy'],''),('Top-5',m['retrieval_top_5_accuracy'],''),('样本数',m['total_samples'],'')])
    script = f"""<script>const M={payload};const c=document.getElementById('sim'),x=c.getContext('2d'),r=M.length,w=c.width/r,h=c.height/r,flat=M.flat(),lo=Math.min(...flat),hi=Math.max(...flat);M.forEach((row,i)=>row.forEach((v,j)=>{{let t=(v-lo)/(hi-lo),red=Math.round(245-170*t),green=Math.round(248-95*t),blue=Math.round(252-35*t);x.fillStyle=`rgb(${{red}},${{green}},${{blue}})`;x.fillRect(j*w,i*h,w+1,h+1)}}));</script>"""
    return document("任务头 4：VideoCaption 视频-文本对齐", body, script)


def classification_page(g4: dict[str, Any], g5: dict[str, Any]) -> str:
    prediction = g4["prediction"]
    m = g5["passes"][0]["metrics"]["CaptionClassification"]
    bars = []
    for item in prediction["top5"]:
        width = 100 * item["probability"]
        bars.append(f"<tr><td>{html.escape(item['label'])}</td><td>{width:.2f}%</td><td><div class='bar'><i style='width:{width:.3f}%'></i></div></td></tr>")
    body = io(
        "1. 30帧全局语义的再编码",
        "<span class='shape'>[1,30,1024]</span>视觉骨干对每帧给出一个1024维全局向量。",
        "先LayerNorm，再经过2层Transformer Encoder，让30帧进一步交换事件级信息。随后对时间维求平均。",
        "<span class='shape'>[1,1024]</span>整段视频的事件特征。",
        "G4真实记录features=[1,1024]、float32、全部有限。",
    ) + io(
        "2. 线性分类器",
        "<span class='shape'>[1,1024]</span>整段视频事件特征。",
        "一个线性层为23种比赛事件各输出一个logit；Softmax只在解释/评估时把logit转成概率。",
        "<span class='shape'>[1,23]</span>例如goal、corner、yellow card、substitution等。",
        "G4真实记录logits=[1,23]、float32、全部有限。",
        "purple",
    )
    body += f"<h2>一个真实样本的23类输出</h2><div class='grid'><div class='card'><p>真实标签：<b>{html.escape(prediction['ground_truth_label'])}</b></p><p>模型Top-1：<b>{html.escape(prediction['top5'][0]['label'])}</b></p><p>真实标签排名：{prediction['ground_truth_rank']}</p></div><div class='card'><table><thead><tr><th>类别</th><th>概率</th><th></th></tr></thead><tbody>{''.join(bars)}</tbody></table></div></div>"
    body += metrics([('固定集准确率',m['classification_accuracy'],''),('Macro F1',m['macro_f1'],''),('平均置信度',m['avg_confidence'],''),('样本数',m['total_samples'],'')])
    body += "<div class='note'><b>与VideoCaption的区别：</b>VideoCaption比较视频和任意文字的相似度；CaptionClassification直接从固定23个事件类别中选一个。它不是自然语言生成头。</div>"
    return document("任务头 5：CaptionClassification 事件分类", body)


def main() -> int:
    for path in (G4, G5, OVERLAY_0, OVERLAY_1):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    g4 = json.loads(G4.read_text(encoding="utf-8"))
    g5 = json.loads(G5.read_text(encoding="utf-8"))
    if g4.get("status") != "passed" or g5.get("status") != "passed":
        raise RuntimeError("Source G4/G5 evidence must be passed")
    temporary = Path(tempfile.mkdtemp(prefix=".task_head_explanations.", dir=OUTPUT_PARENT))
    pages = {
        "index.html": index_page(g5),
        "detection.html": detection_page(g5),
        "lines.html": lines_page(g5),
        "keypoints.html": keypoints_page(g5),
        "video_caption.html": video_caption_page(g5),
        "caption_classification.html": classification_page(g4, g5),
    }
    for name, content in pages.items():
        (temporary / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "scope": "html_explanation_generated_from_existing_g4_g5_evidence",
        "model_imported": False,
        "gpu_used": False,
        "inference_executed": False,
        "training_executed": False,
        "sources": {
            str(G4): {"bytes": G4.stat().st_size, "sha256": sha256(G4)},
            str(G5): {"bytes": G5.stat().st_size, "sha256": sha256(G5)},
            str(OVERLAY_0): {"bytes": OVERLAY_0.stat().st_size, "sha256": sha256(OVERLAY_0)},
            str(OVERLAY_1): {"bytes": OVERLAY_1.stat().st_size, "sha256": sha256(OVERLAY_1)},
        },
        "pages": {
            name: {"bytes": (temporary / name).stat().st_size, "sha256": sha256(temporary / name)}
            for name in sorted(pages)
        },
        "limitations": [
            "G5 is a fixed small evaluation, not a paper-scale test result.",
            "Full detection, line, and keypoint raw output tensors were not persisted by G5.",
            "Module input transformations are code-confirmed; output fingerprints and metrics are run-confirmed.",
        ],
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.rename(OUTPUT_DIR)
    print(json.dumps({"status": "passed", "output": str(OUTPUT_DIR), "pages": sorted(pages)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
