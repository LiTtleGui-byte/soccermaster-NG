#!/usr/bin/env python3
"""Compare Step-3 ReID KMeans teams with the fixed cross-frame color method.

This is a CPU-only development replay.  It reads the Refiner and no-ReID
Step-3 archives, but does not write labels back to either archive.  Manual
labels are loaded only after both unsupervised predictions have been made.
"""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


REPO = Path("/home/tianlin/SoccerMaster")
sys.path.insert(0, str(REPO / "experiments/soccerfactory_visualization"))
from replay_team_assignment_candidate import extract as extract_track_features  # noqa: E402


VIDEO_ID = "10004"
REFINER_ARCHIVE = REPO / ".runtime/g10/sngs10004_refiner_probe/run2/output/refined_sn-gamestate.pklz"
NO_REID_ARCHIVE = REPO / ".runtime/g10/sngs10004_step3_no_reid/run1/states/sn-gamestate.pklz"
ANNOTATIONS = REPO / "reports/g10/20260818_team_color_diagnostic/sngs10004_track_annotations.json"
OUTPUT_DIR = REPO / "reports/g10/20260819_step3_color_team_replay"
MARGIN_THRESHOLD = 0.15
MIN_VALID_CROPS = 3


def load_archive(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
        images = pd.read_pickle(archive.open(f"{VIDEO_ID}_image.pkl"))
    return detections, images


def load_track_ids(path: Path) -> pd.Series:
    with zipfile.ZipFile(path) as archive:
        detections = pd.read_pickle(archive.open(f"{VIDEO_ID}.pkl"))
    track_ids = detections.track_id.astype(int).copy()
    del detections
    return track_ids


def choose_mapping(track_ids: list[int], truth: dict[int, str], predictions: dict[int, int]) -> dict[str, Any]:
    options = []
    for mapping in ({0: "blue", 1: "claret"}, {0: "claret", 1: "blue"}):
        incorrect = [track_id for track_id in track_ids if mapping[predictions[track_id]] != truth[track_id]]
        blue = [track_id for track_id in track_ids if truth[track_id] == "blue"]
        claret = [track_id for track_id in track_ids if truth[track_id] == "claret"]
        blue_correct = sum(mapping[predictions[track_id]] == "blue" for track_id in blue)
        claret_correct = sum(mapping[predictions[track_id]] == "claret" for track_id in claret)
        options.append(
            {
                "cluster_to_team": {str(key): value for key, value in mapping.items()},
                "correct": len(track_ids) - len(incorrect),
                "total": len(track_ids),
                "accuracy": (len(track_ids) - len(incorrect)) / len(track_ids),
                "balanced_accuracy": 0.5 * (blue_correct / len(blue) + claret_correct / len(claret)),
                "incorrect_track_ids_original": incorrect,
            }
        )
    return max(options, key=lambda row: (row["accuracy"], row["balanced_accuracy"]))


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, value: float, color: str, label: str) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 34), radius=8, fill="#27313c")
    draw.rounded_rectangle((x, y, x + round(width * value), y + 34), radius=8, fill=color)
    draw.text((x + 10, y + 5), f"{label}: {value:.1%}", fill="white", font=font(18))


def draw_summary(result: dict[str, Any], tracks: list[dict[str, Any]]) -> Image.Image:
    image = Image.new("RGB", (1500, 900), "#10161d")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), "Step 3 team assignment: ReID KMeans vs jersey color", fill="white", font=font(34))
    draw.text((40, 78), "SNGS-10004 | 49 tracks | evaluation on 39 manually reviewed outfield tracks", fill="#cbd5df", font=font(21))
    draw.text((40, 145), "Manual outfield accuracy", fill="white", font=font(25))
    bar(draw, 40, 190, 650, result["kmeans_evaluation"]["accuracy"], "#de6a73", "ReID KMeans")
    bar(draw, 40, 240, 650, result["color_evaluation"]["accuracy"], "#4dbb87", "Jersey color")
    draw.text((40, 320), "Track counts (all 49; anonymous clusters)", fill="white", font=font(25))
    kcounts = result["all_track_counts"]["kmeans_cluster"]
    ccounts = result["all_track_counts"]["color_cluster"]
    draw.text((55, 370), f"ReID KMeans:  {kcounts['0']} / {kcounts['1']}", fill="#f0b3b8", font=font(22))
    draw.text((55, 410), f"Jersey color: {ccounts['0']} / {ccounts['1']}", fill="#9fe0bd", font=font(22))
    draw.text((40, 480), "Per-track result", fill="white", font=font(25))
    draw.text((40, 520), "green = correct, red = incorrect, gray = no outfield team label", fill="#cbd5df", font=font(18))
    cell_w, cell_h = 27, 90
    start_x, start_y = 42, 570
    for position, row in enumerate(tracks):
        x = start_x + position * cell_w
        truth = row["manual_team"]
        color = "#697580"
        if truth in {"blue", "claret"} and row["manual_role"] == "outfield_player":
            color = "#4dbb87" if row["color_team_evaluation"] == truth else "#de6a73"
        draw.rectangle((x, start_y, x + cell_w - 3, start_y + cell_h), fill=color)
        draw.text((x + 4, start_y + 5), str(row["original_track_id"]), fill="white", font=font(12))
        marker = "!" if row["methods_disagree_after_mapping"] else ""
        draw.text((x + 7, start_y + 55), marker, fill="#11161c", font=font(18))
    draw.text((40, 690), f"Methods disagree on {result['method_disagreement']['all_tracks']} / 49 tracks after evaluation mapping.", fill="white", font=font(21))
    draw.text((40, 735), f"Safe color acceptance: {result['safe_acceptance']['accepted']} / 49 tracks; role gate is not trustworthy because Step 3 labels every track as player.", fill="#ffd479", font=font(19))
    draw.text((40, 795), "Decision: keep color as the team candidate; do not call it production-safe until goalkeeper/referee gating is fixed.", fill="#9fe0bd", font=font(21))
    return image


def context_crop(image: Image.Image, bbox: Any) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox)
    pad_x, pad_y = max(16, 0.4 * width), max(12, 0.15 * height)
    return image.crop((max(0, int(x - pad_x)), max(0, int(y - pad_y)), min(image.width, int(x + width + pad_x)), min(image.height, int(y + height + pad_y)))).convert("RGB")


def draw_conflicts(detections: pd.DataFrame, images: pd.DataFrame, rows: list[dict[str, Any]]) -> Image.Image:
    conflicts = [row for row in rows if row["methods_disagree_after_mapping"]]
    reps = detections.loc[detections.groupby("track_id").bbox_conf.idxmax()].set_index("track_id")
    path_by_id = dict(zip(images.id, images.file_path))
    panel_w, panel_h, columns = 260, 245, 5
    header_h = 90
    sheet = Image.new("RGB", (panel_w * columns, header_h + panel_h * math.ceil(len(conflicts) / columns)), "#10161d")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "Tracks where ReID KMeans and jersey color disagree", fill="white", font=font(28))
    draw.text((20, 52), "IDs are original Refiner track IDs; manual truth is shown only for evaluation.", fill="#cbd5df", font=font(17))
    for position, row in enumerate(conflicts):
        rep = reps.loc[row["new_track_id"]]
        crop = context_crop(Image.open(Path(str(path_by_id[rep.image_id]))).convert("RGB"), rep.bbox_ltwh)
        crop.thumbnail((panel_w - 16, 155), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), "#202a34")
        panel.paste(crop, ((panel_w - crop.width) // 2, 5))
        pd = ImageDraw.Draw(panel)
        pd.text((8, 166), f"ID {row['original_track_id']} -> {row['new_track_id']}", fill="white", font=font(16))
        pd.text((8, 190), f"KMeans {row['kmeans_team_evaluation']} | color {row['color_team_evaluation']}", fill="#ffd479", font=font(14))
        pd.text((8, 214), f"manual {row['manual_team']} ({row['manual_role']})", fill="#cbd5df", font=font(14))
        sheet.paste(panel, ((position % columns) * panel_w, header_h + (position // columns) * panel_h))
    return sheet


def main() -> None:
    outputs = [OUTPUT_DIR / name for name in ("result.json", "team_method_comparison.png", "team_assignment_conflicts.png", "README.md")]
    occupied = [str(path) for path in outputs if path.exists()]
    if occupied:
        raise FileExistsError(f"Refusing to overwrite: {occupied}")

    refiner_track_ids = load_track_ids(REFINER_ARCHIVE)
    detections, images = load_archive(NO_REID_ARCHIVE)
    if len(refiner_track_ids) != 3176 or len(detections) != 3176 or detections.track_id.nunique() != 49 or len(images) != 255:
        raise AssertionError("Fixed Step-3 input identity changed")
    if not refiner_track_ids.index.equals(detections.index):
        raise AssertionError("Cannot establish Refiner-to-no-ReID row lineage")
    lineage = pd.DataFrame({"original": refiner_track_ids, "new": detections.track_id.astype(int)})
    if lineage.groupby("original").new.nunique().max() != 1 or lineage.groupby("new").original.nunique().max() != 1:
        raise AssertionError("No-ReID Step 3 unexpectedly merged or split a track")
    old_to_new = lineage.drop_duplicates().set_index("original").new.astype(int).to_dict()
    new_to_old = {new: old for old, new in old_to_new.items()}

    extracted_rows, features = extract_track_features(VIDEO_ID, detections, images)
    track_table = pd.DataFrame(extracted_rows)
    labels = KMeans(n_clusters=2, random_state=0).fit_predict(features).astype(int)
    model = KMeans(n_clusters=2, random_state=0).fit(features)
    if not np.array_equal(labels, model.labels_):
        raise AssertionError("Deterministic color clustering changed within one run")
    distances = model.transform(features)
    color_by_new = {int(track_id): int(label) for track_id, label in zip(track_table.track_id, labels)}
    kmeans_by_new = detections.groupby("track_id").team_cluster.first().astype(int).to_dict()

    annotation = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    manual = {int(row["track_id"]): row for row in annotation["labels"]}
    if set(manual) != set(old_to_new):
        raise AssertionError("Manual labels and Refiner track identity differ")
    outfield = sorted(track_id for track_id, row in manual.items() if row["role"] == "outfield_player" and row["team"] in {"blue", "claret"})
    truth = {track_id: manual[track_id]["team"] for track_id in outfield}
    kmeans_by_old = {old: kmeans_by_new[new] for old, new in old_to_new.items()}
    color_by_old = {old: color_by_new[new] for old, new in old_to_new.items()}
    kmeans_eval = choose_mapping(outfield, truth, kmeans_by_old)
    color_eval = choose_mapping(outfield, truth, color_by_old)
    kmap = {int(key): value for key, value in kmeans_eval["cluster_to_team"].items()}
    cmap = {int(key): value for key, value in color_eval["cluster_to_team"].items()}

    margins: dict[int, float] = {}
    valid_crops = dict(zip(track_table.track_id.astype(int), track_table.valid_color_crops.astype(int)))
    for index, track_id in enumerate(track_table.track_id.astype(int)):
        label = int(labels[index])
        other = 1 - label
        margins[track_id] = float((distances[index, other] - distances[index, label]) / max(distances[index, other], 1e-12))

    rows = []
    accepted = 0
    for old in sorted(old_to_new):
        new = old_to_new[old]
        decision = "accepted"
        if valid_crops[new] < MIN_VALID_CROPS:
            decision = "unknown_low_evidence"
        elif margins[new] < MARGIN_THRESHOLD:
            decision = "review_required_color"
        accepted += decision == "accepted"
        rows.append(
            {
                "original_track_id": old,
                "new_track_id": new,
                "manual_role": manual[old]["role"],
                "manual_team": manual[old]["team"],
                "kmeans_cluster": kmeans_by_new[new],
                "kmeans_team_side_saved": str(detections.loc[detections.track_id == new, "team"].iloc[0]),
                "kmeans_team_evaluation": kmap[kmeans_by_new[new]],
                "color_cluster": color_by_new[new],
                "color_team_evaluation": cmap[color_by_new[new]],
                "valid_color_crops": valid_crops[new],
                "color_margin": margins[new],
                "color_decision_without_reliable_role_gate": decision,
                "methods_disagree_after_mapping": kmap[kmeans_by_new[new]] != cmap[color_by_new[new]],
            }
        )

    result = {
        "status": "passed",
        "scope": "CPU-only fixed-match Step-3 team-method replay; no archive write-back",
        "inputs": {"refiner": str(REFINER_ARCHIVE), "step3_no_reid": str(NO_REID_ARCHIVE), "manual_evaluation": str(ANNOTATIONS)},
        "seed": 0,
        "gpu_used": False,
        "labels_written_back": False,
        "unchanged_contract": {"rows": 3176, "tracks": 49, "images": 255, "track_lineage_one_to_one": True, "track_id_role_bbox_pitch_modified": False},
        "manual_labels_used_for_candidate_generation": False,
        "manual_outfield_tracks": len(outfield),
        "kmeans_evaluation": kmeans_eval,
        "color_evaluation": color_eval,
        "all_track_counts": {
            "kmeans_cluster": {str(key): int(value) for key, value in pd.Series(kmeans_by_new).value_counts().sort_index().items()},
            "color_cluster": {str(key): int(value) for key, value in pd.Series(color_by_new).value_counts().sort_index().items()},
        },
        "method_disagreement": {
            "all_tracks": sum(row["methods_disagree_after_mapping"] for row in rows),
            "original_track_ids": [row["original_track_id"] for row in rows if row["methods_disagree_after_mapping"]],
        },
        "safe_acceptance": {
            "accepted": accepted,
            "total": 49,
            "warning": "Step-3 role is player for all tracks, so this acceptance policy cannot reliably exclude goalkeepers/referees.",
        },
        "decision": "prefer_color_candidate_over_reid_kmeans_but_do_not_connect_as_production_team_output_until_role_gate_is_fixed",
        "track_comparison": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_summary(result, rows).save(OUTPUT_DIR / "team_method_comparison.png")
    draw_conflicts(detections, images, rows).save(OUTPUT_DIR / "team_assignment_conflicts.png")
    readme = f"""# Step 3 球队方法对比（SNGS-10004）

- ReID KMeans：{kmeans_eval['correct']}/{len(outfield)}（{kmeans_eval['accuracy']:.1%}）
- 跨帧球衣颜色：{color_eval['correct']}/{len(outfield)}（{color_eval['accuracy']:.1%}）
- 两种方法在全部49条轨迹中有{result['method_disagreement']['all_tracks']}条结论不同。
- 颜色法保留为球队模块候选，但当前Step 3把所有轨迹都标成player，无法可靠排除门将/裁判，因此本轮不写回生产字段。

本实验只读取固定Refiner/no-ReID归档并生成比较结果；没有使用GPU，没有修改track_id、role、bbox_pitch或任何源归档。
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "passed", "kmeans": kmeans_eval["accuracy"], "color": color_eval["accuracy"], "disagreements": result["method_disagreement"]["all_tracks"], "accepted": accepted}))


if __name__ == "__main__":
    main()
