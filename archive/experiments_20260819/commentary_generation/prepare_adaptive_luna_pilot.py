#!/usr/bin/env python3
"""Build dense, time-correct evidence sheets for the five-video Luna pilot.

This script is CPU-only.  It consumes blind first-pass review decisions and
extracts only the requested local windows from the original videos.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/commentary_adaptive_luna_pilot_5_20260818/coarse_manifest.json"
DEFAULT_DECISIONS = ROOT / "reports/commentary_adaptive_luna_pilot_5_20260818/trigger_decisions.json"
DEFAULT_OUTPUT = ROOT / "reports/commentary_adaptive_luna_pilot_5_20260818"
CELL_WIDTH = 480
CELL_HEIGHT = 270
TILE_COLUMNS = 4
MAX_FRAMES_PER_SHEET = 12


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def probe_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def image_token_estimate(width: int, height: int) -> int:
    """GPT-5.6 original-detail patch estimate; exact usage needs API counting."""
    return math.ceil(width / 32) * math.ceil(height / 32)


def detect_scene_cuts(video: Path, threshold: float = 0.10) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw = [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    collapsed: list[float] = []
    for value in raw:
        if not collapsed or value - collapsed[-1] > 0.30:
            collapsed.append(value)
    return collapsed


def add_long_shot_boundary_windows(
    windows: list[dict], cuts: list[float], duration: float
) -> tuple[list[dict], list[dict]]:
    """Cover cuts followed by a sustained shot of at least five seconds.

    A long player/referee close-up after play is often the only coarse clue that
    the preceding one or two seconds contain the event outcome.
    """
    candidates = [cut for cut in cuts if 0.5 <= cut < duration - 0.5]
    if not candidates:
        return windows, []
    spans = []
    for cut in candidates:
        following = next((other for other in cuts if other > cut + 0.30), duration)
        spans.append((following - cut, cut, following))
    updated = [dict(window) for window in windows]
    guards = []
    for shot_length, cut, following in spans:
        if shot_length < 5.0:
            continue
        # Broadcast reaction close-ups commonly lag the decisive action.  Bias
        # the evidence window backwards so the causal play is not cropped out.
        required_start = max(0.0, cut - 4.0)
        required_end = min(duration, cut + 1.0)
        containing = next(
            (index for index, window in enumerate(updated)
             if float(window["start_seconds"]) <= required_start
             and float(window["end_seconds"]) >= required_end),
            None,
        )
        reason = "画面切换检测：向前追溯持续特写/长镜头所对应的场上动作与结果。"
        if containing is None:
            replaced = []
            survivors = []
            for window in updated:
                window_start = float(window["start_seconds"])
                window_end = float(window["end_seconds"])
                overlap = max(0.0, min(window_end, required_end) - max(window_start, required_start))
                denominator = min(window_end - window_start, required_end - required_start)
                if denominator > 0 and overlap / denominator >= 0.5:
                    replaced.append([window_start, window_end])
                else:
                    survivors.append(window)
            updated = survivors
            updated.append(
                {
                    "start_seconds": round(required_start, 3),
                    "end_seconds": round(required_end, 3),
                    "target_fps": 8,
                    "reason": reason,
                    "added_by": "scene_cut_sustained_following_shot",
                    "replaced_overlapping_windows": replaced,
                }
            )
        else:
            window = updated[containing]
            window["target_fps"] = max(int(window["target_fps"]), 8)
            window["scene_cut_guard_added"] = True
        guards.append(
            {
                "selected_cut_seconds": cut,
                "following_shot_end_seconds": following,
                "following_shot_duration_seconds": round(shot_length, 3),
                "required_window": [round(required_start, 3), round(required_end, 3)],
            }
        )
    updated.sort(key=lambda window: (float(window["start_seconds"]), float(window["end_seconds"])))
    return updated, guards


def normalize_review_windows(
    windows: list[dict], duration: float, max_seconds: float = 4.0
) -> list[dict]:
    normalized = []
    for original in windows:
        window = dict(original)
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
        if end - start > max_seconds:
            midpoint = (start + end) / 2
            start = max(0.0, midpoint - max_seconds / 2)
            end = min(duration, start + max_seconds)
            start = max(0.0, end - max_seconds)
            window["original_requested_window"] = [
                float(original["start_seconds"]), float(original["end_seconds"])
            ]
            window["window_normalized_to_seconds"] = max_seconds
        window["start_seconds"] = round(start, 3)
        window["end_seconds"] = round(end, 3)
        normalized.append(window)
    return normalized


def validate_decisions(source: dict, decisions: dict) -> None:
    source_ids = [item["annotation_id"] for item in source["items"]]
    rows = decisions.get("items")
    if not isinstance(rows, list) or [row.get("annotation_id") for row in rows] != source_ids:
        raise ValueError("trigger decision IDs/order must exactly match the five source items")
    for row in rows:
        windows = row.get("review_windows", [])
        if bool(windows) != bool(row.get("needs_dense_review")):
            raise ValueError(f"inconsistent dense-review decision for {row['annotation_id']}")
        for window in windows:
            start = float(window["start_seconds"])
            end = float(window["end_seconds"])
            fps = int(window["target_fps"])
            if not (0 <= start < end <= 30.1):
                raise ValueError(f"invalid window for {row['annotation_id']}: {start}-{end}")
            if fps not in {4, 6, 8}:
                raise ValueError(f"target_fps must be 4, 6, or 8, got {fps}")


def extract_window(
    *, video: Path, output_dir: Path, annotation_id: str, window_index: int,
    start: float, end: float, fps: int,
) -> list[dict]:
    duration = end - start
    max_segment_seconds = MAX_FRAMES_PER_SHEET / fps
    segments = []
    segment_start = start
    part = 1
    while segment_start < end - 1e-6:
        segment_end = min(end, segment_start + max_segment_seconds)
        expected_frames = max(1, math.ceil((segment_end - segment_start) * fps - 1e-9))
        rows = math.ceil(expected_frames / TILE_COLUMNS)
        output = output_dir / (
            f"{annotation_id}_window{window_index:02d}_part{part:02d}_"
            f"{segment_start:05.2f}-{segment_end:05.2f}_{fps}fps.jpg"
        )
        if output.exists():
            raise FileExistsError(output)
        # copyts keeps the labels on the original 0-30 second video timeline.
        filter_graph = (
            f"fps={fps},"
            f"scale={CELL_WIDTH}:{CELL_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_WIDTH}:{CELL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            "drawtext=text='%{pts\\:hms}':x=8:y=8:fontsize=20:"
            "fontcolor=white:box=1:boxcolor=black@0.70,"
            f"tile={TILE_COLUMNS}x{rows}:padding=4:margin=4:color=black"
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-copyts",
                "-ss",
                f"{segment_start:.3f}",
                "-t",
                f"{segment_end - segment_start:.3f}",
                "-i",
                str(video),
                "-vf",
                filter_graph,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(output),
            ]
        )
        width, height = probe_dimensions(output)
        segments.append(
            {
                "part": part,
                "start_seconds": round(segment_start, 3),
                "end_seconds": round(segment_end, 3),
                "fps": fps,
                "expected_frame_count": expected_frames,
                "sheet_path": str(output),
                "sheet_width": width,
                "sheet_height": height,
                "estimated_image_tokens_original_detail": image_token_estimate(width, height),
            }
        )
        segment_start = segment_end
        part += 1
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    validate_decisions(source, decisions)
    source_by_id = {item["annotation_id"]: item for item in source["items"]}

    assets_dir = args.output / "dense_evidence"
    assets_dir.mkdir(parents=True, exist_ok=True)
    result_items = []
    dense_tokens = 0
    for decision in decisions["items"]:
        annotation_id = decision["annotation_id"]
        source_item = source_by_id[annotation_id]
        video = Path(source_item["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        cuts = detect_scene_cuts(video)
        effective_windows = normalize_review_windows(
            decision.get("review_windows", []), float(source_item["duration_seconds"])
        )
        scene_guards = []
        if "sudden_closeup_after_play" in decision.get("trigger_reasons", []):
            effective_windows, scene_guards = add_long_shot_boundary_windows(
                effective_windows, cuts, float(source_item["duration_seconds"])
            )
        rendered_windows = []
        for index, window in enumerate(effective_windows, 1):
            sheets = extract_window(
                video=video,
                output_dir=assets_dir,
                annotation_id=annotation_id,
                window_index=index,
                start=float(window["start_seconds"]),
                end=float(window["end_seconds"]),
                fps=int(window["target_fps"]),
            )
            dense_tokens += sum(sheet["estimated_image_tokens_original_detail"] for sheet in sheets)
            rendered_windows.append({**window, "sheets": sheets})
        result_items.append(
            {
                **decision,
                "luna_review_windows": decision.get("review_windows", []),
                "review_windows": effective_windows,
                "scene_cut_detection": {
                    "threshold": 0.10,
                    "detected_cut_seconds": cuts,
                    "long_shot_guards": scene_guards,
                },
                "video_path": str(video),
                "coarse_contact_sheets": source_item["contact_sheets"],
                "rendered_review_windows": rendered_windows,
            }
        )

    coarse_tokens_per_video = 2 * math.ceil(1948 / 32) * math.ceil(924 / 32)
    output = {
        "schema_version": 1,
        "purpose": "adaptive_luna_dense_evidence_pilot_on_five_development_videos",
        "non_goal": "not a holdout evaluation and not training labels before human approval",
        "source_manifest": str(args.source.resolve()),
        "trigger_decisions": str(args.decisions.resolve()),
        "sampling": {
            "coarse": "2 fps over all 30 seconds, two sheets",
            "dense": "only selected windows, 480x270 cells, at most 12 frames per sheet",
            "timestamp_contract": "all dense-sheet labels are absolute seconds on the original video timeline",
        },
        "token_estimate": {
            "coarse_image_tokens_per_video_original_detail": coarse_tokens_per_video,
            "dense_image_tokens_total_all_five": dense_tokens,
            "note": "Patch estimate only; exact request usage must be measured by the API token-count endpoint.",
        },
        "items": result_items,
    }
    output_path = args.output / "dense_manifest.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
