#!/usr/bin/env python3
"""Build CPU-only MatchTime contact sheets and a record-level review manifest."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_ANNOTATIONS = Path(
    "/remote-home/haolinyang/sports/UniSoccer/train_data/video_clip_json/"
    "MatchTime/classification_test.json"
)
DEFAULT_VIDEO_ROOT = Path(
    "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/SoccerNetv2/"
    "MatchTime/SN-Caption-test-align"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/tianlin/SoccerMaster/reports/"
    "commentary_reference_audit_3256_20260814"
)
NUM_FRAMES = 30
TILE_WIDTH = 320
TILE_HEIGHT = 180
LABEL_HEIGHT = 20
SHEET_COLUMNS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--reuse-contact-sheets",
        action="store_true",
        help="Reuse a complete prior set of contact sheets and only rebuild items.jsonl.",
    )
    return parser.parse_args()


def middle_indices(video_length: int) -> list[int]:
    """Match data/video_caption.py get_frame_indices(..., sample='middle')."""
    if video_length <= 0:
        raise ValueError(f"Video has no decodable frames: {video_length}")
    accepted = min(NUM_FRAMES, video_length)
    intervals = np.linspace(0, video_length, num=accepted + 1).astype(int)
    ranges = [
        (start, intervals[index + 1] - 1)
        for index, start in enumerate(intervals[:-1])
    ]
    indices = [int((start + end) // 2) for start, end in ranges]
    if len(indices) < NUM_FRAMES:
        indices.extend([indices[-1]] * (NUM_FRAMES - len(indices)))
    return indices


def fit_frame(array: np.ndarray) -> Image.Image:
    source = Image.fromarray(array, mode="RGB")
    source.thumbnail((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (24, 24, 24))
    canvas.paste(
        source,
        ((TILE_WIDTH - source.width) // 2, (TILE_HEIGHT - source.height) // 2),
    )
    return canvas


def save_contact_sheet(
    frames: np.ndarray,
    indices: list[int],
    fps: float,
    output_path: Path,
    quality: int,
) -> None:
    rows = (NUM_FRAMES + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new(
        "RGB",
        (SHEET_COLUMNS * TILE_WIDTH, rows * (TILE_HEIGHT + LABEL_HEIGHT)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    for position, (frame, frame_index) in enumerate(zip(frames, indices)):
        column = position % SHEET_COLUMNS
        row = position // SHEET_COLUMNS
        left = column * TILE_WIDTH
        top = row * (TILE_HEIGHT + LABEL_HEIGHT)
        sheet.paste(fit_frame(frame), (left, top))
        seconds = frame_index / fps if fps > 0 else float("nan")
        draw.text(
            (left + 5, top + TILE_HEIGHT + 3),
            f"{position + 1:02d}  frame {frame_index}  {seconds:.2f}s",
            fill=(235, 235, 235),
        )
    temporary_path = output_path.with_suffix(".tmp.jpg")
    sheet.save(temporary_path, format="JPEG", quality=quality, optimize=True)
    os.replace(temporary_path, output_path)


def heartbeat(stop: threading.Event, started: float) -> None:
    while not stop.wait(30):
        print(f"[heartbeat] elapsed_seconds={time.monotonic() - started:.1f}", flush=True)


def main() -> int:
    args = parse_args()
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not args.video_root.is_dir():
        raise NotADirectoryError(args.video_root)
    if args.output_root.exists() and not args.reuse_contact_sheets:
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")
    if not 1 <= args.jpeg_quality <= 95:
        raise ValueError("--jpeg-quality must be between 1 and 95")

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 3256:
        raise RuntimeError(f"Expected 3256 records, got {type(records)} / {len(records)}")
    required = {"video", "caption", "comments_text_anonymized"}
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise RuntimeError(f"Record {index} lacks fields: {sorted(missing)}")

    counts = Counter(record["video"] for record in records)
    unique_videos = list(counts)
    duplicate_paths = {path: count for path, count in counts.items() if count > 1}
    if len(unique_videos) != 3251 or len(duplicate_paths) != 5:
        raise RuntimeError(
            f"Unexpected video cardinality: unique={len(unique_videos)} "
            f"duplicate_paths={len(duplicate_paths)}"
        )

    missing_videos = [
        relative_path
        for relative_path in unique_videos
        if not (args.video_root / relative_path).is_file()
    ]
    if missing_videos:
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} videos; first={missing_videos[0]}"
        )

    sheet_root = args.output_root / "contact_sheets"
    if args.reuse_contact_sheets:
        expected_sheets = {f"video_{index:04d}.jpg" for index in range(3251)}
        actual_sheets = {path.name for path in sheet_root.glob("*.jpg")}
        if actual_sheets != expected_sheets:
            raise RuntimeError(
                "Cannot reuse incomplete contact sheets: "
                f"expected={len(expected_sheets)} actual={len(actual_sheets)}"
            )
    else:
        sheet_root.mkdir(parents=True)
    started = time.monotonic()
    stop = threading.Event()
    monitor = threading.Thread(target=heartbeat, args=(stop, started), daemon=True)
    monitor.start()
    metadata: dict[str, dict[str, object]] = {}
    try:
        print(
            f"[start] records={len(records)} unique_videos={len(unique_videos)} "
            f"output_root={args.output_root}",
            flush=True,
        )
        for unique_index, relative_path in enumerate(unique_videos):
            video_path = args.video_root / relative_path
            reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            video_length = len(reader)
            fps = float(reader.get_avg_fps())
            indices = middle_indices(video_length)
            sheet_name = f"video_{unique_index:04d}.jpg"
            if not args.reuse_contact_sheets:
                frames = reader.get_batch(indices).asnumpy()
                if (
                    frames.shape[0] != NUM_FRAMES
                    or frames.ndim != 4
                    or frames.shape[-1] != 3
                ):
                    raise RuntimeError(
                        f"Unexpected decoded shape for {relative_path}: {frames.shape}"
                    )
                save_contact_sheet(
                    frames,
                    indices,
                    fps,
                    sheet_root / sheet_name,
                    args.jpeg_quality,
                )
            metadata[relative_path] = {
                "unique_video_index": unique_index,
                "contact_sheet": f"contact_sheets/{sheet_name}",
                "video_frame_count": int(video_length),
                "video_fps": fps,
                "sampled_frame_indices": indices,
            }
            if (unique_index + 1) % 25 == 0 or unique_index + 1 == len(unique_videos):
                print(
                    f"[progress] {'indexed' if args.reuse_contact_sheets else 'decoded'}="
                    f"{unique_index + 1}/{len(unique_videos)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )

        items_path = args.output_root / "items.jsonl"
        temporary_items = items_path.with_suffix(".tmp.jsonl")
        with temporary_items.open("w", encoding="utf-8") as handle:
            for dataset_index, record in enumerate(records):
                item = {
                    "dataset_index": dataset_index,
                    "video": record["video"],
                    "caption": record["caption"],
                    "comments_text_anonymized": record["comments_text_anonymized"],
                    **metadata[record["video"]],
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary_items, items_path)
    finally:
        stop.set()
        monitor.join(timeout=1)

    elapsed = time.monotonic() - started
    print(
        f"[done] records={len(records)} unique_videos={len(unique_videos)} "
        f"duplicate_paths={len(duplicate_paths)} elapsed_seconds={elapsed:.1f} exit_code=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
