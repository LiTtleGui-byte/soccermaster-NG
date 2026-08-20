#!/usr/bin/env python3
"""Create CPU-only temporal contact sheets for Luna's first-five event scan."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "reports/commentary_fact_calibration_24_20260818/items.json"
OUTPUT = ROOT / "reports/commentary_multi_event_luna_5_20260818"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> None:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    items = []
    for source in packet["items"][:5]:
        video = Path(source["video_path"])
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        sheets = []
        for half, start in enumerate((0, 15), 1):
            output = OUTPUT / f"{source['review_position']:02d}_{source['annotation_id']}_part{half}.jpg"
            filter_graph = (
                "fps=2,"
                "scale=320:180:force_original_aspect_ratio=decrease,"
                "pad=320:180:(ow-iw)/2:(oh-ih)/2:black,"
                "drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65,"
                "tile=6x5:padding=4:margin=4:color=black"
            )
            run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(start), "-t", "15", "-i", str(video),
                    "-vf", filter_graph, "-frames:v", "1", "-q:v", "3", str(output),
                ]
            )
            sheets.append(str(output))
        candidates = {x["source"]: x["text"] for x in source["anonymous_candidates"]}
        items.append(
            {
                "review_position": source["review_position"],
                "annotation_id": source["annotation_id"],
                "dataset_index": source["dataset_index"],
                "video_path": str(video),
                "duration_seconds": duration,
                "contact_sheets": sheets,
                "contact_sheet_contract": "part1 covers video seconds 0-15; part2 covers 15-30; each has 30 frames sampled at 2 fps in row-major order",
                "reference_commentary": source["reference_commentary"],
                "historical_prediction": candidates["historical"],
                "e1_best_prediction": candidates["e1_best"],
            }
        )
    manifest = {
        "schema_version": 1,
        "purpose": "luna_first_five_multi_event_scan_for_human_verification",
        "method": "CPU ffmpeg, two 2-fps temporal contact sheets per 30-second video",
        "items": items,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT / "manifest.json")


if __name__ == "__main__":
    main()
