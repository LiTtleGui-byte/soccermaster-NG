#!/usr/bin/env python3
"""Regenerate the five 2-fps overview sheets with absolute timestamps."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/commentary_multi_event_luna_5_20260818/manifest.json"
DEFAULT_OUTPUT = ROOT / "reports/commentary_adaptive_luna_pilot_5_20260818"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    assets_dir = args.output / "coarse_evidence"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_items = []
    for item in source["items"]:
        video = Path(item["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        sheets = []
        for part, start in enumerate((0, 15), 1):
            output = assets_dir / f"{item['annotation_id']}_part{part}_absolute_time.jpg"
            if output.exists():
                raise FileExistsError(output)
            filter_graph = (
                "fps=2,"
                "scale=320:180:force_original_aspect_ratio=decrease,"
                "pad=320:180:(ow-iw)/2:(oh-ih)/2:black,"
                "drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=16:"
                "fontcolor=white:box=1:boxcolor=black@0.65,"
                "tile=6x5:padding=4:margin=4:color=black"
            )
            run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-copyts",
                    "-ss",
                    str(start),
                    "-t",
                    "15",
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
            sheets.append(str(output))
        output_items.append(
            {
                **item,
                "contact_sheets": sheets,
                "contact_sheet_contract": (
                    "part1 covers absolute video seconds 0-15 and part2 covers 15-30; "
                    "labels in both sheets use the original 0-30 second timeline"
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "purpose": "adaptive_luna_pilot_corrected_two_fps_overview",
        "known_fix": "part2 timestamps no longer restart from zero",
        "source_manifest": str(args.source.resolve()),
        "items": output_items,
    }
    output_path = args.output / "coarse_manifest.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
