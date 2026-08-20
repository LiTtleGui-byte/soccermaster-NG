#!/usr/bin/env python3
"""Render the eight manually labeled non-outfield tracks and Qwen outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path("/home/tianlin/SoccerMaster")
REPORT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004"
EVALUATION = REPORT / "evaluation.json"
SOURCE_CROPS = REPO / "reports/g10/20260818_team_color_diagnostic/annotation_crops"
OUTPUT = REPORT / "nonplayer_role_summary.jpg"


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(EVALUATION.read_text(encoding="utf-8"))
    rows = [row for row in result["track_details"] if row["manual_role"] != "outfield_player"]
    if len(rows) != 8:
        raise AssertionError(f"Expected 8 non-outfield tracks, got {len(rows)}")
    width, header_h, row_h = 1240, 150, 210
    canvas = Image.new("RGB", (width, header_h + row_h * len(rows)), "#10161d")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "SNGS-10004: same tracks, PRTReID -> Qwen2.5-VL", fill="#f2f6fa", font=font(27))
    draw.text(
        (24, 58),
        "Explicit goalkeeper/referee: PRT 0/8 -> Qwen 3/8, with 0/39 outfield false rejects",
        fill="#8fd2ff",
        font=font(20),
    )
    draw.text(
        (24, 91),
        "Treating 'other' as nonplayer reaches 8/8 but wrongly rejects 25/39 outfield tracks.",
        fill="#ffbd73",
        font=font(18),
    )
    for index, row in enumerate(rows):
        top = header_h + index * row_h
        draw.rectangle((12, top + 5, width - 12, top + row_h - 5), fill="#18222d", outline="#344556", width=2)
        source = Image.open(SOURCE_CROPS / f"track_{row['track_id']:03d}.jpg").convert("RGB")
        source.thumbnail((620, 180), Image.Resampling.LANCZOS)
        canvas.paste(source, (22, top + (row_h - source.height) // 2))
        majority = row["generation_majority_role"]
        correct_specific = majority in {"goalkeeper", "referee"}
        color = "#75e6a4" if correct_specific else "#ffbd73"
        x = 670
        draw.text((x, top + 25), f"track {row['track_id']}   truth: {row['manual_role']}", fill="#f2f6fa", font=font(20))
        draw.text((x, top + 65), f"views: {' / '.join(str(v) for v in row['view_roles'])}", fill="#c7d4df", font=font(18))
        draw.text((x, top + 104), f"Qwen track result: {majority}", fill=color, font=font(21))
        draw.text((x, top + 145), "PRTReID track result: player", fill="#ff7f88", font=font(17))
    canvas.save(OUTPUT, quality=94)
    print(json.dumps({"status": "passed", "output": str(OUTPUT), "tracks": 8}))


if __name__ == "__main__":
    main()
