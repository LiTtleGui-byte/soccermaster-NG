#!/usr/bin/env python3
"""Select 48 diagnostic clips and render blind CPU-only contact sheets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/commentary_attribute_probe_pilot_120_20260818/coordinator_manifest.json"
OUTPUT = ROOT / "reports/commentary_attribute_probe_pilot_48_20260818"
TARGETS = {
    "shot_or_save": 12,
    "cross": 10,
    "pass_or_build_up": 12,
    "foul_or_free_kick": 12,
    "goal": 2,
}


def stable_key(row: dict[str, Any], salt: str) -> str:
    raw = f"{salt}:{row['dataset_index']}:{row['match_id']}".encode()
    return hashlib.sha256(raw).hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["items"]
    selected: list[dict[str, Any]] = []
    match_counts: Counter[str] = Counter()
    for label, count in TARGETS.items():
        pool = [row for row in rows if row["silver_event"] == label]
        if len(pool) < count:
            raise RuntimeError(f"{label} has {len(pool)} candidates, needs {count}")
        for occurrence in range(count):
            remaining = [row for row in pool if row not in selected]
            remaining.sort(
                key=lambda row: (
                    match_counts[row["match_id"]],
                    stable_key(row, f"attribute48:{label}:{occurrence}"),
                )
            )
            chosen = remaining[0]
            selected.append(chosen)
            match_counts[chosen["match_id"]] += 1

    if len(selected) != 48:
        raise RuntimeError(f"expected 48 samples, got {len(selected)}")
    selected.sort(key=lambda row: stable_key(row, "attribute48:blind-order"))

    OUTPUT.mkdir(parents=True)
    sheets_dir = OUTPUT / "coarse_evidence"
    sheets_dir.mkdir()
    blind_items = []
    coordinator_items = []
    for blind_position, source_row in enumerate(selected, 1):
        video = Path(source_row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        sheet_paths = []
        for part, start in enumerate((0, 15), 1):
            output = sheets_dir / f"{blind_position:03d}_{source_row['annotation_id']}_part{part}.jpg"
            filter_graph = (
                "fps=2,"
                "scale=320:180:force_original_aspect_ratio=decrease,"
                "pad=320:180:(ow-iw)/2:(oh-ih)/2:black,"
                "drawtext=text='%{pts\\:hms}':x=6:y=6:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.65,"
                "tile=6x5:padding=4:margin=4:color=black"
            )
            run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start), "-t", "15", "-i", str(video),
                "-vf", filter_graph, "-frames:v", "1", "-q:v", "3", str(output),
            ])
            sheet_paths.append(str(output))
        blind_items.append({
            "blind_position": blind_position,
            "annotation_id": source_row["annotation_id"],
            "dataset_index": source_row["dataset_index"],
            "match_id": source_row["match_id"],
            "video_path": str(video),
            "duration_seconds": duration,
            "contact_sheets": sheet_paths,
            "contact_sheet_contract": "part1=0-15s, part2=15-30s, 2fps, 30 frames per sheet",
        })
        coordinator_items.append({
            **blind_items[-1],
            "selection_silver_event": source_row["silver_event"],
            "selection_silver_multi_event": source_row["silver_multi_event"],
            "selection_silver_primary_ambiguous": source_row["silver_primary_ambiguous"],
        })
        print(f"prepared {blind_position}/48 {source_row['annotation_id']}", flush=True)

    blind_manifest = {
        "schema_version": 1,
        "purpose": "codex_video_only_attribute_development_labels",
        "labels_hidden": ["reference_commentary", "reference_derived_event", "model_predictions"],
        "sample_count": len(blind_items),
        "items": blind_items,
    }
    coordinator_manifest = {
        "schema_version": 1,
        "purpose": "development_attribute_probe_not_holdout",
        "selection_targets": TARGETS,
        "sample_count": len(coordinator_items),
        "unique_match_count": len({row["match_id"] for row in coordinator_items}),
        "items": coordinator_items,
    }
    (OUTPUT / "blind_manifest.json").write_text(
        json.dumps(blind_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "coordinator_manifest.json").write_text(
        json.dumps(coordinator_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = {
        "status": "AWAITING_CODEX_VIDEO_ONLY_LABELS",
        "cpu_contact_sheets_prepared": True,
        "video_content_accessed_for_sheet_rendering": True,
        "reference_content_accessed_during_annotation": False,
        "torch_imported": False,
        "gpu_used": False,
        "feature_extraction_started": False,
        "probe_training_started": False,
    }
    (OUTPUT / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status["status"],
        "sample_count": len(blind_items),
        "unique_match_count": coordinator_manifest["unique_match_count"],
        "output": str(OUTPUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
