#!/usr/bin/env python3
"""Build blinded, unfilled annotation templates without opening any video."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import sys


REPO = Path("/home/tianlin/SoccerMaster")
LOCAL_PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
PREFIX_MANIFEST = REPO / "reports/commentary_prefix_cache_200_20260814_run1/manifest.json"
E1_PREDICTIONS = REPO / "reports/commentary_parallel_20260814/e1_decoder_sweep_run1/predictions.jsonl"
PROTOCOL = REPO / "experiments/commentary_generation/VIDEO_EVENT_ANNOTATION_PROTOCOL_200.md"
OUTPUT_DIR = REPO / "reports/commentary_video_event_annotation_200_20260816"
SAMPLE_COUNT = 200
SEEDS = {"annotator_a": 2026081601, "annotator_b": 2026081602}
FIELDS = [
    "annotation_id",
    "video_path",
    "primary_event",
    "secondary_events",
    "observability",
    "confidence",
    "notes",
    "annotation_complete",
]


def require_runtime() -> None:
    if Path(sys.executable).resolve() != LOCAL_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")
    if os.environ.get("PYTHONPATH") != str(REPO):
        raise RuntimeError("PYTHONPATH must be exactly the local repository")
    if not PREFIX_MANIFEST.is_file() or not E1_PREDICTIONS.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError("A required local input is missing")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")


def load_rows() -> list[dict[str, object]]:
    manifest = json.loads(PREFIX_MANIFEST.read_text(encoding="utf-8"))
    e1_rows = [
        json.loads(line)
        for line in E1_PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != SAMPLE_COUNT or len(e1_rows) != SAMPLE_COUNT:
        raise RuntimeError("Expected exactly 200 aligned local records")
    packet_rows = []
    for sample, e1 in zip(samples, e1_rows):
        identity = (int(sample["ordinal"]), int(sample["dataset_index"]))
        if identity != (int(e1["ordinal"]), int(e1["dataset_index"])):
            raise RuntimeError(f"Identity mismatch: {identity}")
        if str(sample["reference_commentary"]) != str(e1["reference_commentary"]):
            raise RuntimeError(f"Reference identity mismatch: {identity}")
        video_path = str(e1["video_path"])
        if not video_path.startswith("/mnt/nas2/"):
            raise RuntimeError("Unexpected video path namespace")
        packet_rows.append(
            {
                "annotation_id": f"CE200-{identity[0]:03d}",
                "ordinal": identity[0],
                "dataset_index": identity[1],
                "video_path": video_path,
            }
        )
    if len({row["annotation_id"] for row in packet_rows}) != SAMPLE_COUNT:
        raise RuntimeError("Annotation IDs are not unique")
    return packet_rows


def csv_text(rows: list[dict[str, object]], seed: int) -> str:
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow(
            {
                "annotation_id": row["annotation_id"],
                "video_path": row["video_path"],
                "primary_event": "",
                "secondary_events": "",
                "observability": "",
                "confidence": "",
                "notes": "",
                "annotation_complete": "no",
            }
        )
    return buffer.getvalue()


def main() -> int:
    require_runtime()
    rows = load_rows()
    manifest = {
        "status": "prepared_not_annotated",
        "sample_count": SAMPLE_COUNT,
        "protocol": str(PROTOCOL),
        "source_files": [str(PREFIX_MANIFEST), str(E1_PREDICTIONS)],
        "video_content_accessed": False,
        "reference_commentary_exported_to_annotators": False,
        "model_outputs_exported_to_annotators": False,
        "annotation_seeds": SEEDS,
        "linkage": [
            {
                "annotation_id": row["annotation_id"],
                "ordinal": row["ordinal"],
                "dataset_index": row["dataset_index"],
            }
            for row in rows
        ],
    }
    readme = """# Fixed-200 video-event annotation packet

Status: prepared, not annotated.

`annotator_a.csv` and `annotator_b.csv` contain the same 200 fixed video path
strings in different fixed random orders. They intentionally exclude reference
commentary, model output, silver labels, and prefix features. Follow the linked
protocol before comparing the two completed files.

Preparing this packet did not open, stat, decode, or otherwise access any video.
Actual annotation remains blocked on explicit NAS/video authorization and the
availability of two independent human annotators plus a separate adjudicator.
"""
    payloads = {
        "README.md": readme,
        "annotation_manifest.json": json.dumps(manifest, indent=2) + "\n",
        "annotator_a.csv": csv_text(rows, SEEDS["annotator_a"]),
        "annotator_b.csv": csv_text(rows, SEEDS["annotator_b"]),
    }
    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    for name, content in payloads.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
    print(json.dumps({"status": "prepared", "files": sorted(payloads)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
