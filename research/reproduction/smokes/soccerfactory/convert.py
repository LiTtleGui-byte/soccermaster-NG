#!/usr/bin/env python3
"""Convert the current isolated Step-3 state to the historical training-PKL schema."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from soccermaster.integrations.soccerfactory import convert_step3_to_training_frames


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "research/reproduction/smokes/soccerfactory/manifests/conversion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def refuse_output(path: Path) -> None:
    if not path.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError(f"Output must remain in the workspace: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    started = time.monotonic()
    manifest_path = parse_args().manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage")) != (
        1,
        "G10-B",
        "current_step3_to_training_pkl",
    ):
        raise AssertionError("Unexpected manifest identity")

    input_path = Path(manifest["input_step3"])
    output_path = Path(manifest["output_pkl"])
    result_path = Path(manifest["result_json"])
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    refuse_output(output_path)
    refuse_output(result_path)

    print("[PHASE] load_current_step3", flush=True)
    with zipfile.ZipFile(input_path) as archive:
        members = archive.namelist()
        required = {"summary.json", f"{manifest['video_id']}.pkl", f"{manifest['video_id']}_image.pkl"}
        if set(members) != required or archive.testzip() is not None:
            raise AssertionError(f"Unexpected or corrupt Step-3 ZIP members: {members}")
        detections = pd.read_pickle(archive.open(f"{manifest['video_id']}.pkl"))
        images = pd.read_pickle(archive.open(f"{manifest['video_id']}_image.pkl"))

    print("[PHASE] convert", flush=True)
    converted = convert_step3_to_training_frames(detections, images, manifest["video_id"])
    expected = manifest["expected"]
    people = sum(len(frame["people"]) for frame in converted.values())
    valid_camera = sum(bool(frame["valid_cam_params"]) for frame in converted.values())
    assertions = {
        "frame_count": len(converted) == expected["frames"],
        "people_count": people == expected["people"],
        "valid_camera_count": valid_camera == expected["valid_camera_frames"],
        "frame_keys": list(converted) == list(range(1, expected["frames"] + 1)),
        "frame_fields": all(set(frame) == {"people", "K", "R", "P", "valid_cam_params"} for frame in converted.values()),
        "person_fields": all(set(person) == {"id", "bbox_ltwh", "role", "legibility_score", "jersey_number"} for frame in converted.values() for person in frame["people"]),
    }
    if not all(assertions.values()):
        raise AssertionError(f"Conversion assertions failed: {assertions}")

    print("[PHASE] write_and_reload", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=False)
    with output_path.open("xb") as handle:
        pickle.dump(converted, handle, protocol=pickle.DEFAULT_PROTOCOL)
    with output_path.open("rb") as handle:
        reloaded = pickle.load(handle)
    if len(reloaded) != expected["frames"] or sum(len(frame["people"]) for frame in reloaded.values()) != expected["people"]:
        raise AssertionError("Written PKL failed the targeted reload check")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    dirty_files = subprocess.run(["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True).stdout.splitlines()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": manifest["stage"],
        "status": "passed",
        "conclusion": "current_step3_to_training_pkl_conversion_passed",
        "manifest": str(manifest_path),
        "git_commit": commit,
        "git_dirty_files": dirty_files,
        "python": sys.executable,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "input": {"path": str(input_path), "bytes": input_path.stat().st_size, "zip_members": members},
        "input_tables": {"detections": len(detections), "tracks": int(detections.track_id.nunique()), "images": len(images)},
        "output": {"path": str(output_path), "bytes": output_path.stat().st_size, "frames": len(converted), "people": people, "valid_camera_frames": valid_camera},
        "assertions": assertions,
        "timing": {"wall_seconds": round(time.monotonic() - started, 3), "max_rss_kib": usage.ru_maxrss},
        "gpu_used": False,
        "training": False,
        "not_validated": manifest["non_goals"],
        "schema_note": "The preserved historical training-PKL contract does not carry team, track_id, or bbox_pitch fields forward.",
    }
    atomic_json(result, result_path)
    print(f"[RESULT] passed frames={len(converted)} people={people} valid_camera={valid_camera}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
