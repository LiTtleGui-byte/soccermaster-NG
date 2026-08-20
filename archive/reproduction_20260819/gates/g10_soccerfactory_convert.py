#!/usr/bin/env python3
"""Validate and reproduce the preserved SoccerFactory Step-3-to-PKL contract.

This is a local compatibility adapter for a missing historical conversion script.
It does not run TrackLab, Refiner, model inference, evaluation, or training.
"""

from __future__ import annotations

import argparse
import hashlib
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

import numpy as np
import pandas as pd


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_soccerfactory_sngs10004.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_local_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Output must stay inside {REPO}: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable output: {path}")


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.DEFAULT_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("gate") != "G10-B":
        raise AssertionError("Unexpected G10-B manifest identity")
    if manifest.get("stage") != "compatibility_conversion_static_validation":
        raise AssertionError("Unexpected G10-B manifest stage")
    return manifest


def validate_small_file(spec: dict[str, Any]) -> None:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != spec["bytes"]:
        raise AssertionError(f"File size changed: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"SHA256 changed: {path}")


def validate_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    sample = manifest["sample"]
    game = sample["source_game"]
    for path_key, size_key in (
        ("raw_video", "raw_video_bytes"),
        ("camera_labels", "camera_labels_bytes"),
    ):
        path = Path(game[path_key])
        if not path.is_file() or path.stat().st_size != game[size_key]:
            raise AssertionError(f"Source asset missing or size changed: {path}")

    validate_small_file(sample["clip_mapping"])
    mapping = json.loads(Path(sample["clip_mapping"]["path"]).read_text())
    mapping_rows = [row for row in mapping if int(row[-1]) == int(sample["video_id"])]
    expected_mapping = [
        game["competition"],
        game["season"],
        game["game"],
        game["half"],
        game["mapping_start"],
        game["mapping_stop"],
        game["mapping_sequence_id"],
    ]
    if mapping_rows != [expected_mapping]:
        raise AssertionError(f"Unexpected clip mapping: {mapping_rows}")

    validate_small_file(sample["sequence_metadata"])
    metadata = json.loads(Path(sample["sequence_metadata"]["path"]).read_text())
    records = [
        record
        for record in metadata[sample["split"]]
        if record["name"] == sample["sequence"]
    ]
    if records != [sample["sequence_metadata"]["record"]]:
        raise AssertionError(f"Unexpected sequence metadata: {records}")

    frames = sample["prepared_frames"]
    frame_root = Path(frames["root"])
    names = sorted(path.name for path in frame_root.glob("*.jpg"))
    expected_names = [f"{index:06d}.jpg" for index in range(1, frames["count"] + 1)]
    if names != expected_names:
        raise AssertionError("Prepared frame names are not contiguous")
    paths = [frame_root / name for name in names]
    if sum(path.stat().st_size for path in paths) != frames["total_bytes"]:
        raise AssertionError("Prepared frame total byte size changed")
    for endpoint in ("first", "last"):
        spec = frames[endpoint]
        path = frame_root / spec["name"]
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"Prepared frame endpoint changed: {path}")

    step3 = manifest["historical_step3"]
    step3_path = Path(step3["path"])
    if not step3_path.is_file() or step3_path.stat().st_size != step3["bytes"]:
        raise AssertionError("Historical Step-3 archive missing or size changed")
    with zipfile.ZipFile(step3_path) as archive:
        for name, expected_size in step3["members"].items():
            if archive.getinfo(name).file_size != expected_size:
                raise AssertionError(f"Historical Step-3 member changed: {name}")

    validate_small_file(manifest["golden_training_pkl"])
    return {
        "mapping_rows": mapping_rows,
        "prepared_frame_count": len(paths),
        "prepared_frame_bytes": sum(path.stat().st_size for path in paths),
        "mapping_span_inclusive": game["mapping_stop"] - game["mapping_start"] + 1,
    }


def require_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing {label} columns: {missing}")


def load_step3(manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_id = manifest["sample"]["video_id"]
    with zipfile.ZipFile(manifest["historical_step3"]["path"]) as archive:
        with archive.open(f"{video_id}.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open(f"{video_id}_image.pkl") as handle:
            images = pickle.load(handle)
    if not isinstance(detections, pd.DataFrame) or not isinstance(images, pd.DataFrame):
        raise TypeError("Historical Step-3 members must contain pandas DataFrames")
    require_columns(
        detections,
        {"image_id", "bbox_ltwh", "role", "legibility_score", "jersey_number"},
        "detection",
    )
    require_columns(images, {"id", "frame", "video_id", "parameters"}, "image")
    return detections, images


def optional_scalar(value: Any) -> Any:
    return None if pd.isna(value) else value


def convert_step3(
    detections: pd.DataFrame,
    images: pd.DataFrame,
    expected_video_id: str,
) -> dict[int, dict[str, Any]]:
    images = images.sort_values("frame", kind="stable")
    expected_frames = list(range(len(images)))
    if images["frame"].astype(int).tolist() != expected_frames:
        raise AssertionError("Step-3 image frames are not zero-based and contiguous")
    if set(images["video_id"].astype(str)) != {expected_video_id}:
        raise AssertionError("Step-3 image video_id does not match manifest")

    converted: dict[int, dict[str, Any]] = {}
    for image in images.itertuples(index=False):
        image_id = image.id
        frame_id = int(image.frame) + 1
        image_detections = detections.loc[detections["image_id"] == image_id]
        people = []
        for index, detection in image_detections.iterrows():
            people.append(
                {
                    "id": int(index),
                    "bbox_ltwh": np.asarray(detection["bbox_ltwh"]).copy(),
                    "role": optional_scalar(detection["role"]),
                    "legibility_score": float(detection["legibility_score"]),
                    "jersey_number": optional_scalar(detection["jersey_number"]),
                }
            )

        parameters = image.parameters
        if parameters is None or (isinstance(parameters, float) and np.isnan(parameters)):
            intrinsic = None
            extrinsic = None
            projection = None
            valid_camera = False
        else:
            principal = np.asarray(parameters["principal_point"], dtype=np.float64)
            intrinsic = np.array(
                [
                    [parameters["x_focal_length"], 0.0, principal[0]],
                    [0.0, parameters["y_focal_length"], principal[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            rotation = np.asarray(parameters["rotation_matrix"], dtype=np.float64)
            position = np.asarray(parameters["position_meters"], dtype=np.float64)
            translation = -rotation @ position
            extrinsic = np.concatenate((rotation, translation[:, None]), axis=1)
            projection = intrinsic @ extrinsic
            valid_camera = True

        converted[frame_id] = {
            "people": people,
            "K": intrinsic,
            "R": extrinsic,
            "P": projection,
            "valid_cam_params": valid_camera,
        }
    return converted


def compare_to_golden(
    actual: dict[int, dict[str, Any]],
    golden: dict[int, dict[str, Any]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    people_count = 0
    valid_camera_count = 0
    camera_max_abs = 0.0
    if list(actual) != list(golden):
        mismatches.append({"kind": "frame_keys"})

    for frame_id in sorted(set(actual) & set(golden)):
        actual_frame = actual[frame_id]
        golden_frame = golden[frame_id]
        actual_people = actual_frame["people"]
        golden_people = golden_frame["people"]
        people_count += len(actual_people)
        if len(actual_people) != len(golden_people):
            mismatches.append(
                {
                    "kind": "people_count",
                    "frame": frame_id,
                    "actual": len(actual_people),
                    "golden": len(golden_people),
                }
            )
            continue
        for person_index, (actual_person, golden_person) in enumerate(
            zip(actual_people, golden_people)
        ):
            scalar_fields = ("id", "role", "legibility_score", "jersey_number")
            if any(actual_person[key] != golden_person[key] for key in scalar_fields):
                mismatches.append(
                    {"kind": "person_scalar", "frame": frame_id, "person": person_index}
                )
                break
            if not np.array_equal(actual_person["bbox_ltwh"], golden_person["bbox_ltwh"]):
                mismatches.append(
                    {"kind": "person_bbox", "frame": frame_id, "person": person_index}
                )
                break

        if actual_frame["valid_cam_params"] != golden_frame["valid_cam_params"]:
            mismatches.append({"kind": "camera_valid", "frame": frame_id})
            continue
        if actual_frame["valid_cam_params"]:
            valid_camera_count += 1
            for field in ("K", "R", "P"):
                difference = np.max(
                    np.abs(np.asarray(actual_frame[field]) - np.asarray(golden_frame[field]))
                )
                camera_max_abs = max(camera_max_abs, float(difference))
                if not np.allclose(
                    actual_frame[field], golden_frame[field], atol=atol, rtol=rtol
                ):
                    mismatches.append(
                        {"kind": f"camera_{field}", "frame": frame_id, "max_abs": difference}
                    )
        elif any(actual_frame[field] is not None for field in ("K", "R", "P")):
            mismatches.append({"kind": "invalid_camera_payload", "frame": frame_id})

    return {
        "frame_count": len(actual),
        "people_count": people_count,
        "valid_camera_count": valid_camera_count,
        "camera_max_abs": camera_max_abs,
        "semantic_mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
    }


def main() -> int:
    started_wall = time.time()
    started_monotonic = time.monotonic()
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    converted_path = Path(manifest["local_outputs"]["converted_pkl"])
    result_path = Path(manifest["local_outputs"]["result_json"])
    require_local_output(converted_path)
    require_local_output(result_path)

    print("[PHASE] validate_assets", flush=True)
    asset_result = validate_assets(manifest)
    print("[PHASE] load_historical_step3", flush=True)
    detections, images = load_step3(manifest)
    print("[PHASE] convert", flush=True)
    converted = convert_step3(detections, images, manifest["sample"]["video_id"])
    print("[PHASE] compare_golden", flush=True)
    with Path(manifest["golden_training_pkl"]["path"]).open("rb") as handle:
        golden = pickle.load(handle)
    success = manifest["success"]
    comparison = compare_to_golden(
        converted,
        golden,
        atol=float(success["camera_atol"]),
        rtol=float(success["camera_rtol"]),
    )
    assertions = {
        "frame_count": comparison["frame_count"] == success["required_frames"],
        "people_count": comparison["people_count"] == success["required_people"],
        "valid_camera_count": (
            comparison["valid_camera_count"] == success["required_valid_camera_frames"]
        ),
        "semantic_mismatch_count": (
            comparison["semantic_mismatch_count"]
            == success["maximum_semantic_mismatches"]
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(
            f"G10-B compatibility assertions failed: {assertions}; {comparison}"
        )

    print("[PHASE] write_local_output", flush=True)
    atomic_pickle_dump(converted, converted_path)
    converted_sha256 = sha256_file(converted_path)
    golden_sha256 = manifest["golden_training_pkl"]["sha256"]
    ended_wall = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "schema_version": 1,
        "gate": "G10-B",
        "stage": manifest["stage"],
        "conclusion": "compatibility_conversion_passed_generation_not_run",
        "manifest": str(manifest_path),
        "git": git_identity(),
        "python": sys.executable,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input": {
            "sequence": manifest["sample"]["sequence"],
            "historical_step3": manifest["historical_step3"]["path"],
            "golden_training_pkl": manifest["golden_training_pkl"]["path"],
        },
        "asset_validation": asset_result,
        "comparison": comparison,
        "assertions": assertions,
        "output": {
            "path": str(converted_path),
            "bytes": converted_path.stat().st_size,
            "sha256": converted_sha256,
            "golden_sha256": golden_sha256,
            "byte_identical_to_golden": converted_sha256 == golden_sha256,
        },
        "timing": {
            "started_unix": started_wall,
            "ended_unix": ended_wall,
            "wall_seconds": round(time.monotonic() - started_monotonic, 3),
            "max_rss_kib": usage.ru_maxrss,
        },
        "gpu_used": False,
        "inference_run": False,
        "training_run": False,
        "known_gap": manifest["known_gap"],
        "not_validated": [
            "raw-video-to-prepared-frame extraction implementation",
            "new TrackLab Step-1 output",
            "new Refiner output",
            "new TrackLab Step-3 output",
            "independent-run reproducibility of model-generated labels",
            "SoccerMaster dataset consumer contract (reserved for G10-C)"
        ],
    }
    print("[PHASE] write_result", flush=True)
    atomic_json_dump(result, result_path)
    print(
        "[RESULT] compatibility_conversion_passed "
        f"frames={comparison['frame_count']} people={comparison['people_count']} "
        f"mismatches={comparison['semantic_mismatch_count']} "
        f"byte_identical={result['output']['byte_identical_to_golden']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
