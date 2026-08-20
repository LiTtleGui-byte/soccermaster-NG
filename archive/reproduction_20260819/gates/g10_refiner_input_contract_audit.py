#!/usr/bin/env python3
"""Read-only audit of a TrackLab state archive against Refiner input code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_refiner_input_contract_audit_run5.json"


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


def validate_file(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != spec["bytes"]:
        raise AssertionError(f"Size changed: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"SHA256 changed: {path}")
    return path


def require_local_new_output(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Output outside workspace: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def index_contract(index: pd.Index) -> dict[str, Any]:
    return {
        "class": type(index).__name__,
        "dtype": str(index.dtype),
        "name": None if index.name is None else str(index.name),
        "nlevels": index.nlevels,
        "unique": bool(index.is_unique),
        "monotonic_increasing": bool(index.is_monotonic_increasing),
    }


def value_type_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        name = type(value).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "refiner_input_contract_audit_run5":
        raise AssertionError("Unexpected manifest identity")

    outputs = manifest["outputs"]
    report_dir = Path(outputs["report_dir"])
    result_path = Path(outputs["result"])
    log_path = Path(outputs["log"])
    require_local_new_output(report_dir)
    if result_path.parent != report_dir or log_path.parent != report_dir:
        raise AssertionError("Output files must be inside the isolated report directory")
    report_dir.mkdir(parents=True, exist_ok=False)

    log_lines: list[str] = []

    def log(phase: str, message: str) -> None:
        line = f"[{phase}] {message}"
        log_lines.append(line)
        print(line, flush=True)

    try:
        execution = manifest["execution"]
        actual_env = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        }
        if actual_env["CUDA_VISIBLE_DEVICES"] != execution["cuda_visible_devices"]:
            raise AssertionError("CUDA_VISIBLE_DEVICES must be empty")
        if actual_env["PYTHONPATH"] != execution["pythonpath"]:
            raise AssertionError("PYTHONPATH must be empty")
        if actual_env["LD_LIBRARY_PATH"] != execution["ld_library_path"]:
            raise AssertionError("LD_LIBRARY_PATH must be empty")
        if actual_env["PYTHONDONTWRITEBYTECODE"] != "1":
            raise AssertionError("PYTHONDONTWRITEBYTECODE must be 1")
        log("environment", "CPU-only environment guard passed")

        source_paths = {name: validate_file(spec) for name, spec in manifest["refiner_sources"].items()}
        archive_path = validate_file(manifest["input_archive"])
        metadata_path = validate_file(manifest["metadata"])
        log("assets", "all fixed paths, sizes, and SHA256 values passed")

        inference_text = source_paths["inference"].read_text(encoding="utf-8")
        dataset_utils_text = source_paths["dataset_utils"].read_text(encoding="utf-8")
        required_source_fragments = [
            "input_zf.open(f'{vid}.pkl')",
            "input_zf.open(f'{vid}_image.pkl')",
            "process_pipeline_video(",
            "if os.path.exists(output_pklz_path):",
            "os.remove(output_pklz_path)",
            "temp_pred_path = f'temp_{vid}.pkl'",
            "temp_image_path = f'temp_{vid}_image.pkl'",
        ]
        missing_inference_fragments = [item for item in required_source_fragments if item not in inference_text]
        required_dataset_fragments = [
            "frame_preds['embeddings']",
            "frame_preds['bbox_ltwh']",
            "frame_preds['bbox_pitch']",
            "frame_preds['role']",
            "frame_preds['team']",
            "frame_preds['jersey_number']",
            "frame_preds['track_id']",
            "frame_image_preds['parameters']",
            "assert len(image_ids) == max_frames",
        ]
        missing_dataset_fragments = [item for item in required_dataset_fragments if item not in dataset_utils_text]
        if missing_inference_fragments or missing_dataset_fragments:
            raise AssertionError("Pinned Refiner source contract fragments changed")
        log("source_contract", "pinned reads and write-side effects found in preserved Refiner source")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_spec = manifest["metadata"]
        if metadata.get(metadata_spec["split"]) != [metadata_spec["record"]]:
            raise AssertionError("Metadata split or record changed")
        video_id = manifest["contract"]["video_id"]
        derived_video_ids = [record["name"].split("-")[1] for record in metadata[metadata_spec["split"]]]
        if derived_video_ids != [video_id]:
            raise AssertionError("Metadata video ID derivation changed")
        log("metadata", f"split={metadata_spec['split']} resolves exactly video_id={video_id}")

        base_config = yaml.safe_load(source_paths["base_config"].read_text(encoding="utf-8"))["base_config"]
        inference_config = yaml.safe_load(source_paths["inference_config"].read_text(encoding="utf-8"))
        max_frames = int(inference_config.get("data", {}).get("max_frames", base_config["data"]["max_frames"]))
        max_detections = int(inference_config.get("data", {}).get(
            "max_detections_per_frame", base_config["data"]["max_detections_per_frame"]
        ))
        if max_frames != manifest["contract"]["default_max_frames"]:
            raise AssertionError("Pinned Refiner max_frames changed")
        if max_detections != manifest["contract"]["default_max_detections_per_frame"]:
            raise AssertionError("Pinned Refiner max detections changed")
        log("config", f"resolved max_frames={max_frames}, max_detections_per_frame={max_detections}")

        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            if sorted(members) != sorted(manifest["input_archive"]["members_exact"]):
                raise AssertionError(f"Unexpected ZIP members: {members}")
            if len(members) != len(set(members)) or archive.testzip() is not None:
                raise AssertionError("ZIP duplicate member or CRC failure")
            summary = json.loads(archive.read("summary.json"))
            with archive.open(f"{video_id}.pkl") as handle:
                detections = pd.read_pickle(handle)
            with archive.open(f"{video_id}_image.pkl") as handle:
                images = pd.read_pickle(handle)

        if not isinstance(detections, pd.DataFrame) or not isinstance(images, pd.DataFrame):
            raise TypeError("State members must be pandas DataFrames")
        contract = manifest["contract"]
        if len(detections) != contract["required_detection_rows"]:
            raise AssertionError("Detection row count changed")
        if len(images) != contract["required_frames"]:
            raise AssertionError("Image row count changed")
        if not detections.index.is_unique or not images.index.is_unique:
            raise AssertionError("DataFrame indices must be unique")
        if summary.get("columns", {}).get("detection") != detections.columns.tolist():
            raise AssertionError("Detection summary columns differ from DataFrame")
        if summary.get("columns", {}).get("image") != images.columns.tolist():
            raise AssertionError("Image summary columns differ from DataFrame")
        log("archive", f"loaded {len(images)} image rows and {len(detections)} detection rows")

        image_ids = sorted(images["id"].unique().tolist(), key=lambda value: int(value))
        if len(image_ids) != contract["required_frames"]:
            raise AssertionError("Image IDs are not unique per frame")
        if images["frame"].astype(int).tolist() != list(range(contract["required_frames"])):
            raise AssertionError("Image frame order is not exactly 0..254")
        expected_names = [f"{index:06d}.jpg" for index in range(1, contract["required_frames"] + 1)]
        actual_names = [Path(value).name for value in images["file_path"].astype(str)]
        if actual_names != expected_names:
            raise AssertionError("Image paths are not exactly 000001.jpg..000255.jpg")
        if not set(detections["image_id"]).issubset(set(image_ids)):
            raise AssertionError("Detection image foreign key mismatch")
        per_frame_counts = detections.groupby("image_id").size()
        max_per_frame = int(per_frame_counts.max())
        empty_frame_count = int(contract["required_frames"] - per_frame_counts.index.nunique())
        if max_per_frame > max_detections:
            raise AssertionError("Refiner would silently truncate detections above its configured maximum")

        if detections["track_id"].isna().any():
            raise AssertionError("track_id contains null")
        track_values = detections["track_id"].astype(float).to_numpy()
        if not np.isfinite(track_values).all() or not np.equal(track_values, np.floor(track_values)).all():
            raise AssertionError("track_id values are not finite integers")
        if not ((track_values > 0) & (track_values < 150)).all():
            raise AssertionError("track_id values are outside Refiner's accepted 1..149 range")
        unique_track_ids = int(np.unique(track_values).size)
        if unique_track_ids != contract["required_unique_track_ids"]:
            raise AssertionError("Unique track ID count changed")

        bbox_shapes = sorted({tuple(np.asarray(value).shape) for value in detections["bbox_ltwh"]})
        if bbox_shapes != [(4,)]:
            raise AssertionError(f"Unexpected bbox shapes: {bbox_shapes}")
        bbox_values = np.stack(detections["bbox_ltwh"].values)
        if not np.issubdtype(bbox_values.dtype, np.number) or not np.isfinite(bbox_values).all():
            raise AssertionError("bbox values must be finite numeric values")
        embedding_shapes = sorted({tuple(np.asarray(value).shape) for value in detections["embeddings"]})
        if embedding_shapes != [tuple(contract["embedding_shape"])]:
            raise AssertionError(f"Unexpected embedding shapes: {embedding_shapes}")
        embedding_values = np.stack(detections["embeddings"].values)
        if not np.issubdtype(embedding_values.dtype, np.number) or not np.isfinite(embedding_values).all():
            raise AssertionError("Embedding values must be finite numeric values")
        log("types", f"bbox_shapes={bbox_shapes}, embedding_shapes={embedding_shapes}, max_per_frame={max_per_frame}")

        missing_detection = sorted(set(contract["refiner_detection_columns"]) - set(detections.columns))
        missing_image = sorted(set(contract["refiner_image_columns"]) - set(images.columns))
        if missing_detection != sorted(contract["expected_missing_detection_columns"]):
            raise AssertionError(f"Unexpected missing detection columns: {missing_detection}")
        if missing_image != sorted(contract["expected_missing_image_columns"]):
            raise AssertionError(f"Unexpected missing image columns: {missing_image}")
        frame_count_matches_config = len(images) == max_frames
        compatible = not missing_detection and not missing_image and frame_count_matches_config
        if compatible != contract["expected_compatibility"]:
            raise AssertionError("Compatibility verdict differs from pinned expectation")
        log("compatibility", f"compatible={compatible}; missing_detection={missing_detection}; missing_image={missing_image}")

        ended = time.time()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = {
            "schema_version": 1,
            "gate": manifest["gate"],
            "stage": manifest["stage"],
            "audit_outcome": "passed",
            "audit_assertions_passed": True,
            "refiner_input_compatible": compatible,
            "compatibility_verdict": "incompatible_missing_prerefiner_fields_and_frame_config_mismatch",
            "started_utc": started_iso,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": ended - started,
            "process_exit_code": 0,
            "timed_out": False,
            "command": [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest.resolve())],
            "authorization": {
                "user_instruction": "Continue with the unique CPU-only read-only Refiner input contract audit.",
                "authorized_scope": "Read pinned source/config/metadata and all rows of the fixed run5 archive; write only the isolated audit report.",
                "explicit_non_goals": manifest["forbidden"],
            },
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyyaml": yaml.__version__,
            },
            "environment": actual_env,
            "git": git_identity(),
            "inputs": {
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": sha256_file(args.manifest),
                "archive": manifest["input_archive"],
                "metadata": manifest["metadata"],
                "refiner_source_revision": manifest["refiner_source_revision"],
                "refiner_sources": manifest["refiner_sources"],
            },
            "archive": {
                "members": members,
                "summary": summary,
                "detections": {
                    "rows": len(detections),
                    "columns": detections.columns.tolist(),
                    "index": index_contract(detections.index),
                    "column_dtypes": {name: str(dtype) for name, dtype in detections.dtypes.items()},
                    "bbox_value_types": value_type_counts(detections["bbox_ltwh"]),
                    "bbox_shapes": [list(shape) for shape in bbox_shapes],
                    "embedding_value_types": value_type_counts(detections["embeddings"]),
                    "embedding_shapes": [list(shape) for shape in embedding_shapes],
                    "track_id_value_types": value_type_counts(detections["track_id"]),
                    "unique_track_ids": unique_track_ids,
                    "maximum_detections_per_frame": max_per_frame,
                    "empty_frame_count": empty_frame_count,
                },
                "images": {
                    "rows": len(images),
                    "columns": images.columns.tolist(),
                    "index": index_contract(images.index),
                    "column_dtypes": {name: str(dtype) for name, dtype in images.dtypes.items()},
                    "id_value_types": value_type_counts(images["id"]),
                    "frame_value_types": value_type_counts(images["frame"]),
                    "video_id_value_types": value_type_counts(images["video_id"]),
                    "file_path_value_types": value_type_counts(images["file_path"]),
                    "frame_min": int(images["frame"].min()),
                    "frame_max": int(images["frame"].max()),
                },
            },
            "refiner_contract": {
                "required_detection_columns": contract["refiner_detection_columns"],
                "required_image_columns": contract["refiner_image_columns"],
                "missing_detection_columns": missing_detection,
                "missing_image_columns": missing_image,
                "metadata_video_ids": derived_video_ids,
                "configured_max_frames": max_frames,
                "actual_frames": len(images),
                "frame_count_matches_config": frame_count_matches_config,
                "configured_max_detections_per_frame": max_detections,
                "actual_maximum_detections_per_frame": max_per_frame,
                "would_truncate_detections": max_per_frame > max_detections,
            },
            "source_side_effects_if_executed": {
                "creates_output_directory": True,
                "deletes_existing_output_archive": True,
                "writes_temporary_pickles_in_current_working_directory": True,
                "loads_model_and_runs_inference": True,
                "audit_executed_any_of_these": False,
            },
            "resources": {
                "peak_cpu_rss_kib": usage.ru_maxrss,
                "gpu_used": False,
                "gpu_peak_memory_bytes": 0,
            },
            "execution_contract": manifest["execution"],
            "fallbacks_used": [],
            "forbidden_actions_executed": [],
            "confirmed": [
                "The run5 archive structure, rows, indices, scalar types, bbox values, embedding values, and track IDs were read and checked.",
                "The pinned Refiner source unconditionally reads the reported missing columns before model inference.",
                "The preserved default inference configuration expects 750 frames while this fixed archive has 255 frames.",
            ],
            "unknown": [
                "Which missing upstream stage populated pitch, semantic, jersey, and camera fields in the historical Refiner input.",
                "Whether a future local 255-frame Refiner configuration and completed pre-Refiner archive can run correctly.",
            ],
            "not_run": ["Refiner import", "Torch import", "model load", "forward", "GPU", "evaluation", "training", "Step 3", "conversion"],
        }
        atomic_json_dump(result, result_path)
        log("result", f"wrote {result_path}")
        log("exit", "audit_exit_code=0")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        log("error", f"{type(error).__name__}: {error}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
