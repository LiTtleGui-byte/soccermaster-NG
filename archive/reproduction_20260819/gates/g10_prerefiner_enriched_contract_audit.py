#!/usr/bin/env python3
"""Read-only contract and quality audit of the enriched run2 archive."""

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
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_prerefiner_enriched_contract_audit_run2.json"


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


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def scalar_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        key = "<missing>" if is_missing_scalar(value) else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "prerefiner_enriched_contract_audit_run2":
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

        reference_paths = {name: validate_file(spec) for name, spec in manifest["references"].items()}
        source_paths = {name: validate_file(spec) for name, spec in manifest["refiner_sources"].items()}
        archive_path = validate_file(manifest["input_archive"])
        metadata_path = validate_file(manifest["metadata"])
        log("assets", "all fixed paths, sizes, and SHA256 values passed")

        run2_result = json.loads(reference_paths["run2_result"].read_text(encoding="utf-8"))
        if not (
            run2_result.get("outcome") == "passed"
            and run2_result.get("assertions_passed") is True
            and run2_result.get("process_exit_code") == 0
            and run2_result.get("timed_out") is False
            and run2_result.get("fallbacks_used") == []
            and run2_result.get("state_archive", {}).get("sha256") == manifest["input_archive"]["sha256"]
            and run2_result.get("state_archive", {}).get("bytes") == manifest["input_archive"]["bytes"]
        ):
            raise AssertionError("Pinned enrichment run2 result contract changed")
        log("lineage", "run2 passed result and output archive identity are pinned")

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
            "ROLE_MAP = {'player': 0, 'goalkeeper': 1, 'referee': 2, 'unknown': 3}",
            "TEAM_MAP = {'left': 0, 'right': 1, 'nan': 2}",
            "frame_preds['embeddings']",
            "frame_preds['bbox_ltwh']",
            "frame_preds['bbox_pitch']",
            "frame_preds['role']",
            "frame_preds['team']",
            "frame_preds['jersey_number']",
            "frame_preds['track_id']",
            "frame_image_preds['parameters']",
            "assert len(image_ids) == max_frames",
            "num_det = min(len(frame_preds), max_detections)",
            "x_focal_length = camera_params['x_focal_length']",
            "principal_point = np.array(camera_params['principal_point'])",
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
        if not (bbox_values[:, 2:] > 0).all():
            raise AssertionError("bbox width/height values must be positive")
        log("types", f"bbox_shapes={bbox_shapes}, embedding_shapes={embedding_shapes}, max_per_frame={max_per_frame}")

        allowed_roles = set(contract["allowed_roles"])
        allowed_teams = set(contract["allowed_teams"])
        role_counts = scalar_counts(detections["role"])
        team_counts = scalar_counts(detections["team"])
        observed_roles = {str(value) for value in detections["role"] if not is_missing_scalar(value)}
        observed_teams = {str(value) for value in detections["team"] if not is_missing_scalar(value)}
        if "<missing>" in role_counts or not observed_roles.issubset(allowed_roles):
            raise AssertionError(f"Role values violate Refiner mapping: {role_counts}")
        if "<missing>" in team_counts or not observed_teams.issubset(allowed_teams):
            raise AssertionError(f"Team values violate Refiner mapping: {team_counts}")

        jersey_numbers: list[int] = []
        for value in detections["jersey_number"]:
            if is_missing_scalar(value):
                continue
            numeric = float(value)
            if not np.isfinite(numeric) or numeric != np.floor(numeric) or not 0 < numeric < 100:
                raise AssertionError(f"Invalid non-null jersey number: {value}")
            jersey_numbers.append(int(numeric))
        jersey_missing = len(detections) - len(jersey_numbers)
        if len(jersey_numbers) != contract["expected_non_null"]["jersey_number"]:
            raise AssertionError("Jersey-number non-null count changed")

        pitch_x: list[float] = []
        pitch_y: list[float] = []
        pitch_key_sets: set[tuple[str, ...]] = set()
        required_pitch_keys = set(contract["bbox_pitch_required_keys"])
        for value in detections["bbox_pitch"]:
            if not isinstance(value, dict) or not required_pitch_keys.issubset(value):
                raise AssertionError("bbox_pitch must be a dictionary with Refiner coordinate keys")
            pitch_key_sets.add(tuple(sorted(str(key) for key in value)))
            x = float(value["x_bottom_middle"])
            y = float(value["y_bottom_middle"])
            if not np.isfinite([x, y]).all():
                raise AssertionError("bbox_pitch coordinates must be finite")
            pitch_x.append(x)
            pitch_y.append(y)
        x_min, x_max = (float(value) for value in contract["pitch_x_range"])
        y_min, y_max = (float(value) for value in contract["pitch_y_range"])
        pitch_would_clip = sum(
            not (x_min <= x <= x_max and y_min <= y <= y_max)
            for x, y in zip(pitch_x, pitch_y)
        )

        required_camera_keys = set(contract["camera_parameter_required_keys"])
        camera_key_sets: set[tuple[str, ...]] = set()
        rotation_determinants: list[float] = []
        focal_lengths: list[float] = []
        for value in images["parameters"]:
            if not isinstance(value, dict) or not required_camera_keys.issubset(value):
                raise AssertionError("Camera parameters lack keys unconditionally read by Refiner")
            camera_key_sets.add(tuple(sorted(str(key) for key in value)))
            principal = np.asarray(value["principal_point"], dtype=float)
            position = np.asarray(value["position_meters"], dtype=float)
            rotation = np.asarray(value["rotation_matrix"], dtype=float)
            if principal.shape != (2,) or position.shape != (3,) or rotation.shape != (3, 3):
                raise AssertionError("Camera parameter shapes violate Refiner contract")
            numeric = np.concatenate([principal, position, rotation.reshape(-1)])
            if not np.isfinite(numeric).all():
                raise AssertionError("Camera parameter arrays must be finite")
            x_focal = float(value["x_focal_length"])
            y_focal = float(value["y_focal_length"])
            if not np.isfinite([x_focal, y_focal]).all() or x_focal <= 0 or y_focal <= 0:
                raise AssertionError("Camera focal lengths must be finite and positive")
            focal_lengths.extend([x_focal, y_focal])
            rotation_determinants.append(float(np.linalg.det(rotation)))

        actual_non_null = {
            "bbox_pitch": sum(not is_missing_scalar(value) for value in detections["bbox_pitch"]),
            "role": sum(not is_missing_scalar(value) for value in detections["role"]),
            "team": sum(not is_missing_scalar(value) for value in detections["team"]),
            "jersey_number": len(jersey_numbers),
            "parameters": sum(not is_missing_scalar(value) for value in images["parameters"]),
        }
        if actual_non_null != contract["expected_non_null"]:
            raise AssertionError(f"Enriched non-null counts changed: {actual_non_null}")

        per_track = detections.assign(_frame=detections["image_id"].map(dict(zip(images["id"], images["frame"]))))
        if per_track["_frame"].isna().any():
            raise AssertionError("Track coverage contains an unresolved image foreign key")
        track_groups = per_track.groupby("track_id")["_frame"]
        track_rows = track_groups.size().astype(int)
        track_frame_counts = track_groups.nunique().astype(int)
        track_spans = (track_groups.max() - track_groups.min() + 1).astype(int)
        if len(track_rows) != unique_track_ids:
            raise AssertionError("Track coverage group count changed")
        log(
            "quality",
            f"roles={role_counts}; teams={team_counts}; jersey_non_null={len(jersey_numbers)}; "
            f"pitch_would_clip={pitch_would_clip}",
        )

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
        ready_with_255_override = (
            not missing_detection
            and not missing_image
            and len(images) == contract["required_frames"]
            and max_per_frame <= max_detections
            and len(rotation_determinants) == contract["required_frames"]
        )
        if ready_with_255_override != contract["expected_255_frame_static_contract_ready"]:
            raise AssertionError("255-frame static contract verdict differs from expectation")
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
            "refiner_255_frame_static_contract_ready": ready_with_255_override,
            "compatibility_verdict": "incompatible_default_max_frames_only",
            "started_utc": started_iso,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": ended - started,
            "process_exit_code": 0,
            "timed_out": False,
            "command": [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest.resolve())],
            "authorization": {
                "user_instruction": "Continue with the unique CPU-only read-only Refiner input contract audit.",
                "authorized_scope": "Read pinned source/config/metadata and all rows of the fixed enriched run2 archive; write only the isolated audit report.",
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
                "references": manifest["references"],
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
                    "per_frame_count_min": int(per_frame_counts.min()),
                    "per_frame_count_median": float(per_frame_counts.median()),
                    "per_frame_count_max": max_per_frame,
                    "bbox_width_min": float(bbox_values[:, 2].min()),
                    "bbox_height_min": float(bbox_values[:, 3].min()),
                    "role_counts": role_counts,
                    "team_counts": team_counts,
                    "jersey_number_value_types": value_type_counts(detections["jersey_number"]),
                    "jersey_number_non_null": len(jersey_numbers),
                    "jersey_number_missing": jersey_missing,
                    "jersey_number_unique_values": sorted(set(jersey_numbers)),
                    "bbox_pitch_value_types": value_type_counts(detections["bbox_pitch"]),
                    "bbox_pitch_key_sets": [list(keys) for keys in sorted(pitch_key_sets)],
                    "bbox_pitch_x_min": min(pitch_x),
                    "bbox_pitch_x_max": max(pitch_x),
                    "bbox_pitch_y_min": min(pitch_y),
                    "bbox_pitch_y_max": max(pitch_y),
                    "bbox_pitch_values_that_refiner_would_clip": pitch_would_clip,
                    "track_rows_min": int(track_rows.min()),
                    "track_rows_median": float(track_rows.median()),
                    "track_rows_max": int(track_rows.max()),
                    "track_frame_count_min": int(track_frame_counts.min()),
                    "track_frame_count_median": float(track_frame_counts.median()),
                    "track_frame_count_max": int(track_frame_counts.max()),
                    "track_span_min": int(track_spans.min()),
                    "track_span_median": float(track_spans.median()),
                    "track_span_max": int(track_spans.max()),
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
                    "parameters_value_types": value_type_counts(images["parameters"]),
                    "parameters_key_sets": [list(keys) for keys in sorted(camera_key_sets)],
                    "camera_valid_rows": len(rotation_determinants),
                    "focal_length_min": min(focal_lengths),
                    "focal_length_max": max(focal_lengths),
                    "rotation_determinant_min": min(rotation_determinants),
                    "rotation_determinant_max": max(rotation_determinants),
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
                "columns_ready": not missing_detection and not missing_image,
                "ready_with_max_frames_255_override": ready_with_255_override,
                "default_incompatibility_reason": "configured_max_frames_750_vs_actual_255",
            },
            "quality_scope": {
                "ground_truth_used": False,
                "semantic_accuracy_measured": False,
                "distribution_and_source_contract_only": True,
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
                "The enriched run2 archive structure, rows, indices, core values, pitch dictionaries, semantic labels, jersey values, camera parameters, and track coverage were read and checked.",
                "All columns unconditionally read by the pinned Refiner dataset preprocessing source are present.",
                "The archive is statically consumable with a 255-frame override, while the preserved default configuration still asserts 750 frames.",
            ],
            "unknown": [
                "Semantic accuracy of pitch, role, team, jersey, and camera outputs because this audit uses no ground truth.",
                "Whether an isolated 255-frame Refiner configuration can load its model and complete forward correctly.",
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
