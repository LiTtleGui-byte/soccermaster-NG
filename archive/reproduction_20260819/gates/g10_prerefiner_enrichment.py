#!/usr/bin/env python3
"""Static preflight and guarded launcher for G10-B pre-Refiner enrichment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import resource
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_prerefiner_enrichment_run1_sngs10004.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Local path escapes repository: {path}")
    return resolved


def require_absent(path: Path, label: str) -> None:
    inside_repo(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} is already used: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_new(text: str, path: Path) -> None:
    inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    identity = (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage"))
    if identity != (1, "G10-B", "prerefiner_enrichment_run1_prepared"):
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    return manifest


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


def validate_environment(mode: str) -> dict[str, str | None]:
    actual = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }
    common_expected = {
        "PYTHONPATH": "", "LD_LIBRARY_PATH": "", "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, expected in common_expected.items():
        if actual[key] != expected:
            raise AssertionError(f"Environment changed for {key}: {actual[key]}")
    if mode == "preflight":
        if actual["CUDA_VISIBLE_DEVICES"] != "":
            raise AssertionError(f"CPU-only preflight exposed CUDA: {actual['CUDA_VISIBLE_DEVICES']}")
    elif mode == "run":
        if re.fullmatch(r"[0-9]+", actual["CUDA_VISIBLE_DEVICES"] or "") is None:
            raise AssertionError("Run mode requires exactly one numeric CUDA_VISIBLE_DEVICES")
    else:
        raise AssertionError(f"Unexpected mode: {mode}")
    return actual


def validate_references(manifest: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for name, spec in manifest["references"].items():
        path = Path(spec["path"])
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"Pinned reference changed: {path}")
        records[name] = {"path": str(path), "sha256": spec["sha256"]}
    lineage_result = json.loads(Path(manifest["references"]["lineage_result"]["path"]).read_text())
    if lineage_result.get("status") != "passed" or lineage_result.get("verdict") != "historical_prerefiner_lineage_statically_traced":
        raise AssertionError("Pinned lineage result is not passed")
    return records


def validate_config(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["config"]["path"])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_defaults = [
        {"dataset": "soccernet_gs"}, {"eval": "gs_hota"}, {"engine": "offline"},
        {"state": "save"}, {"visualization": "gamestate"},
        {"modules/pitch": "nbjw_calib"},
        {"modules/calibration": "nbjw_calib_decouped"},
        {"modules/apply_camera_params": "nbjw_calib_apply_params"},
        {"modules/legibility": "legibility"},
        {"modules/jersey_number_detect": "qwen2_5vl_ocr_batch"},
        {"modules/tracklet_agg": "voting_role_jn_filter2"},
        {"modules/team": "kmeans_embeddings"},
        {"modules/team_side": "mean_position"}, "_self_",
    ]
    if config["defaults"] != expected_defaults:
        raise AssertionError("Hydra defaults changed")
    if config["pipeline"] != manifest["config"]["pipeline"]:
        raise AssertionError("Enrichment pipeline changed")
    sample = manifest["sample"]
    if config["dataset"]["nframes"] != sample["frame_count"]:
        raise AssertionError("Frame count changed")
    if config["dataset"]["eval_set"] != sample["split"]:
        raise AssertionError("Dataset split changed")
    if config["dataset"]["vids_dict"] != {sample["split"]: [sample["sequence"]]}:
        raise AssertionError("Sequence selection changed")
    if config["test_tracking"] is not True or config["eval_tracking"] is not False:
        raise AssertionError("Tracking/evaluation flags changed")
    if config["visualization"] is not None or config["use_wandb"] is not False:
        raise AssertionError("Visualization/W&B must remain disabled")
    if config["num_cores"] != manifest["adapters"]["dataset_pool_workers"]:
        raise AssertionError("Dataset worker cap changed")
    expected_modules = {
        "pitch": {
            "batch_size": 1,
            "checkpoint_kp": manifest["modules"]["pitch"]["weight_reads"][0]["path"],
            "checkpoint_l": manifest["modules"]["pitch"]["weight_reads"][1]["path"],
        },
        "calibration": {"batch_size": 1},
        "apply_camera_params": {
            "batch_size": 1, "use_h": False, "use_linalg": False,
            "use_prev_homography": True,
        },
        "legibility": {
            "batch_size": 16,
            "cfg": {"legibility_model_path": manifest["modules"]["legibility"]["weight_reads"][0]["path"]},
        },
        "jersey_number_detect": {
            "batch_size": 64,
            "cfg": {
                "model_path": manifest["qwen_assets"]["qwen2_5vl_7b"]["configured_path"],
                "save_jersey_number_full_detection": True,
                "use_legibility_filter": True,
                "legibility_filter_threshold": 0.5,
            },
        },
    }
    if config["modules"] != expected_modules:
        raise AssertionError("Module overrides changed")
    input_path = Path(manifest["input_state"]["path"])
    if Path(config["state"]["load_file"]) != input_path:
        raise AssertionError("Input state path changed")
    run_dir = Path(manifest["outputs"]["hydra_run_dir"])
    output_path = Path(manifest["outputs"]["state_archive"])
    if Path(config["hydra"]["run"]["dir"]) != run_dir:
        raise AssertionError("Hydra run directory changed")
    if output_path != run_dir / config["state"]["save_file"]:
        raise AssertionError("Output state resolution changed")
    if input_path.resolve() == output_path.resolve(strict=False):
        raise AssertionError("Input and output archives must differ")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pipeline": config["pipeline"],
        "module_overrides": config["modules"],
        "input_state": str(input_path),
        "output_state": str(output_path),
        "eval_tracking": False,
        "visualization": None,
        "use_wandb": False,
    }


def validate_frames(sample: dict[str, Any]) -> dict[str, Any]:
    root = Path(sample["frame_root"])
    names = sorted(path.name for path in root.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, sample["frame_count"] + 1)]
    if names != expected:
        raise AssertionError("Frame names/count changed")
    total = sum((root / name).stat().st_size for name in names)
    if total != sample["frame_total_bytes"]:
        raise AssertionError("Frame byte total changed")
    for key in ("first_frame", "last_frame"):
        spec = sample[key]
        if (root / spec["name"]).stat().st_size != spec["bytes"]:
            raise AssertionError(f"{key} size changed")
    return {"root": str(root), "count": len(names), "total_bytes": total, "contents_read": False}


def validate_input_state(manifest: dict[str, Any]) -> dict[str, Any]:
    spec = manifest["input_state"]
    path = Path(spec["path"])
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise AssertionError("Input state size changed")
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise AssertionError("Input state SHA256 changed")
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
        if sorted(members) != sorted(spec["members_exact"]) or archive.testzip() is not None:
            raise AssertionError("Input state ZIP member/CRC contract failed")
        summary = json.loads(archive.read("summary.json"))
    detection_columns = summary["columns"]["detection"]
    image_columns = summary["columns"]["image"]
    if not set(spec["required_detection_columns"]).issubset(detection_columns):
        raise AssertionError("Input state lacks required detection columns")
    if not set(spec["required_image_columns"]).issubset(image_columns):
        raise AssertionError("Input state lacks required image columns")
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": digest,
        "members": members, "detection_columns": detection_columns,
        "image_columns": image_columns, "pickle_members_deserialized": False,
    }


def validate_column_chain(manifest: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    detection = set(state["detection_columns"])
    image = set(state["image_columns"])
    records = []
    for stage in manifest["column_contract"]["stages"]:
        missing_detection = sorted(set(stage["requires_detection"]) - detection)
        missing_image = sorted(set(stage["requires_image"]) - image)
        if missing_detection or missing_image:
            raise AssertionError(
                f"Column chain fails at {stage['module']}: detection={missing_detection}, image={missing_image}"
            )
        detection.update(stage["produces_detection"])
        image.update(stage["produces_image"])
        records.append({
            "module": stage["module"],
            "required_detection_available": True,
            "required_image_available": True,
            "produces_detection": stage["produces_detection"],
            "produces_image": stage["produces_image"],
        })
    required_detection = set(manifest["column_contract"]["refiner_required_detection_columns"])
    required_image = set(manifest["column_contract"]["refiner_required_image_columns"])
    if not required_detection.issubset(detection) or not required_image.issubset(image):
        raise AssertionError("Final static Refiner column contract failed")
    return {
        "stages": records,
        "final_detection_columns": sorted(detection),
        "final_image_columns": sorted(image),
        "refiner_required_columns_statically_satisfied": True,
    }


def validate_sources_and_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    sources = {}
    for name, raw_path in manifest["source_files"].items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        sources[name] = {"path": str(path), "sha256": sha256_file(path), "python_syntax": "valid"}

    lineage_result = json.loads(Path(manifest["references"]["lineage_result"]["path"]).read_text())
    targets = lineage_result["module_targets"]
    assets = {}
    for component, module in manifest["modules"].items():
        if component != "evaluator" and targets.get(component) != module["target"]:
            raise AssertionError(f"Module target differs from traced lineage: {component}")
        reads = []
        for spec in module["weight_reads"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Weight stat changed: {path}")
            reads.append({"path": str(path), "bytes": path.stat().st_size, "contents_read": False})
        assets[component] = {"target": module["target"], "weight_reads": reads}

    qwen_records = {}
    for name, spec in manifest["qwen_assets"].items():
        configured = Path(spec["configured_path"])
        if not configured.is_symlink() or os.readlink(configured) != spec["link_target"]:
            raise AssertionError("Qwen symlink changed")
        root = configured.resolve(strict=True)
        shard_total = 0
        shards = []
        for shard in spec["shards"]:
            path = root / shard["name"]
            if not path.is_file() or path.stat().st_size != shard["bytes"]:
                raise AssertionError(f"Qwen shard changed: {path}")
            shard_total += path.stat().st_size
            shards.append({"path": str(path), "bytes": path.stat().st_size, "contents_read": False})
        if shard_total != spec["logical_shard_bytes"]:
            raise AssertionError("Qwen shard total changed")
        qwen_records[name] = {
            "configured_path": str(configured), "resolved_path": str(root),
            "logical_shard_bytes": shard_total, "shards": shards,
        }

    lineage_manifest = json.loads(Path(manifest["references"]["lineage_manifest"]["path"]).read_text())
    legibility_source_spec = lineage_manifest["small_sources"]["legibility_source"]
    legibility_source = Path(legibility_source_spec["path"])
    if sha256_file(legibility_source) != legibility_source_spec["sha256"]:
        raise AssertionError("Pinned legibility source changed")
    text = legibility_source.read_text(encoding="utf-8")
    for fragment in ("models.resnet34(pretrained=True)", "self.model.load_state_dict(state_dict)"):
        if fragment not in text:
            raise AssertionError("Legibility no-download semantic guard source changed")
    adapter = manifest["adapters"]["legibility_resnet34_no_download"]
    if not adapter["enabled"] or adapter["scope"] != "instantiate_legibility_only":
        raise AssertionError("Legibility no-download adapter changed")
    torchvision_cache = Path("/home/tianlin/.cache/torch/hub/checkpoints/resnet34-b627a593.pth")
    return {
        "sources": sources,
        "declared_module_reads": assets,
        "qwen_assets": qwen_records,
        "legibility_no_download_guard": {
            "source": str(legibility_source),
            "source_sha256": legibility_source_spec["sha256"],
            "strict_load_state_dict_call_present": True,
            "torchvision_pretrained_call_present": True,
            "torchvision_cache_present_at_preflight": torchvision_cache.is_file(),
            "adapter": adapter,
        },
    }


def validate_runtime_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    expected_phases = [
        "import_tracklab_main", "hydra_cli_compose", "init_environment",
        "instantiate_dataset", "instantiate_evaluator", "instantiate_pitch",
        "instantiate_calibration", "instantiate_apply_camera_params",
        "instantiate_legibility", "instantiate_jersey_number_detect",
        "instantiate_tracklet_agg", "instantiate_team", "instantiate_team_side",
        "build_pipeline", "build_tracker_state", "instantiate_engine",
        "track_dataset", "evaluation_skipped",
    ]
    if manifest["phases"] != expected_phases:
        raise AssertionError("Phase order changed")
    for phase in expected_phases:
        if phase not in manifest["timeouts_seconds"]:
            raise AssertionError(f"Missing phase timeout: {phase}")
    if manifest["heartbeat_seconds"] != 30:
        raise AssertionError("Heartbeat interval changed")
    approval = manifest["approval"]
    if approval["required_environment"] != "G10_PREREFINER_ENRICHMENT_RUN1_GPU_APPROVED=YES":
        raise AssertionError("Approval guard changed")
    expected_cuda = {
        "cuda_available": True, "visible_device_count": 1, "logical_device_index": 0,
        "expected_device_name": "NVIDIA H800", "minimum_total_memory_bytes": 80000000000,
        "minimum_free_memory_bytes": 70000000000,
    }
    if manifest["cuda_guard"] != expected_cuda:
        raise AssertionError("CUDA guard changed")
    adapters = manifest["adapters"]
    if adapters["dataset_pool_workers"] != 4:
        raise AssertionError("Dataset worker cap changed")
    if adapters["suppressed_path_exact"] != "/home/tianlin/.cache/mim":
        raise AssertionError("MIM suppression scope changed")
    inside_repo(Path(adapters["local_cache_dir"]))
    expected_cache_env = ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "MPLCONFIGDIR"]
    if adapters["cache_environment"] != expected_cache_env:
        raise AssertionError("Local cache environment changed")
    if adapters["qwen_single_device_assertion"] != {
        "enabled": True, "expected_parameter_devices": ["cuda:0"],
        "scope": "post_instantiate_jersey_number_detect",
    }:
        raise AssertionError("Qwen single-device assertion changed")
    expected_null = {
        "enabled": True, "copy_before_filter": True,
        "deletion_mode": "omegaconf_open_dict", "require_struct_before": True,
        "require_struct_restored": True,
        "expected_original_keys": ["ignored_regions", "progress", "vis"],
        "expected_null_keys": ["vis"],
        "expected_remaining_keys": ["ignored_regions", "progress"],
        "scope": "instantiate_engine_only",
    }
    if adapters["engine_null_callback_filter"] != expected_null:
        raise AssertionError("Engine callback adapter changed")
    outputs = manifest["outputs"]
    for key in ("hydra_run_dir", "report_dir"):
        require_absent(Path(outputs[key]), f"future {key}")
    require_absent(Path(adapters["local_cache_dir"]), "future local cache")
    if Path(outputs["state_archive"]) != Path(outputs["hydra_run_dir"]) / "states/sn-gamestate.pklz":
        raise AssertionError("Output state contract changed")
    return {
        "phases": expected_phases,
        "timeouts_seconds": manifest["timeouts_seconds"],
        "heartbeat_seconds": manifest["heartbeat_seconds"],
        "approval": approval,
        "cuda_guard": expected_cuda,
        "adapters": adapters,
        "future_outputs_unused": True,
    }


def static_preflight(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise AssertionError(f"Wrong Python: {sys.executable}")
    environment = validate_environment(mode)
    references = validate_references(manifest)
    config = validate_config(manifest)
    frames = validate_frames(manifest["sample"])
    state = validate_input_state(manifest)
    columns = validate_column_chain(manifest, state)
    sources_assets = validate_sources_and_assets(manifest)
    runtime = validate_runtime_contract(manifest)
    return {
        "python": manifest["python"],
        "environment": environment,
        "references": references,
        "config": config,
        "sample": frames,
        "input_state": state,
        "column_chain": columns,
        **sources_assets,
        "future_runtime_contract": runtime,
        "tracklab_or_model_imports": False,
        "torch_imported": "torch" in sys.modules,
        "weight_contents_read": False,
        "gpu_operations": [],
    }


def require_columns(frame: Any, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing {label} columns: {missing}")


def validate_output_state(manifest: dict[str, Any]) -> dict[str, Any]:
    input_spec = manifest["input_state"]
    input_path = Path(input_spec["path"])
    if sha256_file(input_path) != manifest["success"]["input_archive_sha256_must_remain"]:
        raise AssertionError("Immutable input archive changed during run")
    path = Path(manifest["outputs"]["state_archive"])
    success = manifest["success"]
    video_id = manifest["sample"]["video_id"]
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
        if sorted(members) != sorted(success["zip_members_exact"]) or archive.testzip() is not None:
            raise AssertionError("Output state ZIP member/CRC contract failed")
        summary = json.loads(archive.read("summary.json"))
        with archive.open(f"{video_id}.pkl") as handle:
            detections = pickle.load(handle)
        with archive.open(f"{video_id}_image.pkl") as handle:
            images = pickle.load(handle)
    require_columns(detections, success["required_detection_columns"], "detection")
    require_columns(images, success["required_image_columns"], "image")
    if summary["columns"]["detection"] != detections.columns.tolist():
        raise AssertionError("Output detection summary differs from pickle")
    if summary["columns"]["image"] != images.columns.tolist():
        raise AssertionError("Output image summary differs from pickle")
    if len(images) != success["required_image_rows"] or len(detections) != success["required_detection_rows"]:
        raise AssertionError("Output row counts changed")
    if images["frame"].astype(int).tolist() != list(range(success["required_image_rows"])):
        raise AssertionError("Output frames are not exactly 0..254")
    if detections["track_id"].isna().any():
        raise AssertionError("Output track_id contains null")
    if int(detections["track_id"].nunique()) != success["required_unique_track_ids"]:
        raise AssertionError("Output unique track count changed")
    non_null = {
        column: int(detections[column].notna().sum())
        for column in ("bbox_pitch", "role", "team", "jersey_number")
    }
    non_null["parameters"] = int(images["parameters"].notna().sum())
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "members": members,
        "image_rows": len(images), "detection_rows": len(detections),
        "unique_track_ids": int(detections["track_id"].nunique()),
        "non_null_target_fields": non_null,
        "input_archive_sha256_after_run": sha256_file(input_path),
        "refiner_required_columns_present": True,
    }


def terminate(process: subprocess.Popen[Any], grace: int) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def latest_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1]) if lines else None


def validate_events(path: Path, phases: list[str]) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(record["status"] == "failed" for record in records):
        raise AssertionError("Worker event stream contains a failed phase")
    observed = [(record["phase"], record["status"]) for record in records if record["phase"] in phases]
    expected = [(phase, status) for phase in phases for status in ("started", "passed")]
    if observed != expected:
        raise AssertionError(f"Worker phase event order changed: {observed}")
    return {
        "path": str(path), "record_count": len(records), "phase_order": phases,
        "all_required_phases_started_and_passed": True,
    }


def run_worker(
    manifest: dict[str, Any], manifest_path: Path, preflight: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    approval_name, separator, approval_value = manifest["approval"]["required_environment"].partition("=")
    if separator != "=" or not approval_name or os.environ.get(approval_name) != approval_value:
        raise PermissionError(f"Run mode requires {manifest['approval']['required_environment']}")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-9]+", cuda) is None:
        raise PermissionError("CUDA_VISIBLE_DEVICES must name exactly one numeric device")
    outputs = manifest["outputs"]
    report_dir = Path(outputs["report_dir"])
    cache = Path(manifest["adapters"]["local_cache_dir"])
    require_absent(report_dir, "run report")
    require_absent(Path(outputs["hydra_run_dir"]), "Hydra run")
    require_absent(cache, "run cache")
    report_dir.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    events = Path(outputs["events"])
    events.touch(exist_ok=False)
    command = [
        manifest["python"], manifest["source_files"]["worker"],
        "--manifest", str(manifest_path), "--events", str(events),
        "--config-dir", manifest["config"]["directory"],
        "--config-name", manifest["config"]["name"],
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": "", "LD_LIBRARY_PATH": "", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true", "WANDB_MODE": "disabled",
        "HYDRA_FULL_ERROR": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface" / "transformers"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
    })
    timeouts = manifest["timeouts_seconds"]
    started_wall = time.time()
    started = time.monotonic()
    timed_out = False
    timeout_phase = None
    phase_seen_at: dict[str, float] = {}
    heartbeat = int(manifest["heartbeat_seconds"])
    with Path(outputs["log"]).open("x", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cuda_visible_devices": cuda}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        next_heartbeat = started + heartbeat
        while process.poll() is None:
            now = time.monotonic()
            event = latest_event(events)
            active = "worker_boot" if event is None else event["phase"]
            if event is not None and event["status"] == "passed":
                active = "worker_transition"
            phase_seen_at.setdefault(active, now)
            allowed = int(timeouts.get(active, timeouts["worker_transition"]))
            if now - started >= timeouts["overall"] or now - phase_seen_at[active] >= allowed:
                timed_out = True
                timeout_phase = active
                terminate(process, int(timeouts["termination_grace"]))
                break
            if now >= next_heartbeat:
                print(f"heartbeat phase={active} elapsed_seconds={now-started:.1f} pid={process.pid}", flush=True)
                next_heartbeat += heartbeat
            time.sleep(1)
        exit_code = process.wait()
    result: dict[str, Any] = {
        "schema_version": 1, "gate": "G10-B",
        "stage": manifest["stage"].removesuffix("_prepared"),
        "started_unix": started_wall, "ended_unix": time.time(),
        "wall_seconds": time.monotonic() - started,
        "process_exit_code": exit_code, "timed_out": timed_out,
        "timeout_phase": timeout_phase, "command": command,
        "cuda_visible_devices": cuda, "events": str(events), "log": outputs["log"],
        "preflight": preflight,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "fallbacks_used": [], "evaluation_started": False, "training_started": False,
    }
    final_exit = exit_code if exit_code else 0
    if not timed_out and exit_code == 0:
        try:
            result["event_contract"] = validate_events(events, manifest["phases"])
            result["state_archive"] = validate_output_state(manifest)
            result.update({"outcome": "passed", "assertions_passed": True})
        except Exception as error:
            result.update({
                "outcome": "failed", "assertions_passed": False,
                "failure_category": "artifact_integrity",
                "error": f"{type(error).__name__}: {error}",
            })
            final_exit = 1
    else:
        result.update({
            "outcome": "failed", "assertions_passed": False,
            "failure_category": "phase_timeout" if timed_out else "worker_failure",
        })
        final_exit = final_exit or 1
    atomic_json(result, Path(outputs["result"]))
    return result, final_exit


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    started = time.time()
    print("heartbeat phase=static_preflight status=started", flush=True)
    preflight = static_preflight(manifest, args.mode)
    if preflight["torch_imported"]:
        raise AssertionError("Torch was unexpectedly imported during static preflight")
    print("heartbeat phase=static_preflight status=passed", flush=True)
    if args.mode == "run":
        result, exit_code = run_worker(manifest, args.manifest, preflight)
        print(json.dumps({"outcome": result["outcome"], "result": manifest["outputs"]["result"]}), flush=True)
        return exit_code
    outputs = manifest["outputs"]
    report_dir = Path(outputs["preflight_report_dir"])
    require_absent(report_dir, "preflight report")
    report_dir.mkdir(parents=True, exist_ok=False)
    result_path = Path(outputs["preflight_result"])
    log_path = Path(outputs["preflight_log"])
    result = {
        "schema_version": 1, "gate": "G10-B",
        "stage": "prerefiner_enrichment_run1_preflight_retry1",
        "outcome": "passed", "assertions_passed": True,
        "started_unix": started, "ended_unix": time.time(),
        "wall_seconds": time.time() - started,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "git": git_identity(), "preflight": preflight,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_operations": [], "tracklab_or_model_imports": False,
        "torch_imported": False, "weight_contents_read": False,
        "inference_started": False, "evaluation_started": False,
        "training_started": False, "fallbacks_used": [],
        "next_step_requires_fresh_gpu_approval": True,
    }
    atomic_json(result, result_path)
    write_new(
        "heartbeat phase=static_preflight status=started\n"
        "heartbeat phase=static_preflight status=passed\nexit_code=0\n",
        log_path,
    )
    print(json.dumps({"outcome": "passed", "result": str(result_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
