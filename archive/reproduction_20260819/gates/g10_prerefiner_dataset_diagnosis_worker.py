#!/usr/bin/env python3
"""Future CPU-only worker for staged pre-Refiner dataset diagnosis run2."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Callable


REPO = Path("/home/tianlin/SoccerMaster")
EXPECTED_IDENTITY = (1, "G10-B", "prerefiner_dataset_diagnosis_run2_prepared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPO):
        raise AssertionError(f"Local path escapes repository: {path}")
    return resolved


def emit(events: Path, phase: str, status: str, started: float, **extra: Any) -> None:
    record = {
        "phase": phase,
        "status": status,
        "unix_time": time.time(),
        "worker_elapsed_seconds": time.monotonic() - started,
        "pid": os.getpid(),
        **extra,
    }
    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


class ReplayPool:
    """Minimal Pool-shaped adapter that replays one already-read video result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def imap_unordered(self, function: Any, values: Any) -> Any:
        del function
        items = list(values)
        if len(items) != 1:
            raise AssertionError(f"ReplayPool expected one job, got {len(items)}")
        return iter([self.result])


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    identity = (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage"))
    if identity != EXPECTED_IDENTITY:
        raise AssertionError(f"Unexpected manifest identity: {identity}")
    inside_repo(args.events)
    approval_name, separator, approval_value = manifest["approval"]["required_environment"].partition("=")
    if separator != "=" or os.environ.get(approval_name) != approval_value:
        raise PermissionError("Missing future CPU diagnosis approval guard")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CPU-only diagnosis requires CUDA_VISIBLE_DEVICES to be explicitly empty")
    if str(Path.home()) != "/home/tianlin":
        raise AssertionError("HOME changed")

    started = time.monotonic()

    def run_phase(name: str, action: Callable[[], Any]) -> Any:
        phase_started = time.monotonic()
        emit(args.events, name, "started", started)
        try:
            value = action()
        except BaseException as error:
            emit(
                args.events,
                name,
                "failed",
                started,
                phase_seconds=time.monotonic() - phase_started,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        output = value[1] if isinstance(value, tuple) and len(value) == 2 else value
        if not isinstance(output, dict):
            output = {}
        emit(
            args.events,
            name,
            "passed",
            started,
            phase_seconds=time.monotonic() - phase_started,
            output=output,
        )
        return value[0] if isinstance(value, tuple) and len(value) == 2 else value

    def import_tracklab_main() -> tuple[Any, dict[str, Any]]:
        import tracklab.main as tracklab_main

        origin = str(Path(tracklab_main.__file__).resolve())
        expected = manifest["source_files"]["tracklab_main"]["path"]
        if origin != expected:
            raise AssertionError(f"Unexpected tracklab.main origin: {origin}")
        return tracklab_main, {"origin": origin}

    run_phase("import_tracklab_main", import_tracklab_main)

    composed: dict[str, Any] = {}

    def hydra_cli_compose() -> tuple[Any, dict[str, Any]]:
        from hydra._internal.hydra import Hydra
        from hydra._internal.utils import create_config_search_path
        from hydra.core.global_hydra import GlobalHydra
        from hydra.types import RunMode
        from omegaconf import OmegaConf

        search_path = create_config_search_path(search_path_dir=manifest["config"]["directory"])
        search_path.append(provider="tracklab_core", path="pkg://tracklab.configs", anchor="main")
        search_path.append(provider="sn_gamestate", path="pkg://sn_gamestate.configs", anchor="main")
        hydra = Hydra.create_main_hydra2(
            task_name="g10_prerefiner_dataset_diagnosis_run2",
            config_search_path=search_path,
        )
        try:
            cfg = hydra.compose_config(
                config_name=manifest["config"]["name"],
                overrides=[],
                run_mode=RunMode.RUN,
                with_log_configuration=False,
            )
        finally:
            GlobalHydra.instance().clear()
        output = {
            "pipeline": list(cfg.pipeline),
            "eval_set": str(cfg.dataset.eval_set),
            "nframes": int(cfg.dataset.nframes),
            "sequence": str(cfg.dataset.vids_dict.sn500[0]),
            "dataset_target": str(cfg.dataset._target_),
        }
        expected = {
            "pipeline": manifest["config"]["pipeline"],
            "eval_set": manifest["sample"]["split"],
            "nframes": manifest["sample"]["frame_count"],
            "sequence": manifest["sample"]["sequence"],
            "dataset_target": manifest["config"]["dataset_target"],
        }
        if output != expected:
            raise AssertionError(f"Composed dataset contract changed: {output}")
        resolved_path = inside_repo(Path(manifest["outputs"]["resolved_config"]))
        resolved_path.parent.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, resolved_path, resolve=True)
        composed["cfg"] = cfg
        return cfg, output

    run_phase("hydra_cli_compose", hydra_cli_compose)

    imported: dict[str, Any] = {}

    def import_dataset_wrapper() -> tuple[Any, dict[str, Any]]:
        original_makedirs = os.makedirs
        blocked = Path(manifest["diagnostic_adapter"]["suppressed_path_exact"]).resolve(strict=False)

        def guarded_makedirs(name: Any, *values: Any, **kwargs: Any) -> None:
            requested = Path(name).resolve(strict=False)
            if requested == blocked:
                return
            original_makedirs(name, *values, **kwargs)

        os.makedirs = guarded_makedirs
        try:
            from tracklab.wrappers.datasets.soccernet import soccernet_game_state
        finally:
            os.makedirs = original_makedirs
        origin = str(Path(soccernet_game_state.__file__).resolve())
        expected = manifest["source_files"]["soccernet_dataset"]["path"]
        if origin != expected:
            raise AssertionError(f"Unexpected dataset wrapper origin: {origin}")
        imported["dataset_module"] = soccernet_game_state
        return soccernet_game_state, {
            "origin": origin,
            "mim_home_cache_creation_suppressed": True,
            "home": str(Path.home()),
        }

    run_phase("import_dataset_wrapper", import_dataset_wrapper)

    def dataset_root_scan() -> dict[str, Any]:
        sample = manifest["sample"]
        dataset_root = Path(sample["dataset_root"])
        split_root = Path(sample["split_root"])
        video_root = Path(sample["video_root"])
        frame_root = Path(sample["frame_root"])
        for path in (dataset_root, split_root, video_root, frame_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if (video_root / "Labels-GameState.json").exists():
            raise AssertionError("Fixed sample unexpectedly gained Labels-GameState.json")
        frames = sorted(frame_root.glob("*.jpg"))
        if len(frames) != sample["frame_count"]:
            raise AssertionError(f"Unexpected frame count: {len(frames)}")
        total_bytes = sum(path.stat().st_size for path in frames)
        if total_bytes != sample["frame_total_bytes"]:
            raise AssertionError(f"Unexpected frame bytes: {total_bytes}")
        if frames[0].name != sample["first_frame"]["name"] or frames[0].stat().st_size != sample["first_frame"]["bytes"]:
            raise AssertionError("First frame metadata changed")
        if frames[-1].name != sample["last_frame"]["name"] or frames[-1].stat().st_size != sample["last_frame"]["bytes"]:
            raise AssertionError("Last frame metadata changed")
        return {
            "video_root": str(video_root),
            "labels_gamestate_present": False,
            "frame_count": len(frames),
            "frame_total_bytes": total_bytes,
            "frame_contents_read": False,
        }

    run_phase("dataset_root_scan", dataset_root_scan)

    pool_state: dict[str, Any] = {"pool": None, "closed": False}

    def pool_create() -> tuple[Any, dict[str, Any]]:
        module = imported["dataset_module"]
        workers = int(manifest["diagnostic_adapter"]["dataset_pool_workers"])
        pool = module.Pool(processes=workers)
        pool_state["pool"] = pool
        return pool, {
            "workers": workers,
            "jobs": int(manifest["diagnostic_adapter"]["pool_jobs"]),
            "start_method": multiprocessing.get_start_method(),
        }

    run_phase("pool_create", pool_create)

    video_result: dict[str, Any] = {}
    try:
        def single_video_worker() -> tuple[dict[str, Any], dict[str, Any]]:
            module = imported["dataset_module"]
            pool = pool_state["pool"]
            sample = manifest["sample"]
            job = {
                "dataset_path": sample["split_root"],
                "video_folder": sample["sequence"],
                "split": sample["split"],
            }
            result = pool.apply_async(module.video_dir_to_dfs, (job,)).get()
            if result is None:
                raise AssertionError("video_dir_to_dfs returned None")
            image_rows = len(result["image_metadata"])
            if image_rows != sample["frame_count"]:
                raise AssertionError(f"Unexpected video worker image rows: {image_rows}")
            if result["detections"] is not None or result["annotations_pitch_camera"] is not None:
                raise AssertionError("Unlabeled fixed sample unexpectedly returned annotations")
            if result["video_level_categories"] != []:
                raise AssertionError("Unlabeled fixed sample unexpectedly returned categories")
            if result["video_metadata"]["name"] != sample["sequence"]:
                raise AssertionError("Video worker sequence changed")
            video_result["value"] = result
            return result, {
                "sequence": result["video_metadata"]["name"],
                "image_rows": image_rows,
                "detections_is_none": True,
                "annotations_pitch_camera_is_none": True,
                "video_level_categories": 0,
            }

        run_phase("single_video_worker", single_video_worker)

        def pool_close_join() -> dict[str, Any]:
            pool = pool_state["pool"]
            pool.close()
            pool.join()
            pool_state["closed"] = True
            return {"closed": True, "joined": True}

        run_phase("pool_close_join", pool_close_join)
    finally:
        if pool_state["pool"] is not None and not pool_state["closed"]:
            pool_state["pool"].terminate()
            pool_state["pool"].join()

    tracking: dict[str, Any] = {}

    def tracking_set_construct() -> tuple[Any, dict[str, Any]]:
        module = imported["dataset_module"]
        original_pool = module.Pool
        replay = ReplayPool(video_result["value"])
        module.Pool = lambda *values, **kwargs: replay
        try:
            tracking_set = module.load_set(
                manifest["sample"]["split_root"],
                nvid=-1,
                vids_filter_set=[manifest["sample"]["sequence"]],
            )
        finally:
            module.Pool = original_pool
        tracking["set"] = tracking_set
        return tracking_set, {
            "video_rows": len(tracking_set.video_metadatas),
            "image_rows": len(tracking_set.image_metadatas),
            "detections_gt_is_none": tracking_set.detections_gt is None,
            "single_video_result_replayed": True,
        }

    run_phase("tracking_set_construct", tracking_set_construct)

    finalized: dict[str, Any] = {}

    def tracking_dataset_finalize() -> tuple[Any, dict[str, Any]]:
        from hydra.utils import instantiate

        module = imported["dataset_module"]
        original_load_set = module.load_set
        module.load_set = lambda *values, **kwargs: tracking["set"]
        try:
            dataset = instantiate(composed["cfg"].dataset)
        finally:
            module.load_set = original_load_set
        finalized["dataset"] = dataset
        tracking_set = dataset.sets[manifest["sample"]["split"]]
        return dataset, {
            "sets": sorted(dataset.sets),
            "video_rows": len(tracking_set.video_metadatas),
            "image_rows": len(tracking_set.image_metadatas),
            "detections_gt_is_none": tracking_set.detections_gt is None,
            "load_set_result_replayed": True,
        }

    run_phase("tracking_dataset_finalize", tracking_dataset_finalize)

    def contract_assertions() -> dict[str, Any]:
        success = manifest["success"]
        tracking_set = finalized["dataset"].sets[manifest["sample"]["split"]]
        if sorted(finalized["dataset"].sets) != success["required_dataset_sets"]:
            raise AssertionError("Dataset sets changed")
        if len(tracking_set.video_metadatas) != success["required_video_rows"]:
            raise AssertionError("Video row contract failed")
        if len(tracking_set.image_metadatas) != success["required_image_rows"]:
            raise AssertionError("Image row contract failed")
        names = tracking_set.video_metadatas["name"].astype(str).tolist()
        if names != [manifest["sample"]["sequence"]]:
            raise AssertionError(f"Sequence contract failed: {names}")
        frames = tracking_set.image_metadatas["frame"].astype(int).tolist()
        if frames != list(range(manifest["sample"]["frame_count"])):
            raise AssertionError("Frame values are not exactly 0..254")
        if tracking_set.detections_gt is not None:
            raise AssertionError("detections_gt must remain None for the unlabeled fixed sample")
        return {
            "sets": sorted(finalized["dataset"].sets),
            "sequence_names": names,
            "video_rows": len(tracking_set.video_metadatas),
            "image_rows": len(tracking_set.image_metadatas),
            "frame_min": min(frames),
            "frame_max": max(frames),
            "detections_gt_is_none": True,
        }

    run_phase("contract_assertions", contract_assertions)
    emit(args.events, "worker", "passed", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
