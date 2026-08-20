#!/usr/bin/env python3
"""Phase-observable worker for guarded G10-B pre-Refiner enrichment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


REPO = Path("/home/tianlin/SoccerMaster")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    return parser.parse_known_args()


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


def main() -> int:
    args, hydra_args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest.get("schema_version"), manifest.get("gate"), manifest.get("stage")) != (
        1, "G10-B", "prerefiner_enrichment_run2_prepared"
    ):
        raise AssertionError("Unexpected enrichment manifest identity")
    events = args.events
    if not events.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError("Events path escapes repository")
    approval_name, separator, approval_value = manifest["approval"]["required_environment"].partition("=")
    if separator != "=" or not approval_name or os.environ.get(approval_name) != approval_value:
        raise PermissionError(f"Missing GPU approval guard: {manifest['approval']['required_environment']}")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible.isdigit() or "," in cuda_visible:
        raise PermissionError("CUDA_VISIBLE_DEVICES must contain exactly one numeric device")

    started = time.monotonic()

    def run_phase(name: str, action: Callable[[], Any], output: dict[str, Any] | None = None) -> Any:
        phase_started = time.monotonic()
        details = output if output is not None else {}
        emit(events, name, "started", started, output=details)
        try:
            value = action()
        except BaseException as error:
            emit(
                events, name, "failed", started,
                phase_seconds=time.monotonic() - phase_started,
                error=f"{type(error).__name__}: {error}",
            )
            raise
        emit(
            events, name, "passed", started,
            phase_seconds=time.monotonic() - phase_started,
            output=details,
        )
        return value

    def import_tracklab() -> Any:
        import tracklab.main as tracklab_main

        return tracklab_main

    tracklab_main = run_phase(
        "import_tracklab_main", import_tracklab,
        {"expected_origin": manifest["source_files"]["tracklab_main"]},
    )
    if str(Path(tracklab_main.__file__).resolve()) != manifest["source_files"]["tracklab_main"]:
        raise AssertionError(f"Unexpected tracklab.main origin: {tracklab_main.__file__}")

    original_instantiate = tracklab_main.instantiate
    original_init_environment = tracklab_main.init_environment
    original_pipeline = tracklab_main.Pipeline
    original_tracker_state = tracklab_main.TrackerState
    original_evaluate = tracklab_main.evaluate
    hydra_started = time.monotonic()
    hydra_finished = False
    model_targets = {spec["target"]: name for name, spec in manifest["modules"].items()}

    def validate_weight_specs(component: str) -> list[dict[str, Any]]:
        validated = []
        for spec in manifest["modules"][component]["weight_reads"]:
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Weight asset changed: {path}")
            validated.append({"path": str(path), "bytes": spec["bytes"]})
        qwen_key = manifest["modules"][component].get("qwen_asset_key")
        if qwen_key is not None:
            spec = manifest["qwen_assets"][qwen_key]
            configured = Path(spec["configured_path"])
            if spec.get("configured_path_is_symlink", True):
                if not configured.is_symlink() or os.readlink(configured) != spec["link_target"]:
                    raise AssertionError("Qwen symlink target changed")
                root = configured.resolve(strict=True)
            else:
                if not configured.is_dir() or configured.is_symlink():
                    raise AssertionError("Local Qwen asset must be a real directory")
                root = configured.resolve(strict=True)
            total = 0
            for shard in spec["shards"]:
                path = root / shard["name"]
                if not path.is_file() or path.stat().st_size != shard["bytes"]:
                    raise AssertionError(f"Qwen shard changed: {path}")
                total += path.stat().st_size
            if total != spec["logical_shard_bytes"]:
                raise AssertionError("Qwen logical shard total changed")
            validated.append({
                "path": str(configured), "resolved_path": str(root),
                "logical_shard_bytes": total, "kind": "local_sharded_model",
            })
        return validated

    def dataset_instantiate(config: Any, *values: Any, **kwargs: Any) -> Any:
        adapters = manifest["adapters"]
        original_makedirs = os.makedirs
        blocked = Path(adapters["suppressed_path_exact"]).resolve(strict=False)

        def guarded_makedirs(name: Any, *args: Any, **inner_kwargs: Any) -> None:
            requested = Path(name).resolve(strict=False)
            if adapters["suppress_mim_home_cache_creation"] and requested == blocked:
                return
            original_makedirs(name, *args, **inner_kwargs)

        os.makedirs = guarded_makedirs
        try:
            from tracklab.wrappers.datasets.soccernet import soccernet_game_state
        finally:
            os.makedirs = original_makedirs

        original_pool = soccernet_game_state.Pool
        worker_count = int(adapters["dataset_pool_workers"])

        def limited_pool(*args: Any, **pool_kwargs: Any) -> Any:
            if not args and "processes" not in pool_kwargs:
                pool_kwargs["processes"] = worker_count
            return original_pool(*args, **pool_kwargs)

        soccernet_game_state.Pool = limited_pool
        try:
            dataset = original_instantiate(config, *values, **kwargs)
        finally:
            soccernet_game_state.Pool = original_pool
        split = manifest["sample"]["split"]
        tracking_set = dataset.sets[split]
        if len(tracking_set.video_metadatas) != 1:
            raise AssertionError("Dataset contract requires exactly one video")
        if len(tracking_set.image_metadatas) != manifest["sample"]["frame_count"]:
            raise AssertionError("Dataset contract requires exactly 255 image rows")
        sequences = set(tracking_set.video_metadatas["name"].astype(str))
        if sequences != {manifest["sample"]["sequence"]}:
            raise AssertionError(f"Unexpected dataset sequence set: {sequences}")
        return dataset

    def instantiate_legibility_without_base_download(
        config: Any, *values: Any, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        adapter = manifest["adapters"]["legibility_resnet34_no_download"]
        if not adapter["enabled"] or adapter["scope"] != "instantiate_legibility_only":
            raise AssertionError("Legibility no-download adapter changed")
        from torchvision import models as torchvision_models

        original_resnet34 = torchvision_models.resnet34
        calls: list[dict[str, Any]] = []

        def no_download_resnet34(*model_args: Any, **model_kwargs: Any) -> Any:
            if model_kwargs.get("pretrained") is not True or "weights" in model_kwargs:
                raise AssertionError(f"Unexpected ResNet34 construction: {model_kwargs}")
            calls.append({"original_pretrained": True, "replacement_pretrained": False})
            model_kwargs["pretrained"] = False
            return original_resnet34(*model_args, **model_kwargs)

        torchvision_models.resnet34 = no_download_resnet34
        try:
            instance = original_instantiate(config, *values, **kwargs)
        finally:
            torchvision_models.resnet34 = original_resnet34
        if calls != [{"original_pretrained": True, "replacement_pretrained": False}]:
            raise AssertionError(f"Unexpected ResNet34 adapter calls: {calls}")
        expected_checkpoint = manifest["modules"]["legibility"]["weight_reads"][0]["path"]
        if str(instance.cfg.legibility_model_path) != expected_checkpoint:
            raise AssertionError("Legibility task checkpoint changed")
        return instance, {
            "resnet34_adapter_calls": calls,
            "strict_task_checkpoint_load_completed": True,
            "task_checkpoint": expected_checkpoint,
        }

    def assert_qwen_single_device(instance: Any) -> dict[str, Any]:
        adapter = manifest["adapters"]["qwen_single_device_assertion"]
        if not adapter["enabled"]:
            raise AssertionError("Qwen device assertion disabled")
        devices = sorted({str(parameter.device) for parameter in instance.model.parameters()})
        if devices != adapter["expected_parameter_devices"]:
            raise AssertionError(f"Qwen parameter devices changed: {devices}")
        device_map = getattr(instance.model, "hf_device_map", None)
        return {
            "parameter_devices": devices,
            "hf_device_map": None if device_map is None else {str(k): str(v) for k, v in device_map.items()},
            "single_visible_cuda_device_asserted": True,
        }

    def staged_instantiate(config: Any, *values: Any, **kwargs: Any) -> Any:
        target = str(config.get("_target_", ""))
        if target == "tracklab.wrappers.SoccerNetGameState":
            return run_phase(
                "instantiate_dataset",
                lambda: dataset_instantiate(config, *values, **kwargs),
                {
                    "pool_workers": manifest["adapters"]["dataset_pool_workers"],
                    "sequence": manifest["sample"]["sequence"],
                },
            )
        component = model_targets.get(target)
        if component is not None:
            phase = f"instantiate_{component}"
            output: dict[str, Any] = {
                "target": target,
                "declared_weight_reads": validate_weight_specs(component),
            }

            def instantiate_component() -> Any:
                if component == "legibility":
                    instance, adapter_output = instantiate_legibility_without_base_download(
                        config, *values, **kwargs
                    )
                    output["legibility_no_download_adapter"] = adapter_output
                else:
                    instance = original_instantiate(config, *values, **kwargs)
                if component != "evaluator" and getattr(instance, "training_enabled", False):
                    raise AssertionError(f"Training unexpectedly enabled for {component}")
                if component == "jersey_number_detect":
                    output["qwen_device_assertion"] = assert_qwen_single_device(instance)
                return instance

            return run_phase(phase, instantiate_component, output)
        if target == "tracklab.engine.OfflineTrackingEngine":
            engine_output: dict[str, Any] = {"target": target}

            def instantiate_engine() -> Any:
                adapter = manifest["adapters"]["engine_null_callback_filter"]
                if not adapter["enabled"] or not adapter["copy_before_filter"]:
                    raise AssertionError("Engine null callback adapter changed")
                from omegaconf import OmegaConf, open_dict

                engine_config = copy.deepcopy(config)
                callbacks = engine_config.get("callbacks")
                if callbacks is None:
                    raise AssertionError("Engine callbacks config is missing")
                original_keys = sorted(str(key) for key in callbacks.keys())
                null_keys = sorted(str(key) for key, value in callbacks.items() if value is None)
                if original_keys != adapter["expected_original_keys"]:
                    raise AssertionError(f"Engine callback keys changed: {original_keys}")
                if null_keys != adapter["expected_null_keys"]:
                    raise AssertionError(f"Engine null callbacks changed: {null_keys}")
                struct_before = bool(OmegaConf.is_struct(callbacks))
                if adapter["require_struct_before"] and not struct_before:
                    raise AssertionError("Engine callbacks are not in struct mode")
                if adapter["deletion_mode"] != "omegaconf_open_dict":
                    raise AssertionError("Engine callback deletion mode changed")
                with open_dict(callbacks):
                    for key in null_keys:
                        del callbacks[key]
                struct_after = bool(OmegaConf.is_struct(callbacks))
                if adapter["require_struct_restored"] and not struct_after:
                    raise AssertionError("Engine callback struct mode was not restored")
                remaining_keys = sorted(str(key) for key in callbacks.keys())
                if remaining_keys != adapter["expected_remaining_keys"]:
                    raise AssertionError(f"Engine remaining callbacks changed: {remaining_keys}")
                if any(value is None for value in callbacks.values()):
                    raise AssertionError("Engine callbacks still contain None")
                engine_output.update({
                    "callback_config_copied": True,
                    "callback_deletion_mode": adapter["deletion_mode"],
                    "callback_struct_before": struct_before,
                    "callback_struct_after": struct_after,
                    "original_callback_keys": original_keys,
                    "filtered_null_callback_keys": null_keys,
                    "remaining_callback_keys": remaining_keys,
                })
                return original_instantiate(engine_config, *values, **kwargs)

            engine = run_phase("instantiate_engine", instantiate_engine, engine_output)
            original_track_dataset = engine.track_dataset

            def staged_track_dataset() -> Any:
                return run_phase(
                    "track_dataset", original_track_dataset,
                    {
                        "required_frames": manifest["sample"]["frame_count"],
                        "required_detection_rows": manifest["sample"]["detection_rows"],
                    },
                )

            engine.track_dataset = staged_track_dataset
            return engine
        return original_instantiate(config, *values, **kwargs)

    def staged_init_environment(cfg: Any) -> Any:
        nonlocal hydra_finished
        config_contract = {
            "pipeline": list(cfg.pipeline),
            "eval_set": str(cfg.dataset.eval_set),
            "nframes": int(cfg.dataset.nframes),
            "sequence": str(cfg.dataset.vids_dict.sn500[0]),
            "state_load_file": str(cfg.state.load_file),
            "state_save_file": str(cfg.state.save_file),
            "eval_tracking": bool(cfg.eval_tracking),
            "visualization_is_null": cfg.visualization is None,
            "use_wandb": bool(cfg.use_wandb),
        }
        expected_contract = {
            "pipeline": manifest["config"]["pipeline"],
            "eval_set": manifest["sample"]["split"],
            "nframes": manifest["sample"]["frame_count"],
            "sequence": manifest["sample"]["sequence"],
            "state_load_file": manifest["input_state"]["path"],
            "state_save_file": "states/sn-gamestate.pklz",
            "eval_tracking": False,
            "visualization_is_null": True,
            "use_wandb": False,
        }
        if config_contract != expected_contract:
            raise AssertionError(f"Hydra contract changed: {config_contract}")
        emit(
            events, "hydra_cli_compose", "passed", started,
            phase_seconds=time.monotonic() - hydra_started,
            output=config_contract,
        )
        hydra_finished = True
        cuda_output: dict[str, Any] = {"cuda_visible_devices": cuda_visible}

        def initialize_and_assert_cuda() -> Any:
            value = original_init_environment(cfg)
            import torch

            guard = manifest["cuda_guard"]
            available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count())
            if available is not guard["cuda_available"]:
                raise AssertionError(f"CUDA availability changed: {available}")
            if device_count != guard["visible_device_count"]:
                raise AssertionError(f"Visible CUDA device count changed: {device_count}")
            logical_index = int(guard["logical_device_index"])
            torch.cuda.set_device(logical_index)
            properties = torch.cuda.get_device_properties(logical_index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
            if properties.name != guard["expected_device_name"]:
                raise AssertionError(f"CUDA device name changed: {properties.name}")
            if total_bytes < guard["minimum_total_memory_bytes"]:
                raise AssertionError(f"CUDA total memory below guard: {total_bytes}")
            if free_bytes < guard["minimum_free_memory_bytes"]:
                raise AssertionError(f"CUDA free memory below guard: {free_bytes}")
            cuda_output.update({
                "cuda_available": available, "visible_device_count": device_count,
                "logical_device_index": logical_index, "device_name": properties.name,
                "total_memory_bytes": int(total_bytes), "free_memory_bytes": int(free_bytes),
            })
            return value

        return run_phase("init_environment", initialize_and_assert_cuda, cuda_output)

    def staged_pipeline(*values: Any, **kwargs: Any) -> Any:
        return run_phase(
            "build_pipeline", lambda: original_pipeline(*values, **kwargs),
            {"modules": manifest["config"]["pipeline"]},
        )

    def staged_tracker_state(*values: Any, **kwargs: Any) -> Any:
        output = {
            "input_state": manifest["input_state"]["path"],
            "output_state": manifest["outputs"]["state_archive"],
        }

        def build_state() -> Any:
            actual_load = Path(kwargs.get("load_file")).resolve()
            actual_save = Path(kwargs.get("save_file")).resolve(strict=False)
            expected_load = Path(manifest["input_state"]["path"]).resolve()
            expected_save = Path(manifest["outputs"]["state_archive"]).resolve(strict=False)
            if actual_load != expected_load or actual_save != expected_save:
                raise AssertionError(
                    f"TrackerState paths changed: load={actual_load}, save={actual_save}"
                )
            if actual_load == actual_save:
                raise AssertionError("TrackerState input/output collision")
            if expected_save.exists() or expected_save.is_symlink():
                raise FileExistsError(f"Output state is already occupied: {expected_save}")
            return original_tracker_state(*values, **kwargs)

        return run_phase("build_tracker_state", build_state, output)

    def skip_evaluation(cfg: Any, evaluator: Any, tracker_state: Any) -> None:
        def assert_skipped() -> None:
            if cfg.eval_tracking is not False:
                raise AssertionError("Evaluation must remain disabled")

        run_phase("evaluation_skipped", assert_skipped, {"eval_tracking": False})

    tracklab_main.instantiate = staged_instantiate
    tracklab_main.init_environment = staged_init_environment
    tracklab_main.Pipeline = staged_pipeline
    tracklab_main.TrackerState = staged_tracker_state
    tracklab_main.evaluate = skip_evaluation

    emit(events, "hydra_cli_compose", "started", started)
    sys.argv = [sys.argv[0], *hydra_args]
    try:
        return_value = tracklab_main.main()
        if not hydra_finished:
            raise AssertionError("Hydra task function was not entered")
        import torch

        gpu_memory = {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        emit(events, "worker", "passed", started, output=gpu_memory)
        return int(return_value or 0)
    except BaseException as error:
        if not hydra_finished:
            emit(
                events, "hydra_cli_compose", "failed", started,
                phase_seconds=time.monotonic() - hydra_started,
                error=f"{type(error).__name__}: {error}",
            )
        traceback.print_exc()
        return 1
    finally:
        tracklab_main.instantiate = original_instantiate
        tracklab_main.init_environment = original_init_environment
        tracklab_main.Pipeline = original_pipeline
        tracklab_main.TrackerState = original_tracker_state
        tracklab_main.evaluate = original_evaluate


if __name__ == "__main__":
    raise SystemExit(main())
