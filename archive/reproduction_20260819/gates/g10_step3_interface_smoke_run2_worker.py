#!/usr/bin/env python3
"""Future worker for the guarded run2 empty-pipeline state interface smoke."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    hydra_args = sys.argv[2:]
    if manifest["stage"] != "step3_interface_smoke_run2_prepared":
        raise AssertionError("Unexpected run2 manifest")
    if not manifest["execution_authorized"]:
        raise PermissionError("Run2 manifest is not execution-authorized")
    if os.environ.get("G10_STEP3_INTERFACE_RUN2_CPU_APPROVED") != "YES":
        raise PermissionError("Missing run2 CPU approval guard")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("CPU smoke must hide CUDA")

    import tracklab.main as tracklab_main

    original_instantiate = tracklab_main.instantiate
    original_init = tracklab_main.init_environment
    original_pipeline = tracklab_main.Pipeline
    original_state = tracklab_main.TrackerState
    original_evaluate = tracklab_main.evaluate
    targets: list[str] = []

    class DisabledEvaluator:
        def run(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("Evaluation must not run")

    def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
        target = str(config.get("_target_", ""))
        targets.append(target)
        if target == "tracklab.wrappers.TrackEvalEvaluator":
            return DisabledEvaluator()
        if target == "tracklab.wrappers.SoccerNetGameState":
            original_makedirs = os.makedirs
            blocked = Path("/home/tianlin/.cache/mim").resolve(strict=False)

            def guarded_makedirs(name: Any, *values: Any, **options: Any) -> Any:
                if Path(name).resolve(strict=False) == blocked:
                    return None
                return original_makedirs(name, *values, **options)

            os.makedirs = guarded_makedirs
            try:
                from tracklab.wrappers.datasets.soccernet import soccernet_game_state
            finally:
                os.makedirs = original_makedirs
            original_pool = soccernet_game_state.Pool

            def limited_pool(*values: Any, **options: Any) -> Any:
                if not values and "processes" not in options:
                    options["processes"] = 4
                return original_pool(*values, **options)

            soccernet_game_state.Pool = limited_pool
            try:
                return original_instantiate(config, *args, **kwargs)
            finally:
                soccernet_game_state.Pool = original_pool
        if target == "tracklab.engine.OfflineTrackingEngine":
            from omegaconf import open_dict

            engine_config = copy.deepcopy(config)
            with open_dict(engine_config.callbacks):
                for key in list(engine_config.callbacks.keys()):
                    if engine_config.callbacks[key] is None:
                        del engine_config.callbacks[key]
            return original_instantiate(engine_config, *args, **kwargs)
        raise AssertionError(f"Unexpected top-level target: {target}")

    def init_environment(cfg: Any) -> str:
        if list(cfg.pipeline) or cfg.eval_tracking is not False:
            raise AssertionError("Runtime config is not the empty-pipeline smoke")
        device = original_init(cfg)
        import torch
        if device != "cpu" or torch.cuda.is_available() or torch.cuda.device_count() != 0:
            raise AssertionError("CUDA became visible")
        return device

    def pipeline(*args: Any, **kwargs: Any) -> Any:
        models = kwargs.get("models", args[0] if args else None)
        if models != []:
            raise AssertionError("A Step-3 module reached run2")
        return original_pipeline(*args, **kwargs)

    def tracker_state(*args: Any, **kwargs: Any) -> Any:
        if Path(kwargs["load_file"]).resolve() != Path(manifest["input_state"]).resolve():
            raise AssertionError("Input state changed")
        output = Path(kwargs["save_file"]).resolve(strict=False)
        if output != Path(manifest["output_state"]).resolve(strict=False):
            raise AssertionError("Output state changed")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output state occupied")
        return original_state(*args, **kwargs)

    def evaluate(cfg: Any, *_args: Any) -> None:
        if cfg.eval_tracking is not False:
            raise AssertionError("Evaluation flag changed")

    tracklab_main.instantiate = instantiate
    tracklab_main.init_environment = init_environment
    tracklab_main.Pipeline = pipeline
    tracklab_main.TrackerState = tracker_state
    tracklab_main.evaluate = evaluate
    sys.argv = [sys.argv[0], *hydra_args]
    try:
        return_code = int(tracklab_main.main() or 0)
        expected = {
            "tracklab.wrappers.SoccerNetGameState",
            "tracklab.wrappers.TrackEvalEvaluator",
            "tracklab.engine.OfflineTrackingEngine",
        }
        if set(targets) != expected:
            raise AssertionError(f"Unexpected target set: {targets}")
        print(json.dumps({"status": "passed", "targets": targets}))
        return return_code
    finally:
        tracklab_main.instantiate = original_instantiate
        tracklab_main.init_environment = original_init
        tracklab_main.Pipeline = original_pipeline
        tracklab_main.TrackerState = original_state
        tracklab_main.evaluate = original_evaluate


if __name__ == "__main__":
    raise SystemExit(main())
