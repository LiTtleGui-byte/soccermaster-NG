#!/usr/bin/env python3
"""Confirm that the fixed SoccerFactory module set imports from local vendor code."""

from __future__ import annotations

import json
import os
from pathlib import Path


REPO = Path("/home/tianlin/SoccerMaster")
VENDOR_ROOTS = (
    REPO / "vendor/soccerfactory/tracklab",
    REPO / "vendor/soccerfactory/sn-gamestate",
    REPO / "vendor/soccerfactory/refiner",
)
BLOCKED_MIM_CACHE = Path("/home/tianlin/.cache/mim")


def main() -> int:
    original_makedirs = os.makedirs
    suppressed: list[str] = []

    def guarded_makedirs(name, *args, **kwargs):
        requested = Path(name).resolve(strict=False)
        if requested == BLOCKED_MIM_CACHE:
            suppressed.append(str(requested))
            return None
        return original_makedirs(name, *args, **kwargs)

    os.makedirs = guarded_makedirs
    try:
        import torch
        import inference as refiner_inference
        import sn_gamestate.calibration.nbjw_calib as calibration
        import sn_gamestate.detect_multiple.yolov8_person_api as detector
        import sn_gamestate.jersey.qwen2_5vl_ocr_api as jersey
        import sn_gamestate.legibility.legibility_api as legibility
        import sn_gamestate.reid.prtreid_api as reid
        import sn_gamestate.team.tracklet_team_clustering_api as team
        import sn_gamestate.tracklet_agg.majority_vote_filter_api as aggregation
        import tracklab.main as tracklab_main
    finally:
        os.makedirs = original_makedirs

    modules = (
        tracklab_main,
        detector,
        reid,
        calibration,
        legibility,
        jersey,
        aggregation,
        team,
        refiner_inference,
    )
    origins = {
        module.__name__: str(Path(module.__file__).resolve()) for module in modules
    }
    for name, origin in origins.items():
        if not any(Path(origin).is_relative_to(root) for root in VENDOR_ROOTS):
            raise AssertionError(f"{name} imported outside local vendor roots: {origin}")
    if torch.cuda.is_initialized():
        raise AssertionError("Source import audit unexpectedly initialized CUDA")

    print(
        json.dumps(
            {
                "status": "passed",
                "origins": origins,
                "suppressed_mim_cache_requests": suppressed,
                "cuda_initialized": False,
                "model_or_checkpoint_loaded": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
