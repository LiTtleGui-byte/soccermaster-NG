"""Import-only check for vendored commentary modules.

No class is instantiated, so this must not read model or checkpoint weights.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time


def main() -> int:
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            print(
                f"[HEARTBEAT] commentary import-only elapsed="
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be an empty string")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        from .dataset.commentary import (
            MatchVisionCommentary_new_benchmark_from_npy_Dataset,
        )
        from .model.MatchVision_part_temporal import VisionTimesformer
        from .model.matchvoice_Qformer import BertConfig, BertLMHeadModel
        from .model.matchvoice_model_all_blocks import matchvoice_model_all_blocks

        symbols = {
            "dataset": MatchVisionCommentary_new_benchmark_from_npy_Dataset.__name__,
            "visual_encoder": VisionTimesformer.__name__,
            "qformer_config": BertConfig.__name__,
            "qformer_model": BertLMHeadModel.__name__,
            "generation_model": matchvoice_model_all_blocks.__name__,
        }
        print(
            json.dumps(
                {
                    "status": "passed",
                    "python": sys.version.split()[0],
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ),
                    "symbols": symbols,
                    "model_instances_created": False,
                    "tokenizer_loaded": False,
                    "checkpoint_loaded": False,
                    "forward_executed": False,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        stop.set()
        thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
