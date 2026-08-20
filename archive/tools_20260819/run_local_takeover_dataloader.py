#!/usr/bin/env python3
"""Run the local SoccerFactory-to-SoccerMaster DataLoader smoke with heartbeat."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


REPO = Path("/home/tianlin/SoccerMaster")
MANIFEST = REPO / "reproduction/manifests/g10_local_takeover_dataloader_sngs10004.json"


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("DataLoader smoke is CPU-only")
    stopped = threading.Event()

    def heartbeat() -> None:
        started = time.monotonic()
        while not stopped.wait(30):
            print(f"heartbeat phase=dataloader_cpu elapsed_seconds={time.monotonic()-started:.1f}", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        from reproduction.gates.g10_current_pipeline_dataloader_smoke import main as smoke_main

        sys.argv = [sys.argv[0], "--manifest", str(MANIFEST)]
        return smoke_main()
    finally:
        stopped.set()
        thread.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
