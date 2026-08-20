#!/usr/bin/env python3
"""Stable root launcher for the editable research training entry."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research/src"))
sys.path.insert(0, str(ROOT / "baseline/code/sn_calibration/src"))
sys.path.insert(
    0,
    str(
        ROOT
        / "research/src/soccermaster/models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
    ),
)
runpy.run_module("soccermaster.training.legacy_entry", run_name="__main__")
