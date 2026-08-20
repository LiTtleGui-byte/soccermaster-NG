#!/usr/bin/env python3
"""Prepare a fresh local-only data view for the five-head inference smoke."""

from __future__ import annotations

import os
from pathlib import Path


REPO = Path("/home/tianlin/SoccerMaster")
VIEW = REPO / ".runtime/local_takeover/soccermaster_five_heads_sngs10004/dataloader_view"
IMAGES = REPO / ".local_assets/data/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1"
PKL = REPO / ".runtime/g10/sngs10004_current_pipeline_conversion/run1/SNGS-10004.pkl"


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Preparation is CPU-only and requires CUDA_VISIBLE_DEVICES='' ")
    if VIEW.exists() or VIEW.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing data view: {VIEW}")
    names = sorted(path.name for path in IMAGES.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(1, 256)]
    if names != expected:
        raise AssertionError("Local image asset is not exactly 255 ordered JPEG frames")
    if not PKL.is_file():
        raise FileNotFoundError(PKL)

    dataset = VIEW / "SN-GSR-2024/SoccerNetGS"
    (dataset / "train").mkdir(parents=True)
    image_link = dataset / "sn500/SNGS-10004/img1"
    image_link.parent.mkdir(parents=True)
    image_link.symlink_to(IMAGES, target_is_directory=True)
    extracted = dataset / "extracted_info"
    extracted.mkdir()
    (extracted / "SNGS-10004.pkl").symlink_to(PKL)
    print(f"prepared_local_data_view={VIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
