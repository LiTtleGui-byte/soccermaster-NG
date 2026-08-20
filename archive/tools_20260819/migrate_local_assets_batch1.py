#!/usr/bin/env python3
"""Guarded, transactional copy of the approved local-takeover asset batch."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/home/tianlin/SoccerMaster")
ASSET_ROOT = REPO / ".local_assets"
APPROVAL_ENV = "SOCCERMASTER_LOCAL_ASSET_BATCH1_APPROVED"
EXPECTED_TOTAL = 26_492_433_131
MINIMUM_POST_COPY_FREE = 100_000_000_000
PER_ASSET_TIMEOUT_SECONDS = 8 * 60 * 60


@dataclass(frozen=True)
class Asset:
    name: str
    source: Path
    target: Path
    expected_bytes: int
    kind: str


ASSETS = (
    Asset(
        "soccermaster_epoch19",
        Path("/remote-home/haolinyang/sports/Soccer-Backbone/outputs/pretrain_large_512_multitask_aug_consine_part_temporal_early_freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19"),
        ASSET_ROOT / "checkpoints/soccermaster_epoch19",
        7_089_887_126,
        "directory",
    ),
    Asset(
        "yolo_person",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/yolo/yolo_v8x6_person_lr_default_best.pt"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/yolo/yolo_v8x6_person_lr_default_best.pt",
        195_209_883,
        "file",
    ),
    Asset(
        "prtreid",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar",
        396_287_605,
        "file",
    ),
    Asset(
        "prtreid_hrnet",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/reid/hrnetv2_w32_imagenet_pretrained.pth",
        165_587_602,
        "file",
    ),
    Asset(
        "calibration_keypoints",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/calibration/SV_kp"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/calibration/SV_kp",
        264_964_645,
        "file",
    ),
    Asset(
        "calibration_lines",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/calibration/SV_lines"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/calibration/SV_lines",
        264_857_893,
        "file",
    ),
    Asset(
        "legibility",
        Path("/remote-home/haolinyang/sports/soccernet/sn-gamestate/pretrained_models/legibility/legibility_resnet34_soccer_20240215.pth"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/legibility/legibility_resnet34_soccer_20240215.pth",
        85_289_629,
        "file",
    ),
    Asset(
        "qwen_jersey_ocr",
        Path("/remote-home/haolinyang/huggingface_models/Qwen/Qwen2.5-VL-7B-Instruct"),
        ASSET_ROOT / "models/soccerfactory/pretrained_models/jn/Qwen2.5-VL-7B-Instruct",
        16_596_000_949,
        "directory",
    ),
    Asset(
        "refiner",
        Path("/remote-home/haolinyang/sports/soccernet/Refiner/outputs/train_timesformer_100clip_coord_only_not_0init_l2_xyflip_seed42_20250328_224427/best_model.pth"),
        ASSET_ROOT / "models/soccerfactory/refiner/best_model.pth",
        323_985_486,
        "file",
    ),
    Asset(
        "source_video",
        Path("/remote-home/haolinyang/public/sports/SoccerNet/dataset-720p/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_720p.mkv"),
        ASSET_ROOT / "data/soccernet/raw/sngs10004/1_720p.mkv",
        979_439_471,
        "file",
    ),
    Asset(
        "camera_labels",
        Path("/remote-home/haolinyang/public/sports/SoccerNet/dataset-cameras/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-cameras.json"),
        ASSET_ROOT / "data/soccernet/cameras/sngs10004/Labels-cameras.json",
        106_194,
        "file",
    ),
    Asset(
        "clip_mapping",
        Path("/remote-home/haolinyang/datasets/SN-GSR-2024/SoccerNetGS/sn_2_clip.json"),
        ASSET_ROOT / "data/SN-GSR-2024/SoccerNetGS/sn_2_clip.json",
        8_322_923,
        "file",
    ),
    Asset(
        "prepared_frames",
        Path("/mnt/nas/haolinyang/sports/Soccer-Backbone/datasets/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1"),
        ASSET_ROOT / "data/SN-GSR-2024/SoccerNetGS/sn500/SNGS-10004/img1",
        122_493_725,
        "directory",
    ),
)


def apparent_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    completed = subprocess.run(
        ["du", "-sb", str(path)],
        check=True,
        text=True,
        capture_output=True,
        timeout=300,
    )
    return int(completed.stdout.split()[0])


def content_identity(path: Path) -> tuple[int, int, int]:
    if path.is_file():
        return (1, path.stat().st_size, 0)
    files = 0
    bytes_total = 0
    symlinks = 0
    for root, directories, names in os.walk(path, followlinks=False):
        directories.sort()
        names.sort()
        for name in names:
            item = Path(root) / name
            stat = item.lstat()
            files += 1
            if item.is_symlink():
                symlinks += 1
                bytes_total += len(os.readlink(item).encode())
            else:
                bytes_total += stat.st_size
    return files, bytes_total, symlinks


def preflight() -> dict:
    if sum(asset.expected_bytes for asset in ASSETS) != EXPECTED_TOTAL:
        raise AssertionError("Asset table total changed")
    records = []
    for asset in ASSETS:
        if not asset.source.exists():
            raise FileNotFoundError(asset.source)
        if asset.kind == "file" and not asset.source.is_file():
            raise AssertionError(f"Expected file: {asset.source}")
        if asset.kind == "directory" and not asset.source.is_dir():
            raise AssertionError(f"Expected directory: {asset.source}")
        observed = apparent_size(asset.source)
        if observed != asset.expected_bytes:
            raise AssertionError(
                f"Size changed for {asset.name}: {observed} != {asset.expected_bytes}"
            )
        partial = asset.target.with_name(asset.target.name + ".partial")
        if asset.target.exists() or asset.target.is_symlink():
            raise FileExistsError(f"Final target already exists: {asset.target}")
        if partial.exists() or partial.is_symlink():
            raise FileExistsError(f"Partial target already exists: {partial}")
        records.append(
            {
                "name": asset.name,
                "source": str(asset.source),
                "target": str(asset.target),
                "bytes": observed,
                "kind": asset.kind,
            }
        )
    free = shutil.disk_usage(REPO).free
    if free - EXPECTED_TOTAL < MINIMUM_POST_COPY_FREE:
        raise OSError(
            f"Insufficient safety margin: free={free}, copy={EXPECTED_TOTAL}, "
            f"required_post_copy={MINIMUM_POST_COPY_FREE}"
        )
    return {
        "status": "preflight_passed",
        "asset_count": len(ASSETS),
        "expected_total_bytes": EXPECTED_TOTAL,
        "free_bytes_before": free,
        "expected_free_bytes_after": free - EXPECTED_TOTAL,
        "minimum_post_copy_free_bytes": MINIMUM_POST_COPY_FREE,
        "targets_absent": True,
        "assets": records,
    }


def heartbeat(stop: threading.Event, asset_name: str) -> None:
    while not stop.wait(30):
        print(
            json.dumps(
                {
                    "event": "heartbeat",
                    "asset": asset_name,
                    "unix_time": time.time(),
                    "free_bytes": shutil.disk_usage(REPO).free,
                }
            ),
            flush=True,
        )


def copy_asset(asset: Asset) -> dict:
    partial = asset.target.with_name(asset.target.name + ".partial")
    asset.target.parent.mkdir(parents=True, exist_ok=True)
    source_identity = content_identity(asset.source)
    command = ["rsync", "-a", "--no-owner", "--no-group", "--info=progress2"]
    if asset.kind == "directory":
        command.extend([str(asset.source) + "/", str(partial) + "/"])
    else:
        command.extend([str(asset.source), str(partial)])
    print(json.dumps({"event": "copy_started", "asset": asset.name}), flush=True)
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(stop, asset.name), daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=PER_ASSET_TIMEOUT_SECONDS,
        )
    finally:
        stop.set()
        thread.join(timeout=2)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync failed for {asset.name}: {completed.returncode}")
    target_identity = content_identity(partial)
    if target_identity != source_identity:
        raise AssertionError(
            f"Copied content identity differs for {asset.name}: "
            f"{target_identity} != {source_identity}"
        )
    os.replace(partial, asset.target)
    result = {
        "name": asset.name,
        "elapsed_seconds": time.monotonic() - started,
        "content_identity": {
            "files": source_identity[0],
            "file_bytes": source_identity[1],
            "symlinks": source_identity[2],
        },
        "target": str(asset.target),
    }
    print(json.dumps({"event": "copy_completed", **result}), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = preflight()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        return 0
    if os.environ.get(APPROVAL_ENV) != "YES":
        raise PermissionError(f"Missing exact approval guard: {APPROVAL_ENV}=YES")

    runtime = REPO / ".runtime/local_takeover/batch1"
    runtime.mkdir(parents=True, exist_ok=True)
    result_path = runtime / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    started = time.time()
    result = {
        "status": "running",
        "started_unix": started,
        "approval_guard": f"{APPROVAL_ENV}=YES",
        "copies": [],
    }
    try:
        for asset in ASSETS:
            result["copies"].append(copy_asset(asset))
        result["status"] = "completed"
        result["free_bytes_after"] = shutil.disk_usage(REPO).free
        return_code = 0
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        return_code = 1
    finally:
        result["finished_unix"] = time.time()
        result["elapsed_seconds"] = result["finished_unix"] - started
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
