#!/usr/bin/env python3
"""CPU-only identity and checkpoint-metadata validation for the NAS bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT = REPO / "experiments/commentary_generation"
REPORT = REPO / "reports/commentary_nas_bundle_preflight_20260817"
OUTPUT = REPORT / "result.json"
EXPECTED_ROOT = Path(
    "/mnt/nas/tianlin/SoccerMasterAssets/commentary_generation_epoch11_20260817"
)

if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from runtime.paths import (  # noqa: E402
    ASSET_LAYOUT,
    ASSET_ROOT,
    BERT_ROOT,
    GENERATION_CHECKPOINT,
    LLAMA_ROOT,
    SIGLIP2_ROOT,
    TEST_ANNOTATIONS,
    TRAIN_ANNOTATIONS,
    VISUAL_BACKBONE,
    WORD_WORLD,
)


EXPECTED_FILES = {
    TRAIN_ANNOTATIONS: (
        11_318_394,
        "0c4580786a3e6db7bcd38040190b7f19b80fc28e51eac6307ae94354759ddfa8",
    ),
    TEST_ANNOTATIONS: (
        1_541_678,
        "d163ac7a6b7705d6c98a1ccbb1325a004fcdfc477380425548ab6a8274e4c870",
    ),
    WORD_WORLD: (
        9_092,
        "654f03e1d4678cd0c3e8ca587af027e4bc14489e94e90bd30ad856242dab2d94",
    ),
    VISUAL_BACKBONE: (
        1_435_281_181,
        "fc64d2acbabb5c20a3e0bf996906954c81838d8495b7724a9862199c0af4c977",
    ),
    GENERATION_CHECKPOINT: (
        17_615_455_530,
        "e1ff7fef61a480576d52f4c2761ccedca16d8af3ccd6cdc39a83d36fc5a32317",
    ),
}

EXPECTED_DIRECTORY_COUNTS = {
    LLAMA_ROOT: (35, 32_132_574_429),
    BERT_ROOT: (49, 3_454_103_935),
    SIGLIP2_ROOT: (28, 3_568_008_790),
}

COMPONENT_PREFIXES = {
    "llama_model": ("llama_model.",),
    "visual_encoder": ("visual_encoder.",),
    "video_Qformer": ("video_Qformer.",),
    "video_query_tokens": ("video_query_tokens",),
    "ln_vision": ("ln_vision.",),
    "llama_proj": ("llama_proj.",),
    "video_frame_position_embedding": ("video_frame_position_embedding.",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def directory_inventory(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def main() -> int:
    if ASSET_LAYOUT != "nas_bundle_v1" or ASSET_ROOT != EXPECTED_ROOT:
        raise RuntimeError(
            "Set SOCCERMASTER_COMMENTARY_ASSET_ROOT to the frozen NAS bundle"
        )
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    checks: dict[str, object] = {}
    for path, (expected_size, expected_hash) in EXPECTED_FILES.items():
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Identity mismatch for {path}")
        checks[str(path)] = {
            "bytes": expected_size,
            "sha256": sha256(path),
            "expected_sha256": expected_hash,
        }
        if checks[str(path)]["sha256"] != expected_hash:  # type: ignore[index]
            raise RuntimeError(f"SHA-256 mismatch for {path}")

    directories: dict[str, object] = {}
    for path, expected in EXPECTED_DIRECTORY_COUNTS.items():
        observed = directory_inventory(path)
        if observed != expected:
            raise RuntimeError(
                f"Directory inventory mismatch for {path}: {observed} != {expected}"
            )
        directories[str(path)] = {"files": observed[0], "bytes": observed[1]}

    train = json.loads(TRAIN_ANNOTATIONS.read_text(encoding="utf-8"))
    test = json.loads(TEST_ANNOTATIONS.read_text(encoding="utf-8"))
    if len(train) != 24_027 or len(test) != 3_256:
        raise RuntimeError("Annotation record-count mismatch")

    import torch

    checkpoint = torch.load(
        GENERATION_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if checkpoint.get("epoch") != 11:
        raise RuntimeError(f"Unexpected checkpoint epoch: {checkpoint.get('epoch')}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or len(state_dict) != 953:
        raise RuntimeError("Expected checkpoint state_dict with 953 keys")
    keys = tuple(state_dict)
    coverage = {
        name: sum(key.startswith(prefixes) for key in keys)
        for name, prefixes in COMPONENT_PREFIXES.items()
    }
    missing_components = [name for name, count in coverage.items() if count == 0]
    if missing_components:
        raise RuntimeError(f"Missing checkpoint components: {missing_components}")

    result = {
        "status": "passed",
        "scope": "CPU-only NAS identity and mmap checkpoint metadata validation",
        "asset_root": str(ASSET_ROOT),
        "asset_layout": ASSET_LAYOUT,
        "files": checks,
        "directories": directories,
        "annotations": {"train_records": len(train), "test_records": len(test)},
        "checkpoint": {
            "epoch": checkpoint["epoch"],
            "state_key_count": len(state_dict),
            "component_key_counts": coverage,
            "map_location": "cpu",
            "mmap": True,
        },
        "gpu_query_executed": False,
        "forward_executed": False,
        "generate_executed": False,
        "training_executed": False,
    }
    REPORT.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "state_keys": len(state_dict)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
