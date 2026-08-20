#!/usr/bin/env python3
"""Run the bounded two-rank G8 multi-task training verification."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO = Path("/home/tianlin/SoccerMaster")
OPS_BUILD = REPO / "models/deformable_detr/ops/build/lib.linux-x86_64-cpython-310"
MANIFEST_PATH = REPO / "reproduction/manifests/g8_multitask.json"
CONFIG_PATH = REPO / (
    "configs/pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution.yaml"
)
DEFAULT_CONFIG_PATH = REPO / "configs/default.yaml"
MODEL_PATH = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/"
    "pretrained_models/google/siglip2-large-patch16-512"
)
CHECKPOINT_DIR = Path(
    "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
    "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
    "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000_high_resolution/epoch_19"
)
RUN_ID = "20260812_run5"
REPORT_DIR = REPO / "reports/g8" / RUN_ID
RESULT_PATH = REPORT_DIR / "result.json"
OUTPUT_ROOT = REPO / "outputs/g8" / RUN_ID
CHECKPOINT_PATH = OUTPUT_ROOT / "step_000003"
CHECKPOINT_TEMP_PATH = OUTPUT_ROOT / ".step_000003.tmp"
DATA_VIEW_BASE = REPO / ".runtime/data_views/g8" / RUN_ID
EXPECTED_HEADS = (
    "SoccerNetGSR_Detection",
    "LinesDetection",
    "KeypointsDetection",
    "VideoCaption",
    "CaptionClassification",
)
HEARTBEAT_SECONDS = 30
MIN_FREE_GIB = 30
GIB = 1024**3
FUNCTIONAL_RESUME_TOLERANCES = {
    "head_parameters": {"atol": 1e-7, "rtol": 1e-5},
    "optimizer": {"atol": 1e-7, "rtol": 1e-5},
    "gradients": {"atol": 1e-6, "rtol": 1e-4},
}

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OPS_BUILD))
os.chdir(REPO)


class Phase:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self._value = "startup"
        self._lock = threading.Lock()

    def set(self, value: str) -> None:
        with self._lock:
            self._value = value
        print(f"[PHASE] rank={self.rank} {value}", flush=True)

    def get(self) -> str:
        with self._lock:
            return self._value


def heartbeat(stop: threading.Event, started: float, phase: Phase) -> None:
    while not stop.wait(HEARTBEAT_SECONDS):
        print(
            f"[HEARTBEAT] rank={phase.rank} elapsed={time.monotonic()-started:.1f}s "
            f"phase={phase.get()}",
            flush=True,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_files": dirty}


def set_seed(torch: Any, numpy: Any, seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_seed(seed: int, split: str, epoch: int, identifier: str) -> int:
    value = f"{seed}:{split}:{epoch}:{identifier}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def load_manifest() -> dict[str, Any]:
    from reproduction.gates.g8_multitask import load_manifest as validate_manifest

    return validate_manifest()


def ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Existing link has wrong target: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace data-view path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)


def prepare_data_view(manifest: dict[str, Any], rank: int) -> Path:
    source = Path(manifest["source_roots"]["gsr"])
    view = DATA_VIEW_BASE / f"rank_{rank}"
    dataset_root = view / "SN-GSR-2024"
    for logical_split in ("train", "valid"):
        source_split = manifest["detection"][f"{logical_split}_source_split"]
        sequences = sorted(
            {item["sequence"] for item in manifest["detection"][logical_split]}
        )
        for sequence in sequences:
            ensure_symlink(
                dataset_root / "SoccerNetGS" / logical_split / sequence,
                source / "SoccerNetGS" / source_split / sequence,
            )
            ensure_symlink(
                dataset_root / "camera_params" / logical_split / f"{sequence}.json",
                source / "camera_params" / source_split / f"{sequence}.json",
            )
        ensure_symlink(
            dataset_root / "legibility_jn" / f"{logical_split}.json",
            source / "legibility_jn" / "train.json",
        )
    return view


def load_config(manifest: dict[str, Any], data_root: Path) -> dict[str, Any]:
    from configs.util import load_super_config, yaml_to_dict

    config = load_super_config(yaml_to_dict(str(CONFIG_PATH)), str(DEFAULT_CONFIG_PATH))
    config.update(
        {
            "SUPER_CONFIG_PATH": str(DEFAULT_CONFIG_PATH),
            "DEVICE": "cpu",
            "CKPT_TYPE": "soccer_master",
            "LOAD_CHECKPOINTS": False,
            "LOAD_HEADS": True,
            "CKPT_PATH": str(MODEL_PATH),
            "TEXT_ENCODER_CKPT_PATH": str(MODEL_PATH),
            "STAGE_1_CKPT_DIR": str(CHECKPOINT_DIR),
            "DATA_ROOT": str(data_root),
            "USE_EXTRA_DATA": False,
            "EXTRA_DATA_ONLY": False,
            "NUM_WORKERS": 0,
            "VIDEO_CAPTION_NUM_WORKERS": 0,
            "DATASETS_TO_HEADS": {
                "SoccerNetGSR_Detection": [
                    "SoccerNetGSR_Detection", "LinesDetection", "KeypointsDetection"
                ],
                "VideoCaption": ["VideoCaption", "CaptionClassification"],
            },
            "FREEZE_VISION_ENCODER": True,
            "FREEZE_TEXT_ENCODER": True,
            "VIDEO_CAPTION_DATA_ROOT": manifest["source_roots"]["caption"],
            "VIDEO_CAPTION_DATASETS": [manifest["caption"]["dataset"]],
        }
    )
    contract = manifest["training_contract"]
    assertions = {
        "NUM_FRAMES": contract["num_frames"],
        "BACKBONE_TYPE": "video",
        "SIGLIP_BACKBONE_TYPE": "unisoccer_part_temporal",
        "VIDEO_CAPTION_SIGLIP_LOSS_WEIGHT": contract["video_caption_loss_weight"],
    }
    for key, expected in assertions.items():
        if config.get(key) != expected:
            raise AssertionError(f"Unexpected resolved config {key}={config.get(key)!r}")
    if config["AUG_RANDOM_RESIZE"] != [contract["image_size"]]:
        raise AssertionError("G8 image size differs from the high-resolution config")
    return config


def prepare_caption_records(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    root = Path(manifest["source_roots"]["caption"])
    video_root = root / manifest["caption"]["video_subdirectory"]
    records: dict[str, list[dict[str, Any]]] = {}
    relative_names: dict[str, set[str]] = {}
    for split in ("train", "valid"):
        label_path = (
            root / "video_clip_json" / manifest["caption"]["dataset"]
            / f"classification_{split}.json"
        )
        if sha256_file(label_path) != manifest["caption"]["label_sha256"][split]:
            raise AssertionError(f"Caption {split} label SHA256 changed")
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        split_records = []
        for class_spec in manifest["caption"][split]:
            for json_index, expected_bytes in class_spec["index_and_bytes"]:
                item = labels[json_index]
                path = video_root / item["video"]
                if item["caption"] != class_spec["label"]:
                    raise AssertionError(f"Caption label changed at {split}[{json_index}]")
                if not path.exists() or path.stat().st_size != expected_bytes:
                    raise AssertionError(f"Caption video missing or changed: {path}")
                split_records.append(
                    {
                        "split": split,
                        "json_index": int(json_index),
                        "class_index": int(class_spec["class_index"]),
                        "label": class_spec["label"],
                        "text": item["comments_text_anonymized"],
                        "relative_video": item["video"],
                        "video": path,
                        "video_bytes": int(expected_bytes),
                    }
                )
        records[split] = split_records
        relative_names[split] = {item["relative_video"] for item in split_records}
    if relative_names["train"] & relative_names["valid"]:
        raise AssertionError("Caption train/valid leakage")
    return records


def detection_indices(dataset: Any, specs: list[dict[str, Any]]) -> list[int]:
    lookup = {
        (sequence, int(start)): index
        for index, (sequence, start) in enumerate(dataset.sample_position)
    }
    result = []
    for spec in specs:
        key = (spec["sequence"], int(spec["start_frame_zero_based"]))
        if key not in lookup:
            raise AssertionError(f"Detection clip absent from dataset: {key}")
        result.append(lookup[key])
    return result


def rank_order(length: int, seed: int, epoch: int, rank: int, world: int) -> list[int]:
    order = list(range(length))
    random.Random(seed + 1009 * epoch).shuffle(order)
    selected = order[rank::world]
    if len(selected) * world != length:
        raise AssertionError("Fixed records do not divide evenly across ranks")
    return selected


def batches(order: list[int], size: int) -> list[list[int]]:
    if len(order) % size:
        raise AssertionError("Rank sample count does not divide by batch size")
    return [order[index:index + size] for index in range(0, len(order), size)]


def move_nested(torch: Any, value: Any, device: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested(torch, item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested(torch, item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested(torch, item, device) for item in value)
    return value


def make_detection_batch(
    torch: Any, numpy: Any, dataset: Any, index: int, logical_split: str,
    epoch: int, seed: int, device: Any,
) -> tuple[Any, Any, tuple[Any, ...], list[str]]:
    sequence, start = dataset.sample_position[index]
    set_seed(torch, numpy, sample_seed(seed, logical_split, epoch, f"{sequence}:{start}"))
    images, annotations, meta = dataset[index]
    if tuple(images.shape) != (30, 3, 512, 512):
        raise AssertionError(f"Unexpected detection input shape {tuple(images.shape)}")
    if meta["sequence"] != sequence or int(meta["start_frame"]) != int(start):
        raise AssertionError("Detection dataset returned the wrong fixed clip")
    return (
        images.unsqueeze(0).to(device),
        [move_nested(torch, annotations, device)],
        (meta,),
        [f"{logical_split}:{sequence}:{start}"],
    )


def make_caption_batch(
    torch: Any, numpy: Any, records: list[dict[str, Any]], indices: list[int],
    split: str, epoch: int, seed: int, config: dict[str, Any], transform: Any,
    read_frames: Any, device: Any,
) -> tuple[Any, list[dict[str, Any]], tuple[dict[str, Any], ...], list[str], list[str]]:
    images, annotations, metas, identifiers, texts = [], [], [], [], []
    for index in indices:
        record = records[index]
        set_seed(
            torch, numpy,
            sample_seed(seed, split, epoch, f"caption:{record['json_index']}"),
        )
        frames, frame_indices, _ = read_frames(
            str(record["video"]), config["NUM_FRAMES"], config["VIDEO_CAPTION_SAMPLE"],
            config["VIDEO_CAPTION_FIX_START"], config["VIDEO_CAPTION_MAX_NUM_FRAMES"],
            config["VIDEO_CAPTION_TRIMMED30"],
        )
        annotation = {
            "caption": record["label"],
            "caption_index": torch.tensor(record["class_index"], dtype=torch.long),
            "text": record["text"],
        }
        meta = {"task": "VideoCaption", "video": str(record["video"])}
        image, annotation, meta = transform(frames, annotation, meta)
        if tuple(image.shape) != (30, 3, 512, 512):
            raise AssertionError(f"Unexpected caption input shape {tuple(image.shape)}")
        meta["frame_indices"] = [int(value) for value in frame_indices]
        images.append(image)
        annotations.append(move_nested(torch, annotation, device))
        metas.append(meta)
        identifiers.append(f"{split}:{record['json_index']}")
        texts.append(record["text"])
    return torch.stack(images).to(device), annotations, tuple(metas), identifiers, texts


def weighted_losses(
    torch: Any, outputs: dict[str, Any], annotations: Any, metas: Any,
    heads: list[str], loss_functions: dict[str, Any],
) -> tuple[Any, dict[str, dict[str, float]]]:
    total = None
    report: dict[str, dict[str, float]] = {}
    for head in heads:
        if head == "VideoCaption":
            value = loss_functions[head](outputs[head], annotations, metas)
        else:
            value = loss_functions[head](outputs[head], annotations)
        raw, weights = value[0], value[1]
        weighted = {key: tensor * weights[key] for key, tensor in raw.items() if key in weights}
        head_total = sum(weighted.values())
        if not bool(torch.isfinite(head_total).item()):
            raise FloatingPointError(f"Non-finite loss for {head}")
        total = head_total if total is None else total + head_total
        report[head] = {
            "raw_sum": float(sum(raw[key] for key in weighted).detach().item()),
            "weighted_sum": float(head_total.detach().item()),
        }
    if total is None:
        raise AssertionError("No losses were produced")
    return total, report


def optimizer_for_heads(torch: Any, model: Any, config: dict[str, Any]) -> Any:
    mapping = {
        "SoccerNetGSR_Detection": config["LR_SOCCERNET_GSR_DETECTION"],
        "LinesDetection": config["LR_LINES_DETECTION"],
        "KeypointsDetection": config["LR_KEYPOINTS_DETECTION"],
        "VideoCaption": config["LR_VIDEO_CAPTION"],
        "CaptionClassification": config["LR_CAPTION_CLASSIFICATION"],
    }
    groups = []
    for name in EXPECTED_HEADS:
        head = model.multi_task_head[name]
        if name == "CaptionClassification":
            classifier, other = [], []
            for parameter_name, parameter in head.named_parameters():
                (classifier if "classifier" in parameter_name.lower() else other).append(parameter)
            if other:
                groups.append({"params": other, "lr": mapping[name], "name": f"{name}_other"})
            if classifier:
                groups.append(
                    {
                        "params": classifier,
                        "lr": config["LR_CAPTION_CLASSIFICATION_CLASSIFIER"],
                        "name": f"{name}_classifier",
                    }
                )
        else:
            groups.append({"params": list(head.parameters()), "lr": mapping[name], "name": name})
    return torch.optim.AdamW(groups, weight_decay=config["WEIGHT_DECAY"])


def trainable_parameters(model: Any) -> list[Any]:
    return [parameter for head in model.multi_task_head.values() for parameter in head.parameters()]


def sync_gradients(torch: Any, dist: Any, model: Any, world: int) -> dict[str, float]:
    norms: dict[str, float] = {}
    for head_name in EXPECTED_HEADS:
        square = torch.zeros((), device=torch.cuda.current_device())
        for parameter in model.multi_task_head[head_name].parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world)
            square += parameter.grad.detach().float().pow(2).sum()
        norm = float(square.sqrt().item())
        if not (norm > 0.0):
            raise AssertionError(f"Head has zero synchronized gradient: {head_name}")
        norms[head_name] = norm
    return norms


def state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for head_name in EXPECTED_HEADS:
        for name, tensor in sorted(model.multi_task_head[head_name].named_parameters()):
            digest.update(f"{head_name}:{name}".encode("utf-8"))
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(optimizer: Any) -> str:
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(f"group:{group_index}:{group.get('name', '')}".encode("utf-8"))
        for key in ("lr", "weight_decay", "betas", "eps"):
            digest.update(f"{key}:{group.get(key)!r}".encode("utf-8"))
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key, value in sorted(state.items()):
                digest.update(f"{parameter_index}:{key}".encode("utf-8"))
                if hasattr(value, "detach"):
                    digest.update(value.detach().contiguous().cpu().numpy().tobytes())
                else:
                    digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def parameter_snapshot(model: Any) -> dict[str, dict[str, Any]]:
    return {
        head_name: {
            name: parameter.detach().contiguous().cpu().clone()
            for name, parameter in model.multi_task_head[head_name].named_parameters()
        }
        for head_name in EXPECTED_HEADS
    }


def gradient_snapshot(model: Any) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for head_name in EXPECTED_HEADS:
        values = {}
        for name, parameter in model.multi_task_head[head_name].named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"Missing gradient during diagnostic capture: {head_name}.{name}")
            values[name] = parameter.grad.detach().contiguous().cpu().clone()
        snapshot[head_name] = values
    return snapshot


def snapshot_hashes(snapshot: dict[str, dict[str, Any]]) -> dict[str, str]:
    hashes = {}
    for group_name, values in snapshot.items():
        digest = hashlib.sha256()
        for name, tensor in sorted(values.items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.contiguous().numpy().tobytes())
        hashes[group_name] = digest.hexdigest()
    return hashes


def optimizer_snapshot(optimizer: Any) -> list[dict[str, Any]]:
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        states = []
        for parameter_index, parameter in enumerate(group["params"]):
            item = {}
            for key, value in optimizer.state.get(parameter, {}).items():
                item[key] = (
                    value.detach().contiguous().cpu().clone()
                    if hasattr(value, "detach")
                    else copy.deepcopy(value)
                )
            states.append({"parameter_index": parameter_index, "state": item})
        options = {
            key: copy.deepcopy(value)
            for key, value in group.items()
            if key != "params"
        }
        groups.append(
            {
                "group_index": group_index,
                "name": group.get("name", f"group_{group_index}"),
                "options": options,
                "states": states,
            }
        )
    return groups


def tensor_difference(
    torch: Any, reference: Any, candidate: Any, *, atol: float, rtol: float,
) -> dict[str, Any]:
    candidate_cpu = candidate.detach().contiguous().cpu()
    result = {
        "exact": False,
        "shape_equal": tuple(reference.shape) == tuple(candidate_cpu.shape),
        "dtype_equal": reference.dtype == candidate_cpu.dtype,
        "max_abs": None,
        "max_rel": None,
        "within_tolerance": False,
    }
    if not result["shape_equal"] or not result["dtype_equal"]:
        return result
    result["exact"] = bool(torch.equal(reference, candidate_cpu))
    if result["exact"]:
        result["max_abs"] = 0.0
        result["max_rel"] = 0.0
        result["within_tolerance"] = True
        return result
    if reference.is_floating_point() or reference.is_complex():
        reference_float = reference.detach().float()
        candidate_float = candidate_cpu.detach().float()
        absolute = (reference_float - candidate_float).abs()
        denominator = torch.maximum(reference_float.abs(), candidate_float.abs()).clamp_min(1e-30)
        result["max_abs"] = float(absolute.max().item())
        result["max_rel"] = float((absolute / denominator).max().item())
        result["within_tolerance"] = bool(
            torch.allclose(reference, candidate_cpu, atol=atol, rtol=rtol)
        )
    return result


def compare_named_tensor_snapshots(
    torch: Any,
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    overall_exact = True
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    overall_within_tolerance = True
    first_difference = None
    for group_name in reference:
        group_result = {
            "exact": True,
            "differing_tensors": 0,
            "first_difference": None,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "within_tolerance": True,
        }
        if set(reference[group_name]) != set(candidate[group_name]):
            raise AssertionError(f"Diagnostic tensor names changed for {group_name}")
        for name in sorted(reference[group_name]):
            difference = tensor_difference(
                torch,
                reference[group_name][name],
                candidate[group_name][name],
                atol=atol,
                rtol=rtol,
            )
            if not difference["exact"]:
                group_result["exact"] = False
                group_result["differing_tensors"] += 1
                if group_result["first_difference"] is None:
                    group_result["first_difference"] = {"name": name, **difference}
                if first_difference is None:
                    first_difference = {"group": group_name, "name": name, **difference}
            if difference["max_abs"] is not None:
                group_result["max_abs"] = max(group_result["max_abs"], difference["max_abs"])
                overall_max_abs = max(overall_max_abs, difference["max_abs"])
            if difference["max_rel"] is not None:
                group_result["max_rel"] = max(group_result["max_rel"], difference["max_rel"])
                overall_max_rel = max(overall_max_rel, difference["max_rel"])
            group_result["within_tolerance"] = (
                group_result["within_tolerance"] and difference["within_tolerance"]
            )
        groups[group_name] = group_result
        overall_exact = overall_exact and group_result["exact"]
        overall_within_tolerance = (
            overall_within_tolerance and group_result["within_tolerance"]
        )
    return {
        "exact": overall_exact,
        "first_difference": first_difference,
        "max_abs": overall_max_abs,
        "max_rel": overall_max_rel,
        "within_tolerance": overall_within_tolerance,
        "atol": atol,
        "rtol": rtol,
        "groups": groups,
    }


def live_parameter_view(model: Any) -> dict[str, dict[str, Any]]:
    return {
        head_name: dict(model.multi_task_head[head_name].named_parameters())
        for head_name in EXPECTED_HEADS
    }


def compare_optimizer_snapshot(
    torch: Any, reference: list[dict[str, Any]], optimizer: Any, *, atol: float, rtol: float,
) -> dict[str, Any]:
    if len(reference) != len(optimizer.param_groups):
        raise AssertionError("Optimizer group count changed during diagnostic")
    overall_exact = True
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    overall_within_tolerance = True
    first_difference = None
    groups = []
    for reference_group, current_group in zip(reference, optimizer.param_groups):
        group_result = {
            "group_index": reference_group["group_index"],
            "name": reference_group["name"],
            "exact": True,
            "differing_states": 0,
            "first_difference": None,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "options_equal": all(
                current_group.get(key) == value
                for key, value in reference_group["options"].items()
            ),
            "within_tolerance": True,
        }
        if not group_result["options_equal"]:
            group_result["exact"] = False
            group_result["within_tolerance"] = False
            group_result["first_difference"] = {"kind": "group_options"}
        if len(reference_group["states"]) != len(current_group["params"]):
            raise AssertionError(f"Optimizer parameter count changed for {reference_group['name']}")
        for state_spec, parameter in zip(reference_group["states"], current_group["params"]):
            reference_state = state_spec["state"]
            current_state = optimizer.state.get(parameter, {})
            if set(reference_state) != set(current_state):
                raise AssertionError(
                    f"Optimizer state keys changed for {reference_group['name']} "
                    f"parameter {state_spec['parameter_index']}"
                )
            for state_key in sorted(reference_state):
                reference_value = reference_state[state_key]
                current_value = current_state[state_key]
                if hasattr(reference_value, "detach") and hasattr(current_value, "detach"):
                    difference = tensor_difference(
                        torch, reference_value, current_value, atol=atol, rtol=rtol
                    )
                    exact = difference["exact"]
                else:
                    exact = reference_value == current_value
                    difference = {
                        "exact": exact,
                        "shape_equal": None,
                        "dtype_equal": None,
                        "max_abs": None,
                        "max_rel": None,
                        "within_tolerance": exact,
                    }
                if not exact:
                    group_result["exact"] = False
                    group_result["differing_states"] += 1
                    item = {
                        "parameter_index": state_spec["parameter_index"],
                        "state_key": state_key,
                        **difference,
                    }
                    if group_result["first_difference"] is None:
                        group_result["first_difference"] = item
                    if first_difference is None:
                        first_difference = {"group": reference_group["name"], **item}
                if difference["max_abs"] is not None:
                    group_result["max_abs"] = max(group_result["max_abs"], difference["max_abs"])
                    overall_max_abs = max(overall_max_abs, difference["max_abs"])
                if difference["max_rel"] is not None:
                    group_result["max_rel"] = max(group_result["max_rel"], difference["max_rel"])
                    overall_max_rel = max(overall_max_rel, difference["max_rel"])
                group_result["within_tolerance"] = (
                    group_result["within_tolerance"] and difference["within_tolerance"]
                )
        groups.append(group_result)
        overall_exact = overall_exact and group_result["exact"]
        overall_within_tolerance = (
            overall_within_tolerance and group_result["within_tolerance"]
        )
    return {
        "exact": overall_exact,
        "first_difference": first_difference,
        "max_abs": overall_max_abs,
        "max_rel": overall_max_rel,
        "within_tolerance": overall_within_tolerance,
        "atol": atol,
        "rtol": rtol,
        "groups": groups,
    }


def assert_rank_state_equal(dist: Any, model: Any, world: int) -> str:
    local = state_digest(model)
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    if len(set(gathered)) != 1:
        raise AssertionError(f"Head state differs across ranks: {gathered}")
    return local


def rng_state(torch: Any, numpy: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(torch: Any, numpy: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def save_checkpoint(
    torch: Any, dist: Any, model: Any, optimizer: Any, scheduler: Any,
    rank: int, world: int, progress: dict[str, int], manifest_sha: str,
) -> None:
    states = [None for _ in range(world)]
    dist.all_gather_object(states, rng_state(torch, __import__("numpy")))
    if rank == 0:
        if CHECKPOINT_PATH.exists() or CHECKPOINT_TEMP_PATH.exists():
            raise FileExistsError("Refusing to overwrite G8 checkpoint")
        CHECKPOINT_TEMP_PATH.mkdir(parents=True)
        torch.save(
            {name: model.multi_task_head[name].state_dict() for name in EXPECTED_HEADS},
            CHECKPOINT_TEMP_PATH / "heads.pt",
        )
        torch.save(optimizer.state_dict(), CHECKPOINT_TEMP_PATH / "optimizer.pt")
        torch.save(scheduler.state_dict(), CHECKPOINT_TEMP_PATH / "scheduler.pt")
        torch.save(states, CHECKPOINT_TEMP_PATH / "rng_by_rank.pt")
        (CHECKPOINT_TEMP_PATH / "training_state.json").write_text(
            json.dumps(progress, sort_keys=True) + "\n", encoding="utf-8"
        )
        required = ["heads.pt", "optimizer.pt", "scheduler.pt", "rng_by_rank.pt", "training_state.json"]
        files = []
        for name in required:
            path = CHECKPOINT_TEMP_PATH / name
            files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        document = {
            "schema_version": 1,
            "gate": "G8",
            "manifest_sha256": manifest_sha,
            "world_size": world,
            "files": files,
        }
        (CHECKPOINT_TEMP_PATH / "manifest.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (CHECKPOINT_TEMP_PATH / "COMPLETE").write_text("complete\n", encoding="utf-8")
        CHECKPOINT_TEMP_PATH.rename(CHECKPOINT_PATH)
    dist.barrier()


def load_checkpoint(
    torch: Any, dist: Any, model: Any, optimizer: Any, scheduler: Any,
    rank: int, manifest_sha: str,
) -> dict[str, Any]:
    if not (CHECKPOINT_PATH / "COMPLETE").is_file():
        raise FileNotFoundError("G8 checkpoint COMPLETE marker missing")
    document = json.loads((CHECKPOINT_PATH / "manifest.json").read_text(encoding="utf-8"))
    if document["manifest_sha256"] != manifest_sha:
        raise AssertionError("G8 checkpoint manifest identity changed")
    for item in document["files"]:
        path = CHECKPOINT_PATH / item["name"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise AssertionError(f"G8 checkpoint integrity failed: {path}")
    heads = torch.load(CHECKPOINT_PATH / "heads.pt", map_location="cpu")
    for name in EXPECTED_HEADS:
        model.multi_task_head[name].load_state_dict(heads[name], strict=True)
    optimizer.load_state_dict(torch.load(CHECKPOINT_PATH / "optimizer.pt", map_location="cpu"))
    scheduler.load_state_dict(torch.load(CHECKPOINT_PATH / "scheduler.pt", map_location="cpu"))
    states = torch.load(CHECKPOINT_PATH / "rng_by_rank.pt", map_location="cpu")
    restore_rng(torch, __import__("numpy"), states[rank])
    dist.barrier()
    return json.loads((CHECKPOINT_PATH / "training_state.json").read_text(encoding="utf-8"))


def run_step(
    torch: Any, numpy: Any, dist: Any, model: Any, optimizer: Any,
    loss_functions: dict[str, Any], detection_dataset: Any, detection_index: int,
    caption_records: list[dict[str, Any]], caption_indices: list[int], config: dict[str, Any],
    caption_transform: Any, read_frames: Any, device: Any, seed: int, epoch: int,
    rank: int, world: int, capture_gradient_diagnostic: bool = False,
) -> dict[str, Any]:
    model.train()
    model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    images, annotations, metas, detection_ids = make_detection_batch(
        torch, numpy, detection_dataset, detection_index, "train", epoch, seed, device
    )
    detection_input_sha = tensor_sha256(images)
    outputs = model(images, "SoccerNetGSR_Detection", metas=metas, text=None)
    detection_loss, detection_report = weighted_losses(
        torch, outputs, annotations, metas,
        ["SoccerNetGSR_Detection", "LinesDetection", "KeypointsDetection"], loss_functions,
    )
    detection_loss.backward()
    del images, annotations, outputs
    images, annotations, metas, caption_ids, texts = make_caption_batch(
        torch, numpy, caption_records, caption_indices, "train", epoch, seed,
        config, caption_transform, read_frames, device,
    )
    caption_input_sha = tensor_sha256(images)
    outputs = model(images, "VideoCaption", metas=metas, text=texts)
    expected_global = world * len(caption_indices)
    if tuple(outputs["VideoCaption"]["vision_features"].shape[:1]) != (expected_global,):
        raise AssertionError("VideoCaption distributed gather size changed")
    if tuple(outputs["VideoCaption"]["text_features"].shape[:1]) != (expected_global,):
        raise AssertionError("VideoCaption distributed text gather size changed")
    if tuple(outputs["VideoCaption"]["valid_text_mask"].shape) != (expected_global,):
        raise AssertionError("VideoCaption distributed mask gather size changed")
    caption_loss, caption_report = weighted_losses(
        torch, outputs, annotations, metas,
        ["VideoCaption", "CaptionClassification"], loss_functions,
    )
    caption_loss.backward()
    del images, annotations, outputs
    gradients = sync_gradients(torch, dist, model, world)
    total_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters(model), config["MAX_CLIP_NORM"]
    )
    if not bool(torch.isfinite(total_norm).item()):
        raise FloatingPointError("Non-finite total gradient norm")
    captured_gradients = gradient_snapshot(model) if capture_gradient_diagnostic else None
    gradient_hashes = snapshot_hashes(captured_gradients) if captured_gradients is not None else None
    optimizer.step()
    dist.barrier()
    digest = assert_rank_state_equal(dist, model, world)
    return {
        "rank": rank,
        "detection_ids": detection_ids,
        "caption_ids": caption_ids,
        "detection_input_sha256": detection_input_sha,
        "caption_input_sha256": caption_input_sha,
        "losses": {**detection_report, **caption_report},
        "gradient_norms": gradients,
        "gradient_sha256_by_head": gradient_hashes,
        "clipped_total_norm": float(total_norm.item()),
        "head_state_sha256": digest,
        "_gradient_snapshot": captured_gradients,
    }


def evaluate(
    torch: Any, numpy: Any, dist: Any, model: Any, loss_functions: dict[str, Any],
    detection_dataset: Any, detection_indices_all: list[int], caption_records: list[dict[str, Any]],
    config: dict[str, Any], caption_transform: Any, read_frames: Any, device: Any,
    seed: int, epoch: int, rank: int, world: int,
) -> dict[str, Any]:
    model.eval()
    sums = {name: 0.0 for name in EXPECTED_HEADS}
    counts = {name: 0 for name in EXPECTED_HEADS}
    correct, classification_count = 0, 0
    detection_rank = rank_order(len(detection_indices_all), seed + 17, epoch, rank, world)
    caption_rank = rank_order(len(caption_records), seed + 29, epoch, rank, world)
    with torch.no_grad():
        for position in detection_rank:
            images, annotations, metas, _ = make_detection_batch(
                torch, numpy, detection_dataset, detection_indices_all[position],
                "valid", epoch, seed, device,
            )
            outputs = model(images, "SoccerNetGSR_Detection", metas=metas, text=None)
            _, report = weighted_losses(
                torch, outputs, annotations, metas,
                ["SoccerNetGSR_Detection", "LinesDetection", "KeypointsDetection"],
                loss_functions,
            )
            for name, value in report.items():
                sums[name] += value["weighted_sum"]
                counts[name] += 1
        for batch_indices in batches(caption_rank, 2):
            images, annotations, metas, _, texts = make_caption_batch(
                torch, numpy, caption_records, batch_indices, "valid", epoch, seed,
                config, caption_transform, read_frames, device,
            )
            outputs = model(images, "VideoCaption", metas=metas, text=texts)
            _, report = weighted_losses(
                torch, outputs, annotations, metas,
                ["VideoCaption", "CaptionClassification"], loss_functions,
            )
            for name, value in report.items():
                sums[name] += value["weighted_sum"]
                counts[name] += 1
            predictions = outputs["CaptionClassification"]["logits"].argmax(dim=-1)
            labels = torch.stack([item["caption_index"] for item in annotations])
            correct += int((predictions == labels).sum().item())
            classification_count += len(annotations)
    payload = torch.tensor(
        [item for name in EXPECTED_HEADS for item in (sums[name], counts[name])]
        + [correct, classification_count], dtype=torch.float64, device=device,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    result: dict[str, Any] = {}
    offset = 0
    for name in EXPECTED_HEADS:
        total, count = float(payload[offset].item()), int(payload[offset + 1].item())
        if count <= 0:
            raise AssertionError(f"Missing validation metric for {name}")
        result[name] = {"weighted_loss_mean": total / count, "batches": count}
        offset += 2
    global_correct = int(payload[offset].item())
    global_count = int(payload[offset + 1].item())
    result["CaptionClassification"]["accuracy"] = global_correct / global_count
    result["CaptionClassification"]["samples"] = global_count
    return result


def main() -> int:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "-1"))
    started = time.monotonic()
    stop = threading.Event()
    phase = Phase(rank)
    thread = threading.Thread(target=heartbeat, args=(stop, started, phase), daemon=True)
    thread.start()
    results: dict[str, Any] = {
        "gate": "G8", "run_id": RUN_ID, "status": "failed", "rank": rank,
        "started_monotonic": started, "error": None,
    }
    torch = None
    dist = None
    try:
        if world != 2 or local_rank not in (0, 1) or rank not in (0, 1):
            raise RuntimeError(f"G8 requires exactly two ranks; rank={rank} local={local_rank} world={world}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
            raise RuntimeError("G8 run5 requires external CUDA_VISIBLE_DEVICES=0,1")
        if RESULT_PATH.exists() or CHECKPOINT_PATH.exists() or CHECKPOINT_TEMP_PATH.exists():
            raise FileExistsError("Refusing to overwrite G8 run5 artifacts")
        phase.set("import_framework")
        import numpy
        import torch
        import torch.distributed as dist
        from data.soccernet_gsr_detection import build_gsr_detection_dataset
        from data.video_caption import build_transforms as build_caption_transform
        from data.video_caption import read_frames_decord
        from models.build import build_loss_fn
        from models.multi_task import MultiTaskingSigLIP

        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("G8 requires exactly two visible CUDA devices")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if dist.is_initialized():
            if dist.get_backend() != "nccl":
                raise RuntimeError(
                    f"Existing process group backend is {dist.get_backend()!r}, expected 'nccl'"
                )
            if dist.get_rank() != rank or dist.get_world_size() != world:
                raise RuntimeError(
                    "Existing process group identity differs from torchrun environment: "
                    f"dist_rank={dist.get_rank()} rank={rank} "
                    f"dist_world={dist.get_world_size()} world={world}"
                )
            print(
                f"[DISTRIBUTED] rank={rank} reusing existing NCCL process group",
                flush=True,
            )
        else:
            dist.init_process_group(backend="nccl")
            print(
                f"[DISTRIBUTED] rank={rank} initialized NCCL process group",
                flush=True,
            )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes < MIN_FREE_GIB * GIB:
            raise RuntimeError(f"Rank {rank} has less than {MIN_FREE_GIB} GiB free")
        results["environment"] = {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "cuda_build": torch.version.cuda,
            "visible_physical_gpus": [0, 1],
            "logical_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "free_bytes_at_start": int(free_bytes),
            "total_bytes": int(total_bytes),
            "dtype": "float32",
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        }
        phase.set("manifest_and_assets")
        manifest = load_manifest()
        contract = manifest["training_contract"]
        seed = int(contract["seed"])
        set_seed(torch, numpy, seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        data_view = prepare_data_view(manifest, rank)
        config = load_config(manifest, data_view)
        caption_records = prepare_caption_records(manifest)
        phase.set("build_datasets")
        detection_train = build_gsr_detection_dataset(config=config, split="train")
        detection_valid = build_gsr_detection_dataset(config=config, split="valid")
        detection_train_indices = detection_indices(detection_train, manifest["detection"]["train"])
        detection_valid_indices = detection_indices(detection_valid, manifest["detection"]["valid"])
        caption_train_transform = build_caption_transform(config, "train")
        caption_valid_transform = build_caption_transform(config, "valid")
        phase.set("build_model_cpu")
        model = MultiTaskingSigLIP(config=config)
        phase.set("load_epoch19_cpu")
        load_result = model.load_checkpoint(
            str(CHECKPOINT_DIR), "soccer_master", logger=None, load_heads=True
        )
        results["checkpoint_load_return"] = str(load_result)
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for head in model.multi_task_head.values():
            for parameter in head.parameters():
                parameter.requires_grad_(True)
        initial_heads = {
            name: {
                key: value.detach().cpu().clone()
                for key, value in model.multi_task_head[name].named_parameters()
            }
            for name in EXPECTED_HEADS
        }
        phase.set("move_model_gpu")
        model.to(device)
        frozen_versions = {
            name: parameter._version for name, parameter in model.backbone.named_parameters()
        }
        loss_functions = build_loss_fn(config)
        for function in loss_functions.values():
            function.to(device)
        optimizer = optimizer_for_heads(torch, model, config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=contract["epochs"], eta_min=config["SCHEDULER_MIN_LR"]
        )
        results["git"] = git_identity()
        results["manifest_sha256"] = sha256_file(MANIFEST_PATH)
        results["steps"] = []
        results["validation"] = []
        gradient_seen = {name: False for name in EXPECTED_HEADS}
        global_step = 0
        exact_resume = None
        for epoch in range(contract["epochs"]):
            detection_order = rank_order(
                len(detection_train_indices), seed, epoch, rank, world
            )
            caption_order = rank_order(len(caption_records["train"]), seed, epoch, rank, world)
            caption_batches = batches(caption_order, contract["caption_batch_size"])
            if len(detection_order) != contract["steps_per_epoch"] or len(caption_batches) != contract["steps_per_epoch"]:
                raise AssertionError("G8 rank schedule does not contain four steps")
            for step_in_epoch in range(contract["steps_per_epoch"]):
                phase.set(f"train epoch={epoch} step={step_in_epoch} global={global_step+1}")
                capture_resume_diagnostic = (
                    global_step + 1
                    == contract["resume_probe_after_optimizer_step"] + 1
                )
                step_result = run_step(
                    torch, numpy, dist, model, optimizer, loss_functions,
                    detection_train, detection_train_indices[detection_order[step_in_epoch]],
                    caption_records["train"], caption_batches[step_in_epoch], config,
                    caption_train_transform, read_frames_decord, device, seed, epoch, rank, world,
                    capture_gradient_diagnostic=capture_resume_diagnostic,
                )
                reference_gradient_snapshot = step_result.pop("_gradient_snapshot")
                global_step += 1
                for name, norm in step_result["gradient_norms"].items():
                    gradient_seen[name] = gradient_seen[name] or norm > 0
                gathered_steps = [None for _ in range(world)]
                dist.all_gather_object(gathered_steps, step_result)
                if rank == 0:
                    results["steps"].extend(gathered_steps)
                if global_step == contract["resume_probe_after_optimizer_step"]:
                    phase.set("checkpoint_step_3")
                    save_checkpoint(
                        torch, dist, model, optimizer, scheduler, rank, world,
                        {"epoch": epoch, "completed_step_in_epoch": step_in_epoch + 1, "global_step": global_step},
                        results["manifest_sha256"],
                    )
                elif global_step == contract["resume_probe_after_optimizer_step"] + 1:
                    reference = step_result
                    reference_digest = state_digest(model)
                    reference_optimizer_digest = optimizer_digest(optimizer)
                    reference_scheduler_state = dict(scheduler.state_dict())
                    reference_parameter_snapshot = parameter_snapshot(model)
                    reference_optimizer_snapshot = optimizer_snapshot(optimizer)
                    if reference_gradient_snapshot is None:
                        raise AssertionError("Reference gradient diagnostic was not captured")
                    phase.set("restore_step_3_and_repeat_step_4")
                    progress = load_checkpoint(
                        torch, dist, model, optimizer, scheduler, rank,
                        results["manifest_sha256"],
                    )
                    resumed = run_step(
                        torch, numpy, dist, model, optimizer, loss_functions,
                        detection_train, detection_train_indices[detection_order[step_in_epoch]],
                        caption_records["train"], caption_batches[step_in_epoch], config,
                        caption_train_transform, read_frames_decord, device, seed, epoch, rank, world,
                        capture_gradient_diagnostic=True,
                    )
                    resumed_gradient_snapshot = resumed.pop("_gradient_snapshot")
                    if resumed_gradient_snapshot is None:
                        raise AssertionError("Resumed gradient diagnostic was not captured")
                    resumed_digest = state_digest(model)
                    resumed_optimizer_digest = optimizer_digest(optimizer)
                    parameter_difference = compare_named_tensor_snapshots(
                        torch,
                        reference_parameter_snapshot,
                        live_parameter_view(model),
                        **FUNCTIONAL_RESUME_TOLERANCES["head_parameters"],
                    )
                    gradient_difference = compare_named_tensor_snapshots(
                        torch,
                        reference_gradient_snapshot,
                        resumed_gradient_snapshot,
                        **FUNCTIONAL_RESUME_TOLERANCES["gradients"],
                    )
                    optimizer_difference = compare_optimizer_snapshot(
                        torch,
                        reference_optimizer_snapshot,
                        optimizer,
                        **FUNCTIONAL_RESUME_TOLERANCES["optimizer"],
                    )
                    exact_resume = {
                        "saved_progress": progress,
                        "input_ids_equal": (
                            reference["detection_ids"] == resumed["detection_ids"]
                            and reference["caption_ids"] == resumed["caption_ids"]
                        ),
                        "input_sha256_equal": (
                            reference["detection_input_sha256"] == resumed["detection_input_sha256"]
                            and reference["caption_input_sha256"] == resumed["caption_input_sha256"]
                        ),
                        "losses_equal": reference["losses"] == resumed["losses"],
                        "gradient_hashes_equal": (
                            reference["gradient_sha256_by_head"]
                            == resumed["gradient_sha256_by_head"]
                        ),
                        "head_state_equal": reference_digest == resumed_digest,
                        "optimizer_state_equal": (
                            reference_optimizer_digest == resumed_optimizer_digest
                        ),
                        "scheduler_state_equal": (
                            reference_scheduler_state == scheduler.state_dict()
                        ),
                        "diagnostic": {
                            "gradients": gradient_difference,
                            "head_parameters": parameter_difference,
                            "optimizer": optimizer_difference,
                        },
                        "functional_tolerances": FUNCTIONAL_RESUME_TOLERANCES,
                    }
                    required_resume_checks = (
                        "input_ids_equal",
                        "input_sha256_equal",
                        "losses_equal",
                        "gradient_hashes_equal",
                        "head_state_equal",
                        "optimizer_state_equal",
                        "scheduler_state_equal",
                    )
                    exact_resume["passed"] = all(
                        exact_resume[key] for key in required_resume_checks
                    )
                    required_functional_checks = (
                        "input_ids_equal",
                        "input_sha256_equal",
                        "losses_equal",
                        "scheduler_state_equal",
                    )
                    exact_resume["functional_resume_passed"] = (
                        all(exact_resume[key] for key in required_functional_checks)
                        and gradient_difference["within_tolerance"]
                        and parameter_difference["within_tolerance"]
                        and optimizer_difference["within_tolerance"]
                    )
                    exact_resume["verdict"] = (
                        "exact_resume"
                        if exact_resume["passed"]
                        else (
                            "functional_resume"
                            if exact_resume["functional_resume_passed"]
                            else "failed"
                        )
                    )
                    results["exact_resume_diagnostic"] = exact_resume
                    if not exact_resume["functional_resume_passed"]:
                        raise AssertionError(
                            f"G8 functional-resume probe exceeded tolerance: {exact_resume}"
                        )
            scheduler.step()
            phase.set(f"validate epoch={epoch}")
            metrics = evaluate(
                torch, numpy, dist, model, loss_functions, detection_valid,
                detection_valid_indices, caption_records["valid"], config,
                caption_valid_transform, read_frames_decord, device, seed, epoch, rank, world,
            )
            if rank == 0:
                results["validation"].append({"epoch": epoch, "metrics": metrics})
        phase.set("final_assertions")
        if global_step != contract["epochs"] * contract["steps_per_epoch"]:
            raise AssertionError("G8 global step count changed")
        if not all(gradient_seen.values()):
            raise AssertionError(f"Some heads never received nonzero gradients: {gradient_seen}")
        for name, parameter in model.backbone.named_parameters():
            if parameter._version != frozen_versions[name] or parameter.grad is not None:
                raise AssertionError(f"Frozen backbone changed or received gradient: {name}")
        changed = {}
        for head_name in EXPECTED_HEADS:
            changed[head_name] = any(
                not torch.equal(initial_heads[head_name][key], value.detach().cpu())
                for key, value in model.multi_task_head[head_name].named_parameters()
            )
        if not all(changed.values()):
            raise AssertionError(f"Some heads did not update: {changed}")
        final_digest = assert_rank_state_equal(dist, model, world)
        torch.cuda.synchronize(device)
        local_memory = {
            "rank": rank,
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        all_memory = [None for _ in range(world)]
        dist.all_gather_object(all_memory, local_memory)
        if exact_resume is None:
            raise AssertionError("G8 exact-resume probe did not run")
        if rank == 0:
            detection_ids = [identifier for step in results["steps"] for identifier in step["detection_ids"]]
            caption_ids = [identifier for step in results["steps"] for identifier in step["caption_ids"]]
            expected_detection_positions = contract["epochs"] * len(manifest["detection"]["train"])
            expected_caption_positions = contract["epochs"] * sum(
                len(item["index_and_bytes"]) for item in manifest["caption"]["train"]
            )
            if len(detection_ids) != expected_detection_positions:
                raise AssertionError("G8 did not process every fixed detection position")
            if len(caption_ids) != expected_caption_positions:
                raise AssertionError("G8 did not process every fixed caption position")
        results["assertions"] = {
            "all_fixed_train_steps_processed": True,
            "all_five_heads_finite_nonzero_gradients": all(gradient_seen.values()),
            "all_five_heads_updated": all(changed.values()),
            "frozen_backbone_and_text_unchanged": True,
            "rank_head_parameters_equal": True,
            "video_caption_gather_shape_equal": True,
            "exact_resume": exact_resume,
            "functional_resume_passed": exact_resume["functional_resume_passed"],
            "validation_metric_structure_complete": len(results["validation"]) == 2,
            "test_split_accessed": False,
            "fallback_used": False,
            "determinism_compatibility_exception": (
                "torch deterministic algorithms enabled with warn_only=True because "
                "PyTorch 2.4.1 has no deterministic CUDA cumsum implementation"
            ),
        }
        results["final_head_state_sha256"] = final_digest
        results["gpu_memory"] = all_memory
        results["elapsed_seconds"] = time.monotonic() - started
        results["status"] = "passed"
        if rank == 0:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            RESULT_PATH.write_text(
                json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)
        dist.barrier()
        return 0
    except BaseException as error:
        results["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        results["elapsed_seconds"] = time.monotonic() - started
        print(f"[G8_ERROR] rank={rank}\n{results['error']['traceback']}", flush=True)
        if rank == 0:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            if not RESULT_PATH.exists():
                RESULT_PATH.write_text(
                    json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
        return 1
    finally:
        phase.set("shutdown")
        if dist is not None and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
        stop.set()
        thread.join(timeout=2)
        print(f"[EXIT] rank={rank} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
