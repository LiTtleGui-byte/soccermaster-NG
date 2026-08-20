#!/usr/bin/env python3
"""Capture one smoke sample or one deterministic development shard on one GPU.

This is inference only. It never creates an optimizer or scheduler and never
calls backward or train. Each process sees exactly one physical GPU and writes
one new, isolated shard directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path("/home/tianlin/SoccerMaster")
EXPERIMENT = REPO / "experiments/commentary_generation"
PYTHON = REPO / ".local_envs/SoccerMaster-repro/bin/python"
REPORTS = REPO / "reports"
VIDEO_ROOT = Path(
    "/mnt/nas2/homes/jiayuanrao/UniSoccer_training_videos/"
    "SoccerNetv2/MatchTime/SN-Caption-test-align"
)

if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from runtime.paths import (  # noqa: E402
    ASSET_LAYOUT,
    BERT_ROOT,
    GENERATION_CHECKPOINT,
    LLAMA_ROOT,
    SIGLIP2_ROOT,
    TEST_ANNOTATIONS,
    VISUAL_BACKBONE,
    WORD_WORLD,
)


DATASET_SIZE = 3_256
CHECKPOINT_SIZE = 17_615_455_530
BACKBONE_SIZE = 1_435_281_181
CHECKPOINT_EPOCH = 11
MODEL_PARAMETERS = 8_418_890_760
MODEL_STATE_KEYS = 953
NUM_FRAMES = 30
MODEL_SEED = 20_260_817
MIN_CPU_AVAILABLE_BYTES = 64 * 1024**3
MIN_GPU_FREE_MIB = 60_000
EXPECTED_SHAPES = {
    "visual_frame_global": (30, 1024),
    "layer_normalized": (30, 1024),
    "temporal_output": (30, 1024),
    "qformer_input": (30, 1024),
    "qformer_output": (32, 768),
    "projector_output": (32, 4096),
}


class Heartbeat:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.stage = "preflight"
        self.progress = "not_started"
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, stage: str, progress: str = "") -> None:
        self.stage = stage
        self.progress = progress
        print(f"[STAGE] {stage} {progress}".rstrip(), flush=True)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop.wait(30):
            print(
                f"[HEARTBEAT] stage={self.stage} progress={self.progress} "
                f"elapsed_seconds={time.monotonic() - self.started:.1f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "shard"), required=True)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=MIN_GPU_FREE_MIB)
    parser.add_argument("--allow-existing-pid", type=int, action="append", default=[])
    return parser.parse_args()


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def require_single_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = [part.strip() for part in visible.split(",") if part.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError(
            "Exactly one numeric physical GPU must be exposed through "
            f"CUDA_VISIBLE_DEVICES; got {visible!r}"
        )
    return values[0]


def require_idle_physical_gpu(
    physical_gpu: str,
    minimum_free_mib: int,
    allowed_existing_pids: set[int],
) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = {}
    for line in query.stdout.splitlines():
        index, free_mib, utilization = [part.strip() for part in line.split(",")]
        rows[index] = {
            "free_mib": int(free_mib),
            "utilization_percent": int(utilization),
        }
    if physical_gpu not in rows:
        raise RuntimeError(f"Physical GPU {physical_gpu} is missing from nvidia-smi")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    gpu_uuid = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    uuid_by_index = {
        parts[0]: parts[1]
        for parts in (
            [part.strip() for part in line.split(",", 1)]
            for line in gpu_uuid.stdout.splitlines()
        )
    }
    target_uuid = uuid_by_index[physical_gpu]
    active = []
    for line in processes.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] == target_uuid:
            active.append({"pid": int(parts[1]), "raw": line})
    unexpected_pids = sorted(
        {process["pid"] for process in active} - allowed_existing_pids
    )
    state = rows[physical_gpu]
    if (
        state["free_mib"] < minimum_free_mib
        or state["utilization_percent"] > 5
        or unexpected_pids
    ):
        raise RuntimeError(
            f"Physical GPU {physical_gpu} is no longer idle: "
            f"state={state} active_pids={[item['pid'] for item in active]} "
            f"unexpected_pids={unexpected_pids}"
        )
    return {
        **state,
        "active_pids": [item["pid"] for item in active],
        "allowed_existing_pids": sorted(allowed_existing_pids),
    }


def require_file(path: Path, size: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"Unexpected size for {path}: {path.stat().st_size}")


def checked_output(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.output_root.resolve()
    if not root.is_relative_to(REPORTS.resolve()):
        raise RuntimeError(f"Output root must be inside {REPORTS}: {root}")
    if args.mode == "smoke":
        name = "smoke"
    else:
        if args.shard_id is None or not 0 <= args.shard_id < args.num_shards:
            raise RuntimeError("shard mode requires 0 <= shard-id < num-shards")
        name = f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    final = root / name
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite {final}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    return final, temporary


def main() -> int:
    args = parse_args()
    heartbeat = Heartbeat()
    heartbeat.start()
    started = time.monotonic()
    temporary: Path | None = None
    hooks: list[Any] = []
    torch = None
    result: dict[str, Any] = {
        "status": "failed",
        "mode": args.mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "training_executed": False,
        "backward_executed": False,
        "optimizer_created": False,
        "scheduler_created": False,
    }
    try:
        if Path(sys.executable).resolve() != PYTHON.resolve():
            raise RuntimeError(f"Wrong Python: {sys.executable}")
        os.chdir(REPO)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))

        heartbeat.update("asset_and_output_preflight")
        physical_gpu = require_single_gpu()
        if args.minimum_free_mib < 30_000:
            raise RuntimeError("minimum-free-mib cannot be below 30000")
        if args.mode == "shard" and (
            args.minimum_free_mib != MIN_GPU_FREE_MIB or args.allow_existing_pid
        ):
            raise RuntimeError(
                "Shard mode requires 60000 MiB and permits no existing process"
            )
        immediate_gpu_state = require_idle_physical_gpu(
            physical_gpu,
            args.minimum_free_mib,
            set(args.allow_existing_pid),
        )
        if ASSET_LAYOUT != "nas_bundle_v1":
            raise RuntimeError("The automatic run requires the verified NAS bundle")
        require_file(GENERATION_CHECKPOINT, CHECKPOINT_SIZE)
        require_file(VISUAL_BACKBONE, BACKBONE_SIZE)
        require_file(TEST_ANNOTATIONS)
        require_file(WORD_WORLD)
        for directory in (LLAMA_ROOT, BERT_ROOT, SIGLIP2_ROOT, VIDEO_ROOT):
            if not directory.is_dir():
                raise NotADirectoryError(directory)
        if available_memory_bytes() < MIN_CPU_AVAILABLE_BYTES:
            raise RuntimeError("Less than 64 GiB CPU memory is available")
        final, temporary = checked_output(args)
        result["physical_gpu"] = physical_gpu
        result["immediate_nvidia_smi"] = immediate_gpu_state
        result["output_dir"] = str(final)

        annotations = json.loads(TEST_ANNOTATIONS.read_text(encoding="utf-8"))
        if len(annotations) != DATASET_SIZE:
            raise RuntimeError(f"Expected {DATASET_SIZE} annotations")
        if args.mode == "smoke":
            indices = [0]
        else:
            indices = list(range(args.shard_id, DATASET_SIZE, args.num_shards))
        result["assigned_indices"] = len(indices)

        heartbeat.update("import_framework")
        import torch as torch_module
        from safetensors.torch import save_file

        torch = torch_module
        torch.set_num_threads(1)
        from research.experiments.commentary_generation.runtime.dataset.commentary import (
            MatchVisionCommentary_new_benchmark_from_npy_Dataset,
        )
        from research.experiments.commentary_generation.runtime.model.matchvoice_model_all_blocks import (
            matchvoice_model_all_blocks,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device")
        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        if free_bytes // 1024**2 < args.minimum_free_mib:
            raise RuntimeError(
                f"GPU free memory {free_bytes // 1024**2} MiB is below "
                f"{args.minimum_free_mib} MiB"
            )
        result["gpu"] = {
            "physical_index": physical_gpu,
            "name": torch.cuda.get_device_name(device),
            "free_mib_before": free_bytes // 1024**2,
            "total_mib": total_bytes // 1024**2,
        }

        heartbeat.update("construct_dataset")
        dataset = MatchVisionCommentary_new_benchmark_from_npy_Dataset(
            json_file=[str(TEST_ANNOTATIONS)],
            video_base_dir=[str(VIDEO_ROOT)],
            num_frames=NUM_FRAMES,
            sample="middle",
            tokenizer_name=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
        )
        if len(dataset) != DATASET_SIZE:
            raise RuntimeError(f"Unexpected dataset length: {len(dataset)}")

        heartbeat.update("construct_and_load_model")
        model = matchvoice_model_all_blocks(
            num_features=1024,
            need_temporal=True,
            need_spatial=False,
            use_local_features=False,
            open_visual_encoder=True,
            open_llm_decoder=True,
            file_path=str(WORD_WORLD),
            tokenizer_ckpt=str(LLAMA_ROOT),
            llm_ckpt=str(LLAMA_ROOT),
            visual_encoder_model_name=str(SIGLIP2_ROOT),
            visual_encoder_checkpoint=str(VISUAL_BACKBONE),
            timesformer_type="unisoccer_part_temporal",
            encoder_type="spatial_and_temporal",
            num_video_query_token=32,
            use_mlp=False,
        )
        if sum(parameter.numel() for parameter in model.parameters()) != MODEL_PARAMETERS:
            raise RuntimeError("Model parameter-count contract changed")
        if len(model.state_dict()) != MODEL_STATE_KEYS:
            raise RuntimeError("Model state-key contract changed")
        checkpoint = torch.load(
            GENERATION_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if checkpoint.get("epoch") != CHECKPOINT_EPOCH:
            raise RuntimeError(f"Unexpected checkpoint epoch: {checkpoint.get('epoch')}")
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint key mismatch: {incompatible}")
        del checkpoint
        gc.collect()

        captures: dict[str, Any] = {}
        generated: dict[str, Any] = {}

        def capture(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
                captures[name] = value.detach().float().cpu()
            return hook

        def capture_qformer_input(_module: Any, _args: Any, kwargs: Any) -> None:
            captures["qformer_input"] = (
                kwargs["encoder_hidden_states"].detach().float().cpu()
            )

        hooks.extend(
            (
                model.visual_encoder.register_forward_hook(capture("visual_frame_global")),
                model.ln_vision.register_forward_hook(capture("layer_normalized")),
                model.video_Qformer.bert.register_forward_pre_hook(
                    capture_qformer_input, with_kwargs=True
                ),
                model.video_Qformer.bert.register_forward_hook(capture("qformer_output")),
                model.llama_proj.register_forward_hook(capture("projector_output")),
            )
        )
        original_generate = model.llama_model.generate

        def traced_generate(*positional: Any, **keywords: Any) -> Any:
            keywords["return_dict_in_generate"] = True
            output = original_generate(*positional, **keywords)
            generated["token_ids"] = output.sequences.detach().cpu()[0].tolist()
            return output.sequences

        model.llama_model.generate = traced_generate

        heartbeat.update("move_model_to_gpu")
        model = model.to(device).eval()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        layer_rows: dict[str, list[Any]] = {name: [] for name in EXPECTED_SHAPES}
        prediction_rows: list[dict[str, Any]] = []
        for ordinal, dataset_index in enumerate(indices, start=1):
            heartbeat.progress = f"{ordinal}/{len(indices)} index={dataset_index}"
            item = dataset[dataset_index]
            annotation = annotations[dataset_index]
            expected_video = VIDEO_ROOT / annotation["video"]
            if Path(item["video_path"]) != expected_video:
                raise RuntimeError(f"Video identity mismatch at {dataset_index}")
            if item["caption_text"] != annotation["comments_text_anonymized"]:
                raise RuntimeError(f"Reference identity mismatch at {dataset_index}")
            if tuple(item["frames"].shape) != (3, NUM_FRAMES, 512, 512):
                raise RuntimeError(f"Frame shape mismatch at {dataset_index}")

            per_sample_seed = MODEL_SEED + dataset_index
            random.seed(per_sample_seed)
            torch.manual_seed(per_sample_seed)
            torch.cuda.manual_seed_all(per_sample_seed)
            captures.clear()
            generated.clear()
            samples = dataset.collater([item])
            for key, value in samples.items():
                if isinstance(value, torch.Tensor):
                    samples[key] = value.to(device)
            with torch.inference_mode():
                texts, references, video_paths = model(samples, True)
            if references != [item["caption_text"]] or video_paths != [item["video_path"]]:
                raise RuntimeError(f"Returned identity mismatch at {dataset_index}")
            if len(texts) != 1 or not texts[0].strip():
                raise RuntimeError(f"Empty generation at {dataset_index}")
            if "token_ids" not in generated:
                raise RuntimeError(f"Generated token IDs missing at {dataset_index}")

            captures["temporal_output"] = captures["qformer_input"].clone()
            captures["layer_normalized"] = captures["layer_normalized"].squeeze(2)
            for name, shape in EXPECTED_SHAPES.items():
                tensor = captures.get(name)
                if tensor is None or tuple(tensor.shape[1:]) != shape:
                    observed = None if tensor is None else tuple(tensor.shape)
                    raise RuntimeError(f"Invalid {name} at {dataset_index}: {observed}")
                if not tensor.isfinite().all().item():
                    raise RuntimeError(f"Non-finite {name} at {dataset_index}")
                layer_rows[name].append(tensor.contiguous())

            prediction_rows.append(
                {
                    "dataset_index": dataset_index,
                    "match_id": str(Path(annotation["video"]).parent),
                    "video_path": str(expected_video),
                    "frame_indices": item["frame_indices"],
                    "video_duration_seconds": item["video_duration_seconds"],
                    "preprocess_id": "siglip2-large-patch16-512_middle30_runtime_v1",
                    "checkpoint_id": "commentary_epoch_11_sha256_e1ff7fef61a4",
                    "seed": per_sample_seed,
                    "reference_commentary": item["caption_text"],
                    "generated_commentary": texts[0],
                    "generated_token_ids": generated["token_ids"],
                }
            )
            print(f"[SAMPLE_OK] {ordinal}/{len(indices)} index={dataset_index}", flush=True)
            del samples
            del item

        heartbeat.update("write_shard")
        tensors = {
            name: torch.cat(rows, dim=0).contiguous()
            for name, rows in layer_rows.items()
        }
        tensors["dataset_indices"] = torch.tensor(indices, dtype=torch.int64)
        save_file(
            tensors,
            str(temporary / "layers.safetensors"),
            metadata={
                "schema_version": "1",
                "mode": args.mode,
                "shard_id": "none" if args.shard_id is None else str(args.shard_id),
                "num_shards": str(args.num_shards),
                "checkpoint_epoch": str(CHECKPOINT_EPOCH),
            },
        )
        with (temporary / "predictions.jsonl").open("x", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        result.update(
            {
                "status": "passed",
                "samples_completed": len(indices),
                "dataset_indices_first": indices[:5],
                "dataset_indices_last": indices[-5:],
                "stored_shapes": {name: list(value.shape) for name, value in tensors.items()},
                "stored_dtypes": {name: str(value.dtype) for name, value in tensors.items()},
                "decoder_diagnostics": {
                    "generated_text": True,
                    "generated_token_ids": True,
                    "first_token_logits": False,
                    "fact_token_nll": False,
                    "deferred_to_review_gate": True,
                },
                "gpu_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        (temporary / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(final)
        temporary = None
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": args.mode,
                    "samples": len(indices),
                    "output": str(final),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
        if temporary is not None:
            try:
                (temporary / "failure.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except BaseException:
                traceback.print_exc()
        return 1
    finally:
        for hook in hooks:
            hook.remove()
        heartbeat.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


if __name__ == "__main__":
    raise SystemExit(main())
