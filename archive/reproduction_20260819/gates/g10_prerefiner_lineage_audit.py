#!/usr/bin/env python3
"""CPU-only static audit of the historical pre-Refiner production lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_prerefiner_lineage_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_small_file(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in spec and path.stat().st_size != spec["bytes"]:
        raise AssertionError(f"Size changed: {path}")
    if sha256_file(path) != spec["sha256"]:
        raise AssertionError(f"SHA256 changed: {path}")
    return path


def require_new_local_output(path: Path) -> None:
    if not path.resolve(strict=False).is_relative_to(REPO):
        raise AssertionError(f"Output outside workspace: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def normalized_variant(config: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(config)
    value["experiment_subname"] = "<split>"
    value["dataset"]["eval_set"] = "<split>"
    value["dataset"]["vids_dict"] = {"<split>": []}
    return value


def main() -> int:
    args = parse_args()
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "prerefiner_lineage_audit":
        raise AssertionError("Unexpected manifest identity")

    report_dir = Path(manifest["outputs"]["report_dir"])
    result_path = Path(manifest["outputs"]["result"])
    log_path = Path(manifest["outputs"]["log"])
    require_new_local_output(report_dir)
    if result_path.parent != report_dir or log_path.parent != report_dir:
        raise AssertionError("Output files must remain in the isolated report directory")
    report_dir.mkdir(parents=True, exist_ok=False)

    log_lines: list[str] = []

    def log(phase: str, message: str) -> None:
        line = f"[{phase}] {message}"
        log_lines.append(line)
        print(line, flush=True)

    try:
        execution = manifest["execution"]
        actual_env = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        }
        if actual_env != execution["required_environment"]:
            raise AssertionError(f"Environment guard mismatch: {actual_env}")
        log("environment", "CPU-only/no-import environment guard passed")

        sources = {
            name: validate_small_file(spec)
            for name, spec in manifest["small_sources"].items()
        }
        log("sources", f"validated {len(sources)} pinned small files by SHA256")

        variants: dict[str, dict[str, Any]] = {}
        for split in ("train", "valid", "test"):
            config = yaml.safe_load(sources[f"combination_{split}"].read_text(encoding="utf-8"))
            variants[split] = config
            if config["pipeline"] != manifest["contract"]["pipeline"]:
                raise AssertionError(f"Pipeline changed for {split}")
            if config["dataset"]["eval_set"] != split:
                raise AssertionError(f"Split mismatch for {split}")
            if config["dataset"]["vids_dict"] != {split: []}:
                raise AssertionError(f"vids_dict mismatch for {split}")
            if config["state"] != {"save_file": "states/sn-gamestate.pklz", "load_file": None}:
                raise AssertionError(f"State contract changed for {split}")
            if not config["test_tracking"] or not config["eval_tracking"]:
                raise AssertionError(f"Historical tracking/evaluation flags changed for {split}")
        baseline = normalized_variant(variants["train"])
        if normalized_variant(variants["valid"]) != baseline or normalized_variant(variants["test"]) != baseline:
            raise AssertionError("Train/valid/test configs differ beyond split and experiment_subname")
        log("configs", "three production configs share one exact pipeline and differ only by split/name")

        module_targets = {}
        for module_name, expectation in manifest["contract"]["module_configs"].items():
            cfg = yaml.safe_load(sources[expectation["source_key"]].read_text(encoding="utf-8"))
            if cfg["_target_"] != expectation["target"]:
                raise AssertionError(f"Target changed for {module_name}")
            module_targets[module_name] = cfg["_target_"]
        log("modules", "all configured module targets matched the pinned contract")

        for source_key, fragments in manifest["contract"]["required_source_fragments"].items():
            text = sources[source_key].read_text(encoding="utf-8")
            missing = [fragment for fragment in fragments if fragment not in text]
            if missing:
                raise AssertionError(f"Missing source fragments in {source_key}: {missing}")
        log("field_lineage", "source-level field producers and dependencies matched")

        weight_assets: dict[str, Any] = {}
        for name, spec in manifest["weight_assets"].items():
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Weight asset missing or size changed: {path}")
            weight_assets[name] = {"path": str(path), "bytes": path.stat().st_size}

        qwen = manifest["qwen_asset"]
        qwen_link = Path(qwen["path"])
        if not qwen_link.is_symlink() or os.readlink(qwen_link) != qwen["link_target"]:
            raise AssertionError("Qwen symlink target changed")
        qwen_root = qwen_link.resolve(strict=True)
        shard_total = 0
        shard_records = []
        for shard in qwen["shards"]:
            shard_path = qwen_root / shard["name"]
            if not shard_path.is_file() or shard_path.stat().st_size != shard["bytes"]:
                raise AssertionError(f"Qwen shard missing or size changed: {shard_path}")
            shard_total += shard_path.stat().st_size
            shard_records.append({"path": str(shard_path), "bytes": shard_path.stat().st_size})
        if shard_total != qwen["logical_shard_bytes"]:
            raise AssertionError("Qwen logical shard byte total changed")
        log("weights", "all referenced local weights exist with pinned byte sizes; no weight content opened")

        archive_summaries = {}
        expected_columns = manifest["contract"]["historical_archive_columns"]
        for split, spec in manifest["historical_archives"].items():
            path = Path(spec["path"])
            if not path.is_file() or path.stat().st_size != spec["bytes"]:
                raise AssertionError(f"Historical archive missing or size changed: {path}")
            with zipfile.ZipFile(path, "r") as archive:
                summary = json.loads(archive.read("summary.json"))
            if summary.get("columns") != expected_columns:
                raise AssertionError(f"Historical archive schema changed: {split}")
            archive_summaries[split] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "summary_columns": summary["columns"],
                "full_archive_sha256": "not_computed_large_read_only_asset",
            }
        log("archives", "train/valid/test historical ZIP summaries all contain the Refiner-required fields")

        result = {
            "schema_version": 1,
            "gate": "G10-B",
            "stage": manifest["stage"],
            "status": "passed",
            "verdict": "historical_prerefiner_lineage_statically_traced",
            "started_utc": started_iso,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.time() - started,
            "environment": {
                **actual_env,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "yaml_version": yaml.__version__,
                "torch_imported": "torch" in sys.modules,
                "gpu_queried": False,
                "model_loaded": False,
                "model_forward": False,
                "evaluation_run": False,
                "training_run": False,
                "fallback": None,
            },
            "git": git_identity(),
            "upstream_revisions_current_preserved_checkout": manifest["upstream_revisions"],
            "historical_execution_revision": "unknown_not_recorded_in_examined_archives",
            "pipeline": manifest["contract"]["pipeline"],
            "module_targets": module_targets,
            "field_lineage": manifest["contract"]["field_lineage"],
            "weight_assets": weight_assets,
            "qwen_asset": {
                "configured_path": str(qwen_link),
                "symlink_target": qwen["link_target"],
                "resolved_path": str(qwen_root),
                "logical_shard_bytes": shard_total,
                "shards": shard_records,
            },
            "historical_archives": archive_summaries,
            "configured_side_effects_if_upstream_entry_were_run": manifest["contract"]["side_effects"],
            "conclusion_scope": {
                "confirmed": [
                    "The preserved combination configs form the complete pre-Refiner field-production chain.",
                    "All three existing historical state archives advertise the Refiner-required detection and image fields.",
                    "The current run5 Step-1 archive lacks fields produced only after track in this combination pipeline.",
                ],
                "not_verified": [
                    "Exact source/dependency revision used for the March 2025 historical archive runs.",
                    "Field values, quality, determinism, or compatibility of any newly produced SNGS-10004 archive.",
                    "Any module initialization, checkpoint deserialization, GPU inference, evaluation, training, Refiner forward, Step 3, or conversion.",
                ],
            },
            "resource": {
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_peak_bytes": None,
            },
            "error": None,
        }
        if result["environment"]["torch_imported"]:
            raise AssertionError("Torch was unexpectedly imported")
        atomic_json_dump(result, result_path)
        log("result", f"wrote {result_path}")
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "gate": "G10-B",
            "stage": manifest.get("stage"),
            "status": "failed",
            "started_utc": started_iso,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.time() - started,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if not result_path.exists():
            atomic_json_dump(failure, result_path)
        log("error", f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
