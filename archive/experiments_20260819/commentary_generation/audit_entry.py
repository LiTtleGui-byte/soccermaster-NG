#!/usr/bin/env python3
"""Pure-standard-library inventory check for the commentary generation branch.

This file intentionally does not import torch, transformers, peft, or project code.
It does not load tokenizers, models, checkpoints, or pickle files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPECTED_PYTHON = Path(
    "/home/tianlin/SoccerMaster/.local_envs/SoccerMaster-repro/bin/python"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_versions(expected: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    actual: dict[str, str] = {}
    errors: list[str] = []
    for name, expected_version in sorted(expected.items()):
        try:
            actual_version = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual_version = "MISSING"
        actual[name] = actual_version
        if actual_version != expected_version:
            errors.append(
                f"distribution {name}: expected {expected_version}, got {actual_version}"
            )
    return actual, errors


def check_assets(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in entries:
        path = Path(entry["path"])
        kind = entry["kind"]
        exists = path.is_dir() if kind == "directory" else path.is_file()
        result: dict[str, Any] = {
            "name": entry["name"],
            "path": str(path),
            "exists": exists,
        }
        if not exists:
            errors.append(f"missing {kind}: {path}")
            results.append(result)
            continue
        if kind == "file":
            actual_size = path.stat().st_size
            result["size"] = actual_size
            if actual_size != entry["size"]:
                errors.append(
                    f"size mismatch: {path}: expected {entry['size']}, got {actual_size}"
                )
            if "sha256" in entry:
                actual_hash = sha256(path)
                result["sha256"] = actual_hash
                if actual_hash != entry["sha256"]:
                    errors.append(f"sha256 mismatch: {path}")
            else:
                result["sha256"] = "deferred_for_large_file"
        results.append(result)
    return results, errors


def check_sources(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in entries:
        path = Path(entry["source"])
        exists = path.is_file()
        result: dict[str, Any] = {
            "role": entry["role"],
            "source": str(path),
            "target": entry["target"],
            "exists": exists,
            "vendored": entry["vendored"],
        }
        if not exists:
            errors.append(f"missing source: {path}")
            results.append(result)
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        result.update(size=actual_size, sha256=actual_hash)
        if actual_size != entry["size"]:
            errors.append(f"source size mismatch: {path}")
        if actual_hash != entry["sha256"]:
            errors.append(f"source sha256 mismatch: {path}")
        if entry["vendored"]:
            local_path = ROOT / entry["target"]
            result["local_path"] = str(local_path)
            result["local_exists"] = local_path.is_file()
            if not local_path.is_file():
                errors.append(f"missing vendored source: {local_path}")
            else:
                local_ast_valid = True
                try:
                    ast.parse(
                        local_path.read_text(encoding="utf-8"),
                        filename=str(local_path),
                    )
                except (SyntaxError, UnicodeDecodeError) as error:
                    local_ast_valid = False
                    errors.append(f"invalid vendored Python: {local_path}: {error}")
                result["local_ast_valid"] = local_ast_valid
        results.append(result)
    return results, errors


def check_known_blockers(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in entries:
        paths = [Path(value) for value in entry["paths"]]
        results.append(
            {
                "id": entry["id"],
                "description": entry["description"],
                "paths": [str(path) for path in paths],
                "paths_present": [path.exists() for path in paths],
                "active": not all(path.exists() for path in paths),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when a registered blocker is still active.",
    )
    args = parser.parse_args()

    assets_manifest = load_json(ROOT / "assets.json")
    sources_manifest = load_json(ROOT / "sources.json")

    errors: list[str] = []
    actual_python = Path(sys.executable).resolve()
    if actual_python != EXPECTED_PYTHON.resolve():
        errors.append(f"wrong Python: expected {EXPECTED_PYTHON}, got {actual_python}")

    versions, version_errors = distribution_versions(
        assets_manifest["required_distributions"]
    )
    asset_results, asset_errors = check_assets(assets_manifest["assets"])
    source_results, source_errors = check_sources(sources_manifest["sources"])
    blockers = check_known_blockers(assets_manifest["known_blockers"])
    errors.extend(version_errors)
    errors.extend(asset_errors)
    errors.extend(source_errors)

    active_blockers = [item["id"] for item in blockers if item["active"]]
    result = {
        "status": "failed" if errors else "inventory_consistent",
        "mode": "strict" if args.strict else "inventory",
        "python": str(actual_python),
        "distribution_versions": versions,
        "assets": asset_results,
        "sources": source_results,
        "active_blockers": active_blockers,
        "blocker_details": blockers,
        "errors": errors,
        "ready_for_model_load": not errors,
        "ready_for_end_to_end_inference": not errors and not active_blockers,
        "model_imported": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

    if errors:
        return 1
    if args.strict and active_blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
