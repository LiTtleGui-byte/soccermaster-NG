#!/usr/bin/env python3
"""Run-2 wrapper for the G10 Refiner probe on read-only foreign-owned Git assets."""

from __future__ import annotations

import copy
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import g10_refiner_probe_run1 as base


REPO = Path("/home/tianlin/SoccerMaster")
DEFAULT_MANIFEST = REPO / "reproduction/manifests/g10_refiner_probe_run2_sngs10004.json"
RUN2_PREPARED_STAGE = "refiner_255_probe_run2_prepared"
RUN2_STAGE = "refiner_255_probe_run2"
RUN2_PREFLIGHT_STAGE = "refiner_255_probe_run2_preflight"

_validate_local_contract_run1 = base.validate_local_contract
_run_probe_run1 = base.run_probe


def requested_manifest() -> Path:
    if "--manifest" not in sys.argv:
        return DEFAULT_MANIFEST
    index = sys.argv.index("--manifest")
    if index + 1 >= len(sys.argv):
        raise ValueError("--manifest requires a path")
    return Path(sys.argv[index + 1])


def requested_mode() -> str:
    if "--mode" not in sys.argv:
        return "preflight"
    index = sys.argv.index("--mode")
    if index + 1 >= len(sys.argv):
        raise ValueError("--mode requires a value")
    return sys.argv[index + 1]


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    identity = (raw.get("schema_version"), raw.get("gate"), raw.get("stage"))
    if identity != (1, "G10-B", RUN2_PREPARED_STAGE):
        raise AssertionError(f"Unexpected run2 manifest identity: {identity}")
    inherited = json.loads(
        base.validate_file(raw["base_manifest"]).read_text(encoding="utf-8")
    )
    inherited_identity = (
        inherited.get("schema_version"), inherited.get("gate"), inherited.get("stage")
    )
    if inherited_identity != (1, "G10-B", "refiner_255_probe_run1_prepared"):
        raise AssertionError(f"Unexpected inherited manifest: {inherited_identity}")
    resolved = copy.deepcopy(inherited)
    resolved["stage"] = RUN2_PREPARED_STAGE
    resolved["description"] = raw["description"]
    resolved["runtime"] = copy.deepcopy(raw["runtime"])
    resolved["cuda_guard"] = copy.deepcopy(raw["cuda_guard"])
    resolved["preflight_outputs"] = copy.deepcopy(raw["preflight_outputs"])
    resolved["future_command_arguments"] = copy.deepcopy(
        raw["future_command_arguments"]
    )
    return resolved


def read_git_revision(root: Path) -> str:
    """Resolve HEAD without invoking Git or changing safe.directory config."""
    git_dir = root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_name = head.removeprefix("ref: ").strip()
        if not re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref_name):
            raise AssertionError(f"Invalid Git HEAD ref: {ref_name!r}")
        loose_ref = git_dir / ref_name
        if loose_ref.is_file():
            revision = loose_ref.read_text(encoding="utf-8").strip()
        else:
            revision = ""
            packed_refs = git_dir / "packed-refs"
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                candidate, candidate_ref = line.split(" ", 1)
                if candidate_ref == ref_name:
                    revision = candidate
                    break
            if not revision:
                raise AssertionError(f"Git HEAD ref is unresolved: {ref_name}")
    else:
        revision = head
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AssertionError(f"Invalid Git revision: {revision!r}")
    return revision


def validate_runtime_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["refiner"]["root"])
    revision = read_git_revision(root)
    if revision != manifest["refiner"]["revision"]:
        raise AssertionError("Refiner revision changed")
    sources = {
        name: str(base.validate_file(spec))
        for name, spec in manifest["refiner"]["sources"].items()
    }
    checkpoint = base.validate_file(manifest["refiner"]["checkpoint"])
    python = Path(manifest["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    return {
        "revision": revision,
        "revision_method": "read_only_git_head_resolution_without_git_config",
        "sources": sources,
        "checkpoint": str(checkpoint),
    }


def validate_local_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    approval = manifest["cuda_guard"]["approval_environment"]
    checked = copy.deepcopy(manifest)
    checked["cuda_guard"]["approval_environment"] = (
        "G10_REFINER_PROBE_RUN1_GPU_APPROVED=YES"
    )
    result = _validate_local_contract_run1(checked)
    result["cuda_guard"] = copy.deepcopy(manifest["cuda_guard"])
    if approval != "G10_REFINER_PROBE_RUN2_GPU_APPROVED=YES":
        raise AssertionError("Run2 GPU approval guard changed")
    return result


def run_probe(manifest: dict[str, Any], preflight: dict[str, Any]) -> int:
    exit_code = _run_probe_run1(manifest, preflight)
    result_path = Path(manifest["runtime"]["result"])
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["stage"] = RUN2_STAGE
        base.atomic_json(result, result_path)
    return exit_code


def write_early_failure(error: Exception, started: float) -> None:
    try:
        manifest = load_manifest(requested_manifest())
        runtime = manifest["runtime"]
        report_dir = Path(runtime["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        result_path = Path(runtime["result"])
        if result_path.exists() or result_path.is_symlink():
            return
        result = {
            "schema_version": 1,
            "gate": "G10-B",
            "stage": RUN2_STAGE,
            "outcome": "failed",
            "assertions_passed": False,
            "failure_category": "launcher_preflight_or_runtime_asset",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "started_unix": started,
            "ended_unix": time.time(),
            "fallbacks_used": [],
            "retry_used": False,
            "evaluation_started": False,
            "visualization_started": False,
            "training_started": False,
        }
        base.atomic_json(result, result_path)
    except Exception:
        traceback.print_exc()


def main() -> int:
    started = time.time()
    base.DEFAULT_MANIFEST = DEFAULT_MANIFEST
    base.__file__ = __file__
    base.load_manifest = load_manifest
    base.validate_local_contract = validate_local_contract
    base.validate_runtime_assets = validate_runtime_assets
    base.run_probe = run_probe
    try:
        exit_code = base.main()
        if requested_mode() == "preflight" and exit_code == 0:
            manifest = load_manifest(requested_manifest())
            result_path = Path(manifest["preflight_outputs"]["result"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["stage"] = RUN2_PREFLIGHT_STAGE
            base.atomic_json(result, result_path)
        return exit_code
    except Exception as error:
        traceback.print_exc()
        if requested_mode() == "run":
            write_early_failure(error, started)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
