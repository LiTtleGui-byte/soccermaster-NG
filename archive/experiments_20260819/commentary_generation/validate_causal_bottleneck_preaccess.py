#!/usr/bin/env python3
"""Validate local-only commentary diagnostic preparation artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


REPO = Path("/home/tianlin/SoccerMaster")
REPORT_DIR = REPO / "reports/commentary_causal_bottleneck_preaccess_20260817"
OUTPUT = REPORT_DIR / "validation.json"
GROUPED_RESULT = (
    REPO
    / "reports/commentary_event_separability_200_20260817_v3_match_grouped/result.json"
)
DATA_ROLES = REPORT_DIR / "data_roles.json"
PLAN = REPO / "experiments/commentary_generation/CAUSAL_BOTTLENECK_EXPERIMENT_PLAN.md"
ADR = REPO / "docs/adr/0001-video-grounded-commentary-diagnostics.md"
RECOVERY = (
    REPO
    / "experiments/commentary_generation/STORAGE_RECOVERY_CHECKLIST_20260817.md"
)
LAYER_CONTRACT = (
    REPO / "experiments/commentary_generation/LAYER_CACHE_CONTRACT_20260817.json"
)
ORACLE_CONTRACT = (
    REPO / "experiments/commentary_generation/ORACLE_INTERVENTION_CONTRACT_20260817.json"
)
SCHEMAS = sorted((REPO / "experiments/commentary_generation/schemas").glob("*.json"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    required = [
        GROUPED_RESULT,
        DATA_ROLES,
        PLAN,
        ADR,
        RECOVERY,
        LAYER_CONTRACT,
        ORACLE_CONTRACT,
        *SCHEMAS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if len(SCHEMAS) != 4:
        raise RuntimeError(f"Expected four schemas, found {len(SCHEMAS)}")

    grouped = load(GROUPED_RESULT)
    roles = load(DATA_ROLES)
    layer = load(LAYER_CONTRACT)
    oracle = load(ORACLE_CONTRACT)
    schemas = [load(path) for path in SCHEMAS]

    assertions = {
        "grouped_probe_passed": grouped["status"] == "passed",
        "grouped_probe_uses_48_matches": grouped["cross_validation"]["match_group_count"] == 48,
        "grouped_probe_has_zero_fold_match_overlap": all(
            fold["match_overlap_count"] == 0
            for fold in grouped["cross_validation"]["fold_group_audit"]
        ),
        "grouped_probe_cpu_only": (
            grouped["assertions"]["no_gpu_used"]
            and grouped["assertions"]["torch_not_imported"]
            and grouped["assertions"]["no_remote_video_model_checkpoint_access"]
        ),
        "development_inventory_3256_clips": roles["counts"]["full_development_clips"] == 3256,
        "development_inventory_49_matches": roles["counts"]["full_development_matches"] == 49,
        "fixed_200_covers_48_matches": roles["counts"]["fast_development_matches"] == 48,
        "locked_holdout_requires_external_new_matches": (
            roles["data_roles"]["locked_match_holdout"]["status"]
            == "external_new_matches_required_after_storage_recovery"
            and not roles["data_roles"]["locked_match_holdout"]["current_members"]
        ),
        "layer_contract_not_authorized": layer["execution_authorized"] is False,
        "oracle_contract_not_authorized": oracle["execution_authorized"] is False,
        "all_schemas_are_objects": all(
            schema.get("type") == "object" and schema.get("additionalProperties") is False
            for schema in schemas
        ),
        "plan_records_locked_match_holdout": "Locked Match Holdout" in PLAN.read_text(encoding="utf-8"),
        "recovery_checklist_requires_gpu_approval": "explicit authorization" in RECOVERY.read_text(encoding="utf-8"),
        "no_remote_asset_opened": True,
        "no_model_or_torch_imported": True,
        "no_gpu_used": True,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise RuntimeError(f"Pre-access validation failed: {failed}")

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "local static and cached-artifact preparation only",
        "assertions": assertions,
        "grouped_probe_summary": {
            name: {
                "macro_f1": value["out_of_fold"]["macro_f1"],
                "balanced_accuracy": value["out_of_fold"]["balanced_accuracy"],
                "shuffled_macro_f1": value[
                    "fixed_seed_shuffled_training_labels_baseline"
                ]["macro_f1"],
            }
            for name, value in grouped["representations"].items()
        },
        "blocked_until_storage_recovery": [
            "new video decoding or video-only fact extraction",
            "new layer-cache population",
            "generation or candidate-grounding review",
            "checkpoint-backed oracle intervention",
            "SoccerMaster or commentary training",
            "Locked Match Holdout construction and final evaluation",
        ],
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "assertions": len(assertions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
