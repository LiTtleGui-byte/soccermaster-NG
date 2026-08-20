#!/usr/bin/env python3
"""Build local-only data-role evidence for commentary bottleneck experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


REPO = Path("/home/tianlin/SoccerMaster")
ITEMS = REPO / "reports/commentary_reference_audit_3256_20260814/items.jsonl"
REVIEWS = REPO / "reports/commentary_reference_audit_3256_20260814/reviews.jsonl"
FIXED_200 = REPO / "reports/commentary_prefix_cache_200_20260814_run1/manifest.json"
OUTPUT_DIR = REPO / "reports/commentary_causal_bottleneck_preaccess_20260817"

EVENT_FAMILY_MAP = {
    "ball out of play": "restart_or_out",
    "ball possession": "open_play",
    "clearance": "open_play",
    "corner": "corner",
    "end of half game": "restart_or_out",
    "foul with no card": "foul_or_free_kick",
    "free kick": "foul_or_free_kick",
    "goal": "shot_or_goal",
    "injury": "injury",
    "lead to corner": "corner",
    "off side": "offside",
    "penalty": "penalty",
    "red card": "card",
    "saved by goal-keeper": "shot_or_goal",
    "second yellow card": "card",
    "shot off target": "shot_or_goal",
    "show added time": "restart_or_out",
    "start of half game": "restart_or_out",
    "substitution": "substitution",
    "throw in": "restart_or_out",
    "yellow card": "card",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def match_id(video_path: str) -> str:
    normalized = video_path.replace("\\", "/")
    marker = "/SN-Caption-test-align/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    parts = normalized.split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive match from {video_path!r}")
    return "/".join(parts[:-1])


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    for path in (ITEMS, REVIEWS, FIXED_200):
        if not path.is_file():
            raise FileNotFoundError(path)

    items = load_jsonl(ITEMS)
    reviews = load_jsonl(REVIEWS)
    fixed = json.loads(FIXED_200.read_text(encoding="utf-8"))["samples"]
    if len(items) != 3256 or len(reviews) != 3256 or len(fixed) != 200:
        raise RuntimeError("Unexpected source row count")

    items_by_index = {int(row["dataset_index"]): row for row in items}
    reviews_by_index = {int(row["dataset_index"]): row for row in reviews}
    if set(items_by_index) != set(range(3256)) or set(reviews_by_index) != set(range(3256)):
        raise RuntimeError("Dataset/review indices are not exactly 0..3255")

    per_match: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dataset_indices": [],
            "caption_counts": Counter(),
            "event_family_counts": Counter(),
            "reference_support_verdict_counts": Counter(),
        }
    )
    for index in range(3256):
        item = items_by_index[index]
        review = reviews_by_index[index]
        if str(item["video"]) != str(review["video"]):
            raise RuntimeError(f"Item/review video mismatch at {index}")
        caption = str(item["caption"])
        if caption not in EVENT_FAMILY_MAP:
            raise RuntimeError(f"Unmapped caption {caption!r}")
        group = match_id(str(item["video"]))
        record = per_match[group]
        record["dataset_indices"].append(index)
        record["caption_counts"][caption] += 1
        record["event_family_counts"][EVENT_FAMILY_MAP[caption]] += 1
        record["reference_support_verdict_counts"][str(review["verdict"])] += 1

    fixed_indices = [int(row["dataset_index"]) for row in fixed]
    if len(set(fixed_indices)) != 200:
        raise RuntimeError("Fixed-200 indices are not unique")
    fixed_groups = {match_id(str(row["video_path"])) for row in fixed}
    all_groups = set(per_match)
    unseen_groups = sorted(all_groups - fixed_groups)
    if len(all_groups) != 49 or len(fixed_groups) != 48 or len(unseen_groups) != 1:
        raise RuntimeError(
            f"Unexpected group counts: all={len(all_groups)} fixed={len(fixed_groups)} "
            f"unseen={len(unseen_groups)}"
        )

    matches = []
    for group in sorted(per_match):
        record = per_match[group]
        matches.append(
            {
                "match_id": group,
                "sample_count": len(record["dataset_indices"]),
                "dataset_indices": record["dataset_indices"],
                "caption_counts": dict(sorted(record["caption_counts"].items())),
                "event_family_counts": dict(
                    sorted(record["event_family_counts"].items())
                ),
                "reference_support_verdict_counts": dict(
                    sorted(record["reference_support_verdict_counts"].items())
                ),
                "represented_in_fixed_200": group in fixed_groups,
            }
        )

    unseen_group = unseen_groups[0]
    unseen_count = len(per_match[unseen_group]["dataset_indices"])
    result = {
        "created_at_utc": utc_now(),
        "status": "prepared_local_only",
        "inputs": {
            "items": str(ITEMS),
            "reviews": str(REVIEWS),
            "fixed_200_manifest": str(FIXED_200),
            "remote_paths_embedded_but_not_accessed": True,
        },
        "event_family_mapping": EVENT_FAMILY_MAP,
        "counts": {
            "full_development_clips": len(items),
            "full_development_matches": len(all_groups),
            "fast_development_clips": len(fixed),
            "fast_development_matches": len(fixed_groups),
            "matches_not_represented_in_fixed_200": len(unseen_groups),
            "clips_in_only_unrepresented_match": unseen_count,
        },
        "data_roles": {
            "fixed_200": "fast development and protocol debugging only",
            "full_3256": "development baseline and module localization only",
            "only_match_absent_from_fixed_200": {
                "match_id": unseen_group,
                "sample_count": unseen_count,
                "role": "development sensitivity check only; not a sufficient locked holdout",
            },
            "locked_match_holdout": {
                "status": "external_new_matches_required_after_storage_recovery",
                "current_members": [],
                "requirements": [
                    "match absent from current 49 development matches",
                    "match and clips absent from relevant generation/backbone training data",
                    "never used for prompt, taxonomy, threshold, or model selection",
                ],
            },
        },
        "matches": matches,
        "assertions": {
            "items_3256": len(items) == 3256,
            "reviews_3256": len(reviews) == 3256,
            "fixed_200_unique": len(set(fixed_indices)) == 200,
            "full_match_count_49": len(all_groups) == 49,
            "fixed_200_match_count_48": len(fixed_groups) == 48,
            "only_one_match_absent_from_fixed_200": len(unseen_groups) == 1,
            "no_remote_asset_opened": True,
        },
    }

    readme = f"""# Commentary causal-bottleneck pre-access data roles

Prepared from local JSON only; no NAS/GPFS path was opened.

- Full development set: 3,256 clips from {len(all_groups)} matches.
- Existing fixed-200: 200 clips from {len(fixed_groups)} matches.
- Match absent from fixed-200: `{unseen_group}` ({unseen_count} clips).
- Locked Match Holdout: not available inside the current 3,256 clips; new matches and training-overlap checks are required after storage recovery.

The 3,256 clips and fixed-200 are development data. The single internally unseen match is too narrow to support the final claim.
"""

    OUTPUT_DIR.mkdir(parents=False, exist_ok=False)
    (OUTPUT_DIR / "data_roles.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "prepared", "matches": 49, "fixed_matches": 48}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
