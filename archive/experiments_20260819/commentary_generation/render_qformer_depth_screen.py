#!/usr/bin/env python3
"""Render a small CPU summary of the existing Q-Former layer screen."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/home/tianlin/SoccerMaster")
INPUT = REPO / "reports/commentary_stage2_layer_probe_48_20260818/result.json"
REPORT = REPO / "reports/commentary_stage2_layer_probe_48_20260818"
RESULT = REPORT / "qformer_depth_screen.json"
FIGURE = REPORT / "qformer_input_output_delta.png"


def write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if RESULT.exists() or FIGURE.exists():
        raise FileExistsError("Refusing to overwrite Q-Former depth screen outputs")
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    input_tasks = source["representations"]["qformer_input"]["tasks"]
    output_tasks = source["representations"]["qformer_output"]["tasks"]
    rows: list[dict[str, Any]] = []
    for task, before in input_tasks.items():
        after = output_tasks.get(task, {})
        before_auc = before.get("metrics", {}).get("roc_auc")
        after_auc = after.get("metrics", {}).get("roc_auc")
        if before_auc is None or after_auc is None:
            continue
        rows.append(
            {
                "task": task,
                "family": task.split(":", 1)[0],
                "qformer_input_roc_auc": float(before_auc),
                "qformer_output_roc_auc": float(after_auc),
                "delta_output_minus_input": float(after_auc - before_auc),
            }
        )
    rows.sort(key=lambda row: row["delta_output_minus_input"])
    families: dict[str, dict[str, Any]] = {}
    for family in sorted({row["family"] for row in rows}):
        values = np.asarray(
            [row["delta_output_minus_input"] for row in rows if row["family"] == family],
            dtype=float,
        )
        families[family] = {
            "task_count": int(len(values)),
            "mean_delta": float(values.mean()),
            "drop_count": int((values < 0).sum()),
            "improvement_count": int((values > 0).sum()),
        }

    REPORT.mkdir(parents=True, exist_ok=True)
    labels = [row["task"] for row in rows]
    values = [row["delta_output_minus_input"] for row in rows]
    colors = ["#d95f59" if value < 0 else "#4c9f70" for value in values]
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.barh(np.arange(len(rows)), values, color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(np.arange(len(rows)), labels)
    axis.set_xlabel("ROC-AUC: Q-Former output minus Q-Former input")
    axis.set_title("Existing layer screen: information change across a 2-layer Q-Former")
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)

    result = {
        "status": "passed",
        "gpu_used": False,
        "source": str(INPUT),
        "label_source": source.get("label_source"),
        "not_independent_human_gold": source.get("not_independent_human_gold"),
        "task_count": len(rows),
        "rows": rows,
        "families": families,
        "interpretation": [
            "This is a layer-information screen, not a causal depth test.",
            "Negative delta means a simple probe read less of the provisional label after Q-Former; it does not prove the layer is incorrect.",
            "A deeper-Q-Former experiment still requires fixed downstream modules and trained extra layers.",
        ],
        "figure": str(FIGURE),
    }
    write_exclusive(RESULT, result)
    print(json.dumps({"status": "passed", "task_count": len(rows), "result": str(RESULT), "figure": str(FIGURE)}))


if __name__ == "__main__":
    main()
