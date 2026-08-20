#!/usr/bin/env python3
"""Render compact method comparison for the SNGS-10004 role experiments."""

from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path("/home/tianlin/SoccerMaster")
OUTPUT = REPO / "reports/g10/20260819_qwen_role_swap_sngs10004/method_comparison.png"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    methods = ["PRTReID\nexplicit", "Qwen single-view\nexplicit", "Qwen multi-view\nexplicit", "Qwen single-view\n+ other"]
    recall = [0.0, 37.5, 25.0, 100.0]
    false_reject = [0.0, 0.0, 0.0, 64.1]
    x = range(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    bars = axes[0].bar(x, recall, color=["#68798a", "#4da6ff", "#ffb84d", "#e36b73"])
    axes[0].set_title("Goalkeeper/referee recall (8 tracks)")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_xticks(list(x), methods)
    axes[0].bar_label(bars, fmt="%.1f%%")
    bars = axes[1].bar(x, false_reject, color=["#68798a", "#4da6ff", "#ffb84d", "#e36b73"])
    axes[1].set_title("Outfield false rejection (39 tracks)")
    axes[1].set_ylabel("False rejection (%)")
    axes[1].set_ylim(0, 70)
    axes[1].set_xticks(list(x), methods)
    axes[1].bar_label(bars, fmt="%.1f%%")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle("SNGS-10004 role-module diagnostics", fontsize=14)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
