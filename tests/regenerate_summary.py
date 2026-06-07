"""Regenerate benchmark_summary.png with corrected y-axis interpretation.

The original plot showed "Mean ΔCD vs Baseline (%)" which reads as "worse"
when the value is negative (e.g., -86%) even though it actually means
"improvement of 86%" (CD dropped from 0.5 to 0.07).

This version uses absolute ΔCD (CD reduction in counts), which is
semantically unambiguous and is the standard interpretation in the
optimization literature.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

PALETTE = {
    "baseline": "#3370AC",
    "rule": "#D44B3F",
    "threshold": "#E8A838",
    "mlp": "#2A8C6A",
}
LABELS = {
    "baseline": "Baseline (8w IPOPT)",
    "rule": "Rule",
    "threshold": "Threshold",
    "mlp": "Adaptive Router",
}
METHODS = ["baseline", "rule", "threshold", "mlp"]
SERIF = "serif"


def get(df: pd.DataFrame, method: str, airfoil: str) -> dict | None:
    rows = df[(df["method"] == method) & (df["airfoil"] == airfoil)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def main() -> None:
    df = pd.read_csv(RESULTS / "benchmark_stats.csv")
    with open(DATA / "benchmark_airfoils.json") as f:
        cats = json.load(f)
    normal = cats["normal"]
    hard = cats["hard"]

    cat_afs = [normal, hard]
    cat_names = ["Normal", "Hard"]
    x = np.arange(2)
    w = 0.21

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.40))

    # (a) Mean absolute CD reduction (positive = improvement)
    ax = axes[0]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            imps = []
            for af in afs:
                base = get(df, "baseline", af)
                meth = get(df, m, af)
                if base is None or meth is None:
                    continue
                base_cd = base["cd_mean"]
                meth_cd = meth["cd_mean"]
                if not np.isfinite(meth_cd) or meth_cd > 1e10:
                    continue
                if not np.isfinite(base_cd) or base_cd > 1e10 or base_cd <= 0:
                    # Baseline failed (cd = inf): cannot compute absolute reduction.
                    # Skip rather than imputing an extreme value.
                    continue
                imps.append(base_cd - meth_cd)
            vals.append(np.mean(imps) * 1e4 if imps else 0)  # counts × 1e4
        ax.bar(x + (i - 1.5) * w, vals, w, color=PALETTE[m],
               edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=8, fontfamily=SERIF)
    ax.set_ylabel("Mean $C_D$ reduction (counts $\\times 10^4$)", fontsize=8, fontfamily=SERIF)
    ax.set_title("(a) CD Improvement vs. Baseline", fontsize=9, fontfamily=SERIF, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, direction="in")
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.25), ncol=4)

    # (b) Time speedup (baseline time / method time; > 1 means method is faster)
    ax = axes[1]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            base_times = []
            meth_times = []
            for af in afs:
                base = get(df, "baseline", af)
                meth = get(df, m, af)
                if base and np.isfinite(base["time_mean"]) and base["time_mean"] > 0:
                    base_times.append(base["time_mean"])
                if meth and meth["time_mean"] > 0:
                    meth_times.append(meth["time_mean"])
            mean_base = np.mean(base_times) if base_times else 0
            mean_meth = np.mean(meth_times) if meth_times else 1
            vals.append(mean_base / mean_meth if mean_meth > 0 else 1.0)
        ax.bar(x + (i - 1.5) * w, vals, w, color=PALETTE[m],
               edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.axhline(y=1.0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=8, fontfamily=SERIF)
    ax.set_ylabel("Time ratio (baseline / method)", fontsize=8, fontfamily=SERIF)
    ax.set_title("(b) Relative Time", fontsize=9, fontfamily=SERIF, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, direction="in")
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)

    # (c) Success rate
    ax = axes[2]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            rates = [get(df, m, af) for af in afs]
            vals.append(np.mean([r["success_rate"] for r in rates if r]) * 100)
        ax.bar(x + (i - 1.5) * w, vals, w, color=PALETTE[m],
               edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=8, fontfamily=SERIF)
    ax.set_ylabel("Success rate (%)", fontsize=8, fontfamily=SERIF)
    ax.set_title("(c) Success Rate", fontsize=9, fontfamily=SERIF, pad=4)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, direction="in")
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)

    fig.suptitle("Cross-Category Summary (105 airfoils)",
                 fontsize=10, fontfamily=SERIF, fontweight="bold", y=1.06)
    plt.savefig(RESULTS / "benchmark_summary.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved benchmark_summary.png")


if __name__ == "__main__":
    main()
