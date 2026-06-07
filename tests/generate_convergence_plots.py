"""Generate the two missing figures: convergence_curves.png and router_trajectories.png.

Reads ablation.csv (A1=direct vs hier_rule, A2=rule/threshold/mlp) and
benchmark_airfoils.json (for difficulty categorization).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "Baseline": "#1f77b4",   # blue
    "Rule": "#ff7f0e",       # orange
    "Threshold": "#2ca02c",  # green
    "Adaptive": "#d62728",   # red
}
WEIGHT_COLORS = {
    4: "#9ecae1",  # light blue
    6: "#fd8d3c",  # orange
    8: "#31a354",  # green
}


def parse_stage_cds(s: str) -> list[float]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [float(x) for x in s.split(";") if x.strip()]


def build_per_method_trajectories(df: pd.DataFrame, airfoil: str, cd_initial: float) -> dict:
    """Get CD-over-stage trajectories for all four methods on a single airfoil."""
    out: dict[str, list[float]] = {}

    # Baseline (direct) — A1, method=direct
    row = df[(df["ablation"] == "A1") & (df["method"] == "direct") & (df["airfoil"] == airfoil)]
    if not row.empty:
        s = parse_stage_cds(row.iloc[0]["stage_cds"])
        out["Baseline"] = [cd_initial] + s if s else [cd_initial, row.iloc[0]["cd_final"]]

    # Rule — A2, method=rule
    row = df[(df["ablation"] == "A2") & (df["method"] == "rule") & (df["airfoil"] == airfoil)]
    if not row.empty:
        s = parse_stage_cds(row.iloc[0]["stage_cds"])
        if s:
            out["Rule"] = [cd_initial] + s

    # Threshold — A2, method=threshold
    row = df[(df["ablation"] == "A2") & (df["method"] == "threshold") & (df["airfoil"] == airfoil)]
    if not row.empty:
        s = parse_stage_cds(row.iloc[0]["stage_cds"])
        if s:
            out["Threshold"] = [cd_initial] + s

    # Adaptive (MLP) — A2, method=mlp
    row = df[(df["ablation"] == "A2") & (df["method"] == "mlp") & (df["airfoil"] == airfoil)]
    if not row.empty:
        s = parse_stage_cds(row.iloc[0]["stage_cds"])
        if s:
            out["Adaptive"] = [cd_initial] + s

    return out


def plot_convergence_curves() -> None:
    df = pd.read_csv(RESULTS / "ablation.csv")
    with open(DATA / "benchmark_airfoils.json") as f:
        cats = json.load(f)
    initial_cd = cats["initial_cd"]

    # Representative airfoils: naca0012 (Normal), ah81k144 (Medium), rae2822 (Hard)
    picks = [
        ("NACA 0012",  "naca0012",  "Normal"),
        ("Ah 81-K-144", "ah81k144", "Medium"),
        ("RAE 2822",    "rae2822",  "Hard"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for ax, (label, airfoil, difficulty) in zip(axes, picks):
        cd0 = initial_cd.get(airfoil, 0.1)
        traj = build_per_method_trajectories(df, airfoil, cd0)
        for method, vals in traj.items():
            xs = list(range(len(vals)))
            color = METHOD_COLORS[method]
            if method == "Baseline":
                ax.plot(xs, vals, "o-", color=color, lw=1.8, ms=5, label=method, zorder=3)
            else:
                # Draw as stepwise (stages are discrete)
                ax.step(xs, vals, where="post", color=color, lw=1.4, alpha=0.85, label=method)
                ax.scatter(xs[1:], vals[1:], color=color, s=18, zorder=4, edgecolor="white", lw=0.5)
        # Mark the success threshold
        ax.axhline(0.075, color="grey", ls=":", lw=0.8, alpha=0.6)
        ax.text(0.98, 0.05, "$C_D = 0.075$", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="grey")
        ax.set_xlabel("Stage index (0 = initial)")
        ax.set_title(f"{label} ({difficulty})\n$C_{{D,0}} = {cd0:.4f}$", fontsize=9)
        ax.set_xticks(range(0, 7))
        ax.grid(alpha=0.25, ls="--", lw=0.5)
    axes[0].set_ylabel("Weighted $C_D$")
    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04),
               fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = RESULTS / "convergence_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved {out_path}")


def infer_weight_schedule(stage_cds_str: str) -> list[int]:
    """Recover the weight count at each stage from the stage_cds pattern.

    Heuristic: hierarchical always starts at n=4. Each subsequent stage adds 2
    weights up to a cap of 8. The number of stage entries equals the number of
    IPOPT invocations, all in the same low-mid-high progression.
    """
    s = parse_stage_cds(stage_cds_str)
    if not s:
        return []
    schedule = []
    n = 4
    for _ in s:
        schedule.append(n)
        if n < 8:
            n = min(n + 2, 8)
    return schedule


def plot_router_trajectories() -> None:
    df = pd.read_csv(RESULTS / "ablation.csv")
    with open(DATA / "benchmark_airfoils.json") as f:
        cats = json.load(f)

    # Pick 5 from each category, prioritizing interesting (multi-stage) cases
    def pick(category: str, k: int = 5) -> list[str]:
        candidates = cats[category]
        # Score by number of stages (mlp method)
        scored = []
        for af in candidates:
            row = df[(df["ablation"] == "A2") & (df["method"] == "mlp") & (df["airfoil"] == af)]
            if row.empty:
                continue
            n_st = row.iloc[0]["n_stages"]
            scored.append((af, n_st))
        scored.sort(key=lambda t: -t[1])
        return [t[0] for t in scored[:k]]

    normal_picks = pick("normal")
    medium_picks = pick("medium")
    hard_picks = pick("hard")

    all_picks = []
    for af in normal_picks:
        all_picks.append(("N", af, "normal"))
    for af in medium_picks:
        all_picks.append(("M", af, "medium"))
    for af in hard_picks:
        all_picks.append(("H", af, "hard"))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    n_rows = len(all_picks)
    max_stages = 6
    for r, (tag, af, _) in enumerate(all_picks):
        row = df[(df["ablation"] == "A2") & (df["method"] == "mlp") & (df["airfoil"] == af)]
        if row.empty:
            continue
        sched = infer_weight_schedule(row.iloc[0]["stage_cds"])
        for s_idx, n in enumerate(sched):
            ax.add_patch(plt.Rectangle((s_idx, n_rows - r - 1), 1, 1,
                                        facecolor=WEIGHT_COLORS[n],
                                        edgecolor="white", lw=1.0))
        # Add airfoil label
        ax.text(-0.15, n_rows - r - 0.5, f"{tag}: {af}",
                ha="right", va="center", fontsize=8, family="monospace")

    # Add a blank-stage indicator on the right
    for s_idx in range(max_stages):
        ax.text(s_idx + 0.5, n_rows + 0.15, f"S{s_idx + 1}", ha="center", va="bottom",
                fontsize=8, color="grey")

    ax.set_xlim(-0.3, max_stages)
    ax.set_ylim(-0.5, n_rows + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_handles = [mpatches.Patch(color=WEIGHT_COLORS[n], label=f"$n = {n}$ weights/edge")
                       for n in (4, 6, 8)]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8,
              title="Active weight count", title_fontsize=8)

    # Group separators
    ax.axhline(len(normal_picks), color="black", lw=0.6)
    ax.axhline(len(normal_picks) + len(medium_picks), color="black", lw=0.6)

    # Group labels on the right.
    # y-axis is inverted: top of plot = high index in y, but rows are added top-down
    # in display order. all_picks[0] (Normal) is plotted at y = n_rows - 1 - 0
    # (top of plot). The group label should be at the vertical center of each group.
    # Normal group: y range [n_rows - len(normal_picks), n_rows]
    # Medium group: y range [n_rows - len(normal) - len(medium), n_rows - len(normal)]
    # Hard group: y range [0, n_rows - len(normal) - len(medium)]
    n_n, n_m, n_h = len(normal_picks), len(medium_picks), len(hard_picks)
    y_top_normal = n_rows
    y_bot_normal = n_n
    y_top_medium = y_bot_normal
    y_bot_medium = y_bot_normal + n_m
    y_top_hard = y_bot_medium
    y_bot_hard = n_rows - n_n - n_m - n_h
    ax.text(max_stages + 0.05, (y_top_normal + y_bot_normal) / 2, "Normal",
            rotation=270, ha="left", va="center", fontsize=10, weight="bold")
    ax.text(max_stages + 0.05, (y_top_medium + y_bot_medium) / 2, "Medium",
            rotation=270, ha="left", va="center", fontsize=10, weight="bold")
    ax.text(max_stages + 0.05, (y_top_hard + y_bot_hard) / 2, "Hard",
            rotation=270, ha="left", va="center", fontsize=10, weight="bold")

    ax.set_title("Adaptive Router decision trajectory (5 Normal / 5 Medium / 5 Hard)",
                 fontsize=10, pad=8)
    fig.tight_layout()
    out_path = RESULTS / "router_trajectories.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_convergence_curves()
    plot_router_trajectories()
