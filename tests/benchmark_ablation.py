"""
Ablation Study for PIERN Airfoil Optimization System.

Designed for journal publication. Four ablations:

  A1: Hierarchical CST vs Direct (full 8-weight IPOPT)
      - Hypothesis: hierarchical converges faster with comparable final CD

  A2: Router Strategy Effect (rule / threshold / mlp)
      - Hypothesis: learned routers improve speed/quality tradeoff

  A3: Starting CST Dimension (start_weights = 4, 6, 8)
      - Hypothesis: lower starting dimension accelerates early convergence

  A4: Per-Stage CST Dimension Contribution (4w -> 6w -> 8w)
      - Hypothesis: each dimension expansion yields diminishing returns

Usage:
  uv run python tests/benchmark_ablation.py

Output:
  results/ablation_1_hierarchical_vs_direct.png
  results/ablation_2_router_effect.png
  results/ablation_3_starting_dimension.png
  results/ablation_4_dimension_contribution.png
  results/sensitivity.png
  results/ablation.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Publication Style ─────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }
)

# ── Color Palette (distinct, colorblind-friendly) ────────────────────────

COLORS = {
    "direct": "#2166AC",      # blue
    "hierarchical": "#B2182B", # red
    "rule": "#B2182B",         # red
    "threshold": "#D6604D",    # salmon
    "mlp": "#4393C3",          # light blue
    "start4": "#B2182B",       # red
    "start6": "#D6604D",       # salmon
    "start8": "#4393C3",       # light blue
    "stage_4w": "#2166AC",     # blue
    "stage_6w": "#D6604D",     # salmon
    "stage_8w": "#B2182B",     # red
    "initial": "#999999",      # gray
}

MARKERS = {
    "direct": "s",
    "hierarchical": "o",
    "rule": "o",
    "threshold": "D",
    "mlp": "^",
    "start4": "o",
    "start6": "D",
    "start8": "^",
}

# ── Problem Definition ────────────────────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03
BENCHMARK_JSON = Path(__file__).parent.parent / "data" / "benchmark_airfoils.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── Data Structures ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AblationResult:
    """Single ablation run result."""
    ablation: str          # "A1", "A2", "A3", "A4"
    method: str            # method identifier
    airfoil_name: str
    cd_final: float
    cd_initial: float
    time_s: float
    n_stages: int
    stage_cds: tuple       # per-stage CD values (for A4)
    success: bool          # CD < 0.15


@dataclass
class AggregatedStats:
    """Aggregated statistics across airfoils for one method."""
    method: str
    label: str
    n_airfoils: int
    cd_mean: float
    cd_median: float
    cd_std: float
    time_mean: float
    time_median: float
    time_std: float
    stages_mean: float
    success_rate: float
    cd_improvement_pct: float  # vs initial, mean percentage


# ── Benchmark Loading ─────────────────────────────────────────────────────


def load_benchmark() -> tuple[list[str], list[str], list[str]]:
    """Load the fixed benchmark airfoil set."""
    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    return bench["normal"], bench["medium"], bench["hard"]


# ── CD Evaluation ─────────────────────────────────────────────────────────


def evaluate_cd(airfoil) -> float:
    """Evaluate weighted CD using the shared evaluation function."""
    from piern_airfoil.eval import evaluate_weighted_cd

    return evaluate_weighted_cd(airfoil, CL_TARGETS, RE, CL_WEIGHTS, mach=MACH)


# ── IPOPT Suppression ────────────────────────────────────────────────────


def _suppress_ipopt():
    """Suppress IPOPT stdout."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old_stdout


def _restore_stdout(old_fd: int):
    """Restore stdout."""
    os.dup2(old_fd, 1)
    os.close(old_fd)


# ── Optimization Runners ─────────────────────────────────────────────────


def run_direct(airfoil_name: str) -> AblationResult:
    """Run direct 8-weight IPOPT optimization (baseline)."""
    af = asb.KulfanAirfoil(airfoil_name)
    initial_cd = evaluate_cd(af)

    old_fd = _suppress_ipopt()
    t0 = time.perf_counter()
    try:
        from piern_airfoil.optimizer import NeuralOptimizer

        opt = NeuralOptimizer(
            airfoil=af,
            CL_targets=CL_TARGETS,
            CL_weights=CL_WEIGHTS,
            RE=RE,
            mach=MACH,
        )
        opt.update()
        elapsed = time.perf_counter() - t0
        _restore_stdout(old_fd)

        final_cd = evaluate_cd(opt.airfoil)
        return AblationResult(
            ablation="A1",
            method="direct",
            airfoil_name=airfoil_name,
            cd_final=final_cd,
            cd_initial=initial_cd,
            time_s=elapsed,
            n_stages=1,
            stage_cds=(final_cd,),
            success=final_cd < 0.15,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        try:
            _restore_stdout(old_fd)
        except OSError:
            pass
        return AblationResult(
            ablation="A1",
            method="direct",
            airfoil_name=airfoil_name,
            cd_final=float("inf"),
            cd_initial=initial_cd,
            time_s=elapsed,
            n_stages=0,
            stage_cds=(),
            success=False,
        )


def run_hierarchical(
    airfoil_name: str,
    router_mode: str = "rule",
    start_weights: int = 4,
    method_label: str | None = None,
    ablation_tag: str = "A1",
    threshold: float | None = None,
) -> AblationResult:
    """Run hierarchical CST optimization with configurable router and start dimension.

    Args:
        threshold: Override the router's improvement_threshold. Only used for
            rule/threshold modes. If None, uses the default (0.01 or learned).
    """
    af = asb.KulfanAirfoil(airfoil_name)
    initial_cd = evaluate_cd(af)

    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
    from piern.router.opt_router import OptRouter

    default_thresh = threshold if threshold is not None else 0.01

    if router_mode == "mlp":
        try:
            router = OptRouter.from_mlp()
        except FileNotFoundError:
            router = OptRouter(improvement_threshold=default_thresh, mode="rule")
    elif router_mode == "threshold":
        try:
            router = OptRouter.from_trained()
        except FileNotFoundError:
            router = OptRouter(improvement_threshold=default_thresh, mode="rule")
    else:
        router = OptRouter(improvement_threshold=default_thresh, mode="rule")

    old_fd = _suppress_ipopt()
    t0 = time.perf_counter()
    try:
        optimizer = AdaptiveHierarchicalOptimizer(
            CL_targets=CL_TARGETS,
            CL_weights=CL_WEIGHTS,
            Re=RE,
            mach=MACH,
            start_weights=start_weights,
            router=router,
        )
        result = optimizer.optimize(af)
        elapsed = time.perf_counter() - t0
        _restore_stdout(old_fd)

        stage_cds = tuple(s.cd for s in result.stages)
        label = method_label or f"hier_{router_mode}_sw{start_weights}"

        return AblationResult(
            ablation=ablation_tag,
            method=label,
            airfoil_name=airfoil_name,
            cd_final=result.final_cd,
            cd_initial=initial_cd,
            time_s=elapsed,
            n_stages=len(result.stages),
            stage_cds=stage_cds,
            success=result.final_cd < 0.15,
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        try:
            _restore_stdout(old_fd)
        except OSError:
            pass
        label = method_label or f"hier_{router_mode}_sw{start_weights}"
        return AblationResult(
            ablation=ablation_tag,
            method=label,
            airfoil_name=airfoil_name,
            cd_final=float("inf"),
            cd_initial=initial_cd,
            time_s=elapsed,
            n_stages=0,
            stage_cds=(),
            success=False,
        )


# ── Statistical Aggregation ──────────────────────────────────────────────


def aggregate(
    results: list[AblationResult], method: str, label: str
) -> AggregatedStats:
    """Aggregate results for a single method across airfoils."""
    runs = [r for r in results if r.method == method]
    if not runs:
        return AggregatedStats(
            method=method, label=label, n_airfoils=0,
            cd_mean=0, cd_median=0, cd_std=0,
            time_mean=0, time_median=0, time_std=0,
            stages_mean=0, success_rate=0, cd_improvement_pct=0,
        )

    successes = np.array([r.success for r in runs])
    ok_runs = [r for r in runs if r.success]

    if not ok_runs:
        return AggregatedStats(
            method=method, label=label, n_airfoils=len(runs),
            cd_mean=float("inf"), cd_median=float("inf"), cd_std=0,
            time_mean=0, time_median=0, time_std=0,
            stages_mean=0, success_rate=float(np.mean(successes)),
            cd_improvement_pct=0,
        )

    cds = np.array([r.cd_final for r in ok_runs])
    times = np.array([r.time_s for r in ok_runs])
    stages = np.array([r.n_stages for r in ok_runs])
    improvements = np.array(
        [
            (r.cd_initial - r.cd_final) / r.cd_initial * 100
            for r in ok_runs
            if r.cd_initial > 0
        ]
    )

    return AggregatedStats(
        method=method,
        label=label,
        n_airfoils=len(runs),
        cd_mean=float(np.mean(cds)),
        cd_median=float(np.median(cds)),
        cd_std=float(np.std(cds)),
        time_mean=float(np.mean(times)),
        time_median=float(np.median(times)),
        time_std=float(np.std(times)),
        stages_mean=float(np.mean(stages)),
        success_rate=float(np.mean(successes)),
        cd_improvement_pct=float(np.mean(improvements)) if len(improvements) > 0 else 0.0,
    )


# ── Ablation 1: Hierarchical vs Direct ──────────────────────────────────


def run_ablation_1(airfoils: list[str]) -> list[AblationResult]:
    """A1: Hierarchical CST (4->8, rule router) vs Direct 8-weight IPOPT."""
    print("\n" + "=" * 70)
    print("ABLATION 1: Hierarchical CST vs Direct Optimization")
    print("=" * 70)
    print(f"Airfoils: {len(airfoils)}")
    print(f"Methods:  direct (8w IPOPT), hierarchical (4->8, rule router)")
    print()

    results: list[AblationResult] = []
    total = len(airfoils) * 2

    for i, name in enumerate(airfoils):
        # Direct
        print(f"  [A1 {2*i+1}/{total}] {name} direct...", end=" ", flush=True)
        r = run_direct(name)
        results.append(r)
        print(f"CD={r.cd_final:.6f} {r.time_s:.1f}s" if r.success else "FAILED")

        # Hierarchical
        print(f"  [A1 {2*i+2}/{total}] {name} hierarchical...", end=" ", flush=True)
        r = run_hierarchical(name, router_mode="rule", start_weights=4, ablation_tag="A1")
        results.append(r)
        status = f"CD={r.cd_final:.6f} {r.time_s:.1f}s stages={r.n_stages}" if r.success else "FAILED"
        print(status)

    # Summary
    s_direct = aggregate(results, "direct", "Direct (8w)")
    s_hier = aggregate(results, "hier_rule_sw4", "Hierarchical (4->8)")

    print(f"\n{'Method':<28} {'CD Mean':>10} {'CD Std':>10} {'Time Mean':>10} {'Stages':>8} {'Success':>8}")
    print("-" * 78)
    for s in [s_direct, s_hier]:
        print(
            f"{s.label:<28} {s.cd_mean:>10.6f} {s.cd_std:>10.6f} "
            f"{s.time_mean:>10.1f} {s.stages_mean:>8.1f} {s.success_rate:>8.0%}"
        )

    # Speedup
    if s_direct.time_mean > 0:
        speedup = s_direct.time_mean / s_hier.time_mean if s_hier.time_mean > 0 else 0
        cd_diff = (s_hier.cd_mean - s_direct.cd_mean) / s_direct.cd_mean * 100
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"CD difference: {cd_diff:+.2f}% (negative = hierarchical better)")

    return results


def visualize_ablation_1(results: list[AblationResult], airfoils: list[str]):
    """Publication figure: Hierarchical vs Direct."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    methods = ["direct", "hier_rule_sw4"]
    labels = ["Direct (8w)", "Hierarchical (4→8)"]
    colors = [COLORS["direct"], COLORS["hierarchical"]]

    # ── Panel (a): CD comparison scatter ──
    ax = axes[0]
    sorted_airfoils = sorted(
        airfoils,
        key=lambda n: next(
            (r.cd_initial for r in results if r.airfoil_name == n), 0
        ),
    )
    for method, label, color in zip(methods, labels, colors):
        cds = []
        for name in sorted_airfoils:
            run = next(
                (r for r in results if r.method == method and r.airfoil_name == name),
                None,
            )
            cds.append(run.cd_final if run and run.success else np.nan)
        xs = np.arange(len(sorted_airfoils))
        ax.scatter(xs, cds, c=color, label=label, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("Airfoil index (sorted by initial CD)")
    ax.set_ylabel("Final Weighted CD")
    ax.set_title("(a) Final CD")
    ax.legend(loc="upper left")

    # ── Panel (b): Time comparison ──
    ax = axes[1]
    data_times = []
    for method in methods:
        times = [r.time_s for r in results if r.method == method and r.success]
        data_times.append(times)
    bp = ax.boxplot(
        data_times, labels=labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Optimization Time (s)")
    ax.set_title("(b) Runtime Distribution")

    # ── Panel (c): CD improvement over initial ──
    ax = axes[2]
    improvements = {}
    for method, label, color in zip(methods, labels, colors):
        imps = [
            (r.cd_initial - r.cd_final) / r.cd_initial * 100
            for r in results
            if r.method == method and r.success and r.cd_initial > 0
        ]
        improvements[label] = imps
    bp = ax.boxplot(
        [improvements[l] for l in labels],
        labels=labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("CD Improvement over Initial (%)")
    ax.set_title("(c) Optimization Gain")

    fig.suptitle("Ablation 1: Hierarchical CST vs Direct Optimization", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = RESULTS_DIR / "ablation_1_hierarchical_vs_direct.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Ablation 2: Router Effect ────────────────────────────────────────────


def run_ablation_2(airfoils: list[str]) -> list[AblationResult]:
    """A2: Compare rule / threshold / mlp routing strategies."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Router Strategy Effect")
    print("=" * 70)
    print(f"Airfoils: {len(airfoils)}")
    print(f"Methods:  rule, threshold, mlp (all start from 4 weights)")
    print()

    router_modes = ["rule", "threshold", "mlp"]
    results: list[AblationResult] = []
    total = len(airfoils) * len(router_modes)

    for i, name in enumerate(airfoils):
        for j, mode in enumerate(router_modes):
            idx = i * len(router_modes) + j + 1
            print(f"  [A2 {idx}/{total}] {name} {mode}...", end=" ", flush=True)
            r = run_hierarchical(
                name, router_mode=mode, start_weights=4,
                method_label=mode, ablation_tag="A2",
            )
            results.append(r)
            status = f"CD={r.cd_final:.6f} {r.time_s:.1f}s stages={r.n_stages}" if r.success else "FAILED"
            print(status)

    # Summary
    s_rule = aggregate(results, "rule", "Rule (threshold=0.01)")
    s_thresh = aggregate(results, "threshold", "Threshold (learned)")
    s_mlp = aggregate(results, "mlp", "MLP (learned)")

    print(f"\n{'Method':<28} {'CD Mean':>10} {'CD Std':>10} {'Time Mean':>10} {'Stages':>8} {'Success':>8}")
    print("-" * 78)
    for s in [s_rule, s_thresh, s_mlp]:
        print(
            f"{s.label:<28} {s.cd_mean:>10.6f} {s.cd_std:>10.6f} "
            f"{s.time_mean:>10.1f} {s.stages_mean:>8.1f} {s.success_rate:>8.0%}"
        )

    return results


def visualize_ablation_2(results: list[AblationResult], airfoils: list[str]):
    """Publication figure: Router strategy comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    modes = ["rule", "threshold", "mlp"]
    labels = ["Rule (fixed)", "Threshold (learned)", "MLP (learned)"]
    colors = [COLORS["rule"], COLORS["threshold"], COLORS["mlp"]]
    markers = [MARKERS["rule"], MARKERS["threshold"], MARKERS["mlp"]]

    # Sort airfoils by initial CD for consistent x-axis
    sorted_airfoils = sorted(
        airfoils,
        key=lambda n: next(
            (r.cd_initial for r in results if r.airfoil_name == n), 0
        ),
    )

    # ── Panel (a): CD per airfoil ──
    ax = axes[0, 0]
    for mode, label, color, marker in zip(modes, labels, colors, markers):
        cds = []
        for name in sorted_airfoils:
            run = next(
                (r for r in results if r.method == mode and r.airfoil_name == name), None
            )
            cds.append(run.cd_final if run and run.success else np.nan)
        xs = np.arange(len(sorted_airfoils))
        ax.scatter(xs, cds, c=color, label=label, marker=marker, alpha=0.6, s=18, edgecolors="none")
    ax.set_xlabel("Airfoil index")
    ax.set_ylabel("Final Weighted CD")
    ax.set_title("(a) Final CD by Router Strategy")
    ax.legend(loc="upper left", fontsize=8)

    # ── Panel (b): Time distribution ──
    ax = axes[0, 1]
    data_times = []
    for mode in modes:
        times = [r.time_s for r in results if r.method == mode and r.success]
        data_times.append(times)
    bp = ax.boxplot(
        data_times, labels=labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Runtime Distribution")

    # ── Panel (c): Number of stages ──
    ax = axes[1, 0]
    data_stages = []
    for mode in modes:
        stages = [r.n_stages for r in results if r.method == mode and r.success]
        data_stages.append(stages)
    bp = ax.boxplot(
        data_stages, labels=labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Number of Stages")
    ax.set_title("(c) Stages to Convergence")

    # ── Panel (d): CD improvement vs initial ──
    ax = axes[1, 1]
    data_imps = []
    for mode in modes:
        imps = [
            (r.cd_initial - r.cd_final) / r.cd_initial * 100
            for r in results
            if r.method == mode and r.success and r.cd_initial > 0
        ]
        data_imps.append(imps)
    bp = ax.boxplot(
        data_imps, labels=labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("CD Improvement over Initial (%)")
    ax.set_title("(d) Optimization Gain")

    fig.suptitle("Ablation 2: Router Strategy Effect", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / "ablation_2_router_effect.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Ablation 3: Starting Dimension ───────────────────────────────────────


def run_ablation_3(airfoils: list[str]) -> list[AblationResult]:
    """A3: Test start_weights = 4, 6, 8 with the same rule router."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Starting CST Dimension")
    print("=" * 70)
    print(f"Airfoils: {len(airfoils)}")
    print(f"Methods:  start_weights=4, 6, 8 (all use rule router)")
    print()

    starts = [4, 6, 8]
    results: list[AblationResult] = []
    total = len(airfoils) * len(starts)

    for i, name in enumerate(airfoils):
        for j, sw in enumerate(starts):
            idx = i * len(starts) + j + 1
            label = f"sw{sw}"
            print(f"  [A3 {idx}/{total}] {name} start={sw}...", end=" ", flush=True)
            r = run_hierarchical(
                name, router_mode="rule", start_weights=sw,
                method_label=label, ablation_tag="A3",
            )
            results.append(r)
            status = f"CD={r.cd_final:.6f} {r.time_s:.1f}s stages={r.n_stages}" if r.success else "FAILED"
            print(status)

    # Summary
    for sw in starts:
        s = aggregate(results, f"sw{sw}", f"Start={sw} weights")
        if sw == 4:
            stats_list = [s]
        else:
            stats_list.append(s)

    print(f"\n{'Method':<28} {'CD Mean':>10} {'CD Std':>10} {'Time Mean':>10} {'Stages':>8} {'Success':>8}")
    print("-" * 78)
    for s in stats_list:
        print(
            f"{s.label:<28} {s.cd_mean:>10.6f} {s.cd_std:>10.6f} "
            f"{s.time_mean:>10.1f} {s.stages_mean:>8.1f} {s.success_rate:>8.0%}"
        )

    return results


def visualize_ablation_3(results: list[AblationResult], airfoils: list[str]):
    """Publication figure: Starting dimension comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    starts = [4, 6, 8]
    labels = ["Start = 4 weights", "Start = 6 weights", "Start = 8 weights"]
    colors = [COLORS["start4"], COLORS["start6"], COLORS["start8"]]
    markers = [MARKERS["start4"], MARKERS["start6"], MARKERS["start8"]]

    sorted_airfoils = sorted(
        airfoils,
        key=lambda n: next(
            (r.cd_initial for r in results if r.airfoil_name == n), 0
        ),
    )

    # ── Panel (a): CD per airfoil ──
    ax = axes[0, 0]
    for sw, label, color, marker in zip(starts, labels, colors, markers):
        method = f"sw{sw}"
        cds = []
        for name in sorted_airfoils:
            run = next(
                (r for r in results if r.method == method and r.airfoil_name == name), None
            )
            cds.append(run.cd_final if run and run.success else np.nan)
        xs = np.arange(len(sorted_airfoils))
        ax.scatter(xs, cds, c=color, label=label, marker=marker, alpha=0.6, s=18, edgecolors="none")
    ax.set_xlabel("Airfoil index")
    ax.set_ylabel("Final Weighted CD")
    ax.set_title("(a) Final CD by Starting Dimension")
    ax.legend(loc="upper left", fontsize=8)

    # ── Panel (b): Time distribution ──
    ax = axes[0, 1]
    data_times = []
    for sw in starts:
        method = f"sw{sw}"
        times = [r.time_s for r in results if r.method == method and r.success]
        data_times.append(times)
    bp = ax.boxplot(
        data_times, labels=[f"sw={sw}" for sw in starts], patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Time (s)")
    ax.set_title("(b) Runtime Distribution")

    # ── Panel (c): Stages ──
    ax = axes[1, 0]
    data_stages = []
    for sw in starts:
        method = f"sw{sw}"
        stages = [r.n_stages for r in results if r.method == method and r.success]
        data_stages.append(stages)
    bp = ax.boxplot(
        data_stages, labels=[f"sw={sw}" for sw in starts], patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Number of Stages")
    ax.set_title("(c) Stages to Convergence")

    # ── Panel (d): CD vs Time Pareto ──
    ax = axes[1, 1]
    for sw, label, color, marker in zip(starts, labels, colors, markers):
        method = f"sw{sw}"
        runs_ok = [r for r in results if r.method == method and r.success]
        if runs_ok:
            mean_cd = np.mean([r.cd_final for r in runs_ok])
            mean_time = np.mean([r.time_s for r in runs_ok])
            ax.scatter(
                mean_time, mean_cd, c=color, marker=marker, s=120,
                label=label, edgecolors="black", linewidths=0.8, zorder=5,
            )
    ax.set_xlabel("Mean Time (s)")
    ax.set_ylabel("Mean Weighted CD")
    ax.set_title("(d) Pareto Front (CD vs Time)")
    ax.legend(fontsize=8)

    fig.suptitle("Ablation 3: Starting CST Dimension Effect", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / "ablation_3_starting_dimension.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Ablation 4: Dimension Contribution ──────────────────────────────────


def run_ablation_4(airfoils: list[str]) -> list[AblationResult]:
    """A4: Track CD at each stage to measure per-dimension contribution.

    Uses rule router with start_weights=4 so we can observe 4w -> 6w -> 8w transitions.
    """
    print("\n" + "=" * 70)
    print("ABLATION 4: CST Dimension Contribution (Per-Stage CD)")
    print("=" * 70)
    print(f"Airfoils: {len(airfoils)}")
    print(f"Method:   hierarchical rule router, start=4, tracking each stage")
    print()

    results: list[AblationResult] = []
    total = len(airfoils)

    for i, name in enumerate(airfoils):
        print(f"  [A4 {i+1}/{total}] {name}...", end=" ", flush=True)
        r = run_hierarchical(
            name, router_mode="rule", start_weights=4,
            method_label="hier_rule_sw4", ablation_tag="A4",
        )
        results.append(r)
        if r.success and r.stage_cds:
            stage_str = " -> ".join(f"{cd:.5f}" for cd in r.stage_cds)
            print(f"stages: [{stage_str}]  final={r.cd_final:.6f} {r.time_s:.1f}s")
        else:
            print("FAILED")

    # Summary of per-stage contributions
    ok_runs = [r for r in results if r.success and len(r.stage_cds) >= 2]
    if ok_runs:
        # Compute per-stage improvement
        first_stage_cds = [r.stage_cds[0] for r in ok_runs]
        final_cds = [r.cd_final for r in ok_runs]
        initial_cds = [r.cd_initial for r in ok_runs]

        print(f"\nStage contribution summary ({len(ok_runs)} successful runs):")
        print(f"  Initial CD (mean):     {np.mean(initial_cds):.6f}")
        print(f"  After 1st stage (mean): {np.mean(first_stage_cds):.6f}")
        print(f"  Final CD (mean):       {np.mean(final_cds):.6f}")
        print(f"  1st stage improvement: {(np.mean(initial_cds) - np.mean(first_stage_cds)) / np.mean(initial_cds) * 100:.2f}%")
        print(f"  Further improvement:   {(np.mean(first_stage_cds) - np.mean(final_cds)) / np.mean(first_stage_cds) * 100:.2f}%")

    return results


def visualize_ablation_4(results: list[AblationResult], airfoils: list[str]):
    """Publication figure: Per-stage dimension contribution."""
    ok_runs = [r for r in results if r.success and len(r.stage_cds) >= 2]
    if not ok_runs:
        print("  Skipped: no successful runs with multi-stage data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ── Panel (a): Stage-wise CD trajectory (sample of airfoils) ──
    ax = axes[0, 0]
    # Pick a representative sample: 3 from each difficulty tier
    sample_names = []
    for tier_start, tier_end in [(0, 30), (30, 74), (74, 105)]:
        tier = [n for n in airfoils[tier_start:tier_end] if any(r.airfoil_name == n and r.success and len(r.stage_cds) >= 2 for r in results)]
        sample_names.extend(tier[:3])

    cmap = plt.cm.viridis
    for idx, name in enumerate(sample_names):
        run = next(r for r in results if r.airfoil_name == name and r.success)
        stages = np.arange(1, len(run.stage_cds) + 1)
        color = cmap(idx / max(len(sample_names) - 1, 1))
        ax.plot(stages, run.stage_cds, "-o", color=color, alpha=0.7, markersize=4, linewidth=1)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Weighted CD")
    ax.set_title("(a) Per-Stage CD Trajectory (sample)")
    ax.set_xticks(range(1, 7))

    # ── Panel (b): Stage-wise CD reduction (all airfoils, boxplot) ──
    ax = axes[0, 1]
    # Compute CD reduction at each stage transition
    max_stages = max(len(r.stage_cds) for r in ok_runs)
    stage_reductions = {i: [] for i in range(max_stages)}
    for r in ok_runs:
        for i, cd in enumerate(r.stage_cds):
            stage_reductions[i].append(cd)

    stage_labels = [f"Stage {i+1}" for i in range(max_stages)]
    stage_data = [stage_reductions[i] for i in range(max_stages) if stage_reductions[i]]
    stage_labels = stage_labels[:len(stage_data)]

    bp = ax.boxplot(
        stage_data, labels=stage_labels, patch_artist=True, widths=0.5,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    stage_colors_list = [COLORS["stage_4w"], COLORS["stage_4w"], COLORS["stage_6w"],
                         COLORS["stage_6w"], COLORS["stage_8w"], COLORS["stage_8w"]]
    for i, (patch, color) in enumerate(zip(bp["boxes"], stage_colors_list[:len(bp["boxes"])])):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Weighted CD")
    ax.set_title("(b) CD Distribution by Stage")

    # ── Panel (c): Marginal improvement per stage ──
    ax = axes[1, 0]
    marginal_imps = []
    for r in ok_runs:
        imps = []
        for i in range(1, len(r.stage_cds)):
            imp = (r.stage_cds[i - 1] - r.stage_cds[i]) / r.stage_cds[i - 1] * 100
            imps.append(imp)
        marginal_imps.append(imps)

    # Average marginal improvement at each transition
    max_transitions = max(len(imps) for imps in marginal_imps)
    avg_imps = []
    imp_stds = []
    for t in range(max_transitions):
        vals = [imps[t] for imps in marginal_imps if len(imps) > t]
        avg_imps.append(np.mean(vals) if vals else 0)
        imp_stds.append(np.std(vals) if vals else 0)

    x = np.arange(1, max_transitions + 1)
    ax.bar(
        x, avg_imps, yerr=imp_stds, capsize=4,
        color=[COLORS["stage_4w"], COLORS["stage_6w"], COLORS["stage_8w"]][:max_transitions],
        alpha=0.7, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Stage Transition")
    ax.set_ylabel("Marginal CD Improvement (%)")
    ax.set_title("(c) Average Marginal Improvement per Transition")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}->S{i+1}" for i in range(max_transitions)])

    # ── Panel (d): Cumulative improvement breakdown ──
    ax = axes[1, 1]
    # For airfoils with exactly 3+ stages, show improvement breakdown
    three_stage = [r for r in ok_runs if len(r.stage_cds) >= 3]
    if three_stage:
        initial_mean = np.mean([r.cd_initial for r in three_stage])
        stage_means = []
        for s_idx in range(min(3, max(len(r.stage_cds) for r in three_stage))):
            cds_at_stage = [r.stage_cds[s_idx] for r in three_stage if len(r.stage_cds) > s_idx]
            stage_means.append(np.mean(cds_at_stage))

        # Stacked bar: improvement from each stage
        cumulative = [initial_mean]
        for sm in stage_means:
            cumulative.append(sm)

        bar_labels = ["Initial"] + [f"After S{i+1}" for i in range(len(stage_means))]
        bar_colors_list = [COLORS["initial"]] + [
            COLORS["stage_4w"], COLORS["stage_6w"], COLORS["stage_8w"]
        ][:len(stage_means)]

        bars = ax.bar(
            range(len(cumulative)), cumulative, 0.6,
            color=bar_colors_list, alpha=0.7, edgecolor="white", linewidth=0.5,
        )
        # Annotate improvement
        for i in range(1, len(cumulative)):
            imp = (cumulative[i - 1] - cumulative[i]) / cumulative[i - 1] * 100
            mid_y = (cumulative[i - 1] + cumulative[i]) / 2
            ax.annotate(
                f"-{imp:.1f}%",
                xy=(i - 0.5, mid_y), fontsize=8, ha="center",
                fontweight="bold", color="#333333",
            )
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel("Weighted CD")
        ax.set_title("(d) Cumulative CD Reduction")

    fig.suptitle("Ablation 4: CST Dimension Contribution Analysis", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / "ablation_4_dimension_contribution.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Sensitivity Analysis ─────────────────────────────────────────────────


def run_sensitivity_analysis() -> list[AblationResult]:
    """Parameter sensitivity sweep over threshold and start_weights.

    Tests:
      1. Threshold sweep: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
         - Uses OptRouter(improvement_threshold=t) for each value
      2. Start weights sweep: [4, 5, 6, 7, 8]
         - Uses AdaptiveHierarchicalOptimizer(start_weights=w) for each value

    Runs on a fixed subset of 10 airfoils (5 normal, 3 medium, 2 hard).
    """
    normal, medium, hard = load_benchmark()
    subset = normal[:5] + medium[:3] + hard[:2]

    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    start_weights_list = [4, 5, 6, 7, 8]

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Threshold & Start Weights Sweep")
    print("=" * 70)
    print(f"Subset:    {len(subset)} airfoils (5 normal, 3 medium, 2 hard)")
    print(f"Threshold: {thresholds}")
    print(f"Weights:   {start_weights_list}")
    print()

    results: list[AblationResult] = []

    # ── Part 1: Threshold sweep ──
    total_thresh = len(subset) * len(thresholds)
    for i, name in enumerate(subset):
        for j, t in enumerate(thresholds):
            idx = i * len(thresholds) + j + 1
            label = f"thresh_{t}"
            print(
                f"  [S-thresh {idx}/{total_thresh}] {name} t={t}...",
                end=" ",
                flush=True,
            )
            r = run_hierarchical(
                name,
                router_mode="rule",
                start_weights=4,
                method_label=label,
                ablation_tag="S-thresh",
                threshold=t,
            )
            results.append(r)
            status = (
                f"CD={r.cd_final:.6f} {r.time_s:.1f}s stages={r.n_stages}"
                if r.success
                else "FAILED"
            )
            print(status)

    # ── Part 2: Start weights sweep ──
    total_sw = len(subset) * len(start_weights_list)
    for i, name in enumerate(subset):
        for j, sw in enumerate(start_weights_list):
            idx = i * len(start_weights_list) + j + 1
            label = f"sens_sw{sw}"
            print(
                f"  [S-sw {idx}/{total_sw}] {name} start={sw}...",
                end=" ",
                flush=True,
            )
            r = run_hierarchical(
                name,
                router_mode="rule",
                start_weights=sw,
                method_label=label,
                ablation_tag="S-sw",
            )
            results.append(r)
            status = (
                f"CD={r.cd_final:.6f} {r.time_s:.1f}s stages={r.n_stages}"
                if r.success
                else "FAILED"
            )
            print(status)

    # ── Summary table ──
    print(f"\n{'Parameter':<28} {'CD Mean':>10} {'Time Mean':>10} {'Stages':>8} {'N':>4} {'Success':>8}")
    print("-" * 72)

    for t in thresholds:
        label = f"thresh_{t}"
        ok = [r for r in results if r.method == label and r.success]
        n = len([r for r in results if r.method == label])
        if ok:
            print(
                f"threshold={t:<20} {np.mean([r.cd_final for r in ok]):>10.6f} "
                f"{np.mean([r.time_s for r in ok]):>10.1f} "
                f"{np.mean([r.n_stages for r in ok]):>8.1f} "
                f"{n:>4} {len(ok)/n:>8.0%}"
            )
        else:
            print(f"threshold={t:<20} {'---':>10} {'---':>10} {'---':>8} {n:>4} {'0%':>8}")

    for sw in start_weights_list:
        label = f"sens_sw{sw}"
        ok = [r for r in results if r.method == label and r.success]
        n = len([r for r in results if r.method == label])
        if ok:
            print(
                f"start_weights={sw:<16} {np.mean([r.cd_final for r in ok]):>10.6f} "
                f"{np.mean([r.time_s for r in ok]):>10.1f} "
                f"{np.mean([r.n_stages for r in ok]):>8.1f} "
                f"{n:>4} {len(ok)/n:>8.0%}"
            )
        else:
            print(f"start_weights={sw:<16} {'---':>10} {'---':>10} {'---':>8} {n:>4} {'0%':>8}")

    return results


def visualize_sensitivity(results: list[AblationResult]):
    """Publication figure: parameter sensitivity with 2 subplots.

    Each subplot shows grouped bars for mean CD, mean time, and mean stages.
    """
    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    start_weights_list = [4, 5, 6, 7, 8]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bar_width = 0.25
    bar_alpha = 0.75
    metric_colors = {
        "cd": "#2166AC",
        "time": "#D6604D",
        "stages": "#4393C3",
    }

    def _collect(methods: list[str]) -> tuple[list[float], list[float], list[float]]:
        """Return (cd_means, time_means, stages_means) for each method."""
        cds, times, stages = [], [], []
        for m in methods:
            ok = [r for r in results if r.method == m and r.success]
            if ok:
                cds.append(float(np.mean([r.cd_final for r in ok])))
                times.append(float(np.mean([r.time_s for r in ok])))
                stages.append(float(np.mean([r.n_stages for r in ok])))
            else:
                cds.append(0.0)
                times.append(0.0)
                stages.append(0.0)
        return cds, times, stages

    # ── Subplot 1: Threshold effect ──
    ax = axes[0]
    thresh_methods = [f"thresh_{t}" for t in thresholds]
    cds, times, stages = _collect(thresh_methods)

    x = np.arange(len(thresholds))
    ax.bar(x - bar_width, cds, bar_width, label="Mean CD", color=metric_colors["cd"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.bar(x, times, bar_width, label="Mean Time (s)", color=metric_colors["time"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.bar(x + bar_width, stages, bar_width, label="Mean Stages", color=metric_colors["stages"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds], rotation=30, ha="right")
    ax.set_xlabel("Improvement Threshold")
    ax.set_ylabel("Value")
    ax.set_title("(a) Threshold Sensitivity")
    ax.legend(fontsize=8)

    # Annotate CD bars with values
    for xi, v in zip(x, cds):
        if v > 0:
            ax.annotate(f"{v:.4f}", xy=(xi - bar_width, v), fontsize=7,
                        ha="center", va="bottom", color=metric_colors["cd"])

    # ── Subplot 2: Start weights effect ──
    ax = axes[1]
    sw_methods = [f"sens_sw{w}" for w in start_weights_list]
    cds, times, stages = _collect(sw_methods)

    x = np.arange(len(start_weights_list))
    ax.bar(x - bar_width, cds, bar_width, label="Mean CD", color=metric_colors["cd"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.bar(x, times, bar_width, label="Mean Time (s)", color=metric_colors["time"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.bar(x + bar_width, stages, bar_width, label="Mean Stages", color=metric_colors["stages"],
           alpha=bar_alpha, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in start_weights_list])
    ax.set_xlabel("Starting CST Weights")
    ax.set_ylabel("Value")
    ax.set_title("(b) Start Weights Sensitivity")
    ax.legend(fontsize=8)

    # Annotate CD bars with values
    for xi, v in zip(x, cds):
        if v > 0:
            ax.annotate(f"{v:.4f}", xy=(xi - bar_width, v), fontsize=7,
                        ha="center", va="bottom", color=metric_colors["cd"])

    fig.suptitle("Parameter Sensitivity Analysis", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = RESULTS_DIR / "sensitivity.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── CSV Export ────────────────────────────────────────────────────────────


def export_csv(all_results: list[AblationResult]):
    """Export all ablation results to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "ablation.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ablation", "method", "airfoil", "cd_initial", "cd_final",
            "time_s", "n_stages", "stage_cds", "success",
        ])
        for r in all_results:
            writer.writerow([
                r.ablation,
                r.method,
                r.airfoil_name,
                f"{r.cd_initial:.6f}",
                f"{r.cd_final:.6f}" if r.cd_final < float("inf") else "inf",
                f"{r.time_s:.3f}",
                r.n_stages,
                ";".join(f"{cd:.6f}" for cd in r.stage_cds),
                int(r.success),
            ])
    print(f"\nCSV saved: {save_path}")


# ── Final Summary Table ──────────────────────────────────────────────────


def print_final_summary(all_results: list[AblationResult]):
    """Print a consolidated summary across all ablations."""
    print("\n" + "=" * 80)
    print("CONSOLIDATED SUMMARY")
    print("=" * 80)

    # Group by method
    methods = sorted(set(r.method for r in all_results))
    print(f"\n{'Method':<28} {'Ablation':>8} {'N':>4} {'CD Mean':>10} {'CD Std':>10} {'Time Mean':>10} {'Stages':>8} {'Success':>8}")
    print("-" * 90)
    for method in methods:
        runs = [r for r in all_results if r.method == method]
        ablation = runs[0].ablation
        ok = [r for r in runs if r.success]
        if not ok:
            print(f"{method:<28} {ablation:>8} {len(runs):>4} {'FAILED':>10}")
            continue
        cds = [r.cd_final for r in ok]
        times = [r.time_s for r in ok]
        stages = [r.n_stages for r in ok]
        print(
            f"{method:<28} {ablation:>8} {len(runs):>4} "
            f"{np.mean(cds):>10.6f} {np.std(cds):>10.6f} "
            f"{np.mean(times):>10.1f} {np.mean(stages):>8.1f} "
            f"{sum(r.success for r in runs)/len(runs):>8.0%}"
        )


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    """Run the full ablation study."""
    print("=" * 80)
    print("PIERN Airfoil Optimization — Ablation Study")
    print("=" * 80)

    normal, medium, hard = load_benchmark()
    all_airfoils = normal + medium + hard
    print(f"Benchmark airfoils: {len(all_airfoils)} (normal={len(normal)}, medium={len(medium)}, hard={len(hard)})")
    print(f"CL targets:  {CL_TARGETS}")
    print(f"CL weights:  {CL_WEIGHTS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()
    all_results: list[AblationResult] = []

    # ── Ablation 1 ──
    r1 = run_ablation_1(all_airfoils)
    all_results.extend(r1)
    visualize_ablation_1(r1, all_airfoils)

    # ── Ablation 2 ──
    r2 = run_ablation_2(all_airfoils)
    all_results.extend(r2)
    visualize_ablation_2(r2, all_airfoils)

    # ── Ablation 3 ──
    r3 = run_ablation_3(all_airfoils)
    all_results.extend(r3)
    visualize_ablation_3(r3, all_airfoils)

    # ── Ablation 4 ──
    r4 = run_ablation_4(all_airfoils)
    all_results.extend(r4)
    visualize_ablation_4(r4, all_airfoils)

    # ── Sensitivity Analysis ──
    rs = run_sensitivity_analysis()
    all_results.extend(rs)
    visualize_sensitivity(rs)

    # ── Export ──
    export_csv(all_results)
    print_final_summary(all_results)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total runs: {len(all_results)}")
    print("Done.")


if __name__ == "__main__":
    main()
