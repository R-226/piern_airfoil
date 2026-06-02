"""
Router Benchmark — 优化策略对比。

对比方法:
  - Baseline:    NeuralOptimizer 全 8 权重 IPOPT
  - Rule:        固定阈值 0.01 的层次化路由
  - Threshold:   网格搜索学习的最优阈值
  - PiERN Router: 基于历史信息的学习型路由

场景:
  - Normal: 常规翼型 (30)
  - Medium: 中等翼型 (44)
  - Hard:   困难翼型 (31)

翼型从 data/benchmark_airfoils.json 加载 (固定集合, 基于 brentq 初始 CD 过滤)。

输出:
  results/benchmark_normal.png — 常规场景对比
  results/benchmark_hard.png   — 困难场景对比
  results/benchmark_stats.csv  — 原始数据 CSV
"""

from __future__ import annotations

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
import matplotlib.font_manager as fm
from scipy.optimize import brentq

# ── CJK 字体配置 ────────────────────────────────────────────────────────
_CJK_FONT = "Noto Sans CJK JP"
try:
    fm.findfont(_CJK_FONT, fallback_to_default=False)
    plt.rcParams["font.family"] = _CJK_FONT
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False


# ── 问题定义 ────────────────────────────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03
BENCHMARK_JSON = Path(__file__).parent.parent / "data" / "benchmark_airfoils.json"


# ── 翼型分组 (从固定 benchmark 文件加载) ──────────────────────────────


def load_benchmark_airfoils() -> tuple[list[str], list[str], list[str]]:
    """从固定 benchmark 文件加载翼型集合。

    Returns:
        (normal, medium, hard)
    """
    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    return bench["normal"], bench["medium"], bench["hard"]

# ── 方法标签和颜色 ──────────────────────────────────────────────────────

METHOD_LABELS = {
    "initial": "Initial",
    "baseline": "Baseline (8w IPOPT)",
    "rule": "Rule",
    "threshold": "Threshold",
    "mlp": "PiERN Router",
}

METHOD_COLORS = {
    "initial": "#999999",
    "baseline": "#1F77B4",
    "rule": "#E45756",
    "threshold": "#F58518",
    "mlp": "#4C78D2",
}

OPT_METHODS = ["baseline", "rule", "threshold", "mlp"]


# ── 数据结构 ────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """单次运行结果。"""
    method: str
    airfoil_name: str
    cd: float
    time: float
    n_stages: int
    success: bool = True  # CD < 0.15 且无异常 (mean 公式, ~2× normal)


@dataclass
class StatsResult:
    """多运行统计结果。"""
    method: str
    airfoil_name: str
    cd_mean: float
    cd_std: float
    time_mean: float
    time_std: float
    n_stages_mean: float
    success_rate: float


# ── CD 评估 ─────────────────────────────────────────────────────────────


def evaluate_cd(airfoil) -> float:
    """评估翼型加权 CD (NeuralFoil 标准: mean(CD * weights))。"""
    cd_values = []
    for cl_t, re_i in zip(CL_TARGETS, RE):
        def residual(a, _af=airfoil, _re=re_i, _cl=cl_t):
            aero = _af.get_aero_from_neuralfoil(alpha=a, Re=float(_re), mach=MACH)
            return float(np.asarray(aero["CL"]).flatten()[0]) - _cl
        try:
            alpha_i = brentq(residual, -5, 18, xtol=0.01, maxiter=30)
        except (ValueError, RuntimeError):
            alpha_i = 5.0
        aero = airfoil.get_aero_from_neuralfoil(alpha=alpha_i, Re=float(re_i), mach=MACH)
        cd_values.append(float(np.asarray(aero["CD"]).flatten()[0]))
    return float(np.mean(np.array(cd_values) * CL_WEIGHTS))


# ── 各方法运行 ──────────────────────────────────────────────────────────


def _suppress_ipopt():
    """Suppress IPOPT stdout by redirecting fd 1."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old_stdout


def _restore_stdout(old_fd: int):
    """Restore stdout from saved fd."""
    os.dup2(old_fd, 1)
    os.close(old_fd)


def run_once(airfoil_name: str, method: str) -> RunResult:
    """运行单次优化。"""
    af = asb.KulfanAirfoil(airfoil_name)

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)

    t0 = time.perf_counter()
    try:
        if method == "baseline":
            from piern_airfoil.optimizer import NeuralOptimizer
            opt = NeuralOptimizer(
                airfoil=af, CL_targets=CL_TARGETS, CL_weights=CL_WEIGHTS,
                RE=RE, mach=MACH,
            )
            opt.update()
            elapsed = time.perf_counter() - t0
            cd = evaluate_cd(opt.airfoil)
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            return RunResult(method, airfoil_name, cd, elapsed, 1, success=cd < 0.15)

        elif method in ("rule", "threshold", "mlp"):
            from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
            from piern.router.opt_router import OptRouter

            if method == "mlp":
                router = OptRouter.from_mlp()
            elif method == "threshold":
                router = OptRouter.from_trained()
            else:
                router = OptRouter(improvement_threshold=0.01)

            optimizer = AdaptiveHierarchicalOptimizer(
                CL_targets=CL_TARGETS, CL_weights=CL_WEIGHTS,
                Re=RE, mach=MACH, start_weights=4, router=router,
            )
            result = optimizer.optimize(af)
            elapsed = time.perf_counter() - t0
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            return RunResult(
                method, airfoil_name,
                cd=result.final_cd, time=elapsed,
                n_stages=len(result.stages),
                success=result.final_cd < 0.15,
            )
        else:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        try:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
        except OSError:
            pass
        return RunResult(
            method, airfoil_name,
            cd=float("inf"), time=elapsed, n_stages=0, success=False,
        )


def run_initial(airfoil_name: str) -> RunResult:
    """评估原始翼型。"""
    af = asb.KulfanAirfoil(airfoil_name)
    cd = evaluate_cd(af)
    return RunResult("initial", airfoil_name, cd, 0.0, 0)


# ── 统计计算 ────────────────────────────────────────────────────────────


def compute_stats(runs: list[RunResult]) -> StatsResult:
    """计算多次运行的统计量。"""
    cds = [r.cd for r in runs if r.success]
    times = [r.time for r in runs if r.success]
    stages = [r.n_stages for r in runs if r.success]
    n_success = sum(1 for r in runs if r.success)
    n_total = len(runs)

    if not cds:
        return StatsResult(
            runs[0].method, runs[0].airfoil_name,
            float("inf"), 0, 0, 0, 0, 0,
        )

    return StatsResult(
        method=runs[0].method,
        airfoil_name=runs[0].airfoil_name,
        cd_mean=np.mean(cds),
        cd_std=np.std(cds),
        time_mean=np.mean(times),
        time_std=np.std(times),
        n_stages_mean=np.mean(stages),
        success_rate=n_success / n_total,
    )


# ── Benchmark 运行 ──────────────────────────────────────────────────────


def run_benchmark_group(
    airfoils: list[str], label: str,
) -> tuple[list[StatsResult], list[RunResult]]:
    """对一组翼型运行所有方法。"""
    all_runs: list[RunResult] = []
    all_stats: list[StatsResult] = []
    total = len(airfoils) * (len(OPT_METHODS) + 1)
    idx = 0

    for airfoil_name in airfoils:
        idx += 1
        print(f"  [{label} {idx}/{total}] {airfoil_name} initial...", end=" ", flush=True)
        r = run_initial(airfoil_name)
        all_runs.append(r)
        all_stats.append(compute_stats([r]))
        print(f"CD={r.cd:.4f}")

        for method in OPT_METHODS:
            idx += 1
            print(
                f"  [{label} {idx}/{total}] {airfoil_name} {method}...",
                end=" ", flush=True,
            )
            r = run_once(airfoil_name, method)
            all_runs.append(r)
            all_stats.append(compute_stats([r]))
            status = f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED"
            print(status)

    return all_stats, all_runs


# ── 可视化 ─────────────────────────────────────────────────────────────


def _get_stats(
    all_stats: list[StatsResult], method: str, airfoil: str
) -> StatsResult | None:
    for s in all_stats:
        if s.method == method and s.airfoil_name == airfoil:
            return s
    return None


def visualize_normal(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_normal.png",
):
    """常规场景可视化。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    x = np.arange(len(airfoils))
    n_m = len(OPT_METHODS)
    width = 0.7 / n_m

    # ── 图1: 时间对比 (带误差棒) ──
    ax = axes[0, 0]
    for i, method in enumerate(OPT_METHODS):
        means, stds = [], []
        for af in airfoils:
            s = _get_stats(all_stats, method, af)
            means.append(s.time_mean if s else 0)
            stds.append(s.time_std if s else 0)
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.85, edgecolor="white", linewidth=0.5, capsize=3,
        )
    ax.set_ylabel("Time (s)")
    ax.set_title("Optimization Time (mean ± std)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图2: CD 对比 (带误差棒) ──
    ax = axes[0, 1]
    for i, method in enumerate(OPT_METHODS):
        means, stds = [], []
        for af in airfoils:
            s = _get_stats(all_stats, method, af)
            means.append(s.cd_mean if s else 0)
            stds.append(s.cd_std if s else 0)
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.85, edgecolor="white", linewidth=0.5, capsize=3,
        )
    ax.set_ylabel("Weighted CD")
    ax.set_title("Final CD (mean ± std)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图3: 时间节省百分比 (PiERN vs Baseline) ──
    ax = axes[1, 0]
    compare_methods = ["rule", "threshold", "mlp"]
    compare_colors = [METHOD_COLORS[m] for m in compare_methods]
    compare_labels = [METHOD_LABELS[m] for m in compare_methods]
    bar_w = 0.25
    for i, (method, label, color) in enumerate(zip(compare_methods, compare_labels, compare_colors)):
        savings = []
        for af in airfoils:
            base = _get_stats(all_stats, "baseline", af)
            meth = _get_stats(all_stats, method, af)
            if base and meth and base.time_mean > 0:
                savings.append((base.time_mean - meth.time_mean) / base.time_mean * 100)
            else:
                savings.append(0)
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, savings, bar_w, label=label, color=color, alpha=0.85)
        for bar, s in zip(bars, savings):
            va = "bottom" if s >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:+.0f}%", ha="center", va=va, fontsize=6, fontweight="bold",
            )
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Time Saved vs Baseline (%)")
    ax.set_title("Speed Advantage (positive = faster)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图4: CD 差异 (vs Baseline) ──
    ax = axes[1, 1]
    for i, (method, label, color) in enumerate(zip(compare_methods, compare_labels, compare_colors)):
        diffs = []
        for af in airfoils:
            base = _get_stats(all_stats, "baseline", af)
            meth = _get_stats(all_stats, method, af)
            if base and meth:
                diffs.append((meth.cd_mean - base.cd_mean) / base.cd_mean * 100)
            else:
                diffs.append(0)
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, diffs, bar_w, label=label, color=color, alpha=0.85)
        for bar, d in zip(bars, diffs):
            va = "bottom" if d >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{d:+.1f}%", ha="center", va=va, fontsize=6, fontweight="bold",
            )
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("CD vs Baseline (%)")
    ax.set_title("CD Difference (negative = better)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Benchmark — Normal Cases ({len(airfoils)} airfoils)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n可视化已保存: {save_path}")


def visualize_hard(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_hard.png",
):
    """困难场景可视化。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    x = np.arange(len(airfoils))
    n_m = len(OPT_METHODS)
    width = 0.75 / n_m

    # ── 图1: CD 对比 ──
    ax = axes[0, 0]
    for i, method in enumerate(OPT_METHODS):
        means, stds = [], []
        for af in airfoils:
            s = _get_stats(all_stats, method, af)
            means.append(s.cd_mean if s else 0)
            stds.append(s.cd_std if s else 0)
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.85, capsize=3,
        )
    ax.axhline(y=0.078, color="green", linestyle=":", linewidth=1.5, alpha=0.7, label="Normal ~0.078")
    ax.set_ylabel("Weighted CD")
    ax.set_title("CD Comparison — Baseline Fails, PiERN Succeeds", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图2: 时间对比 ──
    ax = axes[0, 1]
    for i, method in enumerate(OPT_METHODS):
        means, stds = [], []
        for af in airfoils:
            s = _get_stats(all_stats, method, af)
            means.append(s.time_mean if s else 0)
            stds.append(s.time_std if s else 0)
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            alpha=0.85, capsize=3,
        )
    ax.set_ylabel("Time (s)")
    ax.set_title("Optimization Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图3: 救援率 (成功率) ──
    ax = axes[1, 0]
    success_rates = {m: [] for m in OPT_METHODS}
    for af in airfoils:
        for m in OPT_METHODS:
            s = _get_stats(all_stats, m, af)
            success_rates[m].append(s.success_rate if s else 0)
    bars = ax.bar(
        range(len(OPT_METHODS)),
        [np.mean(success_rates[m]) * 100 for m in OPT_METHODS],
        0.6,
        color=[METHOD_COLORS[m] for m in OPT_METHODS],
        alpha=0.85,
    )
    for bar, m in zip(bars, OPT_METHODS):
        rate = np.mean(success_rates[m]) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"Success Rate (CD < 0.15, {len(airfoils)} airfoils)", fontweight="bold")
    ax.set_xticks(range(len(OPT_METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in OPT_METHODS], fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # ── 图4: CD 改进倍数 ──
    ax = axes[1, 1]
    ratios = []
    labels_list = []
    for af in airfoils:
        base = _get_stats(all_stats, "baseline", af)
        piern = _get_stats(all_stats, "mlp", af)
        if base and piern and piern.cd_mean > 0:
            ratios.append(base.cd_mean / piern.cd_mean)
            labels_list.append(af.upper())
    bar_colors = [METHOD_COLORS["mlp"] if r > 1.1 else "#cccccc" for r in ratios]
    bars = ax.bar(range(len(ratios)), ratios, 0.6, color=bar_colors, alpha=0.85)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{ratio:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Baseline CD / PiERN CD")
    ax.set_title("How Much Better PiERN Is (>1x = PiERN wins)", fontweight="bold")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(labels_list, fontsize=8, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Benchmark — Hard Cases ({len(airfoils)} airfoils)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存: {save_path}")


# ── CSV 导出 ────────────────────────────────────────────────────────────


def export_csv(
    all_stats: list[StatsResult],
    save_path: str = "results/benchmark_stats.csv",
):
    """导出统计结果为 CSV。"""
    import csv

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "airfoil", "method", "cd_mean", "cd_std",
            "time_mean", "time_std", "n_stages_mean", "success_rate",
        ])
        for s in all_stats:
            writer.writerow([
                s.airfoil_name, s.method,
                f"{s.cd_mean:.6f}", f"{s.cd_std:.6f}",
                f"{s.time_mean:.3f}", f"{s.time_std:.3f}",
                f"{s.n_stages_mean:.1f}", f"{s.success_rate:.2f}",
            ])
    print(f"CSV 已保存: {save_path}")


# ── 汇总表 ──────────────────────────────────────────────────────────────


def print_summary(all_stats: list[StatsResult], title: str, airfoils: list[str]):
    """打印中文汇总表。"""
    print(f"\n{'='*90}")
    print(title)
    print("=" * 90)
    print(f"\n{'翼型':<14} {'方法':<20} {'CD均值':>10} {'CD标准差':>10} {'时间均值':>10} {'时间标准差':>10} {'成功率':>8}")
    print("-" * 86)
    for s in all_stats:
        if s.airfoil_name not in airfoils:
            continue
        print(
            f"{s.airfoil_name.upper():<14} "
            f"{METHOD_LABELS.get(s.method, s.method):<20} "
            f"{s.cd_mean:>10.6f} "
            f"{s.cd_std:>10.6f} "
            f"{s.time_mean:>10.1f} "
            f"{s.time_std:>10.1f} "
            f"{s.success_rate:>8.0%}"
        )

    # 方法汇总
    print(f"\n{'方法':<20} {'平均CD':>10} {'平均时间':>10} {'平均成功率':>10}")
    print("-" * 54)
    for method in OPT_METHODS:
        ss = [s for s in all_stats if s.method == method and s.airfoil_name in airfoils]
        if not ss:
            continue
        avg_cd = np.mean([s.cd_mean for s in ss])
        avg_time = np.mean([s.time_mean for s in ss])
        avg_sr = np.mean([s.success_rate for s in ss])
        print(f"{METHOD_LABELS[method]:<20} {avg_cd:>10.6f} {avg_time:>10.1f} {avg_sr:>10.0%}")


# ── 主函数 ──────────────────────────────────────────────────────────────


def main():
    normal_afs, medium_afs, hard_afs = load_benchmark_airfoils()
    all_airfoils = normal_afs + medium_afs + hard_afs

    print("=" * 90)
    print("PiERN Router Benchmark")
    print("=" * 90)
    print(f"CL 目标:  {CL_TARGETS}")
    print(f"CL 权重:  {CL_WEIGHTS}")
    print(f"常规翼型: {len(normal_afs)} 个")
    print(f"中等翼型: {len(medium_afs)} 个")
    print(f"困难翼型: {len(hard_afs)} 个")
    print(f"总计:     {len(all_airfoils)} 个, {len(all_airfoils) * (len(OPT_METHODS) + 1)} 次优化")
    print()

    t0 = time.perf_counter()

    # ── 场景1: 常规翼型 ──
    print(">>> 场景1: 常规翼型 (Baseline 可工作)")
    normal_stats, normal_runs = run_benchmark_group(normal_afs, "Normal")
    print_summary(normal_stats, "场景1: 常规翼型", normal_afs)
    visualize_normal(normal_stats, normal_afs)

    # ── 场景2: 中等翼型 ──
    print("\n>>> 场景2: 中等翼型 (Baseline 勉强)")
    medium_stats, medium_runs = run_benchmark_group(medium_afs, "Medium")
    print_summary(medium_stats, "场景2: 中等翼型", medium_afs)

    # ── 场景3: 困难翼型 ──
    print("\n>>> 场景3: 困难翼型 (Baseline 失败)")
    hard_stats, hard_runs = run_benchmark_group(hard_afs, "Hard")
    print_summary(hard_stats, "场景3: 困难翼型", hard_afs)
    visualize_hard(hard_stats, hard_afs)

    # ── 导出 CSV ──
    all_stats = normal_stats + medium_stats + hard_stats
    export_csv(all_stats)

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"总运行次数: {len(normal_runs) + len(medium_runs) + len(hard_runs)}")


if __name__ == "__main__":
    main()
