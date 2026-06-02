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
from matplotlib.axes import Axes
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
    "xfoil": "XFoil (ground truth)",
}

METHOD_COLORS = {
    "initial": "#8C8C8C",
    "baseline": "#3370AC",
    "rule": "#D44B3F",
    "threshold": "#E8A838",
    "mlp": "#2A8C6A",
    "xfoil": "#7B3294",
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


def run_xfoil_baseline(airfoil_name: str) -> RunResult:
    """用 XFoil 评估原始翼型的加权 CD (高保真基线)。"""
    from piern_airfoil.xfoil_baseline import xfoil_cd

    t0 = time.perf_counter()
    try:
        cd = xfoil_cd(airfoil_name, CL_TARGETS, RE, CL_WEIGHTS, MACH)
        elapsed = time.perf_counter() - t0
        return RunResult("xfoil", airfoil_name, cd, elapsed, 0, success=cd < 0.5)
    except Exception:
        elapsed = time.perf_counter() - t0
        return RunResult("xfoil", airfoil_name, float("inf"), elapsed, 0, success=False)


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
    total = len(airfoils) * (len(OPT_METHODS) + 2)  # +2: initial + xfoil
    idx = 0

    for airfoil_name in airfoils:
        idx += 1
        print(f"  [{label} {idx}/{total}] {airfoil_name} initial...", end=" ", flush=True)
        r = run_initial(airfoil_name)
        all_runs.append(r)
        all_stats.append(compute_stats([r]))
        print(f"CD={r.cd:.4f}")

        # XFoil ground truth evaluation
        idx += 1
        print(f"  [{label} {idx}/{total}] {airfoil_name} xfoil...", end=" ", flush=True)
        r = run_xfoil_baseline(airfoil_name)
        all_runs.append(r)
        all_stats.append(compute_stats([r]))
        status = f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED"
        print(status)

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


# ── 可视化 — 出版级质量 ────────────────────────────────────────────────
#
# 设计规范:
#   - 字体: Liberation Serif (Times New Roman 兼容), LaTeX 论文标准
#   - 配色: 优化色盲友好调色板, 兼顾打印和屏幕显示
#   - 尺寸: 7 英寸宽 (双栏), 300 DPI, 符合期刊出版要求
#   - 仅保留左/下轴脊柱, 去除多余网格线, 最小化 chartjunk
#   - 关键数据直接标注在图表上


def _get_stats(
    all_stats: list[StatsResult], method: str, airfoil: str
) -> StatsResult | None:
    for s in all_stats:
        if s.method == method and s.airfoil_name == airfoil:
            return s
    return None


# ── 出版级配色 (色盲友好, 高对比度) ────────────────────────────────────
#
# 基于 Wong (2011, Nature Methods) 色盲友好调色板优化:
#   - Baseline (深蓝):   #3370AC  — 稳重, 作为基准参照
#   - Rule (深红):       #D44B3F  — 温暖, 与蓝色形成强对比
#   - Threshold (琥珀):  #E8A838  — 中性, 区分红/绿色盲
#   - PiERN (深绿):      #2A8C6A  — 清凉, 在打印和屏幕均清晰可辨
# 所有颜色在灰度打印下仍有可辨识的亮度差异。

_SERIF_FONT = "Liberation Serif"
_PALETTE = {
    "baseline": "#3370AC",
    "rule": "#D44B3F",
    "threshold": "#E8A838",
    "mlp": "#2A8C6A",
}


class _PubStyle:
    """Publication figure constants for journal-sized figures.

    All sizes calibrated for 7-inch (full-page) figures at 300 DPI.
    Font sizes in points: title=10, axis=8.5, tick=7.5, annotation=6.5.
    """
    FIG_W = 7.0
    ROW_H = 2.6
    TITLE_SIZE = 10
    AXIS_SIZE = 8.5
    TICK_SIZE = 7.5
    ANNOT_SIZE = 6.5
    BAR_WIDTH = 0.17
    INTRA_GROUP_GAP = 0.02
    INTER_GROUP_GAP = 0.06
    TICK_PARAMS = dict(direction="in", top=False, right=False, labelsize=TICK_SIZE, pad=3)
    SPINE_PARAMS = dict(linewidth=0.6)
    LEGEND_KW = dict(
        fontsize=TICK_SIZE, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=4, handletextpad=0.4, columnspacing=1.0,
    )
    GRID_KW = dict(axis="y", linewidth=0.3, alpha=0.35)


def _style_axes(ax: Axes, x_label: str, y_label: str, title: str) -> None:
    """Apply publication axis styling: spines, ticks, labels, subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(_PubStyle.SPINE_PARAMS["linewidth"])
    ax.tick_params(**_PubStyle.TICK_PARAMS)
    ax.set_xlabel(x_label, fontsize=_PubStyle.AXIS_SIZE, fontfamily=_SERIF_FONT, labelpad=4)
    ax.set_ylabel(y_label, fontsize=_PubStyle.AXIS_SIZE, fontfamily=_SERIF_FONT, labelpad=4)
    ax.set_title(title, fontsize=_PubStyle.TITLE_SIZE, fontfamily=_SERIF_FONT, pad=6)
    ax.yaxis.grid(True, linewidth=_PubStyle.GRID_KW["linewidth"],
                  alpha=_PubStyle.GRID_KW["alpha"])
    ax.set_axisbelow(True)


def _annotate_bars(ax: Axes, bars, values: list[float], suffix: str = "%",
                   offset_ratio: float = 0.02) -> None:
    """Annotate bars with values, handling positive/negative positions."""
    if not bars:
        return
    all_vals = [v for v in values if np.isfinite(v)]
    if not all_vals:
        return
    y_range = max(all_vals) - min(all_vals) if len(all_vals) > 1 else 1.0
    offset = y_range * offset_ratio if y_range > 0 else 0.01
    for bar, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        va = "bottom" if v >= 0 else "top"
        y_pos = bar.get_height() + offset if v >= 0 else bar.get_height() - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{v:+.1f}{suffix}", ha="center", va=va,
            fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
        )


def _grouped_bars(
    ax: Axes, x: np.ndarray, data: dict[str, list[float]],
    method_keys: list[str], bar_w: float, annotate: bool = False,
    suffix: str = "%",
) -> None:
    """Render grouped bars with publication styling.

    Args:
        data: {method_key: [value_per_airfoil]}
    """
    n = len(method_keys)
    for i, method in enumerate(method_keys):
        vals = data[method]
        offset = (i - (n - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            label=METHOD_LABELS[method],
            color=_PALETTE.get(method, METHOD_COLORS.get(method, "#888888")),
            edgecolor="white", linewidth=0.3,
        )
        if annotate:
            _annotate_bars(ax, bars, vals, suffix=suffix)


def visualize_normal(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_normal.png",
):
    """Normal scenario visualization — journal publication quality.

    Layout (2x2):
      (a) CD improvement vs baseline (%)      — grouped bars, all 3 methods
      (b) Time speedup vs baseline (ratio)    — grouped bars with speedup labels
      (c) Success rate (%)                    — per-method bar chart
      (d) Mean improvement summary            — bar chart of aggregated metrics
    """
    n_af = len(airfoils)
    x = np.arange(n_af)

    fig, axes = plt.subplots(
        2, 2, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 2),
        gridspec_kw=dict(hspace=0.55, wspace=0.38),
    )

    methods_3 = ["rule", "threshold", "mlp"]
    bar_w = _PubStyle.BAR_WIDTH
    af_labels = [n.upper()[:12] for n in airfoils]

    # ── (a) CD improvement vs baseline ──
    ax = axes[0, 0]
    cd_imp = {m: [] for m in methods_3}
    for af in airfoils:
        base = _get_stats(all_stats, "baseline", af)
        for m in methods_3:
            meth = _get_stats(all_stats, m, af)
            if base and meth and base.cd_mean > 0:
                cd_imp[m].append((meth.cd_mean - base.cd_mean) / base.cd_mean * 100)
            else:
                cd_imp[m].append(0.0)

    _grouped_bars(ax, x, cd_imp, methods_3, bar_w)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "$\\Delta$CD vs Baseline (%)", "(a) CD Improvement")
    ax.legend(**_PubStyle.LEGEND_KW)

    # ── (b) Time speedup vs baseline ──
    ax = axes[0, 1]
    time_sp = {m: [] for m in methods_3}
    for af in airfoils:
        base = _get_stats(all_stats, "baseline", af)
        for m in methods_3:
            meth = _get_stats(all_stats, m, af)
            if base and meth and meth.time_mean > 0:
                time_sp[m].append(base.time_mean / meth.time_mean)
            else:
                time_sp[m].append(1.0)

    _grouped_bars(ax, x, time_sp, methods_3, bar_w)
    ax.axhline(y=1.0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Speedup (baseline / method)", "(b) Time Speedup")
    # Annotate mean speedup per method in the top-right corner
    for m in methods_3:
        vals = [v for v in time_sp[m] if np.isfinite(v)]
        if vals:
            mean_sp = np.mean(vals)
            ax.annotate(
                f"{METHOD_LABELS[m]}: {mean_sp:.1f}$\\times$",
                xy=(0.98, 0.95 - methods_3.index(m) * 0.10),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
                color=_PALETTE.get(m, "#333333"),
            )

    # ── (c) Success rate ──
    ax = axes[1, 0]
    methods_all = OPT_METHODS
    success_rates = []
    for m in methods_all:
        rates = [_get_stats(all_stats, m, af) for af in airfoils]
        rates = [s.success_rate for s in rates if s]
        success_rates.append(np.mean(rates) * 100 if rates else 0)

    bars = ax.bar(
        range(len(methods_all)), success_rates, 0.55,
        color=[_PALETTE.get(m, "#888888") for m in methods_all],
        edgecolor="white", linewidth=0.3,
    )
    for bar, rate in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{rate:.0f}%", ha="center", va="bottom",
            fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT, fontweight="bold",
        )
    ax.set_xticks(range(len(methods_all)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in methods_all],
        fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT,
    )
    ax.set_ylim(0, 110)
    _style_axes(ax, "", "Success Rate (%)", f"(c) Optimization Success (CD < 0.15)")

    # ── (d) Mean CD improvement summary ──
    ax = axes[1, 1]
    mean_imp = []
    for m in methods_3:
        vals = cd_imp[m]
        mean_imp.append(np.mean(vals))
    bars = ax.bar(
        range(len(methods_3)), mean_imp, 0.45,
        color=[_PALETTE.get(m, "#888888") for m in methods_3],
        edgecolor="white", linewidth=0.3,
    )
    for bar, val in zip(bars, mean_imp):
        va = "bottom" if val >= 0 else "top"
        y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.2f}%", ha="center", va=va,
            fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT, fontweight="bold",
        )
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(range(len(methods_3)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in methods_3],
        fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT,
    )
    _style_axes(ax, "", "Mean $\\Delta$CD vs Baseline (%)", "(d) Mean CD Improvement")

    fig.suptitle(
        f"Router Benchmark — Normal Cases ({n_af} airfoils)",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=0.99,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPublication figure saved: {save_path}")


def visualize_hard(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_hard.png",
):
    """Hard scenario visualization — journal publication quality.

    Layout (2x2):
      (a) CD comparison across all 4 methods  — grouped bars
      (b) Optimization time                   — grouped bars
      (c) Rescue rate (%)                     — bar chart with exact percentages
      (d) CD improvement ratio (baseline/M)   — how much better PiERN is
    """
    n_af = len(airfoils)
    x = np.arange(n_af)
    bar_w = _PubStyle.BAR_WIDTH

    fig, axes = plt.subplots(
        2, 2, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 2),
        gridspec_kw=dict(hspace=0.55, wspace=0.38),
    )

    af_labels = [n.upper()[:12] for n in airfoils]

    # ── (a) CD comparison ──
    ax = axes[0, 0]
    cd_data = {m: [] for m in OPT_METHODS}
    for af in airfoils:
        for m in OPT_METHODS:
            s = _get_stats(all_stats, m, af)
            cd_data[m].append(s.cd_mean if s and s.cd_mean < 1e10 else 0.0)

    _grouped_bars(ax, x, cd_data, OPT_METHODS, bar_w, annotate=False)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Weighted CD", "(a) CD Comparison")
    ax.legend(**_PubStyle.LEGEND_KW)

    # ── (b) Optimization time ──
    ax = axes[0, 1]
    time_data = {m: [] for m in OPT_METHODS}
    for af in airfoils:
        for m in OPT_METHODS:
            s = _get_stats(all_stats, m, af)
            time_data[m].append(s.time_mean if s else 0.0)

    _grouped_bars(ax, x, time_data, OPT_METHODS, bar_w, annotate=False)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Time (s)", "(b) Optimization Time")

    # ── (c) Rescue rate ──
    ax = axes[1, 0]
    success_rates = []
    for m in OPT_METHODS:
        rates = [_get_stats(all_stats, m, af) for af in airfoils]
        rates = [s.success_rate for s in rates if s]
        success_rates.append(np.mean(rates) * 100 if rates else 0)

    bars = ax.bar(
        range(len(OPT_METHODS)), success_rates, 0.55,
        color=[_PALETTE.get(m, "#888888") for m in OPT_METHODS],
        edgecolor="white", linewidth=0.3,
    )
    for bar, rate in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{rate:.0f}%", ha="center", va="bottom",
            fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT, fontweight="bold",
        )
    ax.set_xticks(range(len(OPT_METHODS)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in OPT_METHODS],
        fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT,
    )
    ax.set_ylim(0, 110)
    _style_axes(ax, "", "Success Rate (%)", "(c) Rescue Rate (CD < 0.15)")

    # ── (d) CD improvement ratio ──
    ax = axes[1, 1]
    ratios, ratio_labels = [], []
    for af in airfoils:
        base = _get_stats(all_stats, "baseline", af)
        piern = _get_stats(all_stats, "mlp", af)
        if base and piern and piern.cd_mean > 0 and base.cd_mean < 1e10:
            ratios.append(base.cd_mean / piern.cd_mean)
            ratio_labels.append(af.upper()[:12])

    ratio_colors = [_PALETTE["mlp"] if r >= 1.0 else "#C0C0C0" for r in ratios]
    bars = ax.bar(
        range(len(ratios)), ratios, 0.55,
        color=ratio_colors, edgecolor="white", linewidth=0.3,
    )
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{ratio:.1f}$\\times$", ha="center", va="bottom",
            fontsize=5.5, fontfamily=_SERIF_FONT,
        )
    ax.axhline(y=1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(ratio_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(
        ax, "", "Baseline CD / PiERN CD",
        "(d) PiERN Improvement Ratio",
    )
    # Count wins
    n_wins = sum(1 for r in ratios if r >= 1.0)
    ax.annotate(
        f"PiERN wins {n_wins}/{len(ratios)}",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
        color=_PALETTE["mlp"],
    )

    fig.suptitle(
        f"Router Benchmark — Hard Cases ({n_af} airfoils)",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=0.99,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Publication figure saved: {save_path}")


def visualize_xfoil_comparison(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_xfoil_comparison.png",
):
    """NeuralFoil vs XFoil comparison — scatter plot for journal publication.

    Shows per-airfoil NeuralFoil initial CD vs XFoil CD.
    Perfect agreement falls on the y=x diagonal.
    """
    nf_cds, xf_cds, names = [], [], []
    for af in airfoils:
        nf = _get_stats(all_stats, "initial", af)
        xf = _get_stats(all_stats, "xfoil", af)
        if nf and xf and nf.cd_mean < 1e10 and xf.cd_mean < 1e10:
            nf_cds.append(nf.cd_mean)
            xf_cds.append(xf.cd_mean)
            names.append(af)

    if not nf_cds:
        print("  Skipped xfoil comparison: no data")
        return

    nf_cds = np.array(nf_cds)
    xf_cds = np.array(xf_cds)

    fig, ax = plt.subplots(1, 1, figsize=(_PubStyle.FIG_W * 0.6, _PubStyle.ROW_H))
    ax.scatter(nf_cds, xf_cds, c=_PALETTE["mlp"], s=25, alpha=0.7,
               edgecolors="white", linewidths=0.3, zorder=5)

    # y=x reference line
    lims = [0, max(nf_cds.max(), xf_cds.max()) * 1.1]
    ax.plot(lims, lims, "k--", linewidth=0.6, alpha=0.5, label="y = x")

    # Per-point annotation for outliers (>20% deviation)
    for nf_v, xf_v, name in zip(nf_cds, xf_cds, names):
        if abs(xf_v - nf_v) / max(nf_v, 1e-6) > 0.20:
            ax.annotate(
                name.upper()[:8],
                xy=(nf_v, xf_v), fontsize=5, fontfamily=_SERIF_FONT,
                xytext=(4, 4), textcoords="offset points",
            )

    # Stats annotation
    mape = np.mean(np.abs(xf_cds - nf_cds) / nf_cds) * 100
    r2 = 1 - np.sum((xf_cds - nf_cds) ** 2) / np.sum((xf_cds - np.mean(xf_cds)) ** 2)
    ax.annotate(
        f"MAPE = {mape:.1f}%\nR$^2$ = {r2:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"),
    )

    _style_axes(ax, "NeuralFoil CD", "XFoil CD", "NeuralFoil vs XFoil (initial airfoils)")
    ax.set_aspect("equal")
    ax.legend(fontsize=_PubStyle.TICK_SIZE, frameon=False)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"XFoil comparison figure saved: {save_path}")


def visualize_summary(
    normal_stats: list[StatsResult],
    hard_stats: list[StatsResult],
    normal_afs: list[str],
    hard_afs: list[str],
    save_path: str = "results/benchmark_summary.png",
):
    """Cross-category summary figure for journal publication.

    Layout (1x3):
      (a) Mean CD improvement vs baseline (%)  — grouped by category
      (b) Mean time speedup                    — grouped by category
      (c) Success rate                         — grouped by category
    """
    categories = ["Normal", "Hard"]
    cat_stats = [normal_stats, hard_stats]
    cat_airfoils = [normal_afs, hard_afs]
    methods = ["rule", "threshold", "mlp"]
    x = np.arange(len(categories))
    bar_w = 0.22

    fig, axes = plt.subplots(
        1, 3, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 0.85),
        gridspec_kw=dict(wspace=0.40),
    )

    # ── (a) Mean CD improvement ──
    ax = axes[0]
    for i, m in enumerate(methods):
        vals = []
        for stats, afs in zip(cat_stats, cat_airfoils):
            imps = []
            for af in afs:
                base = _get_stats(stats, "baseline", af)
                meth = _get_stats(stats, m, af)
                if base and meth and base.cd_mean > 0:
                    imps.append((meth.cd_mean - base.cd_mean) / base.cd_mean * 100)
            vals.append(np.mean(imps) if imps else 0)
        offset = (i - 1) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            label=METHOD_LABELS[m], color=_PALETTE[m],
            edgecolor="white", linewidth=0.3,
        )
        _annotate_bars(ax, bars, vals, suffix="%", offset_ratio=0.04)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT)
    _style_axes(ax, "", "Mean $\\Delta$CD vs Baseline (%)", "(a) CD Improvement")
    ax.legend(**_PubStyle.LEGEND_KW)

    # ── (b) Mean time speedup ──
    ax = axes[1]
    for i, m in enumerate(methods):
        vals = []
        for stats, afs in zip(cat_stats, cat_airfoils):
            speedups = []
            for af in afs:
                base = _get_stats(stats, "baseline", af)
                meth = _get_stats(stats, m, af)
                if base and meth and meth.time_mean > 0:
                    speedups.append(base.time_mean / meth.time_mean)
            vals.append(np.mean(speedups) if speedups else 1.0)
        offset = (i - 1) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            label=METHOD_LABELS[m], color=_PALETTE[m],
            edgecolor="white", linewidth=0.3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.1f}$\\times$", ha="center", va="bottom",
                fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
            )
    ax.axhline(y=1.0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT)
    _style_axes(ax, "", "Speedup (baseline / method)", "(b) Time Speedup")

    # ── (c) Success rate ──
    ax = axes[2]
    for i, m in enumerate(methods):
        vals = []
        for stats, afs in zip(cat_stats, cat_airfoils):
            rates = [_get_stats(stats, m, af) for af in afs]
            rates = [s.success_rate for s in rates if s]
            vals.append(np.mean(rates) * 100 if rates else 0)
        offset = (i - 1) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            label=METHOD_LABELS[m], color=_PALETTE[m],
            edgecolor="white", linewidth=0.3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", va="bottom",
                fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT)
    ax.set_ylim(0, 110)
    _style_axes(ax, "", "Success Rate (%)", "(c) Optimization Success")

    fig.suptitle(
        "Router Benchmark Summary",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Publication summary figure saved: {save_path}")


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

    # ── 汇总可视化 ──
    visualize_summary(normal_stats, hard_stats, normal_afs, hard_afs)

    # ── XFoil 对比 ──
    all_stats_for_xfoil = normal_stats + medium_stats + hard_stats
    all_afs_for_xfoil = normal_afs + medium_afs + hard_afs
    visualize_xfoil_comparison(all_stats_for_xfoil, all_afs_for_xfoil)

    # ── 导出 CSV ──
    all_stats = normal_stats + medium_stats + hard_stats
    export_csv(all_stats)

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"总运行次数: {len(normal_runs) + len(medium_runs) + len(hard_runs)}")


if __name__ == "__main__":
    main()
