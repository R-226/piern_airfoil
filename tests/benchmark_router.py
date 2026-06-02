"""
Router Benchmark — 优化策略对比。

对比方法:
  - Baseline:     NeuralOptimizer 全 8 权重 IPOPT
  - Rule:         固定阈值 0.01 的层次化路由
  - Threshold:    网格搜索学习的最优阈值
  - PiERN Router: 基于 MLP 的学习型路由
  - XFoil+DE:     经典基线 — 差分进化 + XFoil 黑箱评估

场景:
  - Normal: 常规翼型 (30)
  - Medium: 中等翼型 (44)
  - Hard:   困难翼型 (31)

翼型从 data/benchmark_airfoils.json 加载 (固定集合, 基于 brentq 初始 CD 过滤)。

输出:
  results/benchmark_stats.csv            — 原始数据 CSV
  results/table_router_full.csv          — 完整结果表 (类别×方法)
  results/table_router_latex.tex         — LaTeX 格式结果表
  results/table_significance.csv         — 统计显著性检验 (Mann-Whitney U)
  results/benchmark_normal.png           — 常规场景对比
  results/benchmark_medium.png           — 中等场景对比
  results/benchmark_hard.png             — 困难场景对比
  results/benchmark_summary.png          — 跨类别汇总
  results/benchmark_method_comparison.png — NeuralFoil vs XFoil+DE
  results/benchmark_dist_*.png           — CD 分布箱线图 (4张)
  results/benchmark_diff_*.png           — 难度-改善散点图 (4张)
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
from scipy.stats import mannwhitneyu

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
    "xfoil_de": "XFoil+DE (classic)",
}

METHOD_COLORS = {
    "initial": "#8C8C8C",
    "baseline": "#3370AC",
    "rule": "#D44B3F",
    "threshold": "#E8A838",
    "mlp": "#2A8C6A",
    "xfoil_de": "#7B3294",
}

OPT_METHODS = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]


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
    cd_initial: float = 0.0  # 初始翼型 CD (用于计算改善率)


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


def run_once(airfoil_name: str, method: str, initial_cd: float = 0.0) -> RunResult:
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
            return RunResult(method, airfoil_name, cd, elapsed, 1,
                             success=cd < 0.15, cd_initial=initial_cd)

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
                cd_initial=initial_cd,
            )

        elif method == "xfoil_de":
            from piern_airfoil.xfoil_optimizer import xfoil_optimize
            result = xfoil_optimize(
                airfoil_name, CL_TARGETS, RE, CL_WEIGHTS, MACH,
                maxiter=8, popsize=4,  # 小参数: ~3min/翼型
            )
            elapsed = time.perf_counter() - t0
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            return RunResult(
                method, airfoil_name,
                cd=result.final_cd, time=elapsed,
                n_stages=1,
                success=result.success,
                cd_initial=initial_cd,
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
            cd_initial=initial_cd,
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
    total = len(airfoils) * (len(OPT_METHODS) + 1)  # +1: initial
    idx = 0

    for airfoil_name in airfoils:
        idx += 1
        print(f"  [{label} {idx}/{total}] {airfoil_name} initial...", end=" ", flush=True)
        r = run_initial(airfoil_name)
        all_runs.append(r)
        all_stats.append(compute_stats([r]))
        initial_cd = r.cd
        print(f"CD={r.cd:.4f}")

        for method in OPT_METHODS:
            idx += 1
            print(
                f"  [{label} {idx}/{total}] {airfoil_name} {method}...",
                end=" ", flush=True,
            )
            r = run_once(airfoil_name, method, initial_cd=initial_cd)
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
    "xfoil_de": "#7B3294",
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

    methods_all = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]
    bar_w = _PubStyle.BAR_WIDTH * 0.8  # narrower to fit 5 methods
    af_labels = [n.upper()[:12] for n in airfoils]

    # ── (a) CD comparison (absolute) ──
    ax = axes[0, 0]
    cd_data = {m: [] for m in methods_all}
    for af in airfoils:
        for m in methods_all:
            s = _get_stats(all_stats, m, af)
            cd_data[m].append(s.cd_mean if s and s.cd_mean < 1e10 else 0.0)

    _grouped_bars(ax, x, cd_data, methods_all, bar_w)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Weighted CD", "(a) Final CD")
    ax.legend(**_PubStyle.LEGEND_KW)

    # ── (b) Time comparison ──
    ax = axes[0, 1]
    time_data = {m: [] for m in methods_all}
    for af in airfoils:
        for m in methods_all:
            s = _get_stats(all_stats, m, af)
            time_data[m].append(s.time_mean if s else 0.0)

    _grouped_bars(ax, x, time_data, methods_all, bar_w)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Time (s)", "(b) Optimization Time")

    # ── (c) Success rate ──
    ax = axes[1, 0]
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

    # ── (d) CD improvement over initial ──
    ax = axes[1, 1]
    imp_data = {m: [] for m in methods_all}
    for af in airfoils:
        init_s = _get_stats(all_stats, "initial", af)
        for m in methods_all:
            meth = _get_stats(all_stats, m, af)
            if init_s and meth and init_s.cd_mean > 0 and meth.cd_mean < 1e10:
                imp_data[m].append((init_s.cd_mean - meth.cd_mean) / init_s.cd_mean * 100)
            else:
                imp_data[m].append(0.0)

    _grouped_bars(ax, x, imp_data, methods_all, bar_w)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "CD Improvement over Initial (%)", "(d) Optimization Gain")

    fig.suptitle(
        f"Router Benchmark — Normal Cases ({n_af} airfoils)",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=0.99,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPublication figure saved: {save_path}")


def visualize_medium(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_medium.png",
):
    """Medium scenario visualization — same layout as normal."""
    n_af = len(airfoils)
    x = np.arange(n_af)

    fig, axes = plt.subplots(
        2, 2, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 2),
        gridspec_kw=dict(hspace=0.55, wspace=0.38),
    )

    methods_all = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]
    bar_w = _PubStyle.BAR_WIDTH * 0.8
    af_labels = [n.upper()[:12] for n in airfoils]

    # (a) CD
    ax = axes[0, 0]
    cd_data = {m: [] for m in methods_all}
    for af in airfoils:
        for m in methods_all:
            s = _get_stats(all_stats, m, af)
            cd_data[m].append(s.cd_mean if s and s.cd_mean < 1e10 else 0.0)
    _grouped_bars(ax, x, cd_data, methods_all, bar_w)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Weighted CD", "(a) Final CD")
    ax.legend(**_PubStyle.LEGEND_KW)

    # (b) Time
    ax = axes[0, 1]
    time_data = {m: [] for m in methods_all}
    for af in airfoils:
        for m in methods_all:
            s = _get_stats(all_stats, m, af)
            time_data[m].append(s.time_mean if s else 0.0)
    _grouped_bars(ax, x, time_data, methods_all, bar_w)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "Time (s)", "(b) Optimization Time")

    # (c) Success rate
    ax = axes[1, 0]
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
    _style_axes(ax, "", "Success Rate (%)", "(c) Optimization Success (CD < 0.15)")

    # (d) CD improvement
    ax = axes[1, 1]
    imp_data = {m: [] for m in methods_all}
    for af in airfoils:
        init_s = _get_stats(all_stats, "initial", af)
        for m in methods_all:
            meth = _get_stats(all_stats, m, af)
            if init_s and meth and init_s.cd_mean > 0 and meth.cd_mean < 1e10:
                imp_data[m].append((init_s.cd_mean - meth.cd_mean) / init_s.cd_mean * 100)
            else:
                imp_data[m].append(0.0)
    _grouped_bars(ax, x, imp_data, methods_all, bar_w)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(af_labels, fontsize=5.5, rotation=90, ha="center")
    _style_axes(ax, "", "CD Improvement over Initial (%)", "(d) Optimization Gain")

    fig.suptitle(
        f"Router Benchmark — Medium Cases ({n_af} airfoils)",
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


def visualize_method_comparison(
    all_stats: list[StatsResult],
    airfoils: list[str],
    save_path: str = "results/benchmark_method_comparison.png",
):
    """NeuralFoil 方法 vs XFoil+DE 基线 — 对比优化质量、时间、成功率。

    Layout (1x3):
      (a) Final CD comparison — grouped bars across all methods
      (b) Optimization time  — grouped bars
      (c) CD vs Time scatter — Pareto-style, each method a different color
    """
    methods = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]

    fig, axes = plt.subplots(
        1, 3, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H),
        gridspec_kw=dict(wspace=0.40),
    )

    # ── (a) Mean CD per method ──
    ax = axes[0]
    mean_cds = []
    for m in methods:
        ok = [s for s in all_stats if s.method == m and s.cd_mean < 1e10 and s.airfoil_name in airfoils]
        mean_cds.append(np.mean([s.cd_mean for s in ok]) if ok else 0)

    bars = ax.bar(
        range(len(methods)), mean_cds, 0.55,
        color=[_PALETTE.get(m, "#888") for m in methods],
        edgecolor="white", linewidth=0.3,
    )
    for bar, v in zip(bars, mean_cds):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=6, fontfamily=_SERIF_FONT)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods],
                       fontsize=7, fontfamily=_SERIF_FONT, rotation=20, ha="right")
    _style_axes(ax, "", "Mean Weighted CD", "(a) Optimization Quality")

    # ── (b) Mean time per method ──
    ax = axes[1]
    mean_times = []
    for m in methods:
        ok = [s for s in all_stats if s.method == m and s.airfoil_name in airfoils]
        mean_times.append(np.mean([s.time_mean for s in ok]) if ok else 0)

    bars = ax.bar(
        range(len(methods)), mean_times, 0.55,
        color=[_PALETTE.get(m, "#888") for m in methods],
        edgecolor="white", linewidth=0.3,
    )
    for bar, v in zip(bars, mean_times):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}s", ha="center", va="bottom",
                    fontsize=6, fontfamily=_SERIF_FONT)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods],
                       fontsize=7, fontfamily=_SERIF_FONT, rotation=20, ha="right")
    _style_axes(ax, "", "Mean Time (s)", "(b) Optimization Speed")

    # ── (c) CD vs Time scatter (Pareto) ──
    ax = axes[2]
    for m in methods:
        ok = [s for s in all_stats if s.method == m and s.cd_mean < 1e10 and s.airfoil_name in airfoils]
        if ok:
            mean_cd = np.mean([s.cd_mean for s in ok])
            mean_time = np.mean([s.time_mean for s in ok])
            ax.scatter(mean_time, mean_cd, c=_PALETTE.get(m, "#888"), s=80,
                       label=METHOD_LABELS[m], edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel("Mean Time (s)", fontsize=8, fontfamily=_SERIF_FONT)
    ax.set_ylabel("Mean CD", fontsize=8, fontfamily=_SERIF_FONT)
    ax.set_title("(c) Pareto Front (CD vs Time)", fontsize=10, fontfamily=_SERIF_FONT)
    ax.legend(fontsize=6, frameon=False)
    _style_axes(ax, "", "", "")

    fig.suptitle("NeuralFoil Methods vs XFoil+DE Baseline",
                 fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Method comparison figure saved: {save_path}")


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


# ── 统计显著性检验 ──────────────────────────────────────────────────────


def _rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Mann-Whitney U 的 rank-biserial effect size (r = 1 - 2U/(n1*n2))。"""
    return 1.0 - 2.0 * u_stat / (n1 * n2)


def run_significance_tests(
    all_stats: list[StatsResult],
    airfoils: list[str],
    title: str = "",
) -> list[dict]:
    """对每个方法 vs baseline 做 Mann-Whitney U 检验 (CD)。

    Returns:
        list of dicts with keys: method, category, u_stat, p_value,
        effect_size_r, n_baseline, n_method, median_baseline, median_method
    """
    results = []
    methods = ["rule", "threshold", "mlp", "xfoil_de"]

    for method in methods:
        # 收集 baseline 和 method 的 CD 值
        cds_base = []
        cds_meth = []
        for af in airfoils:
            s_base = _get_stats(all_stats, "baseline", af)
            s_meth = _get_stats(all_stats, method, af)
            if s_base and s_meth and s_base.cd_mean < 1e10 and s_meth.cd_mean < 1e10:
                cds_base.append(s_base.cd_mean)
                cds_meth.append(s_meth.cd_mean)

        if len(cds_base) < 5 or len(cds_meth) < 5:
            continue

        cds_base = np.array(cds_base)
        cds_meth = np.array(cds_meth)

        # Mann-Whitney U (alternative='less': method 的 CD 是否显著低于 baseline)
        try:
            u_stat, p_value = mannwhitneyu(cds_meth, cds_base, alternative="less")
        except ValueError:
            u_stat, p_value = 0.0, 1.0

        effect_r = _rank_biserial_r(u_stat, len(cds_base), len(cds_meth))

        results.append({
            "title": title,
            "method": method,
            "u_stat": u_stat,
            "p_value": p_value,
            "effect_size_r": effect_r,
            "n_baseline": len(cds_base),
            "n_method": len(cds_meth),
            "median_baseline": float(np.median(cds_base)),
            "median_method": float(np.median(cds_meth)),
        })

    return results


def print_significance_tests(
    test_results: list[dict],
    title: str = "",
):
    """打印显著性检验结果表。"""
    if not test_results:
        return

    print(f"\n{'='*90}")
    print(f"统计显著性检验 — {title}")
    print(f"方法: Mann-Whitney U (one-sided: method CD < baseline CD)")
    print(f"Effect size: rank-biserial r (0=无差异, ±1=完全分离)")
    print(f"{'='*90}")
    print(
        f"{'方法':<16} {'n':>4} {'Median CD':>12} {'vs Baseline':>12} "
        f"{'U':>8} {'p-value':>10} {'r':>8} {'显著?':>6}"
    )
    print("-" * 90)

    for r in test_results:
        sig = "***" if r["p_value"] < 0.001 else (
            "**" if r["p_value"] < 0.01 else (
                "*" if r["p_value"] < 0.05 else "n.s."
            )
        )
        print(
            f"{METHOD_LABELS.get(r['method'], r['method']):<16} "
            f"{r['n_method']:>4} "
            f"{r['median_method']:>12.6f} "
            f"{r['median_baseline']:>12.6f} "
            f"{r['u_stat']:>8.0f} "
            f"{r['p_value']:>10.4f} "
            f"{r['effect_size_r']:>+8.3f} "
            f"{sig:>6}"
        )


# ── 综合结果表 ──────────────────────────────────────────────────────────


def generate_results_table(
    all_stats: list[StatsResult],
    normal_afs: list[str],
    medium_afs: list[str],
    hard_afs: list[str],
    sig_results: list[dict],
    save_dir: str = "results",
):
    """生成完整的论文结果表 (CSV + LaTeX)。

    输出:
      results/table_router_full.csv   — 按类别×方法的完整指标
      results/table_router_latex.tex  — LaTeX 格式
      results/table_significance.csv  — 显著性检验结果
    """
    import csv

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        ("Normal", normal_afs),
        ("Medium", medium_afs),
        ("Hard", hard_afs),
        ("All", normal_afs + medium_afs + hard_afs),
    ]

    methods = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]

    # ── 1. 完整 CSV ──
    rows = []
    for cat_name, afs in categories:
        for method in methods:
            ss = [s for s in all_stats if s.method == method and s.airfoil_name in afs]
            ss_valid = [s for s in ss if s.cd_mean < 1e10]
            if not ss_valid:
                rows.append({
                    "category": cat_name,
                    "method": METHOD_LABELS.get(method, method),
                    "n": len(ss),
                    "cd_mean": "",
                    "cd_std": "",
                    "cd_median": "",
                    "cd_p25": "",
                    "cd_p75": "",
                    "time_mean": "",
                    "time_std": "",
                    "time_median": "",
                    "success_rate": "",
                    "n_stages_mean": "",
                })
                continue

            cds = np.array([s.cd_mean for s in ss_valid])
            times = np.array([s.time_mean for s in ss_valid])
            srs = np.array([s.success_rate for s in ss_valid])

            rows.append({
                "category": cat_name,
                "method": METHOD_LABELS.get(method, method),
                "n": len(ss_valid),
                "cd_mean": f"{np.mean(cds):.6f}",
                "cd_std": f"{np.std(cds):.6f}",
                "cd_median": f"{np.median(cds):.6f}",
                "cd_p25": f"{np.percentile(cds, 25):.6f}",
                "cd_p75": f"{np.percentile(cds, 75):.6f}",
                "time_mean": f"{np.mean(times):.2f}",
                "time_std": f"{np.std(times):.2f}",
                "time_median": f"{np.median(times):.2f}",
                "success_rate": f"{np.mean(srs):.4f}",
                "n_stages_mean": f"{np.mean([s.n_stages_mean for s in ss_valid]):.1f}",
            })

    csv_path = save_dir / "table_router_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"完整结果表 CSV: {csv_path}")

    # ── 2. LaTeX 表 ──
    tex_path = save_dir / "table_router_latex.tex"
    with open(tex_path, "w") as f:
        f.write("% 自动生成 — 不要手动编辑\n")
        f.write("% 用法: \\input{table_router_latex.tex}\n\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Optimization benchmark results across difficulty tiers.}\n")
        f.write("\\label{tab:benchmark}\n")
        f.write("\\smallskip\n")
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\toprule\n")
        f.write("Category & Method & $n$ & CD$_{\\mathrm{mean}}$ & CD$_{\\mathrm{std}}$ & "
                "CD$_{\\mathrm{med}}$ & Time (s) & SR (\\%) \\\\\n")
        f.write("\\midrule\n")

        prev_cat = None
        for row in rows:
            if row["category"] != prev_cat:
                if prev_cat is not None:
                    f.write("\\midrule\n")
                prev_cat = row["category"]

            sr_pct = f"{float(row['success_rate'])*100:.0f}" if row["success_rate"] else "-"
            cd_mean = row["cd_mean"] if row["cd_mean"] else "-"
            cd_std = row["cd_std"] if row["cd_std"] else "-"
            cd_med = row["cd_median"] if row["cd_median"] else "-"
            time_m = row["time_mean"] if row["time_mean"] else "-"
            n = row["n"]

            f.write(
                f"{row['category']} & {row['method']} & {n} & "
                f"{cd_mean} & {cd_std} & {cd_med} & {time_m} & {sr_pct} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"LaTeX 结果表: {tex_path}")

    # ── 3. 显著性检验 CSV ──
    if sig_results:
        sig_path = save_dir / "table_significance.csv"
        with open(sig_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sig_results[0].keys())
            writer.writeheader()
            writer.writerows(sig_results)
        print(f"显著性检验表: {sig_path}")


# ── 分布可视化 ──────────────────────────────────────────────────────────


def visualize_distributions(
    all_stats: list[StatsResult],
    airfoils: list[str],
    category: str = "All",
    save_path: str = "results/benchmark_distributions.png",
):
    """CD 分布箱线图 — 展示各方法的 CD 分布而非仅均值。

    Layout (1x2):
      (a) CD distribution (boxplot + strip) — all methods
      (b) CD improvement over baseline (per-airfoil) — strip + mean
    """
    methods = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]
    n_af = len(airfoils)

    fig, axes = plt.subplots(
        1, 2, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 0.9),
        gridspec_kw=dict(wspace=0.35),
    )

    # ── (a) CD distribution boxplot ──
    ax = axes[0]
    box_data = []
    for m in methods:
        cds = []
        for af in airfoils:
            s = _get_stats(all_stats, m, af)
            if s and s.cd_mean < 1e10:
                cds.append(s.cd_mean)
        box_data.append(cds)

    bp = ax.boxplot(
        box_data,
        positions=range(len(methods)),
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(_PALETTE.get(m, "#888888"))
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    # Strip plot overlay
    for i, m in enumerate(methods):
        cds = box_data[i]
        if cds:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(cds))
            ax.scatter(
                np.full_like(cds, i, dtype=float) + jitter,
                cds, s=12, alpha=0.4, color=_PALETTE.get(m, "#888"),
                edgecolors="none", zorder=3,
            )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in methods],
        fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT, rotation=20, ha="right",
    )
    _style_axes(ax, "", "Weighted CD", f"(a) CD Distribution — {category}")

    # ── (b) CD improvement over baseline (per-airfoil) ──
    ax = axes[1]
    for i, m in enumerate(["rule", "threshold", "mlp", "xfoil_de"]):
        imps = []
        for af in airfoils:
            base = _get_stats(all_stats, "baseline", af)
            meth = _get_stats(all_stats, m, af)
            if base and meth and base.cd_mean > 0 and meth.cd_mean < 1e10:
                imps.append((base.cd_mean - meth.cd_mean) / base.cd_mean * 100)
        if imps:
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(imps))
            ax.scatter(
                np.full_like(imps, i, dtype=float) + jitter,
                imps, s=18, alpha=0.5, color=_PALETTE.get(m, "#888"),
                edgecolors="none", zorder=3,
            )
            mean_imp = np.mean(imps)
            ax.scatter([i], [mean_imp], s=80, color=_PALETTE.get(m, "#888"),
                       edgecolors="black", linewidths=0.8, zorder=5, marker="D")
            ax.annotate(
                f"{mean_imp:+.1f}%", (i, mean_imp),
                textcoords="offset points", xytext=(8, 0),
                fontsize=_PubStyle.ANNOT_SIZE, fontfamily=_SERIF_FONT,
                color=_PALETTE.get(m, "#888"), fontweight="bold",
            )

    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in ["rule", "threshold", "mlp", "xfoil_de"]],
        fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT, rotation=20, ha="right",
    )
    _style_axes(ax, "", "CD Improvement over Baseline (%)", f"(b) Per-Airfoil Improvement — {category}")

    fig.suptitle(
        f"CD Distribution — {category} ({n_af} airfoils)",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Distribution figure saved: {save_path}")


# ── 难度-改善散点图 ────────────────────────────────────────────────────


def visualize_difficulty_improvement(
    all_stats: list[StatsResult],
    airfoils: list[str],
    category: str = "All",
    save_path: str = "results/benchmark_difficulty.png",
):
    """初始 CD (难度) vs CD 改善率散点图。

    Layout (1x2):
      (a) Per-airfoil scatter — x: initial CD, y: improvement %
      (b) Binned mean improvement — grouped by difficulty quartiles
    """
    methods = ["rule", "threshold", "mlp", "xfoil_de"]

    fig, axes = plt.subplots(
        1, 2, figsize=(_PubStyle.FIG_W, _PubStyle.ROW_H * 0.9),
        gridspec_kw=dict(wspace=0.35),
    )

    # ── (a) Per-airfoil scatter ──
    ax = axes[0]
    for m in methods:
        x_vals, y_vals = [], []
        for af in airfoils:
            init = _get_stats(all_stats, "initial", af)
            meth = _get_stats(all_stats, m, af)
            if init and meth and init.cd_mean > 0 and meth.cd_mean < 1e10:
                x_vals.append(init.cd_mean)
                y_vals.append((init.cd_mean - meth.cd_mean) / init.cd_mean * 100)

        if x_vals:
            ax.scatter(
                x_vals, y_vals,
                s=25, alpha=0.5, color=_PALETTE.get(m, "#888"),
                label=METHOD_LABELS[m], edgecolors="none", zorder=3,
            )
            # LOWESS 趋势线
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                x_arr, y_arr = np.array(x_vals), np.array(y_vals)
                order = np.argsort(x_arr)
                smoothed = lowess(y_arr[order], x_arr[order], frac=0.6, return_sorted=True)
                ax.plot(
                    smoothed[:, 0], smoothed[:, 1],
                    color=_PALETTE.get(m, "#888"), linewidth=1.5, alpha=0.8, zorder=4,
                )
            except ImportError:
                pass

    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xscale("log")
    _style_axes(ax, "Initial CD (airfoil difficulty)", "CD Improvement over Baseline (%)",
                f"(a) Difficulty vs Improvement — {category}")
    ax.legend(fontsize=6, frameon=False)

    # ── (b) Binned mean improvement ──
    ax = axes[1]
    # 按初始 CD 分四分位
    init_cds = []
    for af in airfoils:
        s = _get_stats(all_stats, "initial", af)
        if s and s.cd_mean > 0:
            init_cds.append((af, s.cd_mean))
    if not init_cds:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        init_cds.sort(key=lambda x: x[1])
        n = len(init_cds)
        quartiles = [
            init_cds[:n//4],
            init_cds[n//4:n//2],
            init_cds[n//2:3*n//4],
            init_cds[3*n//4:],
        ]
        q_labels = ["Q1\n(easiest)", "Q2", "Q3", "Q4\n(hardest)"]
        x = np.arange(4)
        bar_w = 0.18

        for i, m in enumerate(methods):
            vals = []
            for q_afs in quartiles:
                imps = []
                for af, _ in q_afs:
                    init = _get_stats(all_stats, "initial", af)
                    meth = _get_stats(all_stats, m, af)
                    if init and meth and init.cd_mean > 0 and meth.cd_mean < 1e10:
                        imps.append((init.cd_mean - meth.cd_mean) / init.cd_mean * 100)
                vals.append(np.mean(imps) if imps else 0)
            offset = (i - 1.5) * bar_w
            bars = ax.bar(
                x + offset, vals, bar_w,
                label=METHOD_LABELS[m], color=_PALETTE.get(m, "#888"),
                edgecolor="white", linewidth=0.3,
            )
            _annotate_bars(ax, bars, vals, suffix="%", offset_ratio=0.04)

        ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=_PubStyle.TICK_SIZE, fontfamily=_SERIF_FONT)
        _style_axes(ax, "Difficulty Quartile", "Mean CD Improvement (%)",
                    f"(b) Improvement by Difficulty — {category}")
        ax.legend(fontsize=6, frameon=False)

    fig.suptitle(
        f"Difficulty vs Improvement — {category} ({len(airfoils)} airfoils)",
        fontsize=11, fontfamily=_SERIF_FONT, fontweight="bold", y=1.02,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Difficulty-improvement figure saved: {save_path}")


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
    visualize_medium(medium_stats, medium_afs)

    # ── 场景3: 困难翼型 ──
    print("\n>>> 场景3: 困难翼型 (Baseline 失败)")
    hard_stats, hard_runs = run_benchmark_group(hard_afs, "Hard")
    print_summary(hard_stats, "场景3: 困难翼型", hard_afs)
    visualize_hard(hard_stats, hard_afs)

    # ── 汇总可视化 ──
    visualize_summary(normal_stats, hard_stats, normal_afs, hard_afs)

    # ── NeuralFoil vs XFoil+DE 对比 ──
    all_stats_combined = normal_stats + medium_stats + hard_stats
    all_afs_combined = normal_afs + medium_afs + hard_afs
    visualize_method_comparison(all_stats_combined, all_afs_combined)

    # ── 分布可视化 ──
    visualize_distributions(all_stats_combined, normal_afs, "Normal", "results/benchmark_dist_normal.png")
    visualize_distributions(all_stats_combined, medium_afs, "Medium", "results/benchmark_dist_medium.png")
    visualize_distributions(all_stats_combined, hard_afs, "Hard", "results/benchmark_dist_hard.png")
    visualize_distributions(all_stats_combined, all_afs_combined, "All", "results/benchmark_dist_all.png")

    # 难度-改善散点图
    visualize_difficulty_improvement(all_stats_combined, normal_afs, "Normal", "results/benchmark_diff_normal.png")
    visualize_difficulty_improvement(all_stats_combined, medium_afs, "Medium", "results/benchmark_diff_medium.png")
    visualize_difficulty_improvement(all_stats_combined, hard_afs, "Hard", "results/benchmark_diff_hard.png")
    visualize_difficulty_improvement(all_stats_combined, all_afs_combined, "All", "results/benchmark_diff_all.png")

    # ── 统计显著性检验 ──
    all_stats = normal_stats + medium_stats + hard_stats
    print("\n>>> 统计显著性检验")
    sig_normal = run_significance_tests(all_stats, normal_afs, "Normal")
    sig_medium = run_significance_tests(all_stats, medium_afs, "Medium")
    sig_hard = run_significance_tests(all_stats, hard_afs, "Hard")
    sig_all = run_significance_tests(all_stats, all_afs_combined, "All")
    print_significance_tests(sig_normal, "Normal")
    print_significance_tests(sig_medium, "Medium")
    print_significance_tests(sig_hard, "Hard")
    print_significance_tests(sig_all, "All")

    # ── 综合结果表 ──
    generate_results_table(
        all_stats, normal_afs, medium_afs, hard_afs,
        sig_all, save_dir="results",
    )

    # ── 导出 CSV ──
    export_csv(all_stats)

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"总运行次数: {len(normal_runs) + len(medium_runs) + len(hard_runs)}")


if __name__ == "__main__":
    main()
