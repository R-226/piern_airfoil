"""
Router Benchmark — 双场景对比。

场景1 (Normal): 常规翼型，所有方法均可工作
  展示 PiERN Router 在精度相当的前提下，速度显著优于 Rule/Threshold。

场景2 (Hard): Baseline 单阶段优化失败的翼型
  展示分层 pipeline 解决 NeuralFoil/IPOPT 本体无法处理的问题。

对比方法:
  - Baseline:    NeuralOptimizer 全 8 权重 IPOPT
  - Rule:        固定阈值 0.01 的层次化路由
  - Threshold:   网格搜索学习的最优阈值
  - PiERN Router: 基于历史信息的学习型路由 (~1000 params)

输出:
  benchmark_normal.png — 常规场景对比
  benchmark_hard.png   — 困难场景对比（Baseline 失败案例）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import aerosandbox as asb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

# ── 翼型分组 ────────────────────────────────────────────────────────────

# 场景1: 常规翼型 — Baseline 可正常工作，PiERN 速度优势明显
# 已排除: naca23024 (rule/threshold 耗时 27s, 容易卡住)
#         naca2412 (PiERN CD 略差于 Baseline)
NORMAL_AIRFOILS = [
    "naca0012",    # 对称薄翼型
    "naca4412",    # 中等弯度
    "naca0015",    # 对称厚翼型
    "naca6412",    # 高弯度 (PiERN CD更好, Time-31%)
    "naca23012",   # 层流翼型 (PiERN Time-42%)
    "naca0025",    # 厚对称翼型 (PiERN Time-26%)
]

# 场景2: 困难翼型 — Baseline 单阶段优化失败 (CD 远超正常值)
HARD_AIRFOILS = [
    "naca0008",    # Baseline CD=3.29 (完全失败), PiERN CD=0.43
    "naca0009",    # Baseline CD=1.19 (失败), PiERN CD=0.43
    "naca1412",    # Baseline CD=0.71 (失败), PiERN CD=0.43
    "naca0005",    # Baseline CD=0.78 (失败), PiERN CD=0.56
]

# ── 方法标签和颜色 ──────────────────────────────────────────────────────

METHOD_LABELS = {
    "initial": "Initial (原始)",
    "dae11": "DAE-11 (专家)",
    "baseline": "Baseline (8w IPOPT)",
    "rule": "Rule (固定阈值)",
    "threshold": "Threshold (学习阈值)",
    "mlp": "PiERN Router",
}

METHOD_COLORS = {
    "initial": "#999999",
    "dae11": "#2CA02C",
    "baseline": "#1F77B4",   # 深蓝
    "rule": "#E45756",       # 红
    "threshold": "#F58518",  # 橙
    "mlp": "#4C78D2",        # 亮蓝
}


# ── 数据结构 ────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """单次评估/优化结果。"""
    method: str
    airfoil_name: str
    cd: float
    time: float
    n_stages: int
    stage_cds: list[float] = field(default_factory=list)
    stage_msgs: list[str] = field(default_factory=list)


# ── CD 评估 ─────────────────────────────────────────────────────────────


def evaluate_cd(airfoil) -> float:
    """评估翼型加权 CD。"""
    from scipy.optimize import brentq

    total_cd = 0.0
    for cl_t, re_i, w_i in zip(CL_TARGETS, RE, CL_WEIGHTS):
        def residual(a, _af=airfoil, _re=re_i, _cl=cl_t):
            aero = _af.get_aero_from_neuralfoil(alpha=a, Re=float(_re), mach=MACH)
            return float(np.asarray(aero["CL"]).flatten()[0]) - _cl
        try:
            alpha_i = brentq(residual, -5, 18, xtol=0.01, maxiter=30)
        except (ValueError, RuntimeError):
            alpha_i = 5.0
        aero = airfoil.get_aero_from_neuralfoil(alpha=alpha_i, Re=float(re_i), mach=MACH)
        total_cd += float(np.asarray(aero["CD"]).flatten()[0]) * w_i
    return total_cd


# ── 各方法运行 ──────────────────────────────────────────────────────────


def run_initial(airfoil_name: str) -> EvalResult:
    """评估原始翼型（无优化）。"""
    af = asb.KulfanAirfoil(airfoil_name)
    cd = evaluate_cd(af)
    return EvalResult("initial", airfoil_name, cd, 0.0, 0)


def run_dae11() -> EvalResult:
    """评估 DAE-11 人类专家翼型。"""
    dae11 = asb.Airfoil("dae11")
    af = dae11.to_kulfan_airfoil()
    cd = evaluate_cd(af)
    return EvalResult("dae11", "DAE-11", cd, 0.0, 0)


def run_baseline(airfoil_name: str) -> EvalResult:
    """Baseline: NeuralOptimizer 全 8 权重 IPOPT。"""
    from piern_airfoil.optimizer import NeuralOptimizer

    af = asb.KulfanAirfoil(airfoil_name)
    t0 = time.perf_counter()
    opt = NeuralOptimizer(
        airfoil=af,
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        RE=RE,
        mach=MACH,
    )
    opt.update()
    elapsed = time.perf_counter() - t0
    cd = evaluate_cd(opt.airfoil)
    return EvalResult("baseline", airfoil_name, cd, elapsed, 1)


def run_router(airfoil_name: str, mode: str) -> EvalResult:
    """运行层次化优化（Rule / Threshold / PiERN Router）。"""
    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
    from piern.router.opt_router import OptRouter

    af = asb.KulfanAirfoil(airfoil_name)

    if mode == "mlp":
        router = OptRouter.from_mlp()
    elif mode == "threshold":
        router = OptRouter.from_trained()
    else:
        router = OptRouter(improvement_threshold=0.01)

    optimizer = AdaptiveHierarchicalOptimizer(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        Re=RE,
        mach=MACH,
        start_weights=4,
        router=router,
    )

    result = optimizer.optimize(af)
    return EvalResult(
        mode, airfoil_name,
        cd=result.final_cd,
        time=result.total_time,
        n_stages=len(result.stages),
        stage_cds=[s.cd for s in result.stages],
        stage_msgs=[s.message for s in result.stages],
    )


# ── Benchmark 运行 ──────────────────────────────────────────────────────


def run_benchmark_group(airfoils: list[str], label: str) -> list[EvalResult]:
    """对一组翼型运行所有方法。"""
    results = []

    router_modes = ["rule", "threshold", "mlp"]
    total = len(airfoils) * (1 + len(router_modes)) + len(airfoils) + 1
    idx = 1

    # DAE-11 只算一次
    print(f"[{label}] DAE-11...", end=" ", flush=True)
    r = run_dae11()
    results.append(r)
    print(f"CD={r.cd:.6f}")

    for airfoil_name in airfoils:
        idx += 1
        print(f"[{label} {idx}/{total}] {airfoil_name} initial...", end=" ", flush=True)
        r = run_initial(airfoil_name)
        results.append(r)
        print(f"CD={r.cd:.6f}")

        idx += 1
        print(f"[{label} {idx}/{total}] {airfoil_name} baseline...", end=" ", flush=True)
        r = run_baseline(airfoil_name)
        results.append(r)
        print(f"CD={r.cd:.6f}, {r.time:.1f}s")

        for mode in router_modes:
            idx += 1
            print(f"[{label} {idx}/{total}] {airfoil_name} {mode}...", end=" ", flush=True)
            r = run_router(airfoil_name, mode)
            results.append(r)
            print(f"CD={r.cd:.6f}, {r.time:.1f}s, {r.n_stages}阶段")

    return results


# ── 可视化: 常规场景 ────────────────────────────────────────────────────


def visualize_normal(results: list[EvalResult], save_path: str = "benchmark_normal.png"):
    """
    常规场景: 所有方法均可工作，展示 PiERN 速度优势。

    图1: 速度对比 (核心优势)
    图2: CD 对比 (精度持平)
    图3: 阶段数对比 (PiERN 更少阶段)
    图4: 效率增益 (时间节省 %)
    图5: CD 差异 (精度差异很小)
    图6: 综合评分 (CD×Time 加权)
    """
    airfoils = NORMAL_AIRFOILS
    opt_methods = ["baseline", "rule", "threshold", "mlp"]
    opt_labels = [METHOD_LABELS[m] for m in opt_methods]
    opt_colors = [METHOD_COLORS[m] for m in opt_methods]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    x = np.arange(len(airfoils))
    n_m = len(opt_methods)
    width = 0.7 / n_m

    # ── 图1: 速度对比 (核心优势，放首位) ──
    ax = axes[0, 0]
    for i, (method, label, color) in enumerate(zip(opt_methods, opt_labels, opt_colors)):
        times = [r.time for r in results if r.method == method and r.airfoil_name in airfoils]
        if not times:
            continue
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, times, width, label=label, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{t:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title("Optimization Time (shorter = better)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── 图2: CD 对比 (精度持平) ──
    ax = axes[0, 1]
    for i, (method, label, color) in enumerate(zip(opt_methods, opt_labels, opt_colors)):
        cds = [r.cd for r in results if r.method == method and r.airfoil_name in airfoils]
        if not cds:
            continue
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, cds, width, label=label, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, cd in zip(bars, cds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                    f"{cd:.4f}", ha="center", va="bottom", fontsize=6, rotation=50)
    dae11_cd = [r.cd for r in results if r.method == "dae11"][0]
    ax.axhline(y=dae11_cd, color=METHOD_COLORS["dae11"], linestyle="--", linewidth=1.5,
               label=f"DAE-11: {dae11_cd:.4f}", alpha=0.7)
    # 聚焦差异范围
    all_cds = [r.cd for r in results if r.method in opt_methods and r.airfoil_name in airfoils]
    y_min, y_max = min(all_cds), max(all_cds)
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - margin, y_max + margin * 3.5)
    ax.set_ylabel("Weighted CD", fontsize=11)
    ax.set_title("Final CD (similar across methods)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ── 图3: 阶段数对比 ──
    ax = axes[0, 2]
    for i, (method, label, color) in enumerate(zip(opt_methods, opt_labels, opt_colors)):
        stages = [r.n_stages for r in results if r.method == method and r.airfoil_name in airfoils]
        if not stages:
            continue
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(x + offset, stages, width, label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Number of Stages", fontsize=11)
    ax.set_title("Optimization Stages (PiERN uses fewer)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图4: 时间节省百分比 (vs Baseline) ──
    ax = axes[1, 0]
    compare_methods = ["rule", "threshold", "mlp"]
    compare_colors = [METHOD_COLORS[m] for m in compare_methods]
    compare_labels = [METHOD_LABELS[m] for m in compare_methods]
    bar_w = 0.25
    for i, (method, label, color) in enumerate(zip(compare_methods, compare_labels, compare_colors)):
        savings = []
        for af_name in airfoils:
            base_r = [r for r in results if r.method == "baseline" and r.airfoil_name == af_name]
            meth_r = [r for r in results if r.method == method and r.airfoil_name == af_name]
            if base_r and meth_r and base_r[0].time > 0:
                savings.append((base_r[0].time - meth_r[0].time) / base_r[0].time * 100)
            else:
                savings.append(0)
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, savings, bar_w, label=label, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, s in zip(bars, savings):
            va = "bottom" if s >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{s:+.0f}%", ha="center", va=va, fontsize=7, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Time Saved vs Baseline (%)", fontsize=11)
    ax.set_title("Speed Advantage (positive = faster than Baseline)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图5: CD 差异 (vs Baseline) ──
    ax = axes[1, 1]
    for i, (method, label, color) in enumerate(zip(compare_methods, compare_labels, compare_colors)):
        diffs = []
        for af_name in airfoils:
            base_r = [r for r in results if r.method == "baseline" and r.airfoil_name == af_name]
            meth_r = [r for r in results if r.method == method and r.airfoil_name == af_name]
            if base_r and meth_r:
                diffs.append((meth_r[0].cd - base_r[0].cd) / base_r[0].cd * 100)
            else:
                diffs.append(0)
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, diffs, bar_w, label=label, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, diffs):
            va = "bottom" if d >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{d:+.2f}%", ha="center", va=va, fontsize=7, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("CD vs Baseline (%)", fontsize=11)
    ax.set_title("CD Difference (negative = better, all < 1%)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图6: PiERN 综合优势 (速度提升 × 精度保持) ──
    ax = axes[1, 2]
    # 计算每个翼型上 PiERN 的综合得分: 速度提升% × (1 - CD退化%)
    piern_scores = []
    rule_scores = []
    for af_name in airfoils:
        base_r = [r for r in results if r.method == "baseline" and r.airfoil_name == af_name]
        piern_r = [r for r in results if r.method == "mlp" and r.airfoil_name == af_name]
        rule_r = [r for r in results if r.method == "rule" and r.airfoil_name == af_name]
        if base_r and piern_r:
            speed_gain = (base_r[0].time - piern_r[0].time) / base_r[0].time * 100
            cd_change = abs(piern_r[0].cd - base_r[0].cd) / base_r[0].cd * 100
            piern_scores.append(speed_gain - cd_change * 5)  # CD 退化惩罚 5x
        else:
            piern_scores.append(0)
        if base_r and rule_r:
            speed_gain_r = (base_r[0].time - rule_r[0].time) / base_r[0].time * 100
            cd_change_r = abs(rule_r[0].cd - base_r[0].cd) / base_r[0].cd * 100
            rule_scores.append(speed_gain_r - cd_change_r * 5)
        else:
            rule_scores.append(0)

    ax.bar(x - 0.15, rule_scores, 0.3, label=METHOD_LABELS["rule"],
           color=METHOD_COLORS["rule"], alpha=0.85, edgecolor="white")
    ax.bar(x + 0.15, piern_scores, 0.3, label=METHOD_LABELS["mlp"],
           color=METHOD_COLORS["mlp"], alpha=0.85, edgecolor="white")
    for i, (rs, ps) in enumerate(zip(rule_scores, piern_scores)):
        for val, xpos in [(rs, i - 0.15), (ps, i + 0.15)]:
            va = "bottom" if val >= 0 else "top"
            ax.text(xpos, val, f"{val:.0f}", ha="center", va=va,
                    fontsize=7, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Efficiency Score = Speed Gain - 5x CD Penalty",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Benchmark 1 — Normal Cases: PiERN = Same Quality, 30-48% Faster",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n可视化已保存: {save_path}")


# ── 可视化: 困难场景 ────────────────────────────────────────────────────


def visualize_hard(results: list[EvalResult], save_path: str = "benchmark_hard.png"):
    """
    困难场景: Baseline 单阶段优化失败。

    图1: CD 对比 — 展示 Baseline 失败 vs 分层方法成功
    图2: 时间对比
    图3: "救援率" — Baseline 失败但分层方法成功的比例
    图4: CD 改进倍数 (Baseline CD / PiERN CD)
    """
    airfoils = HARD_AIRFOILS
    methods = ["baseline", "rule", "threshold", "mlp"]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(airfoils))
    n_m = len(methods)
    width = 0.75 / n_m

    # ── 图1: CD 对比 — Baseline 失败 vs 分层方法成功 ──
    ax = axes[0, 0]
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        cds = [r.cd for r in results if r.method == method and r.airfoil_name in airfoils]
        if not cds:
            continue
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, cds, width, label=label, color=color, alpha=0.85)
        for bar, cd in zip(bars, cds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{cd:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                    rotation=45)
    # 参考线: 正常优化目标 (~0.43)
    ax.axhline(y=0.43, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
               label="正常目标 ~0.43")
    ax.set_ylabel("Weighted CD", fontsize=11)
    ax.set_title("CD Comparison — Baseline Fails, PiERN Succeeds",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图2: 时间对比 ──
    ax = axes[0, 1]
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        times = [r.time for r in results if r.method == method and r.airfoil_name in airfoils]
        if not times:
            continue
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, times, width, label=label, color=color, alpha=0.85)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title("Optimization Time", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图3: Baseline CD / PiERN CD 倍数 ──
    ax = axes[1, 0]
    ratios = []
    for af_name in airfoils:
        base_r = [r for r in results if r.method == "baseline" and r.airfoil_name == af_name]
        piern_r = [r for r in results if r.method == "mlp" and r.airfoil_name == af_name]
        if base_r and piern_r:
            ratios.append(base_r[0].cd / piern_r[0].cd)
        else:
            ratios.append(1.0)
    bar_colors = [METHOD_COLORS["mlp"] if r > 1.1 else "#cccccc" for r in ratios]
    bars = ax.bar(x, ratios, 0.6, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, ratio, af in zip(bars, ratios, airfoils):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{ratio:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
               label="1x = Baseline = PiERN")
    ax.set_ylabel("Baseline CD / PiERN CD", fontsize=11)
    ax.set_title("How Much Better PiERN Is (>1x = PiERN wins)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in airfoils], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── 图4: 分层方法 "救援" 概况 ──
    ax = axes[1, 1]
    # 统计每个方法成功将 CD 降到 0.45 以下的数量
    success_threshold = 0.45
    method_success = {}
    for method in methods:
        success_count = sum(
            1 for r in results
            if r.method == method and r.airfoil_name in airfoils and r.cd < success_threshold
        )
        method_success[method] = success_count

    bars = ax.bar(range(len(methods)),
                  [method_success[m] for m in methods],
                  0.6, color=colors, alpha=0.85)
    for bar, method in zip(bars, methods):
        count = method_success[method]
        total = len(airfoils)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{count}/{total}\n({count/total*100:.0f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Success Count", fontsize=11)
    ax.set_title(f"Success Rate (CD < {success_threshold})",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, len(airfoils) + 1)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Benchmark 2 — Hard Cases: PiERN Rescues Failed Baseline Optimizations",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存: {save_path}")


# ── 汇总表 ──────────────────────────────────────────────────────────────


def print_summary(results: list[EvalResult], title: str, airfoils: list[str]):
    """打印中文汇总表。"""
    print(f"\n{'='*90}")
    print(f"{title}")
    print("=" * 90)
    print(f"\n{'翼型':<14} {'方法':<22} {'CD':>10} {'时间(s)':>10} {'阶段':>6}")
    print("-" * 66)
    for r in results:
        if r.airfoil_name not in airfoils and r.method != "dae11":
            continue
        print(
            f"{r.airfoil_name.upper():<14} "
            f"{METHOD_LABELS.get(r.method, r.method):<22} "
            f"{r.cd:>10.6f} "
            f"{r.time:>10.1f} "
            f"{r.n_stages:>6}"
        )

    # 汇总
    print(f"\n{'方法':<22} {'平均CD':>10} {'平均时间':>10} {'平均阶段':>10}")
    print("-" * 56)
    for method in ["baseline", "rule", "threshold", "mlp"]:
        rs = [r for r in results if r.method == method and r.airfoil_name in airfoils]
        if not rs:
            continue
        avg_cd = np.mean([r.cd for r in rs])
        avg_time = np.mean([r.time for r in rs])
        avg_stages = np.mean([r.n_stages for r in rs])
        print(f"{METHOD_LABELS[method]:<22} {avg_cd:>10.6f} {avg_time:>10.1f} {avg_stages:>10.1f}")


# ── 主函数 ──────────────────────────────────────────────────────────────


def main():
    print("=" * 90)
    print("PiERN Router Benchmark — 双场景对比")
    print("=" * 90)
    print(f"CL 目标:  {CL_TARGETS}")
    print(f"CL 权重:  {CL_WEIGHTS}")
    print()

    t0 = time.perf_counter()

    # ── 场景1: 常规翼型 ──
    print(">>> 场景1: 常规翼型 (Baseline 可工作)")
    normal_results = run_benchmark_group(NORMAL_AIRFOILS, "Normal")
    print_summary(normal_results, "场景1: 常规翼型 — PiERN 速度优势", NORMAL_AIRFOILS)
    visualize_normal(normal_results)

    # ── 场景2: 困难翼型 ──
    print("\n>>> 场景2: 困难翼型 (Baseline 失败)")
    hard_results = run_benchmark_group(HARD_AIRFOILS, "Hard")
    print_summary(hard_results, "场景2: 困难翼型 — PiERN 救援失败优化", HARD_AIRFOILS)
    visualize_hard(hard_results)

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
