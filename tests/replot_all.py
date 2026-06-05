"""从 CSV 数据重新绘制所有 benchmark 图表 (无需重新运行 benchmark)。

读取:
  results/benchmark_stats.csv
  results/ablation.csv
  results/pipeline_benchmark.csv
  data/benchmark_airfoils.json

输出: results/*.png (22 张图)
  - Router: 14 张 (场景图、汇总、XFoil分析、分布图、难度-改善图、案例分析)
  - Pipeline: 5 张 (3 类别 + 1 汇总)
  - Ablation: 5 张 (4 实验 + 1 灵敏度分析)
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 字体 ──
_SERIF = "Liberation Serif"
try:
    fm.findfont(_SERIF, fallback_to_default=False)
except Exception:
    _SERIF = "serif"

# ── 配色 ──
_PALETTE = {
    "baseline": "#3370AC",
    "rule": "#D44B3F",
    "threshold": "#E8A838",
    "mlp": "#2A8C6A",
    "xfoil_de": "#7B3294",
}
LABELS = {
    "baseline": "Baseline (8w IPOPT)",
    "rule": "Rule",
    "threshold": "Threshold",
    "mlp": "PiERN Router",
    "xfoil_de": "XFoil+DE",
}
METHODS = ["baseline", "rule", "threshold", "mlp"]


# ── 样式 ──
class S:
    FIG_W = 7.0
    ROW_H = 2.6
    TITLE = 10
    AXIS = 8.5
    TICK = 7.5
    ANNOT = 6.5
    BAR_W = 0.17


def _style(ax, xlabel="", ylabel="", title=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.tick_params(direction="in", top=False, right=False, labelsize=S.TICK, pad=3)
    ax.set_xlabel(xlabel, fontsize=S.AXIS, fontfamily=_SERIF, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=S.AXIS, fontfamily=_SERIF, labelpad=4)
    if title:
        ax.set_title(title, fontsize=S.TITLE, fontfamily=_SERIF, pad=6)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)


def _grouped_bars(ax, x, data, methods, w):
    for i, m in enumerate(methods):
        vals = data.get(m, [])
        if not vals:
            continue
        offset = (i - (len(methods) - 1) / 2) * w
        ax.bar(x + offset, vals, w, color=_PALETTE.get(m, "#888"),
               edgecolor="white", linewidth=0.3, label=LABELS.get(m, m))


def _numbered_labels(airfoils):
    labels = [str(i + 1) for i in range(len(airfoils))]
    lines = []
    row = []
    for i, af in enumerate(airfoils):
        row.append(f"{i + 1}: {af.upper()[:14]}")
        if len(row) == 5:
            lines.append("  ".join(row))
            row = []
    if row:
        lines.append("  ".join(row))
    return labels, "Airfoil Index:\n" + "\n".join(lines)


# ── 数据加载 ──
def load_stats(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "airfoil": r["airfoil"],
                "method": r["method"],
                "cd_mean": float(r["cd_mean"]),
                "cd_std": float(r["cd_std"]),
                "time_mean": float(r["time_mean"]),
                "time_std": float(r["time_std"]),
                "n_stages_mean": float(r["n_stages_mean"]),
                "success_rate": float(r["success_rate"]),
            })
    return rows


def load_airfoils(path):
    with open(path) as f:
        d = json.load(f)
    return d["normal"], d["medium"], d["hard"]


def get(rows, method, airfoil):
    for r in rows:
        if r["method"] == method and r["airfoil"] == airfoil:
            return r
    return None


# ── 场景图 (2x2) ──
def plot_scenario(rows, airfoils, name, save_path):
    n = len(airfoils)
    x = np.arange(n)
    labels, legend_text = _numbered_labels(airfoils)
    w = S.BAR_W * 0.9

    fig, axes = plt.subplots(2, 2, figsize=(S.FIG_W, S.ROW_H * 2),
                             gridspec_kw=dict(hspace=0.55, wspace=0.38))

    # (a) CD
    ax = axes[0, 0]
    cd = {m: [] for m in METHODS}
    for af in airfoils:
        for m in METHODS:
            s = get(rows, m, af)
            cd[m].append(s["cd_mean"] if s and s["cd_mean"] < 1e10 else 0.0)
    _grouped_bars(ax, x, cd, METHODS, w)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.5, ha="center")
    _style(ax, "", "Weighted CD", "(a) Final CD")
    ax.legend(fontsize=S.TICK, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=4, handletextpad=0.4)

    # (b) Time
    ax = axes[0, 1]
    td = {m: [] for m in METHODS}
    for af in airfoils:
        for m in METHODS:
            s = get(rows, m, af)
            td[m].append(s["time_mean"] if s else 0.0)
    _grouped_bars(ax, x, td, METHODS, w)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.5, ha="center")
    _style(ax, "", "Time (s)", "(b) Optimization Time")

    # (c) Success rate
    ax = axes[1, 0]
    sr = []
    for m in METHODS:
        rates = [get(rows, m, af) for af in airfoils]
        sr.append(np.mean([r["success_rate"] for r in rates if r]) * 100)
    bars = ax.bar(range(len(METHODS)), sr, 0.55,
                  color=[_PALETTE[m] for m in METHODS], edgecolor="white", linewidth=0.3)
    for bar, v in zip(bars, sr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom",
                fontsize=S.TICK, fontfamily=_SERIF, fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=S.TICK, fontfamily=_SERIF)
    ax.set_ylim(0, 110)
    _style(ax, "", "Success Rate (%)", "(c) Optimization Success (CD < 0.15)")

    # (d) Improvement
    ax = axes[1, 1]
    imp = {m: [] for m in METHODS}
    for af in airfoils:
        init = get(rows, "initial", af)
        for m in METHODS:
            meth = get(rows, m, af)
            if init and meth and meth["cd_mean"] < 1e10:
                if np.isinf(init["cd_mean"]):
                    # Baseline failed, method succeeded → rescue
                    imp[m].append(100.0)
                elif init["cd_mean"] > 0:
                    imp[m].append((init["cd_mean"] - meth["cd_mean"]) / init["cd_mean"] * 100)
                else:
                    imp[m].append(0.0)
            else:
                imp[m].append(0.0)
    _grouped_bars(ax, x, imp, METHODS, w)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.5, ha="center")
    _style(ax, "", "CD Improvement over Initial (%)", "(d) Optimization Gain")

    fig.suptitle(f"Router Benchmark — {name} Cases ({n} airfoils)",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=0.99)
    fig.text(0.5, -0.06, legend_text, fontsize=5.5, fontfamily=_SERIF,
             ha="center", va="top", transform=fig.transFigure,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8", alpha=0.8))
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── 汇总图 (1x3) ──
def plot_summary(rows, normal_afs, hard_afs, save_path):
    cats = ["Normal", "Hard"]
    cat_afs = [normal_afs, hard_afs]
    x = np.arange(2)
    w = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W, S.ROW_H * 0.85),
                             gridspec_kw=dict(wspace=0.40))

    # (a) CD improvement
    ax = axes[0]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            imps = []
            for af in afs:
                base = get(rows, "baseline", af)
                meth = get(rows, m, af)
                if base and meth and meth["cd_mean"] < 1e10:
                    if np.isinf(base["cd_mean"]):
                        imps.append(-100.0)  # method better than failed baseline
                    elif base["cd_mean"] > 0:
                        imps.append((meth["cd_mean"] - base["cd_mean"]) / base["cd_mean"] * 100)
            vals.append(np.mean(imps) if imps else 0)
        bars = ax.bar(x + (i - 1) * w, vals, w, color=_PALETTE[m],
                      edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=S.TICK, fontfamily=_SERIF)
    _style(ax, "", "Mean $\\Delta$CD vs Baseline (%)", "(a) CD Improvement")
    ax.legend(fontsize=S.TICK, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3)

    # (b) Time speedup
    ax = axes[1]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            speedups = []
            for af in afs:
                base = get(rows, "baseline", af)
                meth = get(rows, m, af)
                if base and meth and meth["time_mean"] > 0:
                    speedups.append(base["time_mean"] / meth["time_mean"])
            vals.append(np.mean(speedups) if speedups else 1.0)
        bars = ax.bar(x + (i - 1) * w, vals, w, color=_PALETTE[m],
                      edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.axhline(y=1.0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=S.TICK, fontfamily=_SERIF)
    _style(ax, "", "Time Ratio (baseline / method)", "(b) Relative Time")

    # (c) Success rate
    ax = axes[2]
    for i, m in enumerate(METHODS):
        vals = []
        for afs in cat_afs:
            rates = [get(rows, m, af) for af in afs]
            vals.append(np.mean([r["success_rate"] for r in rates if r]) * 100)
        bars = ax.bar(x + (i - 1) * w, vals, w, color=_PALETTE[m],
                      edgecolor="white", linewidth=0.3, label=LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=S.TICK, fontfamily=_SERIF)
    ax.set_ylim(0, 115)
    _style(ax, "", "Success Rate (%)", "(c) Success Rate")

    fig.suptitle("Cross-Category Summary", fontsize=11, fontfamily=_SERIF,
                 fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── XFoil 分析图 (1x3) ──
def plot_xfoil(rows, airfoils, save_path):
    """XFoil+DE 优化困难分析 — 仅展示时间对比。

    说明 XFoil+DE 的计算代价远高于 NeuralFoil 方法。
    CD 和成功率因评价标准不同，不在此图展示。
    """
    fig, ax = plt.subplots(1, 1, figsize=(S.FIG_W * 0.6, S.ROW_H))

    all_m = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]
    mt = []
    for m in all_m:
        ok = [r for r in rows if r["method"] == m and r["airfoil"] in airfoils]
        mt.append(np.mean([r["time_mean"] for r in ok]) if ok else 0)

    bars = ax.bar(range(len(all_m)), mt, 0.55,
                  color=[_PALETTE[m] for m in all_m],
                  edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(all_m)))
    ax.set_xticklabels([LABELS[m] for m in all_m], fontsize=7, fontfamily=_SERIF,
                       rotation=20, ha="right")
    _style(ax, "", "Mean Time (s)", "Optimization Time — XFoil+DE vs NeuralFoil Methods")

    # 标注 XFoil 柱子
    xfoil_bar = bars[4]
    ax.annotate(
        f"{mt[4]:.0f}s\n({mt[4]/mt[0]:.0f}× baseline)",
        xy=(xfoil_bar.get_x() + xfoil_bar.get_width() / 2, mt[4]),
        textcoords="offset points", xytext=(10, 0),
        fontsize=8, fontfamily=_SERIF, fontweight="bold", color=_PALETTE["xfoil_de"],
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── 分布图 (1x2) ──
def plot_distribution(rows, airfoils, cat_name, save_path):
    n = len(airfoils)

    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    # (a) Boxplot
    ax = axes[0]
    box_data = []
    for m in METHODS:
        cds = [r["cd_mean"] for r in rows
               if r["method"] == m and r["airfoil"] in airfoils and r["cd_mean"] < 1e10]
        box_data.append(cds)

    bp = ax.boxplot(box_data, positions=range(len(METHODS)), widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(_PALETTE[m])
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    for i, m in enumerate(METHODS):
        cds = box_data[i]
        if cds:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(cds))
            ax.scatter(np.full_like(cds, i, dtype=float) + jitter, cds,
                       s=12, alpha=0.4, color=_PALETTE[m], edgecolors="none", zorder=3)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=S.TICK,
                       fontfamily=_SERIF, rotation=20, ha="right")
    _style(ax, "", "Weighted CD", f"(a) CD Distribution — {cat_name}")

    # (b) Improvement strip
    ax = axes[1]
    for i, m in enumerate(["rule", "threshold", "mlp"]):
        imps = []
        for af in airfoils:
            base = get(rows, "baseline", af)
            meth = get(rows, m, af)
            if base and meth and meth["cd_mean"] < 1e10:
                if np.isinf(base["cd_mean"]):
                    imps.append(100.0)  # rescue
                elif base["cd_mean"] > 0:
                    imps.append((base["cd_mean"] - meth["cd_mean"]) / base["cd_mean"] * 100)
        if imps:
            jitter = np.random.default_rng(42 + i).uniform(-0.15, 0.15, len(imps))
            ax.scatter(np.full_like(imps, i, dtype=float) + jitter, imps,
                       s=18, alpha=0.5, color=_PALETTE[m], edgecolors="none", zorder=3)
            mean_imp = np.mean(imps)
            ax.scatter([i], [mean_imp], s=80, color=_PALETTE[m],
                       edgecolors="black", linewidths=0.8, zorder=5, marker="D")
            ax.annotate(f"{mean_imp:+.1f}%", (i, mean_imp),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=S.ANNOT, fontfamily=_SERIF,
                        color=_PALETTE[m], fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[m] for m in ["rule", "threshold", "mlp"]],
                       fontsize=S.TICK, fontfamily=_SERIF, rotation=20, ha="right")
    _style(ax, "", "CD Improvement over Baseline (%)", f"(b) Per-Airfoil Improvement — {cat_name}")

    fig.suptitle(f"CD Distribution — {cat_name} ({n} airfoils)",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── 难度-改善散点图 (1x2) ──
def plot_difficulty(rows, airfoils, cat_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    # (a) Scatter
    ax = axes[0]
    for i, m in enumerate(["rule", "threshold", "mlp"]):
        xv, yv = [], []
        for af in airfoils:
            init = get(rows, "initial", af)
            meth = get(rows, m, af)
            if init and meth and meth["cd_mean"] < 1e10:
                if np.isinf(init["cd_mean"]):
                    xv.append(0.3)  # rescue case, use arbitrary x
                    yv.append(100.0)
                elif init["cd_mean"] > 0:
                    xv.append(init["cd_mean"])
                    yv.append((init["cd_mean"] - meth["cd_mean"]) / init["cd_mean"] * 100)
        if xv:
            ax.scatter(xv, yv, s=25, alpha=0.5, color=_PALETTE[m],
                       label=LABELS[m], edgecolors="none", zorder=3)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xscale("log")
    _style(ax, "Initial CD (airfoil difficulty)", "CD Improvement over Baseline (%)",
           f"(a) Difficulty vs Improvement — {cat_name}")
    ax.legend(fontsize=6, frameon=False)

    # (b) Binned
    ax = axes[1]
    init_cds = []
    for af in airfoils:
        s = get(rows, "initial", af)
        if s and s["cd_mean"] > 0:
            init_cds.append((af, s["cd_mean"]))
    if init_cds:
        init_cds.sort(key=lambda x: x[1])
        n = len(init_cds)
        qs = [init_cds[:n//4], init_cds[n//4:n//2], init_cds[n//2:3*n//4], init_cds[3*n//4:]]
        q_labels = ["Q1\n(easiest)", "Q2", "Q3", "Q4\n(hardest)"]
        x = np.arange(4)
        bw = 0.18
        for i, m in enumerate(["rule", "threshold", "mlp"]):
            vals = []
            for q_afs in qs:
                imps = []
                for af, _ in q_afs:
                    init = get(rows, "initial", af)
                    meth = get(rows, m, af)
                    if init and meth and meth["cd_mean"] < 1e10:
                        if np.isinf(init["cd_mean"]):
                            imps.append(100.0)
                        elif init["cd_mean"] > 0:
                            imps.append((init["cd_mean"] - meth["cd_mean"]) / init["cd_mean"] * 100)
                vals.append(np.mean(imps) if imps else 0)
            bars = ax.bar(x + (i - 1) * bw, vals, bw, color=_PALETTE[m],
                          edgecolor="white", linewidth=0.3, label=LABELS[m])
        ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=S.TICK, fontfamily=_SERIF)
        _style(ax, "Difficulty Quartile", "Mean CD Improvement (%)",
               f"(b) Improvement by Difficulty — {cat_name}")
        ax.legend(fontsize=6, frameon=False)

    fig.suptitle(f"Difficulty vs Improvement — {cat_name} ({len(airfoils)} airfoils)",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── NACA 0012 案例分析图 ──
def plot_case_study(rows, airfoil_name, save_path):
    """NACA 0012 案例分析: 展示各方法优化后的翼型形状、CD 值和时间。

    Layout (1x3):
      (a) 各方法优化后的翼型轮廓叠加对比
      (b) CD 值对比柱状图
      (c) 优化时间对比柱状图
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    import aerosandbox as asb
    from piern_airfoil.eval import evaluate_weighted_cd

    methods = ["baseline", "rule", "threshold", "mlp", "xfoil_de"]

    # CL/Re 参数 (与 benchmark 一致)
    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
    MACH = 0.03

    # 运行各方法获取优化后的翼型
    af_init = asb.KulfanAirfoil(airfoil_name)
    initial_cd = evaluate_weighted_cd(af_init, CL_TARGETS, RE, CL_WEIGHTS, MACH)

    optimized_airfoils = {}
    cds = []
    times = []

    for method in methods:
        print(f"  Case study: {method}...", end=" ", flush=True)
        t0 = time.time()
        try:
            if method == "baseline":
                from piern_airfoil.optimizer import NeuralOptimizer
                opt = NeuralOptimizer(
                    airfoil=af_init, CL_targets=CL_TARGETS, CL_weights=CL_WEIGHTS,
                    RE=RE, mach=MACH,
                )
                opt.update()
                elapsed = time.time() - t0
                cd = evaluate_weighted_cd(opt.airfoil, CL_TARGETS, RE, CL_WEIGHTS, MACH)
                optimized_airfoils[method] = opt.airfoil
                cds.append(cd)
                times.append(elapsed)
            elif method == "xfoil_de":
                from piern_airfoil.xfoil_optimizer import xfoil_optimize
                result = xfoil_optimize(
                    airfoil_name, CL_TARGETS, RE, CL_WEIGHTS, MACH,
                    maxiter=5, popsize=3,
                )
                elapsed = time.time() - t0
                optimized_airfoils[method] = result.optimized_airfoil
                cds.append(result.final_cd)
                times.append(elapsed)
            else:
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
                result = optimizer.optimize(af_init)
                elapsed = time.time() - t0
                optimized_airfoils[method] = result.airfoil
                cds.append(result.final_cd)
                times.append(elapsed)

            print(f"CD={cds[-1]:.4f} {times[-1]:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED: {e}")
            optimized_airfoils[method] = af_init
            cds.append(float("inf"))
            times.append(elapsed)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W * 1.2, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    # (a) 翼型轮廓叠加
    ax = axes[0]
    coords_init = af_init.coordinates
    ax.plot(coords_init[:, 0], coords_init[:, 1],
            color="#888888", linewidth=1.0, linestyle="--", label="Initial", alpha=0.6)
    for m in methods:
        af = optimized_airfoils[m]
        coords = af.coordinates
        ax.plot(coords[:, 0], coords[:, 1],
                color=_PALETTE[m], linewidth=1.2, label=LABELS[m], alpha=0.8)
    ax.set_aspect("equal")
    _style(ax, "x/c", "y/c", "(a) Optimized Airfoil Shapes")
    ax.legend(fontsize=6, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.15))

    # (b) CD 对比
    ax = axes[1]
    cds_plot = [initial_cd] + cds
    colors_plot = ["#888888"] + [_PALETTE[m] for m in methods]
    labels_plot = ["Initial"] + [LABELS[m] for m in methods]
    bars = ax.bar(range(len(cds_plot)), cds_plot, 0.6,
                  color=colors_plot, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(labels_plot)))
    ax.set_xticklabels(labels_plot, fontsize=6.5, fontfamily=_SERIF, rotation=20, ha="right")
    _style(ax, "", "Weighted CD", "(b) Drag Coefficient")

    # (c) 时间对比
    ax = axes[2]
    times_plot = [0.0] + times
    colors_times = ["#888888"] + [_PALETTE[m] for m in methods]
    labels_times = ["Initial"] + [LABELS[m] for m in methods]
    bars = ax.bar(range(len(times_plot)), times_plot, 0.6,
                  color=colors_times, edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(labels_times)))
    ax.set_xticklabels(labels_times, fontsize=6.5, fontfamily=_SERIF, rotation=20, ha="right")
    _style(ax, "", "Time (s)", "(c) Optimization Time")

    fig.suptitle(f"Case Study — {airfoil_name.upper()}",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Pipeline Benchmark 重绘 ──
def load_pipeline_csv(path):
    """加载 pipeline_benchmark.csv。"""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "airfoil": r["airfoil"],
                "pipeline_type": r["pipeline_type"],
                "cd": float(r["cd"]),
                "cd_gap": float(r["cd_gap"]),
                "time": float(r["time"]),
                "extraction_time": float(r["extraction_time"]),
                "optimization_time": float(r["optimization_time"]),
                "kulfan_fit_error": float(r["kulfan_fit_error"]),
                "success": r["success"] == "True",
            })
    return rows


def plot_pipeline_category(rows, airfoils, category, save_path):
    """Pipeline 按类别可视化 (1x3)。"""
    pipeline_types = ["ground_truth", "image"]
    pipeline_colors = {"ground_truth": "#2CA02C", "image": "#E45756"}
    pipeline_labels = {"ground_truth": "Ground Truth", "image": "Image Pipeline"}

    cat_rows = [r for r in rows if r["airfoil"] in airfoils]
    if not cat_rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W * 1.2, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))
    x = np.arange(len(airfoils))
    n_p = len(pipeline_types)
    width = 0.7 / n_p

    # (a) CD 对比
    ax = axes[0]
    for i, pt in enumerate(pipeline_types):
        cds = []
        for af in airfoils:
            rs = [r for r in cat_rows if r["airfoil"] == af and r["pipeline_type"] == pt]
            cds.append(rs[0]["cd"] if rs and rs[0]["success"] else float("nan"))
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, cds, width, label=pipeline_labels[pt], color=pipeline_colors[pt], alpha=0.85)
    _style(ax, "", "Weighted CD", f"(a) CD — {category}")
    ax.legend(fontsize=6, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=5, rotation=45, ha="right")

    # (b) CD Gap
    ax = axes[1]
    gaps = []
    for af in airfoils:
        rs = [r for r in cat_rows if r["airfoil"] == af and r["pipeline_type"] == "image"]
        gaps.append(rs[0]["cd_gap"] if rs and rs[0]["success"] else float("nan"))
    ax.bar(x, gaps, 0.5, label="image", color=pipeline_colors["image"], alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    _style(ax, "", "CD Gap vs Ground Truth", "(b) Extraction Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=5, rotation=45, ha="right")

    # (c) 时间
    ax = axes[2]
    for i, pt in enumerate(pipeline_types):
        times = []
        for af in airfoils:
            rs = [r for r in cat_rows if r["airfoil"] == af and r["pipeline_type"] == pt]
            times.append(rs[0]["time"] if rs and rs[0]["success"] else 0)
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, times, width, label=pipeline_labels[pt], color=pipeline_colors[pt], alpha=0.85)
    _style(ax, "", "Time (s)", "(c) End-to-End Time")
    ax.legend(fontsize=6, frameon=False)
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=5, rotation=45, ha="right")

    fig.suptitle(f"Pipeline Benchmark — {category} ({len(airfoils)} airfoils)",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pipeline_summary(rows, normal_afs, medium_afs, hard_afs, save_path):
    """Pipeline 汇总图 (1x3)。"""
    categories = ["Normal", "Medium", "Hard"]
    cat_airfoils = [normal_afs, medium_afs, hard_afs]
    pipeline_types = ["ground_truth", "image"]
    pipeline_colors = {"ground_truth": "#2CA02C", "image": "#E45756"}
    pipeline_labels = {"ground_truth": "Ground Truth", "image": "Image Pipeline"}

    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.40))
    x = np.arange(len(categories))
    w = 0.35

    # (a) 成功率
    ax = axes[0]
    for i, pt in enumerate(pipeline_types):
        rates = []
        for afs in cat_airfoils:
            total = sum(1 for r in rows if r["airfoil"] in afs and r["pipeline_type"] == pt)
            success = sum(1 for r in rows if r["airfoil"] in afs and r["pipeline_type"] == pt and r["success"])
            rates.append(success / total * 100 if total > 0 else 0)
        bars = ax.bar(x + (i - 0.5) * w, rates, w, color=pipeline_colors[pt],
                      edgecolor="white", linewidth=0.3, label=pipeline_labels[pt])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=S.TICK, fontfamily=_SERIF)
    ax.set_ylim(0, 115)
    _style(ax, "", "Success Rate (%)", "(a) Pipeline Success Rate")
    ax.legend(fontsize=S.TICK, frameon=False)

    # (b) 平均 CD Gap
    ax = axes[1]
    gaps = []
    for afs in cat_airfoils:
        rs = [r for r in rows if r["airfoil"] in afs and r["pipeline_type"] == "image" and r["success"]]
        gaps.append(np.mean([r["cd_gap"] for r in rs]) if rs else 0)
    bars = ax.bar(x, gaps, 0.5, color=pipeline_colors["image"],
                  edgecolor="white", linewidth=0.3)
    ax.axhline(y=0, color="black", linewidth=0.6, alpha=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=S.TICK, fontfamily=_SERIF)
    _style(ax, "", "Mean CD Gap", "(b) Extraction Accuracy")

    # (c) 平均时间
    ax = axes[2]
    for i, pt in enumerate(pipeline_types):
        times = []
        for afs in cat_airfoils:
            rs = [r for r in rows if r["airfoil"] in afs and r["pipeline_type"] == pt and r["success"]]
            times.append(np.mean([r["time"] for r in rs]) if rs else 0)
        bars = ax.bar(x + (i - 0.5) * w, times, w, color=pipeline_colors[pt],
                      edgecolor="white", linewidth=0.3, label=pipeline_labels[pt])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=S.TICK, fontfamily=_SERIF)
    _style(ax, "", "Time (s)", "(c) End-to-End Time")

    fig.suptitle("Pipeline Summary — Ground Truth vs Image Pipeline",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Ablation Benchmark 重绘 ──
def load_ablation_csv(path):
    """加载 ablation.csv。"""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            stage_cds = [float(x) for x in r["stage_cds"].split(";") if x] if r["stage_cds"] else []
            rows.append({
                "ablation": r["ablation"],
                "method": r["method"],
                "airfoil": r["airfoil"],
                "cd_initial": float(r["cd_initial"]),
                "cd_final": float(r["cd_final"]),
                "time_s": float(r["time_s"]),
                "n_stages": int(r["n_stages"]),
                "stage_cds": stage_cds,
                "success": r["success"] == "1",
            })
    return rows


def plot_ablation_1(rows, airfoils, save_path):
    """Ablation 1: Hierarchical vs Direct (1x3)。"""
    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W * 1.4, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    methods = ["direct", "hier_rule_sw4"]
    labels = ["Direct (8w)", "Hierarchical (4→8)"]
    colors = ["#3370AC", "#D44B3F"]

    # 按初始CD排序
    sorted_airfoils = sorted(
        airfoils,
        key=lambda n: next((r["cd_initial"] for r in rows if r["airfoil"] == n), 0),
    )

    # (a) CD 对比
    ax = axes[0]
    for method, label, color in zip(methods, labels, colors):
        cds = []
        for name in sorted_airfoils:
            run = next((r for r in rows if r["method"] == method and r["airfoil"] == name), None)
            cds.append(run["cd_final"] if run and run["success"] else np.nan)
        xs = np.arange(len(sorted_airfoils))
        ax.scatter(xs, cds, c=color, label=label, alpha=0.6, s=15, edgecolors="none")
    _style(ax, "Airfoil Index", "Final CD", "(a) Final CD")
    ax.legend(fontsize=6, frameon=False)

    # (b) 时间分布
    ax = axes[1]
    data_times = []
    for method in methods:
        times = [r["time_s"] for r in rows if r["method"] == method and r["success"]]
        data_times.append(times)
    bp = ax.boxplot(data_times, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Time (s)", "(b) Runtime Distribution")

    # (c) CD 改善
    ax = axes[2]
    improvements = {}
    for method, label, color in zip(methods, labels, colors):
        imps = [
            (r["cd_initial"] - r["cd_final"]) / r["cd_initial"] * 100
            for r in rows
            if r["method"] == method and r["success"] and r["cd_initial"] > 0
        ]
        improvements[label] = imps
    bp = ax.boxplot([improvements[l] for l in labels], tick_labels=labels, patch_artist=True,
                    widths=0.5, showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "CD Improvement (%)", "(c) Optimization Gain")

    fig.suptitle("A1: Hierarchical CST vs Direct Optimization",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_2(rows, airfoils, save_path):
    """Ablation 2: Router Strategy Effect (2x2)。"""
    fig, axes = plt.subplots(2, 2, figsize=(S.FIG_W, S.ROW_H * 2),
                             gridspec_kw=dict(hspace=0.45, wspace=0.38))

    modes = ["rule", "threshold", "mlp"]
    labels = ["Rule (fixed)", "Threshold (learned)", "MLP (learned)"]
    colors = ["#D44B3F", "#E8A838", "#2A8C6A"]

    sorted_airfoils = sorted(
        airfoils,
        key=lambda n: next((r["cd_initial"] for r in rows if r["airfoil"] == n), 0),
    )

    # (a) CD per airfoil
    ax = axes[0, 0]
    for mode, label, color in zip(modes, labels, colors):
        cds = []
        for name in sorted_airfoils:
            run = next((r for r in rows if r["method"] == mode and r["airfoil"] == name), None)
            cds.append(run["cd_final"] if run and run["success"] else np.nan)
        xs = np.arange(len(sorted_airfoils))
        ax.scatter(xs, cds, c=color, label=label, alpha=0.6, s=15, edgecolors="none")
    _style(ax, "Airfoil Index", "Final CD", "(a) Final CD")
    ax.legend(fontsize=6, frameon=False)

    # (b) Time distribution
    ax = axes[0, 1]
    data_times = []
    for mode in modes:
        times = [r["time_s"] for r in rows if r["method"] == mode and r["success"]]
        data_times.append(times)
    bp = ax.boxplot(data_times, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Time (s)", "(b) Runtime Distribution")

    # (c) Stages distribution
    ax = axes[1, 0]
    data_stages = []
    for mode in modes:
        stages = [r["n_stages"] for r in rows if r["method"] == mode and r["success"]]
        data_stages.append(stages)
    bp = ax.boxplot(data_stages, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Number of Stages", "(c) Stage Count")

    # (d) Time vs CD scatter
    ax = axes[1, 1]
    for mode, label, color in zip(modes, labels, colors):
        rs = [r for r in rows if r["method"] == mode and r["success"]]
        if rs:
            times = [r["time_s"] for r in rs]
            cds = [r["cd_final"] for r in rs]
            ax.scatter(times, cds, s=15, alpha=0.5, color=color, label=label, edgecolors="none")
    _style(ax, "Time (s)", "Final CD", "(d) Time vs CD")
    ax.legend(fontsize=6, frameon=False)

    fig.suptitle("A2: Router Strategy Effect",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_3(rows, airfoils, save_path):
    """Ablation 3: Starting Dimension (1x3)。"""
    fig, axes = plt.subplots(1, 3, figsize=(S.FIG_W * 1.4, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    methods = ["sw4", "sw6", "sw8"]
    labels = ["Start 4w", "Start 6w", "Start 8w"]
    colors = ["#2A8C6A", "#E8A838", "#D44B3F"]

    # (a) Time distribution
    ax = axes[0]
    data_times = []
    for method in methods:
        times = [r["time_s"] for r in rows if r["method"] == method and r["success"]]
        data_times.append(times)
    bp = ax.boxplot(data_times, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Time (s)", "(a) Runtime Distribution")

    # (b) Final CD
    ax = axes[1]
    data_cds = []
    for method in methods:
        cds = [r["cd_final"] for r in rows if r["method"] == method and r["success"]]
        data_cds.append(cds)
    bp = ax.boxplot(data_cds, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Final CD", "(b) Final CD")

    # (c) Stages
    ax = axes[2]
    data_stages = []
    for method in methods:
        stages = [r["n_stages"] for r in rows if r["method"] == method and r["success"]]
        data_stages.append(stages)
    bp = ax.boxplot(data_stages, tick_labels=labels, patch_artist=True, widths=0.5,
                    showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.4),
                    medianprops=dict(color="black", linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    _style(ax, "", "Number of Stages", "(c) Stage Count")

    fig.suptitle("A3: Starting CST Dimension",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_4(rows, airfoils, save_path):
    """Ablation 4: Stage Contribution (1x2)。"""
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    a4_rows = [r for r in rows if r["ablation"] == "A4" and r["success"]]

    # (a) 各阶段 CD 变化
    ax = axes[0]
    if a4_rows:
        max_stages = max(r["n_stages"] for r in a4_rows)
        stage_means = []
        stage_stds = []
        for s in range(max_stages):
            cds_at_stage = [r["stage_cds"][s] for r in a4_rows if s < len(r["stage_cds"])]
            stage_means.append(np.mean(cds_at_stage) if cds_at_stage else np.nan)
            stage_stds.append(np.std(cds_at_stage) if cds_at_stage else 0)
        ax.errorbar(range(1, len(stage_means) + 1), stage_means, yerr=stage_stds,
                    fmt="o-", color="#2A8C6A", linewidth=1.5, markersize=5, alpha=0.8, capsize=3)
    _style(ax, "Stage", "Mean CD", "(a) CD by Stage")

    # (b) 各阶段改善率
    ax = axes[1]
    if a4_rows:
        max_stages = max(r["n_stages"] for r in a4_rows)
        stage_improvements = []
        for s in range(1, max_stages):
            imps = []
            for r in a4_rows:
                if s < len(r["stage_cds"]) and s - 1 < len(r["stage_cds"]):
                    prev_cd = r["stage_cds"][s - 1]
                    curr_cd = r["stage_cds"][s]
                    if prev_cd > 0:
                        imps.append((prev_cd - curr_cd) / prev_cd * 100)
            stage_improvements.append(np.mean(imps) if imps else 0)
        ax.bar(range(2, len(stage_improvements) + 2), stage_improvements, 0.6,
               color="#2A8C6A", edgecolor="white", linewidth=0.3)
    _style(ax, "Stage Transition", "Mean CD Improvement (%)", "(b) Improvement by Stage")

    fig.suptitle("A4: Per-Stage CST Dimension Contribution",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sensitivity(rows, save_path):
    """Sensitivity Analysis (1x2)。"""
    fig, axes = plt.subplots(1, 2, figsize=(S.FIG_W, S.ROW_H * 0.9),
                             gridspec_kw=dict(wspace=0.35))

    sens_rows = [r for r in rows if r["ablation"] == "sensitivity"]

    # (a) Threshold sensitivity
    ax = axes[0]
    threshold_rows = [r for r in sens_rows if "threshold" in r["method"]]
    if threshold_rows:
        thresholds = sorted(set(r["method"] for r in threshold_rows))
        means = []
        for t in thresholds:
            cds = [r["cd_final"] for r in threshold_rows if r["method"] == t and r["success"]]
            means.append(np.mean(cds) if cds else np.nan)
        ax.plot(range(len(thresholds)), means, "o-", color="#E8A838", linewidth=1.5, markersize=5)
    _style(ax, "Threshold Index", "Mean Final CD", "(a) Threshold Sensitivity")

    # (b) Start weight sensitivity
    ax = axes[1]
    sw_rows = [r for r in sens_rows if "sw" in r["method"]]
    if sw_rows:
        sws = sorted(set(r["method"] for r in sw_rows))
        means = []
        for sw in sws:
            cds = [r["cd_final"] for r in sw_rows if r["method"] == sw and r["success"]]
            means.append(np.mean(cds) if cds else np.nan)
        ax.plot(range(len(sws)), means, "o-", color="#2A8C6A", linewidth=1.5, markersize=5)
    _style(ax, "Start Weight Index", "Mean Final CD", "(b) Start Weight Sensitivity")

    fig.suptitle("Sensitivity Analysis",
                 fontsize=11, fontfamily=_SERIF, fontweight="bold", y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ── 主函数 ──
def main():
    base = Path(__file__).resolve().parent.parent
    results = base / "results"
    data = base / "data"

    rows = load_stats(results / "benchmark_stats.csv")
    normal, medium, hard = load_airfoils(data / "benchmark_airfoils.json")
    all_afs = normal + medium + hard

    print("Replotting all figures from CSV data...\n")

    # 场景图
    print("Scenario figures:")
    plot_scenario(rows, normal, "Normal", str(results / "benchmark_normal.png"))
    plot_scenario(rows, medium, "Medium", str(results / "benchmark_medium.png"))
    plot_scenario(rows, hard, "Hard", str(results / "benchmark_hard.png"))

    # 汇总
    print("\nSummary figure:")
    plot_summary(rows, normal, hard, str(results / "benchmark_summary.png"))

    # XFoil 分析
    print("\nXFoil analysis:")
    plot_xfoil(rows, all_afs, str(results / "benchmark_xfoil_analysis.png"))

    # 分布图
    print("\nDistribution figures:")
    for name, afs in [("Normal", normal), ("Medium", medium), ("Hard", hard), ("All", all_afs)]:
        plot_distribution(rows, afs, name, str(results / f"benchmark_dist_{name.lower()}.png"))

    # 难度-改善散点图
    print("\nDifficulty-improvement figures:")
    for name, afs in [("Normal", normal), ("Medium", medium), ("Hard", hard), ("All", all_afs)]:
        plot_difficulty(rows, afs, name, str(results / f"benchmark_diff_{name.lower()}.png"))

    # NACA 0012 案例分析
    print("\nCase study (NACA 0012):")
    plot_case_study(rows, "naca0012", str(results / "benchmark_case_study.png"))

    # Pipeline Benchmark
    print("\nPipeline benchmark:")
    pipeline_rows = load_pipeline_csv(results / "pipeline_benchmark.csv")
    for name, afs in [("Normal", normal), ("Medium", medium), ("Hard", hard)]:
        plot_pipeline_category(pipeline_rows, afs, name, str(results / f"pipeline_{name.lower()}.png"))
    plot_pipeline_summary(pipeline_rows, normal, medium, hard, str(results / "pipeline_summary.png"))

    # Ablation Benchmark
    print("\nAblation benchmark:")
    ablation_rows = load_ablation_csv(results / "ablation.csv")
    plot_ablation_1(ablation_rows, all_afs, str(results / "ablation_1_hierarchical_vs_direct.png"))
    plot_ablation_2(ablation_rows, all_afs, str(results / "ablation_2_router_effect.png"))
    plot_ablation_3(ablation_rows, all_afs, str(results / "ablation_3_starting_dimension.png"))
    plot_ablation_4(ablation_rows, all_afs, str(results / "ablation_4_dimension_contribution.png"))
    plot_sensitivity(ablation_rows, str(results / "sensitivity.png"))

    n_figs = 3 + 1 + 1 + 4 + 4 + 1 + 4 + 5
    print(f"\nDone! Generated {n_figs} figures in {results}/")


if __name__ == "__main__":
    main()
