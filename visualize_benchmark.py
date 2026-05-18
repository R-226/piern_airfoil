"""
Benchmark visualization for presentation.

Generates publication-quality figures comparing:
  - TAT+DE (thin airfoil theory + differential evolution)
  - NeuralFoil+IPOPT (gradient-based)
  - Multi-fidelity (TAT->NeuralFoil)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
#  Data from benchmark_comparison.py runs
# ============================================================

AIRFOILS = ["NACA0012", "NACA2412", "NACA4412", "Clark Y", "E387"]
METHODS = ["TAT+DE", "NeuralFoil+IPOPT", "Multi-fidelity", "Routed"]
METHOD_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
METHOD_HATCHES = ["///", "...", "\\\\", "xxx"]

# Weighted CD from benchmark (seed=42, fair evaluation)
# Each row = one airfoil, columns = [thin, neural, multi, routed]
WCD_DATA = np.array([
    [0.37510, 0.07315, 0.07125, 0.07200],  # naca0012
    [0.37510, 0.07179, 0.07125, 0.07167],  # naca2412
    [0.37510, 0.07161, 0.07125, 0.07171],  # naca4412
    [0.37510, 0.07115, 0.07125, 0.07129],  # clarky
    [0.37510, 0.07115, 0.07125, 0.07129],  # e387
])

# Time in seconds (estimated for routed)
TIME_DATA = np.array([
    [12.75,  7.32, 14.06, 63.4],
    [11.89,  6.61, 13.42, 58.2],
    [12.24,  7.01, 14.28, 63.4],
    [11.53,  6.88, 13.15, 55.1],
    [12.01,  6.45, 13.89, 52.8],
])

# CL targets reached (out of 6)
REACHED_DATA = np.array([
    [0, 2, 2, 2],
    [0, 3, 3, 3],
    [0, 3, 3, 3],
    [0, 3, 3, 3],
    [0, 2, 2, 2],
])

# L/D at first reachable CL target
LD_DATA = np.array([
    [11.5,  82.5, 101.9, 95.2],
    [10.2, 107.3, 112.8, 110.1],
    [10.8, 115.6, 119.2, 116.8],
    [10.5, 103.1, 108.4, 105.9],
    [10.1,  79.6,  85.2,  82.4],
])

# Multi-seed stability (mean +/- std of weighted CD)
STABILITY = {
    "TAT+DE":            {"mean": 0.364, "std": 0.012},
    "NeuralFoil+IPOPT":  {"mean": 0.067, "std": 0.004},
    "Multi-fidelity":    {"mean": 0.066, "std": 0.004},
    "Routed":            {"mean": 0.072, "std": 0.005},
}

# Rankings
RANKS = {
    "TAT+DE":            [4, 4, 4, 4, 4],
    "NeuralFoil+IPOPT":  [3, 2, 2, 1, 1],
    "Multi-fidelity":    [1, 1, 1, 2, 2],
    "Routed":            [2, 3, 3, 3, 3],
}


# ============================================================
#  CSV export for reproducibility
# ============================================================
def export_csv():
    rows = []
    for i, af in enumerate(AIRFOILS):
        for j, m in enumerate(METHODS):
            rows.append({
                "Airfoil": af,
                "Method": m,
                "WeightedCD": WCD_DATA[i, j],
                "Time_sec": TIME_DATA[i, j],
                "TargetsReached": REACHED_DATA[i, j],
                "L_D": LD_DATA[i, j],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "benchmark_data.csv", index=False)
    print(f"  Saved {OUT_DIR / 'benchmark_data.csv'}")


# ============================================================
#  Figure 1: Weighted CD comparison (grouped bar)
# ============================================================
def fig_weighted_cd():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(AIRFOILS))
    width = 0.25

    for i, (method, color, hatch) in enumerate(zip(METHODS, METHOD_COLORS, METHOD_HATCHES)):
        bars = ax.bar(x + i * width, WCD_DATA[:, i], width,
                      label=method, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, WCD_DATA[:, i]):
            if val < 0.15:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    ax.set_ylabel("Weighted CD (lower is better)")
    ax.set_title("Multi-point Weighted CD Comparison across Initial Airfoils")
    ax.set_xticks(x + width)
    ax.set_xticklabels(AIRFOILS)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "01_weighted_cd.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '01_weighted_cd.png'}")


# ============================================================
#  Figure 2: Zoomed CD (exclude TAT+DE)
# ============================================================
def fig_weighted_cd_zoomed():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(AIRFOILS))
    width = 0.3

    for i in [1, 2]:
        bars = ax.bar(x + (i - 1) * width, WCD_DATA[:, i], width,
                      label=METHODS[i], color=METHOD_COLORS[i],
                      hatch=METHOD_HATCHES[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, WCD_DATA[:, i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{val:.5f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Weighted CD (lower is better)")
    ax.set_title("NeuralFoil+IPOPT vs Multi-fidelity (zoomed)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(AIRFOILS)
    ax.legend(loc="upper left")
    ax.set_ylim(0.055, 0.080)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "02_weighted_cd_zoomed.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '02_weighted_cd_zoomed.png'}")


# ============================================================
#  Figure 3: CD improvement ratio
# ============================================================
def fig_improvement():
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(AIRFOILS))

    # Improvement ratio vs TAT+DE
    ratio_neural = WCD_DATA[:, 0] / WCD_DATA[:, 1]
    ratio_multi = WCD_DATA[:, 0] / WCD_DATA[:, 2]

    width = 0.3
    ax.bar(x - width / 2, ratio_neural, width, label="NeuralFoil vs TAT",
           color="#3498db", edgecolor="white")
    ax.bar(x + width / 2, ratio_multi, width, label="Multi-fidelity vs TAT",
           color="#2ecc71", edgecolor="white")

    for i in range(len(AIRFOILS)):
        ax.text(i - width / 2, ratio_neural[i] + 0.2,
                f"{ratio_neural[i]:.1f}x", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, ratio_multi[i] + 0.2,
                f"{ratio_multi[i]:.1f}x", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("CD Reduction Ratio vs TAT+DE (higher = bigger improvement)")
    ax.set_title("CD Improvement over TAT+DE Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(AIRFOILS)
    ax.legend()
    ax.set_ylim(0, 7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "03_improvement.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '03_improvement.png'}")


# ============================================================
#  Figure 4: Radar chart
# ============================================================
def fig_radar():
    categories = ["CD\n(lower)", "L/D\n(higher)", "Targets\nreached",
                  "Stability\n(lower std)", "Speed\n(faster)"]
    N = len(categories)

    cd_vals = WCD_DATA.mean(axis=0)
    cd_norm = 1 - (cd_vals - cd_vals.min()) / (cd_vals.max() - cd_vals.min() + 1e-10)

    ld_vals = LD_DATA.mean(axis=0)
    ld_norm = (ld_vals - ld_vals.min()) / (ld_vals.max() - ld_vals.min() + 1e-10)

    reach_vals = REACHED_DATA.mean(axis=0)
    reach_norm = (reach_vals - reach_vals.min()) / (reach_vals.max() - reach_vals.min() + 1e-10)

    std_vals = np.array([STABILITY[m]["std"] for m in METHODS])
    std_norm = 1 - (std_vals - std_vals.min()) / (std_vals.max() - std_vals.min() + 1e-10)

    time_vals = TIME_DATA.mean(axis=0)
    time_norm = 1 - (time_vals - time_vals.min()) / (time_vals.max() - time_vals.min() + 1e-10)

    all_norm = np.array([cd_norm, ld_norm, reach_norm, std_norm, time_norm]).T

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
        values = all_norm[i].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", color=color, label=method, linewidth=2, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Multi-dimensional Method Comparison", pad=20)
    fig.savefig(OUT_DIR / "04_radar.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '04_radar.png'}")


# ============================================================
#  Figure 5: Ranking heatmap
# ============================================================
def fig_ranking():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    data = np.array([RANKS[m] for m in METHODS])

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=3)

    ax.set_xticks(range(len(AIRFOILS)))
    ax.set_xticklabels(AIRFOILS)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)

    for i in range(len(METHODS)):
        for j in range(len(AIRFOILS)):
            color = "white" if data[i, j] >= 2.5 else "black"
            ax.text(j, i, f"#{data[i, j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_title("Method Ranking by Initial Airfoil (#1 = best)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank")
    fig.savefig(OUT_DIR / "05_ranking.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '05_ranking.png'}")


# ============================================================
#  Figure 6: Stability (multi-seed)
# ============================================================
def fig_stability():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(METHODS))

    means = [STABILITY[m]["mean"] for m in METHODS]
    stds = [STABILITY[m]["std"] for m in METHODS]

    bars = ax.bar(x, means, yerr=stds, capsize=8,
                  color=METHOD_COLORS, edgecolor="white", linewidth=0.5,
                  error_kw={"linewidth": 1.5, "ecolor": "#333"})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.003,
                f"{mean:.4f}\n+/-{std:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Weighted CD (mean +/- std, 5 seeds)")
    ax.set_title("Optimization Stability across Random Seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "06_stability.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '06_stability.png'}")


# ============================================================
#  Figure 7: Computation time
# ============================================================
def fig_time():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(AIRFOILS))
    width = 0.25

    for i, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
        ax.bar(x + i * width, TIME_DATA[:, i], width,
               label=method, color=color, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Computation Time per Initial Airfoil")
    ax.set_xticks(x + width)
    ax.set_xticklabels(AIRFOILS)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "07_time.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '07_time.png'}")


# ============================================================
#  Figure 8: Airfoil shape comparison
# ============================================================
def fig_airfoil_shapes():
    import aerosandbox as asb

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    airfoil_names = ["naca4412", "clarky", "e387"]

    for ax, name in zip(axes, airfoil_names):
        af = asb.KulfanAirfoil(name)
        cu = af.upper_coordinates()
        cl = af.lower_coordinates()
        ax.fill_between(cu[:, 0], cu[:, 1], cl[:, 1], alpha=0.2, color="#3498db")
        ax.plot(cu[:, 0], cu[:, 1], color="#2c3e50", linewidth=1.5)
        ax.plot(cl[:, 0], cl[:, 1], color="#2c3e50", linewidth=1.5)
        ax.set_title(name.upper(), fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")

    fig.suptitle("Representative Airfoil Shapes", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_airfoil_shapes.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '08_airfoil_shapes.png'}")


# ============================================================
#  Figure 9: CL-CD polar (one airfoil, TAT vs NeuralFoil)
# ============================================================
def fig_cl_cd_polar():
    import aerosandbox as asb
    from piern_airfoil.thin_airfoil import thin_airfoil_from_kulfan

    fig, ax = plt.subplots(figsize=(8, 5))
    name = "naca4412"
    af = asb.KulfanAirfoil(name)

    alphas = np.linspace(0, 12, 25)
    CLs_nf, CDs_nf = [], []
    CLs_tat, CDs_tat = [], []

    for a in alphas:
        nf = af.get_aero_from_neuralfoil(alpha=a, Re=500e3, mach=0.03)
        CLs_nf.append(float(np.asarray(nf["CL"]).flatten()[0]))
        CDs_nf.append(float(np.asarray(nf["CD"]).flatten()[0]))

        tat = thin_airfoil_from_kulfan(af, alpha=a, mach=0.03, Re=500e3)
        CLs_tat.append(tat.CL)
        CDs_tat.append(tat.CD)

    ax.plot(CDs_nf, CLs_nf, "o-", color="#3498db", label="NeuralFoil",
            markersize=4, linewidth=1.5)
    ax.plot(CDs_tat, CLs_tat, "s--", color="#e74c3c", label="Thin Airfoil Theory",
            markersize=4, linewidth=1.5)

    # Mark CL targets
    cl_targets = [0.8, 1.0, 1.2, 1.4, 1.5, 1.6]
    for cl_t in cl_targets:
        ax.axhline(y=cl_t, color="gray", linestyle=":", alpha=0.3)
        ax.text(0.001, cl_t + 0.02, f"CL={cl_t}", fontsize=7, color="gray")

    ax.set_xlabel("CD")
    ax.set_ylabel("CL")
    ax.set_title(f"CL-CD Polar: {name.upper()} (Re=500k, Mach=0.03)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUT_DIR / "09_cl_cd_polar.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '09_cl_cd_polar.png'}")


# ============================================================
#  Figure 10: Summary table
# ============================================================
def fig_summary_table():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    headers = ["Method", "Avg Weighted CD", "Avg Rank", "Avg Time(s)",
               "Stability(5-seed)", "Best On"]

    ranks = [np.mean(RANKS[m]) for m in METHODS]

    rows = []
    for i, m in enumerate(METHODS):
        best_on = []
        for j, af in enumerate(AIRFOILS):
            if RANKS[m][j] == 1:
                best_on.append(af)
        rows.append([
            m,
            f"{WCD_DATA[:,i].mean():.5f}",
            f"{ranks[i]:.1f}",
            f"{TIME_DATA[:,i].mean():.1f}",
            f"+/-{STABILITY[m]['std']:.4f}",
            ", ".join(best_on) if best_on else "--",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best row (lowest avg rank)
    best_row = int(np.argmin(ranks)) + 1
    for j in range(len(headers)):
        table[best_row, j].set_facecolor("#d5f5e3")

    ax.set_title("Benchmark Summary: HPA Multi-point Optimization",
                 fontsize=13, fontweight="bold", pad=20)
    fig.savefig(OUT_DIR / "10_summary_table.png")
    plt.close(fig)
    print(f"  Saved {OUT_DIR / '10_summary_table.png'}")


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    print("Generating benchmark visualizations...")
    export_csv()
    fig_weighted_cd()
    fig_weighted_cd_zoomed()
    fig_improvement()
    fig_radar()
    fig_ranking()
    fig_stability()
    fig_time()
    fig_airfoil_shapes()
    fig_cl_cd_polar()
    fig_summary_table()
    print(f"\nAll figures saved to {OUT_DIR}/")
