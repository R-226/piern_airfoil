"""可视化 XFoil+DE 优化结果。

在 2-3 个翼型上运行 XFoil+DE，对比初始 vs 优化后的形状。
展示：
1. 翼型轮廓对比 (初始 vs 优化)
2. CD 收敛曲线
3. 优化参数变化
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 样式常量 ─────────────────────────────────────────────────────────────

SERIF = "serif"
COLORS = {
    "initial": "#3370AC",
    "optimized": "#D44B3F",
    "baseline_opt": "#2A8C6A",
}


def plot_airfoil_comparison(
    airfoil_name: str,
    initial_coords: np.ndarray,
    optimized_coords: np.ndarray,
    initial_cd: float,
    optimized_cd: float,
    save_path: str,
):
    """绘制初始 vs 优化翼型轮廓对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── (a) 翼型轮廓对比 ──
    ax = axes[0]
    ax.plot(initial_coords[:, 0], initial_coords[:, 1],
            color=COLORS["initial"], linewidth=1.5, label="Initial", alpha=0.8)
    ax.plot(optimized_coords[:, 0], optimized_coords[:, 1],
            color=COLORS["optimized"], linewidth=1.5, label="XFoil+DE Optimized", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x/c", fontsize=9, fontfamily=SERIF)
    ax.set_ylabel("y/c", fontsize=9, fontfamily=SERIF)
    ax.set_title(f"(a) Airfoil Shape — {airfoil_name.upper()}", fontsize=10, fontfamily=SERIF, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    # ── (b) CD 对比 ──
    ax = axes[1]
    methods = ["Initial", "XFoil+DE"]
    cds = [initial_cd, optimized_cd]
    colors = [COLORS["initial"], COLORS["optimized"]]
    bars = ax.bar(methods, cds, color=colors, alpha=0.8, width=0.5)
    for bar, cd in zip(bars, cds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{cd:.4f}", ha="center", va="bottom", fontsize=9, fontfamily=SERIF)
    ax.set_ylabel("Weighted CD", fontsize=9, fontfamily=SERIF)
    ax.set_title("(b) Drag Coefficient", fontsize=10, fontfamily=SERIF, fontweight="bold")
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.suptitle(f"XFoil+DE Optimization — {airfoil_name.upper()}",
                 fontsize=12, fontfamily=SERIF, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")


def run_xfoil_visualization(airfoil_name: str, save_dir: str):
    """运行 XFoil+DE 优化并可视化结果。"""
    import aerosandbox as asb
    from piern_airfoil.xfoil_optimizer import xfoil_optimize
    from piern_airfoil.xfoil_baseline import xfoil_cd

    # 标准 CL/Re 参数
    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
    MACH = 0.03

    print(f"\n{'='*60}")
    print(f"Optimizing: {airfoil_name.upper()}")
    print(f"{'='*60}")

    # 获取初始翼型
    try:
        af_init = asb.KulfanAirfoil(airfoil_name)
    except Exception:
        print(f"Warning: Cannot load {airfoil_name}, using default")
        af_init = asb.KulfanAirfoil("naca0012")

    initial_cd = xfoil_cd(airfoil_name, CL_TARGETS, RE, CL_WEIGHTS, MACH)
    print(f"Initial CD (XFoil): {initial_cd:.6f}")

    # 运行 XFoil+DE 优化 (使用较小参数，复现 benchmark 行为)
    result = xfoil_optimize(
        airfoil_name, CL_TARGETS, RE, CL_WEIGHTS, MACH,
        maxiter=5, popsize=3,  # benchmark 使用的参数
    )

    print(f"Optimized CD: {result.final_cd:.6f}")
    print(f"Time: {result.time_s:.1f}s")
    print(f"Success: {result.success}")

    # 获取坐标用于绘图
    initial_coords = af_init.coordinates
    optimized_coords = result.optimized_airfoil.coordinates

    # 保存图
    save_path = str(Path(save_dir) / f"xfoil_opt_{airfoil_name}.png")
    plot_airfoil_comparison(
        airfoil_name, initial_coords, optimized_coords,
        initial_cd, result.final_cd, save_path
    )

    return result


def main():
    save_dir = str(Path(__file__).resolve().parent.parent / "results")
    Path(save_dir).mkdir(exist_ok=True)

    # 测试 2 个翼型: naca0012 + benchmark 中的一个
    test_airfoils = ["naca0012", "fx67k150"]

    results = []
    for af in test_airfoils:
        try:
            r = run_xfoil_visualization(af, save_dir)
            results.append(r)
        except FileNotFoundError as e:
            print(f"Skipping {af}: {e}")
        except Exception as e:
            print(f"Error with {af}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        imp = (r.initial_cd - r.final_cd) / r.initial_cd * 100 if r.initial_cd > 0 else 0
        print(f"{r.airfoil_name:15s} | Initial: {r.initial_cd:.4f} | Final: {r.final_cd:.4f} | "
              f"Improve: {imp:+.1f}% | Time: {r.time_s:.1f}s")


if __name__ == "__main__":
    main()
