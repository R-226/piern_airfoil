"""
翼型形状可视化 — 检查优化结果的物理合理性。

生成:
  results/airfoil_shapes_normal.png   — 常规翼型 (best/median/worst)
  results/airfoil_shapes_medium.png   — 中等翼型 (best/median/worst)
  results/airfoil_shapes_hard.png     — 困难翼型 (best/median/worst)
  results/airfoil_dae11.png           — DAE-11 参考翼型

每个子图展示: 初始翼型 (灰色) vs 优化后翼型 (蓝色), 附带 CD 值。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CJK_FONT = "Noto Sans CJK JP"
try:
    fm.findfont(_CJK_FONT, fallback_to_default=False)
    plt.rcParams["font.family"] = _CJK_FONT
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# ── 问题定义 (与 benchmark 一致) ─────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03

SCREENING_CSV = Path(__file__).parent.parent / "results" / "airfoil_screening.csv"


# ── 工具函数 ──────────────────────────────────────────────────────────


def optimize_and_get_shapes(airfoil_name: str) -> dict:
    """优化翼型并返回初始/优化后的坐标和 CD。

    Returns:
        dict with keys: name, init_coords, opt_coords, init_cd, opt_cd, success
    """
    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
    from piern.router.opt_router import OptRouter
    from piern_airfoil.eval import evaluate_weighted_cd

    af_init = asb.KulfanAirfoil(airfoil_name)
    init_coords = af_init.coordinates
    init_cd = evaluate_weighted_cd(af_init, CL_TARGETS, RE, CL_WEIGHTS, MACH)

    router = OptRouter.from_mlp()
    optimizer = AdaptiveHierarchicalOptimizer(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        Re=RE,
        mach=MACH,
        start_weights=4,
        router=router,
    )

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        result = optimizer.optimize(af_init)
        sys.stdout.close()
        sys.stdout = old_stdout

        opt_cd = result.final_cd
        opt_af = result.airfoil
        opt_coords = opt_af.coordinates

        return {
            "name": airfoil_name,
            "init_coords": init_coords,
            "opt_coords": opt_coords,
            "init_cd": init_cd,
            "opt_cd": opt_cd,
            "success": True,
        }
    except Exception as e:
        sys.stdout.close()
        sys.stdout = old_stdout
        return {
            "name": airfoil_name,
            "init_coords": init_coords,
            "opt_coords": None,
            "init_cd": init_cd,
            "opt_cd": float("inf"),
            "success": False,
        }


def plot_airfoil_comparison(ax, data: dict, title: str = ""):
    """在 axes 上绘制初始 vs 优化后的翼型。"""
    ax.plot(
        data["init_coords"][:, 0], data["init_coords"][:, 1],
        color="#cccccc", linewidth=2, label=f'Initial (CD={data["init_cd"]:.4f})',
        alpha=0.8,
    )
    if data["success"] and data["opt_coords"] is not None:
        ax.plot(
            data["opt_coords"][:, 0], data["opt_coords"][:, 1],
            color="#1F77B4", linewidth=2, label=f'Optimized (CD={data["opt_cd"]:.4f})',
        )
    else:
        ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="red", fontweight="bold")

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# ── 加载翼型列表 ─────────────────────────────────────────────────────


def load_airfoils_by_category() -> dict:
    """从筛选结果加载翼型，按类别返回 best/median/worst。"""
    with open(SCREENING_CSV) as f:
        rows = list(csv.DictReader(f))

    result = {}
    for cat in ["normal", "medium", "hard"]:
        cat_rows = sorted(
            [r for r in rows if r["category"] == cat],
            key=lambda r: float(r["cd"]),
        )
        n = len(cat_rows)
        result[cat] = {
            "best": cat_rows[0]["name"],
            "median": cat_rows[n // 2]["name"],
            "worst": cat_rows[-1]["name"],
        }
    return result


# ── 主函数 ──────────────────────────────────────────────────────────────


def main():
    import time

    categories = load_airfoils_by_category()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 按类别生成可视化 ──
    for cat in ["normal", "medium", "hard"]:
        examples = categories[cat]
        print(f"\n{'='*60}")
        print(f"类别: {cat.upper()}")
        print(f"  Best:   {examples['best']}")
        print(f"  Median: {examples['median']}")
        print(f"  Worst:  {examples['worst']}")
        print("=" * 60)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (label, name) in zip(axes, examples.items()):
            print(f"  优化 {name} ({label})...", end=" ", flush=True)
            t0 = time.perf_counter()
            data = optimize_and_get_shapes(name)
            elapsed = time.perf_counter() - t0
            status = f"CD={data['opt_cd']:.4f} ({elapsed:.1f}s)" if data["success"] else "FAILED"
            print(status)

            title = f"{cat.title()} {label}: {name}"
            plot_airfoil_comparison(ax, data, title)

        plt.suptitle(
            f"Airfoil Shapes — {cat.title()} (Initial vs Optimized)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = output_dir / f"airfoil_shapes_{cat}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  保存: {save_path}")

    # ── DAE-11 参考翼型 ──
    print(f"\n{'='*60}")
    print("DAE-11 参考翼型")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # DAE-11 原始形状
    try:
        dae11 = asb.Airfoil("dae11")
        coords = dae11.coordinates  # type: ignore[attr-defined]
        axes[0].plot(coords[:, 0], coords[:, 1], color="#2CA02C", linewidth=2)
        axes[0].set_aspect("equal")
        axes[0].set_title("DAE-11 Airfoil Shape", fontweight="bold")
        axes[0].grid(True, alpha=0.2)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
    except Exception:
        axes[0].text(0.5, 0.5, "DAE-11 not found", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=14, color="red")

    # DAE-11 优化
    print("  优化 DAE-11...", end=" ", flush=True)
    t0 = time.perf_counter()
    data = optimize_and_get_shapes("dae11")
    elapsed = time.perf_counter() - t0
    status = f"CD={data['opt_cd']:.4f} ({elapsed:.1f}s)" if data["success"] else "FAILED"
    print(status)

    plot_airfoil_comparison(axes[1], data, "DAE-11: Initial vs Optimized")

    plt.suptitle("DAE-11 Reference Airfoil", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = output_dir / "airfoil_dae11.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {save_path}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("汇总: 各类别 best/median/worst 优化后 CD")
    print("=" * 60)
    print(f"{'类别':<10} {'位置':<10} {'翼型':<20} {'初始CD':>10} {'优化CD':>10}")
    print("-" * 62)
    for cat in ["normal", "medium", "hard"]:
        for label in ["best", "median", "worst"]:
            name = categories[cat][label]
            data = optimize_and_get_shapes(name)
            init_cd = f"{data['init_cd']:.4f}"
            opt_cd = f"{data['opt_cd']:.4f}" if data["success"] else "FAILED"
            print(f"{cat.title():<10} {label:<10} {name:<20} {init_cd:>10} {opt_cd:>10}")


if __name__ == "__main__":
    main()
