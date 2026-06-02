"""
Pipeline Benchmark — 翼型提取精度对比。

对比:
  1. Ground Truth: 直接从 aerosandbox 加载 KulfanAirfoil
  2. Image Pipeline: 预渲染图片 → edge detection → Kulfan 拟合 → 优化

分解指标:
  - extraction_time: 轮廓提取耗时
  - optimization_time: 优化耗时
  - kulfan_fit_error: 提取轮廓 vs Kulfan 拟合轮廓的 RMS 距离

翼型来源: data/benchmark_airfoils.json (固定集合)
图片来源: data/benchmark_images/ (预渲染)

输出:
  results/pipeline_normal.png    — 常规翼型 pipeline 对比
  results/pipeline_medium.png    — 中等翼型 pipeline 对比
  results/pipeline_hard.png      — 困难翼型 pipeline 对比
  results/pipeline_summary.png   — 汇总图
  results/pipeline_benchmark.csv — 原始数据 (含分解指标)
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
import matplotlib.font_manager as fm

_CJK_FONT = "Noto Sans CJK JP"
try:
    fm.findfont(_CJK_FONT, fallback_to_default=False)
    plt.rcParams["font.family"] = _CJK_FONT
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# ── 问题定义 ──────────────────────────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03

BENCHMARK_JSON = Path(__file__).parent.parent / "data" / "benchmark_airfoils.json"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "benchmark_images"


# ── 翼型加载 ──────────────────────────────────────────────────────────


def load_benchmark_airfoils() -> tuple[list[str], list[str], list[str]]:
    """从固定 benchmark 文件加载翼型集合。"""
    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    return bench["normal"], bench["medium"], bench["hard"]


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    airfoil_name: str
    pipeline_type: str  # "ground_truth", "image"
    cd: float
    time: float
    success: bool
    cd_gap: float = 0.0
    extraction_time: float = 0.0  # 轮廓提取耗时
    optimization_time: float = 0.0  # 优化耗时
    kulfan_fit_error: float = 0.0  # Kulfan 拟合 RMS 误差


# ── 工具函数 ──────────────────────────────────────────────────────────


def _suppress_stdout():
    """Suppress stdout by redirecting fd 1 (fd-based, safe for exceptions)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    return old_stdout


def _restore_stdout(old_stdout: int):
    """Restore stdout from saved fd."""
    os.dup2(old_stdout, 1)
    os.close(old_stdout)


def optimize_airfoil(airfoil) -> tuple[float, float]:
    """运行 PiERN Router 优化，返回 (CD, time)。"""
    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
    from piern.router.opt_router import OptRouter

    router = OptRouter.from_mlp()
    optimizer = AdaptiveHierarchicalOptimizer(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        Re=RE,
        mach=MACH,
        start_weights=4,
        router=router,
    )

    old_stdout = _suppress_stdout()
    t0 = time.perf_counter()
    try:
        result = optimizer.optimize(airfoil)
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = time.perf_counter() - t0
        _restore_stdout(old_stdout)
        raise
    _restore_stdout(old_stdout)
    return result.final_cd, elapsed


def _contour_to_kulfan(contour) -> asb.KulfanAirfoil:
    return asb.Airfoil(coordinates=contour.to_selig_coords()).to_kulfan_airfoil()


def _kulfan_fit_rms(contour, kaf: asb.KulfanAirfoil) -> float:
    """提取轮廓 vs Kulfan 拟合轮廓的 RMS 距离。"""
    from scipy.interpolate import interp1d

    # 提取轮廓 (Selig 格式)
    extracted = contour.to_selig_coords()  # (M, 2)

    # Kulfan 拟合轮廓
    fitted = kaf.coordinates  # (K, 2)

    # 用弧长参数化做点对点匹配
    def _arc_length_param(coords):
        dx = np.diff(coords[:, 0])
        dy = np.diff(coords[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        return s / s[-1] if s[-1] > 0 else s

    s_ext = _arc_length_param(extracted)
    s_fit = _arc_length_param(fitted)

    # 在 [0, 1] 上均匀采样
    n_sample = 200
    t = np.linspace(0, 1, n_sample)

    ext_x = interp1d(s_ext, extracted[:, 0], kind="linear")(t)
    ext_y = interp1d(s_ext, extracted[:, 1], kind="linear")(t)
    fit_x = interp1d(s_fit, fitted[:, 0], kind="linear")(t)
    fit_y = interp1d(s_fit, fitted[:, 1], kind="linear")(t)

    dist = np.sqrt((ext_x - fit_x)**2 + (ext_y - fit_y)**2)
    return float(np.sqrt(np.mean(dist**2)))


# ── Pipeline 测试 ─────────────────────────────────────────────────────


def test_ground_truth(name: str) -> PipelineResult:
    try:
        af = asb.KulfanAirfoil(name)
        cd, t = optimize_airfoil(af)
        return PipelineResult(name, "ground_truth", cd, t, True)
    except Exception:
        return PipelineResult(name, "ground_truth", float("inf"), 0, False)


def test_image_pipeline(name: str) -> PipelineResult:
    try:
        from piern.view.extract import extract_airfoil

        img_path = IMAGES_DIR / f"{name}.png"
        if not img_path.exists():
            return PipelineResult(name, "image", float("inf"), 0, False)

        # Step 1: 轮廓提取
        t0 = time.perf_counter()
        contour = extract_airfoil(img_path, method="edge")
        extraction_time = time.perf_counter() - t0

        # Step 2: Kulfan 拟合
        t1 = time.perf_counter()
        kaf = _contour_to_kulfan(contour)
        fit_time = time.perf_counter() - t1

        # Kulfan 拟合误差
        fit_error = _kulfan_fit_rms(contour, kaf)

        # Step 3: 优化
        cd, optimization_time = optimize_airfoil(kaf)

        total_time = extraction_time + fit_time + optimization_time
        return PipelineResult(
            name, "image", cd, total_time, True,
            extraction_time=extraction_time,
            optimization_time=optimization_time,
            kulfan_fit_error=fit_error,
        )
    except Exception:
        return PipelineResult(name, "image", float("inf"), 0, False)


# ── Benchmark 运行 ────────────────────────────────────────────────────


def run_pipeline_benchmark(
    airfoils: list[str],
) -> list[PipelineResult]:
    """运行 ground_truth / image pipeline。"""
    results = []
    total = len(airfoils) * 2
    idx = 0

    for name in airfoils:
        idx += 1
        print(f"  [{idx}/{total}] {name} ground_truth...", end=" ", flush=True)
        r = test_ground_truth(name)
        results.append(r)
        print(f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED")

        idx += 1
        print(f"  [{idx}/{total}] {name} image...", end=" ", flush=True)
        r = test_image_pipeline(name)
        results.append(r)
        print(f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED")

    # 计算 CD Gap (vs ground truth)
    gt_map = {
        r.airfoil_name: r.cd
        for r in results
        if r.pipeline_type == "ground_truth" and r.success
    }
    for r in results:
        if r.success and r.airfoil_name in gt_map:
            r.cd_gap = r.cd - gt_map[r.airfoil_name]

    return results


# ── 可视化 ─────────────────────────────────────────────────────────────


def visualize_by_category(
    results: list[PipelineResult],
    airfoils: list[str],
    category: str,
    save_path: str,
):
    """按类别可视化 pipeline 对比。"""
    pipeline_types = ["ground_truth", "image"]
    pipeline_colors = {
        "ground_truth": "#2CA02C",
        "image": "#E45756",
    }
    pipeline_labels = {
        "ground_truth": "Ground Truth",
        "image": "Image Pipeline",
    }

    cat_results = [r for r in results if r.airfoil_name in airfoils]
    if not cat_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(max(18, len(airfoils) * 0.6), 6))
    x = np.arange(len(airfoils))
    n_p = len(pipeline_types)
    width = 0.7 / n_p

    # 图1: CD 对比
    ax = axes[0]
    for i, pt in enumerate(pipeline_types):
        cds = []
        for af in airfoils:
            rs = [r for r in cat_results if r.airfoil_name == af and r.pipeline_type == pt]
            cds.append(rs[0].cd if rs and rs[0].success else float("nan"))
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, cds, width, label=pt, color=pipeline_colors[pt], alpha=0.85)
    ax.set_ylabel("Weighted CD")
    ax.set_title(f"CD — {category}", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 图2: CD Gap (vs Ground Truth)
    ax = axes[1]
    gaps = []
    for af in airfoils:
        rs = [r for r in cat_results if r.airfoil_name == af and r.pipeline_type == "image"]
        gaps.append(rs[0].cd_gap if rs and rs[0].success else float("nan"))
    ax.bar(x, gaps, 0.5, label="image", color=pipeline_colors["image"], alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("CD Gap vs Ground Truth")
    ax.set_title("Extraction Accuracy (0 = perfect)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 图3: 时间
    ax = axes[2]
    for i, pt in enumerate(pipeline_types):
        times = []
        for af in airfoils:
            rs = [r for r in cat_results if r.airfoil_name == af and r.pipeline_type == pt]
            times.append(rs[0].time if rs and rs[0].success else 0)
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, times, width, label=pt, color=pipeline_colors[pt], alpha=0.85)
    ax.set_ylabel("Time (s)")
    ax.set_title("End-to-End Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n[:10] for n in airfoils], fontsize=6, rotation=45, ha="right")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Pipeline Benchmark — {category} ({len(airfoils)} airfoils)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存: {save_path}")


def visualize_summary(
    results: list[PipelineResult],
    normal_afs: list[str],
    medium_afs: list[str],
    hard_afs: list[str],
    save_path: str = "results/pipeline_summary.png",
):
    """汇总统计图: 各类别成功率、平均 CD Gap、平均时间。"""
    categories = ["Normal", "Medium", "Hard"]
    cat_airfoils = [normal_afs, medium_afs, hard_afs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(categories))

    # 图1: 成功率
    ax = axes[0]
    for i, pt in enumerate(["ground_truth", "image"]):
        rates = []
        for afs in cat_airfoils:
            total = sum(1 for r in results if r.airfoil_name in afs and r.pipeline_type == pt)
            success = sum(1 for r in results if r.airfoil_name in afs and r.pipeline_type == pt and r.success)
            rates.append(success / total * 100 if total > 0 else 0)
        offset = (i - 0.5) * 0.35
        color = "#2CA02C" if pt == "ground_truth" else "#E45756"
        bars = ax.bar(x + offset, rates, 0.35, label=pt, color=color, alpha=0.85)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Pipeline Success Rate", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # 图2: 平均 CD Gap (image only)
    ax = axes[1]
    gaps = []
    for afs in cat_airfoils:
        rs = [r for r in results if r.airfoil_name in afs and r.pipeline_type == "image" and r.success]
        gaps.append(np.mean([r.cd_gap for r in rs]) if rs else 0)
    bars = ax.bar(x, gaps, 0.5, label="image", color="#E45756", alpha=0.85)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{gap:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean CD Gap")
    ax.set_title("Extraction Accuracy (0 = perfect)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 图3: 平均时间
    ax = axes[2]
    for i, pt in enumerate(["ground_truth", "image"]):
        times = []
        for afs in cat_airfoils:
            rs = [r for r in results if r.airfoil_name in afs and r.pipeline_type == pt and r.success]
            times.append(np.mean([r.time for r in rs]) if rs else 0)
        offset = (i - 0.5) * 0.35
        color = "#2CA02C" if pt == "ground_truth" else "#E45756"
        ax.bar(x + offset, times, 0.35, label=pt, color=color, alpha=0.85)
    ax.set_ylabel("Mean Time (s)")
    ax.set_title("End-to-End Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Pipeline Benchmark Summary", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"汇总可视化已保存: {save_path}")


# ── CSV 导出 ────────────────────────────────────────────────────────────


def export_csv(
    results: list[PipelineResult],
    save_path: str = "results/pipeline_benchmark.csv",
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "airfoil", "pipeline_type", "cd", "cd_gap", "time",
            "extraction_time", "optimization_time", "kulfan_fit_error", "success",
        ])
        for r in results:
            writer.writerow([
                r.airfoil_name, r.pipeline_type,
                f"{r.cd:.6f}" if r.success else "inf",
                f"{r.cd_gap:.6f}" if r.success else "inf",
                f"{r.time:.3f}",
                f"{r.extraction_time:.3f}" if r.success else "0",
                f"{r.optimization_time:.3f}" if r.success else "0",
                f"{r.kulfan_fit_error:.6f}" if r.success else "inf",
                r.success,
            ])
    print(f"CSV 已保存: {save_path}")


# ── 汇总表 ──────────────────────────────────────────────────────────────


def print_summary(results: list[PipelineResult], category: str, airfoils: list[str]):
    print(f"\n{'='*100}")
    print(f"{category} ({len(airfoils)} airfoils)")
    print("=" * 100)

    for pt in ["ground_truth", "image"]:
        rs = [r for r in results if r.pipeline_type == pt and r.airfoil_name in airfoils]
        rs_success = [r for r in rs if r.success]
        total = len(rs)
        if rs_success:
            avg_cd = np.mean([r.cd for r in rs_success])
            avg_gap = np.mean([r.cd_gap for r in rs_success])
            avg_time = np.mean([r.time for r in rs_success])
            sr = len(rs_success) / total * 100 if total > 0 else 0
            print(f"  {pt:<14} CD={avg_cd:.4f}  Gap={avg_gap:+.4f}  Time={avg_time:.1f}s  SR={sr:.0f}%")

            # image pipeline 额外显示分解指标
            if pt == "image":
                avg_extract = np.mean([r.extraction_time for r in rs_success])
                avg_opt = np.mean([r.optimization_time for r in rs_success])
                avg_fit_err = np.mean([r.kulfan_fit_error for r in rs_success])
                print(f"  {'':<14} Extract={avg_extract:.2f}s  Opt={avg_opt:.1f}s  FitErr={avg_fit_err:.4f}")
        else:
            print(f"  {pt:<14} (no success)")


# ── 主函数 ──────────────────────────────────────────────────────────────


def main():
    normal_afs, medium_afs, hard_afs = load_benchmark_airfoils()
    all_airfoils = normal_afs + medium_afs + hard_afs

    print("=" * 80)
    print("Pipeline Benchmark")
    print("=" * 80)
    print(f"Normal: {len(normal_afs)}, Medium: {len(medium_afs)}, Hard: {len(hard_afs)}")
    print(f"总计: {len(all_airfoils)} 个翼型, {len(all_airfoils) * 2} 次优化")
    print()

    t0 = time.perf_counter()

    results = run_pipeline_benchmark(all_airfoils)

    # 按类别输出统计
    print_summary(results, "Normal", normal_afs)
    print_summary(results, "Medium", medium_afs)
    print_summary(results, "Hard", hard_afs)

    # 可视化
    visualize_by_category(results, normal_afs, "Normal", "results/pipeline_normal.png")
    visualize_by_category(results, medium_afs, "Medium", "results/pipeline_medium.png")
    visualize_by_category(results, hard_afs, "Hard", "results/pipeline_hard.png")
    visualize_summary(results, normal_afs, medium_afs, hard_afs)

    export_csv(results)

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
