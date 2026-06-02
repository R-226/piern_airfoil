"""
全流程 Benchmark — 从图像/.dat/prompt 到优化结果。

对比:
  1. Ground Truth: 直接从 aerosandbox 加载 KulfanAirfoil
  2. .dat Pipeline: 翼型 → .dat 文件 → load_dat → 拟合 → 优化
  3. Image Pipeline: 预渲染图片 → edge detection → 拟合 → 优化
  4. Prompt Pipeline: 中文 prompt → 参数提取 → 优化

翼型来源: data/benchmark_airfoils.json (固定集合, 基于 brentq 初始 CD 过滤)
  - Normal: 30
  - Medium: 44
  - Hard:   31

图片来源: data/benchmark_images/ (预渲染)

输出:
  results/pipeline_normal.png   — 常规翼型 pipeline 对比
  results/pipeline_medium.png   — 中等翼型 pipeline 对比
  results/pipeline_hard.png     — 困难翼型 pipeline 对比
  results/pipeline_benchmark.csv — 原始数据
"""

from __future__ import annotations

import json
import os
import sys
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

# ── 通用 prompt 模板 ─────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "设计一个翼型，马赫数0.03，CL目标值[0.8, 1.0, 1.2, 1.4, 1.5, 1.6]，"
    "权重[5, 6, 7, 8, 9, 10]，厚度@33%c>=0.128，力矩系数>=-0.133，后缘角>=6.03°"
)


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
    pipeline_type: str  # "ground_truth", "dat", "image", "prompt"
    cd: float
    time: float
    success: bool
    cd_gap: float = 0.0


# ── 工具函数 ──────────────────────────────────────────────────────────


def generate_dat_file(name: str, output_path: Path) -> Path:
    """从 aerosandbox 生成 .dat 文件。"""
    af = asb.Airfoil(name)
    coords = af.coordinates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.7f} {y:.7f}\n")
    return output_path


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

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    t0 = time.perf_counter()
    try:
        result = optimizer.optimize(airfoil)
        elapsed = time.perf_counter() - t0
    except Exception:
        elapsed = time.perf_counter() - t0
        sys.stdout.close()
        sys.stdout = old_stdout
        raise
    sys.stdout.close()
    sys.stdout = old_stdout
    return result.final_cd, elapsed


def _contour_to_kulfan(contour) -> asb.KulfanAirfoil:
    return asb.Airfoil(coordinates=contour.to_selig_coords()).to_kulfan_airfoil()


# ── Pipeline 测试 ─────────────────────────────────────────────────────


def test_ground_truth(name: str) -> PipelineResult:
    try:
        af = asb.KulfanAirfoil(name)
        cd, t = optimize_airfoil(af)
        return PipelineResult(name, "ground_truth", cd, t, True)
    except Exception:
        return PipelineResult(name, "ground_truth", float("inf"), 0, False)


def test_dat_pipeline(name: str, tmp_dir: Path) -> PipelineResult:
    try:
        from piern.view.extract import load_dat

        dat_path = tmp_dir / f"{name}.dat"
        generate_dat_file(name, dat_path)

        t0 = time.perf_counter()
        contour = load_dat(dat_path)
        kaf = _contour_to_kulfan(contour)
        cd, _ = optimize_airfoil(kaf)
        total_time = time.perf_counter() - t0
        return PipelineResult(name, "dat", cd, total_time, True)
    except Exception:
        return PipelineResult(name, "dat", float("inf"), 0, False)


def test_image_pipeline(name: str) -> PipelineResult:
    try:
        from piern.view.extract import extract_airfoil

        img_path = IMAGES_DIR / f"{name}.png"
        if not img_path.exists():
            return PipelineResult(name, "image", float("inf"), 0, False)

        t0 = time.perf_counter()
        contour = extract_airfoil(img_path, method="edge")
        kaf = _contour_to_kulfan(contour)
        cd, _ = optimize_airfoil(kaf)
        total_time = time.perf_counter() - t0
        return PipelineResult(name, "image", cd, total_time, True)
    except Exception:
        return PipelineResult(name, "image", float("inf"), 0, False)


def test_prompt_pipeline() -> PipelineResult:
    """Prompt Pipeline: 中文 prompt → 参数提取 → 优化。只跑一次。"""
    try:
        from piern.prompt2data.encoder_extractor import (
            extract, CharTokenizer, FieldClassifier,
            NUM_FIELDS, SAVE_DIR, DEVICE,
        )
        import torch

        tokenizer = CharTokenizer(max_len=512)
        model = FieldClassifier(
            vocab_size=tokenizer.vocab_size, d_model=128, nhead=4,
            num_layers=3, dim_ff=512, max_len=512, num_fields=NUM_FIELDS,
        ).to(DEVICE)
        model.load_state_dict(torch.load(SAVE_DIR, map_location=DEVICE, weights_only=True))
        model.eval()

        t0 = time.perf_counter()
        params = extract(model, tokenizer, PROMPT_TEMPLATE)
        af = asb.KulfanAirfoil("naca0012")
        cd, _ = optimize_airfoil(af)
        total_time = time.perf_counter() - t0
        return PipelineResult("prompt_global", "prompt", cd, total_time, True)
    except Exception:
        return PipelineResult("prompt_global", "prompt", float("inf"), 0, False)


# ── Benchmark 运行 ────────────────────────────────────────────────────


def run_pipeline_benchmark(
    airfoils: list[str],
    tmp_dir: Path,
) -> list[PipelineResult]:
    """运行 ground_truth / dat / image pipeline，prompt 只跑一次。"""
    results = []
    total = len(airfoils) * 3 + 1  # 3 per airfoil + 1 prompt
    idx = 0

    for name in airfoils:
        idx += 1
        print(f"  [{idx}/{total}] {name} ground_truth...", end=" ", flush=True)
        r = test_ground_truth(name)
        results.append(r)
        print(f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED")

        idx += 1
        print(f"  [{idx}/{total}] {name} dat...", end=" ", flush=True)
        r = test_dat_pipeline(name, tmp_dir)
        results.append(r)
        print(f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED")

        idx += 1
        print(f"  [{idx}/{total}] {name} image...", end=" ", flush=True)
        r = test_image_pipeline(name)
        results.append(r)
        print(f"CD={r.cd:.4f} {r.time:.1f}s" if r.success else "FAILED")

    # Prompt pipeline 只跑一次
    idx += 1
    print(f"  [{idx}/{total}] prompt...", end=" ", flush=True)
    r = test_prompt_pipeline()
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
    pipeline_types = ["ground_truth", "dat", "image"]
    pipeline_colors = {
        "ground_truth": "#2CA02C",
        "dat": "#1F77B4",
        "image": "#E45756",
    }
    pipeline_labels = {
        "ground_truth": "Ground Truth",
        "dat": ".dat Pipeline",
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
    gap_types = ["dat", "image"]
    for i, pt in enumerate(gap_types):
        gaps = []
        for af in airfoils:
            rs = [r for r in cat_results if r.airfoil_name == af and r.pipeline_type == pt]
            gaps.append(rs[0].cd_gap if rs and rs[0].success else float("nan"))
        offset = (i - 0.5) * 0.35
        ax.bar(x + offset, gaps, 0.35, label=pt, color=pipeline_colors[pt], alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("CD Gap vs Ground Truth")
    ax.set_title("CD Gap (0 = perfect extraction)", fontweight="bold")
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
    pipeline_types = ["ground_truth", "dat", "image"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(categories))
    n_p = len(pipeline_types)
    width = 0.7 / n_p
    colors = {"ground_truth": "#2CA02C", "dat": "#1F77B4", "image": "#E45756"}

    # 图1: 成功率
    ax = axes[0]
    for i, pt in enumerate(pipeline_types):
        rates = []
        for afs in cat_airfoils:
            total = sum(1 for r in results if r.airfoil_name in afs and r.pipeline_type == pt)
            success = sum(1 for r in results if r.airfoil_name in afs and r.pipeline_type == pt and r.success)
            rates.append(success / total * 100 if total > 0 else 0)
        offset = (i - (n_p - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width, label=pt, color=colors[pt], alpha=0.85)
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

    # 图2: 平均 CD Gap
    ax = axes[1]
    for i, pt in enumerate(["dat", "image"]):
        gaps = []
        for afs in cat_airfoils:
            rs = [r for r in results if r.airfoil_name in afs and r.pipeline_type == pt and r.success]
            gaps.append(np.mean([r.cd_gap for r in rs]) if rs else 0)
        offset = (i - 0.5) * 0.35
        ax.bar(x + offset, gaps, 0.35, label=pt, color=colors[pt], alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean CD Gap")
    ax.set_title("Extraction Accuracy (0 = perfect)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 图3: 平均时间
    ax = axes[2]
    for i, pt in enumerate(pipeline_types):
        times = []
        for afs in cat_airfoils:
            rs = [r for r in results if r.airfoil_name in afs and r.pipeline_type == pt and r.success]
            times.append(np.mean([r.time for r in rs]) if rs else 0)
        offset = (i - (n_p - 1) / 2) * width
        ax.bar(x + offset, times, width, label=pt, color=colors[pt], alpha=0.85)
    ax.set_ylabel("Mean Time (s)")
    ax.set_title("End-to-End Time", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Pipeline Benchmark Summary", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
        writer.writerow(["airfoil", "pipeline_type", "cd", "cd_gap", "time", "success"])
        for r in results:
            writer.writerow([
                r.airfoil_name, r.pipeline_type,
                f"{r.cd:.6f}" if r.success else "inf",
                f"{r.cd_gap:.6f}" if r.success else "inf",
                f"{r.time:.3f}",
                r.success,
            ])
    print(f"CSV 已保存: {save_path}")


# ── 汇总表 ──────────────────────────────────────────────────────────────


def print_summary(results: list[PipelineResult], category: str, airfoils: list[str]):
    print(f"\n{'='*80}")
    print(f"{category} ({len(airfoils)} airfoils)")
    print("=" * 80)

    for pt in ["ground_truth", "dat", "image", "prompt"]:
        rs = [r for r in results if r.pipeline_type == pt and r.airfoil_name in airfoils or r.pipeline_type == "prompt"]
        rs_success = [r for r in rs if r.success]
        rs_cat = [r for r in results if r.pipeline_type == pt and (r.airfoil_name in airfoils if pt != "prompt" else True)]
        rs_cat_success = [r for r in rs_cat if r.success]
        total = len([r for r in results if r.pipeline_type == pt and (r.airfoil_name in airfoils if pt != "prompt" else True)])
        if rs_cat_success:
            avg_cd = np.mean([r.cd for r in rs_cat_success])
            avg_gap = np.mean([r.cd_gap for r in rs_cat_success])
            avg_time = np.mean([r.time for r in rs_cat_success])
            sr = len(rs_cat_success) / total * 100 if total > 0 else 0
            print(f"  {pt:<14} CD={avg_cd:.4f}  Gap={avg_gap:+.4f}  Time={avg_time:.1f}s  SR={sr:.0f}%")
        else:
            print(f"  {pt:<14} (no success)")


# ── 主函数 ──────────────────────────────────────────────────────────────


def main():
    import tempfile

    normal_afs, medium_afs, hard_afs = load_benchmark_airfoils()
    all_airfoils = normal_afs + medium_afs + hard_afs

    print("=" * 80)
    print("Pipeline Benchmark")
    print("=" * 80)
    print(f"Normal: {len(normal_afs)}, Medium: {len(medium_afs)}, Hard: {len(hard_afs)}")
    print(f"总计: {len(all_airfoils)} 个翼型, {len(all_airfoils) * 3 + 1} 次优化")
    print()

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="piern_pipe_") as tmp_dir:
        results = run_pipeline_benchmark(all_airfoils, Path(tmp_dir))

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
