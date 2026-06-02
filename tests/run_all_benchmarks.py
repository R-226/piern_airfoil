"""
一键运行全部 benchmark — 适合在租借算力上后台执行。

用法:
    nohup uv run python tests/run_all_benchmarks.py > results/run_all.log 2>&1 &

输出目录:
    results/
    ├── run_all.log                    — 运行日志
    ├── benchmark_stats.csv            — Router benchmark 原始数据
    ├── benchmark_normal.png           — Router: 常规翼型对比
    ├── benchmark_medium.png           — Router: 中等翼型对比
    ├── benchmark_hard.png             — Router: 困难翼型对比
    ├── benchmark_summary.png          — Router: 汇总图
    ├── benchmark_method_comparison.png — NeuralFoil vs XFoil+DE 对比
    ├── benchmark_dist_normal.png      — Router: 常规翼型 CD 分布
    ├── benchmark_dist_medium.png      — Router: 中等翼型 CD 分布
    ├── benchmark_dist_hard.png        — Router: 困难翼型 CD 分布
    ├── benchmark_dist_all.png         — Router: 全部翼型 CD 分布
    ├── table_router_full.csv          — 完整结果表 (类别×方法)
    ├── table_router_latex.tex         — LaTeX 格式结果表
    ├── table_significance.csv         — 统计显著性检验结果
    ├── pipeline_benchmark.csv         — Pipeline benchmark 原始数据
    ├── pipeline_normal.png            — Pipeline: 常规翼型对比
    ├── pipeline_medium.png            — Pipeline: 中等翼型对比
    ├── pipeline_hard.png              — Pipeline: 困难翼型对比
    ├── pipeline_summary.png           — Pipeline: 汇总图
    ├── ablation.csv                   — 消融实验原始数据
    ├── ablation_1_hierarchical_vs_direct.png
    ├── ablation_2_router_effect.png
    ├── ablation_3_starting_dimension.png
    ├── ablation_4_dimension_contribution.png
    └── sensitivity.png

运行完成后, 把 results/ 目录打包传回来即可:
    tar czf results.tar.gz results/
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg: str):
    """带时间戳的日志输出。"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_router_benchmark():
    """运行 Router Benchmark。"""
    log("=" * 70)
    log("PART 1: Router Benchmark")
    log("=" * 70)

    from tests.benchmark_router import (
        load_benchmark_airfoils,
        run_benchmark_group,
        print_summary,
        visualize_normal,
        visualize_medium,
        visualize_hard,
        visualize_summary,
        visualize_method_comparison,
        visualize_distributions,
        visualize_difficulty_improvement,
        run_significance_tests,
        print_significance_tests,
        generate_results_table,
        export_csv,
    )

    normal_afs, medium_afs, hard_afs = load_benchmark_airfoils()
    all_afs = normal_afs + medium_afs + hard_afs
    log(f"翼型: normal={len(normal_afs)} medium={len(medium_afs)} hard={len(hard_afs)}")

    t0 = time.perf_counter()

    normal_stats, _ = run_benchmark_group(normal_afs, "Normal")
    print_summary(normal_stats, "Normal", normal_afs)
    visualize_normal(normal_stats, normal_afs)

    medium_stats, _ = run_benchmark_group(medium_afs, "Medium")
    print_summary(medium_stats, "Medium", medium_afs)
    visualize_medium(medium_stats, medium_afs)

    hard_stats, _ = run_benchmark_group(hard_afs, "Hard")
    print_summary(hard_stats, "Hard", hard_afs)
    visualize_hard(hard_stats, hard_afs)

    visualize_summary(normal_stats, hard_stats, normal_afs, hard_afs)

    all_stats = normal_stats + medium_stats + hard_stats
    all_afs_combined = normal_afs + medium_afs + hard_afs
    visualize_method_comparison(all_stats, all_afs_combined)

    # 分布可视化
    visualize_distributions(all_stats, normal_afs, "Normal", str(RESULTS_DIR / "benchmark_dist_normal.png"))
    visualize_distributions(all_stats, medium_afs, "Medium", str(RESULTS_DIR / "benchmark_dist_medium.png"))
    visualize_distributions(all_stats, hard_afs, "Hard", str(RESULTS_DIR / "benchmark_dist_hard.png"))
    visualize_distributions(all_stats, all_afs_combined, "All", str(RESULTS_DIR / "benchmark_dist_all.png"))

    # 难度-改善散点图
    visualize_difficulty_improvement(all_stats, normal_afs, "Normal", str(RESULTS_DIR / "benchmark_diff_normal.png"))
    visualize_difficulty_improvement(all_stats, medium_afs, "Medium", str(RESULTS_DIR / "benchmark_diff_medium.png"))
    visualize_difficulty_improvement(all_stats, hard_afs, "Hard", str(RESULTS_DIR / "benchmark_diff_hard.png"))
    visualize_difficulty_improvement(all_stats, all_afs_combined, "All", str(RESULTS_DIR / "benchmark_diff_all.png"))

    # 统计显著性检验
    sig_all = run_significance_tests(all_stats, all_afs_combined, "All")
    print_significance_tests(sig_all, "All")

    # 综合结果表
    generate_results_table(all_stats, normal_afs, medium_afs, hard_afs, sig_all, save_dir=str(RESULTS_DIR))

    export_csv(all_stats)
    elapsed = time.perf_counter() - t0
    log(f"Router Benchmark 完成: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def run_pipeline_benchmark():
    """运行 Pipeline Benchmark。"""
    log("=" * 70)
    log("PART 2: Pipeline Benchmark")
    log("=" * 70)

    from tests.benchmark_pipeline import (
        load_benchmark_airfoils,
        run_pipeline_benchmark as run_pipeline,
        visualize_by_category,
        visualize_summary,
        export_csv,
    )

    normal_afs, medium_afs, hard_afs = load_benchmark_airfoils()
    all_afs = normal_afs + medium_afs + hard_afs
    log(f"翼型: normal={len(normal_afs)} medium={len(medium_afs)} hard={len(hard_afs)}")

    t0 = time.perf_counter()

    results = run_pipeline(all_afs)

    visualize_by_category(results, normal_afs, "Normal", str(RESULTS_DIR / "pipeline_normal.png"))
    visualize_by_category(results, medium_afs, "Medium", str(RESULTS_DIR / "pipeline_medium.png"))
    visualize_by_category(results, hard_afs, "Hard", str(RESULTS_DIR / "pipeline_hard.png"))
    visualize_summary(results, normal_afs, medium_afs, hard_afs)
    export_csv(results, str(RESULTS_DIR / "pipeline_benchmark.csv"))

    elapsed = time.perf_counter() - t0
    log(f"Pipeline Benchmark 完成: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def run_ablation_study():
    """运行消融实验。"""
    log("=" * 70)
    log("PART 3: Ablation Study")
    log("=" * 70)

    from tests.benchmark_ablation import (
        load_benchmark,
        run_ablation_1,
        run_ablation_2,
        run_ablation_3,
        run_ablation_4,
        run_sensitivity_analysis,
        visualize_ablation_1,
        visualize_ablation_2,
        visualize_ablation_3,
        visualize_ablation_4,
        visualize_sensitivity,
        export_csv,
        print_final_summary,
    )

    normal, medium, hard = load_benchmark()
    all_airfoils = normal + medium + hard
    log(f"翼型: {len(all_airfoils)} (normal={len(normal)} medium={len(medium)} hard={len(hard)})")

    t0 = time.perf_counter()
    all_results = []

    r1 = run_ablation_1(all_airfoils)
    all_results.extend(r1)
    visualize_ablation_1(r1, all_airfoils)

    r2 = run_ablation_2(all_airfoils)
    all_results.extend(r2)
    visualize_ablation_2(r2, all_airfoils)

    r3 = run_ablation_3(all_airfoils)
    all_results.extend(r3)
    visualize_ablation_3(r3, all_airfoils)

    r4 = run_ablation_4(all_airfoils)
    all_results.extend(r4)
    visualize_ablation_4(r4, all_airfoils)

    rs = run_sensitivity_analysis()
    all_results.extend(rs)
    visualize_sensitivity(rs)

    export_csv(all_results)
    print_final_summary(all_results)

    elapsed = time.perf_counter() - t0
    log(f"Ablation Study 完成: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("PIERN-Airfoil 全量 Benchmark")
    log(f"输出目录: {RESULTS_DIR}")
    log(f"Python: {sys.version}")
    log("")

    t_total = time.perf_counter()

    try:
        run_router_benchmark()
    except Exception as e:
        log(f"Router Benchmark 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        run_pipeline_benchmark()
    except Exception as e:
        log(f"Pipeline Benchmark 失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        run_ablation_study()
    except Exception as e:
        log(f"Ablation Study 失败: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.perf_counter() - t_total
    log("")
    log("=" * 70)
    log(f"全部完成! 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"结果目录: {RESULTS_DIR}")
    log("打包命令: tar czf results.tar.gz results/")
    log("=" * 70)


if __name__ == "__main__":
    main()
