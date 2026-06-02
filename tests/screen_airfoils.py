"""
全量翼型筛选脚本 — 测试 2174 个翼型的优化器兼容性并分类。

优化策略:
  1. Phase 1: 快速 NeuralFoil 预筛 (~0.01s/airfoil)，排除明显不合格的翼型
  2. Phase 2: 多进程 IPOPT 优化，直接用 optimizer 内部 CD，无需 evaluate_cd

输出: results/airfoil_screening.csv
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import aerosandbox as asb

# Suppress aerosandbox optimization warnings in workers
warnings.filterwarnings("ignore", message="Optimization failed")

# ── 问题定义 (与 benchmark 一致) ──────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class ScreenResult:
    name: str
    cd: float
    category: str  # "normal", "medium", "hard", "failed"


# ── Phase 1: 快速预筛 ────────────────────────────────────────────────


def quick_prescreen(name: str) -> bool:
    """快速检查翼型是否值得优化 (NeuralFoil 评估, ~0.01s)。"""
    try:
        kaf = asb.KulfanAirfoil(name)
        coords = kaf.coordinates
        if len(coords) < 5:
            return False
        chord = coords[:, 0].max() - coords[:, 0].min()
        if chord < 0.5:
            return False
    except Exception:
        return False

    try:
        aero = kaf.get_aero_from_neuralfoil(alpha=5.0, Re=250e3, mach=MACH)
        cl = float(np.asarray(aero["CL"]).flatten()[0])
        conf = float(np.asarray(aero["analysis_confidence"]).flatten()[0])
        return cl > 0.3 and conf > 0.5
    except Exception:
        return False


# ── Phase 2: IPOPT 优化 (单进程 worker) ──────────────────────────────


def optimize_single(name: str) -> ScreenResult:
    """优化单个翼型，直接用 optimizer 内部 CD。"""
    try:
        kaf = asb.KulfanAirfoil(name)
    except Exception:
        return ScreenResult(name, float("inf"), "failed")

    try:
        from piern_airfoil.optimizer import NeuralOptimizer

        opt = NeuralOptimizer(
            airfoil=kaf,
            CL_targets=CL_TARGETS,
            CL_weights=CL_WEIGHTS,
            RE=RE,
            mach=MACH,
        )
        opt.update()
    except Exception:
        return ScreenResult(name, float("inf"), "failed")

    # 直接用 optimizer 内部 aero 数据计算目标值
    try:
        cd_values = np.asarray(opt.aero["CD"]).flatten()
        if len(cd_values) != len(CL_WEIGHTS):
            return ScreenResult(name, float("inf"), "failed")
        cd = float(np.mean(cd_values * CL_WEIGHTS))
    except Exception:
        return ScreenResult(name, float("inf"), "failed")

    if cd >= 10 or np.isnan(cd) or np.isinf(cd):
        return ScreenResult(name, float("inf"), "failed")

    if cd < 0.45:
        category = "normal"
    elif cd < 0.8:
        category = "medium"
    else:
        category = "hard"

    return ScreenResult(name, cd, category)


# ── 主流程 ────────────────────────────────────────────────────────────


def main():
    import multiprocessing

    # 获取所有翼型名
    asb_dir = Path(asb.__file__).parent
    dat_dir = asb_dir / "geometry" / "airfoil" / "airfoil_database"
    all_names = sorted([f.stem for f in dat_dir.glob("*.dat")])

    print(f"Total airfoils in database: {len(all_names)}")

    # ── Phase 1: 快速预筛 ──
    print("\nPhase 1: Quick pre-screening...")
    t1 = time.perf_counter()
    qualified = []
    prescreened_out = 0

    for i, name in enumerate(all_names):
        if quick_prescreen(name):
            qualified.append(name)
        else:
            prescreened_out += 1

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t1
            print(f"  [{i+1}/{len(all_names)}] qualified={len(qualified)} "
                  f"rejected={prescreened_out} ({elapsed:.1f}s)")

    elapsed1 = time.perf_counter() - t1
    print(f"  Pre-screening done: {len(qualified)} qualified, "
          f"{prescreened_out} rejected ({elapsed1:.1f}s)")

    # ── Phase 2: 多进程 IPOPT 优化 ──
    # Use 8 workers to avoid memory issues with too many IPOPT processes
    n_workers = min(8, max(1, multiprocessing.cpu_count() - 1))
    print(f"\nPhase 2: IPOPT optimization ({n_workers} workers)...")

    results: list[ScreenResult] = []
    t2 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(optimize_single, name): name for name in qualified}
        done_count = 0

        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result(timeout=120)
                if result.category != "failed":
                    results.append(result)
            except Exception:
                pass

            if done_count % 50 == 0:
                elapsed = time.perf_counter() - t2
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = len(qualified) - done_count
                eta = remaining / rate if rate > 0 else 0
                passed = len(results)
                normal = sum(1 for r in results if r.category == "normal")
                medium = sum(1 for r in results if r.category == "medium")
                hard = sum(1 for r in results if r.category == "hard")
                print(
                    f"  [{done_count}/{len(qualified)}] passed={passed} "
                    f"(normal={normal}, medium={medium}, hard={hard}) "
                    f"rate={rate:.1f}/s ETA={eta/60:.1f}min"
                )

    elapsed2 = time.perf_counter() - t2
    total_elapsed = time.perf_counter() - t1
    print(f"\nPhase 2 done: {len(results)}/{len(qualified)} passed ({elapsed2:.1f}s)")
    print(f"Total: {len(results)}/{len(all_names)} airfoils ({total_elapsed/60:.1f}min)")

    # ── 保存结果 ──
    results.sort(key=lambda x: x.cd)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "airfoil_screening.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "cd", "category"])
        writer.writeheader()
        for r in results:
            writer.writerow({"name": r.name, "cd": r.cd, "category": r.category})
    print(f"Saved to {csv_path}")

    # ── 统计 ──
    normal = [r for r in results if r.category == "normal"]
    medium = [r for r in results if r.category == "medium"]
    hard = [r for r in results if r.category == "hard"]

    print(f"\n{'='*60}")
    print(f"Normal (CD<0.45): {len(normal)}")
    print(f"Medium (0.45-0.8): {len(medium)}")
    print(f"Hard (>=0.8): {len(hard)}")

    if normal:
        print(f"\nNormal samples:")
        for r in normal[:20]:
            print(f"  {r.name:<20} CD={r.cd:.4f}")
        if len(normal) > 20:
            print(f"  ... ({len(normal)} total)")

    if medium:
        print(f"\nMedium samples:")
        for r in medium[:10]:
            print(f"  {r.name:<20} CD={r.cd:.4f}")

    if hard:
        print(f"\nHard samples:")
        for r in hard[:10]:
            print(f"  {r.name:<20} CD={r.cd:.4f}")


if __name__ == "__main__":
    main()
