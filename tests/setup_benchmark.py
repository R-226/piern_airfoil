"""
固定 Benchmark 翼型集 — 一次性生成，后续 benchmark 读取此文件。

流程:
  1. 从 airfoil_screening.csv 加载所有翼型
  2. 用 brentq 评估初始 CD (非优化后的 CD)
  3. 过滤掉初始 CD < 0.075 的 (已经优于优化目标，无需优化)
  4. Normal 采样 30 个, Medium/Hard 全量
  5. 保存到 data/benchmark_airfoils.json
  6. 预渲染图片到 data/benchmark_images/

输出:
  data/benchmark_airfoils.json — 固定翼型列表 + 初始 CD
  data/benchmark_images/*.png  — 预渲染的翼型图片
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import aerosandbox as asb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 问题定义 (与其他 benchmark 一致) ──────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03

SCREENING_CSV = Path(__file__).parent.parent / "results" / "airfoil_screening.csv"
BENCHMARK_JSON = Path(__file__).parent.parent / "data" / "benchmark_airfoils.json"
IMAGES_DIR = Path(__file__).parent.parent / "data" / "benchmark_images"
NORMAL_SAMPLE_SIZE = 30
CD_THRESHOLD = 0.075  # 低于此值的翼型不参与 benchmark


# ── Brentq CD 评估 ────────────────────────────────────────────────────


def brentq_cd(airfoil) -> float:
    """用 brentq 评估翼型初始加权 CD (固定形状, 不优化)。"""
    from scipy.optimize import brentq

    cd_values = []
    for cl_t, re_i in zip(CL_TARGETS, RE):

        def residual(a, _af=airfoil, _re=re_i, _cl=cl_t):
            aero = _af.get_aero_from_neuralfoil(
                alpha=a, Re=float(_re), mach=MACH
            )
            return float(np.asarray(aero["CL"]).flatten()[0]) - _cl

        try:
            alpha_i = brentq(residual, -5, 18, xtol=0.01, maxiter=30)
        except (ValueError, RuntimeError):
            alpha_i = 5.0

        aero = airfoil.get_aero_from_neuralfoil(
            alpha=alpha_i, Re=float(re_i), mach=MACH
        )
        cd_values.append(float(np.asarray(aero["CD"]).flatten()[0]))

    return float(np.mean(np.array(cd_values) * CL_WEIGHTS))


# ── 图片渲染 ──────────────────────────────────────────────────────────


def render_airfoil_image(name: str, output_path: Path) -> None:
    """渲染翼型图片 (与 pipeline benchmark 一致)。"""
    af = asb.Airfoil(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    af.draw(show=False)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ── 主流程 ────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Setup Benchmark: 固定翼型集 + 预渲染图片")
    print("=" * 60)

    # 1. 加载 screening 结果
    with open(SCREENING_CSV) as f:
        rows = list(csv.DictReader(f))

    all_airfoils = []
    for cat in ["normal", "medium", "hard"]:
        cat_rows = sorted(
            [r for r in rows if r["category"] == cat],
            key=lambda r: float(r["cd"]),
        )
        for r in cat_rows:
            all_airfoils.append({"name": r["name"], "category": cat})

    print(f"Screening 总数: {len(all_airfoils)}")

    # 2. 计算 brentq 初始 CD
    print(f"\n计算 brentq 初始 CD (需要时间)...", flush=True)
    t0 = time.perf_counter()

    for i, af_info in enumerate(all_airfoils):
        name = af_info["name"]
        af = asb.KulfanAirfoil(name)
        cd = brentq_cd(af)
        af_info["initial_cd"] = round(cd, 6)
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(all_airfoils)}] {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"完成: {elapsed:.0f}s")

    # 3. 过滤
    filtered = [af for af in all_airfoils if af["initial_cd"] >= CD_THRESHOLD]
    print(f"\n过滤 (CD >= {CD_THRESHOLD}): {len(all_airfoils)} → {len(filtered)}")

    for cat in ["normal", "medium", "hard"]:
        cat_all = [af for af in all_airfoils if af["category"] == cat]
        cat_filtered = [af for af in filtered if af["category"] == cat]
        print(f"  {cat}: {len(cat_all)} → {len(cat_filtered)}")

    # 4. 分组 + 采样
    normal = sorted(
        [af for af in filtered if af["category"] == "normal"],
        key=lambda af: af["initial_cd"],
    )
    medium = sorted(
        [af for af in filtered if af["category"] == "medium"],
        key=lambda af: af["initial_cd"],
    )
    hard = sorted(
        [af for af in filtered if af["category"] == "hard"],
        key=lambda af: af["initial_cd"],
    )

    # Normal: 按 CD 分位数均匀采样
    if len(normal) > NORMAL_SAMPLE_SIZE:
        step = max(1, len(normal) // NORMAL_SAMPLE_SIZE)
        normal = normal[::step][:NORMAL_SAMPLE_SIZE]

    print(f"\n最终 benchmark 集:")
    print(f"  Normal: {len(normal)}")
    print(f"  Medium: {len(medium)}")
    print(f"  Hard:   {len(hard)}")
    print(f"  总计:   {len(normal) + len(medium) + len(hard)}")

    # 5. 保存 JSON
    benchmark = {
        "normal": [af["name"] for af in normal],
        "medium": [af["name"] for af in medium],
        "hard": [af["name"] for af in hard],
        "initial_cd": {
            af["name"]: af["initial_cd"]
            for af in normal + medium + hard
        },
        "config": {
            "cd_threshold": CD_THRESHOLD,
            "normal_sample_size": NORMAL_SAMPLE_SIZE,
            "cl_targets": CL_TARGETS.tolist(),
            "cl_weights": CL_WEIGHTS.tolist(),
        },
    }

    BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_JSON, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"\n保存: {BENCHMARK_JSON}")

    # 6. 预渲染图片
    all_names = [af["name"] for af in normal + medium + hard]
    print(f"\n预渲染 {len(all_names)} 张图片...", flush=True)
    t0 = time.perf_counter()

    for i, name in enumerate(all_names):
        img_path = IMAGES_DIR / f"{name}.png"
        if not img_path.exists():
            render_airfoil_image(name, img_path)
        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(all_names)}] {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"完成: {elapsed:.0f}s")
    print(f"图片目录: {IMAGES_DIR}")

    # 7. 汇总
    print(f"\n{'='*60}")
    print("Benchmark 固定翼型集已就绪:")
    print(f"  配置: {BENCHMARK_JSON}")
    print(f"  图片: {IMAGES_DIR}/")
    print(f"  Normal: {len(normal)} (CD range: {normal[0]['initial_cd']:.4f} ~ {normal[-1]['initial_cd']:.4f})")
    print(f"  Medium: {len(medium)} (CD range: {medium[0]['initial_cd']:.4f} ~ {medium[-1]['initial_cd']:.4f})")
    print(f"  Hard:   {len(hard)} (CD range: {hard[0]['initial_cd']:.4f} ~ {hard[-1]['initial_cd']:.4f})")


if __name__ == "__main__":
    main()
