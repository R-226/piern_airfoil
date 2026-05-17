"""
Benchmark: 同一翼型优化问题的三种方法对比
问题: HPA (Human-Powered Aircraft) 多点优化
  - 6个CL目标: [0.8, 1.0, 1.2, 1.4, 1.5, 1.6]
  - Re = 500e3 * (CL/1.25)^-0.5
  - Mach = 0.03
  - 最小化加权CD
  - 约束: CM >= -0.133, 厚度 >= 0.128 @33%, 厚度 >= 0.014 @90%
"""

import time
import numpy as np
import aerosandbox as asb

from piern_airfoil.neuralfoil import NeuralOptimizer
from piern_airfoil.thin_airfoil import (
    AirfoilConstraints,
    FidelityLevel,
    GlobalAirfoilOptimizer,
    MultiFidelityResult,
    OptimizerConfig,
    multi_fidelity_optimize,
    thin_airfoil_from_kulfan,
)


# --- 问题定义 ---
CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5, 6, 7, 8, 9, 10])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03
INITIAL_AIRFOIL = asb.KulfanAirfoil("naca0012")


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_aero(label: str, aero, indent=2):
    pad = " " * indent
    CL = np.asarray(aero["CL"]).flatten()
    CD = np.asarray(aero["CD"]).flatten()
    CM = np.asarray(aero["CM"]).flatten()
    print(f"{pad}{label}:")
    for i in range(len(CL)):
        ld = CL[i] / CD[i] if CD[i] > 0 else float("inf")
        print(f"{pad}  Point {i+1}: CL={CL[i]:.4f}  CD={CD[i]:.6f}  CM={CM[i]:.4f}  L/D={ld:.1f}")
    print(f"{pad}  Mean CD (weighted): {np.mean(CD * CL_WEIGHTS):.6f}")


def print_geometry(label: str, airfoil, indent=2):
    pad = " " * indent
    t33 = float(np.asarray(airfoil.local_thickness(x_over_c=0.33)).flatten()[0])
    t90 = float(np.asarray(airfoil.local_thickness(x_over_c=0.90)).flatten()[0])
    te = float(np.asarray(airfoil.TE_angle()).flatten()[0])
    print(f"{pad}{label}: thickness@33%={t33:.4f}  thickness@90%={t90:.4f}  TE_angle={te:.2f}")


# ============================================================
#  方法 1: 纯薄翼理论 + 全局优化 (DE)
# ============================================================
def run_thin_only():
    print_header("方法1: 薄翼理论 + 差分进化 (全局搜索)")

    constraints = AirfoilConstraints(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        thickness_at_33_min=0.128,
        thickness_at_90_min=0.014,
        TE_angle_min=6.03,
        CM_min=-0.133,
    )

    config = OptimizerConfig(maxiter=80, popsize=15, seed=42)

    optimizer = GlobalAirfoilOptimizer.for_kulfan_airfoil(
        airfoil=INITIAL_AIRFOIL,
        constraints=constraints,
        alpha=5.0,
        Re=500e3,
        mach=MACH,
        fidelity=FidelityLevel.THIN,
        config=config,
    )

    t0 = time.perf_counter()
    result = optimizer.optimize()
    elapsed = time.perf_counter() - t0

    # 用 NeuralFoil 评估最终结果的"真实"气动性能
    n_upper = len(INITIAL_AIRFOIL.upper_weights)
    n_lower = len(INITIAL_AIRFOIL.lower_weights)
    x = result.x
    final_airfoil = asb.KulfanAirfoil(
        name="ThinOnly",
        upper_weights=np.array(x[:n_upper]),
        lower_weights=np.array(x[n_upper:n_upper + n_lower]),
        leading_edge_weight=float(x[-1]),
        TE_thickness=0.0,
    )

    # 多工况NeuralFoil评估
    CL_list, CD_list, CM_list = [], [], []
    for cl_t, re in zip(CL_TARGETS, RE):
        # 简化: 固定alpha, 实际应找匹配CL的alpha
        aero = final_airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=re, mach=MACH)
        CL_list.append(float(np.asarray(aero["CL"]).flatten()[0]))
        CD_list.append(float(np.asarray(aero["CD"]).flatten()[0]))
        CM_list.append(float(np.asarray(aero["CM"]).flatten()[0]))

    aero_eval = {"CL": np.array(CL_list), "CD": np.array(CD_list), "CM": np.array(CM_list)}

    print(f"  耗时: {elapsed:.2f}s  |  函数评估: {result.nfev}  |  成功: {result.success}")
    print_aero("NeuralFoil评估", aero_eval)
    print_geometry("几何检查", final_airfoil)

    return final_airfoil, elapsed, result.nfev


# ============================================================
#  方法 2: 纯 NeuralFoil 梯度优化 (IPOPT)
# ============================================================
def run_neural_only():
    print_header("方法2: NeuralFoil + IPOPT (梯度优化)")

    t0 = time.perf_counter()
    optimizer = NeuralOptimizer(
        airfoil=INITIAL_AIRFOIL,
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        RE=RE,
        mach=MACH,
    )

    optimizer.update()
    elapsed = time.perf_counter() - t0

    aero = optimizer.aero
    print(f"  耗时: {elapsed:.2f}s")
    print_aero("优化结果", aero)
    print_geometry("几何检查", optimizer.airfoil)

    return optimizer.airfoil, elapsed, 1


# ============================================================
#  方法 3: 多保真度优化 (薄翼全局搜索 → NeuralFoil精修)
# ============================================================
def run_multi_fidelity():
    print_header("方法3: 多保真度 (薄翼全局 → NeuralFoil精修)")

    constraints = AirfoilConstraints(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        thickness_at_33_min=0.128,
        thickness_at_90_min=0.014,
        TE_angle_min=6.03,
        CM_min=-0.133,
    )

    thin_config = OptimizerConfig(maxiter=50, popsize=10, seed=42)

    t0 = time.perf_counter()
    result = multi_fidelity_optimize(
        initial_airfoil=INITIAL_AIRFOIL,
        constraints=constraints,
        alpha=5.0,
        Re=500e3,
        mach=MACH,
        thin_config=thin_config,
        neural_max_iterations=3,
    )
    elapsed = time.perf_counter() - t0

    # 用NeuralFoil评估最终结果
    CL_list, CD_list, CM_list = [], [], []
    for cl_t, re in zip(CL_TARGETS, RE):
        aero = result.airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=re, mach=MACH)
        CL_list.append(float(np.asarray(aero["CL"]).flatten()[0]))
        CD_list.append(float(np.asarray(aero["CD"]).flatten()[0]))
        CM_list.append(float(np.asarray(aero["CM"]).flatten()[0]))

    aero_eval = {"CL": np.array(CL_list), "CD": np.array(CD_list), "CM": np.array(CM_list)}

    print(f"  耗时: {elapsed:.2f}s  |  Stage1评估: {result.stage1_nfev}  |  Stage2迭代: {result.neural_iterations}")
    print_aero("NeuralFoil评估", aero_eval)
    print_geometry("几何检查", result.airfoil)

    return result.airfoil, elapsed, result.stage1_nfev


# ============================================================
#  综合对比
# ============================================================
def main():
    print("=" * 60)
    print("  翼型优化三方法对比: HPA多点优化问题")
    print(f"  初始翼型: NACA0012")
    print(f"  CL目标: {CL_TARGETS}")
    print(f"  Re: {RE.astype(int)}")
    print(f"  Mach: {MACH}")

    # 基准: 初始翼型性能
    print_header("基准: NACA0012 初始性能")
    CL_list, CD_list, CM_list = [], [], []
    for cl_t, re in zip(CL_TARGETS, RE):
        aero = INITIAL_AIRFOIL.get_aero_from_neuralfoil(alpha=5.0, Re=re, mach=MACH)
        CL_list.append(float(np.asarray(aero["CL"]).flatten()[0]))
        CD_list.append(float(np.asarray(aero["CD"]).flatten()[0]))
        CM_list.append(float(np.asarray(aero["CM"]).flatten()[0]))
    print_aero("NACA0012", {"CL": np.array(CL_list), "CD": np.array(CD_list), "CM": np.array(CM_list)})

    airfoil_thin, t_thin, nfev_thin = run_thin_only()
    airfoil_neural, t_neural, nfev_neural = run_neural_only()
    airfoil_multi, t_multi, nfev_multi = run_multi_fidelity()

    # 最终对比表
    print_header("综合对比")

    def eval_airfoil(af):
        CLs, CDs, CMs = [], [], []
        for re in RE:
            a = af.get_aero_from_neuralfoil(alpha=5.0, Re=re, mach=MACH)
            CLs.append(float(np.asarray(a["CL"]).flatten()[0]))
            CDs.append(float(np.asarray(a["CD"]).flatten()[0]))
            CMs.append(float(np.asarray(a["CM"]).flatten()[0]))
        return np.array(CLs), np.array(CDs), np.array(CMs)

    CL0, CD0, CM0 = eval_airfoil(INITIAL_AIRFOIL)
    CL1, CD1, CM1 = eval_airfoil(airfoil_thin)
    CL2, CD2, CM2 = eval_airfoil(airfoil_neural)
    CL3, CD3, CM3 = eval_airfoil(airfoil_multi)

    mean_cd0 = np.mean(CD0 * CL_WEIGHTS)
    mean_cd1 = np.mean(CD1 * CL_WEIGHTS)
    mean_cd2 = np.mean(CD2 * CL_WEIGHTS)
    mean_cd3 = np.mean(CD3 * CL_WEIGHTS)

    print(f"  {'方法':<20} {'加权CD':>10} {'改善%':>8} {'耗时':>8} {'评估次数':>8}")
    print(f"  {'-' * 56}")
    print(f"  {'NACA0012 基准':<20} {mean_cd0:>10.6f} {'--':>8} {'--':>8} {'--':>8}")
    print(f"  {'薄翼理论+DE':<20} {mean_cd1:>10.6f} {(1-mean_cd1/mean_cd0)*100:>7.1f}% {t_thin:>7.2f}s {nfev_thin:>8}")
    print(f"  {'NeuralFoil+IPOPT':<20} {mean_cd2:>10.6f} {(1-mean_cd2/mean_cd0)*100:>7.1f}% {t_neural:>7.2f}s {nfev_neural:>8}")
    print(f"  {'多保真度':<20} {mean_cd3:>10.6f} {(1-mean_cd3/mean_cd0)*100:>7.1f}% {t_multi:>7.2f}s {nfev_multi:>8}")


if __name__ == "__main__":
    main()
