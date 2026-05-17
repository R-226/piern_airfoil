"""
Benchmark: 同一翼型优化问题的三种方法对比
问题: HPA (Human-Powered Aircraft) 多点优化
  - 6个CL目标: [0.8, 1.0, 1.2, 1.4, 1.5, 1.6]
  - Re = 500e3 * (CL/1.25)^-0.5
  - Mach = 0.03
  - 最小化加权CD
  - 约束: CM >= -0.133, 厚度 >= 0.128 @33%, 厚度 >= 0.014 @90%

公平评估: 对每个CL目标用brentq搜索匹配alpha，计算真实CD。
稳定性分析: 多seed + 多初始翼型。
"""

import time
import numpy as np
import aerosandbox as asb
from scipy.optimize import brentq

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


# --- 公平评估工具 ---
def find_alpha_for_cl(airfoil, cl_target, re, mach, solver="neuralfoil"):
    """用brentq搜索匹配CL目标的alpha。"""
    def residual(alpha):
        try:
            if solver == "neuralfoil":
                aero = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=re, mach=mach)
                cl = float(np.asarray(aero["CL"]).flatten()[0])
            else:
                result = thin_airfoil_from_kulfan(airfoil, alpha=alpha, mach=mach)
                cl = result.CL
            return cl - cl_target
        except Exception:
            return 1e3
    try:
        return brentq(residual, -3.0, 20.0, xtol=0.05, maxiter=30)
    except (ValueError, RuntimeError):
        return None


def eval_airfoil_multipoint(airfoil, cl_targets=CL_TARGETS, re_arr=RE, mach=MACH):
    """公平多点评估: 为每个CL目标搜索匹配alpha，计算CD。

    如果某CL目标不可达（brentq失败），则在alpha=5.0处评估并标记。
    """
    CLs, CDs, CMs, alphas, reached = [], [], [], [], []
    for cl_t, re in zip(cl_targets, re_arr):
        a_opt = find_alpha_for_cl(airfoil, cl_t, re, mach, solver="neuralfoil")
        if a_opt is None:
            # Fallback: evaluate at fixed alpha, mark as unreached
            aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=re, mach=mach)
            CLs.append(float(np.asarray(aero["CL"]).flatten()[0]))
            CDs.append(float(np.asarray(aero["CD"]).flatten()[0]))
            CMs.append(float(np.asarray(aero["CM"]).flatten()[0]))
            alphas.append(5.0)
            reached.append(False)
        else:
            aero = airfoil.get_aero_from_neuralfoil(alpha=a_opt, Re=re, mach=mach)
            CLs.append(float(np.asarray(aero["CL"]).flatten()[0]))
            CDs.append(float(np.asarray(aero["CD"]).flatten()[0]))
            CMs.append(float(np.asarray(aero["CM"]).flatten()[0]))
            alphas.append(a_opt)
            reached.append(True)
    return (
        np.array(CLs), np.array(CDs), np.array(CMs), np.array(alphas),
        np.array(reached),
    )


def weighted_cd(CDs, weights=None):
    """计算加权CD（忽略NaN）。"""
    if weights is None:
        weights = CL_WEIGHTS[:len(CDs)]
    weights = np.asarray(weights)
    mask = ~np.isnan(CDs)
    if not mask.any():
        return np.inf
    return float(np.mean(CDs[mask] * weights[mask]))


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_aero(label: str, CLs, CDs, CMs, alphas, reached=None, indent=2):
    pad = " " * indent
    print(f"{pad}{label}:")
    for i in range(len(CLs)):
        tag = "" if (reached is None or reached[i]) else " [UNREACHED]"
        ld = CLs[i] / CDs[i] if CDs[i] > 0 else float("inf")
        print(
            f"{pad}  Point {i+1}: CL*={CL_TARGETS[i]:.2f} a={alphas[i]:5.1f}° "
            f"CL={CLs[i]:.4f} CD={CDs[i]:.6f} CM={CMs[i]:.4f} L/D={ld:.1f}{tag}"
        )
    wcd = weighted_cd(CDs)
    n_reached = sum(reached) if reached is not None else len(CLs)
    print(f"{pad}  Weighted CD: {wcd:.6f}  ({n_reached}/{len(CLs)} targets reached)")


def print_geometry(label: str, airfoil, indent=2):
    pad = " " * indent
    t33 = float(np.asarray(airfoil.local_thickness(x_over_c=0.33)).flatten()[0])
    t90 = float(np.asarray(airfoil.local_thickness(x_over_c=0.90)).flatten()[0])
    te = float(np.asarray(airfoil.TE_angle()).flatten()[0])
    print(f"{pad}{label}: t@33%={t33:.4f}  t@90%={t90:.4f}  TE_angle={te:.2f}")


# ============================================================
#  方法 1: 纯薄翼理论 + 全局优化 (DE)
# ============================================================
def run_thin_only(initial_airfoil, seed=42):
    constraints = AirfoilConstraints(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        thickness_at_33_min=0.128,
        thickness_at_90_min=0.014,
        TE_angle_min=6.03,
        CM_min=-0.133,
    )
    config = OptimizerConfig(maxiter=80, popsize=15, seed=seed)

    optimizer = GlobalAirfoilOptimizer.for_kulfan_airfoil(
        airfoil=initial_airfoil,
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

    n_upper = len(initial_airfoil.upper_weights)
    n_lower = len(initial_airfoil.lower_weights)
    x = result.x
    final_airfoil = asb.KulfanAirfoil(
        name="ThinOnly",
        upper_weights=np.array(x[:n_upper]),
        lower_weights=np.array(x[n_upper:n_upper + n_lower]),
        leading_edge_weight=float(x[-1]),
        TE_thickness=0.0,
    )
    return final_airfoil, elapsed, result.nfev


# ============================================================
#  方法 2: 纯 NeuralFoil 梯度优化 (IPOPT)
# ============================================================
def run_neural_only(initial_airfoil):
    t0 = time.perf_counter()
    optimizer = NeuralOptimizer(
        airfoil=initial_airfoil,
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        RE=RE,
        mach=MACH,
    )
    optimizer.update()
    elapsed = time.perf_counter() - t0
    return optimizer.airfoil, elapsed, 1


# ============================================================
#  方法 3: 多保真度优化 (薄翼全局搜索 → NeuralFoil精修)
# ============================================================
def run_multi_fidelity(initial_airfoil, seed=42):
    constraints = AirfoilConstraints(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        thickness_at_33_min=0.128,
        thickness_at_90_min=0.014,
        TE_angle_min=6.03,
        CM_min=-0.133,
    )
    thin_config = OptimizerConfig(maxiter=50, popsize=10, seed=seed)

    t0 = time.perf_counter()
    result = multi_fidelity_optimize(
        initial_airfoil=initial_airfoil,
        constraints=constraints,
        alpha=5.0,
        Re=500e3,
        mach=MACH,
        thin_config=thin_config,
        neural_max_iterations=3,
    )
    elapsed = time.perf_counter() - t0
    return result.airfoil, elapsed, result.stage1_nfev


# ============================================================
#  单次完整对比
# ============================================================
def run_single_comparison(initial_airfoil, seed=42, label=""):
    """运行三种方法并返回结果字典。"""
    print_header(f"初始翼型: {label}")

    # 基准
    CL0, CD0, CM0, A0, R0 = eval_airfoil_multipoint(initial_airfoil)
    wcd0 = weighted_cd(CD0)
    print_aero("基准", CL0, CD0, CM0, A0, R0)
    print_geometry("基准几何", initial_airfoil)

    results = {"baseline": wcd0}

    # 方法1: 薄翼+DE
    print("\n  [1] 薄翼理论 + DE ...")
    af1, t1, nfev1 = run_thin_only(initial_airfoil, seed=seed)
    CL1, CD1, CM1, A1, R1 = eval_airfoil_multipoint(af1)
    wcd1 = weighted_cd(CD1)
    print_aero("薄翼+DE", CL1, CD1, CM1, A1, R1)
    print_geometry("几何", af1)
    print(f"  耗时: {t1:.2f}s  评估: {nfev1}")
    results["thin"] = {"wcd": wcd1, "time": t1, "nfev": nfev1, "airfoil": af1}

    # 方法2: NeuralFoil+IPOPT
    print("\n  [2] NeuralFoil + IPOPT ...")
    af2, t2, nfev2 = run_neural_only(initial_airfoil)
    CL2, CD2, CM2, A2, R2 = eval_airfoil_multipoint(af2)
    wcd2 = weighted_cd(CD2)
    print_aero("NeuralFoil+IPOPT", CL2, CD2, CM2, A2, R2)
    print_geometry("几何", af2)
    print(f"  耗时: {t2:.2f}s")
    results["neural"] = {"wcd": wcd2, "time": t2, "nfev": nfev2, "airfoil": af2}

    # 方法3: 多保真度
    print("\n  [3] 多保真度 (薄翼→NeuralFoil) ...")
    af3, t3, nfev3 = run_multi_fidelity(initial_airfoil, seed=seed)
    CL3, CD3, CM3, A3, R3 = eval_airfoil_multipoint(af3)
    wcd3 = weighted_cd(CD3)
    print_aero("多保真度", CL3, CD3, CM3, A3, R3)
    print_geometry("几何", af3)
    print(f"  耗时: {t3:.2f}s  Stage1评估: {nfev3}")
    results["multi"] = {"wcd": wcd3, "time": t3, "nfev": nfev3, "airfoil": af3}

    return results


# ============================================================
#  主程序: 多seed + 多翼型稳定性分析
# ============================================================
def main():
    print("=" * 70)
    print("  翼型优化方法对比: HPA多点优化 (公平评估)")
    print(f"  CL目标: {CL_TARGETS}")
    print(f"  Re: {RE.astype(int)}")
    print(f"  Mach: {MACH}")
    print(f"  评估方式: 每个CL目标用brentq搜索匹配alpha")

    # --- 测试1: NACA0012, 多seed稳定性 ---
    SEEDS = [42, 123, 456, 789, 2024]
    airfoil_name = "naca0012"
    initial = asb.KulfanAirfoil(airfoil_name)

    print_header(f"稳定性分析: {airfoil_name}, {len(SEEDS)} seeds")
    all_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        r = run_single_comparison(initial, seed=seed, label=f"{airfoil_name} (seed={seed})")
        all_results.append(r)

    # 汇总稳定性
    print_header(f"稳定性汇总: {airfoil_name}")
    methods = ["thin", "neural", "multi"]
    method_names = {"thin": "薄翼+DE", "neural": "NeuralFoil+IPOPT", "multi": "多保真度"}
    for m in methods:
        wcds = [r[m]["wcd"] for r in all_results]
        times = [r[m]["time"] for r in all_results]
        print(
            f"  {method_names[m]:<20} "
            f"CD: {np.mean(wcds):.6f} ± {np.std(wcds):.6f}  "
            f"Time: {np.mean(times):.2f}s ± {np.std(times):.2f}s  "
            f"Range: [{np.min(wcds):.6f}, {np.max(wcds):.6f}]"
        )

    # --- 测试2: 不同初始翼型, 固定seed ---
    print_header("不同初始翼型排名对比 (seed=42)")
    airfoils = {
        "naca0012": asb.KulfanAirfoil("naca0012"),
        "naca2412": asb.KulfanAirfoil("naca2412"),
        "naca4412": asb.KulfanAirfoil("naca4412"),
        "clarky": asb.KulfanAirfoil("clarky"),
        "e387": asb.KulfanAirfoil("e387"),
    }

    rankings = {m: [] for m in methods}
    for name, af in airfoils.items():
        r = run_single_comparison(af, seed=42, label=name)
        wcds = {m: r[m]["wcd"] for m in methods}
        sorted_methods = sorted(wcds, key=lambda m: wcds[m])
        for rank, m in enumerate(sorted_methods):
            rankings[m].append(rank + 1)
        print(f"  {name} 排名: {', '.join(f'{method_names[m]}={wcds[m]:.6f}' for m in sorted_methods)}")

    print_header("排名统计")
    for m in methods:
        print(
            f"  {method_names[m]:<20} "
            f"平均排名: {np.mean(rankings[m]):.1f}  "
            f"排名: {rankings[m]}"
        )


if __name__ == "__main__":
    main()
