"""XFoil + SciPy 优化器 — 经典基线方案。

用 Differential Evolution (差分进化) 作为全局优化器，
XFoil 作为目标函数评估器。

这是 NeuralFoil 论文中的经典对比基线：
  - 无梯度、黑箱优化
  - 每次迭代调用 XFoil (面板法+边界层)
  - 比 NeuralFoil+IPOPT 慢 10-100x

用法:
    from piern_airfoil.xfoil_optimizer import xfoil_optimize
    result = xfoil_optimize("naca0012", CL_targets, Re, CL_weights, mach)
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from piern_airfoil.xfoil_baseline import _run_xfoil_single, airfoil_to_dat

logger = logging.getLogger(__name__)


@dataclass
class XFOilOptResult:
    """XFoil 优化结果。"""
    airfoil_name: str
    final_cd: float
    initial_cd: float
    time_s: float
    n_evals: int
    success: bool
    optimized_airfoil: object | None = None  # asb.KulfanAirfoil


def _xfoil_objective(
    params: np.ndarray,
    CL_targets: np.ndarray,
    Re: np.ndarray,
    CL_weights: np.ndarray,
    mach: float,
) -> float:
    """XFoil 目标函数: 给定 CST 参数，返回加权 CD。

    用 Differential Evolution 优化此函数。
    """
    import aerosandbox as asb

    n = len(params) // 2
    upper_w = params[:n]
    lower_w = params[n:]
    le_weight = 0.5  # 固定 LE weight

    try:
        af = asb.KulfanAirfoil(
            name="optimizing",
            upper_weights=upper_w,
            lower_weights=lower_w,
            leading_edge_weight=le_weight,
            TE_thickness=0,
        )

        # 检查基本几何合法性
        coords = af.coordinates
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return 1e6

        with tempfile.TemporaryDirectory(prefix="xfoil_opt_") as tmpdir:
            dat_path = Path(tmpdir) / "opt.dat"
            airfoil_to_dat("opt", dat_path, coordinates=coords)

            cd_values = []
            for cl_t, re_i in zip(CL_targets, Re):
                cd = _run_xfoil_single(dat_path, cl_t, float(re_i), mach)
                if cd is not None and np.isfinite(cd):
                    cd_values.append(cd)
                else:
                    # XFoil 未收敛 → 高惩罚
                    return 1e4

            return float(np.mean(np.array(cd_values) * CL_weights))

    except FileNotFoundError:
        raise  # xfoil 缺失时向上抛出，不要静默
    except Exception:
        return 1e6


def xfoil_optimize(
    airfoil_name: str,
    CL_targets: np.ndarray,
    Re: np.ndarray,
    CL_weights: np.ndarray,
    mach: float = 0.03,
    maxiter: int = 30,
    popsize: int = 10,
) -> XFOilOptResult:
    """用 XFoil + Differential Evolution 优化翼型。

    Args:
        airfoil_name: 初始翼型名称 (用于获取初始参数)。
        CL_targets: 目标 CL 值。
        Re: 雷诺数。
        CL_weights: CL 权资。
        mach: 马赫数。
        maxiter: DE 最大迭代数。
        popsize: 种群大小。

    Returns:
        XFOilOptResult 包含最终 CD、耗时、成功状态。
    """
    import aerosandbox as asb
    from piern_airfoil.xfoil_baseline import xfoil_cd, XFOIL_BIN

    # 预检查: XFoil binary 是否存在
    if not Path(XFOIL_BIN).exists():
        raise FileNotFoundError(
            f"XFoil binary not found at {XFOIL_BIN}. "
            "Install XFoil or set the correct path in xfoil_baseline.XFOIL_BIN."
        )

    af_init = asb.KulfanAirfoil(airfoil_name)
    initial_cd = xfoil_cd(airfoil_name, CL_targets, Re, CL_weights, mach)

    # 参数范围: upper_weights[8] + lower_weights[8]
    n = 8
    bounds = [(-0.25, 0.5)] * n + [(-0.5, 0.25)] * n

    t0 = time.perf_counter()

    result = differential_evolution(
        _xfoil_objective,
        bounds=bounds,
        args=(CL_targets, Re, CL_weights, mach),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-4,
        seed=42,
        disp=False,
    )

    elapsed = time.perf_counter() - t0

    # 构造优化后的翼型
    upper_w = result.x[:n]
    lower_w = result.x[n:]
    optimized_af = asb.KulfanAirfoil(
        name=f"{airfoil_name}_xfoil_opt",
        upper_weights=upper_w,
        lower_weights=lower_w,
        leading_edge_weight=0.5,
        TE_thickness=0,
    )

    final_cd = result.fun
    n_evals = result.nfev

    return XFOilOptResult(
        airfoil_name=airfoil_name,
        final_cd=final_cd,
        initial_cd=initial_cd,
        time_s=elapsed,
        n_evals=n_evals,
        success=final_cd < 0.5 and np.isfinite(final_cd),
        optimized_airfoil=optimized_af,
    )


def main():
    """Quick test: optimize NACA 0012 with XFoil + DE."""
    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
    MACH = 0.03

    print("XFoil + Differential Evolution Optimizer")
    print(f"CL targets: {CL_TARGETS}")
    print(f"Max iterations: 20, Population: 8")
    print()

    result = xfoil_optimize(
        "naca0012", CL_TARGETS, RE, CL_WEIGHTS, MACH,
        maxiter=20, popsize=8,
    )

    print(f"Airfoil:      {result.airfoil_name}")
    print(f"Initial CD:   {result.initial_cd:.6f}")
    print(f"Final CD:     {result.final_cd:.6f}")
    print(f"Time:         {result.time_s:.1f}s")
    print(f"FEvals:       {result.n_evals}")
    print(f"Success:      {result.success}")

    if result.initial_cd > 0:
        imp = (result.initial_cd - result.final_cd) / result.initial_cd * 100
        print(f"Improvement:  {imp:+.1f}%")


if __name__ == "__main__":
    main()
