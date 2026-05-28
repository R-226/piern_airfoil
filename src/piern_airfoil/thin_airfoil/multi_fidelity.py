"""
Multi-fidelity airfoil optimization orchestrator.

Workflow:
  Stage 1: L-BFGS-B with NeuralFoil xxsmall (fast, explores CL targets)
  Stage 2: IPOPT with NeuralFoil large (accurate, enforces constraints)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .constraints import AirfoilConstraints, FidelityLevel
from .gradient_optimizer import GradientOptConfig, GradientOptResult, optimize_with_lbfgsb

if TYPE_CHECKING:
    import aerosandbox as asb


@dataclass
class MultiFidelityResult:
    """Result from multi-fidelity optimization."""
    airfoil: object  # asb.KulfanAirfoil
    stage1_result: Optional[GradientOptResult] = None
    stage1_nfev: int = 0
    stage2_nfev: int = 0
    stage2_iterations: int = 0


def multi_fidelity_optimize(
    initial_airfoil: "asb.KulfanAirfoil",
    constraints: AirfoilConstraints,
    alpha: float = 5.0,
    Re: float = 500e3,
    mach: float = 0.03,
    stage1_config: Optional[GradientOptConfig] = None,
    neural_max_iterations: int = 3,
) -> MultiFidelityResult:
    """
    Run multi-fidelity airfoil optimization.

    Stage 1 uses L-BFGS-B with NeuralFoil xxsmall to find a feasible
    solution that satisfies all CL targets with small constraint violations.
    Stage 2 takes this warm-start and refines it using IPOPT with NeuralFoil
    large for accurate constraint enforcement and CD minimization.

    Args:
        initial_airfoil: Starting airfoil (asb.KulfanAirfoil).
        constraints: Unified constraint specification.
        alpha: Angle of attack in degrees.
        Re: Reynolds number(s). Can be scalar or array matching CL_targets.
        mach: Mach number.
        stage1_config: L-BFGS-B configuration for Stage 1.
        neural_max_iterations: Number of NeuralOptimizer.update() calls in Stage 2.

    Returns:
        MultiFidelityResult with the optimized airfoil and statistics.
    """
    if stage1_config is None:
        stage1_config = GradientOptConfig(
            model_size="xxsmall",
            maxiter=500,
            maxfun=10000,
            cl_penalty_scale=5000.0,
        )

    # --- Stage 1: L-BFGS-B with NeuralFoil xxsmall ---
    stage1_result = optimize_with_lbfgsb(
        airfoil=initial_airfoil,
        constraints=constraints,
        alpha=alpha,
        Re=Re,
        mach=mach,
        config=stage1_config,
    )

    # --- Stage 2: IPOPT with NeuralFoil large ---
    from ..neuralfoil import NeuralOptimizer

    cl_targets = constraints.CL_targets
    cl_weights = constraints.CL_weights
    re_array = np.atleast_1d(Re).astype(float)

    neural_optimizer = NeuralOptimizer(
        airfoil=stage1_result.airfoil,
        CL_targets=cl_targets,
        CL_weights=cl_weights,
        RE=re_array,
        mach=mach,
    )

    for _ in range(neural_max_iterations):
        neural_optimizer.update()

    return MultiFidelityResult(
        airfoil=neural_optimizer.airfoil,
        stage1_result=stage1_result,
        stage1_nfev=stage1_result.nfev,
        stage2_nfev=neural_max_iterations,
        stage2_iterations=neural_max_iterations,
    )
