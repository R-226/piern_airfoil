"""
Multi-fidelity airfoil optimization orchestrator.

Workflow:
  Stage 1: Global search with thin airfoil theory (~1ms per eval, fast exploration)
  Stage 2: Local refinement with NeuralFoil (~50-200ms per eval, accurate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .constraints import AirfoilConstraints, FidelityLevel
from .global_optimizer import GlobalAirfoilOptimizer, OptimizerConfig

if TYPE_CHECKING:
    import aerosandbox as asb


@dataclass
class MultiFidelityResult:
    """Result from multi-fidelity optimization."""
    airfoil: object  # asb.KulfanAirfoil
    thin_result: Optional[object] = None  # OptimizationResult from Stage 1
    neural_iterations: int = 0
    stage1_nfev: int = 0
    stage2_nfev: int = 0


def multi_fidelity_optimize(
    initial_airfoil: "asb.KulfanAirfoil",
    constraints: AirfoilConstraints,
    alpha: float = 5.0,
    Re: float = 500e3,
    mach: float = 0.03,
    thin_config: Optional[OptimizerConfig] = None,
    neural_max_iterations: int = 3,
) -> MultiFidelityResult:
    """
    Run multi-fidelity airfoil optimization.

    Stage 1 uses thin airfoil theory with differential evolution to globally
    explore the design space. Stage 2 takes the best candidate and refines it
    using NeuralFoil with gradient-based optimization.

    Args:
        initial_airfoil: Starting airfoil (asb.KulfanAirfoil).
        constraints: Unified constraint specification.
        alpha: Angle of attack in degrees.
        Re: Reynolds number.
        mach: Mach number.
        thin_config: DE configuration for Stage 1. Defaults to fast exploration.
        neural_max_iterations: Number of NeuralOptimizer.update() calls in Stage 2.

    Returns:
        MultiFidelityResult with the optimized airfoil and statistics.
    """
    import aerosandbox as asb

    if thin_config is None:
        thin_config = OptimizerConfig(maxiter=50, popsize=10, seed=42)

    # --- Stage 1: Global search with thin airfoil theory ---
    optimizer_s1 = GlobalAirfoilOptimizer.for_kulfan_airfoil(
        airfoil=initial_airfoil,
        constraints=constraints,
        alpha=alpha,
        Re=Re,
        mach=mach,
        fidelity=FidelityLevel.THIN,
        config=thin_config,
    )

    thin_result = optimizer_s1.optimize()

    # Reconstruct best airfoil from Stage 1
    n_upper = len(initial_airfoil.upper_weights)
    n_lower = len(initial_airfoil.lower_weights)
    x = thin_result.x

    stage1_airfoil = asb.KulfanAirfoil(
        name="Stage1",
        upper_weights=np.array(x[:n_upper]),
        lower_weights=np.array(x[n_upper:n_upper + n_lower]),
        leading_edge_weight=float(x[-1]),
        TE_thickness=0.0,
    )

    # --- Stage 2: Local refinement with NeuralFoil ---
    from ..neuralfoil import NeuralOptimizer

    cl_targets = constraints.CL_targets if constraints.CL_targets is not None else np.array([alpha / (180 / np.pi) * 2 * np.pi])
    cl_weights = constraints.CL_weights if constraints.CL_weights is not None else np.ones_like(cl_targets)
    re_array = np.full_like(cl_targets, Re) if not hasattr(Re, '__len__') else np.asarray(Re)

    neural_optimizer = NeuralOptimizer(
        airfoil=stage1_airfoil,
        CL_targets=cl_targets,
        CL_weights=cl_weights,
        RE=re_array,
        mach=mach,
    )

    for _ in range(neural_max_iterations):
        neural_optimizer.update()

    return MultiFidelityResult(
        airfoil=neural_optimizer.airfoil,
        thin_result=thin_result,
        neural_iterations=neural_max_iterations,
        stage1_nfev=thin_result.nfev,
        stage2_nfev=neural_max_iterations,
    )
