"""
Test: Global optimization with NeuralFoil.

This demonstrates using differential_evolution (global optimizer)
with NeuralFoil (aerosandbox) for airfoil shape optimization.
"""

import numpy as np
import aerosandbox as asb
from src.piern_airfoil.thin_airfoil import GlobalAirfoilOptimizer, OptimizerConfig


def objective_naca_max_cl(x):
    """
    Objective: Maximize CL for NACA0012-like airfoil at alpha=5°.

    Variables:
    - x[0:8] = upper_weights (8 coefficients)
    - x[8:16] = lower_weights (8 coefficients)
    - x[16] = leading_edge_weight

    Returns: negative CL (since we minimize)
    """
    try:
        upper_weights = np.array(x[0:8])
        lower_weights = np.array(x[8:16])
        leading_edge_weight = float(x[16])

        airfoil = asb.KulfanAirfoil(
            name="Optimized",
            upper_weights=upper_weights,
            lower_weights=lower_weights,
            leading_edge_weight=leading_edge_weight,
            TE_thickness=0.0,
        )

        aero = airfoil.get_aero_from_neuralfoil(
            alpha=5.0,
            Re=500e3,
            mach=0.03,
        )

        return -float(np.asarray(aero["CL"]).flatten()[0])

    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e6


if __name__ == "__main__":
    print("Testing Global Optimization with NeuralFoil")
    print("=" * 50)

    # Bounds for NACA0012-like airfoil (8 coefficients per side + leading edge)
    bounds = [
        (-0.5, 0.5),   # upper_weights[0]
        (-0.3, 0.3),   # upper_weights[1]
        (-0.2, 0.2),   # upper_weights[2]
        (-0.1, 0.1),   # upper_weights[3]
        (-0.1, 0.1),   # upper_weights[4]
        (-0.1, 0.1),   # upper_weights[5]
        (-0.1, 0.1),   # upper_weights[6]
        (-0.1, 0.1),   # upper_weights[7]
        (-0.5, 0.25),  # lower_weights[0]
        (-0.3, 0.3),   # lower_weights[1]
        (-0.2, 0.2),   # lower_weights[2]
        (-0.1, 0.1),   # lower_weights[3]
        (-0.1, 0.1),   # lower_weights[4]
        (-0.1, 0.1),   # lower_weights[5]
        (-0.1, 0.1),   # lower_weights[6]
        (-0.1, 0.1),   # lower_weights[7]
        (-1.0, 1.0),   # leading_edge_weight
    ]

    # Initialize with NACA0012 parameters (8 coefficients per side)
    naca0012_upper = np.array([0.3, 0.23, 0.1, 0.07, 0.0, 0.0, 0.0, 0.0])
    naca0012_lower = np.array([-0.1, -0.14, -0.07, -0.05, 0.0, 0.0, 0.0, 0.0])
    naca0012_le = 0.0

    # Initial guess (centered around NACA0012)
    x0 = np.concatenate([
        naca0012_upper,
        naca0012_lower,
        [naca0012_le]
    ])

    print(f"Initial guess CL: {-float(objective_naca_max_cl(x0)):.4f}")

    config = OptimizerConfig(
        maxiter=20,
        popsize=5,
        tol=1e-3,
        seed=42,
    )

    optimizer = GlobalAirfoilOptimizer(
        objective=objective_naca_max_cl,
        bounds=bounds,
        config=config,
    )

    print(f"\nOptimizer: {optimizer}")
    print("Running optimization (this may take a minute)...\n")

    result = optimizer.optimize()

    print(f"\nOptimization Results:")
    print(f"  Optimal x: {result.x}")
    print(f"  Max CL: {-result.fun:.4f}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Iterations: {result.nit}")
