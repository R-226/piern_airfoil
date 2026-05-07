"""
Global optimization using scipy's differential_evolution (genetic algorithm).

Based on the plan: Use NeuralFoil as the solver + global optimization algorithm.
Differential evolution is a stochastic global optimization algorithm that:
- Does not require gradients
- Handles bound constraints
- Good at avoiding local optima

Speed: ~1-5 seconds per evaluation (depends on problem complexity)
Accuracy: Global search capability vs local methods like IPOPT
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult


@dataclass
class OptimizationResult:
    """Result from global optimization."""
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nfev: int
    nit: int


@dataclass
class OptimizerConfig:
    """Configuration for global optimizer."""
    maxiter: int = 100
    popsize: int = 10
    tol: float = 1e-7
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    seed: Optional[int] = 42
    workers: int = 1


class GlobalAirfoilOptimizer:
    """
    Global optimizer for airfoil shape optimization.

    Uses differential evolution (genetic algorithm) to find globally
    optimal airfoil shapes, avoiding local optima that plague gradient-based
    methods like IPOPT.

    Example:
        def objective(x):
            # x = [upper_weights[0:4], lower_weights[0:4], leading_edge_weight]
            airfoil = rebuild_airfoil(x)
            result = neuralfoil.analyze(airfoil)
            return -result.CL / result.CD  # maximize L/D

        optimizer = GlobalAirfoilOptimizer(objective, bounds=bounds)
        result = optimizer.optimize()
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        config: Optional[OptimizerConfig] = None,
    ):
        """
        Initialize global optimizer.

        Args:
            objective: Function to minimize. Takes ndarray x, returns scalar.
            bounds: List of (min, max) tuples for each variable.
            config: Optional OptimizerConfig for fine-tuning.
        """
        self.objective = objective
        self.bounds = bounds
        self.config = config or OptimizerConfig()
        self.result: Optional[OptimizeResult] = None

    def optimize(self) -> OptimizationResult:
        """
        Run global optimization.

        Returns:
            OptimizationResult with optimal x, fun value, and statistics.
        """
        self.result = differential_evolution(
            func=self.objective,
            bounds=self.bounds,
            maxiter=self.config.maxiter,
            popsize=self.config.popsize,
            tol=self.config.tol,
            mutation=self.config.mutation,
            recombination=self.config.recombination,
            seed=self.config.seed,
            workers=self.config.workers,
            disp=False,
        )

        return OptimizationResult(
            x=self.result.x,
            fun=self.result.fun,
            success=self.result.success,
            message=self.result.message,
            nfev=self.result.nfev,
            nit=self.result.nit,
        )

    def __repr__(self) -> str:
        return (
            f"GlobalAirfoilOptimizer("
            f"n_vars={len(self.bounds)}, "
            f"maxiter={self.config.maxiter}, "
            f"popsize={self.config.popsize})"
        )


if __name__ == "__main__":
    # Simple test: minimize Rosenbrock function
    def rosenbrock(x):
        x = np.asarray(x)
        return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

    bounds = [(-5, 5), (-5, 5)]
    optimizer = GlobalAirfoilOptimizer(rosenbrock, bounds)
    result = optimizer.optimize()

    print(f"Optimal x: {result.x}")
    print(f"Optimal f(x): {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Iterations: {result.nit}")
