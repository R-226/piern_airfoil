"""
Global optimization using scipy's differential_evolution (genetic algorithm).

Differential evolution is a stochastic global optimization algorithm that:
- Does not require gradients
- Handles bound constraints
- Good at avoiding local optima
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from .constraints import AirfoilConstraints, FidelityLevel

if TYPE_CHECKING:
    import aerosandbox as asb


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
        import aerosandbox as asb

        def objective(x):
            airfoil = asb.KulfanAirfoil(
                name="candidate",
                upper_weights=np.array(x[0:8]),
                lower_weights=np.array(x[8:16]),
                leading_edge_weight=x[16],
            )
            aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=500e3, mach=0.03)
            return -float(aero["CL"])  # maximize CL

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

    @classmethod
    def for_kulfan_airfoil(
        cls,
        airfoil: "asb.KulfanAirfoil",
        constraints: AirfoilConstraints,
        alpha: float = 5.0,
        Re: float = 500e3,
        mach: float = 0.03,
        fidelity: FidelityLevel = FidelityLevel.THIN,
        objective_fn: Optional[Callable[["asb.KulfanAirfoil", dict], float]] = None,
        config: Optional[OptimizerConfig] = None,
    ) -> "GlobalAirfoilOptimizer":
        """
        Create optimizer for KulfanAirfoil shape optimization.

        Automatically derives bounds and builds an objective function that
        reconstructs a KulfanAirfoil from the design vector, evaluates
        aerodynamics at the specified fidelity, and applies constraint penalties.

        Args:
            airfoil: Template airfoil for initial weight values.
            constraints: Unified constraint specification.
            alpha: Angle of attack in degrees.
            Re: Reynolds number.
            mach: Mach number.
            fidelity: THIN (~1ms eval) or NEURAL (~50-200ms eval).
            objective_fn: Custom (airfoil, aero) -> float. Defaults to weighted CD.
            config: DE optimizer configuration.

        Returns:
            Configured GlobalAirfoilOptimizer instance.
        """
        n_upper = len(airfoil.upper_weights)
        n_lower = len(airfoil.lower_weights)

        bounds = (
            [(-0.25, 0.5)] * n_upper
            + [(-0.5, 0.25)] * n_lower
            + [(-1.0, 1.0)]  # leading_edge_weight
        )

        if fidelity == FidelityLevel.THIN:
            from .thin_airfoil_solver import thin_airfoil_from_kulfan

            def solve(af: "asb.KulfanAirfoil") -> dict:
                result = thin_airfoil_from_kulfan(af, alpha=alpha, mach=mach)
                return {"CL": result.CL, "CD": result.CD, "CM": result.CM}
        else:
            def solve(af: "asb.KulfanAirfoil") -> dict:
                return af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=mach)

        default_obj = objective_fn is None

        def _objective(x: np.ndarray) -> float:
            try:
                af = __import__("aerosandbox", fromlist=["KulfanAirfoil"]).KulfanAirfoil(
                    name="candidate",
                    upper_weights=np.array(x[:n_upper]),
                    lower_weights=np.array(x[n_upper:n_upper + n_lower]),
                    leading_edge_weight=float(x[-1]),
                    TE_thickness=0.0,
                )

                aero = solve(af)

                if default_obj:
                    cd = float(np.asarray(aero["CD"]).flatten()[0])
                    if constraints.CL_weights is not None:
                        weights = np.mean(constraints.CL_weights)
                    else:
                        weights = 1.0
                    base = cd * weights
                else:
                    base = objective_fn(af, aero)

                penalty = constraints.penalty(af, aero, fidelity, CL_target=None)
                return base + penalty

            except Exception:
                return 1e6

        return cls(objective=_objective, bounds=bounds, config=config)

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
