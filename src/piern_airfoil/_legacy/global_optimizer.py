"""
Global optimization using scipy's differential_evolution (genetic algorithm).

Differential evolution is a stochastic global optimization algorithm that:
- Does not require gradients
- Handles bound constraints
- Good at avoiding local optima

The THIN fidelity level uses NeuralFoil xxsmall (~4ms/eval) instead of
classical thin airfoil theory. This provides physically accurate CD
predictions that correlate with the NEURAL fidelity model, so DE finds
airfoils that are good starting points for NeuralOptimizer (IPOPT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
from scipy.optimize import brentq, differential_evolution, OptimizeResult

from ..constraints import AirfoilConstraints, FidelityLevel

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


# NeuralFoil model sizes for each fidelity level
_NF_MODEL_SIZE = {
    FidelityLevel.THIN: "xxsmall",    # ~4ms, fast exploration
    FidelityLevel.NEURAL: "large",    # ~1ms (cached), accurate
}


def _analytical_alpha(cl_target: float, mach: float = 0.03) -> float:
    """Compute approximate alpha (degrees) for a CL target using thin airfoil theory.

    CL = 2*pi*alpha_rad (for symmetric airfoil at small alpha)
    alpha_rad = CL / (2*pi)
    alpha_deg = alpha_rad * 180 / pi

    Returns alpha in degrees (as expected by NeuralFoil).
    This is a reasonable approximation for DE ranking — the exact alpha
    will be found by brentq in the final evaluation.
    """
    if mach > 0 and mach < 1:
        pg_corr = 1 / np.sqrt(1 - mach**2)
    else:
        pg_corr = 1.0
    alpha_rad = cl_target / (2 * np.pi * pg_corr)
    return np.degrees(alpha_rad)


def _check_geometric_validity(af: "asb.KulfanAirfoil") -> bool:
    """Check that upper surface is above lower surface (no self-intersection)."""
    try:
        upper = af.upper_coordinates()
        lower = af.lower_coordinates()
        # Sample at a few chord positions
        for x_check in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]:
            y_u = np.interp(x_check, upper[:, 0], upper[:, 1])
            y_l = np.interp(x_check, lower[:, 0], lower[:, 1])
            if y_u < y_l:
                return False
        return True
    except Exception:
        return False


class GlobalAirfoilOptimizer:
    """
    Global optimizer for airfoil shape optimization.

    Uses differential evolution (genetic algorithm) to find globally
    optimal airfoil shapes, avoiding local optima that plague gradient-based
    methods like IPOPT.

    For THIN fidelity, uses NeuralFoil xxsmall (~4ms/eval) which provides
    physically accurate CD predictions that correlate well with the larger
    NeuralFoil models used by NeuralOptimizer. This ensures DE finds airfoils
    that are good warm-starts for IPOPT.

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
        Re: float | np.ndarray = 500e3,
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

        Fidelity levels:
          - THIN: NeuralFoil xxsmall (~4ms/eval). Fast, physically accurate CD.
            Uses analytical alpha for multi-point (no brentq during DE).
          - NEURAL: NeuralFoil large (~1ms cached, ~50ms uncached). High accuracy.
            Uses brentq for precise multi-point alpha matching.

        Args:
            airfoil: Template airfoil for initial weight values.
            constraints: Unified constraint specification.
            alpha: Fixed angle of attack (used only when CL_targets is None).
            Re: Reynolds number(s). Can be scalar or array matching CL_targets.
            mach: Mach number.
            fidelity: THIN (xxsmall) or NEURAL (large).
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

        has_targets = (
            constraints.CL_targets is not None
            and len(constraints.CL_targets) > 0
        )

        re_scalar = float(np.atleast_1d(Re)[0])
        model_size = _NF_MODEL_SIZE[fidelity]
        default_obj = objective_fn is None

        def _eval_aero(af: "asb.KulfanAirfoil", a: float) -> dict:
            """Evaluate aerodynamics with NeuralFoil at the configured model size."""
            aero = af.get_aero_from_neuralfoil(
                alpha=a, Re=re_scalar, mach=mach, model_size=model_size
            )
            return {
                "CL": float(np.asarray(aero["CL"]).flatten()[0]),
                "CD": float(np.asarray(aero["CD"]).flatten()[0]),
                "CM": float(np.asarray(aero["CM"]).flatten()[0]),
                "analysis_confidence": float(np.asarray(aero["analysis_confidence"]).flatten()[0]),
            }

        def _find_alpha_for_cl(af: "asb.KulfanAirfoil", cl_target: float) -> float | None:
            """Find alpha that achieves cl_target using Brent's method."""
            def cl_residual(a):
                try:
                    r = _eval_aero(af, a)
                    return r["CL"] - cl_target
                except Exception:
                    return 1e3
            try:
                return brentq(cl_residual, -3.0, 20.0, xtol=0.1, maxiter=10)
            except (ValueError, RuntimeError):
                return None

        # Brentq settings: aggressive (fast) for THIN during DE, precise for NEURAL
        if fidelity == FidelityLevel.THIN:
            _brentq_xtol = 0.5    # 0.5 degree tolerance — fast convergence
            _brentq_maxiter = 5   # ~3-5 iterations typical for near-linear CL-alpha
        else:
            _brentq_xtol = 0.05
            _brentq_maxiter = 30

        def _find_alpha_fast(af: "asb.KulfanAirfoil", cl_target: float) -> float | None:
            """Find alpha with aggressive brentq settings (for DE speed)."""
            def cl_residual(a):
                try:
                    r = _eval_aero(af, a)
                    return r["CL"] - cl_target
                except Exception:
                    return 1e3
            try:
                return brentq(cl_residual, -3.0, 20.0, xtol=_brentq_xtol, maxiter=_brentq_maxiter)
            except (ValueError, RuntimeError):
                return None

        def _objective(x: np.ndarray) -> float:
            try:
                asb_mod = __import__("aerosandbox", fromlist=["KulfanAirfoil"])
                af = asb_mod.KulfanAirfoil(
                    name="candidate",
                    upper_weights=np.array(x[:n_upper]),
                    lower_weights=np.array(x[n_upper:n_upper + n_lower]),
                    leading_edge_weight=float(x[-1]),
                    TE_thickness=0.0,
                )

                # Geometric validity check: reject self-intersecting airfoils
                if not _check_geometric_validity(af):
                    return 1e6

                if has_targets:
                    total_cd = 0.0
                    re_arr = np.atleast_1d(Re)
                    weights = (
                        constraints.CL_weights
                        if constraints.CL_weights is not None
                        else np.ones(len(constraints.CL_targets))
                    )

                    for i, cl_t in enumerate(constraints.CL_targets):
                        # Find alpha for each CL target via brentq
                        a_i = _find_alpha_fast(af, float(cl_t))
                        if a_i is None:
                            # Can't achieve this CL target — large penalty
                            # but don't dominate the objective
                            total_cd += 0.1 * float(weights[i])
                            continue

                        aero = _eval_aero(af, a_i)
                        total_cd += aero["CD"] * float(weights[i])

                    return total_cd

                else:
                    # Single-point: evaluate at fixed alpha
                    aero = _eval_aero(af, alpha)

                    if default_obj:
                        cd = aero["CD"]
                        w = (
                            float(np.mean(constraints.CL_weights))
                            if constraints.CL_weights is not None
                            else 1.0
                        )
                        base = cd * w
                    else:
                        base = objective_fn(af, aero)

                    cl_t = (
                        float(constraints.CL_targets[0])
                        if constraints.CL_targets is not None
                        else None
                    )
                    penalty = constraints.penalty(af, aero, fidelity, CL_target=cl_t)
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
