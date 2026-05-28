"""
Routed multi-fidelity airfoil optimizer.

Uses a router to dynamically select optimization actions at each step,
rather than following a fixed two-stage pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .constraints import AirfoilConstraints, FidelityLevel
from .router import FidelityAction, OptimizationRouter, OptimizationState, N_ACTIONS

if TYPE_CHECKING:
    import aerosandbox as asb


@dataclass
class RoutedResult:
    """Result from routed multi-fidelity optimization."""
    airfoil: "asb.KulfanAirfoil"
    best_objective: float
    total_steps: int
    action_log: list[int]  # sequence of FidelityAction values used
    objective_history: list[float]
    total_evaluations: int
    total_time: float


class RoutedMultiFidelityOptimizer:
    """Routed multi-fidelity optimizer for KulfanAirfoil.

    At each step, the router selects an action (fidelity + optimizer + depth)
    based on the current optimization state. The action is executed, the state
    is updated, and the loop continues until the budget is exhausted.

    This replaces the fixed Stage1→Stage2 pipeline with a dynamic,
    state-dependent sequence of optimization steps.
    """

    def __init__(
        self,
        airfoil: "asb.KulfanAirfoil",
        constraints: AirfoilConstraints,
        max_steps: int = 10,
        router: OptimizationRouter | None = None,
        router_mode: str = "rule",
        Re: float | np.ndarray = 500e3,
        mach: float = 0.03,
    ):
        import aerosandbox as asb

        self.initial_airfoil = airfoil
        self.constraints = constraints
        self.max_steps = max_steps
        self.Re = Re
        self.mach = mach

        # Current best airfoil (mutable state)
        self.current_airfoil = airfoil
        self.best_objective = float("inf")

        # Router
        self.router = router or OptimizationRouter(mode=router_mode)

        # Optimization state
        self.state = OptimizationState()

        # Tracking
        self.action_log: list[int] = []
        self.objective_history: list[float] = []
        self.total_evaluations = 0

    def _evaluate(self, airfoil: "asb.KulfanAirfoil") -> tuple[float, float, dict]:
        """Evaluate airfoil with NeuralFoil at the highest fidelity.

        Returns:
            (weighted_cd, max_constraint_violation, aero_dict)
        """
        import aerosandbox as asb

        cl_targets = self.constraints.CL_targets
        cl_weights = self.constraints.CL_weights
        re_arr = np.atleast_1d(self.Re)

        total_cd = 0.0
        max_violation = 0.0
        aero_summary = {}

        if cl_targets is not None and len(cl_targets) > 0:
            from scipy.optimize import brentq

            for i, cl_t in enumerate(cl_targets):
                re_i = float(re_arr[min(i, len(re_arr) - 1)])

                # Find alpha for this CL target
                def residual(a):
                    try:
                        aero = airfoil.get_aero_from_neuralfoil(alpha=a, Re=re_i, mach=self.mach)
                        return float(np.asarray(aero["CL"]).flatten()[0]) - float(cl_t)
                    except Exception:
                        return 1e3

                try:
                    a_opt = brentq(residual, -3.0, 20.0, xtol=0.05, maxiter=20)
                except (ValueError, RuntimeError):
                    a_opt = 5.0  # fallback

                aero = airfoil.get_aero_from_neuralfoil(alpha=a_opt, Re=re_i, mach=self.mach)
                cd = float(np.asarray(aero["CD"]).flatten()[0])
                cl = float(np.asarray(aero["CL"]).flatten()[0])
                cm = float(np.asarray(aero["CM"]).flatten()[0])
                conf = float(np.asarray(aero["analysis_confidence"]).flatten()[0])

                total_cd += cd * float(cl_weights[i])

                # Track violations
                self._update_violations(airfoil, aero, cl_t, max_violation)
                max_violation = max(max_violation, self._constraint_violation(airfoil, aero, float(cl_t)))

                aero_summary = {"CL": cl, "CD": cd, "CM": cm, "confidence": conf}
        else:
            aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=float(re_arr[0]), mach=self.mach)
            cd = float(np.asarray(aero["CD"]).flatten()[0])
            conf = float(np.asarray(aero["analysis_confidence"]).flatten()[0])
            total_cd = cd
            max_violation = self._constraint_violation(airfoil, aero, None)
            aero_summary = {"CL": float(np.asarray(aero["CL"]).flatten()[0]),
                           "CD": cd,
                           "CM": float(np.asarray(aero["CM"]).flatten()[0]),
                           "confidence": conf}

        return total_cd, max_violation, aero_summary

    def _constraint_violation(self, airfoil, aero, cl_target) -> float:
        """Compute max constraint violation."""
        violations = self.constraints.evaluate_geometry(airfoil)
        violations.extend(self.constraints.evaluate_aero(aero, FidelityLevel.NEURAL, cl_target))
        return max(0, max(violations)) if violations else 0.0

    def _update_violations(self, airfoil, aero, cl_target, current_max):
        """Update state's constraint violation."""
        v = self._constraint_violation(airfoil, aero, cl_target)
        self.state.constraint_violation = max(self.state.constraint_violation, v)

    def _run_action(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Execute one optimization action.

        Returns:
            Optimized airfoil.
        """
        if action.is_de:
            return self._run_de(action)
        elif action.is_lbfgsb:
            return self._run_lbfgsb(action)
        else:
            return self._run_neuralfoil(action)

    def _run_de(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Run differential evolution with thin airfoil theory."""
        import aerosandbox as asb
        from .global_optimizer import GlobalAirfoilOptimizer, OptimizerConfig

        if action == FidelityAction.TAT_DE_SHALLOW:
            config = OptimizerConfig(maxiter=20, popsize=5, seed=None)
        else:
            config = OptimizerConfig(maxiter=80, popsize=15, seed=None)

        # For DE, use thin airfoil theory (fast)
        # Simplified: evaluate at fixed alpha, optimize shape
        optimizer = GlobalAirfoilOptimizer.for_kulfan_airfoil(
            airfoil=self.current_airfoil,
            constraints=self.constraints,
            alpha=5.0,
            Re=float(np.atleast_1d(self.Re)[0]),
            mach=self.mach,
            fidelity=FidelityLevel.THIN,
            config=config,
        )

        result = optimizer.optimize()
        self.total_evaluations += result.nfev

        # Reconstruct airfoil from result
        n_upper = len(self.initial_airfoil.upper_weights)
        n_lower = len(self.initial_airfoil.lower_weights)
        x = result.x

        return asb.KulfanAirfoil(
            name="RoutedOpt",
            upper_weights=np.array(x[:n_upper]),
            lower_weights=np.array(x[n_upper:n_upper + n_lower]),
            leading_edge_weight=float(x[-1]),
            TE_thickness=0.0,
        )

    def _run_lbfgsb(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Run L-BFGS-B with NeuralFoil at specified model size."""
        from .gradient_optimizer import GradientOptConfig, optimize_with_lbfgsb

        model_size = action.model_size or "xxsmall"

        config = GradientOptConfig(
            model_size=model_size,
            maxiter=300,
            maxfun=5000,
            cl_penalty_scale=5000.0,
        )

        result = optimize_with_lbfgsb(
            airfoil=self.current_airfoil,
            constraints=self.constraints,
            alpha=5.0,
            Re=self.Re,
            mach=self.mach,
            config=config,
        )

        self.total_evaluations += result.nfev
        return result.airfoil

    def _run_neuralfoil(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Run NeuralFoil IPOPT with specified model size and iterations."""
        model_size = action.model_size
        n_iters = action.n_ipopt_iters

        # Build CL targets and Re for NeuralFoil
        cl_targets = self.constraints.CL_targets
        cl_weights = self.constraints.CL_weights
        re_arr = np.atleast_1d(self.Re)

        optimized = run_neuralfoil_optimization(
            airfoil=self.current_airfoil,
            cl_targets=cl_targets,
            cl_weights=cl_weights,
            Re=re_arr,
            mach=self.mach,
            model_size=model_size,
            n_iterations=n_iters,
            cm_min=self.constraints.CM_min,
            thickness_33_min=self.constraints.thickness_at_33_min,
            thickness_90_min=self.constraints.thickness_at_90_min,
            te_angle_min=self.constraints.TE_angle_min,
        )

        self.total_evaluations += n_iters  # approximate
        return optimized

    def step(self) -> bool:
        """Execute one optimization step.

        Returns:
            True if budget exhausted.
        """
        if self.state.step_count >= self.max_steps:
            return True

        # Router selects action
        action = self.router.select_action(
            self.state,
            deterministic=(self.state.step_count >= self.max_steps - 2),  # deterministic for last 2 steps
        )

        # Execute action
        new_airfoil = self._run_action(action)

        # Evaluate result
        obj, violation, aero = self._evaluate(new_airfoil)

        # Update state
        self.state.step_count += 1
        self.state.budget_used_ratio = self.state.step_count / self.max_steps
        self.state.last_action = int(action)
        self.state.action_history.append(int(action))
        self.state.objective_history.append(obj)
        self.state.constraint_violation = violation
        self.state.confidence = aero.get("confidence", 0.0)

        # Track best
        if obj < self.best_objective and violation < 0.01:
            self.best_objective = obj
            self.current_airfoil = new_airfoil
        elif obj < self.best_objective * 1.5:
            # Accept even if slightly worse, for diversity
            self.current_airfoil = new_airfoil

        # Log
        self.action_log.append(int(action))
        self.objective_history.append(obj)

        return self.state.step_count >= self.max_steps

    def optimize(self) -> RoutedResult:
        """Run the full routed optimization loop.

        Returns:
            RoutedResult with the best airfoil found.
        """
        import time
        t0 = time.perf_counter()

        # Initial evaluation
        obj, violation, aero = self._evaluate(self.current_airfoil)
        self.best_objective = obj
        self.state.best_objective = obj
        self.state.objective_history.append(obj)
        self.state.constraint_violation = violation
        self.objective_history.append(obj)

        # Main loop
        done = False
        while not done:
            done = self.step()

        elapsed = time.perf_counter() - t0

        return RoutedResult(
            airfoil=self.current_airfoil,
            best_objective=self.best_objective,
            total_steps=self.state.step_count,
            action_log=self.action_log,
            objective_history=self.objective_history,
            total_evaluations=self.total_evaluations,
            total_time=elapsed,
        )


def run_neuralfoil_optimization(
    airfoil: "asb.KulfanAirfoil",
    cl_targets: np.ndarray | None,
    cl_weights: np.ndarray | None,
    Re: np.ndarray,
    mach: float,
    model_size: str,
    n_iterations: int,
    cm_min: float = -0.133,
    thickness_33_min: float = 0.128,
    thickness_90_min: float = 0.014,
    te_angle_min: float = 6.03,
) -> "asb.KulfanAirfoil":
    """Run NeuralFoil optimization with specified model size.

    This wraps the NeuralOptimizer to support model_size and iterative warm-starting.
    """
    import aerosandbox as asb

    # Build the optimization problem using asb.Opti directly
    opti = asb.Opti()

    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=opti.variable(
            init_guess=airfoil.lower_weights,
            lower_bound=-0.5,
            upper_bound=0.25,
        ),
        upper_weights=opti.variable(
            init_guess=airfoil.upper_weights,
            lower_bound=-0.25,
            upper_bound=0.5,
        ),
        leading_edge_weight=opti.variable(
            init_guess=airfoil.leading_edge_weight,
            lower_bound=-1,
            upper_bound=1,
        ),
        TE_thickness=0,
    )

    if cl_targets is not None and len(cl_targets) > 0:
        import aerosandbox.numpy as asbnp
        alpha = opti.variable(
            init_guess=asbnp.degrees(cl_targets / (2 * asbnp.pi)),
            lower_bound=-3,
            upper_bound=20,
        )
    else:
        alpha = opti.variable(init_guess=5.0, lower_bound=-3, upper_bound=20)

    # NeuralFoil analysis with specified model size
    aero = optimized_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=mach,
        model_size=model_size,
    )

    # Constraints
    opti.subject_to([
        aero["analysis_confidence"] > 0.90,
        aero["CM"] >= cm_min,
        optimized_airfoil.local_thickness(x_over_c=0.33) >= thickness_33_min,
        optimized_airfoil.local_thickness(x_over_c=0.90) >= thickness_90_min,
        optimized_airfoil.TE_angle() >= te_angle_min,
        optimized_airfoil.lower_weights[0] < -0.05,
        optimized_airfoil.upper_weights[0] > 0.05,
        optimized_airfoil.local_thickness() > 0,
    ])

    if cl_targets is not None and len(cl_targets) > 0:
        opti.subject_to([aero["CL"] == cl_targets])
        if len(cl_targets) > 1:
            import aerosandbox.numpy as asbnp
            opti.subject_to(asbnp.diff(alpha) > 0)

    # Wiggliness constraint (use aerosandbox.numpy for CasADi compatibility)
    import aerosandbox.numpy as asbnp
    get_wiggliness = lambda af: sum(
        asbnp.sum(asbnp.diff(asbnp.diff(array)) ** 2)
        for array in [af.lower_weights, af.upper_weights]
    )
    opti.subject_to(get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(airfoil))

    # Objective
    import aerosandbox.numpy as asbnp
    if cl_weights is not None:
        opti.minimize(asbnp.mean(aero["CD"] * cl_weights))
    else:
        opti.minimize(asbnp.mean(aero["CD"]))

    # Solve with warm-starting
    last_sol = None
    for _ in range(max(n_iterations, 1)):
        if last_sol is not None:
            opti.set_initial_from_sol(last_sol, initialize_primals=True, initialize_duals=True)

        try:
            sol = opti.solve(behavior_on_failure="return_last")
            optimized_airfoil = sol(optimized_airfoil)
            last_sol = sol
        except Exception:
            break

    return optimized_airfoil
