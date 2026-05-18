"""
Routed multi-fidelity airfoil optimizer.

At each step, the router selects an optimization action based on the
current state. The action is executed, the state is updated, and the
loop continues until the budget is exhausted.

This replaces the fixed TAT->NeuralFoil pipeline with a dynamic,
state-dependent sequence of optimization steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import time
import numpy as np

from .router import FidelityAction, OptimizationRouter, OptimizationState, N_ACTIONS

if TYPE_CHECKING:
    import aerosandbox as asb


@dataclass
class RoutedResult:
    """Result from routed optimization."""
    airfoil: object  # asb.KulfanAirfoil
    best_objective: float
    total_steps: int
    action_log: list[int]
    objective_history: list[float]
    total_evaluations: int
    total_time: float


class RoutedOptimizer:
    """Routed multi-fidelity optimizer for KulfanAirfoil.

    At each step:
    1. Router selects an action (fidelity + optimizer + depth)
    2. Action is executed
    3. Result is evaluated with NeuralFoil (ground truth)
    4. State is updated
    5. Loop continues until budget exhausted
    """

    def __init__(
        self,
        airfoil: "asb.KulfanAirfoil",
        constraints,
        max_steps: int = 10,
        router: OptimizationRouter | None = None,
        Re: float | np.ndarray = 500e3,
        mach: float = 0.03,
    ):
        self.initial_airfoil = airfoil
        self.constraints = constraints
        self.max_steps = max_steps
        self.Re = Re
        self.mach = mach

        self.current_airfoil = airfoil
        self.best_objective = float("inf")

        self.router = router or OptimizationRouter(mode="rule")
        self.state = OptimizationState()

        self.action_log: list[int] = []
        self.objective_history: list[float] = []
        self.total_evaluations = 0

    def _evaluate_ground_truth(self, airfoil: "asb.KulfanAirfoil") -> tuple[float, float]:
        """Evaluate with NeuralFoil (ground truth).

        Returns:
            (weighted_cd, max_constraint_violation)
        """
        cl_targets = self.constraints.CL_targets
        cl_weights = self.constraints.CL_weights
        re_arr = np.atleast_1d(self.Re)

        total_cd = 0.0
        max_violation = 0.0

        if cl_targets is not None and len(cl_targets) > 0:
            from scipy.optimize import brentq

            for i, cl_t in enumerate(cl_targets):
                re_i = float(re_arr[min(i, len(re_arr) - 1)])

                def residual(a):
                    try:
                        aero = airfoil.get_aero_from_neuralfoil(alpha=a, Re=re_i, mach=self.mach)
                        return float(np.asarray(aero["CL"]).flatten()[0]) - float(cl_t)
                    except Exception:
                        return 1e3

                try:
                    a_opt = brentq(residual, -3.0, 20.0, xtol=0.05, maxiter=20)
                except (ValueError, RuntimeError):
                    a_opt = 5.0

                aero = airfoil.get_aero_from_neuralfoil(alpha=a_opt, Re=re_i, mach=self.mach)
                cd = float(np.asarray(aero["CD"]).flatten()[0])
                cl = float(np.asarray(aero["CL"]).flatten()[0])
                cm = float(np.asarray(aero["CM"]).flatten()[0])
                conf = float(np.asarray(aero["analysis_confidence"]).flatten()[0])

                total_cd += cd * float(cl_weights[i])

                # Constraint violation
                v = self._constraint_violation(airfoil, aero, float(cl_t))
                max_violation = max(max_violation, v)

                self.state.confidence = conf
        else:
            aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=float(re_arr[0]), mach=self.mach)
            cd = float(np.asarray(aero["CD"]).flatten()[0])
            total_cd = cd
            max_violation = self._constraint_violation(airfoil, aero, None)

        return total_cd, max_violation

    def _constraint_violation(self, airfoil, aero, cl_target) -> float:
        v = self.constraints.penalty(airfoil, aero, None, CL_target=cl_target, scale=1.0)
        return v

    def _run_action(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Execute one optimization action."""
        if action.is_de:
            return self._run_de(action)
        else:
            return self._run_neuralfoil(action)

    def _run_de(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Run differential evolution with thin airfoil theory."""
        import aerosandbox as asb
        import sys
        sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
        from piern_airfoil.thin_airfoil import (
            AirfoilConstraints, GlobalAirfoilOptimizer, OptimizerConfig, FidelityLevel,
        )

        if action == FidelityAction.TAT_DE_SHALLOW:
            config = OptimizerConfig(maxiter=20, popsize=5, seed=None)
        else:
            config = OptimizerConfig(maxiter=80, popsize=15, seed=None)

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

    def _run_neuralfoil(self, action: FidelityAction) -> "asb.KulfanAirfoil":
        """Run NeuralFoil IPOPT with specified model size."""
        import aerosandbox as asb
        import sys
        sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
        from piern_airfoil.thin_airfoil.routed_optimizer import run_neuralfoil_optimization

        cl_targets = self.constraints.CL_targets
        cl_weights = self.constraints.CL_weights
        re_arr = np.atleast_1d(self.Re)

        optimized = run_neuralfoil_optimization(
            airfoil=self.current_airfoil,
            cl_targets=cl_targets,
            cl_weights=cl_weights,
            Re=re_arr,
            mach=self.mach,
            model_size=action.model_size,
            n_iterations=action.n_ipopt_iters,
            cm_min=self.constraints.CM_min,
            thickness_33_min=self.constraints.thickness_at_33_min,
            thickness_90_min=self.constraints.thickness_at_90_min,
            te_angle_min=self.constraints.TE_angle_min,
        )

        self.total_evaluations += action.n_ipopt_iters
        return optimized

    def step(self) -> bool:
        """Execute one optimization step. Returns True if budget exhausted."""
        if self.state.step_count >= self.max_steps:
            return True

        # Router selects action
        action = self.router.select_action(
            self.state,
            deterministic=(self.state.step_count >= self.max_steps - 2),
        )

        # Execute action
        new_airfoil = self._run_action(action)

        # Evaluate result (ground truth)
        obj, violation = self._evaluate_ground_truth(new_airfoil)

        # Update state
        self.state.step_count += 1
        self.state.budget_used_ratio = self.state.step_count / self.max_steps
        self.state.last_action = int(action)
        self.state.action_history.append(int(action))
        self.state.objective_history.append(obj)
        self.state.constraint_violation = violation

        # Track best
        if obj < self.best_objective and violation < 0.01:
            self.best_objective = obj
            self.current_airfoil = new_airfoil
        elif obj < self.best_objective * 1.5:
            self.current_airfoil = new_airfoil

        self.action_log.append(int(action))
        self.objective_history.append(obj)

        return self.state.step_count >= self.max_steps

    def optimize(self) -> RoutedResult:
        """Run the full routed optimization loop."""
        t0 = time.perf_counter()

        # Initial evaluation
        obj, violation = self._evaluate_ground_truth(self.current_airfoil)
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
