"""
Optimization engine for airfoil design.

Provides a unified interface for various optimization algorithms.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from enum import Enum
import numpy as np

from ..analysis.base import AnalysisResult, FlowConditions
from ..parameterization.base import AirfoilGeometry, Parameterization


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    GRADIENT_DESCENT = "gradient_descent"
    LBFGS = "lbfgs"
    GENETIC = "genetic"
    COBYLA = "cobyla"  # Constrained optimization
    SLSQP = "slsqp"    # Sequential least squares


@dataclass
class OptimizationObjective:
    """
    Definition of an optimization objective.

    Can be a minimization or maximization target.
    """
    name: str
    target: Callable[[AnalysisResult], float]
    weight: float = 1.0
    is_minimization: bool = True

    def __call__(self, result: AnalysisResult) -> float:
        value = self.target(result)
        if self.is_minimization:
            return value * self.weight
        return -value * self.weight


@dataclass
class OptimizationConstraint:
    """
    Definition of an optimization constraint.

    Constraint is satisfied when value >= 0.
    """
    name: str
    target: Callable[[AnalysisResult], float]
    constraint_type: str = "ineq"  # "ineq" for >= 0, "eq" for == 0

    def __call__(self, result: AnalysisResult) -> float:
        value = self.target(result)
        if self.constraint_type == "eq":
            return value  # Equality: must be exactly 0
        return value  # Inequality: must be >= 0


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.LBFGS
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    step_size: float = 1.0
    population_size: int = 50  # For genetic algorithms
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    verbose: bool = True
    save_history: bool = True


class OptimizationEngine:
    """
    Unified optimization engine for airfoil design.

    Supports multiple optimization algorithms and constraints.
    """

    def __init__(
        self,
        parameterization: Parameterization,
        analyzer: Any,  # AnalysisResult generator
        config: Optional[OptimizationConfig] = None
    ):
        """
        Initialize optimization engine.

        Args:
            parameterization: Parameterization method for airfoil geometry
            analyzer: Analyzer that takes params and returns AnalysisResult
            config: Optimization configuration
        """
        self.parameterization = parameterization
        self.analyzer = analyzer
        self.config = config or OptimizationConfig()
        self.history = []

    def optimize(
        self,
        initial_params: np.ndarray,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        conditions: FlowConditions
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            initial_params: Starting parameters
            objectives: List of objectives to optimize
            constraints: List of constraints to satisfy
            conditions: Flow conditions for analysis

        Returns:
            Dictionary with optimization results
        """
        if self.config.algorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
            return self._optimize_gradient_descent(initial_params, objectives, constraints, conditions)
        elif self.config.algorithm == OptimizationAlgorithm.LBFGS:
            return self._optimize_lbfgs(initial_params, objectives, constraints, conditions)
        elif self.config.algorithm == OptimizationAlgorithm.GENETIC:
            return self._optimize_genetic(initial_params, objectives, constraints, conditions)
        else:
            return self._optimize_lbfgs(initial_params, objectives, constraints, conditions)

    def _objective_function(self, params: np.ndarray, objectives: List[OptimizationObjective],
                           constraints: List[OptimizationConstraint], conditions: FlowConditions) -> float:
        """Compute total objective value."""
        # Analyze current design
        geometry = self.parameterization.params_to_geometry(params)
        result = self.analyzer.analyze(geometry, conditions)

        # Compute objective value
        total_obj = sum(obj(result) for obj in objectives)

        # Compute penalty for constraints
        penalty = 0.0
        for constraint in constraints:
            constraint_value = constraint(result)
            if constraint_value < 0:
                penalty += (constraint_value ** 2) * 1000  # Quadratic penalty

        return total_obj + penalty

    def _optimize_gradient_descent(
        self,
        initial_params: np.ndarray,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        conditions: FlowConditions
    ) -> Dict[str, Any]:
        """Simple gradient descent optimization."""
        params = initial_params.copy()
        history = []

        lr = self.config.step_size

        for iteration in range(self.config.max_iterations):
            # Compute numerical gradient
            grad = np.zeros_like(params)
            eps = 1e-5

            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps

                f_plus = self._objective_function(params_plus, objectives, constraints, conditions)
                f_minus = self._objective_function(params_minus, objectives, constraints, conditions)
                grad[i] = (f_plus - f_minus) / (2 * eps)

            # Update
            params_new = params - lr * grad

            # Check validity
            is_valid, _ = self.parameterization.validate(params_new)
            if is_valid:
                params = params_new

            # Record history
            geometry = self.parameterization.params_to_geometry(params)
            result = self.analyzer.analyze(geometry, conditions)
            history.append({
                "iteration": iteration,
                "params": params.copy(),
                "objective": sum(obj(result) for obj in objectives),
                "result": result
            })

            if self.config.verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Obj={history[-1]['objective']:.4f}, CL={result.CL:.3f}, CD={result.CD:.5f}")

            # Check convergence
            if iteration > 0 and abs(history[-1]['objective'] - history[-2]['objective']) < self.config.convergence_tolerance:
                break

        return {
            "params": params,
            "history": history,
            "n_iterations": len(history)
        }

    def _optimize_lbfgs(
        self,
        initial_params: np.ndarray,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        conditions: FlowConditions
    ) -> Dict[str, Any]:
        """L-BFGS optimization (if scipy available)."""
        try:
            from scipy.optimize import minimize

            def objective_and_grad(x):
                # Simple gradient approximation for L-BFGS
                f = self._objective_function(x, objectives, constraints, conditions)

                # Numerical gradient
                eps = 1e-5
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    grad[i] = (self._objective_function(x_plus, objectives, constraints, conditions) - f) / eps

                return f, grad

            result = minimize(
                objective_and_grad,
                initial_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': self.config.max_iterations, 'disp': self.config.verbose}
            )

            # Build history from optimization
            history = [{
                "iteration": i,
                "params": initial_params,  # Simplified
                "objective": result.fun
            } for i in range(result.nit)]

            return {
                "params": result.x,
                "history": history,
                "n_iterations": result.nit,
                "success": result.success
            }

        except ImportError:
            # Fall back to gradient descent
            return self._optimize_gradient_descent(initial_params, objectives, constraints, conditions)

    def _optimize_genetic(
        self,
        initial_params: np.ndarray,
        objectives: List[OptimizationObjective],
        constraints: List[OptimizationConstraint],
        conditions: FlowConditions
    ) -> Dict[str, Any]:
        """Simple genetic algorithm optimization."""
        pop_size = self.config.population_size
        mutation_rate = self.config.mutation_rate
        crossover_rate = self.config.crossover_rate

        # Initialize population
        dim = len(initial_params)
        population = np.random.randn(pop_size, dim) * 0.1 + initial_params

        history = []

        for generation in range(self.config.max_iterations):
            # Evaluate fitness
            fitness = []
            for params in population:
                geometry = self.parameterization.params_to_geometry(params)
                result = self.analyzer.analyze(geometry, conditions)
                obj_value = sum(obj(result) for obj in objectives)

                # Penalty for constraint violations
                penalty = 0
                for constraint in constraints:
                    if constraint(result) < 0:
                        penalty += abs(constraint(result)) * 1000

                fitness.append(-obj_value - penalty)  # Negative because we minimize

            fitness = np.array(fitness)

            # Selection (tournament selection)
            new_population = [population[np.argmin(fitness)]]  # Keep best

            while len(new_population) < pop_size:
                # Tournament selection
                idx1, idx2 = np.random.randint(0, pop_size, 2)
                winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
                new_population.append(population[winner].copy())

            population = np.array(new_population)

            # Crossover
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < crossover_rate:
                    alpha = np.random.rand()
                    child1 = alpha * population[i] + (1 - alpha) * population[i + 1]
                    child2 = (1 - alpha) * population[i] + alpha * population[i + 1]
                    population[i] = child1
                    population[i + 1] = child2

            # Mutation
            for i in range(pop_size):
                if np.random.rand() < mutation_rate:
                    mutation = np.random.randn(dim) * 0.1
                    population[i] += mutation

            # Record best
            best_idx = np.argmin(fitness)
            best_params = population[best_idx]
            geometry = self.parameterization.params_to_geometry(best_params)
            result = self.analyzer.analyze(geometry, conditions)

            history.append({
                "generation": generation,
                "params": best_params.copy(),
                "objective": -fitness[best_idx],
                "result": result
            })

            if self.config.verbose and generation % 10 == 0:
                print(f"Gen {generation}: Obj={history[-1]['objective']:.4f}, CL={result.CL:.3f}, CD={result.CD:.5f}")

        return {
            "params": best_params,
            "history": history,
            "n_iterations": len(history)
        }
