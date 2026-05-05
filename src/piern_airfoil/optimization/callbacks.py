"""Optimization callbacks for monitoring and visualization."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class OptimizationCallback(ABC):
    """Base class for optimization callbacks."""

    @abstractmethod
    def on_iteration(self, iteration: int, params: np.ndarray,
                     result: Any, objective_value: float):
        """Called at each iteration."""
        pass

    @abstractmethod
    def on_complete(self, best_params: np.ndarray, best_result: Any):
        """Called when optimization completes."""
        pass


class PrintCallback(OptimizationCallback):
    """Print progress to stdout."""

    def __init__(self, print_every: int = 10):
        self.print_every = print_every

    def on_iteration(self, iteration: int, params: np.ndarray,
                     result: Any, objective_value: float):
        if iteration % self.print_every == 0:
            print(f"Iter {iteration}: objective={objective_value:.4f}, "
                  f"CL={result.CL:.3f}, CD={result.CD:.5f}, "
                  f"confidence={result.confidence:.2f}")

    def on_complete(self, best_params: np.ndarray, best_result: Any):
        print(f"\nOptimization complete!")
        print(f"Best CL={best_result.CL:.3f}, CD={best_result.CD:.5f}")
        print(f"Best L/D={best_result.L/D:.1f}")


class HistoryCallback(OptimizationCallback):
    """Store optimization history."""

    def __init__(self):
        self.history = []

    def on_iteration(self, iteration: int, params: np.ndarray,
                     result: Any, objective_value: float):
        self.history.append({
            "iteration": iteration,
            "params": params.copy(),
            "CL": result.CL,
            "CD": result.CD,
            "CM": result.CM,
            "L_D": result.L/D,
            "confidence": result.confidence,
            "objective": objective_value
        })

    def on_complete(self, best_params: np.ndarray, best_result: Any):
        pass

    def get_history(self) -> List[Dict]:
        return self.history

    def get_best_iteration(self) -> Dict:
        """Return the iteration with best L/D."""
        if not self.history:
            return {}
        return min(self.history, key=lambda x: x.get('CD', float('inf')))


class ConvergenceCallback(OptimizationCallback):
    """Monitor convergence and stop if stalled."""

    def __init__(self, patience: int = 20, tolerance: float = 1e-6):
        self.patience = patience
        self.tolerance = tolerance
        self.best_value = float('inf')
        self.counter = 0

    def on_iteration(self, iteration: int, params: np.ndarray,
                     result: Any, objective_value: float):
        if objective_value < self.best_value - self.tolerance:
            self.best_value = objective_value
            self.counter = 0
        else:
            self.counter += 1

    def is_converged(self) -> bool:
        return self.counter >= self.patience
