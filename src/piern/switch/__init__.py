"""
Switch: Routed multi-fidelity airfoil optimization.

Routes between different fidelity levels and optimization strategies
based on the current optimization state (conditional Markov decision).
"""

from .router import FidelityAction, OptimizationRouter, OptimizationState, N_ACTIONS
from .optimizer import RoutedOptimizer, RoutedResult

__all__ = [
    "FidelityAction",
    "OptimizationRouter",
    "OptimizationState",
    "RoutedOptimizer",
    "RoutedResult",
    "N_ACTIONS",
]
