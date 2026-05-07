"""
Optimization module for global airfoil shape optimization.

Uses differential evolution (genetic algorithm) to find globally
optimal airfoil shapes, avoiding local optima that plague gradient-based
methods like IPOPT.

Also includes thin airfoil theory for rapid approximate analysis.
"""

from .global_optimizer import GlobalAirfoilOptimizer, OptimizerConfig, OptimizationResult
from .thin_airfoil_solver import thin_airfoil_analysis, ThinAirfoilResult

__all__ = [
    "GlobalAirfoilOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "thin_airfoil_analysis",
    "ThinAirfoilResult",
]
