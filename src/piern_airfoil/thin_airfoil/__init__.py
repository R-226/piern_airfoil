"""
Thin airfoil module: fast approximate analysis + global optimization + multi-fidelity workflow.
"""

from .constraints import AirfoilConstraints, FidelityLevel
from .global_optimizer import GlobalAirfoilOptimizer, OptimizerConfig, OptimizationResult
from .multi_fidelity import MultiFidelityResult, multi_fidelity_optimize
from .router import FidelityAction, OptimizationRouter, OptimizationState
from .routed_optimizer import RoutedMultiFidelityOptimizer, RoutedResult
from .thin_airfoil_solver import ThinAirfoilResult, thin_airfoil_analysis, thin_airfoil_from_kulfan, thin_airfoil_multipoint_cd

__all__ = [
    "AirfoilConstraints",
    "FidelityAction",
    "FidelityLevel",
    "GlobalAirfoilOptimizer",
    "MultiFidelityResult",
    "OptimizationResult",
    "OptimizationRouter",
    "OptimizationState",
    "OptimizerConfig",
    "RoutedMultiFidelityOptimizer",
    "RoutedResult",
    "ThinAirfoilResult",
    "multi_fidelity_optimize",
    "thin_airfoil_analysis",
    "thin_airfoil_from_kulfan",
]
