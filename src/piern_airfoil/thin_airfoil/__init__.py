"""
Thin airfoil module: fast approximate analysis + global optimization + multi-fidelity workflow.
"""

from .constraints import AirfoilConstraints, FidelityLevel
from .global_optimizer import GlobalAirfoilOptimizer, OptimizerConfig, OptimizationResult
from .multi_fidelity import MultiFidelityResult, multi_fidelity_optimize
from .thin_airfoil_solver import ThinAirfoilResult, thin_airfoil_analysis, thin_airfoil_from_kulfan

__all__ = [
    "AirfoilConstraints",
    "FidelityLevel",
    "GlobalAirfoilOptimizer",
    "MultiFidelityResult",
    "OptimizationResult",
    "OptimizerConfig",
    "ThinAirfoilResult",
    "multi_fidelity_optimize",
    "thin_airfoil_analysis",
    "thin_airfoil_from_kulfan",
]
