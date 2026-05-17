"""
PIERN-Airfoil: Unified framework for automatic airfoil optimization.

Organized into two main modules:
- neuralfoil: Gradient-based optimization using Aerosandbox + NeuralFoil (NeuralOptimizer)
- thin_airfoil: Multi-fidelity optimization with thin airfoil theory + global search

Example usage:
    import aerosandbox as asb
    from piern_airfoil import NeuralOptimizer

    airfoil = asb.KulfanAirfoil("naca0012")
    optimizer = NeuralOptimizer(
        airfoil=airfoil,
        CL_targets=[1.0],
        CL_weights=[1.0],
        RE=[500e3],
        mach=0.03,
    )
    optimizer.update()
"""

__version__ = "0.3.0"

from .neuralfoil import NeuralOptimizer
from .thin_airfoil import (
    AirfoilConstraints,
    FidelityLevel,
    GlobalAirfoilOptimizer,
    OptimizerConfig,
    multi_fidelity_optimize,
    thin_airfoil_from_kulfan,
)

__all__ = [
    "__version__",
    "NeuralOptimizer",
    "AirfoilConstraints",
    "FidelityLevel",
    "GlobalAirfoilOptimizer",
    "OptimizerConfig",
    "multi_fidelity_optimize",
    "thin_airfoil_from_kulfan",
]
