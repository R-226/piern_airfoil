"""
PIERN-Airfoil: Physics-Infused Expert Reasoning Network for Airfoil Optimization.

Core components:
- NeuralOptimizer: CasADi+IPOPT with NeuralFoil (baseline gradient-based optimizer)
- AdaptiveHierarchicalOptimizer: Hierarchical CST parameterization — the key innovation,
  using parameterization dimension as the fidelity axis for multi-fidelity optimization
- AirfoilConstraints: Unified constraint interface

Legacy components are in `_legacy/` — preserved for reference but not actively maintained.

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

__version__ = "0.4.0"

from .optimizer import NeuralOptimizer
from .hierarchical import AdaptiveHierarchicalOptimizer
from .constraints import AirfoilConstraints, FidelityLevel

__all__ = [
    "__version__",
    "NeuralOptimizer",
    "AdaptiveHierarchicalOptimizer",
    "AirfoilConstraints",
    "FidelityLevel",
]
