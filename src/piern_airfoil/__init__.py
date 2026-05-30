"""
PIERN-Airfoil: Hierarchical CST airfoil optimization with adaptive fidelity routing.

Core components:
- NeuralOptimizer: CasADi+IPOPT with NeuralFoil (baseline gradient-based optimizer)
- AdaptiveHierarchicalOptimizer: Hierarchical CST parameterization — the key innovation,
  using parameterization dimension as the fidelity axis for multi-fidelity optimization
- evaluate_weighted_cd: Shared NeuralFoil CD evaluation utility

Example usage:
    import aerosandbox as asb
    from piern_airfoil import AdaptiveHierarchicalOptimizer
    from piern.router import OptRouter
    import numpy as np

    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5

    router = OptRouter.from_mlp()
    optimizer = AdaptiveHierarchicalOptimizer(
        CL_targets=CL_TARGETS,
        CL_weights=CL_WEIGHTS,
        Re=RE,
        mach=0.03,
        router=router,
    )
    airfoil = asb.KulfanAirfoil("naca0012")
    result = optimizer.optimize(airfoil)
"""

__version__ = "0.5.0"

from .optimizer import NeuralOptimizer
from .hierarchical import AdaptiveHierarchicalOptimizer
from .eval import evaluate_weighted_cd

__all__ = [
    "__version__",
    "NeuralOptimizer",
    "AdaptiveHierarchicalOptimizer",
    "evaluate_weighted_cd",
]
