"""
PIERN-Airfoil: Unified framework for automatic airfoil optimization.

Organized into three main modules:
- neuralfoil: Fast analysis and optimization using Aerosandbox + NeuralFoil
- optimization: Global optimization algorithms (CMA-ES, differential evolution)
- piern: Physics-informed reinforcement learning for intelligent optimization

Example usage:
    from piern_airfoil import NeuralFoilOptimizer

    optimizer = NeuralFoilOptimizer()
    result = optimizer.optimize(
        objective="min_cd",
        constraints=[("cl", ">=", 0.6)],
        initial_guess="naca0012"
    )
"""

__version__ = "0.3.0"

# Optimization module
from .thin_airfoil import GlobalAirfoilOptimizer, OptimizerConfig

__all__ = [
    # Version
    "__version__",

    # NeuralFoil
    "NeuralFoilAnalyzer",
    "NeuralFoilOptimizer",

    # Global Optimization
    "GlobalAirfoilOptimizer",
    "OptimizerConfig",
]
