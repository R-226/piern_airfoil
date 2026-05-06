"""
PIERN-Airfoil: Unified framework for automatic airfoil optimization.

Organized into three main modules:
- neuralfoil: Fast analysis and optimization using Aerosandbox + NeuralFoil
- transolver: High-fidelity CFD analysis (pending retraining)
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

# Transolver module (placeholder)
from .transolver import TransolverAnalyzer, TransolverOptimizer

# PiERN module
from .piern import (
    PiERNPolicyNetwork,
    PiERNValueNetwork,
    PPOAgent,
    DDPGAgent,
    DesignState,
    DesignAction,
    PiERNTrainer,
    TrainingConfig,
)

# UI module
from .ui import AirfoilVisualizer

__all__ = [
    # Version
    "__version__",

    # NeuralFoil
    "NeuralFoilAnalyzer",
    "NeuralFoilOptimizer",

    # Transolver (placeholder)
    "TransolverAnalyzer",
    "TransolverOptimizer",

    # PiERN
    "PiERNPolicyNetwork",
    "PiERNValueNetwork",
    "PPOAgent",
    "DDPGAgent",
    "DesignState",
    "DesignAction",
    "PiERNTrainer",
    "TrainingConfig",

    # UI
    "AirfoilVisualizer",
]
