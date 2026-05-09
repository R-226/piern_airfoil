"""
NeuralFoil module - High-fidelity airfoil analysis and optimization.

Uses Aerosandbox's KulfanAirfoil + Opti framework for symbolic optimization.
"""

from .high_fidelity import LowFidelityOptimizer

__all__ = ["LowFidelityOptimizer"]
