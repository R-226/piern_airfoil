"""
NeuralFoil module - High-fidelity airfoil analysis and optimization.

Uses Aerosandbox's KulfanAirfoil + Opti framework for symbolic optimization.
"""

from .neuralfoil import NeuralOptimizer

__all__ = ["NeuralOptimizer"]
