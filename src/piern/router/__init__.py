"""
Adaptive Router — decision layer for fidelity expansion.

OptRouter: optimization history → fidelity level decision (rule/threshold/mlp)

Training:
- train_threshold: grid search for optimal improvement_threshold
- mlp_router: learned MLP policy (~1000 params)
"""

from .opt_router import OptRouter

__all__ = ["OptRouter"]
