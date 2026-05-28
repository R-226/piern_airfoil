"""
PiERN Router — decision layer for expert model invocation.

Two-level routing:
- SeqRouter: LLM token sequence → trigger expert? (simplified for now)
- OptRouter: optimization history → fidelity level decision

Training:
- train_threshold: grid search for optimal improvement_threshold
"""

from .opt_router import OptRouter

__all__ = ["OptRouter"]
