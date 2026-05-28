"""
OptRouter: optimization-level fidelity router.

Monitors optimization history (CD trajectory, constraint satisfaction)
and decides when to increase CST parameterization dimension.

This is the PiERN architecture applied to optimization decision-making:
  state(CD history, weight dimension, constraint violations) → action(keep/expand)

Supports three modes:
1. Rule-based: fixed improvement_threshold
2. Learned threshold: from grid search training (train_threshold.py)
3. MLP-based: learned 8-dim state → 3 actions (mlp_router.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class OptAction(Enum):
    """Router action."""
    KEEP = "keep"               # Continue at current fidelity
    EXPAND = "expand"           # Open more CST weights (by 2)
    EXPAND_ALL = "expand_all"   # Jump to max weights immediately


@dataclass
class OptState:
    """Observable optimization state for the router."""
    stage: int
    n_active_weights: int
    cd: float
    prev_cd: float | None = None
    initial_cd: float | None = None
    stall_count: int = 0
    max_stages: int = 6

    @property
    def improvement(self) -> float | None:
        """Fractional CD improvement from previous stage."""
        if self.prev_cd is None or self.prev_cd == 0:
            return None
        return (self.prev_cd - self.cd) / self.prev_cd

    def to_mlp_state(self) -> np.ndarray:
        """Convert to 8-dim vector for MLP router."""
        from .mlp_router import MLPOptState

        imp = self.improvement if self.improvement is not None else 0.0
        return MLPOptState(
            cd_current=self.cd,
            cd_improvement=imp,
            cd_improvement_2nd=0.0,  # Not tracked here
            n_active_weights=self.n_active_weights,
            stage=self.stage,
            max_stages=self.max_stages,
            initial_cd=self.initial_cd if self.initial_cd else self.cd,
            stall_count=self.stall_count,
        ).to_vector()


@dataclass
class OptRouter:
    """
    Optimization fidelity router.

    Decides whether to continue at the current CST dimension or expand.
    Mirrors PiERN's SeqRouter but operates on optimization history
    instead of token sequences.

    Modes:
        - rule: fixed threshold (improvement_threshold)
        - threshold: learned threshold from grid search
        - mlp: learned MLP from training episodes
    """

    improvement_threshold: float = 0.01
    max_weights: int = 8
    mode: str = "rule"  # "rule", "threshold", "mlp"
    _mlp_router: object = field(default=None, repr=False)

    @classmethod
    def from_trained(cls, path: Path | str | None = None, **kwargs) -> OptRouter:
        """
        Create a router with a learned threshold from training.

        Args:
            path: Path to optimal_threshold.json. If None, uses default.
            **kwargs: Additional arguments passed to OptRouter.
        """
        if path is None:
            path = Path(__file__).parent / "trained" / "optimal_threshold.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"No trained threshold found at {path}. "
                "Run 'uv run python -m piern.router.train_threshold' first."
            )

        import json
        with open(path) as f:
            data = json.load(f)

        threshold = data["optimal_threshold"]
        return cls(improvement_threshold=threshold, mode="threshold", **kwargs)

    @classmethod
    def from_mlp(cls, path: Path | str | None = None, **kwargs) -> OptRouter:
        """
        Create a router with a trained MLP model.

        Args:
            path: Path to mlp_router.json. If None, uses default.
            **kwargs: Additional arguments passed to OptRouter.
        """
        from .mlp_router import MLPRouter

        if path is None:
            path = Path(__file__).parent / "trained" / "mlp_router.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"No trained MLP router found at {path}. "
                "Run 'uv run python -m piern.router.mlp_router' first."
            )

        mlp = MLPRouter(path)
        return cls(mode="mlp", _mlp_router=mlp, **kwargs)

    def decide(self, state: OptState) -> tuple[OptAction, int, str]:
        """
        Decide next action based on optimization state.

        Args:
            state: Current optimization state.

        Returns:
            (action, new_n_active_weights, reason_string)
        """
        if self.mode == "mlp" and self._mlp_router is not None:
            return self._decide_mlp(state)
        return self._decide_rule(state)

    def _decide_rule(self, state: OptState) -> tuple[OptAction, int, str]:
        """Rule-based decision (original logic)."""
        if state.improvement is None:
            return OptAction.KEEP, state.n_active_weights, "初始阶段，保持当前维度"

        if state.improvement > self.improvement_threshold:
            return (
                OptAction.KEEP,
                state.n_active_weights,
                f"改进显著 ({state.improvement:.3f})，继续 {state.n_active_weights} 权重",
            )

        new_n = min(state.n_active_weights + 2, self.max_weights)
        if new_n > state.n_active_weights:
            return (
                OptAction.EXPAND,
                new_n,
                f"改进不足 ({state.improvement:.3f})，扩展到 {new_n} 权重",
            )

        return (
            OptAction.KEEP,
            state.n_active_weights,
            f"已达最大维度 ({self.max_weights} 权重)",
        )

    def _decide_mlp(self, state: OptState) -> tuple[OptAction, int, str]:
        """MLP-based decision."""
        vec = state.to_mlp_state()
        action_idx, reason = self._mlp_router.decide_from_vector(vec)

        if action_idx == 0:  # KEEP
            return OptAction.KEEP, state.n_active_weights, reason
        elif action_idx == 1:  # EXPAND_2
            new_n = min(state.n_active_weights + 2, self.max_weights)
            return OptAction.EXPAND, new_n, reason
        else:  # EXPAND_ALL
            return OptAction.EXPAND, self.max_weights, reason
