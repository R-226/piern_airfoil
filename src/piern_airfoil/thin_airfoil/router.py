"""
Optimization step router for multi-fidelity airfoil optimization.

Routes between different fidelity levels and optimization strategies
based on the current optimization state (conditional Markov decision).

Analogous to PiERN's SeqRouter but for optimization steps instead of tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class FidelityAction(IntEnum):
    """Discrete optimization action space.

    Each action combines a fidelity level (analyzer + model size),
    an optimizer type, and an optimization depth.
    """
    # Thin airfoil theory + DE (global search, ~1ms/eval)
    TAT_DE_SHALLOW = 0    # maxiter=20, popsize=5
    TAT_DE_DEEP = 1       # maxiter=80, popsize=15

    # NeuralFoil xxsmall + IPOPT (~1ms/eval, low accuracy)
    NF_XXS_IPOPT_1 = 2   # 1 warm-start iteration
    NF_XXS_IPOPT_3 = 3   # 3 warm-start iterations

    # NeuralFoil small + IPOPT (~1.2ms/eval, medium accuracy)
    NF_SMALL_IPOPT_1 = 4
    NF_SMALL_IPOPT_3 = 5

    # NeuralFoil large + IPOPT (~1.6ms/eval, high accuracy)
    NF_LARGE_IPOPT_1 = 6
    NF_LARGE_IPOPT_3 = 7
    NF_LARGE_IPOPT_10 = 8

    # NeuralFoil xlarge + IPOPT (~1.7ms/eval, very high accuracy)
    NF_XL_IPOPT_1 = 9
    NF_XL_IPOPT_3 = 10

    # NeuralFoil xxxlarge + IPOPT (~21ms/eval, highest accuracy)
    NF_XXXL_IPOPT_3 = 11

    @property
    def model_size(self) -> str | None:
        """NeuralFoil model size, or None for TAT."""
        if self.value <= 1:
            return None
        sizes = {
            2: "xxsmall", 3: "xxsmall",
            4: "small", 5: "small",
            6: "large", 7: "large", 8: "large",
            9: "xlarge", 10: "xlarge",
            11: "xxxlarge",
        }
        return sizes[self.value]

    @property
    def n_ipopt_iters(self) -> int:
        """Number of IPOPT warm-start iterations."""
        if self.value == 0:
            return 0
        if self.value == 1:
            return 0
        iters = {2: 1, 3: 3, 4: 1, 5: 3, 6: 1, 7: 3, 8: 10, 9: 1, 10: 3, 11: 3}
        return iters[self.value]

    @property
    def is_de(self) -> bool:
        return self.value <= 1

    @property
    def cost_estimate(self) -> float:
        """Relative cost (normalized to NF_LARGE_IPOPT_3 = 1.0)."""
        costs = {
            0: 0.1,    # TAT_DE_SHALLOW
            1: 0.5,    # TAT_DE_DEEP
            2: 0.05,   # NF_XXS_IPOPT_1
            3: 0.15,   # NF_XXS_IPOPT_3
            4: 0.06,   # NF_SMALL_IPOPT_1
            5: 0.18,   # NF_SMALL_IPOPT_3
            6: 0.1,    # NF_LARGE_IPOPT_1
            7: 1.0,    # NF_LARGE_IPOPT_3 (reference)
            8: 3.0,    # NF_LARGE_IPOPT_10
            9: 0.12,   # NF_XL_IPOPT_1
            10: 1.2,   # NF_XL_IPOPT_3
            11: 15.0,  # NF_XXXL_IPOPT_3
        }
        return costs[self.value]


N_ACTIONS = len(FidelityAction)


@dataclass
class OptimizationState:
    """Snapshot of the optimization process, used as Router input.

    All features are normalized to roughly [0, 1] range for stable learning.
    """
    # Objective tracking
    best_objective: float = float("inf")
    objective_history: list[float] = field(default_factory=list)

    # Constraint tracking
    constraint_violation: float = 0.0  # max violation magnitude

    # Progress
    step_count: int = 0
    budget_used_ratio: float = 0.0  # steps / max_steps

    # Quality signals
    confidence: float = 0.0  # last NeuralFoil analysis_confidence

    # Action tracking
    last_action: int = -1  # index into FidelityAction
    action_history: list[int] = field(default_factory=list)

    def to_feature_vector(self, history_len: int = 5) -> np.ndarray:
        """Convert state to fixed-size feature vector for the router.

        Returns:
            Feature vector of shape (feature_dim,).
        """
        features = []

        # Objective improvement (normalized)
        if len(self.objective_history) >= 2:
            recent = self.objective_history[-history_len:]
            # Relative improvement over last N steps
            if recent[0] != 0:
                improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)
            else:
                improvement = 0.0
            # Convergence rate (std of recent objectives)
            convergence = np.std(recent) / (np.mean(np.abs(recent)) + 1e-8)
        else:
            improvement = 0.0
            convergence = 1.0  # high = not converged

        features.append(np.clip(improvement, -1, 1))
        features.append(np.clip(convergence, 0, 1))

        # Best objective (log-scaled, clipped)
        if self.best_objective > 0 and self.best_objective < float("inf"):
            features.append(np.clip(np.log10(self.best_objective + 1e-8), -3, 1))
        else:
            features.append(0.0)

        # Constraint violation
        features.append(np.clip(self.constraint_violation, 0, 1))

        # Progress
        features.append(np.clip(self.budget_used_ratio, 0, 1))

        # Confidence
        features.append(np.clip(self.confidence, 0, 1))

        # Last action (one-hot encoded, but as index for embedding)
        features.append(float(self.last_action) / N_ACTIONS)

        # Action diversity (unique actions / total)
        if len(self.action_history) > 0:
            unique_ratio = len(set(self.action_history)) / len(self.action_history)
        else:
            unique_ratio = 0.0
        features.append(unique_ratio)

        # History of recent objectives (padded)
        obj_hist = self.objective_history[-history_len:]
        if len(obj_hist) < history_len:
            obj_hist = [0.0] * (history_len - len(obj_hist)) + obj_hist
        # Normalize
        if self.best_objective > 0 and self.best_objective < float("inf"):
            norm_hist = [o / (self.best_objective + 1e-8) for o in obj_hist]
        else:
            norm_hist = [0.0] * history_len
        features.extend(norm_hist)

        return np.array(features, dtype=np.float32)


FEATURE_DIM = 8 + 5  # 8 base features + 5 history features


class OptimizationRouter:
    """Conditional Markov router for optimization step selection.

    Uses a simple MLP to map optimization state → action probabilities.
    Can operate in:
    - Deterministic mode (argmax): always pick the best action
    - Stochastic mode (sample): explore according to probabilities
    - Rule-based mode: hand-crafted policies for bootstrapping

    The router is stateful: it tracks the optimization trajectory and
    makes decisions based on the history, not just the current state.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 64,
        mode: str = "rule",  # "rule", "learned", "random"
    ):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.mode = mode

        # MLP weights (initialized for rule-based or random)
        self._weights1 = None
        self._bias1 = None
        self._weights2 = None
        self._bias2 = None

        if mode == "learned":
            self._init_mlp()

    def _init_mlp(self):
        """Initialize MLP with Xavier initialization."""
        rng = np.random.RandomState(42)
        self._weights1 = rng.randn(self.feature_dim, self.hidden_dim).astype(np.float32) * 0.1
        self._bias1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._weights2 = rng.randn(self.hidden_dim, self.n_actions).astype(np.float32) * 0.1
        self._bias2 = np.zeros(self.n_actions, dtype=np.float32)

    def set_weights(self, w1, b1, w2, b2):
        """Set MLP weights externally (for training)."""
        self._weights1 = w1
        self._bias1 = b1
        self._weights2 = w2
        self._bias2 = b2
        self.mode = "learned"

    def get_action_probs(self, state: OptimizationState) -> np.ndarray:
        """Get action probability distribution.

        Returns:
            Array of shape (n_actions,) with probabilities.
        """
        features = state.to_feature_vector()

        if self.mode == "rule":
            return self._rule_based_probs(state)
        elif self.mode == "random":
            return np.ones(self.n_actions) / self.n_actions
        else:
            return self._mlp_forward(features)

    def _rule_based_probs(self, state: OptimizationState) -> np.ndarray:
        """Hand-crafted policy for bootstrapping.

        Strategy:
        - Early exploration: prefer cheap fast actions (TAT_DE, NF_XXS)
        - Mid optimization: prefer medium fidelity (NF_SMALL, NF_LARGE)
        - Late refinement: prefer high fidelity (NF_XL, NF_XXXL)
        - Stuck (no improvement): switch to different fidelity level
        - High constraint violation: prefer DE (global search)
        """
        probs = np.zeros(self.n_actions, dtype=np.float32)
        ratio = state.budget_used_ratio

        if ratio < 0.2:
            # Early: cheap exploration
            probs[FidelityAction.TAT_DE_SHALLOW] = 0.3
            probs[FidelityAction.NF_XXS_IPOPT_1] = 0.3
            probs[FidelityAction.NF_SMALL_IPOPT_1] = 0.2
            probs[FidelityAction.NF_LARGE_IPOPT_1] = 0.2
        elif ratio < 0.5:
            # Mid: balanced
            probs[FidelityAction.TAT_DE_DEEP] = 0.15
            probs[FidelityAction.NF_SMALL_IPOPT_3] = 0.25
            probs[FidelityAction.NF_LARGE_IPOPT_3] = 0.35
            probs[FidelityAction.NF_XL_IPOPT_1] = 0.25
        elif ratio < 0.8:
            # Late: refinement
            probs[FidelityAction.NF_LARGE_IPOPT_3] = 0.3
            probs[FidelityAction.NF_LARGE_IPOPT_10] = 0.3
            probs[FidelityAction.NF_XL_IPOPT_3] = 0.3
            probs[FidelityAction.NF_XXS_IPOPT_3] = 0.1  # diversity
        else:
            # Final: highest fidelity
            probs[FidelityAction.NF_XL_IPOPT_3] = 0.4
            probs[FidelityAction.NF_XXXL_IPOPT_3] = 0.4
            probs[FidelityAction.NF_LARGE_IPOPT_10] = 0.2

        # If stuck, boost DE (global exploration)
        if len(state.objective_history) >= 3:
            recent = state.objective_history[-3:]
            if max(recent) - min(recent) < 1e-6 * abs(np.mean(recent)):
                probs[FidelityAction.TAT_DE_DEEP] += 0.3
                probs[FidelityAction.NF_XXS_IPOPT_1] += 0.1

        # If high constraint violation, boost DE
        if state.constraint_violation > 0.1:
            probs[FidelityAction.TAT_DE_DEEP] += 0.2
            probs[FidelityAction.TAT_DE_SHALLOW] += 0.1

        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(self.n_actions) / self.n_actions

        return probs

    def _mlp_forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through MLP."""
        # Layer 1: ReLU
        h = features @ self._weights1 + self._bias1
        h = np.maximum(h, 0)  # ReLU

        # Layer 2: logits → softmax
        logits = h @ self._weights2 + self._bias2

        # Softmax (numerically stable)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum() + 1e-8)

        return probs

    def select_action(
        self,
        state: OptimizationState,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> FidelityAction:
        """Select the next optimization action.

        Args:
            state: Current optimization state.
            deterministic: If True, use argmax. If False, sample.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Selected FidelityAction.
        """
        probs = self.get_action_probs(state)

        if temperature != 1.0:
            # Apply temperature
            logits = np.log(probs + 1e-8) / temperature
            logits = logits - np.max(logits)
            probs = np.exp(logits) / np.exp(logits).sum()

        if deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(len(probs), p=probs))

        return FidelityAction(idx)
