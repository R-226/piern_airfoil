"""
Optimization step router for multi-fidelity airfoil optimization.

Routes between fidelity levels and optimization strategies based on
the current optimization state (conditional Markov decision).

Action space:
  - TAT + DE (thin airfoil theory, global search)
  - NeuralFoil-xxsmall/small/large/xlarge/xxxlarge + IPOPT (local refinement)

Router selects actions based on:
  - Budget remaining
  - Convergence rate
  - Constraint violation
  - Analysis confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class FidelityAction(IntEnum):
    """Discrete optimization action space.

    Each action = (analyzer, model_size, optimizer, depth).
    """
    # Thin airfoil theory + DE (global, ~1ms/eval)
    TAT_DE_SHALLOW = 0    # maxiter=20, popsize=5
    TAT_DE_DEEP = 1       # maxiter=80, popsize=15

    # NeuralFoil xxsmall + IPOPT (~1ms/eval)
    NF_XXS_IPOPT_1 = 2
    NF_XXS_IPOPT_3 = 3

    # NeuralFoil small + IPOPT (~1.2ms/eval)
    NF_SMALL_IPOPT_1 = 4
    NF_SMALL_IPOPT_3 = 5

    # NeuralFoil large + IPOPT (~1.6ms/eval)
    NF_LARGE_IPOPT_1 = 6
    NF_LARGE_IPOPT_3 = 7
    NF_LARGE_IPOPT_10 = 8

    # NeuralFoil xlarge + IPOPT (~1.7ms/eval)
    NF_XL_IPOPT_1 = 9
    NF_XL_IPOPT_3 = 10

    # NeuralFoil xxxlarge + IPOPT (~21ms/eval)
    NF_XXXL_IPOPT_3 = 11

    @property
    def model_size(self) -> str | None:
        if self.value <= 1:
            return None
        return {
            2: "xxsmall", 3: "xxsmall",
            4: "small", 5: "small",
            6: "large", 7: "large", 8: "large",
            9: "xlarge", 10: "xlarge",
            11: "xxxlarge",
        }[self.value]

    @property
    def n_ipopt_iters(self) -> int:
        if self.value <= 1:
            return 0
        return {2: 1, 3: 3, 4: 1, 5: 3, 6: 1, 7: 3, 8: 10, 9: 1, 10: 3, 11: 3}[self.value]

    @property
    def is_de(self) -> bool:
        return self.value <= 1

    @property
    def cost_estimate(self) -> float:
        """Relative cost (NF_LARGE_IPOPT_3 = 1.0)."""
        return {
            0: 0.1, 1: 0.5,
            2: 0.05, 3: 0.15,
            4: 0.06, 5: 0.18,
            6: 0.1, 7: 1.0, 8: 3.0,
            9: 0.12, 10: 1.2,
            11: 15.0,
        }[self.value]


N_ACTIONS = len(FidelityAction)


@dataclass
class OptimizationState:
    """Snapshot of the optimization process, used as Router input."""
    best_objective: float = float("inf")
    objective_history: list[float] = field(default_factory=list)
    constraint_violation: float = 0.0
    step_count: int = 0
    budget_used_ratio: float = 0.0
    confidence: float = 0.0
    last_action: int = -1
    action_history: list[int] = field(default_factory=list)

    def to_feature_vector(self, history_len: int = 5) -> np.ndarray:
        """Convert state to fixed-size feature vector (dim=13)."""
        features = []

        # Objective improvement
        if len(self.objective_history) >= 2:
            recent = self.objective_history[-history_len:]
            improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)
            convergence = np.std(recent) / (np.mean(np.abs(recent)) + 1e-8)
        else:
            improvement = 0.0
            convergence = 1.0

        features.append(np.clip(improvement, -1, 1))
        features.append(np.clip(convergence, 0, 1))

        # Best objective (log-scaled)
        if 0 < self.best_objective < float("inf"):
            features.append(np.clip(np.log10(self.best_objective + 1e-8), -3, 1))
        else:
            features.append(0.0)

        features.append(np.clip(self.constraint_violation, 0, 1))
        features.append(np.clip(self.budget_used_ratio, 0, 1))
        features.append(np.clip(self.confidence, 0, 1))
        features.append(float(self.last_action) / N_ACTIONS)

        # Action diversity
        if self.action_history:
            unique_ratio = len(set(self.action_history)) / len(self.action_history)
        else:
            unique_ratio = 0.0
        features.append(unique_ratio)

        # Recent objective history (normalized)
        obj_hist = self.objective_history[-history_len:]
        if len(obj_hist) < history_len:
            obj_hist = [0.0] * (history_len - len(obj_hist)) + obj_hist
        if 0 < self.best_objective < float("inf"):
            norm_hist = [o / (self.best_objective + 1e-8) for o in obj_hist]
        else:
            norm_hist = [0.0] * history_len
        features.extend(norm_hist)

        return np.array(features, dtype=np.float32)


FEATURE_DIM = 13


class OptimizationRouter:
    """Conditional Markov router for optimization step selection.

    Modes:
    - "rule": hand-crafted policy (default, no training needed)
    - "learned": MLP-based policy (requires trained weights)
    - "random": uniform random (baseline)
    """

    def __init__(self, mode: str = "rule", hidden_dim: int = 64):
        self.mode = mode
        self.hidden_dim = hidden_dim
        self._weights1 = None
        self._bias1 = None
        self._weights2 = None
        self._bias2 = None

    def get_action_probs(self, state: OptimizationState) -> np.ndarray:
        if self.mode == "rule":
            return self._rule_probs(state)
        elif self.mode == "random":
            return np.ones(N_ACTIONS) / N_ACTIONS
        else:
            return self._mlp_forward(state.to_feature_vector())

    def select_action(
        self,
        state: OptimizationState,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> FidelityAction:
        probs = self.get_action_probs(state)

        if temperature != 1.0:
            logits = np.log(probs + 1e-8) / temperature
            logits -= np.max(logits)
            probs = np.exp(logits) / np.exp(logits).sum()

        if deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(len(probs), p=probs))

        return FidelityAction(idx)

    def _rule_probs(self, state: OptimizationState) -> np.ndarray:
        """Hand-crafted policy.

        Strategy:
        - Early: cheap exploration (TAT_DE, NF_XXS)
        - Mid: balanced medium fidelity
        - Late: high fidelity refinement
        - Stuck: boost DE for global search
        - High violation: boost DE
        """
        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        r = state.budget_used_ratio

        if r < 0.2:
            probs[FidelityAction.TAT_DE_SHALLOW] = 0.3
            probs[FidelityAction.NF_XXS_IPOPT_1] = 0.3
            probs[FidelityAction.NF_SMALL_IPOPT_1] = 0.2
            probs[FidelityAction.NF_LARGE_IPOPT_1] = 0.2
        elif r < 0.5:
            probs[FidelityAction.TAT_DE_DEEP] = 0.15
            probs[FidelityAction.NF_SMALL_IPOPT_3] = 0.25
            probs[FidelityAction.NF_LARGE_IPOPT_3] = 0.35
            probs[FidelityAction.NF_XL_IPOPT_1] = 0.25
        elif r < 0.8:
            probs[FidelityAction.NF_LARGE_IPOPT_3] = 0.3
            probs[FidelityAction.NF_LARGE_IPOPT_10] = 0.3
            probs[FidelityAction.NF_XL_IPOPT_3] = 0.3
            probs[FidelityAction.NF_XXS_IPOPT_3] = 0.1
        else:
            probs[FidelityAction.NF_XL_IPOPT_3] = 0.4
            probs[FidelityAction.NF_XXXL_IPOPT_3] = 0.4
            probs[FidelityAction.NF_LARGE_IPOPT_10] = 0.2

        # Stuck detection: boost global search
        if len(state.objective_history) >= 3:
            recent = state.objective_history[-3:]
            if max(recent) - min(recent) < 1e-6 * abs(np.mean(recent) + 1e-8):
                probs[FidelityAction.TAT_DE_DEEP] += 0.3
                probs[FidelityAction.NF_XXS_IPOPT_1] += 0.1

        # High constraint violation: boost DE
        if state.constraint_violation > 0.1:
            probs[FidelityAction.TAT_DE_DEEP] += 0.2
            probs[FidelityAction.TAT_DE_SHALLOW] += 0.1

        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(N_ACTIONS) / N_ACTIONS

        return probs

    def _mlp_forward(self, features: np.ndarray) -> np.ndarray:
        h = np.maximum(features @ self._weights1 + self._bias1, 0)
        logits = h @ self._weights2 + self._bias2
        logits -= np.max(logits)
        return np.exp(logits) / np.exp(logits).sum()
