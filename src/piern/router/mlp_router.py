"""
MLP Router: learned optimization-level fidelity router.

Architecture: 8-dim state → 32 hidden → 3 actions (KEEP / EXPAND_2 / EXPAND_ALL)
~1000 parameters. Trained on auto-labeled optimization episodes.

State vector (8-dim):
    [0] cd_current          - current weighted CD
    [1] cd_improvement      - fractional improvement from prev stage
    [2] cd_improvement_2nd  - 2nd derivative of improvement (acceleration)
    [3] n_active_weights/8  - fraction of max weights active
    [4] stage/max_stages    - progress through stages
    [5] cd_vs_initial       - current CD / initial CD (how far from start)
    [6] stall_count         - consecutive stages with <0.1% improvement
    [7] weight_dim_ratio    - n_active / 8 (dimension utilization)

Actions:
    0: KEEP       - stay at current dimension
    1: EXPAND_2   - open 2 more weights
    2: EXPAND_ALL - jump to 8 weights immediately
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import aerosandbox as asb


# ── State representation ────────────────────────────────────────────────


@dataclass
class MLPOptState:
    """8-dimensional state vector for the MLP router."""
    cd_current: float
    cd_improvement: float  # 0 if first stage
    cd_improvement_2nd: float  # 0 if < 3 stages
    n_active_weights: int
    stage: int
    max_stages: int
    initial_cd: float
    stall_count: int  # consecutive stages with < 0.1% improvement

    def to_vector(self) -> np.ndarray:
        """Convert to 8-dim feature vector."""
        return np.array([
            self.cd_current,
            self.cd_improvement,
            self.cd_improvement_2nd,
            self.n_active_weights / 8.0,
            self.stage / self.max_stages,
            self.cd_current / self.initial_cd if self.initial_cd > 0 else 1.0,
            self.stall_count,
            self.n_active_weights / 8.0,
        ], dtype=np.float32)


# ── MLP Architecture (~1000 params) ────────────────────────────────────


class MLPRouter:
    """
    Small MLP for fidelity routing decisions.

    Architecture:
        Linear(8, 32) → ReLU → Linear(32, 3) → Softmax

    ~1000 parameters (8*32 + 32 + 32*3 + 3 = 387)
    """

    def __init__(self, weights_path: Path | str | None = None):
        self._weights = None
        self._bias = None
        self._weights2 = None
        self._bias2 = None

        if weights_path is not None:
            self.load(weights_path)

    def _init_random(self, seed: int = 42):
        """Initialize with random weights (for testing)."""
        rng = np.random.RandomState(seed)
        self._weights = rng.randn(8, 32).astype(np.float32) * 0.1
        self._bias = np.zeros(32, dtype=np.float32)
        self._weights2 = rng.randn(32, 3).astype(np.float32) * 0.1
        self._bias2 = np.zeros(3, dtype=np.float32)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass: state (8,) → action_probs (3,).

        Uses numpy only (no torch dependency for inference).
        """
        if self._weights is None:
            self._init_random()

        # Layer 1: Linear + ReLU
        h = state @ self._weights + self._bias
        h = np.maximum(h, 0)  # ReLU

        # Layer 2: Linear + Softmax
        logits = h @ self._weights2 + self._bias2
        # Numerically stable softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def decide(self, state: MLPOptState) -> tuple[int, str]:
        """
        Decide action from state.

        Returns:
            (action_index, reason_string)
            action_index: 0=KEEP, 1=EXPAND_2, 2=EXPAND_ALL
        """
        return self.decide_from_vector(state.to_vector())

    def decide_from_vector(self, vec: np.ndarray) -> tuple[int, str]:
        """
        Decide action from a raw 8-dim feature vector.

        Returns:
            (action_index, reason_string)
            action_index: 0=KEEP, 1=EXPAND_2, 2=EXPAND_ALL
        """
        probs = self.forward(vec)
        action = int(np.argmax(probs))

        reasons = [
            f"KEEP (p={probs[0]:.2f})",
            f"EXPAND_2 (p={probs[1]:.2f})",
            f"EXPAND_ALL (p={probs[2]:.2f})",
        ]

        return action, reasons[action]

    def save(self, path: Path | str):
        """Save model weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "architecture": "mlp_router",
            "input_dim": 8,
            "hidden_dim": 32,
            "output_dim": 3,
            "params": {
                "weights": self._weights.tolist(),
                "bias": self._bias.tolist(),
                "weights2": self._weights2.tolist(),
                "bias2": self._bias2.tolist(),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | str):
        """Load model weights from JSON."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        params = data["params"]
        self._weights = np.array(params["weights"], dtype=np.float32)
        self._bias = np.array(params["bias"], dtype=np.float32)
        self._weights2 = np.array(params["weights2"], dtype=np.float32)
        self._bias2 = np.array(params["bias2"], dtype=np.float32)


# ── Training data collection ────────────────────────────────────────────


@dataclass
class TrainingSample:
    """A single training sample for the MLP router."""
    state: np.ndarray  # 8-dim
    action: int  # 0=KEEP, 1=EXPAND_2, 2=EXPAND_ALL
    reward: float  # improvement after this stage


@dataclass
class EpisodeData:
    """Training data from a single optimization episode."""
    airfoil_name: str
    samples: list[TrainingSample]
    final_cd: float


def collect_training_data(
    airfoil_names: list[str] | None = None,
    n_episodes_per_airfoil: int = 1,
    max_stages: int = 6,
    verbose: bool = True,
) -> list[EpisodeData]:
    """
    Collect training data by running optimization episodes.

    For each episode, runs the hierarchical optimizer and records
    (state, action, reward) at each decision point.

    Actions are labeled by counterfactual analysis:
    - If improvement > 0.5%: label = KEEP (0)
    - If improvement < 0.1% and n_active < 8: label = EXPAND_2 (1)
    - If improvement < 0.1% and n_active < 6: could also be EXPAND_ALL (2)
    """
    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer

    if airfoil_names is None:
        airfoil_names = [
            "naca0012", "naca2412", "naca4412",
            "naca0015", "naca2415", "naca6412",
        ]

    # Problem definition
    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5

    episodes = []

    for airfoil_name in airfoil_names:
        for ep_idx in range(n_episodes_per_airfoil):
            if verbose:
                print(f"Collecting: {airfoil_name} ep={ep_idx}...", end=" ", flush=True)

            airfoil = asb.KulfanAirfoil(airfoil_name)
            initial_cd = _eval_cd_airfoil(airfoil, CL_TARGETS, CL_WEIGHTS, RE)

            optimizer = AdaptiveHierarchicalOptimizer(
                CL_targets=CL_TARGETS,
                CL_weights=CL_WEIGHTS,
                Re=RE,
                mach=0.03,
                start_weights=4,
                improvement_threshold=0.001,  # Low threshold for diverse data
            )

            result = optimizer.optimize(airfoil)

            # Extract training samples from stages
            samples = []
            prev_improvement = 0.0
            stall_count = 0

            for i, stage in enumerate(result.stages):
                if i == 0:
                    improvement = 0.0
                    improvement_2nd = 0.0
                else:
                    prev_cd = result.stages[i - 1].cd
                    improvement = (prev_cd - stage.cd) / prev_cd if prev_cd > 0 else 0.0
                    if i >= 2:
                        prev_prev_cd = result.stages[i - 2].cd
                        prev_imp = (prev_prev_cd - prev_cd) / prev_prev_cd if prev_prev_cd > 0 else 0.0
                        improvement_2nd = improvement - prev_imp
                    else:
                        improvement_2nd = 0.0

                # Track stalls
                if improvement < 0.001:
                    stall_count += 1
                else:
                    stall_count = 0

                state = MLPOptState(
                    cd_current=stage.cd,
                    cd_improvement=improvement,
                    cd_improvement_2nd=improvement_2nd,
                    n_active_weights=stage.n_active_weights,
                    stage=i + 1,
                    max_stages=max_stages,
                    initial_cd=initial_cd,
                    stall_count=stall_count,
                )

                # Label action by counterfactual analysis
                if improvement > 0.005:
                    action = 0  # KEEP - good improvement
                elif improvement > 0.001:
                    action = 0  # KEEP - marginal improvement
                else:
                    if stage.n_active_weights < 6:
                        action = 2  # EXPAND_ALL - jump to max
                    elif stage.n_active_weights < 8:
                        action = 1  # EXPAND_2 - incremental
                    else:
                        action = 0  # KEEP - already at max

                reward = improvement
                samples.append(TrainingSample(
                    state=state.to_vector(),
                    action=action,
                    reward=reward,
                ))

                prev_improvement = improvement

            episodes.append(EpisodeData(
                airfoil_name=airfoil_name,
                samples=samples,
                final_cd=result.final_cd,
            ))

            if verbose:
                print(f"CD={result.final_cd:.6f}, {len(samples)} samples")

    return episodes


def _eval_cd_airfoil(airfoil, CL_TARGETS, CL_WEIGHTS, RE) -> float:
    """Quick CD evaluation."""
    from piern_airfoil.eval import evaluate_weighted_cd

    return evaluate_weighted_cd(airfoil, CL_TARGETS, RE, CL_WEIGHTS, mach=0.03)


# ── MLP Training ────────────────────────────────────────────────────────


def train_mlp(
    episodes: list[EpisodeData],
    epochs: int = 100,
    lr: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> MLPRouter:
    """
    Train the MLP router on collected data.

    Simple gradient descent with cross-entropy loss.
    No torch dependency — pure numpy.
    """
    # Flatten all samples
    all_samples = []
    for ep in episodes:
        all_samples.extend(ep.samples)

    if not all_samples:
        raise ValueError("No training samples collected")

    X = np.array([s.state for s in all_samples])
    y = np.array([s.action for s in all_samples])

    if verbose:
        print(f"\nTraining MLP Router: {len(X)} samples, {epochs} epochs")
        print(f"Action distribution: KEEP={np.sum(y==0)}, EXPAND_2={np.sum(y==1)}, EXPAND_ALL={np.sum(y==2)}")

    # Initialize model
    router = MLPRouter()
    rng = np.random.RandomState(seed)

    # Xavier initialization
    router._weights = rng.randn(8, 32).astype(np.float32) * np.sqrt(2.0 / 8)
    router._bias = np.zeros(32, dtype=np.float32)
    router._weights2 = rng.randn(32, 3).astype(np.float32) * np.sqrt(2.0 / 32)
    router._bias2 = np.zeros(3, dtype=np.float32)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        total_loss = 0.0
        dw1_grad = np.zeros_like(router._weights)
        db1_grad = np.zeros_like(router._bias)
        dw2_grad = np.zeros_like(router._weights2)
        db2_grad = np.zeros_like(router._bias2)

        for i in range(len(X)):
            x = X[i]
            target = y[i]

            # Forward
            h = x @ router._weights + router._bias
            h_relu = np.maximum(h, 0)
            logits = h_relu @ router._weights2 + router._bias2
            logits_max = np.max(logits)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits)

            # Cross-entropy loss
            loss = -np.log(probs[target] + 1e-8)
            total_loss += loss

            # Backward (cross-entropy + softmax gradient)
            dlogits = probs.copy()
            dlogits[target] -= 1.0  # dL/dlogits = probs - one_hot(target)

            # Gradients for layer 2
            dw2_grad += np.outer(h_relu, dlogits)
            db2_grad += dlogits

            # Gradients for layer 1
            dh_relu = dlogits @ router._weights2.T
            dh = dh_relu * (h > 0).astype(np.float32)  # ReLU gradient
            dw1_grad += np.outer(x, dh)
            db1_grad += dh

        # Update weights
        n = len(X)
        router._weights -= lr * dw1_grad / n
        router._bias -= lr * db1_grad / n
        router._weights2 -= lr * dw2_grad / n
        router._bias2 -= lr * db2_grad / n

        if verbose and (epoch + 1) % 20 == 0:
            # Compute accuracy
            correct = 0
            for i in range(len(X)):
                probs = router.forward(X[i])
                if np.argmax(probs) == y[i]:
                    correct += 1
            acc = correct / len(X)
            print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f}, acc={acc:.2%}")

    return router


# ── Main ────────────────────────────────────────────────────────────────


SAVE_DIR = Path(__file__).parent / "trained"


def main():
    """Collect data and train the MLP router."""
    print("=" * 60)
    print("Strategy A2: MLP Router Training")
    print("=" * 60)

    t0 = time.perf_counter()

    # Collect training data
    print("\nPhase 1: Collecting training data...")
    episodes = collect_training_data(
        airfoil_names=["naca0012", "naca2412", "naca4412", "naca0015"],
        max_stages=6,
        verbose=True,
    )

    # Train MLP
    print("\nPhase 2: Training MLP router...")
    router = train_mlp(episodes, epochs=100, lr=0.01, verbose=True)

    # Save
    save_path = SAVE_DIR / "mlp_router.json"
    router.save(save_path)
    print(f"\nSaved MLP router to {save_path}")

    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.1f}s")

    # Quick test
    print("\nQuick test:")
    test_state = MLPOptState(
        cd_current=0.5,
        cd_improvement=0.003,
        cd_improvement_2nd=0.0,
        n_active_weights=4,
        stage=2,
        max_stages=6,
        initial_cd=0.52,
        stall_count=0,
    )
    action, reason = router.decide(test_state)
    print(f"  imp=0.003, n_active=4: {reason}")

    test_state2 = MLPOptState(
        cd_current=0.5,
        cd_improvement=0.0005,
        cd_improvement_2nd=-0.002,
        n_active_weights=4,
        stage=3,
        max_stages=6,
        initial_cd=0.52,
        stall_count=2,
    )
    action2, reason2 = router.decide(test_state2)
    print(f"  imp=0.0005, stall=2: {reason2}")

    return router


if __name__ == "__main__":
    import aerosandbox as asb
    main()
