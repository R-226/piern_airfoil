"""
Design action definitions for PiERN.

Defines action space and constraints for airfoil optimization.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DesignAction:
    """
    Design action for airfoil optimization.

    Action vector composition (20 dimensions):
    - Parameter delta: 18 (CST weights adjustment, scaled)
    - Analysis choice: 1 (0=fast, 1=precise, 2=both)
    - Exploration rate: 1 (0-1)
    """

    param_delta: np.ndarray       # shape: (18,)
    analysis_choice: int         # 0=fast, 1=precise, 2=both
    exploration_rate: float       # 0-1

    # Labels for analysis choices
    ANALYSIS_LABELS = ["fast", "precise", "both"]

    # Bounds for parameter deltas
    PARAM_DELTA_SCALE = 0.05  # Typical scale for parameter changes
    PARAM_DELTA_MAX = 0.2     # Maximum parameter change per step

    def __post_init__(self):
        """Validate and clip values."""
        self.param_delta = np.array(self.param_delta).flatten()
        if len(self.param_delta) != 18:
            raise ValueError(f"Expected 18 parameter deltas, got {len(self.param_delta)}")

        # Clip parameter deltas
        self.param_delta = np.clip(
            self.param_delta,
            -self.PARAM_DELTA_MAX,
            self.PARAM_DELTA_MAX
        )

        self.analysis_choice = int(np.clip(self.analysis_choice, 0, 2))
        self.exploration_rate = float(np.clip(self.exploration_rate, 0, 1))

    def to_vector(self) -> np.ndarray:
        """Convert to action vector."""
        return np.concatenate([
            self.param_delta,
            [float(self.analysis_choice)],
            [self.exploration_rate]
        ])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "DesignAction":
        """Create from action vector."""
        if len(vector) != 20:
            raise ValueError(f"Expected 20-dimensional vector, got {len(vector)}")

        param_delta = vector[:18]
        analysis_choice = int(vector[18])
        exploration_rate = float(vector[19])

        return cls(
            param_delta=param_delta,
            analysis_choice=analysis_choice,
            exploration_rate=exploration_rate
        )

    @classmethod
    def random(cls, scale: float = 0.1) -> "DesignAction":
        """
        Create random action for exploration.

        Args:
            scale: Scale of random parameter changes

        Returns:
            DesignAction with random values
        """
        return cls(
            param_delta=np.random.randn(18) * scale,
            analysis_choice=np.random.randint(0, 3),
            exploration_rate=np.random.rand()
        )

    @classmethod
    def zero(cls) -> "DesignAction":
        """Create zero action (no change)."""
        return cls(
            param_delta=np.zeros(18),
            analysis_choice=0,
            exploration_rate=0.0
        )

    @classmethod
    def from_gradients(cls, gradients: np.ndarray, step_size: float = 0.01) -> "DesignAction":
        """
        Create action from gradients (for policy gradient methods).

        Args:
            gradients: Gradient vector
            step_size: Step size for gradient descent

        Returns:
            DesignAction representing gradient step
        """
        if len(gradients) != 18:
            raise ValueError(f"Expected 18 gradients, got {len(gradients)}")

        return cls(
            param_delta=-gradients * step_size,  # Gradient descent
            analysis_choice=0,  # Use fast analysis by default
            exploration_rate=0.0
        )

    def get_analysis_label(self) -> str:
        """Get human-readable analysis choice."""
        return self.ANALYSIS_LABELS[self.analysis_choice]

    def scale_param_delta(self, scale: float) -> "DesignAction":
        """Scale parameter delta (for adaptive step size)."""
        return DesignAction(
            param_delta=self.param_delta * scale,
            analysis_choice=self.analysis_choice,
            exploration_rate=self.exploration_rate
        )

    def apply_to_params(self, params: np.ndarray) -> np.ndarray:
        """
        Apply action to parameters.

        Args:
            params: Current parameter array (18,)

        Returns:
            Updated parameters
        """
        if len(params) != 18:
            raise ValueError(f"Expected 18 parameters, got {len(params)}")

        new_params = params + self.param_delta

        # Ensure parameters stay in reasonable bounds
        # Weights typically in [-0.5, 0.5]
        new_params[:16] = np.clip(new_params[:16], -0.5, 0.5)
        # TE thickness in [0, 0.05]
        new_params[17] = np.clip(new_params[17], 0, 0.05)

        return new_params


class ActionConstraints:
    """
    Constraints on actions to ensure valid parameter updates.

    Prevents the agent from taking actions that would lead to
    invalid or extreme airfoil geometries.
    """

    def __init__(
        self,
        param_min: np.ndarray = None,
        param_max: np.ndarray = None,
        max_delta: float = 0.2,
        max_total_delta: float = 0.5
    ):
        """
        Initialize action constraints.

        Args:
            param_min: Minimum allowed parameter values
            param_max: Maximum allowed parameter values
            max_delta: Maximum change per parameter
            max_total_delta: Maximum total parameter change (L2)
        """
        # Default bounds for CST parameters
        if param_min is None:
            self.param_min = np.concatenate([
                np.full(8, -0.5),   # upper_weights
                np.full(8, -0.5),   # lower_weights
                [-0.2],             # leading_edge_weight
                [0]                 # TE_thickness
            ])
        else:
            self.param_min = param_min

        if param_max is None:
            self.param_max = np.concatenate([
                np.full(8, 0.5),    # upper_weights
                np.full(8, 0.5),    # lower_weights
                [0.2],              # leading_edge_weight
                [0.05]              # TE_thickness
            ])
        else:
            self.param_max = param_max

        self.max_delta = max_delta
        self.max_total_delta = max_total_delta

    def clip_action(self, action: DesignAction, current_params: np.ndarray) -> DesignAction:
        """
        Clip action to satisfy constraints.

        Args:
            action: Proposed action
            current_params: Current parameters

        Returns:
            Clipped action
        """
        # Compute proposed new parameters
        proposed_params = action.apply_to_params(current_params)

        # Clip to parameter bounds
        clipped_params = np.clip(proposed_params, self.param_min, self.param_max)

        # Compute delta that respects bounds
        new_delta = clipped_params - current_params

        # Clip individual parameter changes
        new_delta = np.clip(new_delta, -self.max_delta, self.max_delta)

        # Clip total change
        total_change = np.linalg.norm(new_delta)
        if total_change > self.max_total_delta:
            new_delta = new_delta * (self.max_total_delta / total_change)

        return DesignAction(
            param_delta=new_delta,
            analysis_choice=action.analysis_choice,
            exploration_rate=action.exploration_rate
        )

    def is_valid_action(self, action: DesignAction, current_params: np.ndarray) -> bool:
        """
        Check if action is valid.

        Args:
            action: Action to check
            current_params: Current parameters

        Returns:
            True if action is valid
        """
        proposed_params = action.apply_to_params(current_params)

        # Check parameter bounds
        if np.any(proposed_params < self.param_min) or np.any(proposed_params > self.param_max):
            return False

        # Check delta bounds
        delta = action.param_delta
        if np.any(np.abs(delta) > self.max_delta):
            return False

        # Check total delta
        if np.linalg.norm(delta) > self.max_total_delta:
            return False

        return True
