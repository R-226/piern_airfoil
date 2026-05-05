"""
State representation for PiERN.

Defines how design state is represented for the RL agent.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

from ..analysis.base import AnalysisResult
from ..parameterization.base import AirfoilGeometry


@dataclass
class DesignState:
    """
    Design state representation for airfoil optimization.

    Contains all information needed by the RL agent to make decisions.

    State vector composition (28 dimensions):
    - Geometry parameters: 18 (CST weights + LE/TE)
    - Performance metrics: 4 (CL, CD, CM, L/D)
    - Constraint violations: 5 (max 5 constraints)
    - Confidence: 1
    """

    # Geometry (18 dims)
    upper_weights: np.ndarray      # shape: (8,)
    lower_weights: np.ndarray      # shape: (8,)
    leading_edge_weight: float
    te_thickness: float

    # Performance (4 dims)
    CL: float
    CD: float
    CM: float
    L_D: float

    # Constraint violations (5 dims, capped)
    constraint_violations: np.ndarray  # shape: (5,)

    # Confidence (1 dim)
    confidence: float

    # Optional metadata
    iteration: int = 0
    history: Optional[List[Dict]] = None

    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure arrays have correct shape
        self.constraint_violations = np.array(self.constraint_violations).flatten()
        if len(self.constraint_violations) < 5:
            self.constraint_violations = np.pad(
                self.constraint_violations,
                (0, 5 - len(self.constraint_violations)),
                constant_values=0
            )
        elif len(self.constraint_violations) > 5:
            self.constraint_violations = self.constraint_violations[:5]

    @classmethod
    def from_geometry_and_result(
        cls,
        geometry: AirfoilGeometry,
        result: AnalysisResult,
        constraints: List = None,
        n_constraints: int = 5
    ) -> "DesignState":
        """
        Create DesignState from geometry and analysis result.

        Args:
            geometry: Airfoil geometry
            result: Analysis result
            constraints: List of constraint functions
            n_constraints: Number of constraint slots

        Returns:
            DesignState instance
        """
        # Compute constraint violations
        if constraints:
            violations = []
            for c in constraints:
                val = c(result)
                violations.append(max(0, -val))  # Positive if violated
        else:
            violations = [0.0] * n_constraints

        return cls(
            upper_weights=geometry.upper_weights,
            lower_weights=geometry.lower_weights,
            leading_edge_weight=geometry.leading_edge_weight,
            te_thickness=geometry.te_thickness,
            CL=result.CL,
            CD=result.CD,
            CM=result.CM,
            L_D=result.L/D,
            constraint_violations=np.array(violations),
            confidence=result.confidence
        )

    def to_vector(self) -> np.ndarray:
        """
        Convert to state vector for neural network input.

        Returns:
            State vector of shape (28,)
        """
        # Geometry (18)
        geometry = np.concatenate([
            self.upper_weights,  # 8
            self.lower_weights,  # 8
            [self.leading_edge_weight, self.te_thickness]  # 2
        ])

        # Performance (4)
        performance = np.array([
            self.CL, self.CD, self.CM, self.L_D
        ])

        # Constraint violations (5) - clip to prevent extreme values
        violations = np.clip(self.constraint_violations, 0, 10)

        # Confidence (1)
        confidence = np.array([self.confidence])

        return np.concatenate([geometry, performance, violations, confidence])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "DesignState":
        """
        Create DesignState from state vector.

        Args:
            vector: State vector of shape (28,)

        Returns:
            DesignState instance
        """
        if len(vector) != 28:
            raise ValueError(f"Expected 28-dimensional vector, got {len(vector)}")

        idx = 0

        # Geometry (18)
        upper_weights = vector[idx:idx+8]
        idx += 8
        lower_weights = vector[idx:idx+8]
        idx += 8
        leading_edge_weight = vector[idx]
        idx += 1
        te_thickness = vector[idx]
        idx += 1

        # Performance (4)
        CL = vector[idx]
        idx += 1
        CD = vector[idx]
        idx += 1
        CM = vector[idx]
        idx += 1
        L_D = vector[idx]
        idx += 1

        # Constraint violations (5)
        constraint_violations = vector[idx:idx+5]
        idx += 5

        # Confidence (1)
        confidence = vector[idx]

        return cls(
            upper_weights=upper_weights,
            lower_weights=lower_weights,
            leading_edge_weight=leading_edge_weight,
            te_thickness=te_thickness,
            CL=CL,
            CD=CD,
            CM=CM,
            L_D=L_D,
            constraint_violations=constraint_violations,
            confidence=confidence
        )

    def update(
        self,
        new_geometry: AirfoilGeometry,
        new_result: AnalysisResult,
        action: "DesignAction" = None,
        reward: float = 0.0
    ) -> "DesignState":
        """
        Create new state after taking an action.

        Args:
            new_geometry: Updated geometry
            new_result: Updated analysis result
            action: Action taken
            reward: Reward received

        Returns:
            New DesignState
        """
        # Update history
        new_history = self.history.copy() if self.history else []
        new_history.append({
            "params": self.to_vector()[:18].copy(),
            "CL": self.CL,
            "CD": self.CD,
            "reward": reward,
            "action": action
        })

        # Keep only last 10 history entries
        if len(new_history) > 10:
            new_history = new_history[-10:]

        # Create new state
        return DesignState(
            upper_weights=new_geometry.upper_weights,
            lower_weights=new_geometry.lower_weights,
            leading_edge_weight=new_geometry.leading_edge_weight,
            te_thickness=new_geometry.te_thickness,
            CL=new_result.CL,
            CD=new_result.CD,
            CM=new_result.CM,
            L_D=new_result.L/D,
            constraint_violations=self.constraint_violations,  # Will be recomputed
            confidence=new_result.confidence,
            iteration=self.iteration + 1,
            history=new_history
        )

    def compute_reward(
        self,
        target_CL: float = None,
        target_CD: float = None,
        constraint_weight: float = 10.0,
        smoothness_weight: float = 0.1
    ) -> float:
        """
        Compute reward for current state.

        Args:
            target_CL: Target lift coefficient
            target_CD: Target drag coefficient
            constraint_weight: Weight for constraint violations
            smoothness_weight: Weight for parameter change smoothness

        Returns:
            Reward value
        """
        reward = 0.0

        # Lift reward
        if target_CL is not None:
            cl_error = abs(self.CL - target_CL)
            reward -= cl_error * 5  # Penalty for CL deviation

        # Drag reward (minimize)
        if target_CD is not None:
            reward -= self.CD * 100  # Direct minimization
        else:
            reward -= self.CD * 100

        # Constraint violations penalty
        violation_penalty = np.sum(np.maximum(0, self.constraint_violations))
        reward -= violation_penalty * constraint_weight

        # Small bonus for high L/D
        if self.L_D > 50:
            reward += 1.0
        elif self.L_D > 100:
            reward += 5.0

        # Small penalty for low confidence
        if self.confidence < 0.5:
            reward -= 0.5

        return reward

    def is_valid(self) -> bool:
        """Check if state is valid (no severe constraint violations)."""
        return (
            np.all(self.constraint_violations < 1.0) and
            self.confidence > 0.2 and
            -3 < self.CL < 3 and
            0 < self.CD < 1
        )


@dataclass
class DesignAction:
    """
    Design action representation.

    Represents the action taken by the RL agent.

    Action vector composition (20 dimensions):
    - Parameter delta: 18 (CST weights adjustment)
    - Analysis choice: 1 (0=fast, 1=precise, 2=both)
    - Exploration rate: 1 (0-1)
    """

    param_delta: np.ndarray       # shape: (18,)
    analysis_choice: int         # 0=fast, 1=precise, 2=both
    exploration_rate: float       # 0-1

    ANALYSIS_LABELS = ["fast", "precise", "both"]

    def __post_init__(self):
        """Validate and clip values."""
        self.param_delta = np.array(self.param_delta).flatten()
        if len(self.param_delta) != 18:
            raise ValueError(f"Expected 18 parameter deltas, got {len(self.param_delta)}")

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
        """Create random action for exploration."""
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

    def get_analysis_label(self) -> str:
        """Get human-readable analysis choice."""
        return self.ANALYSIS_LABELS[self.analysis_choice]
