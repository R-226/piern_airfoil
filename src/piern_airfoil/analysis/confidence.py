"""
Confidence estimator for multi-fidelity analysis.

Estimates analysis confidence based on multiple factors.
"""

from typing import Optional
import numpy as np

from .base import AnalysisResult, AnalysisConfidence
from ..parameterization.base import AirfoilGeometry
from ..parameterization.validity import AirfoilValidator


class ConfidenceEstimator:
    """
    Multi-factor confidence estimator.

    Computes confidence based on:
    1. NeuralFoil's built-in confidence
    2. Geometry validity
    3. Aerodynamic reasonableness
    4. Parameter distribution distance
    """

    # Weights for each factor
    DEFAULT_WEIGHTS = {
        "neuralfoil": 0.4,
        "geometry": 0.2,
        "aerodynamic": 0.25,
        "distribution": 0.15
    }

    def __init__(
        self,
        weights: Optional[dict] = None,
        validator: Optional[AirfoilValidator] = None
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.validator = validator or AirfoilValidator()

    def estimate(
        self,
        result: AnalysisResult,
        geometry: Optional[AirfoilGeometry] = None,
        params: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate confidence score [0, 1].

        Args:
            result: Analysis result from NeuralFoil
            geometry: Airfoil geometry (optional)
            params: Parameter array (optional)

        Returns:
            Confidence score
        """
        scores = []

        # Factor 1: NeuralFoil built-in confidence
        scores.append(self.weights["neuralfoil"] * result.confidence)

        # Factor 2: Geometry validity
        if geometry is not None and geometry.coordinates is not None:
            geom_score = self._geometry_score(geometry)
        else:
            geom_score = 0.8  # Assume valid if no geometry provided
        scores.append(self.weights["geometry"] * geom_score)

        # Factor 3: Aerodynamic reasonableness
        aero_score = self._aerodynamic_score(result)
        scores.append(self.weights["aerodynamic"] * aero_score)

        # Factor 4: Distribution distance (if params provided)
        if params is not None:
            dist_score = self._distribution_score(params)
            scores.append(self.weights["distribution"] * dist_score)
        else:
            scores.append(self.weights["distribution"])

        return float(np.clip(sum(scores), 0, 1))

    def _geometry_score(self, geometry: AirfoilGeometry) -> float:
        """Score based on geometry validity."""
        if geometry.coordinates is None:
            return 0.5

        # Validate geometry
        validation = self.validator.validate(geometry.coordinates)

        if validation.is_valid:
            return 1.0
        else:
            # Partial credit based on violations
            n_violations = len(validation.violations)
            return max(0.0, 1.0 - 0.2 * n_violations)

    def _aerodynamic_score(self, result: AnalysisResult) -> float:
        """
        Score based on aerodynamic reasonableness.

        Checks:
        - CL in reasonable range [-3, 3]
        - CD positive and < 1.0
        - L/D reasonable for airfoil [-100, 100]
        - CM in reasonable range [-1, 1]
        """
        score = 1.0

        # Check CL
        if abs(result.CL) > 3.0:
            score -= 0.3
        elif abs(result.CL) > 2.0:
            score -= 0.1

        # Check CD
        if result.CD < 0:
            score -= 0.4
        elif result.CD > 0.5:
            score -= 0.2
        elif result.CD > 0.1:
            score -= 0.05

        # Check L/D
        if abs(result.L/D) > 200:
            score -= 0.2
        elif abs(result.L/D) > 100:
            score -= 0.1

        # Check CM
        if abs(result.CM) > 0.5:
            score -= 0.2
        elif abs(result.CM) > 0.2:
            score -= 0.05

        return float(max(0.0, score))

    def _distribution_score(self, params: np.ndarray) -> float:
        """
        Score based on Mahalanobis distance from training distribution.

        This is a simplified version - full implementation would use
        the actual Mahalanobis distance computed during NeuralFoil training.
        """
        # Simplified: use distance from typical CST parameter ranges
        # Weights should be in [-0.5, 0.5] roughly
        # TE thickness should be small [0, 0.02]

        weight_score = 1.0 - 0.1 * np.mean(np.abs(params[:16]) > 0.5)
        te_score = 1.0 - 0.5 * (params[17] > 0.05)

        return float(0.7 * weight_score + 0.3 * te_score)

    def to_confidence_level(self, confidence: float) -> AnalysisConfidence:
        """Convert numeric confidence to enum level."""
        if confidence >= 0.8:
            return AnalysisConfidence.HIGH
        elif confidence >= 0.5:
            return AnalysisConfidence.MEDIUM
        elif confidence >= 0.2:
            return AnalysisConfidence.LOW
        else:
            return AnalysisConfidence.UNKNOWN


class AdaptiveThreshold:
    """
    Adaptive threshold for confidence-based analysis selection.

    Automatically adjusts threshold based on optimization progress.
    """

    def __init__(
        self,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.3,
        max_threshold: float = 0.9,
        adaptation_rate: float = 0.01
    ):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate

    def update(self, iteration: int, improvement_rate: float):
        """
        Update threshold based on optimization progress.

        Args:
            iteration: Current iteration
            improvement_rate: Recent improvement rate
        """
        # Increase threshold when improvement is slow
        if improvement_rate < 0.01:
            self.threshold = min(
                self.max_threshold,
                self.threshold + self.adaptation_rate
            )
        # Decrease threshold when improvement is fast
        elif improvement_rate > 0.1:
            self.threshold = max(
                self.min_threshold,
                self.threshold - self.adaptation_rate
            )

    def should_use_precise(self, confidence: float) -> bool:
        """Decide whether to use precise analysis."""
        return confidence < self.threshold
