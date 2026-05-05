"""
Simple aerodynamic analyzer using potential flow theory.

This provides a lightweight alternative when NeuralFoil/Transolver
weights are not available.
"""

import numpy as np
from typing import Union

from .base import AnalysisResult, FlowConditions, AnalysisConfidence


class SimpleAerodynamicAnalyzer:
    """
    Simple aerodynamic analyzer based on potential flow theory.

    Uses thin airfoil theory and empirical corrections for quick analysis.
    This is intended as a fallback when NeuralFoil is not available.
    """

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze(
        self,
        geometry: "AirfoilGeometry",
        conditions: FlowConditions
    ) -> AnalysisResult:
        """
        Analyze airfoil aerodynamics using simplified methods.

        Args:
            geometry: AirfoilGeometry object
            conditions: Flow conditions

        Returns:
            AnalysisResult with aerodynamic coefficients
        """
        # Compute geometry properties
        thickness = self._compute_thickness(geometry)
        camber = self._compute_camber(geometry)

        # Convert to radians
        alpha_rad = np.deg2rad(conditions.alpha)

        # Thin airfoil theory: CL = 2*pi*(alpha + camber)
        # Add thickness effect (thicker airfoils have slightly higher CL_max)
        CL = 2 * np.pi * (alpha_rad + camber) * (1 + 0.8 * thickness)

        # Drag coefficient: CD = CD0 + CDi
        # CD0: zero-lift drag (roughness dependent)
        CD0 = 0.008 + 0.002 * thickness  # Base drag + thickness correction
        CDi = CL**2 / (np.pi * 5.0)     # Induced drag (AR=5 assumption)

        # Camber contribution to drag
        CD_camber = 0.01 * abs(camber)

        CD = CD0 + CDi + CD_camber

        # Moment coefficient: CM ≈ -0.1 * CL (approximate)
        # Negative camber gives positive moment (nose-down)
        CM = -0.05 * CL - 0.02 * camber

        # Transition location (simplified)
        # Higher camber → earlier transition on upper surface
        Top_Xtr = 0.3 - 2.0 * camber
        Bot_Xtr = 0.3 + 2.0 * camber
        Top_Xtr = np.clip(Top_Xtr, 0.1, 0.9)
        Bot_Xtr = np.clip(Bot_Xtr, 0.1, 0.9)

        # Confidence (lower for simplified model)
        confidence = 0.6
        confidence_level = AnalysisConfidence.MEDIUM

        return AnalysisResult(
            CL=float(CL),
            CD=float(CD),
            CM=float(CM),
            Top_Xtr=float(Top_Xtr),
            Bot_Xtr=float(Bot_Xtr),
            confidence=confidence,
            confidence_level=confidence_level,
            conditions=conditions,
            source="simple_potential"
        )

    def _compute_thickness(self, geometry: "AirfoilGeometry") -> float:
        """Compute approximate thickness ratio."""
        # Thickness is related to the magnitude of CST weights
        upper_sum = np.abs(geometry.upper_weights).mean()
        lower_sum = np.abs(geometry.lower_weights).mean()

        # Rough approximation
        thickness = (upper_sum + lower_sum) * 2 + geometry.te_thickness * 10

        return np.clip(thickness, 0.02, 0.30)

    def _compute_camber(self, geometry: "AirfoilGeometry") -> float:
        """Compute approximate camber (in radians)."""
        # Camber is related to the difference between upper and lower weights
        upper_mean = geometry.upper_weights.mean()
        lower_mean = geometry.lower_weights.mean()

        # Rough approximation (convert to radians)
        camber = (upper_mean - lower_mean) * 0.5 + geometry.leading_edge_weight * 0.1

        return np.clip(camber, -0.15, 0.15)

    def analyze_with_gradient(
        self,
        geometry: "AirfoilGeometry",
        conditions: FlowConditions
    ) -> tuple:
        """
        Analyze and return gradients for optimization.

        Returns:
            (AnalysisResult, gradients_dict)
        """
        result = self.analyze(geometry, conditions)

        # Compute gradients (simplified)
        # dCL/dalpha ≈ 2*pi
        # dCD/dCL ≈ 2*CL/(pi*AR)
        alpha_rad = np.deg2rad(conditions.alpha)
        camber = self._compute_camber(geometry)
        thickness = self._compute_thickness(geometry)

        dCL_dalpha = 2 * np.pi * (1 + 0.8 * thickness)
        dCD_dCL = 2 * CL / (np.pi * 5.0) if 'CL' in dir() else 0

        gradients = {
            'dCL_dalpha': dCL_dalpha,
            'dCD_dCL': dCD_dCL,
            'dCM_dCL': -0.05
        }

        return result, gradients


# Singleton instance for convenience
_default_analyzer = None

def get_simple_analyzer() -> SimpleAerodynamicAnalyzer:
    """Get the default simple analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SimpleAerodynamicAnalyzer()
    return _default_analyzer
