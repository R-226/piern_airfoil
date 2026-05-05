"""
NeuralFoil wrapper for unified airfoil analysis.

Provides a clean interface to NeuralFoil's analysis capabilities.
"""

from typing import Optional, Union
import numpy as np

from .base import AnalysisResult, FlowConditions, AnalysisConfidence
from ..parameterization.base import AirfoilGeometry, Parameterization


class NeuralFoilAnalyzer:
    """
    Wrapper around NeuralFoil for airfoil aerodynamic analysis.

    NeuralFoil is a physics-informed neural network that predicts
    airfoil aerodynamics at ~5ms per evaluation with ~2% error vs XFoil.
    """

    def __init__(self, model_size: str = "xlarge"):
        """
        Initialize NeuralFoil analyzer.

        Args:
            model_size: NeuralFoil model size. Options: xxsmall, xsmall, small,
                       medium, large, xlarge, xxlarge, xxxlarge.
                       Larger models are more accurate but slower.
        """
        self.model_size = model_size
        self._neuralfoil_available = self._check_neuralfoil()

    def _check_neuralfoil(self) -> bool:
        """Check if neuralfoil package is available."""
        try:
            import neuralfoil as nf
            return True
        except ImportError:
            return False

    def analyze(
        self,
        geometry: Union[AirfoilGeometry, np.ndarray],
        conditions: FlowConditions,
        return_bl_data: bool = False
    ) -> AnalysisResult:
        """
        Analyze airfoil aerodynamics.

        Args:
            geometry: AirfoilGeometry or parameter array (18 dims for CST)
            conditions: Flow conditions (alpha, Re, etc.)
            return_bl_data: Whether to return boundary layer data

        Returns:
            AnalysisResult with aerodynamic coefficients
        """
        if not self._neuralfoil_available:
            return self._mock_result(conditions)

        import neuralfoil as nf

        # Convert geometry to appropriate format
        if isinstance(geometry, AirfoilGeometry):
            kulfan_params = self._geometry_to_kulfan(geometry)
        else:
            # Assume it's already a CST parameter array
            kulfan_params = self._array_to_kulfan(geometry)

        # Call NeuralFoil
        try:
            result = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan_params,
                alpha=conditions.alpha,
                Re=conditions.Re,
                n_crit=conditions.n_crit,
                xtr_upper=conditions.xtr_upper,
                xtr_lower=conditions.xtr_lower,
                model_size=self.model_size
            )

            return AnalysisResult(
                CL=float(result["CL"].item()),
                CD=float(result["CD"].item()),
                CM=float(result["CM"].item()),
                Top_Xtr=float(result["Top_Xtr"].item()),
                Bot_Xtr=float(result["Bot_Xtr"].item()),
                confidence=float(result["analysis_confidence"].item()),
                confidence_level=self._confidence_to_level(float(result["analysis_confidence"].item())),
                conditions=conditions,
                source="neuralfoil",
                upper_bl_theta=result.get("upper_bl_theta_i"),
                upper_bl_H=result.get("upper_bl_H_i"),
                lower_bl_theta=result.get("lower_bl_theta_i"),
                lower_bl_H=result.get("lower_bl_H_i"),
            )

        except Exception as e:
            # Return a fallback result if NeuralFoil fails
            return self._mock_result(conditions)

    def _geometry_to_kulfan(self, geometry: AirfoilGeometry) -> dict:
        """Convert AirfoilGeometry to NeuralFoil's kulfan format."""
        return {
            "upper_weights": geometry.upper_weights.tolist(),
            "lower_weights": geometry.lower_weights.tolist(),
            "leading_edge_weight": float(geometry.leading_edge_weight),
            "TE_thickness": float(geometry.te_thickness)
        }

    def _array_to_kulfan(self, arr: np.ndarray) -> dict:
        """Convert parameter array to kulfan format."""
        if len(arr) != 18:
            raise ValueError(f"Expected 18 CST parameters, got {len(arr)}")
        return {
            "upper_weights": arr[:8].tolist(),
            "lower_weights": arr[8:16].tolist(),
            "leading_edge_weight": float(arr[16]),
            "TE_thickness": float(arr[17])
        }

    def _confidence_to_level(self, confidence: float) -> AnalysisConfidence:
        """Convert confidence value to level enum."""
        if confidence >= 0.8:
            return AnalysisConfidence.HIGH
        elif confidence >= 0.5:
            return AnalysisConfidence.MEDIUM
        elif confidence >= 0.2:
            return AnalysisConfidence.LOW
        else:
            return AnalysisConfidence.UNKNOWN

    def _mock_result(self, conditions: FlowConditions) -> AnalysisResult:
        """
        Generate a mock result when NeuralFoil is not available.

        This is for testing/development purposes only.
        """
        # Simple mock based on thin airfoil theory
        alpha_rad = np.deg2rad(conditions.alpha)
        CL = 2 * np.pi * alpha_rad  # Thin airfoil theory

        # Drag polar (simple approximation)
        CD0 = 0.008  # Zero-lift drag
        K = 0.04    # Induced drag factor
        CD = CD0 + K * CL**2

        return AnalysisResult(
            CL=CL,
            CD=CD,
            CM=-0.05 * CL,  # Simple moment approximation
            Top_Xtr=0.3,
            Bot_Xtr=0.3,
            confidence=0.5,
            confidence_level=AnalysisConfidence.MEDIUM,
            conditions=conditions,
            source="mock"
        )

    def batch_analyze(
        self,
        geometries: list,
        conditions: FlowConditions
    ) -> list:
        """
        Batch analyze multiple airfoils.

        Args:
            geometries: List of AirfoilGeometry or parameter arrays
            conditions: Flow conditions (will be broadcast)

        Returns:
            List of AnalysisResult
        """
        results = []
        for geo in geometries:
            result = self.analyze(geo, conditions)
            results.append(result)
        return results

    def sweep_alpha(
        self,
        geometry: Union[AirfoilGeometry, np.ndarray],
        alpha_range: tuple,
        n_points: int = 20,
        Re: float = 3e6
    ) -> dict:
        """
        Perform alpha sweep analysis.

        Args:
            geometry: Airfoil geometry
            alpha_range: (min_alpha, max_alpha) in degrees
            n_points: Number of points in sweep
            Re: Reynolds number

        Returns:
            Dictionary with arrays of alpha, CL, CD, CM, etc.
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)

        results = {
            "alpha": alphas,
            "CL": np.zeros(n_points),
            "CD": np.zeros(n_points),
            "CM": np.zeros(n_points),
            "confidence": np.zeros(n_points)
        }

        for i, alpha in enumerate(alphas):
            conditions = FlowConditions(alpha=alpha, Re=Re)
            result = self.analyze(geometry, conditions)
            results["CL"][i] = result.CL
            results["CD"][i] = result.CD
            results["CM"][i] = result.CM
            results["confidence"][i] = result.confidence

        return results
