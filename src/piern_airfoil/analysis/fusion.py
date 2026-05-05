"""
Multi-fidelity analyzer combining NeuralFoil and Transolver.

Provides adaptive analysis that switches between fast and precise models
based on confidence estimation.
"""

from typing import Optional, Union, Callable
import numpy as np
import time

from .base import AnalysisResult, FlowConditions, AnalysisConfidence, MultiFidelityResult
from .neuralfoil_wrapper import NeuralFoilAnalyzer
from .transolver_wrapper import TransolverWrapper
from .confidence import ConfidenceEstimator, AdaptiveThreshold


class MultiFidelityAnalyzer:
    """
    Adaptive multi-fidelity analyzer.

    Automatically selects between:
    - Fast analysis (NeuralFoil): ~5ms, moderate accuracy
    - Precise analysis (Transolver): ~100ms, high accuracy

    Selection is based on confidence estimation.
    """

    # Default confidence threshold for switching to precise analysis
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        neuralfoil_model_size: str = "xlarge",
        transolver_model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        enable_adaptive_threshold: bool = False,
        device: str = "cuda:0"
    ):
        """
        Initialize multi-fidelity analyzer.

        Args:
            neuralfoil_model_size: NeuralFoil model size
            transolver_model_path: Path to Transolver model (optional)
            confidence_threshold: Threshold for switching to precise analysis
            enable_adaptive_threshold: Whether to adapt threshold dynamically
            device: Computation device for Transolver
        """
        # Fast analyzer
        self.fast = NeuralFoilAnalyzer(model_size=neuralfoil_model_size)

        # Precise analyzer (optional)
        self.precise = None
        if transolver_model_path or enable_adaptive_threshold:
            try:
                self.precise = TransolverWrapper(
                    model_path=transolver_model_path,
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load Transolver: {e}")

        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator()

        # Threshold settings
        self.confidence_threshold = confidence_threshold
        self.enable_adaptive = enable_adaptive_threshold
        self.adaptive_threshold = AdaptiveThreshold(
            initial_threshold=confidence_threshold
        ) if enable_adaptive_threshold else None

        # Statistics
        self.stats = {
            "fast_calls": 0,
            "precise_calls": 0,
            "total_fast_time": 0.0,
            "total_precise_time": 0.0
        }

    def analyze(
        self,
        geometry: Union[np.ndarray, "AirfoilGeometry"],
        conditions: FlowConditions,
        require_precision: Optional[bool] = None,
        return_details: bool = False
    ) -> Union[AnalysisResult, MultiFidelityResult]:
        """
        Analyze airfoil with automatic fidelity selection.

        Args:
            geometry: Airfoil geometry (CST params or AirfoilGeometry)
            conditions: Flow conditions
            require_precision: Force specific analysis type
            return_details: Whether to return detailed MultiFidelityResult

        Returns:
            AnalysisResult or MultiFidelityResult (if return_details=True)
        """
        # Ensure geometry is in correct format
        if isinstance(geometry, np.ndarray):
            from ..parameterization.base import CSTParameterization
            param = CSTParameterization()
            geometry = param.params_to_geometry(geometry)

        # Step 1: Fast analysis
        start_time = time.time()
        fast_result = self._fast_analysis(geometry, conditions)
        fast_time = time.time() - start_time
        self.stats["total_fast_time"] += fast_time
        self.stats["fast_calls"] += 1

        # Step 2: Estimate confidence
        confidence = self.confidence_estimator.estimate(fast_result, geometry)
        confidence_level = self.confidence_estimator.to_confidence_level(confidence)

        # Step 3: Decide if precise analysis is needed
        if require_precision is None:
            if self.enable_adaptive and self.adaptive_threshold:
                require_precision = self.adaptive_threshold.should_use_precise(confidence)
            else:
                require_precision = confidence < self.confidence_threshold

        # Step 4: Optional precise analysis
        precise_result = None
        precise_time = 0.0
        if require_precision and self.precise is not None:
            start_time = time.time()
            precise_result = self._precise_analysis(geometry, conditions)
            precise_time = time.time() - start_time
            self.stats["total_precise_time"] += precise_time
            self.stats["precise_calls"] += 1

        # Step 5: Fuse or select result
        if precise_result is not None:
            fused_result = self._fuse_results(
                fast_result, precise_result, confidence
            )
            source = "fused"
        else:
            fused_result = fast_result
            source = "fast"

        fused_result.confidence = confidence
        fused_result.confidence_level = confidence_level
        fused_result.source = source

        if return_details:
            return MultiFidelityResult(
                fast_result=fast_result,
                precise_result=precise_result,
                fused_result=fused_result,
                fusion_weight=confidence,
                fusion_reason=self._get_fusion_reason(
                    confidence, require_precision, precise_result is not None
                )
            )
        else:
            return fused_result

    def _fast_analysis(self, geometry, conditions) -> AnalysisResult:
        """Run fast NeuralFoil analysis."""
        return self.fast.analyze(geometry, conditions)

    def _precise_analysis(self, geometry, conditions) -> AnalysisResult:
        """Run precise Transolver analysis."""
        if self.precise is None:
            raise RuntimeError("Transolver not available")
        return self.precise.analyze(geometry, conditions)

    def _fuse_results(
        self,
        fast: AnalysisResult,
        precise: AnalysisResult,
        confidence: float
    ) -> AnalysisResult:
        """
        Fuse fast and precise results.

        Uses confidence-weighted fusion:
        result_fused = w * precise + (1-w) * fast

        where w = confidence (more confidence in fast result = less weight on precise)
        """
        # Note: This is counterintuitive but correct
        # Low confidence -> use precise more
        # High confidence -> use fast only
        w = 1.0 - confidence  # Weight for precise result

        return AnalysisResult(
            CL=w * precise.CL + (1 - w) * fast.CL,
            CD=w * precise.CD + (1 - w) * fast.CD,
            CM=w * precise.CM + (1 - w) * fast.CM,
            Top_Xtr=w * precise.Top_Xtr + (1 - w) * fast.Top_Xtr,
            Bot_Xtr=w * precise.Bot_Xtr + (1 - w) * fast.Bot_Xtr,
            confidence=confidence,
            conditions=fast.conditions,
            source="fused"
        )

    def _get_fusion_reason(
        self,
        confidence: float,
        require_precision: bool,
        has_precise: bool
    ) -> str:
        """Explain why fusion was chosen."""
        if not has_precise:
            return "transolver_not_available"
        if require_precision:
            return f"low_confidence_{confidence:.2f}"
        return "high_confidence"

    def get_statistics(self) -> dict:
        """Get analysis statistics."""
        total_calls = self.stats["fast_calls"] + self.stats["precise_calls"]
        avg_fast_time = (
            self.stats["total_fast_time"] / self.stats["fast_calls"]
            if self.stats["fast_calls"] > 0 else 0
        )
        avg_precise_time = (
            self.stats["total_precise_time"] / self.stats["precise_calls"]
            if self.stats["precise_calls"] > 0 else 0
        )

        return {
            "total_calls": total_calls,
            "fast_calls": self.stats["fast_calls"],
            "precise_calls": self.stats["precise_calls"],
            "precise_ratio": (
                self.stats["precise_calls"] / total_calls
                if total_calls > 0 else 0
            ),
            "avg_fast_time_ms": avg_fast_time * 1000,
            "avg_precise_time_ms": avg_precise_time * 1000,
            "estimated_speedup": (
                (avg_precise_time / avg_fast_time)
                if avg_fast_time > 0 else 1.0
            )
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "fast_calls": 0,
            "precise_calls": 0,
            "total_fast_time": 0.0,
            "total_precise_time": 0.0
        }

    def batch_analyze(
        self,
        geometries: list,
        conditions: FlowConditions,
        parallel: bool = False
    ) -> list:
        """
        Batch analyze multiple airfoils.

        Args:
            geometries: List of geometries
            conditions: Flow conditions
            parallel: Whether to use parallel processing

        Returns:
            List of AnalysisResult
        """
        if parallel:
            # Parallel processing (requires joblib or similar)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    lambda g: self.analyze(g, conditions),
                    geometries
                ))
            return results
        else:
            # Sequential
            return [self.analyze(g, conditions) for g in geometries]


class CachedMultiFidelityAnalyzer(MultiFidelityAnalyzer):
    """
    Multi-fidelity analyzer with result caching.

    Caches analysis results to avoid redundant computations.
    """

    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size

    def _get_cache_key(
        self,
        geometry: "AirfoilGeometry",
        conditions: FlowConditions
    ) -> str:
        """Generate cache key for geometry/conditions pair."""
        params = geometry.to_array()
        return f"{params.tobytes()}_{conditions.alpha}_{conditions.Re}"

    def analyze(
        self,
        geometry: Union[np.ndarray, "AirfoilGeometry"],
        conditions: FlowConditions,
        require_precision: Optional[bool] = None,
        return_details: bool = False
    ) -> Union[AnalysisResult, MultiFidelityResult]:
        """Analyze with caching."""
        # Ensure geometry format
        if isinstance(geometry, np.ndarray):
            from ..parameterization.base import CSTParameterization
            param = CSTParameterization()
            geometry = param.params_to_geometry(geometry)

        # Check cache
        cache_key = self._get_cache_key(geometry, conditions)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if return_details:
                return cached
            else:
                return cached.fused_result

        # Compute result
        result = super().analyze(geometry, conditions, require_precision, True)

        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Simple FIFO eviction
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[cache_key] = result

        if return_details:
            return result
        else:
            return result.fused_result

    def clear_cache(self):
        """Clear the result cache."""
        self.cache.clear()
