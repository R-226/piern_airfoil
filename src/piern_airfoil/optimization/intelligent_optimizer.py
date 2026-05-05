"""
Intelligent optimizer that combines all components.

This module provides the main entry point for the complete
airfoil optimization system.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np

from .parameterization.base import (
    Parameterization,
    CSTParameterization,
    AirfoilGeometry
)
from .analysis.base import AnalysisResult, FlowConditions
from .analysis.fusion import MultiFidelityAnalyzer, CachedMultiFidelityAnalyzer
from .optimization.engine import (
    OptimizationEngine,
    OptimizationConfig,
    OptimizationObjective,
    OptimizationConstraint,
    OptimizationAlgorithm
)
from .piern.agent import PPOAgent, DDPGAgent
from .piern.trainer import PiERNTrainer, TrainingConfig
from .ui.visualization import AirfoilVisualizer


@dataclass
class DesignSpec:
    """
    Design specification for airfoil optimization.

    Defines objectives, constraints, and conditions for optimization.
    """
    # Objectives
    objectives: List[OptimizationObjective]

    # Constraints
    constraints: List[OptimizationConstraint]

    # Flow conditions
    conditions: FlowConditions

    # Optimization parameters
    max_iterations: int = 1000
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.LBFGS

    # Target values (optional)
    target_CL: Optional[float] = None
    target_CD: Optional[float] = None

    # Multi-fidelity settings
    use_multifidelity: bool = True
    confidence_threshold: float = 0.7

    # PiERN settings
    use_piern: bool = False
    piern_episodes: int = 500


class IntelligentAirfoilOptimizer:
    """
    Main optimizer combining all components.

    Provides a unified interface for:
    - Parameterization
    - Multi-fidelity analysis (NeuralFoil + Transolver)
    - Classical optimization (gradient descent, genetic, etc.)
    - PiERN RL-based optimization (optional)
    - Visualization

    Example:
        optimizer = IntelligentAirfoilOptimizer()
        result = optimizer.optimize(design_spec)
        optimizer.visualize(result)
    """

    def __init__(
        self,
        parameterization: Optional[Parameterization] = None,
        analyzer: Optional[MultiFidelityAnalyzer] = None,
        use_cache: bool = True
    ):
        """
        Initialize optimizer.

        Args:
            parameterization: Parameterization method (default: CST)
            analyzer: Multi-fidelity analyzer (default: created automatically)
            use_cache: Whether to use result caching
        """
        # Setup parameterization
        self.param = parameterization or CSTParameterization()

        # Setup analyzer
        if analyzer is not None:
            self.analyzer = analyzer
        elif use_cache:
            self.analyzer = CachedMultiFidelityAnalyzer(
                confidence_threshold=0.7,
                enable_adaptive_threshold=True
            )
        else:
            self.analyzer = MultiFidelityAnalyzer(
                confidence_threshold=0.7,
                enable_adaptive_threshold=True
            )

        # Setup visualizer
        self.visualizer = AirfoilVisualizer()

        # Internal state
        self.optimization_history = []
        self.best_result = None
        self.best_params = None

    def optimize(
        self,
        spec: DesignSpec,
        initial_params: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run optimization according to design spec.

        Args:
            spec: Design specification
            initial_params: Initial parameters (default: random)

        Returns:
            Optimization result dict
        """
        # Initialize parameters
        if initial_params is None:
            geometry = self.param.random()
            initial_params = geometry.to_array()
        else:
            geometry = self.param.params_to_geometry(initial_params)

        # Ensure valid initial params
        is_valid, _ = self.param.validate(initial_params)
        if not is_valid:
            raise ValueError("Initial parameters are invalid")

        # Run optimization
        if spec.use_piern:
            result = self._optimize_with_piern(spec, initial_params)
        else:
            result = self._optimize_classical(spec, initial_params)

        # Store best
        self.best_params = result["params"]
        self.best_result = result
        self.optimization_history.append(result)

        return result

    def _optimize_classical(
        self,
        spec: DesignSpec,
        initial_params: np.ndarray
    ) -> Dict:
        """Run classical optimization."""
        # Create optimizer
        optimizer = OptimizationEngine(
            parameterization=self.param,
            analyzer=self.analyzer,
            config=OptimizationConfig(
                algorithm=spec.algorithm,
                max_iterations=spec.max_iterations
            )
        )

        # Run
        result = optimizer.optimize(
            initial_params,
            spec.objectives,
            spec.constraints,
            spec.conditions
        )

        return result

    def _optimize_with_piern(
        self,
        spec: DesignSpec,
        initial_params: np.ndarray
    ) -> Dict:
        """Run PiERN-based optimization."""
        # Create agent
        agent = PPOAgent(
            state_dim=28,
            action_dim=20,
            hidden_dims=[256, 128, 64]
        )

        # Create trainer
        trainer = PiERNTrainer(
            agent=agent,
            parameterization=self.param,
            analyzer=lambda g, c: self.analyzer.analyze(g, c),
            constraints=spec.constraints,
            config=TrainingConfig(
                n_episodes=spec.piern_episodes,
                max_steps_per_episode=200
            )
        )

        # Train
        history = trainer.train(initial_params, spec.conditions)

        # Get best parameters
        best_params = trainer.best_params
        if best_params is None:
            best_params = initial_params

        # Evaluate best
        geometry = self.param.params_to_geometry(best_params)
        best_result = self.analyzer.analyze(geometry, spec.conditions)

        return {
            "params": best_params,
            "result": best_result,
            "history": history,
            "method": "piern"
        }

    def batch_optimize(
        self,
        specs: List[DesignSpec],
        initial_params_list: List[np.ndarray] = None
    ) -> List[Dict]:
        """
        Run multiple optimizations in parallel.

        Args:
            specs: List of design specifications
            initial_params_list: List of initial parameters

        Returns:
            List of optimization results
        """
        results = []

        for i, spec in enumerate(specs):
            initial = (
                initial_params_list[i]
                if initial_params_list is not None
                else None
            )
            result = self.optimize(spec, initial)
            results.append(result)

        return results

    def visualize(self, result: Dict, show: bool = True):
        """
        Visualize optimization result.

        Args:
            result: Optimization result
            show: Whether to show plots
        """
        params = result["params"]
        geometry = self.param.params_to_geometry(params)

        # Get coordinates
        if geometry.coordinates is None:
            coords = self.param.get_coordinates(geometry)
        else:
            coords = geometry.coordinates

        # Plot airfoil
        self.visualizer.plot_airfoil(
            coords,
            title=f"Optimized Airfoil: CL={result.get('result', {}).CL:.3f}, "
                  f"CD={result.get('result', {}).CD:.5f}"
        )

        if show:
            import matplotlib.pyplot as plt
            plt.show()

    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {}

        final_result = self.optimization_history[-1].get("result")
        if final_result is None and "history" in self.optimization_history[-1]:
            # PiERN result
            history = self.optimization_history[-1]["history"]
            return {
                "n_iterations": len(history.get("episode_rewards", [])),
                "best_reward": max(history.get("episode_rewards", [-float('inf')])),
                "final_reward": history.get("episode_rewards", [0])[-1]
            }
        else:
            return {
                "n_iterations": len(self.optimization_history),
                "best_CL": self.best_params[0] if self.best_result else None,  # Placeholder
                "best_CD": self.best_result.CD if self.best_result else None
            }
