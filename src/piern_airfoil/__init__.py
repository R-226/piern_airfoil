"""
PIERN-Airfoil: Unified framework for automatic airfoil optimization.

Combines Transolver (precise CFD), NeuralFoil (fast analysis),
and PiERN (Token-Level Router) for intelligent airfoil design.

Example usage:
    from piern_airfoil import (
        CSTParameterization,
        FlowConditions,
        OptimizationEngine,
        OptimizationObjective
    )

    # Setup
    param = CSTParameterization()
    optimizer = OptimizationEngine(param, lambda g, c: None)  # Use mock analyzer

    # Run optimization
    initial = param.random()
    result = optimizer.optimize(
        initial.to_array(),
        [OptimizationObjective(name="min_drag", target=lambda r: r.CD)],
        [],
        FlowConditions(alpha=5, Re=3e6)
    )
"""

__version__ = "0.2.0"

# Parameterization
from .parameterization.base import (
    Parameterization,
    CSTParameterization,
    AirfoilGeometry,
)
from .parameterization.validity import AirfoilValidator

# Analysis
from .analysis.base import (
    AnalysisResult,
    FlowConditions,
    AnalysisConfidence,
    MultiFidelityResult
)
from .analysis.neuralfoil_wrapper import NeuralFoilAnalyzer
from .analysis.transolver_wrapper import TransolverWrapper
from .analysis.confidence import ConfidenceEstimator, AdaptiveThreshold
from .analysis.fusion import MultiFidelityAnalyzer, CachedMultiFidelityAnalyzer
from .analysis.simple_aero import SimpleAerodynamicAnalyzer

# Optimization
from .optimization.engine import (
    OptimizationEngine,
    OptimizationConfig,
    OptimizationObjective,
    OptimizationConstraint,
    OptimizationAlgorithm
)
from .optimization.callbacks import (
    OptimizationCallback,
    PrintCallback,
    HistoryCallback,
    ConvergenceCallback
)

# PiERN
from .piern.policy_network import PiERNPolicyNetwork
from .piern.agent import PPOAgent, DDPGAgent
from .piern.state_representation import DesignState, DesignAction
from .piern.trainer import PiERNTrainer, TrainingConfig, CurriculumTrainer

# UI
from .ui.visualization import AirfoilVisualizer

__all__ = [
    # Version
    "__version__",

    # Parameterization
    "Parameterization",
    "CSTParameterization",
    "AirfoilGeometry",
    "AirfoilValidator",

    # Analysis
    "AnalysisResult",
    "FlowConditions",
    "AnalysisConfidence",
    "MultiFidelityResult",
    "NeuralFoilAnalyzer",
    "TransolverWrapper",
    "ConfidenceEstimator",
    "AdaptiveThreshold",
    "MultiFidelityAnalyzer",
    "CachedMultiFidelityAnalyzer",
    "SimpleAerodynamicAnalyzer",

    # Optimization
    "OptimizationEngine",
    "OptimizationConfig",
    "OptimizationObjective",
    "OptimizationConstraint",
    "OptimizationAlgorithm",
    "OptimizationCallback",
    "PrintCallback",
    "HistoryCallback",
    "ConvergenceCallback",

    # PiERN
    "PiERNPolicyNetwork",
    "PPOAgent",
    "DDPGAgent",
    "DesignState",
    "DesignAction",
    "PiERNTrainer",
    "TrainingConfig",
    "CurriculumTrainer",

    # UI
    "AirfoilVisualizer"
]
