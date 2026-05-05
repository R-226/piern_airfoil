"""Analysis module for airfoil performance evaluation."""

from .base import AnalysisResult, FlowConditions, AnalysisConfidence
from .neuralfoil_wrapper import NeuralFoilAnalyzer

__all__ = ["AnalysisResult", "FlowConditions", "AnalysisConfidence", "NeuralFoilAnalyzer"]
