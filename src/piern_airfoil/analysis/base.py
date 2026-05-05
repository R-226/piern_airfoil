"""Base classes for airfoil analysis.

================================================================================
术语解释
================================================================================

气动系数 (Aerodynamic Coefficients):
    - CL (Lift Coefficient): 升力系数，衡量翼型产生的升力
        CL = L / (0.5 × ρ × V² × S)
        其中 L 是升力，ρ 是空气密度，V 是来流速度，S 是参考面积

    - CD (Drag Coefficient): 阻力系数，衡量翼型产生的阻力
        CD = D / (0.5 × ρ × V² × S)
        其中 D 是阻力

    - CM (Moment Coefficient): 力矩系数，衡量翼型产生的俯仰力矩
        CM = M / (0.5 × ρ × V² × S × c)
        其中 M 是力矩，c 是弦长

L/D (升阻比):
    衡量翼型气动效率的关键指标，L/D越大效率越高
    典型滑翔机翼型 L/D 可达 50-100

过渡位置 (Transition Location):
    边界层从层流转变为湍流的位置，通常用 x/c 表示
    - xtr_upper: 上表面过渡位置
    - xtr_lower: 下表面过渡位置
    提前过渡意味着更大的摩擦阻力

雷诺数 (Reynolds Number):
    Re = ρ × V × c / μ
    表征惯性力与粘性力之比的无量纲数
    Re > 5×10⁵ 通常认为需要考虑湍流边界层

攻角 (Angle of Attack / Alpha):
    翼型弦线与来流方向之间的夹角，单位：度
    临界攻角附近会出现失速 (stall)

================================================================================
分析结果 (AnalysisResult)
================================================================================

主要气动系数:
    CL, CD, CM - 气动系数（核心结果）
    Top_Xtr, Bot_Xtr - 过渡位置
    confidence - 分析置信度 [0, 1]

置信度等级:
    HIGH (≥0.8): 结果可靠，可直接使用
    MEDIUM (≥0.5): 结果可参考，建议结合其他信息
    LOW (≥0.2): 结果需谨慎对待
    UNKNOWN (<0.2): 置信度未知

来源标识:
    - "neuralfoil": NeuralFoil快速分析
    - "transolver": Transolver高保真CFD
    - "fused": 多保真度融合结果
    - "potential_flow": 势流理论计算
    - "mock": 模拟结果（开发测试用）

================================================================================
流场条件 (FlowConditions)
================================================================================

    alpha: 攻角 [度]，正值表示机头向上
    Re: 雷诺数 [1/m]，基于弦长的流动特征数
    Ma: 马赫数，可压缩效应（目前未充分考虑）
    n_crit: 临界放大因子，用于过渡预测（Michael-Hubble方法）
    xtr_upper, xtr_lower: 固定过渡位置（覆盖n_crit计算）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np


class AnalysisConfidence(Enum):
    """Analysis confidence level."""
    HIGH = "high"      # 置信度高，可直接使用
    MEDIUM = "medium"  # 置信度中等，建议参考
    LOW = "low"        # 置信度低，需要精确分析
    UNKNOWN = "unknown"  # 未知


@dataclass
class FlowConditions:
    """Flow conditions for analysis."""
    alpha: float = 0.0       # Angle of attack [deg]
    Re: float = 3e6          # Reynolds number
    Ma: float = 0.0           # Mach number (for compressible effects)
    n_crit: float = 9.0       # Critical amplification factor
    xtr_upper: float = 1.0   # Upper surface transition location [x/c]
    xtr_lower: float = 1.0   # Lower surface transition location [x/c]

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "Re": self.Re,
            "Ma": self.Ma,
            "n_crit": self.n_crit,
            "xtr_upper": self.xtr_upper,
            "xtr_lower": self.xtr_lower
        }


@dataclass
class AnalysisResult:
    """
    Unified result from airfoil analysis.

    Contains aerodynamic coefficients and metadata.
    """
    # Primary aerodynamic coefficients
    CL: float                  # Lift coefficient
    CD: float                  # Drag coefficient
    CM: float                  # Moment coefficient

    # Transition locations
    Top_Xtr: float = 0.0      # Upper surface transition [x/c]
    Bot_Xtr: float = 0.0      # Lower surface transition [x/c]

    # Confidence metrics
    confidence: float = 1.0   # Analysis confidence [0, 1]
    confidence_level: AnalysisConfidence = AnalysisConfidence.HIGH

    # Flow conditions used
    conditions: Optional[FlowConditions] = None

    # Additional data
    source: str = "unknown"   # "neuralfoil", "transolver", "fused", etc.

    # Boundary layer data (if available)
    upper_bl_theta: Optional[np.ndarray] = None  # Momentum thickness
    upper_bl_H: Optional[np.ndarray] = None     # Shape factor
    lower_bl_theta: Optional[np.ndarray] = None
    lower_bl_H: Optional[np.ndarray] = None

    # Error estimate (if available)
    error_estimate: Optional[Dict[str, float]] = None

    @property
    def lift_to_drag(self) -> float:
        """Lift-to-drag ratio."""
        if self.CD <= 0:
            return 0.0
        return self.CL / self.CD

    # Alias for backward compatibility
    @property
    def L_over_D(self) -> float:
        return self.lift_to_drag

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "CL": self.CL,
            "CD": self.CD,
            "CM": self.CM,
            "L/D": self.lift_to_drag,
            "Top_Xtr": self.Top_Xtr,
            "Bot_Xtr": self.Bot_Xtr,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "source": self.source,
        }

    def is_valid(self, min_CL: float = -2.0, max_CL: float = 3.0,
                  max_CD: float = 1.0) -> bool:
        """Check if result is physically reasonable."""
        if self.confidence < 0.3:
            return False
        if self.CL < min_CL or self.CL > max_CL:
            return False
        if self.CD < 0 or self.CD > max_CD:
            return False
        return True


@dataclass
class MultiFidelityResult:
    """
    Result from multi-fidelity analysis.

    Combines fast (NeuralFoil) and precise (Transolver) results.
    """
    fast_result: Optional[AnalysisResult] = None
    precise_result: Optional[AnalysisResult] = None
    fused_result: Optional[AnalysisResult] = None

    # Fusion metadata
    fusion_weight: float = 1.0  # Weight given to precise result [0, 1]
    fusion_reason: str = "initial"  # Why fusion was chosen

    @property
    def best_result(self) -> AnalysisResult:
        """Get the best available result."""
        if self.fused_result is not None:
            return self.fused_result
        if self.precise_result is not None:
            return self.precise_result
        if self.fast_result is not None:
            return self.fast_result
        raise ValueError("No result available")
