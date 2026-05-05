"""
Base classes for airfoil parameterization.

================================================================================
术语解释
================================================================================

CST (Class-Shape Transformation) / Kulfan参数化:
    一种将翼型几何表示为"类函数 × 形状函数"乘积的参数化方法。
    - "类函数(Class function)"定义翼型的大致轮廓形状
    - "形状函数(Shape function)"使用Bernstein多项式定义上下表面的细节形状
    - 每表面8个权重，共计16个权重，加上前缘权重和后缘厚度共18个参数

Kulfan权重 (Kulfan Parameters):
    描述翼型上下表面形状的8个权重。上表面和下表面各有8个权重。
    权重值通常在[-1, 1]范围内，描述了该位置的"偏距"程度。

AirfoilGeometry:
    统一的翼型几何表示，内部使用CST参数，可转换为坐标点用于可视化。

================================================================================
参数格式 (18维)
================================================================================

    indices 0-7:   upper_weights (上表面8个Kulfan权重)
    indices 8-15:  lower_weights (下表面8个Kulfan权重)
    index 16:      leading_edge_weight (前缘权重，控制前缘形状)
    index 17:      te_thickness (后缘厚度，翼型后缘的开合程度)

示例:
    params = [0.1, 0.2, -0.1, 0.05, ...]  # 18个float值
    geo = AirfoilGeometry.from_array(params)

================================================================================
翼型坐标生成流程
================================================================================

    坐标 → fit() → AirfoilGeometry → get_coordinates() → 坐标点

实现委托:
    - fit() 委托给 fitting.py 的 AeroSandbox get_kulfan_parameters
    - get_coordinates() 委托给 fitting.py 的 AeroSandbox get_kulfan_coordinates
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class AirfoilGeometry:
    """
    统一的翼型几何表示。

    内部使用CST (Class-Shape Transformation) 参数化方法存储翼型形状。
    可以转换为18维参数数组或(N, 2)的坐标点用于可视化。

    属性:
        upper_weights: 上表面8个Kulfan权重，定义翼型上表面的形状
        lower_weights: 下表面8个Kulfan权重，定义翼型下表面的形状
        leading_edge_weight: 前缘权重，控制翼型前缘的尖锐程度
            - 值越大，前缘越尖锐
            - 通常在[-0.5, 0.5]范围内
        te_thickness: 后缘厚度，控制翼型后缘的开合程度
            - 值越大，后缘越厚，翼型越"钝"
            - 通常在[0, 0.1]范围内（相对于弦长）
        coordinates: 缓存的坐标点 (N, 2)，避免重复计算

    示例:
        # 从18维参数数组创建
        params = np.random.randn(18)
        geo = AirfoilGeometry.from_array(params)

        # 随机生成翼型
        geo = AirfoilGeometry.random(seed=42)

        # 转换为参数数组
        arr = geo.to_array()  # shape: (18,)

        # 获取坐标点用于可视化
        coords = CSTParameterization().get_coordinates(geo)  # shape: (199, 2)
    """
    # CST parameters (18 dimensions)
    upper_weights: np.ndarray  # shape: (8,) - 上表面Kulfan权重
    lower_weights: np.ndarray  # shape: (8,) - 下表面Kulfan权重
    leading_edge_weight: float  # 前缘权重，控制前缘尖锐程度
    te_thickness: float  # 后缘厚度，相对弦长的比例

    # Normalized coordinates (for visualization)
    coordinates: Optional[np.ndarray] = None  # shape: (N, 2), 缓存坐标

    @property
    def n_params(self) -> int:
        return 18

    def to_array(self) -> np.ndarray:
        """Convert to parameter array."""
        return np.concatenate([
            self.upper_weights,
            self.lower_weights,
            [self.leading_edge_weight, self.te_thickness]
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "AirfoilGeometry":
        """Create from parameter array."""
        if len(arr) != 18:
            raise ValueError(f"Expected 18 parameters, got {len(arr)}")
        return cls(
            upper_weights=arr[:8],
            lower_weights=arr[8:16],
            leading_edge_weight=arr[16],
            te_thickness=arr[17]
        )

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "AirfoilGeometry":
        """Generate random airfoil parameters."""
        if seed is not None:
            np.random.seed(seed)
        return cls(
            upper_weights=np.random.randn(8) * 0.1,
            lower_weights=np.random.randn(8) * 0.1,
            leading_edge_weight=np.random.randn() * 0.05,
            te_thickness=np.random.rand() * 0.02
        )


class Parameterization(ABC):
    """Abstract base class for airfoil parameterization."""

    @abstractmethod
    def params_to_geometry(self, params: np.ndarray) -> AirfoilGeometry:
        """Convert parameter array to geometry."""
        pass

    @abstractmethod
    def geometry_to_params(self, geometry: AirfoilGeometry) -> np.ndarray:
        """Convert geometry to parameter array."""
        pass

    @abstractmethod
    def get_coordinates(self, geometry: AirfoilGeometry, n_points: int = 201) -> np.ndarray:
        """Get (x, y) coordinates for visualization."""
        pass

    @abstractmethod
    def validate(self, params: np.ndarray) -> Tuple[bool, str]:
        """Validate parameters. Returns (is_valid, message)."""
        pass


class CSTParameterization(Parameterization):
    """
    CST (Class-Shape Transformation) 参数化方法，也称为Kulfan参数化。

    翼型设计领域最广泛使用的参数化方法之一。

    参数数量: 18个
        - 上表面权重: 8个
        - 下表面权重: 8个
        - 前缘权重: 1个
        - 后缘厚度: 1个

    实现:
        fit() 和 get_coordinates() 委托给 fitting.py 的 AeroSandbox 实现。

    使用示例:
        param = CSTParameterization()
        geo = param.fit(coords)  # 坐标 → AirfoilGeometry
        coords_out = param.get_coordinates(geo)  # AirfoilGeometry → 坐标
    """

    def __init__(self, n_weights_per_side: int = 8):
        """
        初始化CST参数化器。

        Args:
            n_weights_per_side: 每表面的权重数量，默认8
        """
        self.n_weights_per_side = n_weights_per_side

    def params_to_geometry(self, params: np.ndarray) -> AirfoilGeometry:
        if len(params) != 18:
            raise ValueError(f"CST requires 18 parameters, got {len(params)}")
        return AirfoilGeometry(
            upper_weights=params[:8],
            lower_weights=params[8:16],
            leading_edge_weight=params[16],
            te_thickness=params[17]
        )

    def geometry_to_params(self, geometry: AirfoilGeometry) -> np.ndarray:
        return geometry.to_array()

    def get_coordinates(self, geometry: AirfoilGeometry, n_points: int = 201) -> np.ndarray:
        """
        Generate airfoil coordinates from CST parameters.

        Uses AeroSandbox's Kulfan implementation for high-precision results.
        """
        # Delegate to AeroSandbox implementation in fitting.py
        from .fitting import get_coordinates_from_geometry as _get_coords
        return _get_coords(geometry, n_points_per_side=n_points // 2)

    def validate(self, params: np.ndarray) -> Tuple[bool, str]:
        """Validate CST parameters."""
        if len(params) != 18:
            return False, f"Expected 18 parameters, got {len(params)}"

        # Check weight magnitudes
        if np.any(np.abs(params[:16]) > 1.0):
            return False, "Weights should be in range [-1, 1]"

        # Check thickness
        if params[17] < 0 or params[17] > 0.1:
            return False, "TE thickness should be in range [0, 0.1]"

        return True, "Valid"

    def random(self, seed: int | None = None) -> AirfoilGeometry:
        """Generate random airfoil geometry."""
        return AirfoilGeometry.random(seed=seed)

    def fit(self, coords: np.ndarray) -> AirfoilGeometry:
        """
        Fit airfoil coordinates to CST parameters.

        Delegates to fitting.py's AeroSandbox implementation.
        """
        from .fitting import fit_airfoil_coords as _fit
        return _fit(coords, n_weights_per_side=self.n_weights_per_side)
