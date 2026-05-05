"""
翼型坐标拟合器 - 基于AeroSandbox的Kulfan参数提取

================================================================================
设计说明
================================================================================

本模块使用AeroSandbox的get_kulfan_parameters函数进行翼型坐标拟合。
AeroSandbox是MIT开发的航空优化框架，其Kulfan实现经过充分测试。

参数化维度: 18维 Kulfan/CST参数
    - upper_weights: 8个上表面权重 (indices 0-7)
    - lower_weights: 8个下表面权重 (indices 8-15)
    - leading_edge_weight: 1个前缘修改权重 (index 16)
    - te_thickness: 1个后缘厚度 (index 17)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, Dict

import numpy as np

from .base import AirfoilGeometry

# AeroSandbox imports
try:
    import aerosandbox as asb
    from aerosandbox.geometry.airfoil.airfoil_families import (
        get_kulfan_parameters,
        get_kulfan_coordinates,
    )
    HAS_AEROSANDBOX = True
except ImportError:
    HAS_AEROSANDBOX = False
    asb = None


def _check_aerosandbox():
    """检查AeroSandbox是否可用"""
    if not HAS_AEROSANDBOX:
        raise ImportError(
            "AeroSandbox is required for Kulfan fitting. "
            "Install with: pip install aerosandbox"
        )


@dataclass
class FitResult:
    """拟合结果容器"""
    geometry: AirfoilGeometry
    final_params: np.ndarray
    final_loss: float
    success: bool
    message: str
    method: str = "aerosandbox"


def fit_airfoil_coords(
    coords: np.ndarray,
    n_weights_per_side: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
    use_leading_edge_modification: bool = True,
    method: str = "least_squares"
) -> AirfoilGeometry:
    """
    将翼型坐标拟合为Kulfan (CST) 参数。

    Args:
        coords: 翼型坐标 (N, 2)
        n_weights_per_side: 每表面的Kulfan权重数量
        N1: Kulfan函数的N1参数
        N2: Kulfan函数的N2参数
        use_leading_edge_modification: 是否使用前缘修改
        method: 拟合方法，默认为 "least_squares"

    Returns:
        AirfoilGeometry对象

    示例:
        coords = np.loadtxt('airfoil.dat')
        geo = fit_airfoil_coords(coords)
    """
    _check_aerosandbox()

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        n = len(coords) // 2
        coords = coords.reshape(n, 2)

    params = get_kulfan_parameters(
        coordinates=coords,
        n_weights_per_side=n_weights_per_side,
        N1=N1,
        N2=N2,
        normalize_coordinates=True,
        use_leading_edge_modification=use_leading_edge_modification,
        method=method
    )

    return AirfoilGeometry(
        upper_weights=np.array(params['upper_weights']),
        lower_weights=np.array(params['lower_weights']),
        leading_edge_weight=float(params['leading_edge_weight']),
        te_thickness=float(params['TE_thickness'])
    )


def get_coordinates_from_geometry(
    geometry: AirfoilGeometry,
    n_points_per_side: int = 100,
    N1: float = 0.5,
    N2: float = 1.0
) -> np.ndarray:
    """
    从AirfoilGeometry生成翼型坐标。

    使用AeroSandbox的Kulfan坐标生成。

    Args:
        geometry: AirfoilGeometry对象
        n_points_per_side: 每表面的点数
        N1: Kulfan函数的N1参数
        N2: Kulfan函数的N2参数

    Returns:
        翼型坐标 (2*n_points_per_side-1, 2)
    """
    _check_aerosandbox()

    return get_kulfan_coordinates(
        upper_weights=geometry.upper_weights,
        lower_weights=geometry.lower_weights,
        leading_edge_weight=geometry.leading_edge_weight,
        TE_thickness=geometry.te_thickness,
        n_points_per_side=n_points_per_side,
        N1=N1,
        N2=N2
    )


def fit_airfoil_with_quality(
    coords: np.ndarray,
    N1: float = 0.5,
    N2: float = 1.0
) -> Tuple[AirfoilGeometry, Dict[str, float]]:
    """
    拟合并评估质量。

    Args:
        coords: 翼型坐标 (N, 2)
        N1: Kulfan函数的N1参数
        N2: Kulfan函数的N2参数

    Returns:
        Tuple of (AirfoilGeometry, quality_dict)

    quality_dict包含:
        - mse_upper, mse_lower, mse_total
        - max_error_upper, max_error_lower, max_error_total
        - thickness, normalized_mse
    """
    _check_aerosandbox()

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        n = len(coords) // 2
        coords = coords.reshape(n, 2)

    # 获取归一化坐标（AeroSandbox自动处理）
    asb_af = asb.Airfoil(name="Target", coordinates=coords)
    normalized = asb_af.coordinates

    # 拟合
    geo = fit_airfoil_coords(coords, N1=N1, N2=N2)

    # 重建
    fitted = get_coordinates_from_geometry(geo, N1=N1, N2=N2)

    # 计算质量指标
    # 注意：normalized可能有多余数据（如多圈轮廓），用fitted的LE来对齐
    le_idx_fitted = np.argmin(fitted[:, 0])
    n_upper_fitted = le_idx_fitted + 1

    # 使用AeroSandbox归一化后的坐标作为参考
    # 由于原始数据可能有非标准顺序，我们直接比较重建质量
    # 通过插值到相同的x网格上来比较

    # Upper surface: x从1到0（TE到LE）
    x_u_f = fitted[:n_upper_fitted, 0][::-1]  # 0到1
    y_u_f = fitted[:n_upper_fitted, 1][::-1]

    # 对normalized的upper surface (TE到LE)插值到fitted的x上
    x_u_n = normalized[:le_idx_fitted+1, 0][::-1] if le_idx_fitted < len(normalized) else normalized[:len(normalized)//2, 0][::-1]
    y_u_n = normalized[:len(x_u_n), 1][::-1] if len(x_u_n) <= len(normalized) else normalized[:le_idx_fitted+1, 1][::-1]

    # 确保x_u_n和x_u_f方向一致
    if len(x_u_n) > 0 and len(x_u_f) > 0:
        y_u_n_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(x_u_n)), y_u_n)
        y_u_f_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(x_u_f)), y_u_f)
        mse_upper = float(np.mean((y_u_n_interp - y_u_f_interp) ** 2))
        max_u = float(np.max(np.abs(y_u_n_interp - y_u_f_interp)))
    else:
        mse_upper = 0.0
        max_u = 0.0

    # Lower surface: x从0到1（LE到TE）
    x_l_f = fitted[le_idx_fitted:, 0]
    y_l_f = fitted[le_idx_fitted:, 1]

    # 取normalized的后半部分作为lower surface
    n_lower_start = len(normalized) // 2
    x_l_n = normalized[n_lower_start:, 0]
    y_l_n = normalized[n_lower_start:, 1]

    if len(x_l_n) > 0 and len(x_l_f) > 0:
        y_l_n_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(x_l_n)), y_l_n)
        y_l_f_interp = np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, len(x_l_f)), y_l_f)
        mse_lower = float(np.mean((y_l_n_interp - y_l_f_interp) ** 2))
        max_l = float(np.max(np.abs(y_l_n_interp - y_l_f_interp)))
    else:
        mse_lower = 0.0
        max_l = 0.0

    thickness = float(normalized[:, 1].max() - normalized[:, 1].min())
    mse_total = (mse_upper + mse_lower) / 2

    return geo, {
        'mse_upper': mse_upper,
        'mse_lower': mse_lower,
        'mse_total': mse_total,
        'max_error_upper': max_u,
        'max_error_lower': max_l,
        'max_error_total': max(max_u, max_l),
        'thickness': thickness,
        'normalized_mse': mse_total / thickness**2 if thickness > 0 else 0.0,
    }

def fit_naca_airfoil(naca_digits: str, n_points: int = 201) -> AirfoilGeometry:
    """
    拟合NACA标准翼型。

    Args:
        naca_digits: NACA翼型代号，如 '4412'
        n_points: 采样点数

    Returns:
        AirfoilGeometry对象
    """
    _check_aerosandbox()

    af = asb.Airfoil(f'naca{naca_digits}')
    return fit_airfoil_coords(af.coordinates, n_points_per_side=n_points // 2)

# =============================================================================
# 文件加载
# =============================================================================

def load_contour_dat(filepath: Union[str, Path]) -> np.ndarray:
    """加载 contour .dat 格式（通用格式）"""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        try:
            float(first_line.split()[0])
            skiprows = 0
        except (ValueError, IndexError):
            skiprows = 1
    return np.loadtxt(filepath, skiprows=skiprows)


def load_selig_format(filepath: Union[str, Path]) -> np.ndarray:
    """加载 Selig 格式的翼型坐标文件 (UIUC数据库常用)"""
    return np.loadtxt(filepath, skiprows=1)


def load_xfoil_format(filepath: Union[str, Path]) -> np.ndarray:
    """加载 XFOIL/AFLR 格式的翼型坐标文件"""
    return np.loadtxt(filepath)


class AirfoilFitter:
    """
    翼型拟合器类。

    示例:
        fitter = AirfoilFitter()
        result = fitter.fit(coords)
        geo = result.geometry
    """

    def __init__(self, n_points: int = 201, N1: float = 0.5, N2: float = 1.0):
        self.n_points = n_points
        self.N1 = N1
        self.N2 = N2

    def fit(self, coords: np.ndarray) -> FitResult:
        """拟合翼型坐标"""
        geo, quality = fit_airfoil_with_quality(coords, N1=self.N1, N2=self.N2)
        return FitResult(
            geometry=geo,
            final_params=geo.to_array(),
            final_loss=quality['mse_total'],
            success=True,
            message="AeroSandbox Kulfan least-squares fit"
        )
    
if __name__ == "__main__":
    # 示例用法
    import matplotlib.pyplot as plt
    coords = load_contour_dat('contour_points.dat')
    # 绘制原始翼型
    plt.figure(figsize=(10, 5))
    plt.plot(coords[:, 0], coords[:, 1], label="Existing Airfoil")
    plt.title("Existing Airfoil from contour_points.dat")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.savefig("kulfan_airfoil.png")
    plt.show()
    fitter = AirfoilFitter()
    result = fitter.fit(coords)
    coords_fitted = get_coordinates_from_geometry(result.geometry, n_points_per_side=100)
    # 绘制拟合结果
    plt.figure(figsize=(10, 5))
    plt.plot(coords[:, 0], coords[:, 1], label="Existing Airfoil", linestyle='--')
    plt.plot(coords_fitted[:, 0], coords_fitted[:, 1], label="fitted Airfoil")
    plt.title("Kulfan Fit to Existing Airfoil")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.savefig("kulfan_fit.png")
    plt.show()