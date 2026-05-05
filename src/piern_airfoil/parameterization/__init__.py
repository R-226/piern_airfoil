"""Parameterization module for airfoil geometry representation."""

from .base import Parameterization, CSTParameterization, AirfoilGeometry
from .validity import AirfoilValidator, ValidityCheckResult
from .fitting import (
    AirfoilFitter,
    FitResult,
    fit_airfoil_coords,
    fit_airfoil_with_quality,
    fit_naca_airfoil,
    get_coordinates_from_geometry,
    load_selig_format,
    load_xfoil_format,
    load_contour_dat,
)

__all__ = [
    "Parameterization",
    "CSTParameterization",
    "AirfoilGeometry",
    "ValidityCheckResult",
    "AirfoilValidator",
    "AirfoilFitter",
    "FitResult",
    "fit_airfoil_coords",
    "fit_airfoil_with_quality",
    "fit_naca_airfoil",
    "get_coordinates_from_geometry",
    "load_selig_format",
    "load_xfoil_format",
    "load_contour_dat",
]
