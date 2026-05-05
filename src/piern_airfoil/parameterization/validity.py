"""Airfoil validity checker."""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class ValidityCheckResult:
    """Result of airfoil validity check."""
    is_valid: bool
    message: str
    thickness_ratio: float = 0.0
    max_camber_ratio: float = 0.0
    violations: List[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class AirfoilValidator:
    """
    Validates airfoil geometry for aerodynamic and manufacturing constraints.

    Common constraints:
    - Minimum thickness (for structural requirements)
    - Maximum thickness (for drag requirements)
    - Maximum camber (for stall behavior)
    - Leading edge radius (for stall characteristics)
    - Trailing edge angle (for manufacturing)
    """

    def __init__(
        self,
        min_thickness: float = 0.03,  # 3% chord minimum
        max_thickness: float = 0.25,  # 25% chord maximum
        max_camber: float = 0.10,     # 10% chord maximum
        min_le_radius: float = 0.005,  # 0.5% chord minimum
    ):
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.max_camber = max_camber
        self.min_le_radius = min_le_radius

    def validate(self, coordinates: np.ndarray) -> ValidityCheckResult:
        """
        Validate airfoil coordinates.

        Args:
            coordinates: Array of shape (N, 2) with x, y coordinates

        Returns:
            ValidityCheckResult with validation details
        """
        violations = []

        # Check basic properties
        if len(coordinates) < 10:
            return ValidityCheckResult(
                is_valid=False,
                message="Too few points for airfoil",
                violations=["insufficient_points"]
            )

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        # Check x range
        if x.max() > 1.1 or x.min() < -0.1:
            violations.append("x coordinates out of expected range [0, 1]")

        # Find leading edge (minimum x)
        le_idx = np.argmin(x)
        le_x, le_y = x[le_idx], y[le_idx]

        # Find trailing edge
        te_mask = (x > 0.95)
        if np.any(te_mask):
            te_y_upper = y[:le_idx][x[:le_idx] > 0.95].mean() if np.any(x[:le_idx] > 0.95) else 0
            te_y_lower = y[le_idx:][x[le_idx:] > 0.95].mean() if np.any(x[le_idx:] > 0.95) else 0
            te_thickness = abs(te_y_upper - te_y_lower)
        else:
            te_thickness = abs(y[0] - y[-1]) if len(y) > 1 else 0

        # Compute thickness distribution
        upper_y, lower_y = self._split_surfaces(x, y, le_idx)

        # Maximum thickness
        thickness = upper_y - lower_y
        max_thickness_val = thickness.max()
        max_thickness_loc = x[le_idx:le_idx+len(thickness)][np.argmax(thickness)]

        # Maximum camber (of camber line)
        camber = (upper_y + lower_y) / 2
        max_camber_val = camber.max()
        min_camber_val = camber.min()

        # Leading edge radius approximation
        le_radius = self._estimate_le_radius(coordinates, le_idx)

        # Check constraints
        if max_thickness_val < self.min_thickness:
            violations.append(f"Thickness {max_thickness_val:.3f} below minimum {self.min_thickness:.3f}")

        if max_thickness_val > self.max_thickness:
            violations.append(f"Thickness {max_thickness_val:.3f} above maximum {self.max_thickness:.3f}")

        if max(abs(max_camber_val), abs(min_camber_val)) > self.max_camber:
            violations.append(f"Camber {max(abs(max_camber_val), abs(min_camber_val)):.3f} above maximum {self.max_camber:.3f}")

        if le_radius < self.min_le_radius:
            violations.append(f"LE radius {le_radius:.5f} below minimum {self.min_le_radius:.5f}")

        is_valid = len(violations) == 0

        return ValidityCheckResult(
            is_valid=is_valid,
            message="Valid" if is_valid else f"Invalid: {', '.join(violations)}",
            thickness_ratio=max_thickness_val,
            max_camber_ratio=max(abs(max_camber_val), abs(min_camber_val)),
            violations=violations
        )

    def _split_surfaces(self, x: np.ndarray, y: np.ndarray, le_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split coordinates into upper and lower surfaces."""
        # Upper surface: from LE (min x) going "up" (positive y direction)
        # Lower surface: from LE going "down"
        upper_x, upper_y = x[:le_idx+1], y[:le_idx+1]
        lower_x, lower_y = x[le_idx:], y[le_idx:]

        # Sort by x
        upper_sort = np.argsort(upper_x)
        lower_sort = np.argsort(lower_x)

        return np.interp(np.linspace(0, 1, 100), upper_x[upper_sort], upper_y[upper_sort]), \
               np.interp(np.linspace(0, 1, 100), lower_x[lower_sort], lower_y[lower_sort])

    def _estimate_le_radius(self, coords: np.ndarray, le_idx: int, window: int = 5) -> float:
        """Estimate leading edge radius using local curvature."""
        # Simple approximation: fit a circle to points near LE
        start = max(0, le_idx - window)
        end = min(len(coords), le_idx + window + 1)

        le_points = coords[start:end]

        if len(le_points) < 3:
            return 0.01  # Default small radius

        # Fit circle using 3 points
        try:
            p1, p2, p3 = le_points[0], le_points[len(le_points)//2], le_points[-1]

            # Circumcircle calculation
            ax, ay = p1[0], p1[1]
            bx, by = p2[0], p2[1]
            cx, cy = p3[0], p3[1]

            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

            if abs(d) < 1e-10:
                return 0.01

            ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
            uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d

            radius = np.sqrt((p1[0] - ux)**2 + (p1[1] - uy)**2)
            return radius

        except Exception:
            return 0.01  # Default small radius on error