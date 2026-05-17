"""
Unified constraint interface for multi-fidelity airfoil optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import aerosandbox as asb


class FidelityLevel(Enum):
    """Fidelity level for aerodynamic analysis."""
    THIN = "thin"       # ~1ms, classical thin airfoil theory
    NEURAL = "neural"   # ~50-200ms, NeuralFoil neural network


@dataclass
class AirfoilConstraints:
    """
    Unified constraints applicable at any fidelity level.

    Geometry constraints (thickness, TE angle) are always enforced since they
    depend only on the airfoil shape, not the solver.

    Aero constraints (confidence) are only enforced at NEURAL fidelity since
    thin airfoil theory does not provide confidence estimates.
    """
    CL_targets: np.ndarray | None = None
    CL_weights: np.ndarray | None = None
    CM_min: float = -0.133
    thickness_at_33_min: float = 0.128
    thickness_at_90_min: float = 0.014
    TE_angle_min: float = 6.03
    confidence_min: float = 0.90
    max_wiggliness_ratio: float = 2.0

    def evaluate_geometry(self, airfoil: "asb.KulfanAirfoil") -> list[float]:
        """
        Evaluate geometry-only constraints (fidelity-independent).

        Returns list of constraint violations (negative = feasible).
        """
        violations = []

        t33 = float(np.asarray(airfoil.local_thickness(x_over_c=0.33)).flatten()[0])
        violations.append(self.thickness_at_33_min - t33)

        t90 = float(np.asarray(airfoil.local_thickness(x_over_c=0.90)).flatten()[0])
        violations.append(self.thickness_at_90_min - t90)

        te = float(np.asarray(airfoil.TE_angle()).flatten()[0])
        violations.append(self.TE_angle_min - te)

        t_max = float(np.asarray(airfoil.local_thickness()).flatten()[0])
        violations.append(-t_max)  # thickness > 0 → violation = -thickness < 0

        return violations

    def evaluate_aero(
        self,
        aero: dict,
        fidelity: FidelityLevel,
        CL_target: float | None = None,
    ) -> list[float]:
        """
        Evaluate aerodynamic constraints.

        Returns list of constraint violations (negative = feasible).
        """
        violations = []

        CL = float(np.asarray(aero["CL"]).flatten()[0])
        CM = float(np.asarray(aero["CM"]).flatten()[0])

        if CL_target is not None:
            violations.append(abs(CL - CL_target) - 0.01)  # small tolerance

        violations.append(self.CM_min - CM)

        if fidelity == FidelityLevel.NEURAL:
            confidence = float(np.asarray(aero["analysis_confidence"]).flatten()[0])
            violations.append(self.confidence_min - confidence)

        return violations

    def evaluate(
        self,
        airfoil: "asb.KulfanAirfoil",
        aero: dict,
        fidelity: FidelityLevel,
        CL_target: float | None = None,
    ) -> list[float]:
        """
        Evaluate all constraints.

        Returns list of constraint violations (negative = feasible).
        """
        violations = self.evaluate_geometry(airfoil)
        violations.extend(self.evaluate_aero(aero, fidelity, CL_target))
        return violations

    def penalty(
        self,
        airfoil: "asb.KulfanAirfoil",
        aero: dict,
        fidelity: FidelityLevel,
        CL_target: float | None = None,
        scale: float = 100.0,
    ) -> float:
        """
        Compute total penalty from constraint violations.

        Returns 0 if all constraints are satisfied, positive otherwise.
        """
        violations = self.evaluate(airfoil, aero, fidelity, CL_target)
        return scale * sum(max(0, v) ** 2 for v in violations)
