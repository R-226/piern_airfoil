"""Shared weighted CD evaluation using NeuralFoil.

Used by hierarchical optimizer, pipeline, router training, and benchmark.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def evaluate_weighted_cd(
    airfoil,
    CL_targets: np.ndarray,
    Re: np.ndarray,
    CL_weights: np.ndarray,
    mach: float = 0.03,
    alpha_range: tuple[float, float] = (-5.0, 18.0),
) -> float:
    """Evaluate weighted CD across multiple CL design points.

    For each CL target, finds the angle of attack that produces that CL
    using NeuralFoil, then sums weighted CD values.

    Args:
        airfoil: asb.KulfanAirfoil instance.
        CL_targets: target CL values for each design point.
        Re: Reynolds numbers for each design point.
        CL_weights: optimization weights for each design point.
        mach: Mach number.
        alpha_range: (min, max) angle of attack in degrees for brentq search.

    Returns:
        Total weighted CD (scalar).
    """
    total_cd = 0.0
    for cl_t, re_i, w_i in zip(CL_targets, Re, CL_weights):

        def residual(a, _af=airfoil, _re=re_i, _cl=cl_t):
            aero = _af.get_aero_from_neuralfoil(
                alpha=a, Re=float(_re), mach=mach
            )
            return float(np.asarray(aero["CL"]).flatten()[0]) - _cl

        try:
            alpha_i = brentq(residual, alpha_range[0], alpha_range[1], xtol=0.01, maxiter=30)
        except (ValueError, RuntimeError):
            alpha_i = 5.0

        aero = airfoil.get_aero_from_neuralfoil(
            alpha=alpha_i, Re=float(re_i), mach=mach
        )
        cd = float(np.asarray(aero["CD"]).flatten()[0])
        total_cd += cd * w_i

    return total_cd
