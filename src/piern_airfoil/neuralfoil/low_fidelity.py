from dataclasses import dataclass
from typing import Optional
import aerosandbox.numpy as np
import aerosandbox.tools.pretty_plots as p
import aerosandbox as asb


@dataclass
class AnalysisResult:
    """Result from low-fidelity airfoil analysis."""
    CL: float
    CD: float
    CM: float
    confidence: float
    CL_func: Optional[any] = None  # for aero.derivative


@dataclass
class AirfoilGeometry:
    """Airfoil geometry parameters."""
    upper_weights: np.ndarray
    lower_weights: np.ndarray
    leading_edge_weight: float
    te_thickness: float = 0.0

'''
Low-fidelity airfoil optimization using Aerosandbox's KulfanAirfoil and Opti framework.
'''

class LowFidelityOptimizer():
    def __init__(self, airfoil: asb.KulfanAirfoil, CL_targets: np.array[float], CL_weights: np.array[float], TE_thickness: float = 0.0, alpha: Optional[asb.OptiVariable] = None, RE: Optional[asb.OptiVariable] = None, mach: Optional[asb.OptiVariable] = None, aoa_low_bound: float = -5, aoa_high_bound: float = 18):
        self.airfoil = airfoil
        self.CL_targets = CL_targets
        self.CL_weights = CL_weights
        self.TE_thickness = TE_thickness
        self.RE = RE
        self.mach = mach
        self.aoa_low_bound = aoa_low_bound
        self.aoa_high_bound = aoa_high_bound
        self.opti = asb.Opti()
        self.last_sol = None
        self._build_problem()

    def _build_problem(self):
        self.optimized_airfoil = asb.KulfanAirfoil(
            name="Optimized",
            lower_weights=self.opti.variable(
                init_guess=self.airfoil.lower_weights,
                lower_bound=-0.5,
                upper_bound=0.25,
            ),
            upper_weights=self.opti.variable(
                init_guess=self.airfoil.upper_weights,
                lower_bound=-0.25,
                upper_bound=0.5,
            ),
            leading_edge_weight=self.opti.variable(
                init_guess=self.airfoil.leading_edge_weight,
                lower_bound=-1,
                upper_bound=1,
            ),
            TE_thickness=0,
        )

        self.alpha = self.opti.variable(
            init_guess=np.degrees(self.CL_targets / (2 * np.pi)),
            lower_bound=self.aoa_low_bound,
            upper_bound=self.aoa_high_bound,
        )

        self.aero = self.optimized_airfoil.get_aero_from_neuralfoil(
            alpha=self.alpha,
            Re=self.RE,
            mach=self.mach,
        )

        self.opti.subject_to(
            [
                self.aero["analysis_confidence"] > 0.90,
                self.aero["CL"] == self.CL_targets,
                np.diff(self.alpha) > 0,
                self.aero["CM"] >= -0.133,
                self.optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
                self.optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
                self.optimized_airfoil.TE_angle()
                >= 6.03,  # Modified from Drela's 6.25 to match DAE-11 case
                self.optimized_airfoil.lower_weights[0] < -0.05,
                self.optimized_airfoil.upper_weights[0] > 0.05,
                self.optimized_airfoil.local_thickness() > 0,
            ]
        )

        get_wiggliness = lambda af: sum(
            [
                np.sum(np.diff(np.diff(array)) ** 2)
                for array in [af.lower_weights, af.upper_weights]
            ]
        )

        self.opti.subject_to(
            get_wiggliness(self.optimized_airfoil) < 2 * get_wiggliness(self.airfoil)
        )

        self.opti.minimize(np.mean(self.aero["CD"] * self.CL_weights))


    def update(self):
        if self.last_sol is not None:
            self.opti.set_initial_from_sol(
                self.last_sol,
                initialize_primals=True,
                initialize_duals=True
            )

        sol = self.opti.solve(
            behavior_on_failure="return_last",
        )

        self.optimized_airfoil = sol(self.optimized_airfoil)
        self.aero = sol(self.aero)
        self.airfoil = self.optimized_airfoil
        self.last_sol = sol


if __name__ == "__main__":
    airfoil = asb.KulfanAirfoil('naca0012')
    CL_multipoint_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_multipoint_weights = np.array([5, 6, 7, 8, 9, 10])
    Re = 500e3 * (CL_multipoint_targets / 1.25) ** -0.5
    mach = 0.03
    Low_optimizer = LowFidelityOptimizer(airfoil=airfoil, CL_targets=CL_multipoint_targets, CL_weights=CL_multipoint_weights, RE=Re, mach=mach)
    
    import matplotlib.pyplot as plt

    Low_optimizer.update()
    fig, ax = plt.subplots(figsize=(6, 2))
    Low_optimizer.airfoil.draw()
    plt.savefig(f"test.png", dpi=300)
