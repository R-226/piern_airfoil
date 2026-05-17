"""
Thin Airfoil Theory Solver for rapid aerodynamic predictions.

Based on classical thin airfoil theory:
- Uses vortex sheet distribution along chord
- Analytical solution for symmetric airfoils
- Approximate solution for cambered airfoils via lifting line

Speed: ~1ms per evaluation (extremely fast)
Accuracy: Approximate - valid for thin airfoils at small angles
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import NamedTuple


class ThinAirfoilResult(NamedTuple):
    """Result from thin airfoil theory analysis."""
    CL: float
    CD: float  # Induced drag only (CDi = CL^2 / (pi * AR))
    CM: float
    Cp_upper: np.ndarray
    Cp_lower: np.ndarray
    x_chord: np.ndarray


def thin_airfoil_analysis(
    contour_x: np.ndarray,
    contour_y: np.ndarray,
    alpha: float = 0.0,
    mach: float = 0.0,
    AR: float = 6.0  # Aspect ratio for induced drag
) -> ThinAirfoilResult:
    """
    Analyze airfoil using thin airfoil theory.

    Args:
        contour_x: x-coordinates of airfoil contour (normalized, 0 to 1)
        contour_y: y-coordinates of airfoil contour
        alpha: angle of attack in degrees
        mach: Mach number (for compressibility correction)
        AR: aspect ratio (used for induced drag calculation)

    Returns:
        ThinAirfoilResult with CL, CD, CM and surface Cp distributions
    """
    alpha_rad = np.radians(alpha)

    # Separate upper and lower surfaces by y-sign at each x
    # Find unique x values and corresponding upper/lower y
    x_unique = np.unique(contour_x)
    y_upper = np.zeros_like(x_unique)
    y_lower = np.zeros_like(x_unique)

    for i, x in enumerate(x_unique):
        mask = np.isclose(contour_x, x)
        y_vals = contour_y[mask]
        y_upper[i] = np.max(y_vals)  # Upper surface (positive y)
        y_lower[i] = np.min(y_vals)  # Lower surface (negative y)

    # Sort by x
    sort_idx = np.argsort(x_unique)
    x_unique = x_unique[sort_idx]
    y_upper = y_upper[sort_idx]
    y_lower = y_lower[sort_idx]

    # Compute camber line: y_c = (y_upper + y_lower) / 2
    y_camber = (y_upper + y_lower) / 2

    # Compute thickness: h = y_upper - y_lower
    thickness = y_upper - y_lower

    # Thin airfoil theory: CL = 2 * pi * (alpha + theta_0)
    # where theta_0 is the angle of attack due to camber at quarter-chord

    # For a symmetric airfoil (camber = 0): CL = 2 * pi * alpha
    # For a cambered airfoil, use approximate formula:
    # CL = 2 * pi * (alpha + alpha_0)
    # where alpha_0 is the zero-lift angle of attack (from camber)

    # Approximate alpha_0 from camber line slope at x = 0.25
    # dyc/dx at x = 0.25 approximates the zero-lift angle
    if len(x_unique) >= 5:
        # Interpolate camber line
        f_camber = interp1d(x_unique, y_camber, kind='cubic', fill_value='extrapolate')
        x_test = np.array([0.2, 0.3])
        dyc_dx = np.gradient(f_camber(x_test), x_test).mean()
        alpha_0 = np.degrees(np.arctan(dyc_dx))
    else:
        alpha_0 = 0.0

    # Prandtl-Glauert compressibility correction for Mach
    if mach > 0:
        beta = np.sqrt(1 - mach**2)
        corr_factor = 1 / beta
    else:
        corr_factor = 1.0

    # CL = 2 * pi * (alpha_rad + alpha_0_rad) * corr
    alpha_0_rad = np.radians(alpha_0)
    CL = 2 * np.pi * (alpha_rad + alpha_0_rad) * corr_factor

    # Induced drag (from lifting line theory): CDi = CL^2 / (pi * AR)
    CD = CL**2 / (np.pi * AR)

    # CM about quarter-chord (approximation)
    # For symmetric airfoil: CM ~ -0.25 * CL
    # For cambered: add moment due to camber
    CM = -0.25 * CL - 0.01 * alpha_0  # Approximate

    # Compute surface Cp distribution
    # Using: Cp = -2 * (V/Vinf - 1) for thin airfoil
    # For small angles: V/Vinf ~ 1 + dyc/dx * alpha
    # Simplified: Cp_u = -2*(alpha + theta_u), Cp_l = -2*(alpha + theta_l)

    # Generate chordwise distribution
    x_chord = np.linspace(0, 1, 80)

    # Interpolate upper/lower surfaces to common x distribution
    f_upper = interp1d(x_unique, y_upper, kind='linear', fill_value='extrapolate')
    f_lower = interp1d(x_unique, y_lower, kind='linear', fill_value='extrapolate')

    yu = f_upper(x_chord)
    yl = f_lower(x_chord)

    # Approximate surface slope: theta = arctan(dy/dx)
    # For thin airfoil: Cp = -2 * (alpha + theta)
    # Upper surface: theta > 0 for positive alpha
    # Lower surface: theta < 0 for positive alpha
    dydx_u = np.gradient(yu, x_chord)
    dydx_l = np.gradient(yl, x_chord)

    theta_u = np.arctan(dydx_u)
    theta_l = np.arctan(dydx_l)

    # Cp = -2 * (alpha_rad + theta)
    # Upper surface has negative Cp (suction) for positive alpha
    Cp_upper = -2 * (alpha_rad + theta_u)
    Cp_lower = -2 * (alpha_rad + theta_l)

    return ThinAirfoilResult(
        CL=float(CL),
        CD=float(CD),
        CM=float(CM),
        Cp_upper=Cp_upper,
        Cp_lower=Cp_lower,
        x_chord=x_chord
    )


def thin_airfoil_from_kulfan(
    airfoil,
    alpha: float = 0.0,
    mach: float = 0.0,
    AR: float = 6.0,
) -> ThinAirfoilResult:
    """
    Analyze a KulfanAirfoil using thin airfoil theory.

    Bridge function that converts a KulfanAirfoil into contour coordinates
    and delegates to thin_airfoil_analysis().

    Args:
        airfoil: asb.KulfanAirfoil instance
        alpha: angle of attack in degrees
        mach: Mach number (for compressibility correction)
        AR: aspect ratio (used for induced drag calculation)

    Returns:
        ThinAirfoilResult with CL, CD, CM and surface Cp distributions
    """
    upper = airfoil.upper_coordinates()
    lower = airfoil.lower_coordinates()

    contour_x = np.concatenate([upper[:, 0], lower[:, 0]])
    contour_y = np.concatenate([upper[:, 1], lower[:, 1]])

    return thin_airfoil_analysis(contour_x, contour_y, alpha=alpha, mach=mach, AR=AR)


def compare_with_neuralfoil():
    """Compare thin airfoil theory with NeuralFoil."""
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/src')
    import aerosandbox as asb

    # NACA0012 at 5 degrees
    airfoil = asb.KulfanAirfoil("naca0012")
    aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=500e3, mach=0.03)

    def to_float(v):
        return float(np.asarray(v).flatten()[0])

    print("NeuralFoil: CL={:.4f}, CD={:.6f}, CM={:.4f}".format(
        to_float(aero['CL']), to_float(aero['CD']), to_float(aero['CM'])))

    # Thin airfoil theory
    result = thin_airfoil_analysis(
        contour_x=np.linspace(0, 1, 100),
        contour_y=np.zeros(100),  # Symmetric airfoil
        alpha=5.0,
        mach=0.03,
        AR=6.0
    )
    print("Thin Airfoil: CL={:.4f}, CD={:.6f}, CM={:.4f}".format(
        result.CL, result.CD, result.CM))


if __name__ == "__main__":
    compare_with_neuralfoil()
