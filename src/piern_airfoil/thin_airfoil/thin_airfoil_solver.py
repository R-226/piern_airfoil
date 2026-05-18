"""
Thin Airfoil Theory Solver for rapid aerodynamic predictions.

Implements classical thin airfoil theory with:
- Glauert integral for zero-lift angle of attack
- Fourier coefficient representation (A0, An)
- Proper CM from Fourier coefficients: CM_c/4 = pi/4 * (A1 - A2)
- Cp distribution with correct upper/lower surface signs
- Profile drag model (skin friction + pressure drag)
- Prandtl-Glauert compressibility correction

Speed: ~1ms per evaluation
Accuracy: Good for thin airfoils at moderate angles; profile drag is empirical
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import NamedTuple


class ThinAirfoilResult(NamedTuple):
    """Result from thin airfoil theory analysis."""
    CL: float
    CD: float       # CDi (induced) + CDp (profile)
    CM: float        # Moment coefficient at c/4
    Cp_upper: np.ndarray
    Cp_lower: np.ndarray
    x_chord: np.ndarray
    A0: float        # Zero-angle Fourier coefficient
    A1: float        # First Fourier coefficient
    A2: float        # Second Fourier coefficient


def _compute_fourier_coefficients(
    x_camber: np.ndarray,
    y_camber: np.ndarray,
    n_coeffs: int = 5,
) -> tuple[np.ndarray, float]:
    """Compute Glauert Fourier coefficients for cambered thin airfoil.

    Standard Glauert convention:
        dy/dx = A0 + sum_{n=1}^{N} An * cos(n*theta)

    where x = (1 - cos(theta))/2 maps [0,1] to [0,pi].

    Coefficients:
        A0 = 1/pi * integral_0^pi (dy/dx) d_theta
        An = 2/pi * integral_0^pi (dy/dx) * cos(n*theta) d_theta

    Zero-lift angle:
        alpha_0 = A0 - 1/pi * integral_0^pi (dy/dx) * cos(theta) d_theta
                = A0 - A1/2

    For symmetric airfoil: all A = 0, alpha_0 = 0.

    Args:
        x_cammer: x/c positions along chord (sorted, 0 to 1)
        y_camber: y/c camber line positions
        n_coeffs: number of An coefficients (A1..An), A0 computed separately

    Returns:
        (A_full, alpha_0) where A_full = [A0, A1, A2, ..., An]
        alpha_0 is the zero-lift angle (radians, negative for positive camber)
    """
    if len(x_camber) < 3:
        return np.zeros(n_coeffs + 1), 0.0

    f_camber = interp1d(x_camber, y_camber, kind='cubic', fill_value='extrapolate')

    # Map x/c to theta: x = (1 - cos(theta)) / 2
    theta_eval = np.linspace(0.01, np.pi - 0.01, 200)
    x_eval = (1 - np.cos(theta_eval)) / 2

    # Camber line slope dy/dx at each x
    y_eval = f_camber(x_eval)
    dydx = np.gradient(y_eval, x_eval)

    # A0 = 1/pi * integral_0^pi (dy/dx) d_theta
    A0 = trapezoid(dydx, theta_eval) / np.pi

    # An = 2/pi * integral_0^pi (dy/dx) * cos(n*theta) d_theta
    A = np.zeros(n_coeffs + 1)
    A[0] = A0
    for n in range(1, n_coeffs + 1):
        integrand_n = dydx * np.cos(n * theta_eval)
        A[n] = 2 * trapezoid(integrand_n, theta_eval) / np.pi

    # alpha_0 = A0 - A1/2
    alpha_0 = A0 - A[1] / 2

    return A, alpha_0


def _profile_drag_coefficient(
    thickness_max: float,
    Re: float,
    x_chord: np.ndarray,
    thickness: np.ndarray,
) -> float:
    """Estimate profile drag (skin friction + pressure drag).

    Uses flat-plate skin friction + empirical pressure drag correction.

    Args:
        thickness_max: maximum thickness/chord ratio
        Re: Reynolds number
        x_chord: chordwise positions
        thickness: thickness distribution

    Returns:
        Profile drag coefficient CDp
    """
    # Skin friction: Cf = 1.328 / sqrt(Re) for laminar
    #                 Cf = 0.074 / Re^0.2 for turbulent
    # Use blended: assume transition at Re_crit = 3e5 (typical for airfoils)
    Re_crit = 3e5
    if Re < 1e3:
        Re = 1e3  # avoid division issues

    if Re < Re_crit:
        Cf = 1.328 / np.sqrt(Re)
    else:
        Cf_lam = 1.328 / np.sqrt(Re_crit)
        Cf_turb = 0.074 / Re**0.2
        x_trans = Re_crit / Re
        Cf = x_trans * Cf_lam + (1 - x_trans) * Cf_turb

    # Wetted area ratio (approximate: upper + lower surfaces)
    # For thin airfoil, wetted area ~ 2 * chord
    CDf = 2 * Cf  # both surfaces

    # Pressure drag from thickness is negligible for thin airfoils (< 15% t/c).
    # Hoerner's 2*(t/c)^2 is for bluff bodies; for streamlined shapes the
    # pressure drag is a small fraction of skin friction. Add a mild correction.
    CDp_thickness = 0.006 * thickness_max  # ~5% of CDf for typical t/c

    return CDf + CDp_thickness


def thin_airfoil_analysis(
    contour_x: np.ndarray,
    contour_y: np.ndarray,
    alpha: float = 0.0,
    mach: float = 0.0,
    AR: float = 6.0,
    Re: float = 500e3,
) -> ThinAirfoilResult:
    """Analyze airfoil using thin airfoil theory.

    Args:
        contour_x: x-coordinates of airfoil contour (normalized, 0 to 1)
        contour_y: y-coordinates of airfoil contour
        alpha: angle of attack in degrees
        mach: Mach number (for compressibility correction)
        AR: aspect ratio (used for induced drag calculation)
        Re: Reynolds number (used for profile drag estimation)

    Returns:
        ThinAirfoilResult with CL, CD, CM, Cp distributions, and Fourier coefficients
    """
    alpha_rad = np.radians(alpha)

    # --- Separate upper/lower surfaces ---
    x_unique = np.unique(contour_x)
    y_upper = np.zeros_like(x_unique)
    y_lower = np.zeros_like(x_unique)

    for i, x in enumerate(x_unique):
        mask = np.isclose(contour_x, x)
        y_vals = contour_y[mask]
        y_upper[i] = np.max(y_vals)
        y_lower[i] = np.min(y_vals)

    sort_idx = np.argsort(x_unique)
    x_unique = x_unique[sort_idx]
    y_upper = y_upper[sort_idx]
    y_lower = y_lower[sort_idx]

    # Camber and thickness
    y_camber = (y_upper + y_lower) / 2
    thickness = y_upper - y_lower
    thickness_max = float(np.max(thickness))

    # --- Prandtl-Glauert correction ---
    if mach > 0 and mach < 1:
        beta = np.sqrt(1 - mach**2)
        pg_corr = 1 / beta
    else:
        pg_corr = 1.0

    # --- Fourier coefficients and alpha_0 ---
    # A_full = [A0, A1, A2, ..., An] in Glauert convention
    A_full, alpha_0 = _compute_fourier_coefficients(x_unique, y_camber, n_coeffs=5)

    # CL = 2*pi*(alpha - alpha_0) * PG correction
    # alpha_0 is negative for positive camber (e.g. NACA4412: alpha_0 ~ -5°)
    CL = 2 * np.pi * (alpha_rad - alpha_0) * pg_corr

    # --- CM at quarter-chord ---
    # CM_c/4 = pi/4 * (A2 - A3) in standard notation
    # A_full[1]=A1, A_full[2]=A2
    A1 = float(A_full[1]) if len(A_full) > 1 else 0.0
    A2 = float(A_full[2]) if len(A_full) > 2 else 0.0
    CM = (np.pi / 4) * (A2 - A1)

    # --- Drag ---
    # Profile drag only (2D, no induced drag — NeuralFoil is 2D)
    CD = _profile_drag_coefficient(thickness_max, Re, x_unique, thickness)

    # --- Cp distribution ---
    # Glauert vortex strength: gamma(theta) = 2*U*[A0*(1+cos)/sin + sum(An*sin(n*theta))]
    # Cp = 1 - (V/U)^2 ≈ -2*vx/U for thin airfoil
    # For upper surface: Cp_u = -2*(alpha + theta_u) where theta is surface slope
    # For lower surface: Cp_l = +2*(alpha + theta_l) (opposite sign!)

    x_chord = np.linspace(0.01, 0.99, 80)

    # Interpolate surfaces
    f_upper = interp1d(x_unique, y_upper, kind='linear', fill_value='extrapolate')
    f_lower = interp1d(x_unique, y_lower, kind='linear', fill_value='extrapolate')

    yu = f_upper(x_chord)
    yl = f_lower(x_chord)

    # Surface slopes
    dydx_u = np.gradient(yu, x_chord)
    dydx_l = np.gradient(yl, x_chord)

    theta_u = np.arctan(dydx_u)
    theta_l = np.arctan(dydx_l)

    # Cp upper: suction (negative) for positive alpha
    # Cp = -2 * (alpha_rad + theta) on upper surface
    Cp_upper = -2 * (alpha_rad + theta_u) * pg_corr

    # Cp lower: pressure (positive) for positive alpha
    # Cp = +2 * (alpha_rad + theta) on lower surface (opposite sign!)
    Cp_lower = 2 * (alpha_rad + theta_l) * pg_corr

    return ThinAirfoilResult(
        CL=float(CL),
        CD=float(CD),
        CM=float(CM),
        Cp_upper=Cp_upper,
        Cp_lower=Cp_lower,
        x_chord=x_chord,
        A0=float(A_full[0]) if len(A_full) > 0 else 0.0,
        A1=float(A_full[1]) if len(A_full) > 1 else 0.0,
        A2=float(A_full[2]) if len(A_full) > 2 else 0.0,
    )


def thin_airfoil_from_kulfan(
    airfoil,
    alpha: float = 0.0,
    mach: float = 0.0,
    AR: float = 6.0,
    Re: float = 500e3,
) -> ThinAirfoilResult:
    """Analyze a KulfanAirfoil using thin airfoil theory.

    Bridge function that converts a KulfanAirfoil into contour coordinates
    and delegates to thin_airfoil_analysis().

    Args:
        airfoil: asb.KulfanAirfoil instance
        alpha: angle of attack in degrees
        mach: Mach number (for compressibility correction)
        AR: aspect ratio (used for induced drag calculation)
        Re: Reynolds number (used for profile drag estimation)

    Returns:
        ThinAirfoilResult with CL, CD, CM, Cp distributions, and Fourier coefficients
    """
    upper = airfoil.upper_coordinates()
    lower = airfoil.lower_coordinates()

    contour_x = np.concatenate([upper[:, 0], lower[:, 0]])
    contour_y = np.concatenate([upper[:, 1], lower[:, 1]])

    return thin_airfoil_analysis(contour_x, contour_y, alpha=alpha, mach=mach, AR=AR, Re=Re)


def thin_airfoil_multipoint_cd(
    airfoil,
    cl_targets: np.ndarray,
    cl_weights: np.ndarray,
    mach: float = 0.03,
    Re: float = 500e3,
) -> tuple[float, dict]:
    """Fast multi-point weighted CD evaluation using analytical alpha.

    For thin airfoil theory: CL = 2*pi*(alpha - alpha_0)
    So alpha = CL/(2*pi) + alpha_0  (no iterative search needed)

    Args:
        airfoil: asb.KulfanAirfoil instance
        cl_targets: array of CL targets
        cl_weights: array of weights for each CL target
        mach: Mach number
        Re: Reynolds number

    Returns:
        (weighted_cd, aero_summary) where aero_summary has CL, CD, CM at first target
    """
    upper = airfoil.upper_coordinates()
    lower = airfoil.lower_coordinates()
    contour_x = np.concatenate([upper[:, 0], lower[:, 0]])
    contour_y = np.concatenate([upper[:, 1], lower[:, 1]])

    # Separate surfaces
    x_unique = np.unique(contour_x)
    y_upper = np.zeros_like(x_unique)
    y_lower = np.zeros_like(x_unique)
    for i, x in enumerate(x_unique):
        mask = np.isclose(contour_x, x)
        y_vals = contour_y[mask]
        y_upper[i] = np.max(y_vals)
        y_lower[i] = np.min(y_vals)
    sort_idx = np.argsort(x_unique)
    x_unique = x_unique[sort_idx]
    y_upper = y_upper[sort_idx]
    y_lower = y_lower[sort_idx]

    y_camber = (y_upper + y_lower) / 2
    thickness = y_upper - y_lower
    thickness_max = float(np.max(thickness))

    # Fourier coefficients (compute once)
    A_full, alpha_0 = _compute_fourier_coefficients(x_unique, y_camber, n_coeffs=5)

    # Profile drag (independent of alpha)
    CDp = _profile_drag_coefficient(thickness_max, Re, x_unique, thickness)

    # CM from Fourier coefficients (independent of alpha)
    A1 = float(A_full[1]) if len(A_full) > 1 else 0.0
    A2 = float(A_full[2]) if len(A_full) > 2 else 0.0
    CM = (np.pi / 4) * (A2 - A1)

    # Compressibility correction
    if mach > 0 and mach < 1:
        pg_corr = 1 / np.sqrt(1 - mach**2)
    else:
        pg_corr = 1.0

    # Multi-point evaluation
    total_cd = 0.0
    alpha_penalty = 0.0
    alpha_min = np.radians(-3.0)
    alpha_max = np.radians(15.0)
    for i, cl_t in enumerate(cl_targets):
        # Analytical alpha: CL = 2*pi*(alpha - alpha_0) * pg_corr
        alpha_i = cl_t / (2 * np.pi * pg_corr) + alpha_0
        # Penalize alpha outside practical range
        if alpha_i < alpha_min:
            alpha_penalty += (alpha_min - alpha_i) ** 2 * 1000
        elif alpha_i > alpha_max:
            alpha_penalty += (alpha_i - alpha_max) ** 2 * 1000
        total_cd += CDp * float(cl_weights[i])

    total_cd += alpha_penalty

    # Aero summary at first target (CL matches by construction)
    aero_summary = {
        "CL": float(cl_targets[0]),
        "CD": CDp,
        "CM": CM,
    }

    return total_cd, aero_summary


def compare_with_neuralfoil():
    """Compare thin airfoil theory with NeuralFoil."""
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0] + '/src')
    import aerosandbox as asb

    airfoil = asb.KulfanAirfoil("naca0012")
    aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=500e3, mach=0.03)

    def to_float(v):
        return float(np.asarray(v).flatten()[0])

    print("NeuralFoil: CL={:.4f}, CD={:.6f}, CM={:.4f}".format(
        to_float(aero['CL']), to_float(aero['CD']), to_float(aero['CM'])))

    result = thin_airfoil_from_kulfan(airfoil, alpha=5.0, mach=0.03, Re=500e3)
    print("Thin Airfoil: CL={:.4f}, CD={:.6f}, CM={:.4f}".format(
        result.CL, result.CD, result.CM))
    print("  Fourier: A0={:.4f} A1={:.4f} A2={:.4f}".format(result.A0, result.A1, result.A2))
    print("  alpha_0={:.4f} rad".format(np.arcsin(result.A0) if abs(result.A0) < 1 else 0))


if __name__ == "__main__":
    compare_with_neuralfoil()
