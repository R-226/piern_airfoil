"""XFoil baseline evaluator for airfoil optimization comparison.

Provides XFOIL-based CD evaluation as a high-fidelity baseline comparison
for the NeuralFoil-based evaluator in eval.py. Uses the system XFoil binary
via subprocess calls.
"""

from __future__ import annotations

import logging
import re
import tempfile
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

XFOIL_BIN = "/usr/bin/xfoil"
XFOIL_TIMEOUT = 30  # seconds per CL evaluation


def airfoil_to_dat(
    name: str,
    output_path: str | Path,
    coordinates: np.ndarray | None = None,
) -> Path:
    """Write airfoil coordinates to a Selig-format .dat file.

    Args:
        name: Airfoil name (used for the first line of the .dat file).
        output_path: Destination path for the .dat file.
        coordinates: Optional (N, 2) array of [x, y] coordinates.
            If None, loads coordinates from aerosandbox for the given name.

    Returns:
        Path to the written .dat file.
    """
    output_path = Path(output_path)
    if coordinates is None:
        import aerosandbox as asb

        af = asb.Airfoil(name)
        coordinates = af.coordinates

    lines = [f"{name}\n"]
    for x, y in coordinates:
        lines.append(f" {x: .7f} {y: .7f}\n")

    output_path.write_text("".join(lines))
    return output_path


def _parse_xfoil_output(text: str) -> dict[str, float] | None:
    """Parse XFoil stdout to extract aerodynamic quantities from iteration output.

    Extracts CL and CD from the last converged iteration block, which has lines like:
        a =  6.947      CL =  0.8000
        Cm = -0.0050     CD =  0.01303   =>   CDf =  0.00673    CDp =  0.00630
    """
    cl_matches = re.findall(r"CL\s*=\s*([0-9.eE+-]+)", text)
    cd_matches = re.findall(r"CD\s*=\s*([0-9.eE+-]+)", text)
    alpha_matches = re.findall(r"\ba\s*=\s*([0-9.eE+-]+)", text)
    cm_matches = re.findall(r"Cm\s*=\s*([0-9.eE+-]+)", text)

    if not cl_matches or not cd_matches:
        return None

    try:
        return {
            "CL": float(cl_matches[-1]),
            "CD": float(cd_matches[-1]),
            "alpha": float(alpha_matches[-1]) if alpha_matches else 0.0,
            "CM": float(cm_matches[-1]) if cm_matches else 0.0,
        }
    except (ValueError, IndexError):
        return None


def _run_xfoil_single(
    dat_path: str | Path,
    cl_target: float,
    Re: float,
    mach: float,
) -> float | None:
    """Run XFoil for a single (CL, Re, mach) operating point.

    Args:
        dat_path: Path to the .dat airfoil file.
        cl_target: Target lift coefficient.
        Re: Reynolds number.
        mach: Mach number.

    Returns:
        CD at the target CL, or None if convergence failed.
    """
    commands = (
        f"LOAD {dat_path}\nPANE\nOPER\nVISC {Re:.0f}\nM {mach}\nCL {cl_target}\nQUIT\n"
    )

    try:
        result = subprocess.run(
            [XFOIL_BIN],
            input=commands,
            capture_output=True,
            text=True,
            timeout=XFOIL_TIMEOUT,
        )
    except FileNotFoundError:
        logger.error("XFoil binary not found at %s", XFOIL_BIN)
        raise
    except subprocess.TimeoutExpired:
        logger.warning(
            "XFoil timed out after %ds for CL=%.2f Re=%.0f",
            XFOIL_TIMEOUT,
            cl_target,
            Re,
        )
        return None

    combined = result.stdout + result.stderr
    parsed = _parse_xfoil_output(combined)

    if parsed is None:
        logger.warning(
            "XFoil failed to converge for CL=%.2f Re=%.0f M=%.3f",
            cl_target,
            Re,
            mach,
        )
        return None

    return parsed["CD"]


def xfoil_cd(
    airfoil_name: str,
    CL_targets: np.ndarray,
    Re: np.ndarray,
    CL_weights: np.ndarray,
    mach: float = 0.03,
) -> float:
    """Evaluate weighted CD using XFoil.

    Same formula as evaluate_weighted_cd in eval.py: mean(CD * weights).

    Args:
        airfoil_name: Name of the airfoil (resolved via aerosandbox).
        CL_targets: Target CL values for each design point.
        Re: Reynolds numbers for each design point.
        CL_weights: Optimization weights for each design point.
        mach: Mach number.

    Returns:
        Weighted mean CD (scalar).
    """
    with tempfile.TemporaryDirectory(prefix="xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / f"{airfoil_name}.dat"
        airfoil_to_dat(airfoil_name, dat_path)

        cd_values = []
        for cl_t, re_i in zip(CL_targets, Re):
            cd = _run_xfoil_single(dat_path, cl_t, float(re_i), mach)
            if cd is not None:
                cd_values.append(cd)
            else:
                cd_values.append(float("inf"))

    return float(np.mean(np.array(cd_values) * CL_weights))


def xfoil_cd_from_coordinates(
    coordinates: np.ndarray,
    CL_targets: np.ndarray,
    Re: np.ndarray,
    CL_weights: np.ndarray,
    mach: float = 0.03,
    name: str = "optimized",
) -> float:
    """Evaluate weighted CD using XFoil from explicit coordinates.

    Args:
        coordinates: (N, 2) array of airfoil coordinates.
        CL_targets: Target CL values for each design point.
        Re: Reynolds numbers for each design point.
        CL_weights: Optimization weights for each design point.
        mach: Mach number.
        name: Airfoil name for the .dat file header.

    Returns:
        Weighted mean CD (scalar).
    """
    with tempfile.TemporaryDirectory(prefix="xfoil_") as tmpdir:
        dat_path = Path(tmpdir) / f"{name}.dat"
        airfoil_to_dat(name, dat_path, coordinates=coordinates)

        cd_values = []
        for cl_t, re_i in zip(CL_targets, Re):
            cd = _run_xfoil_single(dat_path, cl_t, float(re_i), mach)
            if cd is not None:
                cd_values.append(cd)
            else:
                cd_values.append(float("inf"))

    return float(np.mean(np.array(cd_values) * CL_weights))


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a quick test comparing XFoil and NeuralFoil on NACA 0012."""
    CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
    MACH = 0.03

    print("XFoil baseline test — NACA 0012")
    print(f"CL targets: {CL_TARGETS}")
    print(f"CL weights: {CL_WEIGHTS}")
    print(f"Re: {RE}")
    print(f"Mach: {MACH}")
    print()

    # XFoil evaluation
    cd_xfoil = xfoil_cd("naca0012", CL_TARGETS, RE, CL_WEIGHTS, MACH)
    print(f"XFoil weighted CD:   {cd_xfoil:.6f}")

    # NeuralFoil evaluation (via eval.py)
    from piern_airfoil.eval import evaluate_weighted_cd
    import aerosandbox as asb

    af = asb.KulfanAirfoil("naca0012")
    cd_nf = evaluate_weighted_cd(af, CL_TARGETS, RE, CL_WEIGHTS, MACH)
    print(f"NeuralFoil weighted CD: {cd_nf:.6f}")

    if cd_nf > 0:
        diff_pct = (cd_xfoil - cd_nf) / cd_nf * 100
        print(f"Difference: {diff_pct:+.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
