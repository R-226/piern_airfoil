"""
Strategy A1: Learn optimal improvement_threshold from data.

Method: Grid search over threshold values, evaluated on multiple
optimization episodes with different initial airfoils.

Usage:
    uv run python -m piern.router.train_threshold
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import aerosandbox as asb


# ── Problem definition ──────────────────────────────────────────────────

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5
MACH = 0.03

# Initial airfoils for training episodes
TRAINING_AIRFOILS = [
    "naca0012",
    "naca2412",
    "naca4412",
    "naca0015",
    "naca2415",
    "naca6412",
]


# ── Episode data ────────────────────────────────────────────────────────


@dataclass
class DecisionPoint:
    """A single routing decision during optimization."""
    stage: int
    n_active_weights: int
    improvement: float | None  # None for first stage
    action: str  # "KEEP" or "EXPAND"
    cd_before: float
    cd_after: float


@dataclass
class Episode:
    """Complete optimization episode."""
    airfoil_name: str
    initial_cd: float
    final_cd: float
    total_time: float
    decision_points: list[DecisionPoint]
    cd_trajectory: list[tuple[int, float]]  # (n_active_weights, cd)


# ── Core functions ──────────────────────────────────────────────────────


def evaluate_cd(airfoil) -> float:
    """Evaluate weighted CD for an airfoil."""
    from scipy.optimize import brentq

    total_cd = 0.0
    for cl_t, re_i, w_i in zip(CL_TARGETS, RE, CL_WEIGHTS):
        def residual(a, _af=airfoil, _re=re_i, _cl=cl_t):
            aero = _af.get_aero_from_neuralfoil(alpha=a, Re=float(_re), mach=MACH)
            return float(np.asarray(aero["CL"]).flatten()[0]) - _cl
        try:
            alpha_i = brentq(residual, -5, 18, xtol=0.01, maxiter=30)
        except (ValueError, RuntimeError):
            alpha_i = 5.0
        aero = airfoil.get_aero_from_neuralfoil(alpha=alpha_i, Re=float(re_i), mach=MACH)
        total_cd += float(np.asarray(aero["CD"]).flatten()[0]) * w_i
    return total_cd


def run_stage(airfoil, n_active: int, initial_weights=None):
    """Run a single CST optimization stage."""
    import casadi
    import aerosandbox.numpy as asbnp

    opti = asb.Opti()

    initial_upper = airfoil.upper_weights
    initial_lower = airfoil.lower_weights

    upper_vars, lower_vars = [], []
    upper_fixed, lower_fixed = [], []

    for i in range(8):
        if i < n_active:
            init_u = initial_weights[0][i] if initial_weights else initial_upper[i]
            init_l = initial_weights[1][i] if initial_weights else initial_lower[i]
            upper_vars.append(opti.variable(init_guess=float(init_u), lower_bound=-0.25, upper_bound=0.5))
            lower_vars.append(opti.variable(init_guess=float(init_l), lower_bound=-0.5, upper_bound=0.25))
        else:
            upper_fixed.append(float(initial_upper[i]))
            lower_fixed.append(float(initial_lower[i]))

    upper_weights = casadi.vertcat(*upper_vars, *upper_fixed) if upper_fixed else casadi.vertcat(*upper_vars)
    lower_weights = casadi.vertcat(*lower_vars, *lower_fixed) if lower_fixed else casadi.vertcat(*lower_vars)

    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        leading_edge_weight=opti.variable(
            init_guess=airfoil.leading_edge_weight, lower_bound=-1, upper_bound=1
        ),
        TE_thickness=0,
    )

    alpha = opti.variable(
        init_guess=np.degrees(CL_TARGETS / (2 * np.pi)),
        lower_bound=-5, upper_bound=18,
    )

    aero = optimized_airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=RE, mach=MACH)

    opti.subject_to([
        aero["analysis_confidence"] > 0.90,
        aero["CL"] == CL_TARGETS,
        asbnp.diff(alpha) > 0,
        aero["CM"] >= -0.133,
        optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
        optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
        optimized_airfoil.TE_angle() >= 6.03,
        optimized_airfoil.lower_weights[0] < -0.05,
        optimized_airfoil.upper_weights[0] > 0.05,
        optimized_airfoil.local_thickness() > 0,
        optimized_airfoil.LE_radius() > 0,
    ])

    get_wiggliness = lambda af: sum(
        asbnp.sum(asbnp.diff(asbnp.diff(array)) ** 2)
        for array in [af.lower_weights, af.upper_weights]
    )
    opti.subject_to(get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(airfoil))

    opti.minimize(asbnp.mean(aero["CD"] * CL_WEIGHTS))

    sol = opti.solve(
        behavior_on_failure="return_last",
        options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
    )
    result_airfoil = sol(optimized_airfoil)

    result_upper = np.array([float(sol(upper_vars[i])) for i in range(n_active)] +
                           [upper_fixed[i] for i in range(8 - n_active)])
    result_lower = np.array([float(sol(lower_vars[i])) for i in range(n_active)] +
                           [lower_fixed[i] for i in range(8 - n_active)])

    return result_airfoil, (result_upper, result_lower)


def run_episode(
    airfoil_name: str,
    threshold: float,
    start_weights: int = 4,
    max_stages: int = 6,
) -> Episode:
    """
    Run a single optimization episode with a given threshold.

    Records the full CD trajectory and routing decisions.
    """
    airfoil = asb.KulfanAirfoil(airfoil_name)
    init_cd = evaluate_cd(airfoil)

    current_airfoil = airfoil
    current_weights = (airfoil.upper_weights, airfoil.lower_weights)
    n_active = start_weights
    prev_cd = init_cd

    decision_points = []
    cd_trajectory = [(n_active, init_cd)]

    t0 = time.perf_counter()

    for stage_idx in range(max_stages):
        # Force expansion to 8 weights on the final stage,
        # so every episode reaches full dimensionality regardless of threshold.
        if stage_idx == max_stages - 1:
            n_active = 8

        # Run optimization
        result_airfoil, result_weights = run_stage(current_airfoil, n_active, current_weights)
        cd = evaluate_cd(result_airfoil)

        # Compute improvement
        improvement = (prev_cd - cd) / prev_cd if prev_cd > 0 else None

        # Record decision
        action = "KEEP" if (improvement is not None and improvement > threshold) else "EXPAND"
        decision_points.append(DecisionPoint(
            stage=stage_idx + 1,
            n_active_weights=n_active,
            improvement=improvement,
            action=action,
            cd_before=prev_cd,
            cd_after=cd,
        ))

        cd_trajectory.append((n_active, cd))

        # Apply routing decision
        if action == "EXPAND":
            new_n = min(n_active + 2, 8)
            if new_n > n_active:
                n_active = new_n

        # Update state
        current_airfoil = result_airfoil
        current_weights = result_weights
        prev_cd = cd

        # Stop if max weights reached and we've done at least one more stage
        if n_active >= 8 and stage_idx > 0:
            break

    elapsed = time.perf_counter() - t0

    return Episode(
        airfoil_name=airfoil_name,
        initial_cd=init_cd,
        final_cd=prev_cd,
        total_time=elapsed,
        decision_points=decision_points,
        cd_trajectory=cd_trajectory,
    )


# ── Grid search ─────────────────────────────────────────────────────────


@dataclass
class ThresholdResult:
    """Result of evaluating a single threshold value."""
    threshold: float
    avg_final_cd: float
    avg_time: float
    avg_stages: float
    episodes: list[Episode]


def grid_search(
    thresholds: list[float] | None = None,
    n_airfoils: int = 6,
    start_weights: int = 4,
    verbose: bool = True,
) -> list[ThresholdResult]:
    """
    Grid search over threshold values.

    For each threshold, runs optimization on multiple initial airfoils
    and computes the average final CD.
    """
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]

    airfoil_names = TRAINING_AIRFOILS[:n_airfoils]
    results = []

    for threshold in thresholds:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing threshold = {threshold}")
            print(f"{'='*60}")

        episodes = []
        for airfoil_name in airfoil_names:
            if verbose:
                print(f"  Episode: {airfoil_name}...", end=" ", flush=True)
            ep = run_episode(airfoil_name, threshold, start_weights=start_weights)
            episodes.append(ep)
            if verbose:
                print(f"CD={ep.final_cd:.6f}, time={ep.total_time:.1f}s, stages={len(ep.decision_points)}")

        avg_cd = np.mean([ep.final_cd for ep in episodes])
        avg_time = np.mean([ep.total_time for ep in episodes])
        avg_stages = np.mean([len(ep.decision_points) for ep in episodes])

        results.append(ThresholdResult(
            threshold=threshold,
            avg_final_cd=avg_cd,
            avg_time=avg_time,
            avg_stages=avg_stages,
            episodes=episodes,
        ))

        if verbose:
            print(f"  → Average CD: {avg_cd:.6f}, Average time: {avg_time:.1f}s")

    return results


def find_optimal_threshold(results: list[ThresholdResult]) -> tuple[float, ThresholdResult]:
    """Find the threshold with the lowest average final CD."""
    best = min(results, key=lambda r: r.avg_final_cd)
    return best.threshold, best


# ── Save / Load ─────────────────────────────────────────────────────────

SAVE_DIR = Path(__file__).parent / "trained"


def save_threshold(threshold: float, results: list[ThresholdResult], path: Path | None = None):
    """Save the learned threshold and training metadata."""
    if path is None:
        path = SAVE_DIR / "optimal_threshold.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "optimal_threshold": threshold,
        "method": "grid_search",
        "grid": [
            {
                "threshold": r.threshold,
                "avg_final_cd": r.avg_final_cd,
                "avg_time": r.avg_time,
                "avg_stages": r.avg_stages,
            }
            for r in results
        ],
        "problem": {
            "CL_targets": CL_TARGETS.tolist(),
            "CL_weights": CL_WEIGHTS.tolist(),
            "mach": MACH,
        },
        "training_airfoils": TRAINING_AIRFOILS,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved optimal threshold ({threshold}) to {path}")


def load_threshold(path: Path | None = None) -> float:
    """Load the learned threshold."""
    if path is None:
        path = SAVE_DIR / "optimal_threshold.json"

    with open(path) as f:
        data = json.load(f)

    return data["optimal_threshold"]


# ── Main ────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Strategy A1: Threshold Grid Search")
    print("=" * 60)

    t0 = time.perf_counter()

    results = grid_search(
        thresholds=[0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        n_airfoils=4,  # Use 4 airfoils for speed
        start_weights=4,
        verbose=True,
    )

    elapsed = time.perf_counter() - t0

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Threshold':>10} {'Avg CD':>12} {'Avg Time':>10} {'Avg Stages':>12}")
    print("-" * 46)
    for r in results:
        print(f"{r.threshold:>10.4f} {r.avg_final_cd:>12.6f} {r.avg_time:>10.1f} {r.avg_stages:>12.1f}")

    opt_threshold, opt_result = find_optimal_threshold(results)
    print(f"\nOptimal threshold: {opt_threshold}")
    print(f"Average CD at optimal: {opt_result.avg_final_cd:.6f}")
    print(f"Total time: {elapsed:.1f}s")

    save_threshold(opt_threshold, results)

    return opt_threshold, results


if __name__ == "__main__":
    main()
