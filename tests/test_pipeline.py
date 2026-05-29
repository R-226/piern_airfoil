"""Integration test for the PiERN pipeline."""

import time
import numpy as np
import aerosandbox as asb
from pathlib import Path

from piern.pipeline import PiernPipeline


def test_pipeline_with_prompt():
    """Test the pipeline: prompt extraction + optimization."""
    prompt = (
        "在Ma=0.03的飞行条件下，要求CL=[0.8,1.0,1.2,1.4,1.5,1.6]，"
        "权重[5,6,7,8,9,10]。约束：CM≥-0.133，后缘角≥6.03°，"
        "前缘角=180°，前缘厚度>0.128，后缘厚度>0.014。请优化翼型"
    )

    pipeline = PiernPipeline()

    # Step 1: Extract parameters
    params = pipeline.extract_params(prompt)
    print(f"Extracted: Mach={params.Mach}, CL={params.CL_targets}, "
          f"W={params.CL_weights}")

    assert len(params.CL_targets) == 6
    assert len(params.CL_weights) == 6
    assert params.Mach > 0

    # Step 2: Start from NACA0012 (skip image extraction for integration test)
    airfoil = asb.KulfanAirfoil("naca0012")
    initial_cd = pipeline._quick_eval(airfoil, params)
    print(f"Initial CD: {initial_cd:.6f}")

    # Step 3: Optimize
    optimized, opt_time, history = pipeline.optimize(airfoil, params)
    final_cd = pipeline._quick_eval(optimized, params)
    print(f"Final CD: {final_cd:.6f}, Time: {opt_time:.2f}s")
    print(f"Improvement: {(initial_cd - final_cd) / initial_cd * 100:+.2f}%")

    print("\nStage history:")
    for h in history:
        print(f"  Stage {h['stage']}: {h['n_active_weights']}w → "
              f"CD={h['cd']:.6f}  [{h['message']}]")

    assert final_cd < initial_cd * 1.05  # at most 5% worse
    assert len(history) > 0

    # Step 4: Physical check
    t33 = float(np.asarray(optimized.local_thickness(x_over_c=0.33)).flatten()[0])
    t90 = float(np.asarray(optimized.local_thickness(x_over_c=0.90)).flatten()[0])
    te = float(np.asarray(optimized.TE_angle()).flatten()[0])
    print(f"\nPhysical: t33={t33:.4f} (>=0.128), t90={t90:.4f} (>=0.014), TE={te:.2f} (>=6.03)")
    assert t33 >= 0.128 - 0.001
    assert t90 >= 0.014 - 0.0001
    assert te >= 6.03 - 0.01

    print("\nAll checks passed!")


def test_router_decisions():
    """Test opt_router decision logic."""
    from piern.router.opt_router import OptRouter, OptState, OptAction

    router = OptRouter(improvement_threshold=0.01)

    # Test 1: First stage (no previous CD)
    state = OptState(stage=1, n_active_weights=4, cd=0.5, prev_cd=None)
    action, new_n, reason = router.decide(state)
    assert action == OptAction.KEEP
    print(f"  First stage: {reason}")

    # Test 2: Significant improvement
    state = OptState(stage=2, n_active_weights=4, cd=0.45, prev_cd=0.5)
    action, new_n, reason = router.decide(state)
    assert action == OptAction.KEEP
    assert new_n == 4
    print(f"  Good improvement: {reason}")

    # Test 3: Insufficient improvement
    state = OptState(stage=3, n_active_weights=4, cd=0.449, prev_cd=0.45)
    action, new_n, reason = router.decide(state)
    assert action == OptAction.EXPAND
    assert new_n == 6
    print(f"  Poor improvement: {reason}")

    # Test 4: At max weights
    state = OptState(stage=4, n_active_weights=8, cd=0.44, prev_cd=0.45)
    action, new_n, reason = router.decide(state)
    assert action == OptAction.KEEP
    assert new_n == 8
    print(f"  At max weights: {reason}")

    print("All router tests passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Router Unit Tests")
    print("=" * 70)
    test_router_decisions()

    print("\n" + "=" * 70)
    print("Pipeline Integration Test")
    print("=" * 70)
    test_pipeline_with_prompt()
