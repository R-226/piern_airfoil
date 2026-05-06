"""
Test NeuralFoil optimization integration.

Following the official NeuralFoil tutorial patterns.
"""

import numpy as np


class TestNeuralFoilAnalysis:
    """Test NeuralFoil analysis."""

    def test_analyze_by_name(self):
        """Test analyzing airfoil by name."""
        from piern_airfoil.neuralfoil import NeuralFoilAnalyzer

        analyzer = NeuralFoilAnalyzer()
        result = analyzer.analyze("naca4412", alpha=5.0, Re=3e6)

        assert result.cl > 0
        assert result.cd > 0
        print(f"  NACA4412 @ 5°: CL={result.cl:.4f}, CD={result.cd:.6f}, confidence={result.analysis_confidence:.3f}")

    def test_vectorized_analysis(self):
        """Test vectorized analysis across multiple conditions."""
        from piern_airfoil.neuralfoil import NeuralFoilAnalyzer

        analyzer = NeuralFoilAnalyzer()

        alphas = [0, 5, 10]
        for alpha in alphas:
            result = analyzer.analyze("dae11", alpha=alpha, Re=1e6)
            print(f"  alpha={alpha}°: CL={result.cl:.4f}, CD={result.cd:.6f}")


class TestNeuralFoilOptimization:
    """Test NeuralFoil optimization using asb.Opti."""

    def test_simple_optimization(self):
        """Test simple drag minimization with CL constraint."""
        from piern_airfoil.neuralfoil import NeuralFoilOptimizer

        optimizer = NeuralFoilOptimizer()

        # Optimize: minimize CD subject to CL >= 0.6
        result = optimizer.optimize(
            objective="min_cd",
            constraints=[("cl", ">=", 0.6)],
            initial_guess="naca0012",
            alpha_init=5.0,
            Re=3e6,
        )

        if result.success:
            print(f"  Optimized: CL={result.cl:.4f}, CD={result.cd:.6f}, L/D={result.cl_cd:.2f}")
            print(f"  Alpha: {result.alpha:.2f}°")
        else:
            print(f"  Optimization failed: {result.error}")

    def test_max_lift_to_drag(self):
        """Test maximum L/D optimization."""
        from piern_airfoil.neuralfoil import NeuralFoilOptimizer

        optimizer = NeuralFoilOptimizer()

        result = optimizer.optimize(
            objective="max_cl_cd",
            initial_guess="naca0012",
            Re=3e6,
        )

        if result.success:
            print(f"  Max L/D: CL={result.cl:.4f}, CD={result.cd:.6f}, L/D={result.cl_cd:.2f}")
        else:
            print(f"  Optimization failed: {result.error}")


class TestMultipointOptimization:
    """Test multi-point optimization (following official tutorial)."""

    def test_multipoint_hpa(self):
        """Test Human-Powered Aircraft optimization from tutorial."""
        from piern_airfoil.neuralfoil import NeuralFoilOptimizer

        optimizer = NeuralFoilOptimizer()

        # Follow the tutorial: multipoint optimization
        CL_multipoint_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
        CL_multipoint_weights = np.array([5, 6, 7, 8, 9, 10])
        Re = 500e3 * (CL_multipoint_targets / 1.25) ** -0.5

        result = optimizer.multipoint_optimize(
            cl_targets=CL_multipoint_targets,
            Re_values=Re,
            cl_weights=CL_multipoint_weights,
            mach=0.03,
            constraints=[
                ("cm", ">=", -0.133),
            ],
            initial_guess="naca0012",
        )

        if result["success"]:
            print(f"  Multipoint optimization successful!")
            print(f"  Alpha: {result['alpha']:.2f}°")
            for i, (cl, cd, conf) in enumerate(zip(result['CL'], result['CD'], result['confidence'])):
                print(f"    Point {i+1}: CL={cl:.4f}, CD={cd:.6f}, conf={conf:.3f}")
        else:
            print(f"  Optimization failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("=" * 60)
    print("NeuralFoil Optimization Test Suite")
    print("(Following Official Tutorial Patterns)")
    print("=" * 60)

    print("\n[1] Testing NeuralFoil Analysis...")
    test1 = TestNeuralFoilAnalysis()
    test1.test_analyze_by_name()
    test1.test_vectorized_analysis()

    print("\n[2] Testing Single-Point Optimization...")
    test2 = TestNeuralFoilOptimization()
    test2.test_simple_optimization()
    test2.test_max_lift_to_drag()

    print("\n[3] Testing Multi-Point Optimization...")
    test3 = TestMultipointOptimization()
    test3.test_multipoint_hpa()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
