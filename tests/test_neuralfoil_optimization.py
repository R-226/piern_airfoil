"""
Test NeuralFoil analysis and optimization integration.
"""

import numpy as np
import aerosandbox as asb


class TestNeuralFoilAnalysis:
    """Test NeuralFoil analysis via aerosandbox."""

    def test_analyze_by_name(self):
        """Test analyzing airfoil by name."""
        airfoil = asb.KulfanAirfoil("naca4412")
        aero = airfoil.get_aero_from_neuralfoil(alpha=5.0, Re=3e6)

        CL = float(np.asarray(aero["CL"]).flatten()[0])
        CD = float(np.asarray(aero["CD"]).flatten()[0])
        confidence = float(np.asarray(aero["analysis_confidence"]).flatten()[0])

        assert CL > 0
        assert CD > 0
        assert confidence > 0.9
        print(f"  NACA4412 @ 5deg: CL={CL:.4f}, CD={CD:.6f}, confidence={confidence:.3f}")

    def test_vectorized_analysis(self):
        """Test analysis across multiple angles of attack."""
        airfoil = asb.KulfanAirfoil("dae11")

        for alpha in [0, 5, 10]:
            aero = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=1e6)
            CL = float(np.asarray(aero["CL"]).flatten()[0])
            CD = float(np.asarray(aero["CD"]).flatten()[0])
            print(f"  alpha={alpha}deg: CL={CL:.4f}, CD={CD:.6f}")


class TestNeuralFoilOptimization:
    """Test NeuralFoil optimization using NeuralOptimizer."""

    def test_single_point_optimization(self):
        """Test simple drag minimization with CL constraint."""
        from piern_airfoil.neuralfoil import NeuralOptimizer

        airfoil = asb.KulfanAirfoil("naca0012")
        optimizer = NeuralOptimizer(
            airfoil=airfoil,
            CL_targets=np.array([0.6]),
            CL_weights=np.array([1.0]),
            RE=np.array([3e6]),
            mach=0.0,
        )

        optimizer.update()

        CL = float(np.asarray(optimizer.aero["CL"]).flatten()[0])
        CD = float(np.asarray(optimizer.aero["CD"]).flatten()[0])
        print(f"  Optimized: CL={CL:.4f}, CD={CD:.6f}, L/D={CL/CD:.2f}")

        assert CL > 0
        assert CD > 0

    def test_constraints_active(self):
        """Verify thickness constraint is satisfied after optimization."""
        from piern_airfoil.neuralfoil import NeuralOptimizer

        airfoil = asb.KulfanAirfoil("naca0012")
        optimizer = NeuralOptimizer(
            airfoil=airfoil,
            CL_targets=np.array([1.0]),
            CL_weights=np.array([1.0]),
            RE=np.array([500e3]),
            mach=0.03,
        )

        optimizer.update()

        thickness_33 = optimizer.airfoil.local_thickness(x_over_c=0.33)
        thickness_33_val = float(np.asarray(thickness_33).flatten()[0])
        assert thickness_33_val >= 0.127, f"Thickness at 33% = {thickness_33_val} < 0.127"
        print(f"  Thickness at 33% chord: {thickness_33_val:.4f}")


class TestMultipointOptimization:
    """Test multi-point optimization (HPA case)."""

    def test_multipoint_hpa(self):
        """Test Human-Powered Aircraft multipoint optimization."""
        from piern_airfoil.neuralfoil import NeuralOptimizer

        airfoil = asb.KulfanAirfoil("naca0012")
        CL_targets = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
        CL_weights = np.array([5, 6, 7, 8, 9, 10])
        Re = 500e3 * (CL_targets / 1.25) ** -0.5

        optimizer = NeuralOptimizer(
            airfoil=airfoil,
            CL_targets=CL_targets,
            CL_weights=CL_weights,
            RE=Re,
            mach=0.03,
        )

        optimizer.update()

        CL_vals = np.asarray(optimizer.aero["CL"]).flatten()
        CD_vals = np.asarray(optimizer.aero["CD"]).flatten()

        for i, (cl, cd) in enumerate(zip(CL_vals, CD_vals)):
            print(f"  Point {i+1}: CL={cl:.4f}, CD={cd:.6f}")

        assert len(CL_vals) == 6
        assert all(cl > 0 for cl in CL_vals)


if __name__ == "__main__":
    print("=" * 60)
    print("NeuralFoil Optimization Test Suite")
    print("=" * 60)

    print("\n[1] Testing NeuralFoil Analysis...")
    t1 = TestNeuralFoilAnalysis()
    t1.test_analyze_by_name()
    t1.test_vectorized_analysis()

    print("\n[2] Testing Single-Point Optimization...")
    t2 = TestNeuralFoilOptimization()
    t2.test_single_point_optimization()
    t2.test_constraints_active()

    print("\n[3] Testing Multi-Point Optimization...")
    t3 = TestMultipointOptimization()
    t3.test_multipoint_hpa()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
