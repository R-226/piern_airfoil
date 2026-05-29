"""End-to-end PiERN pipeline demo."""
import aerosandbox as asb
import numpy as np
from piern.pipeline import PiernPipeline

print("=" * 60)
print("PiERN Pipeline — End-to-End Demo")
print("=" * 60)

pipeline = PiernPipeline()

# 1. Prompt → Parameters
prompt = (
    "在Ma=0.03的飞行条件下，要求CL=[0.8,1.0,1.2,1.4,1.5,1.6]，"
    "权重[5,6,7,8,9,10]。约束：CM≥-0.133，后缘角≥6.03°，"
    "前缘角=180°，前缘厚度>0.128，后缘厚度>0.014。请优化翼型"
)
params = pipeline.extract_params(prompt)
print(f"\n[1] Prompt2Data")
print(f"    Mach={params.Mach}")
print(f"    CL={params.CL_targets}")
print(f"    W ={params.CL_weights}")
print(f"    CM >= {params.CM_min}")
print(f"    TE >= {params.TE_angle_min} deg")
print(f"    LE =  {params.LE_angle} deg")
print(f"    t@33% >= {params.thickness_33_min}")
print(f"    t@90% >= {params.thickness_90_min}")

# 2. Airfoil (NACA0012)
airfoil = asb.KulfanAirfoil("naca0012")
print(f"\n[2] Image → KulfanAirfoil ({len(airfoil.upper_weights)} CST weights)")

# 3. Initial evaluation
init_cd = pipeline._quick_eval(airfoil, params)
print(f"\n[3] Initial CD = {init_cd:.6f}")

# 4. Hierarchical Optimization
print(f"\n[4] Running AdaptiveHierarchicalOptimizer...")
optimized, opt_time, history = pipeline.optimize(airfoil, params)
final_cd = pipeline._quick_eval(optimized, params)
print(f"    Final CD   = {final_cd:.6f}")
print(f"    Time       = {opt_time:.2f}s")
print(f"    Improvement = {(init_cd - final_cd) / init_cd * 100:+.2f}%")

# 5. Stage history (Router decisions)
print(f"\n[5] OptRouter Decisions ({len(history)} stages):")
head = f"    {'Stage':<8}{'Weights':<10}{'CD':<12}{'Decision'}"
print(head)
print("    " + "-" * 55)
for h in history:
    print(f"    {h['stage']:<8}{h['n_active_weights']:<10}{h['cd']:<12.6f}{h['message']}")

# 6. Physical constraint check
t33 = float(np.asarray(optimized.local_thickness(x_over_c=0.33)).flatten()[0])
t90 = float(np.asarray(optimized.local_thickness(x_over_c=0.90)).flatten()[0])
te = float(np.asarray(optimized.TE_angle()).flatten()[0])
le_r = float(np.asarray(optimized.LE_radius()).flatten()[0])

print(f"\n[6] Physical Constraints:")
print(f"    t@33% = {t33:.4f}  (min 0.128)  {'OK' if t33 >= 0.128 else 'FAIL'}")
print(f"    t@90% = {t90:.4f}  (min 0.014)  {'OK' if t90 >= 0.014 else 'FAIL'}")
print(f"    TE    = {te:.2f}   (min 6.03)   {'OK' if te >= 6.03 else 'FAIL'}")
print(f"    LE_r  = {le_r:.4f}  (min 0)      {'OK' if le_r > 0 else 'FAIL'}")

# 7. Summary
print(f"\n{'=' * 60}")
print(f"Pipeline Summary:")
print(f"  Input:   Chinese prompt + NACA0012 airfoil")
print(f"  CD:      {init_cd:.4f} → {final_cd:.4f}  ({(init_cd - final_cd) / init_cd * 100:+.1f}% improvement)")
print(f"  Time:    {opt_time:.1f}s")
print(f"  Stages:  {len(history)} (via OptRouter)")
print(f"{'=' * 60}")
