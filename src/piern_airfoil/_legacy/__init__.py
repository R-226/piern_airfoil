"""
Legacy optimization code preserved for reference.

These files were part of earlier exploration phases:
- global_optimizer: Differential Evolution (DE) — too slow for practical use
- gradient_optimizer: L-BFGS-B with NeuralFoil model-size switching — proved ineffective
- multi_fidelity: Fixed two-stage pipeline (xxsmall → large) — no advantage over single-stage
- router/routed_optimizer: Conditional Markov decision router — prototype for current PiERN migration

The current approach uses:
- piern_airfoil.optimizer: CasADi+IPOPT with NeuralFoil (NeuralOptimizer)
- piern_airfoil.hierarchical: Hierarchical CST parameterization (AdaptiveHierarchicalOptimizer)
"""
