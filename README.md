# PiERN-Airfoil

Hierarchical CST airfoil optimization with adaptive fidelity routing.

## Overview

PiERN-Airfoil uses **CST parameterization dimension as the fidelity axis** for multi-fidelity airfoil optimization. Instead of switching between different physics models, it adaptively expands the number of CST weights (4 -> 6 -> 8) based on optimization history.

```
Initial Airfoil (NACA xxxx)
    |
    v
Stage 1: Optimize 4 CST weights/edge (low-dimensional, fast)
    |
    v  OptRouter decides: CONTINUE or EXPAND?
Stage 2: Optimize 6 CST weights/edge (medium fidelity)
    |
    v  OptRouter decides: CONTINUE or EXPAND?
Stage 3: Optimize 8 CST weights/edge (full fidelity)
    |
    v
Optimized Airfoil
```

## Installation

```bash
git clone https://github.com/R-226/piern_airfoil.git
cd piern_airfoil
uv sync                          # install core deps
uv pip install -e .              # editable install

# Optional: for LLM prompt extraction training
uv sync --extra train

# Optional: for Gradio web UI
uv sync --extra ui
```

## Quick Start

### Python API

```python
import aerosandbox as asb
from piern_airfoil import AdaptiveHierarchicalOptimizer
from piern.router import OptRouter
import numpy as np

# Define optimization problem
CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5

# Create optimizer with learned router
router = OptRouter.from_mlp()  # or .from_trained() for threshold mode
optimizer = AdaptiveHierarchicalOptimizer(
    CL_targets=CL_TARGETS,
    CL_weights=CL_WEIGHTS,
    Re=RE,
    mach=0.03,
    router=router,
)

# Optimize
airfoil = asb.KulfanAirfoil("naca0012")
result = optimizer.optimize(airfoil)

print(f"Final CD: {result.final_cd:.6f}")
print(f"Stages: {len(result.stages)}")
for stage in result.stages:
    print(f"  Stage {stage.stage}: {stage.n_active_weights}w -> CD={stage.cd:.6f}")
```

### Pipeline (Chinese prompt + image)

```python
from piern.pipeline import PiernPipeline

pipeline = PiernPipeline(router_mode="mlp")
result = pipeline.run(
    prompt="设计一个翼型，马赫数0.03，CL目标是0.8到1.6...",
    airfoil_image="path/to/airfoil.png",
)
pipeline.visualize(result)
```

### Web UI

```bash
uv run python -m piern.view.app
```

## Architecture

```
src/
├── piern_airfoil/              # Core optimization engines
│   ├── optimizer.py            # Baseline: single-stage 8-weight IPOPT
│   ├── hierarchical.py         # Innovation: adaptive 4->8 weight expansion
│   ├── constraints.py          # AirfoilConstraints dataclass
│   ├── thin_airfoil.py         # Classical thin airfoil theory
│   └── _legacy/                # Reference implementations (DE, L-BFGS-B, etc.)
│
├── piern/                      # Integration layer
│   ├── router/                 # Fidelity routing
│   │   ├── opt_router.py       # OptRouter: rule/threshold/mlp modes
│   │   ├── mlp_router.py       # MLP router training (~1000 params)
│   │   ├── train_threshold.py  # Grid search for optimal threshold
│   │   └── trained/            # Saved model weights
│   ├── prompt2data/            # Chinese NL -> structured parameters
│   │   ├── encoder_extractor.py # Regex + Transformer classifier
│   │   └── generate_diverse.py # Synthetic training data generation
│   ├── view/                   # Gradio web UI
│   │   ├── app.py              # Interactive optimization interface
│   │   ├── extract.py          # Airfoil image contour extraction
│   │   └── verify.py           # Visual verification tools
│   └── pipeline.py             # End-to-end orchestration
```

## OptRouter Modes

| Mode | Description | Training |
|------|-------------|----------|
| `rule` | Fixed improvement threshold (0.01) | None |
| `threshold` | Learned threshold via grid search | `train_threshold.py` |
| `mlp` | Learned MLP policy (~1000 params) | `mlp_router.py` |

## Benchmark Results

### Normal Cases (6 airfoils)
PiERN achieves best CD (0.427463) while being 29-33% faster than rule/threshold methods.

| Method | Avg CD | Avg Time | Avg Stages |
|--------|--------|----------|------------|
| Baseline (8w IPOPT) | 0.428923 | 2.1s | 1.0 |
| Rule (fixed threshold) | 0.427871 | 10.4s | 5.5 |
| Threshold (learned) | 0.429097 | 11.1s | 6.0 |
| **PiERN Router** | **0.427463** | **7.4s** | **3.7** |

### Hard Cases (4 airfoils)
Baseline fails completely (0% success), PiERN rescues 75% of cases.

| Method | Success Rate | Avg CD |
|--------|-------------|--------|
| Baseline | 0/4 (0%) | 1.490 |
| Rule | 3/4 (75%) | 0.461 |
| Threshold | 3/4 (75%) | 0.461 |
| **PiERN** | **3/4 (75%)** | **0.460** |

## Tests

```bash
uv run pytest tests/ -v                      # all tests
uv run pytest tests/test_pipeline.py -v      # pipeline integration
uv run pytest tests/ -k "test_router"        # router unit tests
```

## Development

```bash
uv run black .                    # format
uv run ruff check src/            # lint
uv run mypy src/                  # type check
```

## License

MIT
