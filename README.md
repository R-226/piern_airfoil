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
uv sync
uv pip install -e .
```

Optional dependencies:

```bash
uv sync --extra train    # LLM prompt extraction training (torch)
uv sync --extra ui       # Gradio web UI
```

## Quick Start

### Python API

```python
import aerosandbox as asb
from piern_airfoil import AdaptiveHierarchicalOptimizer
from piern.router import OptRouter
import numpy as np

CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
RE = 500e3 * (CL_TARGETS / 1.25) ** -0.5

router = OptRouter.from_mlp()
optimizer = AdaptiveHierarchicalOptimizer(
    CL_targets=CL_TARGETS, CL_weights=CL_WEIGHTS,
    Re=RE, mach=0.03, router=router,
)

airfoil = asb.KulfanAirfoil("naca0012")
result = optimizer.optimize(airfoil)
print(f"Final CD: {result.final_cd:.6f}")
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
├── piern_airfoil/                  # Core optimization engines
│   ├── optimizer.py                # Baseline: single-stage 8-weight IPOPT
│   ├── hierarchical.py             # Core: adaptive 4->8 weight expansion
│   ├── eval.py                     # Shared weighted CD evaluation (NeuralFoil)
│   ├── xfoil_optimizer.py          # Classic baseline: XFoil + Differential Evolution
│   └── _legacy/                    # Ablation baselines (DE, L-BFGS-B, early router)
│
├── piern/                          # Integration layer
│   ├── router/                     # Fidelity routing
│   │   ├── opt_router.py           # OptRouter: rule/threshold/mlp modes
│   │   ├── mlp_router.py           # MLP router training (~1000 params)
│   │   ├── train_threshold.py      # Grid search for optimal threshold
│   │   └── trained/                # Saved model weights
│   ├── prompt2data/                # Chinese NL -> structured parameters
│   │   └── encoder_extractor.py    # Regex + Transformer classifier
│   ├── view/                       # Gradio web UI
│   │   ├── app.py                  # Interactive optimization interface
│   │   └── extract.py              # Airfoil image contour extraction (edge detection)
│   └── pipeline.py                 # End-to-end orchestration
│
tests/
├── benchmark_router.py             # 5 methods x 105 airfoils comparison
├── benchmark_pipeline.py           # Ground truth vs image extraction accuracy
├── benchmark_ablation.py           # 4 ablations + sensitivity analysis
└── run_all_benchmarks.py           # One-click benchmark orchestrator
```

## Optimization Methods

| Method | Description |
|--------|-------------|
| Baseline | Direct 8-weight IPOPT (single-stage) |
| Rule | Hierarchical with fixed threshold (0.01) |
| Threshold | Hierarchical with learned threshold (grid search) |
| **PiERN Router** | Hierarchical with learned MLP policy (~1000 params) |
| XFoil+DE | Classic: Differential Evolution + XFoil black-box evaluation |

## OptRouter Modes

| Mode | Description | Training |
|------|-------------|----------|
| `rule` | Fixed improvement threshold (0.01) | None |
| `threshold` | Learned threshold via grid search | `train_threshold.py` |
| `mlp` | Learned MLP policy (~1000 params) | `mlp_router.py` |

## Benchmark Suite

Three benchmarks covering optimization quality, extraction accuracy, and design choices:

```bash
uv run python tests/run_all_benchmarks.py
```

### Router Benchmark
Compares 5 optimization methods on 105 airfoils (Normal/Medium/Hard).

Output: `benchmark_stats.csv`, `table_router_full.csv`, `table_router_latex.tex`, `table_significance.csv`, plus 14 figures (per-category, distribution, difficulty-improvement, method comparison).

### Pipeline Benchmark
Compares ground truth vs image-based extraction accuracy.

Output: `pipeline_benchmark.csv`, plus 4 figures with decomposition metrics (extraction time, optimization time, Kulfan fit error).

### Ablation Study
4 experiments + sensitivity analysis validating design choices (hierarchical vs direct, router effect, starting dimension, per-stage contribution).

Output: `ablation.csv`, plus 6 figures.

## Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_pipeline.py -v
uv run pytest tests/ -k "test_router"
```

## Development

```bash
uv run black .
uv run ruff check src/
uv run mypy src/
```

## License

MIT
