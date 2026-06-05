# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PIERN-Airfoil (Physics-Infused Expert Reasoning Network) is a unified airfoil shape optimization framework with **Hierarchical CST Parameterization** as the core innovation:

- **CasADi+IPOPT Baseline**: gradient-based via Aerosandbox KulfanAirfoil + IPOPT (~5s, high accuracy)
- **Hierarchical CST**: adaptive multi-stage pipeline using CST parameterization dimension as the fidelity axis
  - Stage 1: Low-dimensional search (4 CST weights/edge) → fast feasible solution
  - Stage 2: Medium refinement (6 CST weights/edge) → mid-chord detail
  - Stage 3: Full fidelity (8 CST weights/edge) → shape detail optimization
  - OptRouter (rule/threshold/mlp) decides when to expand dimensions
- **XFoil+DE Baseline**: classic black-box optimization using Differential Evolution + XFoil evaluation

The project also includes an LLM-powered system that extracts structured optimization parameters from Chinese natural-language prompts.

## Commands

### Install
```bash
uv sync                          # install all deps (including dev)
uv pip install -e .              # editable install
```

### Run Tests
```bash
uv run pytest                    # all tests
uv run pytest tests/test_pipeline.py -v   # single file
uv run pytest -k "test_name"     # single test by name
uv run pytest --cov=src --cov-report=term-missing
```

### Lint & Format
```bash
uv run black .                   # format (line-length=100, target py311)
uv run ruff check src/           # lint
uv run mypy src/                 # type check
```

### Key Entry Points
```bash
uv run python -m piern.view.app                          # Gradio web UI
uv run python -m piern.prompt2data.encoder_extractor     # prompt2data training
uv run python -m piern.router.train_threshold            # A1: learn optimal threshold
uv run python -m piern.router.mlp_router                 # A2: train MLP router
uv run python tests/benchmark_router.py                  # router benchmark (4 methods + XFoil analysis, 105 airfoils)
uv run python tests/benchmark_pipeline.py                # pipeline benchmark (ground truth vs image)
uv run python tests/benchmark_ablation.py                # ablation study (4 experiments + sensitivity)
uv run python tests/run_all_benchmarks.py                # run all benchmarks (one-click)
```

## Architecture

The codebase has two top-level packages under `src/`:

### `piern_airfoil/` — Core optimization engines

- **`optimizer.py`**: `NeuralOptimizer` wraps Aerosandbox's `Opti` + `KulfanAirfoil` + NeuralFoil. Minimizes weighted CD subject to constraints (CL, CM, thickness, TE angle, LE radius, wiggliness). Supports warm-starting via IPOPT.
- **`hierarchical.py`**: **Core innovation** — `AdaptiveHierarchicalOptimizer` uses CST parameterization dimension as the fidelity axis. Starts from low-dimensional (4 weights/edge), adaptively expands to 6 and then 8 weights/edge based on convergence history. Uses `OptRouter` for routing decisions.
- **`eval.py`**: Shared `evaluate_weighted_cd()` — single-point NeuralFoil CD evaluation used across optimizer, pipeline, and router training.
- **`xfoil_optimizer.py`**: Classic baseline — XFoil + Differential Evolution (scipy). Black-box optimization using XFoil as the objective evaluator. Used as comparison baseline in benchmarks.
- **`_legacy/`**: Earlier exploration code kept as **baseline comparisons for ablation studies**:
  - `global_optimizer.py`: Differential Evolution (DE) — global search baseline
  - `gradient_optimizer.py`: L-BFGS-B — local gradient baseline
  - `multi_fidelity.py`: Model-size switching approach — predecessor to hierarchical CST
  - `routed_optimizer.py` / `router.py`: Early router prototypes — predecessors to OptRouter

### `piern/` — LLM integration, router, and UI

- **`router/`**: Optimization fidelity router (`OptRouter`) — decides when to expand CST weights.
  - **`opt_router.py`**: Three modes: `rule` (fixed threshold), `threshold` (grid-search learned), `mlp` (learned 8-dim→3-action MLP, ~1000 params)
  - **`train_threshold.py`**: Grid search for optimal improvement_threshold
  - **`mlp_router.py`**: MLP router training pipeline
  - **`trained/`**: Saved model weights (optimal_threshold.json, mlp_router.json)
- **`prompt2data/`**: Extracts structured params (Mach, CL, weights, constraints) from Chinese text. Active model: `encoder_extractor.py` (regex number extraction + Transformer classifier, 18 output classes, ~3.3M params). `deprecated/` contains earlier MLP approaches kept for reference.
- **`view/`**: Gradio web UI (`app.py`) — accepts Chinese prompt + airfoil image, runs Baseline + PiERN optimization in parallel, compares results. `extract.py` supports color-based and edge-based (Sobel) airfoil contour extraction from images.

### Data flow

```
Chinese prompt + airfoil image
  -> prompt2data (params) + view.extract (coordinates)
  -> asb.Airfoil.to_kulfan_airfoil() (initial guess)
  -> Hierarchical CST optimization
     -> OptRouter (rule/threshold/mlp) decides fidelity transitions
     -> Stage 1: 4 weights → Stage 2: 6-8 weights
  -> comparison & visualization
```

## Key Conventions

- **Package manager**: uv (not pip)
- **Python**: 3.11 target
- **Core domain library**: Aerosandbox (`asb`) — provides `KulfanAirfoil`, `Opti` (IPOPT wrapper), `Airfoil`, and NeuralFoil integration. Most aerodynamic concepts map to Aerosandbox types.
- **Kulfan parameterization**: Airfoil shapes are represented as `KulfanAirfoil` with upper/lower weight arrays + leading edge weight. This is the common currency between all optimization methods.
- **Normalization**: The 18 aerodynamic parameters use mean/std normalization stored in `data/2com/normalization_params.json`.
- **Chinese text**: Prompt templates and training data are in Chinese. The `encoder_extractor` works with Chinese numerical expressions.
- **Checkpoints**: Model weights live in `checkpoint/` (not version-controlled). The Qwen3.5-0.8B base model is in `model/`.
- **Tests**: `tests/output/` is gitignored. Test scripts are tracked in git.
