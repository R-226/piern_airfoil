# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PIERN-Airfoil (Physics-Infused Expert Reasoning Network) is a unified airfoil shape optimization framework with **Hierarchical CST Parameterization** as the core innovation:

- **CasADi+IPOPT Baseline**: gradient-based via Aerosandbox KulfanAirfoil + IPOPT (~5s, high accuracy)
- **Hierarchical CST**: two-stage pipeline with different parameterization dimensions
  - Stage 1: Low-dimensional search (4 CST weights/edge) → fast feasible solution
  - Stage 2: High-dimensional refinement (8 CST weights/edge) → shape detail optimization

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
uv run pytest tests/test_multi_fidelity.py -v   # single file
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
uv run python tests/test_hierarchical_cst.py             # hierarchical CST test
uv run python -m piern_airfoil.neuralfoil.neuralfoil     # NeuralOptimizer demo
uv run python -m piern.prompt2data.encoder_extractor     # prompt2data training
```

## Architecture

The codebase has two top-level packages under `src/`:

### `piern_airfoil/` — Core optimization engines

- **`neuralfoil/neuralfoil.py`**: `NeuralOptimizer` wraps Aerosandbox's `Opti` + `KulfanAirfoil` + NeuralFoil. Minimizes weighted CD subject to constraints (CL, CM, thickness, TE angle, wiggliness). Supports warm-starting.
- **`thin_airfoil/`**: The multi-fidelity subsystem:
  - `thin_airfoil_solver.py`: Classical thin airfoil theory (Glauert Fourier coefficients, Prandtl-Glauert compressibility). Provides `thin_airfoil_from_kulfan()` bridge to Kulfan parameterization.
  - `constraints.py`: `AirfoilConstraints` dataclass — unified interface for geometry (always enforced) and aerodynamic (NEURAL fidelity only) constraints.
  - `hierarchical_optimizer.py`: **Core innovation** — Hierarchical CST parameterization optimization. Stage 1: low-dimensional search (4 weights/edge), Stage 2: high-dimensional refinement (8 weights/edge).
  - `router.py` + `routed_optimizer.py`: Router for selecting optimization strategies (under development).

### `piern/` — LLM integration and UI

- **`prompt2data/`**: Extracts structured params (Mach, CL, weights, constraints) from Chinese text. Active model: `encoder_extractor.py` (regex number extraction + Transformer classifier, 18 output classes, ~3.3M params). Deprecated: `mlp.py`, `mlp_hidden.py`.
- **`seq_level/`**: Binary classifier detecting "trigger boundaries" in LLM output — where reasoning ends and results should begin. Uses frozen Qwen3.5-0.8B embeddings + MLP.
- **`switch/`**: Alternative namespace for routed optimization (delegates to `piern_airfoil.thin_airfoil`).
- **`view/`**: Gradio web UI (`app.py`) — accepts Chinese prompt + airfoil image, runs 3 optimization methods in parallel, compares results. `extract.py` detects blue contour pixels from images.

### Data flow

```
Chinese prompt + airfoil image
  -> prompt2data (params) + view.extract (coordinates)
  -> asb.Airfoil.to_kulfan_airfoil() (initial guess)
  -> optimization engines (TAT+DE, NeuralFoil+IPOPT, multi-fidelity, routed)
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
- **Tests are gitignored**: The `tests/` directory is in `.gitignore`.
- **Deprecated modules**: `piern/prompt2data/mlp.py`, `mlp_hidden.py`, and `piern/switch/` are deprecated. Use `encoder_extractor.py` and `piern_airfoil/thin_airfoil/` instead.
