"""Gradio Web UI for airfoil optimization.

Unified interface combining prompt-based parameter extraction and
image-based coordinate extraction, running three optimization methods
in parallel and comparing results.

Run: uv run python -m piern.view.app
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ── Project root ───────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Lazy-loaded prompt model ───────────────────────────────────────
_tokenizer = None
_model = None


def _load_prompt_model():
    """Lazy-load the encoder-extractor model on first use."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    import torch
    from piern.prompt2data.encoder_extractor import (
        DEVICE,
        SAVE_DIR,
        CharTokenizer,
        FieldClassifier,
        NUM_FIELDS,
    )

    _tokenizer = CharTokenizer(max_len=512)
    _model = FieldClassifier(
        vocab_size=_tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_ff=512,
        max_len=512,
        num_fields=NUM_FIELDS,
    ).to(DEVICE)
    _model.load_state_dict(torch.load(SAVE_DIR, map_location=DEVICE, weights_only=True))
    _model.eval()
    return _tokenizer, _model


# ── Extraction helpers ─────────────────────────────────────────────


def extract_params_from_prompt(prompt: str) -> dict:
    """Extract aerodynamic parameters from a Chinese NL prompt."""
    from piern.prompt2data.encoder_extractor import extract
    if not prompt or not prompt.strip():
        return {}

    tokenizer, model = _load_prompt_model()
    return extract(model, tokenizer, prompt)


def extract_contour_from_image(image_path: str | None):
    """Extract airfoil contour from an uploaded image. Returns (AirfoilContour | None, fig)."""
    if image_path is None:
        return None, None

    from piern.view.extract import extract_airfoil

    contour = extract_airfoil(image_path)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(contour.x_surface, contour.y_upper, "b-", linewidth=1.5, label="Upper")
    ax.plot(contour.x_surface, contour.y_lower, "r-", linewidth=1.5, label="Lower")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Extracted airfoil contour")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return contour, fig


def neuralfoil_optimization(initial_guess_airfoil, inputs):
    import aerosandbox.numpy as np
    import aerosandbox as asb
    CL_multipoint_targets = np.array(inputs["CL"])
    CL_multipoint_weights = np.array(inputs["weights"])
    Re = 500e3 * (CL_multipoint_targets / 1.25) ** -0.5

    opti = asb.Opti()

    optimized_airfoil = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=opti.variable(
            init_guess=initial_guess_airfoil.lower_weights,
            lower_bound=-0.5,
            upper_bound=0.25,
        ),
        upper_weights=opti.variable(
            init_guess=initial_guess_airfoil.upper_weights,
            lower_bound=-0.25,
            upper_bound=0.5,
        ),
        leading_edge_weight=opti.variable(
            init_guess=initial_guess_airfoil.leading_edge_weight,
            lower_bound=-1,
            upper_bound=1,
        ),
        TE_thickness=0,
    )

    alpha = opti.variable(
        init_guess=np.degrees(CL_multipoint_targets / (2 * np.pi)),
        lower_bound=-5,
        upper_bound=18,
    )

    aero = optimized_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=inputs["Mach"],
    )

    opti.subject_to(
        [
            aero["analysis_confidence"] > 0.90,
            aero["CL"] == CL_multipoint_targets,
            np.diff(alpha) > 0,
            aero["CM"] >= inputs["CM_lower_bound"],
            optimized_airfoil.local_thickness(x_over_c=0.33) >= inputs["thickness_head_lower_bound"],
            optimized_airfoil.local_thickness(x_over_c=0.90) >= inputs["thickness_tail_lower_bound"],
            optimized_airfoil.TE_angle()
            >= inputs["Trailing_edge_angle_lower_bound"],  # Modified from Drela's 6.25 to match DAE-11 case
            optimized_airfoil.lower_weights[0] < -0.05,
            optimized_airfoil.upper_weights[0] > 0.05,
            optimized_airfoil.local_thickness() > 0,
        ]
    )

    get_wiggliness = lambda af: sum(
        [
            np.sum(np.diff(np.diff(array)) ** 2)
            for array in [af.lower_weights, af.upper_weights]
        ]
    )

    opti.subject_to(
        get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(initial_guess_airfoil)
    )

    opti.minimize(np.mean(aero["CD"] * CL_multipoint_weights))

    sol = opti.solve(
        behavior_on_failure="return_last",
        options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
    )

    optimized_airfoil = sol(optimized_airfoil)
    aero = sol(aero)
    return optimized_airfoil, aero


def build_constraints(params: dict):
    """Build AirfoilConstraints from extracted prompt parameters."""
    from piern_airfoil.thin_airfoil.constraints import AirfoilConstraints

    cl = params.get("CL", [0.0] * 6)
    weights = params.get("weights", [0] * 6)
    cl_targets = [c for c, w in zip(cl, weights) if w > 0]
    cl_weights = [w for w in weights if w > 0]

    return AirfoilConstraints(
        CL_targets=np.array(cl_targets) if cl_targets else None,
        CL_weights=np.array(cl_weights, dtype=float) if cl_weights else None,
        CM_min=params.get("CM_lower_bound", -0.133),
        thickness_at_33_min=params.get("thickness_head_lower_bound", 0.128),
        thickness_at_90_min=params.get("thickness_tail_lower_bound", 0.014),
        TE_angle_min=params.get("Trailing_edge_angle_lower_bound", 6.03),
    )


# ── Optimization runners ──────────────────────────────────────────


@dataclass
class OptMethodResult:
    """Result from a single optimization method."""

    name: str
    airfoil: object  # asb.KulfanAirfoil
    objective: float
    elapsed: float
    stats: dict


def _run_thin_de(initial_airfoil, constraints, mach: float) -> OptMethodResult:
    """Run thin airfoil theory + differential evolution (global search)."""
    import aerosandbox as asb

    from piern_airfoil.thin_airfoil.constraints import FidelityLevel
    from piern_airfoil.thin_airfoil.global_optimizer import (
        GlobalAirfoilOptimizer,
        OptimizerConfig,
    )

    t0 = time.perf_counter()
    optimizer = GlobalAirfoilOptimizer.for_kulfan_airfoil(
        airfoil=initial_airfoil,
        constraints=constraints,
        alpha=5.0,
        Re=500e3,
        mach=mach,
        fidelity=FidelityLevel.THIN,
        config=OptimizerConfig(maxiter=30, popsize=8, seed=42),
    )
    result = optimizer.optimize()
    elapsed = time.perf_counter() - t0

    n_upper = len(initial_airfoil.upper_weights)
    n_lower = len(initial_airfoil.lower_weights)
    airfoil = asb.KulfanAirfoil(
        name="ThinDE",
        upper_weights=np.array(result.x[:n_upper]),
        lower_weights=np.array(result.x[n_upper : n_upper + n_lower]),
        leading_edge_weight=float(result.x[-1]),
        TE_thickness=0.0,
    )

    return OptMethodResult(
        name="Thin DE",
        airfoil=airfoil,
        objective=result.fun,
        elapsed=elapsed,
        stats={"nfev": result.nfev, "nit": result.nit, "success": result.success},
    )


def _run_neuralfoil(initial_airfoil, params) -> OptMethodResult:
    """Run NeuralFoil + IPOPT gradient optimization (local refinement)."""
    t0 = time.perf_counter()
    optimized_airfoil, _ = neuralfoil_optimization(initial_airfoil, params)
    elapsed = time.perf_counter() - t0
    return OptMethodResult(
        name="NeuralFoil",
        airfoil=optimized_airfoil,
        objective=0.0,
        elapsed=elapsed,
        stats={},
    )



def _run_multifidelity(initial_airfoil, constraints, mach: float) -> OptMethodResult:
    """Run multi-fidelity optimization (L-BFGS-B → IPOPT)."""
    from piern_airfoil.thin_airfoil.gradient_optimizer import GradientOptConfig
    from piern_airfoil.thin_airfoil.multi_fidelity import multi_fidelity_optimize

    t0 = time.perf_counter()
    result = multi_fidelity_optimize(
        initial_airfoil=initial_airfoil,
        constraints=constraints,
        alpha=5.0,
        Re=500e3,
        mach=mach,
        stage1_config=GradientOptConfig(model_size="xxsmall", maxiter=300, maxfun=5000),
        neural_max_iterations=2,
    )
    elapsed = time.perf_counter() - t0

    return OptMethodResult(
        name="Multi-Fidelity",
        airfoil=result.airfoil,
        objective=float(result.stage1_result.best_cd) if result.stage1_result else 0.0,
        elapsed=elapsed,
        stats={
            "stage1_nfev": result.stage1_nfev,
            "stage2_nfev": result.stage2_nfev,
            "stage2_iterations": result.stage2_iterations,
        },
    )


# ── Main callback ─────────────────────────────────────────────────


def run_optimization(prompt: str, image):
    """Extract inputs, run 3 optimizations in parallel, compare results."""
    import aerosandbox as asb

    # 1. Extract parameters from prompt
    params = extract_params_from_prompt(prompt) if prompt.strip() else {}

    # 2. Extract contour from image
    image_path = image if image else None
    contour, contour_fig = extract_contour_from_image(image_path)

    # 3. Build initial airfoil
    if contour is not None:
        upper_surface = np.column_stack([contour.x_surface, contour.y_upper])
        lower_surface = np.column_stack([contour.x_surface, contour.y_lower])
        coordinates = np.vstack([upper_surface, lower_surface[::-1][1:]])
        initial_airfoil = asb.Airfoil(coordinates=coordinates).to_kulfan_airfoil()
    else:
        initial_airfoil = asb.KulfanAirfoil("naca0012")

    # 4. Build constraints
    constraints = build_constraints(params)
    mach = params.get("Mach", 0.03)

    # 5. Run 3 optimizations in parallel
    results: dict[str, OptMethodResult] = {}
    runners = {
        "thin": (_run_thin_de, initial_airfoil, constraints, mach),
        "neural": (_run_neuralfoil, initial_airfoil, params),
        "multi": (_run_multifidelity, initial_airfoil, constraints, mach),
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(fn, *args): key for key, (fn, *args) in runners.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = OptMethodResult(
                    name=key, airfoil=initial_airfoil, objective=float("inf"),
                    elapsed=0.0, stats={"error": str(e)},
                )

    # 6. Build comparison plot
    comparison_fig = _build_comparison_plot(initial_airfoil, results)

    # 7. Build summary JSON
    summary = {}
    for key in ("thin", "neural", "multi"):
        r = results[key]
        summary[r.name] = {
            "objective": f"{r.objective:.6f}",
            "elapsed": f"{r.elapsed:.2f}s",
            **r.stats,
        }

    return params, contour_fig, summary, comparison_fig


def _build_comparison_plot(initial_airfoil, results: dict[str, OptMethodResult]):
    """Plot all optimized airfoils overlaid for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    colors = {"thin": "#2196F3", "neural": "#F44336", "multi": "#4CAF50"}
    labels = {"thin": "Thin DE", "neural": "NeuralFoil", "multi": "Multi-Fidelity"}

    # Left: shape comparison (flip x to match image orientation: TE at x=0)
    ax = axes[0]
    init_coords = initial_airfoil.coordinates
    ax.plot(init_coords[:, 0], init_coords[:, 1], "k--", linewidth=1, alpha=0.5, label="Initial")

    for key in ("thin", "neural", "multi"):
        r = results[key]
        coords = r.airfoil.coordinates
        ax.plot(1.0 - coords[:, 0], coords[:, 1], color=colors[key], linewidth=1.5, label=labels[key])

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Airfoil shape comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of objective + time
    ax = axes[1]
    names = [labels[k] for k in ("thin", "neural", "multi")]
    objs = [results[k].objective for k in ("thin", "neural", "multi")]
    times = [results[k].elapsed for k in ("thin", "neural", "multi")]
    bar_colors = [colors[k] for k in ("thin", "neural", "multi")]

    x = np.arange(len(names))
    width = 0.35

    ax2 = ax.twinx()
    bars1 = ax.bar(x - width / 2, objs, width, color=bar_colors, alpha=0.7, label="Objective")
    bars2 = ax2.bar(x + width / 2, times, width, color=bar_colors, alpha=0.35, label="Time (s)")

    ax.set_ylabel("Objective (weighted CD)")
    ax2.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Objective & runtime comparison")

    # Add value labels on bars
    for bar, val in zip(bars1, objs):
        if val < float("inf"):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.5f}",
                    ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}s",
                 ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    return fig


# ── Gradio UI ──────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PiERN Airfoil Optimizer") as demo:
        gr.Markdown("# PiERN Airfoil Optimizer")
        gr.Markdown(
            "Combine natural language prompts and airfoil images. "
            "Three optimization methods run **in parallel** and results are compared."
        )

        with gr.Row():
            # Left: inputs
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Optimization prompt",
                    placeholder=(
                        "例：设计一个翼型，马赫数0.03，CL目标值[0.8, 1.0, 1.2, 1.4, 1.5, 1.6]，"
                        "权重[5, 6, 7, 8, 9, 10]，力矩系数≥-0.133，后缘角≥6.03°，"
                        "厚度@33%c≥0.128，厚度@90%c≥0.014"
                    ),
                    lines=4,
                )
                image_input = gr.Image(label="Airfoil image", type="filepath")
                run_btn = gr.Button("Extract & Optimize (3 methods)", variant="primary")

            # Right: outputs
            with gr.Column(scale=1):
                params_output = gr.JSON(label="Extracted parameters")
                contour_plot = gr.Plot(label="Extracted contour")

        # Full-width comparison section
        gr.Markdown("## Optimization Comparison")
        with gr.Row():
            result_output = gr.JSON(label="Results summary")
        with gr.Row():
            comparison_plot = gr.Plot(label="Shape & Performance Comparison")

        run_btn.click(
            fn=run_optimization,
            inputs=[prompt_input, image_input],
            outputs=[params_output, contour_plot, result_output, comparison_plot],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
