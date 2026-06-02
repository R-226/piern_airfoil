"""Gradio Web UI for PiERN airfoil optimization.

Uses the hierarchical CST optimizer with adaptive fidelity routing.

Run: uv run python -m piern.view.app
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

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


def extract_contour_from_input(file_path: str | None, method: str = "auto"):
    """Extract airfoil contour from an uploaded image or .dat file."""
    if file_path is None:
        return None, None

    from piern.view.extract import extract_airfoil

    contour = extract_airfoil(file_path, method=method)

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


# ── Optimization ───────────────────────────────────────────────────


def _run_hierarchical(initial_airfoil, params: dict, router_mode: str = "mlp"):
    """Run hierarchical CST optimization with router."""
    from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
    from piern.router.opt_router import OptRouter

    import aerosandbox as asb

    cl = params.get("CL", [0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    weights = params.get("weights", [5, 6, 7, 8, 9, 10])
    mach = params.get("Mach", 0.03)

    CL_targets = np.array(cl, dtype=float)
    CL_weights = np.array(weights, dtype=float)
    RE = 500e3 * (CL_targets / 1.25) ** -0.5

    if router_mode == "mlp":
        router = OptRouter.from_mlp()
    elif router_mode == "threshold":
        router = OptRouter.from_trained()
    else:
        router = OptRouter()

    optimizer = AdaptiveHierarchicalOptimizer(
        CL_targets=CL_targets,
        CL_weights=CL_weights,
        Re=RE,
        mach=mach,
        start_weights=4,
        router=router,
    )

    t0 = time.perf_counter()
    result = optimizer.optimize(initial_airfoil)
    elapsed = time.perf_counter() - t0

    return result.airfoil, result.final_cd, elapsed, result.stages


def _run_baseline(initial_airfoil, params: dict):
    """Run baseline 8-weight IPOPT optimization."""
    from piern_airfoil.optimizer import NeuralOptimizer

    import aerosandbox as asb

    cl = params.get("CL", [0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    weights = params.get("weights", [5, 6, 7, 8, 9, 10])
    mach = params.get("Mach", 0.03)

    CL_targets = np.array(cl, dtype=float)
    CL_weights = np.array(weights, dtype=float)
    RE = 500e3 * (CL_targets / 1.25) ** -0.5

    opt = NeuralOptimizer(
        airfoil=initial_airfoil,
        CL_targets=CL_targets,
        CL_weights=CL_weights,
        RE=RE,
        mach=mach,
    )

    t0 = time.perf_counter()
    opt.update()
    elapsed = time.perf_counter() - t0

    return opt.airfoil, elapsed


# ── Main callback ─────────────────────────────────────────────────


def run_optimization(
    prompt: str,
    image,
    dat_file,
    router_mode: str,
    extract_method: str = "auto",
):
    """Extract inputs, run hierarchical + baseline optimization, compare results."""
    import aerosandbox as asb

    # 1. Extract parameters from prompt
    params = extract_params_from_prompt(prompt) if prompt and prompt.strip() else {}

    # 2. Extract contour from image or .dat file (.dat takes priority)
    file_path = None
    effective_method = extract_method
    if dat_file is not None:
        file_path = dat_file if isinstance(dat_file, str) else dat_file.name
        effective_method = "dat"
    elif image is not None:
        file_path = image
    contour, contour_fig = extract_contour_from_input(file_path, method=effective_method)

    # 3. Build initial airfoil
    if contour is not None:
        upper_surface = np.column_stack([contour.x_surface, contour.y_upper])
        lower_surface = np.column_stack([contour.x_surface, contour.y_lower])
        coordinates = np.vstack([upper_surface, lower_surface[::-1][1:]])
        initial_airfoil = asb.Airfoil(coordinates=coordinates).to_kulfan_airfoil()
    else:
        initial_airfoil = asb.KulfanAirfoil("naca0012")

    # 4. Evaluate initial CD
    from piern.pipeline import PiernPipeline

    pipeline = PiernPipeline()
    init_cd = pipeline._quick_eval(initial_airfoil, _params_to_extraction(params))

    # 5. Run optimizations
    results = {}

    # Baseline
    try:
        baseline_af, baseline_time = _run_baseline(initial_airfoil, params)
        baseline_cd = pipeline._quick_eval(baseline_af, _params_to_extraction(params))
        results["baseline"] = {
            "airfoil": baseline_af,
            "cd": baseline_cd,
            "time": baseline_time,
            "stages": [],
        }
    except Exception as e:
        results["baseline"] = {"error": str(e)}

    # Hierarchical (PiERN)
    try:
        piern_af, piern_cd, piern_time, piern_stages = _run_hierarchical(
            initial_airfoil, params, router_mode
        )
        results["piern"] = {
            "airfoil": piern_af,
            "cd": piern_cd,
            "time": piern_time,
            "stages": piern_stages,
        }
    except Exception as e:
        results["piern"] = {"error": str(e)}

    # 6. Build comparison plot
    comparison_fig = _build_comparison_plot(initial_airfoil, init_cd, results)

    # 7. Build summary
    summary = {}
    if "baseline" in results and "error" not in results["baseline"]:
        r = results["baseline"]
        summary["Baseline (8w IPOPT)"] = {
            "CD": f"{r['cd']:.6f}",
            "Time": f"{r['time']:.2f}s",
        }
    if "piern" in results and "error" not in results["piern"]:
        r = results["piern"]
        summary["PiERN Router"] = {
            "CD": f"{r['cd']:.6f}",
            "Time": f"{r['time']:.2f}s",
            "Stages": len(r["stages"]),
        }
    summary["Initial"] = {"CD": f"{init_cd:.6f}"}

    return params, contour_fig, summary, comparison_fig


def _params_to_extraction(params: dict):
    """Convert raw dict to ExtractionResult for pipeline eval."""
    from piern.pipeline import ExtractionResult

    cl = params.get("CL", [0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
    weights = params.get("weights", [5, 6, 7, 8, 9, 10])

    return ExtractionResult(
        Mach=float(params.get("Mach", 0.03)),
        CL_targets=np.array(cl, dtype=float),
        CL_weights=np.array(weights, dtype=float),
    )


def _build_comparison_plot(initial_airfoil, init_cd: float, results: dict):
    """Plot airfoil shapes and performance comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: shape comparison
    ax = axes[0]
    init_coords = initial_airfoil.coordinates
    ax.plot(init_coords[:, 0], init_coords[:, 1], "k--", linewidth=1, alpha=0.5, label="Initial")

    colors = {"baseline": "#1F77B4", "piern": "#E45756"}
    labels = {"baseline": "Baseline", "piern": "PiERN"}

    for key in ("baseline", "piern"):
        if key in results and "airfoil" in results[key]:
            coords = results[key]["airfoil"].coordinates
            ax.plot(coords[:, 0], coords[:, 1], color=colors[key], linewidth=1.5, label=labels[key])

    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil Shape Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: CD bar chart
    ax = axes[1]
    names = ["Initial"]
    cds = [init_cd]
    bar_colors = ["#999999"]

    for key, label, color in [("baseline", "Baseline", "#1F77B4"), ("piern", "PiERN", "#E45756")]:
        if key in results and "cd" in results[key]:
            names.append(label)
            cds.append(results[key]["cd"])
            bar_colors.append(color)

    bars = ax.bar(range(len(names)), cds, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, cd in zip(bars, cds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{cd:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Weighted CD")
    ax.set_title("Performance Comparison")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ── Gradio UI ──────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PiERN Airfoil Optimizer") as demo:
        gr.Markdown("# PiERN Airfoil Optimizer")
        gr.Markdown(
            "Hierarchical CST optimization with adaptive fidelity routing. "
            "Enter a Chinese prompt and/or upload an airfoil image."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Optimization prompt (Chinese)",
                    placeholder=(
                        "例：设计一个翼型，马赫数0.03，CL目标值[0.8, 1.0, 1.2, 1.4, 1.5, 1.6]，"
                        "权重[5, 6, 7, 8, 9, 10]，力矩系数>=-0.133，后缘角>=6.03°，"
                        "厚度@33%c>=0.128，厚度@90%c>=0.014"
                    ),
                    lines=4,
                )
                image_input = gr.Image(label="Airfoil image (optional)", type="filepath")
                dat_input = gr.File(
                    label="Or upload .dat file (optional)",
                    file_types=[".dat"],
                )
                extract_method = gr.Radio(
                    choices=["auto", "edge", "color", "dat"],
                    value="auto",
                    label="Extraction method",
                )
                router_mode = gr.Radio(
                    choices=["mlp", "threshold", "rule"],
                    value="mlp",
                    label="Router mode",
                )
                run_btn = gr.Button("Optimize", variant="primary")

            with gr.Column(scale=1):
                params_output = gr.JSON(label="Extracted parameters")
                contour_plot = gr.Plot(label="Extracted contour")

        gr.Markdown("## Results")
        with gr.Row():
            result_output = gr.JSON(label="Summary")
        with gr.Row():
            comparison_plot = gr.Plot(label="Shape & Performance Comparison")

        run_btn.click(
            fn=run_optimization,
            inputs=[prompt_input, image_input, dat_input, router_mode, extract_method],
            outputs=[params_output, contour_plot, result_output, comparison_plot],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
