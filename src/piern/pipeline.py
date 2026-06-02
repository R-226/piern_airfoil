"""
PiERN-Airfoil Pipeline — end-to-end orchestration.

Data flow:
    Chinese prompt + airfoil image
        │
        ├──→ prompt2data (NER extraction) ──→ optimization parameters
        │
        └──→ view.extract (image processing) ──→ KulfanAirfoil
                    │
                    └──→ piern_airfoil.hierarchical ──→ optimized airfoil
                              │
                              └──→ comparison + visualization

Usage:
    from piern.pipeline import PiernPipeline

    pipeline = PiernPipeline()
    result = pipeline.run(
        prompt="设计一个翼型，马赫数0.03，CL目标是0.8到1.6...",
        airfoil_image="data/airfoil/naca0012.png",
    )
    pipeline.visualize(result)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import aerosandbox as asb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


# ── Result types ──────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Structured parameters extracted from natural language prompt."""

    Mach: float
    CL_targets: np.ndarray
    CL_weights: np.ndarray
    CM_min: float = -0.133
    TE_angle_min: float = 6.03
    LE_angle: float = 180.0
    thickness_33_min: float = 0.128
    thickness_90_min: float = 0.014

    @property
    def Re(self) -> np.ndarray:
        return 500e3 * (self.CL_targets / 1.25) ** -0.5


@dataclass
class PipelineResult:
    """Complete pipeline output."""

    prompt: str
    extraction: ExtractionResult
    initial_airfoil: object  # asb.KulfanAirfoil
    optimized_airfoil: object  # asb.KulfanAirfoil
    initial_cd: float
    final_cd: float
    optimization_time: float
    stage_history: list[dict]
    optimization_log: str = ""


# ── Pipeline ───────────────────────────────────────────────────────────


class PiernPipeline:
    """
    End-to-end PiERN airfoil optimization pipeline.

    Wires together:
    1. NER-based parameter extraction from Chinese prompts
    2. Airfoil coordinate extraction from images
    3. Hierarchical CST optimization with adaptive fidelity routing

    Call ``load_models()`` once before ``run()`` to initialize the
    prompt2data encoder model (lazy, since it requires GPU/torch).

    Args:
        router_mode: "rule", "threshold", or "mlp" for OptRouter mode.
    """

    def __init__(self, router_mode: str = "mlp"):
        self._tokenizer = None
        self._extractor_model = None
        self._models_loaded = False
        self._router_mode = router_mode

    # ── Model loading ──────────────────────────────────────────────

    def load_models(self) -> None:
        """Load prompt2data encoder model (lazy, GPU required)."""
        if self._models_loaded:
            return

        from piern.prompt2data.encoder_extractor import (
            CharTokenizer,
            FieldClassifier,
            DEVICE,
            SAVE_DIR,
        )
        import torch

        self._tokenizer = CharTokenizer(max_len=512)
        self._extractor_model = FieldClassifier(
            vocab_size=len(self._tokenizer.char2idx),
            d_model=128,
            nhead=4,
            num_layers=3,
            dim_ff=512,
        )
        checkpoint = torch.load(SAVE_DIR, map_location=DEVICE, weights_only=False)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        self._extractor_model.load_state_dict(model_state, strict=False)
        self._extractor_model.to(DEVICE)
        self._extractor_model.eval()
        self._models_loaded = True

    # ── Step 1: Prompt → Parameters ──────────────────────────────────

    def extract_params(self, prompt: str) -> ExtractionResult:
        """Extract optimization parameters from a Chinese prompt."""
        self.load_models()

        from piern.prompt2data.encoder_extractor import extract

        raw = extract(self._extractor_model, self._tokenizer, prompt)

        # Parse CL targets (6 values)
        cl = raw.get("CL", [0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
        if isinstance(cl, list) and all(isinstance(v, (int, float)) for v in cl):
            cl = np.array(cl, dtype=float)

        # Parse CL weights (6 values)
        weights = raw.get("weights", [5, 6, 7, 8, 9, 10])
        if isinstance(weights, list) and all(
            isinstance(v, (int, float)) for v in weights
        ):
            weights = np.array(weights, dtype=float)

        return ExtractionResult(
            Mach=float(raw.get("Mach", 0.03)),
            CL_targets=cl,
            CL_weights=weights,
            CM_min=float(raw.get("CM_lower_bound", -0.133)),
            TE_angle_min=float(raw.get("Trailing_edge_angle_lower_bound", 6.03)),
            LE_angle=float(raw.get("Leading_edge_angle", 180.0)),
            thickness_33_min=float(raw.get("thickness_head_lower_bound", 0.128)),
            thickness_90_min=float(raw.get("thickness_tail_lower_bound", 0.014)),
        )

    # ── Step 2: Image → KulfanAirfoil ─────────────────────────────────

    def extract_airfoil(self, image_path: str | Path) -> asb.KulfanAirfoil:
        """Extract airfoil shape from an image or .dat file."""
        from piern.view.extract import extract_airfoil

        image_path = Path(image_path)

        if image_path.suffix.lower() == ".dat":
            contour = extract_airfoil(image_path, method="dat")
        else:
            contour = extract_airfoil(image_path)

        coordinate_airfoil = asb.Airfoil(
            name="UserInput",
            coordinates=contour.to_selig_coords(),
        )
        return coordinate_airfoil.to_kulfan_airfoil()

    # ── Step 3: Optimize ─────────────────────────────────────────────

    def optimize(
        self,
        initial_airfoil: asb.KulfanAirfoil,
        params: ExtractionResult,
    ) -> tuple[asb.KulfanAirfoil, float, list[dict]]:
        """
        Run hierarchical CST optimization with router-guided fidelity decisions.

        Returns:
            (optimized_airfoil, elapsed_time, stage_history)
        """
        from piern_airfoil.hierarchical import AdaptiveHierarchicalOptimizer
        from piern.router.opt_router import OptRouter

        # Create router based on mode
        if self._router_mode == "mlp":
            router = OptRouter.from_mlp()
        elif self._router_mode == "threshold":
            router = OptRouter.from_trained()
        else:
            router = OptRouter()

        optimizer = AdaptiveHierarchicalOptimizer(
            CL_targets=params.CL_targets,
            CL_weights=params.CL_weights,
            Re=params.Re,
            mach=params.Mach,
            start_weights=4,
            stability_threshold=0.005,
            router=router,
        )

        t0 = time.perf_counter()
        result = optimizer.optimize(initial_airfoil)
        elapsed = time.perf_counter() - t0

        history = [
            {
                "stage": s.stage,
                "n_active_weights": s.n_active_weights,
                "cd": s.cd,
                "message": s.message,
            }
            for s in result.stages
        ]

        return result.airfoil, elapsed, history

    # ── Main entry point ─────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        airfoil_image: str | Path,
    ) -> PipelineResult:
        """
        Run the full PiERN pipeline.

        Args:
            prompt: Chinese natural-language prompt describing optimization goals.
            airfoil_image: Path to airfoil image (blue contour on white background).

        Returns:
            PipelineResult with initial and optimized airfoils, CD values, history.
        """
        # Step 1: Extract parameters from prompt
        params = self.extract_params(prompt)

        # Step 2: Extract airfoil shape from image
        initial_airfoil = self.extract_airfoil(airfoil_image)

        # Step 3: Evaluate initial CD
        initial_cd = self._quick_eval(initial_airfoil, params)

        # Step 4: Hierarchical optimization
        optimized_airfoil, opt_time, history = self.optimize(initial_airfoil, params)

        # Step 5: Evaluate final CD
        final_cd = self._quick_eval(optimized_airfoil, params)

        return PipelineResult(
            prompt=prompt,
            extraction=params,
            initial_airfoil=initial_airfoil,
            optimized_airfoil=optimized_airfoil,
            initial_cd=initial_cd,
            final_cd=final_cd,
            optimization_time=opt_time,
            stage_history=history,
        )

    # ── Evaluation helper ────────────────────────────────────────────

    def _quick_eval(self, airfoil, params: ExtractionResult) -> float:
        """Quick weighted CD evaluation using NeuralFoil."""
        from piern_airfoil.eval import evaluate_weighted_cd

        return evaluate_weighted_cd(
            airfoil, params.CL_targets, params.Re, params.CL_weights, mach=params.Mach,
        )

    # ── Visualization ─────────────────────────────────────────────────

    def visualize(
        self,
        result: PipelineResult,
        save_path: str | Path = "pipeline_output.png",
    ) -> None:
        """Generate comparison visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: Airfoil shape comparison
        ax = axes[0]
        for af, name, color, ls in [
            (result.initial_airfoil, "Initial", "gray", "--"),
            (result.optimized_airfoil, "Optimized", "red", "-"),
        ]:
            coords = af.coordinates
            ax.plot(coords[:, 0], coords[:, 1], color=color, linestyle=ls,
                    linewidth=2, label=name)
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title("Airfoil Shape")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Middle: CD convergence
        ax = axes[1]
        stages = [h["stage"] for h in result.stage_history]
        cds = [h["cd"] for h in result.stage_history]
        ax.plot(stages, cds, "bo-", linewidth=2, markersize=8)
        for s, c in zip(stages, cds):
            ax.annotate(f"{c:.4f}", (s, c), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
        ax.axhline(y=result.initial_cd, color="gray", linestyle="--",
                   label=f"Initial CD={result.initial_cd:.4f}")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Weighted CD")
        ax.set_title("CD Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Info panel
        ax = axes[2]
        ax.axis("off")
        lines = [
            f"Pipeline Result",
            f"",
            f"Initial CD:   {result.initial_cd:.6f}",
            f"Final CD:     {result.final_cd:.6f}",
            f"Improvement:  {(result.initial_cd - result.final_cd) / result.initial_cd * 100:+.2f}%",
            f"Time:         {result.optimization_time:.2f}s",
            f"",
            f"Parameters:",
            f"  Mach:  {result.extraction.Mach}",
            f"  CL:    {result.extraction.CL_targets}",
            f"  W:     {result.extraction.CL_weights}",
            f"",
            f"Stages:",
        ]
        for h in result.stage_history:
            lines.append(
                f"  {h['stage']}: {h['n_active_weights']}w → "
                f"CD={h['cd']:.4f}"
            )
        for i, line in enumerate(lines):
            ax.text(0.05, 0.95 - i * 0.04, line, transform=ax.transAxes,
                    fontfamily="monospace", fontsize=10,
                    verticalalignment="top")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {save_path}")


# ── Convenience function ──────────────────────────────────────────────


def run_pipeline(prompt: str, airfoil_image: str | Path) -> PipelineResult:
    """Run the full pipeline and return results."""
    pipeline = PiernPipeline()
    return pipeline.run(prompt, airfoil_image)
