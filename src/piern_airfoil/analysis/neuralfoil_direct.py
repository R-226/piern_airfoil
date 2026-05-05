"""
Direct NeuralFoil inference using pre-trained .npz weights.

This module provides a standalone inference implementation that loads
the pre-trained weights directly without requiring the neuralfoil pip package.

We use the weights from NeuralFoil/neuralfoil/nn_weights_and_biases/ directory.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict
from dataclasses import dataclass

# Default path to NeuralFoil weights (relative to this project)
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "../../../NeuralFoil/neuralfoil/nn_weights_and_biases"


def _swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with numerical stability"""
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))


@dataclass
class NeuralFoilDirectResult:
    """Result from NeuralFoil direct inference."""
    CL: float
    CD: float
    CM: float
    analysis_confidence: float
    Top_Xtr: float
    Bot_Xtr: float
    # Boundary layer data (32 points each)
    upper_bl_theta: Optional[np.ndarray] = None
    upper_bl_H: Optional[np.ndarray] = None
    upper_bl_ue_over_vinf: Optional[np.ndarray] = None
    lower_bl_theta: Optional[np.ndarray] = None
    lower_bl_H: Optional[np.ndarray] = None
    lower_bl_ue_over_vinf: Optional[np.ndarray] = None


class NeuralFoilDirect:
    """
    Direct NeuralFoil inference without pip package dependency.

    Loads pre-trained weights from NeuralFoil's .npz files and performs
    inference using the same network architecture and preprocessing as
    the original NeuralFoil implementation.
    """

    # Number of boundary layer points
    N_BL = 32

    def __init__(self, weights_path: Optional[Union[str, Path]] = None, model_size: str = "xlarge"):
        """
        Initialize NeuralFoil direct inference.

        Args:
            weights_path: Path to NeuralFoil weights directory.
                        Defaults to NeuralFoil/neuralfoil/nn_weights_and_biases/
            model_size: Model size - one of:
                       xxsmall, xsmall, small, medium, large, xlarge, xxlarge, xxxlarge
        """
        self.model_size = model_size

        if weights_path is None:
            # Try default path relative to project root
            default_path = Path(__file__).parent / "../../../NeuralFoil/neuralfoil/nn_weights_and_biases"
            if default_path.exists():
                self.weights_path = default_path
            else:
                # Fallback to absolute path
                self.weights_path = Path("/home/amiya/code/py/Python/BY/NeuralFoil/neuralfoil/nn_weights_and_biases")
        else:
            self.weights_path = Path(weights_path)

        self._nn_params: Optional[Dict] = None
        self._dist_params: Optional[Dict] = None
        self._layer_indices: Optional[list] = None
        self._loaded = False

    def _load(self):
        """Lazy load weights on first use."""
        if self._loaded:
            return

        weights_path = self.weights_path

        # Load network weights
        nn_path = weights_path / f"nn-{self.model_size}.npz"
        if not nn_path.exists():
            raise FileNotFoundError(
                f"NeuralFoil weights not found at {nn_path}. "
                f"Please ensure NeuralFoil is installed or provide correct weights_path."
            )

        self._nn_params = dict(np.load(nn_path))

        # Load scaling parameters
        dist_path = weights_path / "scaled_input_distribution.npz"
        if not dist_path.exists():
            raise FileNotFoundError(f"Scaling distribution not found at {dist_path}")
        self._dist_params = dict(np.load(dist_path))

        # Determine layer indices from weight keys
        self._layer_indices = sorted(set(int(k.split('.')[1]) for k in self._nn_params.keys()))

        self._loaded = True

    def _net_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Scaled input (N_cases, 25)

        Returns:
            Network output (N_cases, 198)
        """
        x = x.T  # (25, N_cases)
        for i in self._layer_indices:
            w = self._nn_params[f"net.{i}.weight"]
            b = self._nn_params[f"net.{i}.bias"]
            x = w @ x + np.reshape(b, (-1, 1))
            if i != self._layer_indices[-1]:
                x = _swish(x)
        return x.T  # (N_cases, 198)

    def _mahalnobis_distance(self, x_scaled: np.ndarray) -> np.ndarray:
        """
        Compute squared Mahalanobis distance from training distribution.

        Args:
            x_scaled: Scaled inputs (N_cases, 25)

        Returns:
            Squared distances (N_cases,)
        """
        mean = self._dist_params["mean_inputs_scaled"]
        inv_cov = self._dist_params["inv_cov_inputs_scaled"]
        diff = x_scaled - mean
        return np.sum(diff @ inv_cov * diff, axis=1)

    def _build_input(
        self,
        kulfan_params: Dict[str, np.ndarray],
        alpha: float,
        Re: float,
        n_crit: float,
        xtr_upper: float,
        xtr_lower: float
    ) -> np.ndarray:
        """
        Build scaled input vector from kulfan parameters.

        Args:
            kulfan_params: Dict with upper_weights, lower_weights, leading_edge_weight, TE_thickness
            alpha: Angle of attack [deg]
            Re: Reynolds number
            n_crit: Critical amplification factor
            xtr_upper: Upper surface transition location
            xtr_lower: Lower surface transition location

        Returns:
            Scaled input vector (25,)
        """
        # Build raw input rows (same as NeuralFoil main.py)
        input_rows = [
            *kulfan_params["upper_weights"],        # 8
            *kulfan_params["lower_weights"],        # 8
            kulfan_params["leading_edge_weight"],  # 1
            kulfan_params["TE_thickness"] * 50,    # 1 (scaled)
            np.sin(2 * np.deg2rad(alpha)),         # sin(2α)
            np.cos(np.deg2rad(alpha)),             # cos(α)
            1 - np.cos(np.deg2rad(alpha))**2,     # 1-cos²(α)
            (np.log(Re) - 12.5) / 3.5,            # log(Re) normalized
            (n_crit - 9) / 4.5,                   # n_crit normalized
            xtr_upper,
            xtr_lower,
        ]

        x = np.stack([np.atleast_1d(r) for r in input_rows], axis=0)  # (25,)

        # Scale input using training distribution
        mean = self._dist_params["mean_inputs_scaled"]
        x_scaled = x - mean

        return x_scaled

    def _flip_inputs(self, x: np.ndarray) -> np.ndarray:
        """
        Flip inputs for alpha-symmetry embedding.

        Args:
            x: Input vector (25,)

        Returns:
            Flipped input vector (25,)
        """
        x_flip = x.copy()
        # Switch upper and lower weights with sign flip
        x_flip[:8] = x[8:16] * -1
        x_flip[8:16] = x[:8] * -1
        # Flip leading edge weight
        x_flip[16] = -x[16]
        # Flip sin(2α)
        x_flip[18] = -x[18]
        # Switch upper/lower transition
        x_flip[23] = x[24]
        x_flip[24] = x[23]
        return x_flip

    def _flip_outputs(self, y_flipped: np.ndarray) -> np.ndarray:
        """
        Unflip outputs from the flipped inference.

        Args:
            y_flipped: Raw output from flipped forward pass (198,)

        Returns:
            Unflipped output (198,)
        """
        y = y_flipped.copy()
        # CL: negate
        y[1] = -y_flipped[1]
        # CM: negate
        y[3] = -y_flipped[3]
        # Switch Top/Bot Xtr
        y[4] = y_flipped[5]
        y[5] = y_flipped[4]
        # Switch upper/lower boundary layer data
        # Upper theta/H: y[6:6+32] <-> y[6+32*3:6+32*4]
        y[6:6+32] = y_flipped[6+32*3:6+32*4]
        y[6+32*3:6+32*4] = y_flipped[6:6+32]
        # Upper ue/vinf: y[6+32*2:6+32*3] <-> y[6+32*5:6+32*6]
        y[6+32*2:6+32*3] = -y_flipped[6+32*5:6+32*6]
        y[6+32*5:6+32*6] = -y_flipped[6+32*2:6+32*3]
        return y

    def inference(
        self,
        kulfan_params: Dict[str, np.ndarray],
        alpha: float,
        Re: float,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        return_bl_data: bool = False
    ) -> NeuralFoilDirectResult:
        """
        Run NeuralFoil inference.

        Args:
            kulfan_params: Dict with upper_weights (8,), lower_weights (8,),
                          leading_edge_weight (float), TE_thickness (float)
            alpha: Angle of attack [deg]
            Re: Reynolds number
            n_crit: Critical amplification factor
            xtr_upper: Upper surface transition location
            xtr_lower: Lower surface transition location
            return_bl_data: Whether to return boundary layer data

        Returns:
            NeuralFoilDirectResult with aerodynamic coefficients
        """
        self._load()

        # Build scaled input
        x = self._build_input(kulfan_params, alpha, Re, n_crit, xtr_upper, xtr_lower)

        # Forward pass
        y = self._net_forward(x[None, :])[0]  # (198,)

        # Apply Mahalanobis confidence adjustment
        mahal_dist = self._mahalnobis_distance(x[None, :])[0]
        confidence_adj = mahal_dist / (2 * 25)
        y[0] = y[0] - confidence_adj

        # Flipped forward pass for alpha-symmetry
        x_flipped = self._flip_inputs(x)
        y_flipped_raw = self._net_forward(x_flipped[None, :])[0]

        # Apply Mahalanobis to flipped
        mahal_dist_flip = self._mahalnobis_distance(x_flipped[None, :])[0]
        y_flipped_raw[0] = y_flipped_raw[0] - mahal_dist_flip / (2 * 25)

        # Unflip outputs
        y_flipped = self._flip_outputs(y_flipped_raw)

        # Fuse (average) original and flipped
        y_fused = (y + y_flipped) / 2

        # Apply output transforms
        confidence = _sigmoid(y_fused[0])
        CL = y_fused[1] / 2
        CD = np.exp((y_fused[2] - 2) * 2)
        CM = y_fused[3] / 20
        Top_Xtr = np.clip(y_fused[4], 0, 1)
        Bot_Xtr = np.clip(y_fused[5], 0, 1)

        # Boundary layer data
        upper_bl_ue_over_vinf = y_fused[6 + self.N_BL * 2 : 6 + self.N_BL * 3]
        lower_bl_ue_over_vinf = y_fused[6 + self.N_BL * 5 : 6 + self.N_BL * 6]

        upper_theta = ((10 ** y_fused[6:6+self.N_BL] - 0.1) /
                       (np.abs(upper_bl_ue_over_vinf) * Re))
        upper_H = 2.6 * np.exp(y_fused[6 + self.N_BL : 6 + self.N_BL * 2])

        lower_theta = ((10 ** y_fused[6 + self.N_BL * 3 : 6 + self.N_BL * 4] - 0.1) /
                       (np.abs(lower_bl_ue_over_vinf) * Re))
        lower_H = 2.6 * np.exp(y_fused[6 + self.N_BL * 4 : 6 + self.N_BL * 5])

        if return_bl_data:
            return NeuralFoilDirectResult(
                CL=float(CL),
                CD=float(CD),
                CM=float(CM),
                analysis_confidence=float(confidence),
                Top_Xtr=float(Top_Xtr),
                Bot_Xtr=float(Bot_Xtr),
                upper_bl_theta=upper_theta,
                upper_bl_H=upper_H,
                upper_bl_ue_over_vinf=upper_bl_ue_over_vinf,
                lower_bl_theta=lower_theta,
                lower_bl_H=lower_H,
                lower_bl_ue_over_vinf=lower_bl_ue_over_vinf,
            )
        else:
            return NeuralFoilDirectResult(
                CL=float(CL),
                CD=float(CD),
                CM=float(CM),
                analysis_confidence=float(confidence),
                Top_Xtr=float(Top_Xtr),
                Bot_Xtr=float(Bot_Xtr),
            )

    def __call__(
        self,
        kulfan_params: Dict[str, np.ndarray],
        alpha: float,
        Re: float,
        n_crit: float = 9.0,
        xtr_upper: float = 1.0,
        xtr_lower: float = 1.0,
        return_bl_data: bool = False
    ) -> NeuralFoilDirectResult:
        """Convenience callable."""
        return self.inference(kulfan_params, alpha, Re, n_crit, xtr_upper, xtr_lower, return_bl_data)


# Convenience function
def neuralfoil_inference(
    upper_weights: np.ndarray,
    lower_weights: np.ndarray,
    leading_edge_weight: float,
    te_thickness: float,
    alpha: float,
    Re: float,
    n_crit: float = 9.0,
    xtr_upper: float = 1.0,
    xtr_lower: float = 1.0,
    model_size: str = "xlarge"
) -> NeuralFoilDirectResult:
    """
    One-shot NeuralFoil inference.

    Args:
        upper_weights: Upper surface CST weights (8,)
        lower_weights: Lower surface CST weights (8,)
        leading_edge_weight: Leading edge weight
        te_thickness: Trailing edge thickness
        alpha: Angle of attack [deg]
        Re: Reynolds number
        n_crit: Critical amplification factor
        xtr_upper: Upper surface transition location
        xtr_lower: Lower surface transition location
        model_size: Model size

    Returns:
        NeuralFoilDirectResult
    """
    kulfan_params = {
        "upper_weights": np.array(upper_weights),
        "lower_weights": np.array(lower_weights),
        "leading_edge_weight": float(leading_edge_weight),
        "TE_thickness": float(te_thickness),
    }
    nf = NeuralFoilDirect(model_size=model_size)
    return nf.inference(kulfan_params, alpha, Re, n_crit, xtr_upper, xtr_lower)
