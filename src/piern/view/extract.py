"""Airfoil coordinate extraction from images.

Extracts airfoil contour coordinates from images containing blue airfoil
outlines on white backgrounds. Outputs normalized (x, y) arrays compatible
with KulfanAirfoil fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AirfoilContour:
    """Extracted airfoil contour data."""

    x_surface: np.ndarray  # shape (N,), normalized [0, 1]
    y_upper: np.ndarray  # shape (N,), upper surface y-coords
    y_lower: np.ndarray  # shape (N,), lower surface y-coords

    @property
    def contour_x(self) -> np.ndarray:
        """Flat x array (upper + lower concatenated) for KulfanAirfoil fitting."""
        return np.concatenate([self.x_surface, self.x_surface])

    @property
    def contour_y(self) -> np.ndarray:
        """Flat y array (upper + lower concatenated) for KulfanAirfoil fitting."""
        return np.concatenate([self.y_upper, self.y_lower])


def extract_blue_pixels(
    image: Image.Image,
    blue_threshold: int = 150,
    rgb_max: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract blue contour pixel coordinates from an RGBA/RGB image.

    Returns (xs, ys) arrays of pixel coordinates where blue contour is detected.
    """
    arr = np.array(image)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (b > blue_threshold) & (r < rgb_max) & (g < rgb_max)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No blue contour pixels found in image")
    return xs, ys


def _find_clusters(values: np.ndarray, min_gap: int = 3) -> list[list[int]]:
    """Split sorted values into clusters separated by gaps > min_gap pixels."""
    if len(values) == 0:
        return []
    clusters: list[list[int]] = [[int(values[0])]]
    for v in values[1:]:
        if v - clusters[-1][-1] > min_gap:
            clusters.append([int(v)])
        else:
            clusters[-1].append(int(v))
    return clusters


def _compute_surface_centroids(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-x centroids for upper and lower surfaces.

    For each unique x, finds y-pixel clusters and returns their centroids.
    At trailing/leading edges where surfaces merge (1 cluster), that centroid
    is used for both upper and lower (giving zero thickness).

    Returns (unique_x, y_upper_centroids, y_lower_centroids) in pixel coords.
    """
    unique_x = np.unique(xs)
    y_upper_out = np.empty(len(unique_x))
    y_lower_out = np.empty(len(unique_x))

    for i, x_val in enumerate(unique_x):
        y_vals = np.sort(ys[xs == x_val])
        clusters = _find_clusters(y_vals)

        if len(clusters) >= 2:
            # Two separate surfaces: upper is the one with smaller y (higher in image)
            centers = [np.mean(c) for c in clusters]
            centers.sort()
            y_upper_out[i] = centers[0]  # smaller y = upper surface in image coords
            y_lower_out[i] = centers[-1]  # larger y = lower surface in image coords
        else:
            # Single cluster (TE or LE merge point): use centroid for both
            centroid = np.mean(y_vals)
            y_upper_out[i] = centroid
            y_lower_out[i] = centroid

    return unique_x.astype(float), y_upper_out, y_lower_out


def _normalize_and_resample(
    x_px: np.ndarray,
    y_upper_px: np.ndarray,
    y_lower_px: np.ndarray,
    num_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize pixel coords to airfoil convention and resample uniformly.

    - x mapped to [0, 1]: trailing edge at 0, leading edge at 1
    - y normalized by chord length, y-up positive
    - Linear interpolation to uniform x-grid

    Returns (x_grid, y_upper, y_lower).
    """
    # Trailing edge = leftmost (min x), leading edge = rightmost (max x)
    x_min, x_max = x_px.min(), x_px.max()
    chord = x_max - x_min
    if chord == 0:
        raise ValueError("Zero chord length detected")

    x_norm = (x_px - x_min) / chord

    # Vertical center from the full y-range across both surfaces
    y_all = np.concatenate([y_upper_px, y_lower_px])
    y_center = (y_all.min() + y_all.max()) / 2.0
    # Negate because image y increases downward
    y_upper_norm = -(y_upper_px - y_center) / chord
    y_lower_norm = -(y_lower_px - y_center) / chord

    # Ensure upper surface has y >= lower surface after normalization
    # (upper should be the surface with larger y in airfoil coords)
    swap = y_upper_norm < y_lower_norm
    y_upper_norm[swap], y_lower_norm[swap] = y_lower_norm[swap], y_upper_norm[swap]

    # Sort by x for interpolation
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_upper_sorted = y_upper_norm[sort_idx]
    y_lower_sorted = y_lower_norm[sort_idx]

    # Remove duplicate x values (keep last occurrence)
    _, unique_idx = np.unique(x_sorted, return_index=True)
    # Actually we want the full sorted array for interp; np.interp handles duplicates
    x_grid = np.linspace(0.0, 1.0, num_samples)
    y_upper_grid = np.interp(x_grid, x_sorted, y_upper_sorted)
    y_lower_grid = np.interp(x_grid, x_sorted, y_lower_sorted)

    return x_grid, y_upper_grid, y_lower_grid


def extract_airfoil(
    image_path: str | Path,
    num_samples: int = 200,
    blue_threshold: int = 150,
    rgb_max: int = 100,
) -> AirfoilContour:
    """Extract airfoil coordinates from an image file.

    Args:
        image_path: Path to airfoil image (PNG/JPG with blue contour on white bg).
        num_samples: Number of equally-spaced x points for output.
        blue_threshold: Minimum blue channel value to consider a pixel as contour.
        rgb_max: Maximum red/green channel value for blue contour detection.

    Returns:
        AirfoilContour with normalized coordinates.
    """
    image = Image.open(image_path).convert("RGBA")
    xs, ys = extract_blue_pixels(image, blue_threshold, rgb_max)
    x_px, y_upper_px, y_lower_px = _compute_surface_centroids(xs, ys)
    x, y_upper, y_lower = _normalize_and_resample(x_px, y_upper_px, y_lower_px, num_samples)
    return AirfoilContour(x_surface=x, y_upper=y_upper, y_lower=y_lower)


def save_dat(
    contour: AirfoilContour,
    output_path: str | Path,
) -> None:
    """Save extracted contour to a .dat file in Selig-style format."""
    with open(output_path, "w") as f:
        f.write(f"{len(contour.x_surface) * 2}\n")
        # Upper surface: trailing edge to leading edge
        for xi, yi in zip(contour.x_surface, contour.y_upper):
            f.write(f"{xi:.7f} {yi:.7f}\n")
        # Lower surface: leading edge to trailing edge (reversed)
        for xi, yi in zip(contour.x_surface[::-1], contour.y_lower[::-1]):
            f.write(f"{xi:.7f} {yi:.7f}\n")

def get_coordinates_from_img(image_path: str | Path) -> np.ndarray:
    contour = extract_airfoil(image_path)
    coordinates = np.column_stack([contour.contour_x, contour.contour_y])
    return coordinates

if __name__ == "__main__":
    contour = extract_airfoil("/home/amiya/code/py/Python/BY/piern_airfoil/data/airfoil/naca0012.png")
    upper_surface = np.column_stack([contour.x_surface, contour.y_upper])
    lower_surface = np.column_stack([contour.x_surface, contour.y_lower])
    coordinates = np.vstack([upper_surface, lower_surface[::-1][1:]])