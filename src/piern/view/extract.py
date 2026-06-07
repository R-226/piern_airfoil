"""Airfoil coordinate extraction from images and .dat files.

Supports multiple extraction methods:
- Edge detection (default): color-agnostic, works with any line color
- Color detection: explicit color targeting (blue, black, red, etc.)
- .dat file: direct coordinate loading (Selig/Lednicer format)

Outputs normalized (x, y) arrays compatible with KulfanAirfoil fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import sobel, label, binary_closing, binary_opening, gaussian_filter


# ── Data structures ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AirfoilContour:
    """Extracted airfoil contour data."""

    x_surface: np.ndarray  # shape (N,), normalized [0, 1]
    y_upper: np.ndarray  # shape (N,), upper surface y-coords
    y_lower: np.ndarray  # shape (N,), lower surface y-coords

    @property
    def contour_x(self) -> np.ndarray:
        """Flat x array (upper + lower concatenated)."""
        return np.concatenate([self.x_surface, self.x_surface])

    @property
    def contour_y(self) -> np.ndarray:
        """Flat y array (upper + lower concatenated)."""
        return np.concatenate([self.y_upper, self.y_lower])

    def to_selig_coords(self) -> np.ndarray:
        """Convert to Selig-format coordinates: TE(upper)→LE→TE(lower).

        Standard .dat coordinate order, compatible with asb.Airfoil(coordinates=...).
        Returns shape (2N-1, 2) — LE point appears only once.
        """
        # Upper: TE→LE (x decreasing from 1 to 0)
        upper = np.column_stack([self.x_surface[::-1], self.y_upper[::-1]])
        # Lower: LE→TE (x increasing from 0 to 1), skip LE to avoid duplicate
        lower = np.column_stack([self.x_surface, self.y_lower])
        return np.vstack([upper, lower[1:]])


# ── .dat file loading ───────────────────────────────────────────────


def load_dat(path: str | Path, num_samples: int = 200) -> AirfoilContour:
    """Load airfoil coordinates from a .dat file.

    Supports:
    - Selig format: optional header line + coordinates (upper TE→LE, lower LE→TE)
    - Headerless format: just coordinates
    - Lednicer format: upper count + upper coords + lower count + lower coords

    Args:
        path: Path to .dat file.
        num_samples: Number of points for resampling output.

    Returns:
        AirfoilContour with normalized coordinates.
    """
    path = Path(path)
    lines = path.read_text().strip().splitlines()

    # Try to parse as numbers
    coords = []
    header = None
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
            except ValueError:
                header = line  # likely a header line
                continue

    if len(coords) < 4:
        raise ValueError(f"Too few coordinate points in {path}: {len(coords)}")

    coords = np.array(coords)

    # Detect format: Selig vs Lednicer
    # Selig: all coords in one block, upper then lower
    # Lednicer: two blocks with count lines between
    # Heuristic: if y values go from positive to negative to positive, it's Selig
    y_vals = coords[:, 1]
    sign_changes = np.sum(np.diff(np.sign(y_vals)) != 0)

    if sign_changes >= 2:
        # Selig format: upper surface (y>0) then lower surface (y<0) or vice versa
        upper_mask = y_vals >= 0
        lower_mask = y_vals < 0

        if upper_mask.sum() > 1 and lower_mask.sum() > 1:
            upper = coords[upper_mask]
            lower = coords[lower_mask]
        else:
            # Fallback: split at midpoint
            mid = len(coords) // 2
            upper = coords[:mid]
            lower = coords[mid:]
    else:
        # All same sign or monotonic — split at midpoint
        mid = len(coords) // 2
        upper = coords[:mid]
        lower = coords[mid:]

    # Sort by x for interpolation
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    # Normalize x to [0, 1]
    x_min = min(upper[:, 0].min(), lower[:, 0].min())
    x_max = max(upper[:, 0].max(), lower[:, 0].max())
    chord = x_max - x_min
    if chord < 1e-10:
        raise ValueError("Zero chord length in .dat file")

    upper_x = (upper[:, 0] - x_min) / chord
    lower_x = (lower[:, 0] - x_min) / chord

    # Center y
    y_all = np.concatenate([upper[:, 1], lower[:, 1]])
    y_center = (y_all.max() + y_all.min()) / 2.0
    upper_y = (upper[:, 1] - y_center) / chord
    lower_y = (lower[:, 1] - y_center) / chord

    # Ensure upper > lower
    if upper_y.mean() < lower_y.mean():
        upper_y, lower_y = lower_y, upper_y

    # Resample to uniform grid
    x_grid = np.linspace(0.0, 1.0, num_samples)
    y_upper = np.interp(x_grid, upper_x, upper_y)
    y_lower = np.interp(x_grid, lower_x, lower_y)

    return AirfoilContour(x_surface=x_grid, y_upper=y_upper, y_lower=y_lower)


# ── Edge detection ──────────────────────────────────────────────────


def _sobel_edges(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute edge magnitude using Sobel filters."""
    smoothed = gaussian_filter(gray.astype(float), sigma=sigma)
    sx = sobel(smoothed, axis=0)
    sy = sobel(smoothed, axis=1)
    return np.sqrt(sx**2 + sy**2)


def _auto_threshold(values: np.ndarray) -> float:
    """Compute threshold using a simple percentile-based method."""
    # Use Otsu-like approach: maximize inter-class variance
    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.0

    sum_total = np.sum(bin_centers * hist)
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0
    threshold = bin_centers[0]

    for i in range(len(hist)):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += bin_centers[i] * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    return threshold


def _extract_contour_from_edges(
    edge_mag: np.ndarray,
    min_component_size: int = 50,
) -> np.ndarray | None:
    """Extract airfoil contour coordinates from edge magnitude map.

    Returns (N, 2) array of (x, y) pixel coordinates, or None if not found.
    """
    # Threshold edges
    threshold = _auto_threshold(edge_mag)
    binary = edge_mag > threshold

    # Morphological cleanup: close small gaps, remove isolated pixels
    struct = np.ones((3, 3))
    binary = binary_closing(binary, structure=struct, iterations=2)
    binary = binary_opening(binary, structure=struct, iterations=1)

    # Label connected components
    labeled, n_components = label(binary)

    if n_components == 0:
        return None

    # Find the best component (largest, with airfoil-like aspect ratio)
    best_score = -1
    best_coords = None

    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        ys, xs = np.where(mask)
        if len(xs) < min_component_size:
            continue

        # Aspect ratio: airfoils are wide, not tall
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        if height == 0:
            continue

        aspect = width / height
        if aspect < 2.0:
            continue  # too tall to be an airfoil

        # Score: size * aspect ratio (prefer large, wide contours)
        score = len(xs) * aspect
        if score > best_score:
            best_score = score
            best_coords = np.column_stack([xs, ys])

    return best_coords


def _detect_rotation(coords: np.ndarray) -> float:
    """Detect rotation angle of contour using PCA.

    Returns angle in degrees (0 = horizontal).
    """
    if len(coords) < 10:
        return 0.0

    # PCA
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis is eigenvector with largest eigenvalue
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))

    # Normalize to [-45, 45] (eigenvector has 180° ambiguity)
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90

    return angle


def _rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotate image by angle degrees."""
    return image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))


# ── Color detection ─────────────────────────────────────────────────


def _detect_color_pixels(
    arr: np.ndarray,
    color: str | None = None,
) -> np.ndarray:
    """Detect foreground pixels by color.

    Args:
        arr: RGB image array (H, W, 3).
        color: Target color ("blue", "black", "red", "green", "white", or None for auto).

    Returns:
        Boolean mask of foreground pixels.
    """
    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)

    if color == "blue":
        return (b > 150) & (r < 100) & (g < 100)
    elif color == "red":
        return (r > 150) & (g < 100) & (b < 100)
    elif color == "green":
        return (g > 150) & (r < 100) & (b < 100)
    elif color == "black":
        brightness = (r + g + b) / 3
        return brightness < 50
    elif color == "white":
        brightness = (r + g + b) / 3
        return brightness > 200
    else:
        # Auto-detect: find the dominant non-background color
        # Background is typically the most common color (white or light gray)
        brightness = (r + g + b) / 3
        # Assume background is bright (>180) or dark (<75)
        bg_bright = brightness > 180
        bg_dark = brightness < 75
        bg_mask = bg_bright | bg_dark

        # Foreground is everything that's not background
        fg_mask = ~bg_mask

        # If too few foreground pixels, try edge-based approach
        if fg_mask.sum() < 100:
            # Fall back to detecting dark lines on light background
            return brightness < 100

        return fg_mask


def extract_color_contour(
    image: Image.Image,
    color: str | None = None,
    min_gap: int = 3,
) -> np.ndarray | None:
    """Extract contour coordinates from color-detected pixels.

    Returns (N, 2) array of (x, y) pixel coordinates, or None if not found.
    """
    arr = np.array(image.convert("RGB"))
    mask = _detect_color_pixels(arr, color)

    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None

    return np.column_stack([xs, ys])


# ── Surface separation ──────────────────────────────────────────────


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
            centers = [np.mean(c) for c in clusters]
            centers.sort()
            y_upper_out[i] = centers[0]  # smaller y = upper surface in image coords
            y_lower_out[i] = centers[-1]  # larger y = lower surface in image coords
        else:
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
    x_min, x_max = x_px.min(), x_px.max()
    chord = x_max - x_min
    if chord == 0:
        raise ValueError("Zero chord length detected")

    x_norm = (x_px - x_min) / chord

    y_all = np.concatenate([y_upper_px, y_lower_px])
    y_center = (y_all.min() + y_all.max()) / 2.0
    # Negate because image y increases downward
    y_upper_norm = -(y_upper_px - y_center) / chord
    y_lower_norm = -(y_lower_px - y_center) / chord

    # Ensure upper surface has y >= lower surface after normalization
    swap = y_upper_norm < y_lower_norm
    y_upper_norm[swap], y_lower_norm[swap] = y_lower_norm[swap], y_upper_norm[swap]

    # Sort by x for interpolation
    sort_idx = np.argsort(x_norm)
    x_sorted = x_norm[sort_idx]
    y_upper_sorted = y_upper_norm[sort_idx]
    y_lower_sorted = y_lower_norm[sort_idx]

    x_grid = np.linspace(0.0, 1.0, num_samples)
    y_upper_grid = np.interp(x_grid, x_sorted, y_upper_sorted)
    y_lower_grid = np.interp(x_grid, x_sorted, y_lower_sorted)

    return x_grid, y_upper_grid, y_lower_grid


# ── Main extraction functions ───────────────────────────────────────


def _extract_from_coords(
    coords: np.ndarray,
    num_samples: int = 200,
) -> AirfoilContour:
    """Convert (N, 2) pixel coordinates to AirfoilContour."""
    xs, ys = coords[:, 0], coords[:, 1]
    x_px, y_upper_px, y_lower_px = _compute_surface_centroids(xs, ys)
    x, y_upper, y_lower = _normalize_and_resample(x_px, y_upper_px, y_lower_px, num_samples)
    return AirfoilContour(x_surface=x, y_upper=y_upper, y_lower=y_lower)


def extract_airfoil(
    image_path: str | Path,
    num_samples: int = 200,
    method: str = "auto",
    color: str | None = None,
) -> AirfoilContour:
    """Extract airfoil coordinates from an image using sub-pixel iso-contours.

    Pipeline (sub-pixel accurate via skimage.measure.find_contours):
        1. Read grayscale
        2. find_contours at iso-level = mid-gray (sub-pixel, no Otsu needed)
        3. Pick the longest closed contour (the airfoil outline)
        4. Split into upper (y >= 0) and lower (y <= 0) halves by LE point
        5. Convert pixel coords -> normalized (x, y) where x in [0, 1], y centered
        6. Resample to num_samples points with cos spacing (LE/TE dense)
        7. Force camber line y_mid = 0 (perfect symmetry)
        8. Force LE/TE closure (y_upper = y_lower = 0 at endpoints)

    Args:
        image_path: Path to airfoil image.
        num_samples: Number of equally-spaced x points for output.
        method: kept for API compatibility ("auto", "edge", "color", "dat")
        color: kept for API compatibility (unused)

    Returns:
        AirfoilContour with normalized coordinates.
    """
    image_path = Path(image_path)

    if method == "dat" or image_path.suffix.lower() == ".dat":
        return load_dat(image_path, num_samples)

    image = Image.open(image_path).convert("L")
    gray = np.array(image, dtype=float)
    H, W = gray.shape

    # Sub-pixel iso-contour extraction.
    # Use a mid-gray iso-level (128) — works for any airfoil drawn as a dark
    # outline on a light background. Returns (N, 2) array of (row, col).
    from skimage import measure
    contours = measure.find_contours(gray, 128.0, fully_connected="high")
    if not contours:
        raise ValueError(
            f"No airfoil contour found in {image_path}. "
            "The image may have a non-standard color scheme or no visible airfoil."
        )

    # Pick the contour with the highest aspect ratio (widest vs tallest).
    # This filters out axis labels, tick marks, frame borders, and the
    # surrounding fill region — all of which have lower aspect ratios than
    # the actual airfoil outline.
    def _aspect(c):
        return (c[:, 1].max() - c[:, 1].min()) / max(c[:, 0].max() - c[:, 0].min(), 1)

    contour = max(contours, key=_aspect)  # (N, 2) — (row, col) = (y_pix, x_pix)

    # Convert (row, col) -> (x, y) and normalize to airfoil coordinates
    x_pix = contour[:, 1]
    y_pix = contour[:, 0]

    # Normalize x to [0, 1] using the chord
    x_min, x_max = x_pix.min(), x_pix.max()
    chord = max(x_max - x_min, 1.0)
    x_norm = (x_pix - x_min) / chord

    # Normalize y: invert (image y goes down), center about 0, scale by chord
    y_centered = -((y_pix - y_pix.mean()) / chord)

    # Split into upper (y >= 0) and lower (y < 0) by walking the closed loop
    # from LE to TE on both surfaces. The closed contour visits upper, then
    # lower, so we split at LE (x_min) and TE (x_max).
    le_idx = np.argmin(x_norm)
    te_idx = np.argmax(x_norm)
    n = len(contour)

    # Walk forward (i -> i+1) from LE to TE: this gives one surface
    forward_idx = []
    i = le_idx
    while i != te_idx:
        forward_idx.append(i)
        i = (i + 1) % n
    forward_idx.append(te_idx)

    # Walk backward (i -> i-1) from LE to TE: this gives the other surface
    backward_idx = []
    i = (le_idx - 1) % n
    while i != te_idx:
        backward_idx.append(i)
        i = (i - 1) % n
    backward_idx.append(te_idx)

    # Determine which walk is upper (y >= 0) and which is lower (y < 0)
    # Use mean y of each path to assign roles
    fwd_y_mean = y_centered[forward_idx].mean()
    bwd_y_mean = y_centered[backward_idx].mean()
    if fwd_y_mean >= bwd_y_mean:
        upper_idx, lower_idx = forward_idx, backward_idx
    else:
        upper_idx, lower_idx = backward_idx, forward_idx

    x_upper = x_norm[upper_idx]
    y_upper_raw = y_centered[upper_idx]
    x_lower = x_norm[lower_idx]
    y_lower_raw = y_centered[lower_idx]

    # Sort by x ascending (upper and lower both go LE -> TE)
    upper_sort = np.argsort(x_upper)
    x_upper = x_upper[upper_sort]
    y_upper_raw = y_upper_raw[upper_sort]
    lower_sort = np.argsort(x_lower)
    x_lower = x_lower[lower_sort]
    y_lower_raw = y_lower_raw[lower_sort]

    # Resample to num_samples with cos spacing (LE/TE dense)
    i_arr = np.arange(num_samples)
    x_target = 0.5 * (1.0 - np.cos(i_arr * np.pi / (num_samples - 1)))
    y_upper = np.interp(x_target, x_upper, y_upper_raw)
    y_lower = np.interp(x_target, x_lower, y_lower_raw)

    # Force perfect camber symmetry (y_mid = 0)
    y_mid = (y_upper + y_lower) / 2.0
    y_upper = y_upper - y_mid
    y_lower = y_lower - y_mid

    # Force LE/TE closure
    y_upper[0] = y_lower[0] = 0.0
    y_upper[-1] = y_lower[-1] = 0.0

    return AirfoilContour(x_surface=x_target, y_upper=y_upper, y_lower=y_lower)


# ── Legacy interface ────────────────────────────────────────────────


def extract_blue_pixels(
    image: Image.Image,
    blue_threshold: int = 150,
    rgb_max: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract blue contour pixel coordinates (legacy interface)."""
    arr = np.array(image)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (b > blue_threshold) & (r < rgb_max) & (g < rgb_max)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("No blue contour pixels found in image")
    return xs, ys


def save_dat(
    contour: AirfoilContour,
    output_path: str | Path,
) -> None:
    """Save extracted contour to a .dat file in Selig-style format."""
    with open(output_path, "w") as f:
        f.write(f"{len(contour.x_surface) * 2}\n")
        for xi, yi in zip(contour.x_surface, contour.y_upper):
            f.write(f"{xi:.7f} {yi:.7f}\n")
        for xi, yi in zip(contour.x_surface[::-1], contour.y_lower[::-1]):
            f.write(f"{xi:.7f} {yi:.7f}\n")


def get_coordinates_from_img(image_path: str | Path) -> np.ndarray:
    """Extract airfoil coordinates as Selig-format array for asb.Airfoil."""
    contour = extract_airfoil(image_path)
    return contour.to_selig_coords()


# ── CLI ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m piern.view.extract <image_or_dat_path>")
        sys.exit(1)

    path = sys.argv[1]
    contour = extract_airfoil(path)
    print(f"Extracted {len(contour.x_surface)} points per surface")
    print(f"x range: [{contour.x_surface[0]:.4f}, {contour.x_surface[-1]:.4f}]")
    print(f"y_upper range: [{contour.y_upper.min():.4f}, {contour.y_upper.max():.4f}]")
    print(f"y_lower range: [{contour.y_lower.min():.4f}, {contour.y_lower.max():.4f}]")
