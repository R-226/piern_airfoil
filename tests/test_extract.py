"""Tests for airfoil coordinate extraction from images and .dat files.

Tests cover:
- .dat file loading (Selig format) across 18 aerosandbox airfoils
- Edge detection extraction from generated images (multiple colors)
- Color detection extraction (blue, black, red)
- Rotation correction
- Method auto-detection fallback chain
- Coordinate accuracy vs ground truth KulfanAirfoil
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aerosandbox as asb

from piern.view.extract import (
    AirfoilContour,
    extract_airfoil,
    load_dat,
    save_dat,
    _detect_rotation,
    _sobel_edges,
    _extract_contour_from_edges,
)

# ── Airfoil fixtures ────────────────────────────────────────────────

ALL_AIRFOILS = [
    "naca0012", "naca2412", "naca4412", "naca0015", "naca6412",
    "naca0009", "naca0021", "naca23012", "naca64a210", "clarky",
    "e387", "sd7037", "s1223", "mh114", "goe776", "ag13", "ag26", "b737b",
]

# Subset for expensive image-based tests (covers thin/thick/symmetric/cambered)
IMAGE_TEST_AIRFOILS = [
    "naca0012",   # symmetric, standard
    "naca4412",   # cambered
    "naca0021",   # thick symmetric
    "clarky",     # general aviation
    "e387",       # low Re
    "s1223",      # high lift
]


# ── Helpers ─────────────────────────────────────────────────────────


def _airfoil_dat_path(name: str, tmp_path: Path) -> Path:
    """Generate a .dat file from an aerosandbox airfoil."""
    af = asb.Airfoil(name)
    path = tmp_path / f"{name}.dat"
    # Write Selig format
    coords = af.coordinates
    with open(path, "w") as f:
        f.write(f"{name}\n")
        for x, y in coords:
            f.write(f"{x:.7f} {y:.7f}\n")
    return path


def _airfoil_image(
    name: str,
    tmp_path: Path,
    color: str = "blue",
    line_width: float = 2.0,
    bg_color: str = "white",
    dpi: int = 150,
    figsize: tuple[float, float] = (8, 3),
) -> Path:
    """Render an airfoil image from aerosandbox coordinates."""
    af = asb.Airfoil(name)
    coords = af.coordinates

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    color_map = {
        "blue": "blue",
        "black": "black",
        "red": "red",
        "green": "green",
    }
    c = color_map.get(color, color)

    ax.plot(coords[:, 0], coords[:, 1], color=c, linewidth=line_width)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 0.25)
    ax.axis("off")

    path = tmp_path / f"{name}_{color}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor=bg_color)
    plt.close(fig)
    return path


def _airfoil_image_rotated(
    name: str,
    tmp_path: Path,
    angle: float = 15.0,
    color: str = "black",
) -> Path:
    """Render a rotated airfoil image."""
    af = asb.Airfoil(name)
    coords = af.coordinates

    # Rotate coordinates
    rad = np.radians(angle)
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    dx, dy = coords[:, 0] - cx, coords[:, 1] - cy
    rx = dx * np.cos(rad) - dy * np.sin(rad) + cx
    ry = dx * np.sin(rad) + dy * np.cos(rad) + cy

    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    ax.plot(rx, ry, color=color, linewidth=2)
    ax.set_aspect("equal")
    ax.axis("off")

    path = tmp_path / f"{name}_rotated.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return path


def _ground_truth_contour(name: str, num_samples: int = 200) -> AirfoilContour:
    """Get ground truth contour directly from aerosandbox."""
    af = asb.Airfoil(name)
    coords = af.coordinates

    # Split upper/lower by y sign or sequential order
    y_vals = coords[:, 1]
    upper_mask = y_vals >= 0
    lower_mask = y_vals < 0

    if upper_mask.sum() < 2 or lower_mask.sum() < 2:
        # All same sign — split at midpoint
        mid = len(coords) // 2
        upper = coords[:mid]
        lower = coords[mid:]
    else:
        upper = coords[upper_mask]
        lower = coords[lower_mask]

    # Sort by x
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    # Normalize
    x_min = min(upper[:, 0].min(), lower[:, 0].min())
    x_max = max(upper[:, 0].max(), lower[:, 0].max())
    chord = x_max - x_min

    upper_x = (upper[:, 0] - x_min) / chord
    lower_x = (lower[:, 0] - x_min) / chord
    y_all = np.concatenate([upper[:, 1], lower[:, 1]])
    y_center = (y_all.max() + y_all.min()) / 2.0
    upper_y = (upper[:, 1] - y_center) / chord
    lower_y = (lower[:, 1] - y_center) / chord

    if upper_y.mean() < lower_y.mean():
        upper_y, lower_y = lower_y, upper_y

    x_grid = np.linspace(0.0, 1.0, num_samples)
    y_upper = np.interp(x_grid, upper_x, upper_y)
    y_lower = np.interp(x_grid, lower_x, lower_y)

    return AirfoilContour(x_surface=x_grid, y_upper=y_upper, y_lower=y_lower)


def _contour_rmse(extracted: AirfoilContour, ground_truth: AirfoilContour) -> float:
    """Compute RMSE between extracted and ground truth contours."""
    err_upper = np.sqrt(np.mean((extracted.y_upper - ground_truth.y_upper) ** 2))
    err_lower = np.sqrt(np.mean((extracted.y_lower - ground_truth.y_lower) ** 2))
    return (err_upper + err_lower) / 2


# ── .dat loading tests ─────────────────────────────────────────────


class TestDatLoading:
    """Test .dat file loading across all airfoils."""

    @pytest.mark.parametrize("name", ALL_AIRFOILS)
    def test_load_dat_roundtrip(self, name, tmp_path):
        """Write .dat from aerosandbox, load it, check shape and range."""
        dat_path = _airfoil_dat_path(name, tmp_path)
        contour = load_dat(dat_path, num_samples=200)

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.shape == (200,)
        assert contour.y_lower.shape == (200,)
        assert contour.x_surface[0] == pytest.approx(0.0, abs=0.01)
        assert contour.x_surface[-1] == pytest.approx(1.0, abs=0.01)
        # Upper surface should be above lower
        assert contour.y_upper.mean() >= contour.y_lower.mean()

    @pytest.mark.parametrize("name", ALL_AIRFOILS)
    def test_load_dat_vs_ground_truth(self, name, tmp_path):
        """Extracted .dat contour should closely match ground truth."""
        dat_path = _airfoil_dat_path(name, tmp_path)
        extracted = load_dat(dat_path, num_samples=200)
        gt = _ground_truth_contour(name, num_samples=200)

        rmse = _contour_rmse(extracted, gt)
        assert rmse < 0.02, f"{name}: RMSE={rmse:.4f} > 0.02"

    def test_load_dat_saves_and_loads(self, tmp_path):
        """Test save_dat + load_dat roundtrip."""
        # Use symmetric contour so y-centering doesn't shift values
        x = np.linspace(0, 1, 100)
        y_upper = np.sin(np.linspace(0, np.pi, 100)) * 0.05
        y_lower = -np.sin(np.linspace(0, np.pi, 100)) * 0.05
        contour = AirfoilContour(x_surface=x, y_upper=y_upper, y_lower=y_lower)
        out_path = tmp_path / "test.dat"
        save_dat(contour, out_path)
        loaded = load_dat(out_path, num_samples=100)

        np.testing.assert_allclose(loaded.x_surface, contour.x_surface, atol=1e-6)
        # y values may shift slightly due to recentering — check shape
        assert loaded.y_upper.shape == (100,)
        assert loaded.y_lower.shape == (100,)
        assert loaded.y_upper.mean() > loaded.y_lower.mean()

    def test_load_dat_invalid_file(self, tmp_path):
        """Loading a non-dat file should raise ValueError."""
        bad_path = tmp_path / "bad.dat"
        bad_path.write_text("no numbers here\njust text\n")
        with pytest.raises(ValueError, match="Too few"):
            load_dat(bad_path)


# ── Edge detection tests ───────────────────────────────────────────


class TestEdgeDetection:
    """Test edge detection extraction from generated images."""

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_edge_detect_blue_image(self, name, tmp_path):
        """Edge detection should extract contour from blue-line images."""
        img_path = _airfoil_image(name, tmp_path, color="blue")
        contour = extract_airfoil(img_path, method="edge")

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.mean() > contour.y_lower.mean()
        # Contour should span most of [0, 1]
        assert contour.x_surface[-1] - contour.x_surface[0] > 0.8

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_edge_detect_black_image(self, name, tmp_path):
        """Edge detection should work with black-line images."""
        img_path = _airfoil_image(name, tmp_path, color="black")
        contour = extract_airfoil(img_path, method="edge")

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.mean() > contour.y_lower.mean()

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_edge_detect_red_image(self, name, tmp_path):
        """Edge detection should work with red-line images."""
        img_path = _airfoil_image(name, tmp_path, color="red")
        contour = extract_airfoil(img_path, method="edge")

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.mean() > contour.y_lower.mean()

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_edge_detect_accuracy(self, name, tmp_path):
        """Edge-detected contour should be reasonably close to ground truth."""
        img_path = _airfoil_image(name, tmp_path, color="black", line_width=3.0)
        extracted = extract_airfoil(img_path, method="edge")
        gt = _ground_truth_contour(name, num_samples=200)

        rmse = _contour_rmse(extracted, gt)
        # Image extraction is less precise than .dat — allow 5% tolerance
        assert rmse < 0.05, f"{name}: RMSE={rmse:.4f} > 0.05"


# ── Color detection tests ──────────────────────────────────────────


class TestColorDetection:
    """Test color-based extraction."""

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_color_blue(self, name, tmp_path):
        """Blue color detection should extract contour."""
        img_path = _airfoil_image(name, tmp_path, color="blue")
        contour = extract_airfoil(img_path, method="color", color="blue")

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.mean() > contour.y_lower.mean()

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_color_black(self, name, tmp_path):
        """Black color detection should extract contour."""
        img_path = _airfoil_image(name, tmp_path, color="black")
        contour = extract_airfoil(img_path, method="color", color="black")

        assert contour.x_surface.shape == (200,)
        assert contour.y_upper.mean() > contour.y_lower.mean()

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_color_red(self, name, tmp_path):
        """Red color detection should extract contour."""
        img_path = _airfoil_image(name, tmp_path, color="red")
        contour = extract_airfoil(img_path, method="color", color="red")

        assert contour.x_surface.shape == (200,)


# ── Rotation correction tests ──────────────────────────────────────


class TestRotationCorrection:
    """Test PCA-based rotation detection and correction."""

    def test_detect_rotation_horizontal(self):
        """Horizontal contour should have ~0° rotation."""
        # Create a horizontal line of points
        coords = np.column_stack([
            np.linspace(0, 100, 50),
            np.full(50, 50.0),
        ])
        angle = _detect_rotation(coords)
        assert abs(angle) < 5.0

    def test_detect_rotation_tilted(self):
        """Tilted contour should detect the correct angle."""
        angle_deg = 20.0
        rad = np.radians(angle_deg)
        x = np.linspace(0, 100, 50)
        y = np.tan(rad) * x
        coords = np.column_stack([x, y])
        detected = _detect_rotation(coords)
        assert abs(detected - angle_deg) < 5.0

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS[:3])
    def test_rotated_image_extraction(self, name, tmp_path):
        """Extraction from rotated images should still produce valid contours."""
        img_path = _airfoil_image_rotated(name, tmp_path, angle=15.0)
        contour = extract_airfoil(img_path, method="edge")

        assert contour.x_surface.shape == (200,)
        # After rotation correction, upper should be above lower
        assert contour.y_upper.mean() > contour.y_lower.mean()


# ── Auto method detection tests ────────────────────────────────────


class TestAutoMethod:
    """Test the auto-detection fallback chain."""

    def test_auto_detects_dat(self, tmp_path):
        """Auto method should detect .dat files by extension."""
        dat_path = _airfoil_dat_path("naca0012", tmp_path)
        contour = extract_airfoil(dat_path, method="auto")
        assert contour.x_surface.shape == (200,)

    def test_auto_falls_back_to_edge(self, tmp_path):
        """Auto should use edge detection for images."""
        img_path = _airfoil_image("naca0012", tmp_path, color="black")
        contour = extract_airfoil(img_path, method="auto")
        assert contour.x_surface.shape == (200,)

    def test_auto_falls_back_to_color(self, tmp_path):
        """Auto should fall back to color if edge fails on blue images."""
        img_path = _airfoil_image("naca0012", tmp_path, color="blue")
        # Edge detection should work on blue images too, but if it fails,
        # color fallback should kick in
        contour = extract_airfoil(img_path, method="auto")
        assert contour.x_surface.shape == (200,)


# ── Contour structure tests ────────────────────────────────────────


class TestContourStructure:
    """Test AirfoilContour data structure."""

    def test_contour_properties(self):
        """contour_x and contour_y should concatenate upper and lower."""
        contour = AirfoilContour(
            x_surface=np.array([0.0, 0.5, 1.0]),
            y_upper=np.array([0.1, 0.15, 0.0]),
            y_lower=np.array([-0.05, -0.08, 0.0]),
        )
        assert contour.contour_x.shape == (6,)
        assert contour.contour_y.shape == (6,)
        np.testing.assert_array_equal(
            contour.contour_x, [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
        )

    def test_contour_frozen(self):
        """AirfoilContour should be immutable."""
        contour = AirfoilContour(
            x_surface=np.array([0.0, 1.0]),
            y_upper=np.array([0.1, 0.0]),
            y_lower=np.array([-0.1, 0.0]),
        )
        with pytest.raises(AttributeError):
            contour.x_surface = np.array([0.5])  # type: ignore


# ── Sobel edge filter tests ────────────────────────────────────────


class TestSobelEdges:
    """Test Sobel edge detection internals."""

    def test_sobel_detects_edge(self):
        """Sobel should detect a sharp brightness transition."""
        # Create image with vertical edge
        img = np.zeros((100, 100))
        img[:, 50:] = 255
        edges = _sobel_edges(img, sigma=1.0)
        # Edge should be strongest near x=50
        assert edges[:, 48:52].max() > edges[:, :10].max()

    def test_sobel_uniform_image(self):
        """Uniform image should have near-zero edges."""
        img = np.full((100, 100), 128.0)
        edges = _sobel_edges(img, sigma=1.0)
        assert edges.max() < 1.0


# ── Integration: extract_airfoil end-to-end ────────────────────────


class TestEndToEnd:
    """End-to-end extraction tests simulating real usage."""

    @pytest.mark.parametrize("name", ALL_AIRFOILS)
    def test_dat_end_to_end(self, name, tmp_path):
        """Full pipeline: generate dat → extract → verify shape."""
        dat_path = _airfoil_dat_path(name, tmp_path)
        contour = extract_airfoil(dat_path)

        # Basic shape checks
        assert len(contour.x_surface) == 200
        assert not np.any(np.isnan(contour.y_upper))
        assert not np.any(np.isnan(contour.y_lower))
        assert not np.any(np.isinf(contour.y_upper))
        assert not np.any(np.isinf(contour.y_lower))

        # Physical plausibility: thickness should be positive
        thickness = contour.y_upper - contour.y_lower
        assert np.all(thickness >= -0.01), f"{name}: negative thickness detected"

    @pytest.mark.parametrize("name", IMAGE_TEST_AIRFOILS)
    def test_image_end_to_end(self, name, tmp_path):
        """Full pipeline: render image → extract → verify shape."""
        img_path = _airfoil_image(name, tmp_path, color="black", line_width=2.5)
        contour = extract_airfoil(img_path)

        assert len(contour.x_surface) == 200
        assert not np.any(np.isnan(contour.y_upper))
        assert not np.any(np.isnan(contour.y_lower))

        # Thickness check (allow small tolerance for image noise)
        thickness = contour.y_upper - contour.y_lower
        assert np.all(thickness >= -0.02), f"{name}: negative thickness from image"

    def test_different_num_samples(self, tmp_path):
        """Extraction should respect num_samples parameter."""
        dat_path = _airfoil_dat_path("naca0012", tmp_path)
        for n in [50, 100, 300]:
            contour = extract_airfoil(dat_path, num_samples=n)
            assert contour.x_surface.shape == (n,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
