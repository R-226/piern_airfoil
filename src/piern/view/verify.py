"""Verification script: extract airfoil from image, plot alongside original."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from piern.view.extract import AirfoilContour, extract_airfoil, save_dat

matplotlib.rcParams["figure.dpi"] = 150


def plot_comparison(
    contour: AirfoilContour,
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Plot extracted coordinates next to the original image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: original image
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    # Right: extracted coordinates
    ax = axes[1]
    ax.plot(contour.x_surface, contour.y_upper, "b-", linewidth=1.5, label="Upper")
    ax.plot(contour.x_surface, contour.y_lower, "r-", linewidth=1.5, label="Lower")
    ax.set_aspect("equal")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title("Extracted coordinates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
    plt.show()


def plot_roundtrip(
    original_dat: str | Path,
    extracted: AirfoilContour,
    output_path: str | Path | None = None,
) -> None:
    """Plot original .dat coordinates vs extracted coordinates."""
    data = np.loadtxt(original_dat, skiprows=1)
    orig_x, orig_y = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(orig_x, orig_y, "k-", linewidth=1, alpha=0.7, label="Original .dat")
    ax.plot(
        extracted.x_surface,
        extracted.y_upper,
        "b--",
        linewidth=1.5,
        label="Extracted (upper)",
    )
    ax.plot(
        extracted.x_surface,
        extracted.y_lower,
        "r--",
        linewidth=1.5,
        label="Extracted (lower)",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Roundtrip: Original .dat vs Extracted from image")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved roundtrip plot to {output_path}")
    plt.show()

    # Compute error metrics
    # Interpolate original onto extracted x grid for upper surface
    orig_upper_mask = orig_y >= 0
    orig_lower_mask = orig_y <= 0
    # Sort by x for interpolation
    orig_upper_x = orig_x[orig_upper_mask]
    orig_upper_y = orig_y[orig_upper_mask]
    sort_idx = np.argsort(orig_upper_x)
    orig_upper_x, orig_upper_y = orig_upper_x[sort_idx], orig_upper_y[sort_idx]

    orig_lower_x = orig_x[orig_lower_mask]
    orig_lower_y = orig_y[orig_lower_mask]
    sort_idx = np.argsort(orig_lower_x)
    orig_lower_x, orig_lower_y = orig_lower_x[sort_idx], orig_lower_y[sort_idx]

    # Only compare in the range where original data exists
    x_min = max(extracted.x_surface.min(), orig_upper_x.min())
    x_max = min(extracted.x_surface.max(), orig_upper_x.max())
    mask = (extracted.x_surface >= x_min) & (extracted.x_surface <= x_max)

    if mask.any():
        interp_upper = np.interp(extracted.x_surface[mask], orig_upper_x, orig_upper_y)
        interp_lower = np.interp(extracted.x_surface[mask], orig_lower_x, orig_lower_y)

        err_upper = np.abs(extracted.y_upper[mask] - interp_upper)
        err_lower = np.abs(extracted.y_lower[mask] - interp_lower)

        print(f"\n--- Error Metrics ---")
        print(f"Upper surface:  max={err_upper.max():.6f}, mean={err_upper.mean():.6f}")
        print(f"Lower surface:  max={err_lower.max():.6f}, mean={err_lower.mean():.6f}")
        print(
            f"Combined:       max={max(err_upper.max(), err_lower.max()):.6f}, "
            f"mean={(err_upper.mean() + err_lower.mean()) / 2:.6f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    image_path = project_root / "data" / "airfoil" / "naca0012.png"
    dat_path = project_root / "data" / "airfoil" / "naca0012.dat"
    output_dat = project_root / "data" / "airfoil" / "naca0012_extracted.dat"

    print(f"Extracting airfoil from {image_path}...")
    contour = extract_airfoil(image_path, num_samples=200)
    print(f"Extracted {len(contour.x_surface)} points per surface")

    # Save extracted coordinates
    save_dat(contour, output_dat)
    print(f"Saved extracted coordinates to {output_dat}")

    # Plot comparison
    plot_comparison(
        contour,
        image_path,
        output_path=project_root / "data" / "airfoil" / "comparison.png",
    )

    # Plot roundtrip against original .dat
    plot_roundtrip(
        dat_path,
        contour,
        output_path=project_root / "data" / "airfoil" / "roundtrip.png",
    )


if __name__ == "__main__":
    main()
