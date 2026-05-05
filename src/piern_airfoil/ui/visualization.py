"""
Visualization tools for airfoil analysis and optimization results.

Provides plotting functions for airfoil shapes, polars, and optimization history.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class AirfoilVisualizer:
    """
    Visualizer for airfoil geometry and aerodynamic performance.

    Simple matplotlib-based visualization.
    """

    def __init__(self, figsize: tuple = (12, 5)):
        self.figsize = figsize
        self._matplotlib_available = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        try:
            import matplotlib
            return True
        except ImportError:
            return False

    def plot_airfoil(self, coordinates: np.ndarray,
                     title: str = "Airfoil Shape",
                     ax=None, show_camber: bool = True) -> Any:
        """
        Plot airfoil shape.

        Args:
            coordinates: Array of shape (N, 2) with x, y coordinates
            title: Plot title
            ax: Existing axes (optional)
            show_camber: Whether to show camber line

        Returns:
            matplotlib axes object
        """
        if not self._matplotlib_available:
            print("Matplotlib not available, skipping plot")
            return None

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        ax.plot(x, y, 'b-', linewidth=2, label='Airfoil')
        ax.fill(x, y, alpha=0.2)

        if show_camber:
            # Compute camber line
            le_idx = np.argmin(x)
            upper_x, upper_y = x[:le_idx+1], y[:le_idx+1]
            lower_x, lower_y = x[le_idx:], y[le_idx:]

            # Simple camber approximation
            if len(upper_x) > 0 and len(lower_x) > 0:
                camber_x = np.sort(np.concatenate([upper_x, lower_x]))
                camber_y = np.interp(camber_x, upper_x, upper_y)

                ax.plot(camber_x, camber_y, 'r--', alpha=0.5, label='Camber')

        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax

    def plot_polars(self, results_list: List[Dict],
                    labels: Optional[List[str]] = None,
                    figsize: tuple = (12, 5)) -> Any:
        """
        Plot lift and drag polars.

        Args:
            results_list: List of dictionaries with 'alpha', 'CL', 'CD' keys
            labels: Labels for each result set
            figsize: Figure size

        Returns:
            matplotlib figure object
        """
        if not self._matplotlib_available:
            print("Matplotlib not available, skipping plot")
            return None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        if labels is None:
            labels = [f"Set {i+1}" for i in range(len(results_list))]

        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

        # CL vs Alpha
        ax = axes[0]
        for results, label, color in zip(results_list, labels, colors):
            ax.plot(results['alpha'], results['CL'], 'o-',
                   color=color, label=label, markersize=3)
        ax.set_xlabel('Alpha [deg]')
        ax.set_ylabel('CL')
        ax.set_title('Lift Polar')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CD vs CL
        ax = axes[1]
        for results, label, color in zip(results_list, labels, colors):
            ax.plot(results['CD'], results['CL'], 'o-',
                   color=color, label=label, markersize=3)
        ax.set_xlabel('CD')
        ax.set_ylabel('CL')
        ax.set_title('Drag Polar')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_optimization_history(self, history: List[Dict],
                                  figsize: tuple = (12, 8)) -> Any:
        """
        Plot optimization history.

        Args:
            history: List of dictionaries with optimization history
            figsize: Figure size

        Returns:
            matplotlib figure object
        """
        if not self._matplotlib_available:
            print("Matplotlib not available, skipping plot")
            return None

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        iterations = [h.get('iteration', i) for i, h in enumerate(history)]

        # CL history
        ax = axes[0, 0]
        if 'CL' in history[0]:
            ax.plot(iterations, [h['CL'] for h in history], 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CL')
        ax.set_title('Lift Coefficient')
        ax.grid(True, alpha=0.3)

        # CD history
        ax = axes[0, 1]
        if 'CD' in history[0]:
            ax.plot(iterations, [h['CD'] for h in history], 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('CD')
        ax.set_title('Drag Coefficient')
        ax.grid(True, alpha=0.3)

        # L/D history
        ax = axes[1, 0]
        if 'L_D' in history[0]:
            ax.plot(iterations, [h['L_D'] for h in history], 'g-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L/D')
        ax.set_title('Lift-to-Drag Ratio')
        ax.grid(True, alpha=0.3)

        # Confidence history
        ax = axes[1, 1]
        if 'confidence' in history[0]:
            ax.plot(iterations, [h['confidence'] for h in history], 'k-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Confidence')
        ax.set_title('Analysis Confidence')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_comparison(self, coords_list: List[np.ndarray],
                        labels: Optional[List[str]] = None,
                        figsize: tuple = (10, 6)) -> Any:
        """
        Compare multiple airfoil shapes.

        Args:
            coords_list: List of coordinate arrays
            labels: Labels for each airfoil
            figsize: Figure size

        Returns:
            matplotlib figure object
        """
        if not self._matplotlib_available:
            print("Matplotlib not available, skipping plot")
            return None

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if labels is None:
            labels = [f"Airfoil {i+1}" for i in range(len(coords_list))]

        colors = plt.cm.viridis(np.linspace(0, 0.8, len(coords_list)))

        for coords, label, color in zip(coords_list, labels, colors):
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color,
                   linewidth=1.5, label=label, alpha=0.8)

        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title('Airfoil Comparison')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig


def plot_polar(results: Dict, label: str = "Airfoil", figsize: tuple = (10, 5)):
    """
    Plot lift polar (CL vs alpha).

    Args:
        results: Dictionary with 'alpha', 'CL', 'CD' keys
        label: Label for the plot
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(results['alpha'], results['CL'], 'o-', label=label, markersize=4)
        ax.set_xlabel('Alpha [deg]')
        ax.set_ylabel('CL')
        ax.set_title('Lift Polar')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Plotting failed: {e}")
        return None


def plot_optimization_history(history: list, figsize: tuple = (12, 8)):
    """
    Plot optimization history.

    Args:
        history: List of dicts with 'iteration', 'CL', 'CD', etc.
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        iterations = [h.get('iteration', i) for i, h in enumerate(history)]

        # CL history
        if 'CL' in history[0]:
            axes[0, 0].plot(iterations, [h['CL'] for h in history], 'b-')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('CL')
        axes[0, 0].set_title('Lift Coefficient')
        axes[0, 0].grid(True, alpha=0.3)

        # CD history
        if 'CD' in history[0]:
            axes[0, 1].plot(iterations, [h['CD'] for h in history], 'r-')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('CD')
        axes[0, 1].set_title('Drag Coefficient')
        axes[0, 1].grid(True, alpha=0.3)

        # L/D history
        if 'L_D' in history[0]:
            axes[1, 0].plot(iterations, [h['L_D'] for h in history], 'g-')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('L/D')
        axes[1, 0].set_title('Lift-to-Drag Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Confidence history
        if 'confidence' in history[0]:
            axes[1, 1].plot(iterations, [h['confidence'] for h in history], 'k-')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Analysis Confidence')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Plotting failed: {e}")
        return None
