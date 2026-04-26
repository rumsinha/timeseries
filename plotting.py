"""
Shared plotting style and helpers for the Jena Climate series.
Import STYLE_CONTEXT or call apply_style() at the top of any notebook.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palette (consistent across all 11 weekends)
# ---------------------------------------------------------------------------
BLUE = "#2563EB"
GRAY = "#9CA3AF"
ORANGE = "#F59E0B"
RED = "#DC2626"
GREEN = "#16A34A"

# ---------------------------------------------------------------------------
# Global style defaults
# ---------------------------------------------------------------------------
_STYLE_PARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Apply the series-wide rcParams. Call once per notebook."""
    mpl.rcParams.update(_STYLE_PARAMS)


def save_plot(fig: plt.Figure, path: Path, close: bool = False, **kwargs) -> None:
    """Save fig to path, creating parent dirs as needed.

    Parameters
    ----------
    close : bool
        Close the figure after saving. Default False so notebooks can still
        display it inline after calling this function.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    if close:
        plt.close(fig)
