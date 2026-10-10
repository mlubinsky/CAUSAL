"""Minimal, consistent matplotlib styling for the notebooks.

Categorical colors are assigned in a fixed order (blue, orange, aqua, yellow),
never cycled past that; ground-truth reference lines use a dashed neutral ink.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
TRUTH = dict(color=INK_2, linestyle="--", linewidth=1.2)


def use_style() -> None:
    mpl.rcParams.update(
        {
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.figsize": (8, 3.6),
            "figure.dpi": 100,
        }
    )


def truth_line(ax: plt.Axes, value: float, label: str = "truth", axis: str = "v") -> None:
    """Dashed reference line at the true value."""
    if axis == "v":
        ax.axvline(value, label=label, **TRUTH)
    else:
        ax.axhline(value, label=label, **TRUTH)
