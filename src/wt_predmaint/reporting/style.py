"""Shared look and the figure registry for report figures (D-025).

Colours follow the validated reference palette of the dataviz method: categorical hues in a fixed
order, one blue ramp for magnitude, text in neutral ink (never in a series colour), thin marks and a
recessive grid. Target A is always blue and target B always orange, so colour follows the entity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
TARGET_COLOUR = {"A": BLUE, "B": ORANGE}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#eaf1fb", "#9cc1ef", "#2a78d6", "#14407a"]
)
DPI = 200


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "font.size": 9,
            "figure.dpi": 110,
        }
    )


@dataclass
class FigureSpec:
    """One report figure: where it lives, what it shows and where it came from."""

    fig_id: str
    folder: str
    name: str
    title: str
    section: str  # where it belongs in the report
    message: str  # what a reader should take from it
    sources: str  # data and code it is built from
    caveats: str = ""
    func: Callable[..., Figure] | None = field(default=None, repr=False)

    @property
    def filename(self) -> str:
        return f"{self.fig_id}_{self.name}.png"

    @property
    def relpath(self) -> str:
        return f"{self.folder}/{self.filename}"


REGISTRY: list[FigureSpec] = []


def figure(
    fig_id: str,
    folder: str,
    name: str,
    title: str,
    section: str,
    message: str,
    sources: str,
    caveats: str = "",
) -> Callable[[Callable[..., Figure]], Callable[..., Figure]]:
    """Register a figure function; the runner saves its result under reports/figures/<folder>/."""

    def wrap(func: Callable[..., Figure]) -> Callable[..., Figure]:
        REGISTRY.append(
            FigureSpec(fig_id, folder, name, title, section, message, sources, caveats, func)
        )
        return func

    return wrap


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def label_end(
    ax: Any, x: float, y: float, text: str, colour: str = INK_2, dx: float = 3, **kw: Any
) -> None:
    """Direct label next to a line end, in ink (not the series colour)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=colour,
        **kw,
    )
