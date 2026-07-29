"""Stage 18 arm B7: conceptual mechanism schematic (PREDICTIONS.md "Stage
18 preregistration", arm B7). Four panels illustrating the "collapse by
rotation" mechanism (STATUS.md): encoded constellation -> global rotation
-> measured projection -> apparent collapse. Purely illustrative/synthetic
-- NOT a data figure regenerated from any run's metrics.jsonl, so
CLAUDE.md invariant 4 (figures as a pure function of metrics.jsonl) does
not apply here; this is a conceptual diagram for the paper, analogous to
a textbook schematic, not an experimental result plot.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]
_RNG = np.random.default_rng(0)


def _toy_constellation(spread: float, center_radius: float = 1.6) -> list[np.ndarray]:
    """3 synthetic class blobs on a 2D plane (schematic stand-in for the
    class-mean-plus-spread structure the real 4^n-1 dimensional
    constellation has)."""
    angles = np.array([90, 210, 330]) * np.pi / 180
    centers = center_radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return [c + _RNG.normal(scale=spread, size=(40, 2)) for c in centers]


def _rotate(points_by_class: list[np.ndarray], theta: float) -> list[np.ndarray]:
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return [p @ R.T for p in points_by_class]


def _panel_scatter(ax, points_by_class, title, xlabel, ylabel, xlim=(-3, 3), ylim=(-3, 3)):
    for c, pts in enumerate(points_by_class):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7, color=_COLORS[c], edgecolors="none")
        ax.scatter(*pts.mean(axis=0), s=90, marker="*", color=_COLORS[c], edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _draw_mechanism_figure(width_in: float, out_path: Path, grid: bool) -> None:
    """`grid=False`: 1x4 row (journal/double-column width). `grid=True`:
    2x2 grid (single-column width -- 4 panels side-by-side don't fit
    legibly at single-column widths, so the standard fix is a 2x2 grid)."""
    if grid:
        fig, axes2d = plt.subplots(2, 2, figsize=(width_in, width_in * 1.05), constrained_layout=True)
        axes = [axes2d[0, 0], axes2d[0, 1], axes2d[1, 0], axes2d[1, 1]]
        fontsize_title, fontsize_note = 9, 6
    else:
        fig, axes = plt.subplots(1, 4, figsize=(width_in, width_in / 2.7), constrained_layout=True)
        fontsize_title, fontsize_note = 11, 6.5

    constellation = _toy_constellation(spread=0.35)
    _panel_scatter(axes[0], constellation, "(a) Encoded\nconstellation", "measured", "fiber")
    axes[0].title.set_fontsize(fontsize_title)

    theta = np.pi / 3
    rotated = _rotate(constellation, theta)
    _panel_scatter(axes[1], rotated, "(b) Global\nrotation $U(\\theta)$", "measured", "fiber")
    axes[1].title.set_fontsize(fontsize_title)
    arrow = FancyArrowPatch((1.0, -2.4), (1.9, -1.7), connectionstyle="arc3,rad=-0.4",
                             arrowstyle="->", mutation_scale=12, color="gray", linewidth=1.2)
    axes[1].add_patch(arrow)

    # (c) measured projection: collapse the fiber (y) axis to a thin sliver,
    # keep the measured (x) axis at rotated scale -- illustrates that only
    # the measurement-visible component of the same rotated constellation
    # is what a Z-basis readout actually sees.
    projected = [np.stack([p[:, 0], p[:, 1] * 0.12], axis=1) for p in rotated]
    _panel_scatter(axes[2], projected, "(c) Measured\nprojection", "measured (visible)", "fiber (hidden)")
    axes[2].title.set_fontsize(fontsize_title)

    # (d) apparent collapse: only the measured axis shown, class means far
    # apart relative to shrunken within-class spread -- what a measurement-
    # space NC1_z witness reports, with a text note that the fiber variance
    # (not shown on this axis) is unchanged.
    final = [np.stack([p[:, 0] * 0.55, p[:, 1]], axis=1) for p in projected]
    _panel_scatter(axes[3], final, "(d) Apparent\ncollapse", "measured", "fiber (unchanged)")
    axes[3].title.set_fontsize(fontsize_title)
    axes[3].text(
        0, -2.7, "total variance conserved\n(fiber-space absorbs it)",
        ha="center", fontsize=fontsize_note, style="italic", color="dimgray",
    )

    # Short in-image title only; the full explanatory sentence ("a
    # training-driven rotation redistributes a fixed variance budget out
    # of the measured subspace and into fiber directions") belongs in the
    # paper's LaTeX \caption{}, not baked into the image -- standard
    # practice, and the only way both the single-column (2x2) and
    # double-column (1x4) layouts can carry it legibly.
    fig.suptitle("Collapse by rotation", fontsize=fontsize_title + 1, fontweight="bold")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_mechanism_schematic(out_dir: str | Path = "figs") -> tuple[Path, Path]:
    """Writes both a single-column (3.5in wide, 2x2 grid) and a
    double-column/journal (7.0in wide, 1x4 row) version of the four-panel
    schematic."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    single_col = out_dir / "fig18_mechanism_schematic_singlecol.png"
    double_col = out_dir / "fig18_mechanism_schematic_doublecol.png"
    _draw_mechanism_figure(3.5, single_col, grid=True)
    _draw_mechanism_figure(7.0, double_col, grid=False)
    return single_col, double_col


if __name__ == "__main__":
    paths = make_mechanism_schematic()
    print("wrote:", *paths)
