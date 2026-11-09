"""Conceptual bubble-scale transport schematic, not a computed flow solution.

Reproduce: MPLCONFIGDIR=/tmp/ml-schmidt-mpl .venv-quarto/bin/python \
    scripts/create_schmidt_problem_figure.py --output-dir notebooks/media
See figures/schmidt-problem/README.md for geometry, assumptions and sources.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from cycler import cycler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from figure_style import WIDTH_MM, DPI, figure_style
from figure_utils import save_theme_variants

COLORS = plt.colormaps["tab10"].colors
A, B = 1.30, 1.04
THETA = np.linspace(0, 2 * np.pi, 801)
INTERFACE = np.column_stack((A * np.sin(THETA), B * np.cos(THETA)))
NORMAL = np.column_stack((np.sin(THETA) / A, np.cos(THETA) / B))
NORMAL /= np.linalg.norm(NORMAL, axis=1)[:, None]
# Front at theta=0; surface transport runs toward theta=pi (the rear).
# These offsets communicate the topology only; they are deliberately exaggerated.
HYDRO = .20 + .10 * (1 - np.cos(THETA)) / 2
SPECIES = .035 + .11 * ((1 - np.cos(THETA)) / 2) ** 1.4
MEAN = .080


def offset(thickness):
    return INTERFACE + np.asarray(thickness)[..., None] * NORMAL


def arrow(ax, points, i, *, color, direction=1, step=10):
    ax.add_patch(FancyArrowPatch(points[i], points[i + direction * step],
                                arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.1, color=color, zorder=10))


def draw(use_tex=True):
    # Called inside the theme context: reset categorical colors AFTER its styles.
    plt.rcParams.update(figure_style(use_tex=use_tex))
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)
    neutral = plt.rcParams["text.color"]
    flow, species = COLORS[0], COLORS[1]
    fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 120 / 25.4))
    fig.subplots_adjust(left=.025, right=.975, bottom=.04, top=.96)
    text_box = dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=1.5)
    ax.set(xlim=(-3.65, 3.65), ylim=(-2.50, 2.55), aspect="equal")
    ax.axis("off")

    # Schematic exterior streamlines: constant sphere potential-flow streamfunction
    # curves, affinely deformed to the ellipse. They are NOT a spheroid solution.
    yy = np.linspace(2.45, -2.40, 501)
    inner_streamline = None
    for sign in (-1, 1):
        for psi in (.16, .53, 1.03):
            lo, hi = np.full_like(yy, psi), np.full_like(yy, 4.)
            for _ in range(55):
                mid = (lo + hi) / 2
                val = mid**2 * (1 - (mid**2 + yy**2) ** (-1.5))
                lo = np.where(val < psi**2, mid, lo)
                hi = np.where(val >= psi**2, mid, hi)
            points = np.column_stack((sign * A * (lo + hi) / 2, B * yy))
            if sign == 1 and psi == .16:
                inner_streamline = points.copy()
            ax.plot(*points.T, color=flow, linewidth=1.0, alpha=.85, zorder=1)
            arrow(ax, points, 90, color=flow)
            arrow(ax, points, 360, color=flow)

    hyd = offset(HYDRO)
    con = offset(SPECIES)
    mean = offset(np.full_like(THETA, MEAN))
    # Continue the surface concentration region into the downstream wake.
    # The outer envelope joins the lower local-layer contour to the innermost
    # exterior streamline. This is illustrative geometry, not a scalar solution.
    wake_y = np.linspace(0., inner_streamline[-1, 1], 401)
    lower = con[200:401]
    layer_x = np.interp(wake_y, lower[::-1, 1], lower[::-1, 0], left=0.)
    stream_x = np.interp(wake_y, inner_streamline[::-1, 1], inner_streamline[::-1, 0])
    wake_x = np.maximum(layer_x, stream_x)
    front = np.vstack((con[600:], con[1:201]))
    right = np.column_stack((wake_x, wake_y))
    left = np.column_stack((-wake_x[::-1], wake_y[::-1]))
    concentration = np.vstack((front, right[1:], left))
    # The bubble interior is cut out in the page color.
    ax.add_patch(Polygon(hyd, closed=True, facecolor=flow, alpha=.16, edgecolor="none", zorder=2))
    ax.plot(*hyd.T, color=flow, linewidth=1.4, zorder=3)
    ax.add_patch(Polygon(concentration, closed=True, facecolor=species, alpha=.40, edgecolor="none", zorder=3))
    ax.plot(*np.vstack((front, right[1:])).T, color=species, linewidth=1.4, zorder=4)
    ax.plot(*left.T, color=species, linewidth=1.4, zorder=4)
    ax.plot(*mean.T, color=species, linewidth=1.3, linestyle=(0, (4, 3)), zorder=5)
    ax.add_patch(Polygon(INTERFACE, closed=True,
                        facecolor=fig.get_facecolor(), edgecolor=neutral,
                        linewidth=1.6, zorder=6))

    # Two counter-rotating halves of a toroidal gas circulation, in meridional view.
    # Outer branches move down along the interface; central branches return upward.
    phi = np.linspace(0, 2*np.pi, 301)
    for sign in (-1, 1):
        for scale in (1., .60):
            points = np.column_stack((sign * (.56 + .37 * scale * np.cos(phi)),
                                      .67 * scale * np.sin(phi)))
            ax.plot(*points.T, color=flow, linewidth=1.1, zorder=7)
            arrow(ax, points, 45, color=flow, direction=-1, step=30)
            arrow(ax, points, 180, color=flow, direction=-1, step=30)
    ax.text(0, .84, "gas", ha="center", va="center", color=neutral, zorder=8)
    ax.annotate("Internal\nrecirculation", xy=(-.56, -.30), xytext=(-3.35, -.95),
                ha="left", color=neutral, fontsize=10, bbox=text_box,
                arrowprops=dict(arrowstyle="-", color=neutral, lw=.8), zorder=11)

    # Bubble is stationary in this frame; rise arrow refers to its laboratory motion.
    ax.annotate("", xy=(-2.65, 1.98), xytext=(-2.65, 1.12),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=neutral))
    ax.text(-2.43, 1.55, "$U_b$\n(lab frame)", ha="left", va="center", color=neutral)
    ax.text(.55, 2.23, r"Liquid inflow: $-U_b\mathbf{e}_z$", ha="center", color=neutral, bbox=text_box, zorder=11)
    ax.text(0, -2.30, "Bubble-fixed frame; layer widths exaggerated", ha="center", color=neutral, bbox=text_box, zorder=11)

    def label(text, point, pos):
        ax.annotate(text, xy=point, xytext=pos, ha="left", va="center",
                    color=neutral, fontsize=10,
                    arrowprops=dict(arrowstyle="-", lw=.8, color=neutral,
                                    connectionstyle="angle,angleA=0,angleB=90,rad=4"),
                    zorder=9)
    label("Species carried\ninto the wake", right[290], (1.92, -1.35))
    legend = [Line2D([], [], color=neutral, lw=1.6, label="Interface"),
              Line2D([], [], color=flow, lw=1.4, label=r"Velocity layer: $\delta_h$"),
              Line2D([], [], color=species, lw=1.4, label=r"Species layer / wake"),
              Line2D([], [], color=species, lw=1.3, ls="--", label=r"Mean film: $\overline{\delta}_c$")]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(.74, .57),
               ncol=1, frameon=False, fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/media"))
    parser.add_argument("--no-tex", action="store_true", help="Explicitly disable LaTeX rendering")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        save_theme_variants(lambda: draw(not args.no_tex),
                            args.output_dir / f"bubble-schmidt-layers.{ext}", dpi=DPI)


if __name__ == "__main__":
    main()
