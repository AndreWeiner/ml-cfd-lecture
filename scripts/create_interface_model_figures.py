"""Conceptual sharp-interface, VOF, and interface-fitted mesh sketches.

Reproduce with the course environment (no CFD solution is computed)::

    MPLCONFIGDIR=/tmp/ml-interface-mpl .venv-quarto/bin/python \
        scripts/create_interface_model_figures.py --output-dir notebooks/media

Geometry, verification, and references: figures/interface-models/README.md.
"""
from pathlib import Path
import argparse
import json
import sys
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from cycler import cycler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from figure_style import WIDTH_MM, DPI, figure_style
from figure_utils import save_theme_variants

# One analytic, mildly oblate bubble for all three conceptual representations.
# The lengths are arbitrary sketch units, not a measured or simulated bubble.
A, B = 1.38, 1.04
THETA = np.linspace(0, 2 * np.pi, 8193)
BOUNDARY = np.column_stack((A * np.cos(THETA), B * np.sin(THETA)))
COLORS = plt.colormaps["tab10"].colors
HEIGHT_MM = {"sharp-interface": 111, "vof-interface": 133,
             "fitted-interface": 117}


def apply_style(use_tex):
    """The helper is called after the theme, so tab10 is preserved in dark mode."""
    plt.rcParams.update(figure_style(use_tex=use_tex))
    plt.rcParams.update({"axes.prop_cycle": cycler(color=COLORS),
                         "text.latex.preamble": "", "savefig.dpi": DPI})


def canvas(stem, use_tex, *, bounds):
    apply_style(use_tex)
    fig = plt.figure(figsize=(WIDTH_MM["double"] / 25.4,
                              HEIGHT_MM[stem] / 25.4))
    ax = fig.add_axes((.035, .12, .93, .83))
    ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal")
    ax.axis("off")
    return fig, ax, plt.rcParams["text.color"]


def normal(theta):
    outward = np.array([np.cos(theta) / A, np.sin(theta) / B])
    return outward / np.linalg.norm(outward)


def normal_arrow(ax, theta, neutral, *, label_offset=(0., 0.), origin=(0., 0.)):
    point = np.asarray(origin) + [A * np.cos(theta), B * np.sin(theta)]
    outward = normal(theta)
    start, end = point, point + .65 * outward
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=neutral, shrinkA=0, shrinkB=0),
                zorder=9)
    loc = (start + end) / 2 + np.array(label_offset)
    ax.text(*loc, r"$\mathbf{n}_{\Sigma}$", color=neutral,
            ha="center", va="center", zorder=10,
            bbox=dict(facecolor=ax.get_facecolor(), edgecolor="none", pad=1))


def draw_sharp(use_tex=True):
    fig, ax, neutral = canvas("sharp-interface", use_tex,
                              bounds=(-3.40, 3.40, -2.55, 1.75))
    center = .95
    ax.plot(BOUNDARY[:, 0] + center, BOUNDARY[:, 1],
            color=COLORS[0], lw=1.6, zorder=4)
    for x, phase, side in [(-1.95, "Liquid", "+"), (center, "Gas", "-")]:
        ax.text(x, .42, rf"{phase} $\Omega^{side}(t)$",
                color=neutral, ha="center", va="center")
        ax.text(x, .02, rf"$\nabla\cdot\mathbf{{u}}^{side}=0$",
                color=neutral, ha="center", va="center")
        ax.text(x, -.39,
                rf"$\rho^{side}\frac{{\mathrm{{D}}\mathbf{{u}}^{side}}}{{\mathrm{{D}}t}}"
                rf"=\nabla\cdot\mathbf{{T}}^{side}+\rho^{side}\mathbf{{g}}$",
                color=neutral, ha="center", va="center")

    normal_arrow(ax, 1.25, neutral, label_offset=(.28, 0),
                 origin=(center, 0))
    ax.annotate(r"$\Sigma(t)$", xy=BOUNDARY[600] + (center, 0),
                xytext=(2.68, .84), color=neutral, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=.8, color=neutral))
    ax.text(0, -1.40, r"Coupling on $\Sigma(t)$", color=neutral,
            ha="center", va="center")
    ax.text(0, -1.79,
            r"$\dot m^+=\dot m^-=0,\qquad"
            r"\mathbf{u}^+=\mathbf{u}^-,\qquad"
            r"\mathbf{u}^\pm\cdot\mathbf{n}_\Sigma=V_\Sigma$",
            color=neutral, ha="center", va="center")
    ax.text(0, -2.20,
            r"$(\mathbf{T}^+-\mathbf{T}^-)\mathbf{n}_\Sigma="
            r"\gamma\kappa\mathbf{n}_\Sigma-\nabla_\Sigma\gamma$",
            color=neutral, ha="center", va="center")
    return fig


def polygon_area(vertices):
    if len(vertices) < 3:
        return 0.
    return .5 * abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                    - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1)))


def clip_polygon(vertices, direction, threshold):
    """Clip a convex polygon to dot(direction, x) <= threshold."""
    if not len(vertices):
        return vertices
    distances = vertices @ direction - threshold
    inside = distances <= 0.
    if inside.all():
        return vertices.copy()
    if not inside.any():
        return np.empty((0, 2))
    result = []
    for i, point in enumerate(vertices):
        previous = vertices[i - 1]
        if inside[i] != inside[i - 1]:
            t = distances[i - 1] / (distances[i - 1] - distances[i])
            result.append(previous + t * (point - previous))
        if inside[i]:
            result.append(point)
    return np.asarray(result)


def cut_segment(rectangle, direction, threshold):
    distances = rectangle @ direction - threshold
    points = []
    for i, point in enumerate(rectangle):
        previous = rectangle[i - 1]
        if distances[i] * distances[i - 1] < 0:
            t = distances[i - 1] / (distances[i - 1] - distances[i])
            points.append(previous + t * (point - previous))
    if len(points) != 2:
        raise ValueError("Each nondegenerate PLIC cut must have two endpoints")
    return np.asarray(points)


def vof_geometry():
    """True ellipse areas to polygon tolerance, then area-matched local planes.

    The known analytic ellipse supplies local normal directions for this sketch.
    No claim is made that a numerical VOF method recovers those normals exactly.
    """
    x = np.linspace(-2.10, 2.10, 13)
    y = np.linspace(-1.575, 1.575, 10)
    alpha = np.zeros((len(y)-1, len(x)-1))
    segments, errors = [], []
    for j, (bottom, top) in enumerate(zip(y[:-1], y[1:])):
        for i, (left, right) in enumerate(zip(x[:-1], x[1:])):
            rectangle = np.array([[left, bottom], [right, bottom],
                                  [right, top], [left, top]])
            cell_area = (right-left) * (top-bottom)
            if np.all((rectangle[:, 0]/A)**2 + (rectangle[:, 1]/B)**2 < 1):
                fraction = 1.
            elif right < -A or left > A or top < -B or bottom > B:
                fraction = 0.
            else:
                clipped = BOUNDARY[:-1]
                for direction, value in (([1, 0], right), ([-1, 0], -left),
                                         ([0, 1], top), ([0, -1], -bottom)):
                    clipped = clip_polygon(clipped, np.asarray(direction), value)
                fraction = np.clip(polygon_area(clipped) / cell_area, 0., 1.)
            alpha[j, i] = fraction
            if 1e-10 < fraction < 1-1e-10:
                direction = np.array([(left+right)/(2*A*A),
                                      (bottom+top)/(2*B*B)])
                direction /= np.linalg.norm(direction)
                lo, hi = min(rectangle @ direction), max(rectangle @ direction)
                for _ in range(60):
                    mid = .5 * (lo + hi)
                    area = polygon_area(clip_polygon(rectangle, direction, mid))
                    if area < fraction * cell_area:
                        lo = mid
                    else:
                        hi = mid
                threshold = .5 * (lo+hi)
                segments.append(cut_segment(rectangle, direction, threshold))
                reconstructed = polygon_area(clip_polygon(rectangle, direction,
                                                          threshold)) / cell_area
                errors.append(abs(reconstructed - fraction))
    total = np.sum(alpha) * (x[1]-x[0]) * (y[1]-y[0])
    diagnostics = {"mixed_cells": len(segments),
                   "max_plic_fraction_error": max(errors),
                   "ellipse_area_relative_error": abs(total/(np.pi*A*B)-1),
                   "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max())}
    assert diagnostics["max_plic_fraction_error"] < 1e-12
    assert diagnostics["ellipse_area_relative_error"] < 2e-7
    return x, y, alpha, segments, diagnostics


def draw_vof(geometry, use_tex=True):
    apply_style(use_tex)
    x, y, alpha, segments, _ = geometry
    fig = plt.figure(figsize=(WIDTH_MM["double"]/25.4,
                              HEIGHT_MM["vof-interface"]/25.4))
    neutral = plt.rcParams["text.color"]
    ax = fig.add_axes((.055, .26, .735, .66))
    ax.set(aspect="equal", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.axis("off")
    scalar = ax.pcolormesh(x, y, alpha, cmap="viridis", vmin=0, vmax=1,
                           shading="flat", edgecolors="#888888", linewidth=.45)
    # A white keyline and dark core remain visible across the complete scalar map.
    cuts = LineCollection(segments, colors="#111111", linewidths=1.7, zorder=5)
    cuts.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
    ax.add_collection(cuts)
    ax.text(-1.92, 1.37, r"Liquid: $\alpha_g=0$", color="white",
            ha="left", va="center", zorder=6,
            bbox=dict(facecolor=plt.colormaps["viridis"](0.), edgecolor="none", pad=2))
    ax.text(0, 0, r"Gas: $\alpha_g=1$", color="#111111",
            ha="center", va="center", zorder=6,
            bbox=dict(facecolor=plt.colormaps["viridis"](1.), edgecolor="none", pad=2))
    cax = fig.add_axes((.835, .305, .027, .565))
    cb = fig.colorbar(scalar, cax=cax, ticks=(0, .5, 1))
    cb.set_label(r"Gas volume fraction $\alpha_g$", labelpad=9)
    cb.ax.tick_params(length=3, width=.8)
    fig.text(.42, .955, "Fixed grid with cellwise interface reconstruction",
             color=neutral, ha="center", va="center")
    legend_line = Line2D([], [], color="#111111", lw=1.7,
                         path_effects=[pe.Stroke(linewidth=3.2, foreground="white"),
                                       pe.Normal()])
    fig.legend([legend_line], ["PLIC: one straight segment per mixed cell"],
               loc="center", bbox_to_anchor=(.46, .222), frameon=False, fontsize=10)
    fig.text(.5, .138,
             (r"$\displaystyle " if use_tex else "$")
             + r"\alpha_g = \frac{|K\cap\Omega^-(t)|}{|K|}"
             r" = \frac{1}{|K|}\int_K \chi_g(\mathbf{x},t)\,\mathrm{d}V$",
             color=neutral, ha="center", va="center")
    fig.text(.5, .045, "2D conceptual sketch: cell areas represent unit-depth volumes",
             color=neutral, fontsize=10, ha="center", va="center")
    return fig


def draw_fitted(use_tex=True):
    fig, ax, neutral = canvas("fitted-interface", use_tex,
                              bounds=(-2.55, 3.85, -1.90, 2.05))
    theta = np.linspace(0, 2*np.pi, 25)
    scales = np.array([.61, .80, 1., 1.20, 1.42, 1.64])
    ax.add_patch(Ellipse((0, 0), 2*A*scales[-1], 2*B*scales[-1],
                         facecolor=COLORS[0], edgecolor="none", alpha=.065))
    ax.add_patch(Ellipse((0, 0), 2*A, 2*B, facecolor=fig.get_facecolor(),
                         edgecolor="none"))
    ax.add_patch(Ellipse((0, 0), 2*A, 2*B, facecolor=COLORS[1],
                         edgecolor="none", alpha=.12))
    # Curvilinear quadrilateral bands on both sides share the very same s=1 face.
    # Only an interface neighborhood is drawn, not a complete computational mesh.
    for scale in scales:
        ax.plot(*(scale*BOUNDARY).T, color=neutral, lw=.6, alpha=.48, zorder=2)
    for angle in theta[:-1]:
        nodes = scales[:, None] * np.array([A*np.cos(angle), B*np.sin(angle)])
        ax.plot(*nodes.T, color=neutral, lw=.6, alpha=.48, zorder=2)
    ax.plot(*BOUNDARY.T, color=COLORS[0], lw=1.7, zorder=4)
    ax.scatter(A*np.cos(theta[:-1]), B*np.sin(theta[:-1]), s=8,
               color=COLORS[0], zorder=5)
    ax.text(0, .10, r"Gas: $\Omega^-(t)$", color=neutral, ha="center")
    ax.text(0, -.22, "Interior omitted", color=neutral, ha="center", fontsize=10)
    label_box = dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=1.5)
    ax.text(-2.35, 1.88, r"Liquid: $\Omega^+(t)$", color=neutral, ha="left")
    # A prescribed upward direction illustrates interface motion, not a velocity.
    ax.annotate("", xy=(0., 1.89), xytext=(0., B),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=COLORS[2]), zorder=7)
    ax.annotate("Interface motion\n(mesh follows)", xy=(0., 1.72),
                xytext=(1.12, 1.75), color=neutral, ha="left", va="center",
                bbox=label_box,
                arrowprops=dict(arrowstyle="-", lw=.8, color=neutral), zorder=8)
    normal_arrow(ax, -.80, neutral, label_offset=(-.24, -.16))
    ax.annotate(r"Shared boundary $\Sigma(t)$", xy=(A, 0),
                xytext=(2.20, .36), color=neutral, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=.8, color=neutral),
                bbox=label_box, zorder=9)
    ax.text(2.20, -1.18, "Cells on both sides\nfit the interface", color=neutral,
            ha="left", va="center", linespacing=1.3, bbox=label_box, zorder=9)
    fig.text(.5, .055, "Conceptual moving mesh; only the interface neighborhood is shown",
             color=neutral, fontsize=10, ha="center", va="center")
    return fig


def smoke_test(output_dir, use_tex):
    """Exercise actual math/font rendering and both export backends first."""
    with plt.style.context("default"):
        apply_style(use_tex)
        fig, ax = plt.subplots(figsize=(4, 1.4))
        ax.axis("off")
        ax.text(.03, .5, r"$\Omega^+(t),\ \Omega^-(t),\ \Sigma(t),\ "
                r"\mathbf{n}_{\Sigma},\ \alpha_g = |K\cap\Omega^-|/|K|$"
                "\nMass, momentum, species balances", va="center")
        for ext in ("svg", "png"):
            fig.savefig(output_dir / f"renderer-smoke.{ext}", dpi=DPI)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/media"))
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    args = parser.parse_args()
    use_tex = resolve_renderer("mathtext" if args.no_tex else args.renderer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="interface-renderer-") as smoke_dir:
        smoke_test(Path(smoke_dir), use_tex)
    geometry = vof_geometry()
    for stem, draw in (("sharp-interface", lambda: draw_sharp(use_tex)),
                       ("vof-interface", lambda: draw_vof(geometry, use_tex)),
                       ("fitted-interface", lambda: draw_fitted(use_tex))):
        for extension in ("svg", "pdf", "png"):
            save_theme_variants(draw, args.output_dir / f"{stem}.{extension}", dpi=DPI)
    print(json.dumps({"renderer": "MathText" if not use_tex
                      else "LaTeX (text.usetex=True)",
                      "width_mm": WIDTH_MM["double"], "height_mm": HEIGHT_MM,
                      **geometry[-1]}, indent=2))


if __name__ == "__main__":
    main()
