"""Conceptual VOF-to-liquid workflow and axisymmetric boundary coordinates.

Reproduce from the repository root with the course environment::

    MPLCONFIGDIR=/tmp/ml-hybrid-mpl .venv-quarto/bin/python \
        scripts/create_hybrid_workflow_figures.py --output-dir notebooks/media

No CFD, regression fit, or measured data are produced. See
figures/hybrid-workflow/README.md for definitions and interpretation.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
from cycler import cycler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from figure_style import WIDTH_MM, DPI, figure_style
from figure_utils import save_theme_variants
from create_interface_model_figures import vof_geometry

COLORS = plt.colormaps["tab10"].colors
HEIGHT_MM = {"bubble-hybrid-workflow": 74,
             "bubble-interface-coordinates": 118}
ANGLE = np.linspace(0., 2*np.pi, 721)
A, B = .94, .76
ELLIPSE = np.column_stack((A*np.sin(ANGLE), B*np.cos(ANGLE)))


def apply_style(use_tex):
    plt.rcParams.update(figure_style(use_tex=use_tex))
    plt.rcParams.update({"axes.prop_cycle": cycler(color=COLORS),
                         "text.latex.preamble": ""})


def arrow(ax, start, end, color, *, lw=1.5, scale=10, style="-|>", zorder=6):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale,
                            linewidth=lw, color=color, zorder=zorder)
    ax.add_patch(patch)
    return patch


def axis(fig, rect, xlim, ylim):
    ax = fig.add_axes(rect)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.axis("off")
    return ax


def liquid_mesh(ax, neutral):
    scales = (1., 1.16, 1.36, 1.62)
    for scale in scales:
        ax.plot(*(scale*ELLIPSE).T, color=neutral, lw=.55, alpha=.40)
    for theta in np.linspace(0., 2*np.pi, 21)[:-1]:
        nodes = np.asarray(scales)[:, None] * np.array([A*np.sin(theta), B*np.cos(theta)])
        ax.plot(*nodes.T, color=neutral, lw=.55, alpha=.40)
    ax.plot(*ELLIPSE.T, color=COLORS[0], lw=1.6, zorder=5)


def workflow_vof(fig, left, geometry):
    x, y, alpha, segments, _ = geometry
    ax = axis(fig, (left+.010, .30, .200, .50),
              (x[0], x[-1]), (y[0], y[-1]))
    ax.pcolormesh(x, y, alpha, cmap="viridis", vmin=0., vmax=1.,
                  shading="flat", edgecolors="#888888", linewidth=.30)
    # These are the geometrically matched segments from the interface sketch.
    ax.add_collection(LineCollection(segments, colors="white", linewidths=1.5, zorder=4))
    samples = np.asarray(segments)[::4].mean(axis=1)
    ax.scatter(*samples.T, s=8, facecolor=COLORS[3], edgecolor="white",
               linewidth=.4, zorder=5)


def workflow_functions(fig, left, neutral):
    # Rise-speed history; full-meridian angular profiles at one fixed time.
    x = np.linspace(0., 1., 241)
    theta = 2*np.pi*x
    radius = ((np.sin(theta)/A)**2 + (np.cos(theta)/B)**2)**(-.5)
    curves = (.72*(1-np.exp(-3*x)), radius, .66*np.sin(theta))
    labels = (r"$\widehat U_b$", r"$\widehat r$", r"$\widehat u_t$")
    limits = ((-.08, .85), (.65, 1.02), (-.85, .85))
    for i, (offset, values, label, color, ylim) in enumerate(zip(
            (.677, .502, .327), curves, labels, COLORS, limits)):
        ax = fig.add_axes((left+.015, offset, .192, .115))
        ax.set(xlim=(-.04, 1.68), ylim=ylim)
        ax.axis("off")
        baseline = 0. if i != 1 else .66
        ax.plot([0, 1.02], [baseline, baseline], color=neutral, lw=.55, alpha=.6)
        ax.plot(x, values, color=color, lw=1.6)
        ax.text(1.10, sum(ylim)/2, label, color=neutral, ha="left", va="center", fontsize=10)
        ticks = ((0., "0"), (1., r"$t$")) if i == 0 else (
            (0., "0"), (.5, r"$\pi$"), (1., r"$2\pi$"))
        for position, text in ticks:
            ax.text(position, ylim[0], text, color=neutral, ha="center", va="top", fontsize=8)


def workflow_mesh(fig, left, neutral):
    ax = axis(fig, (left+.010, .30, .200, .50), (-1.65, 1.65), (-1.30, 1.60))
    liquid_mesh(ax, neutral)
    ax.text(0, 0, "Void", color=neutral, ha="center", va="center", fontsize=10)
    arrow(ax, (A, 0), (A+.48, 0), COLORS[2], lw=1.5, scale=9)
    arrow(ax, (0, 1.52), (0, 1.03), neutral, lw=1.4, scale=9)
    ax.text(-.12, 1.45, r"$-\widehat U_b$", color=neutral,
            ha="right", va="center", fontsize=9)


def workflow_transport(fig, left, neutral):
    ax = axis(fig, (left+.010, .30, .200, .50), (-1.65, 1.65), (-1.30, 1.60))
    # A single conceptual surface layer opens into a downstream species wake.
    normals = np.column_stack((np.sin(ANGLE)/A, np.cos(ANGLE)/B))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    layer = ELLIPSE + (.045+.055*(1-np.cos(ANGLE))/2)[:, None]*normals
    y = np.linspace(0., -1.30, 181)
    lower = layer[180:361]
    layer_x = np.interp(y, lower[::-1, 1], lower[::-1, 0], left=0.)
    wake_x = np.maximum(layer_x, .13+.68*np.exp(-(y/.65)**2))
    front = np.vstack((layer[540:], layer[1:181]))
    right = np.column_stack((wake_x, y))
    left = np.column_stack((-wake_x[::-1], y[::-1]))
    ax.add_patch(Polygon(np.vstack((front, right[1:], left)),
                         facecolor=COLORS[1], alpha=.40, edgecolor="none", zorder=2))
    ax.plot(*np.vstack((front, right[1:])).T, color=COLORS[1], lw=.9, zorder=3)
    ax.plot(*left.T, color=COLORS[1], lw=.9, zorder=3)
    ax.add_patch(Ellipse((0, 0), 2*A, 2*B,
                         facecolor=fig.get_facecolor(), edgecolor="none", zorder=3))
    ax.plot(*ELLIPSE.T, color=COLORS[0], lw=1.6, zorder=4)
    ax.text(0, 0, "Void", color=neutral, ha="center", va="center", fontsize=10)
    for sign in (-1, 1):
        # A schematic curved flow arrow, not a computed streamline.
        ax.add_patch(FancyArrowPatch((sign*.95, 1.18), (sign*1.10, -.95),
                                    connectionstyle=f"arc3,rad={-sign*.32}",
                                    arrowstyle="-|>", mutation_scale=10,
                                    lw=1.5, color=COLORS[0], zorder=6))
    ax.text(.35, -1.08, r"$c$", color=neutral, fontsize=11, ha="center",
            bbox=dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=.4))


def draw_workflow(geometry, use_tex=True):
    apply_style(use_tex)
    fig = plt.figure(figsize=(WIDTH_MM["double"]/25.4,
                              HEIGHT_MM["bubble-hybrid-workflow"]/25.4))
    neutral = plt.rcParams["text.color"]
    lefts = (.020, .267, .513, .760)
    width = .220
    titles = ("1. Two-phase VOF", "2. Regression",
              "3. Liquid mesh", "4. Transport solve")
    descriptions = (
        (r"Rise speed $U_b$", r"Surface points $\mathbf{x}_\Sigma$",
         r"Liquid velocity $\mathbf{u}^+_\Sigma$"),
        ("Fit speed, radius and", "tangential velocity"),
        ("Apply learned", "motion and velocity"),
        ("Compute liquid flow", "and concentration"),
    )
    drawers = (lambda left: workflow_vof(fig, left, geometry),
               lambda left: workflow_functions(fig, left, neutral),
               lambda left: workflow_mesh(fig, left, neutral),
               lambda left: workflow_transport(fig, left, neutral))
    for left, title, lines, draw in zip(lefts, titles, descriptions, drawers):
        card = FancyBboxPatch((left, .048), width, .902,
                              transform=fig.transFigure, boxstyle="round,pad=.003,rounding_size=.008",
                              facecolor="none", edgecolor=neutral, lw=.6, alpha=.35,
                              zorder=0)
        fig.add_artist(card)
        heading = r"\textbf{" + title + "}" if use_tex else title
        fig.text(left+width/2, .865, heading, color=neutral,
                 ha="center", va="center", fontsize=10)
        for y, line in zip((.244, .170, .096), lines):
            fig.text(left+width/2, y, line, color=neutral,
                     ha="center", va="center", fontsize=9)
        draw(left)
    for left, right in zip(lefts[:-1], lefts[1:]):
        fig.add_artist(FancyArrowPatch((left+width+.002, .53), (right-.002, .53),
                                      transform=fig.transFigure, arrowstyle="-|>",
                                      mutation_scale=8, color=neutral, lw=1.1))
    return fig


def polar_geometry(theta):
    er = np.array([np.sin(theta), np.cos(theta)])
    etheta = np.array([np.cos(theta), -np.sin(theta)])
    inv_r2 = (np.sin(theta)/A)**2 + (np.cos(theta)/B)**2
    radius = inv_r2**(-.5)
    derivative = -radius**3 * np.sin(theta)*np.cos(theta)*(1/A**2-1/B**2)
    tangent = derivative*er + radius*etheta
    tangent /= np.linalg.norm(tangent)
    outward = radius*er-derivative*etheta
    outward /= np.linalg.norm(outward)
    return radius, er, tangent, outward


def draw_coordinates(use_tex=True):
    apply_style(use_tex)
    fig = plt.figure(figsize=(WIDTH_MM["double"]/25.4,
                              HEIGHT_MM["bubble-interface-coordinates"]/25.4))
    neutral = plt.rcParams["text.color"]
    box = dict(facecolor=fig.get_facecolor(), edgecolor="none", pad=1)
    fig.text(.255, .94, "Meridional geometry", ha="center", color=neutral)
    fig.text(.755, .94, "Local boundary velocity", ha="center", color=neutral)

    ax = axis(fig, (.035, .285, .445, .605), (-1.40, 1.55), (-1.14, 1.50))
    ax.add_patch(Ellipse((0, 0), 2*A, 2*B, color=COLORS[1], alpha=.10, lw=0))
    ax.plot(*ELLIPSE.T, color=COLORS[0], lw=1.6)
    ax.plot([0, 0], [-1.06, 1.40], color=neutral, lw=.7, ls=(0, (3, 3)), alpha=.65)
    arrow(ax, (0, 0), (0, 1.38), neutral, lw=1.1)
    ax.text(.08, 1.34, r"$+\mathbf{e}_z$", color=neutral, ha="left", va="center")
    ax.text(-1.23, 1.05, "Liquid\n" + r"$\Omega^+(t)$", color=neutral, ha="left", va="center")
    ax.text(-.32, -.38, "Gas\n" + r"$\Omega^-(t)$", color=neutral, ha="center", va="center")
    ax.text(.08, -1.03, "Symmetry axis", color=neutral, fontsize=10, ha="left")

    theta = .80
    radius, er, tangent, outward = polar_geometry(theta)
    point = radius*er
    ax.scatter([0, point[0]], [0, point[1]], s=10, color=neutral, zorder=7)
    ax.plot([0, point[0]], [0, point[1]], color=neutral, lw=1.1, zorder=6)
    ax.text(.43, -.10, r"$r(t,\vartheta)$", color=neutral, ha="center", bbox=box, zorder=8)
    arc_theta = np.linspace(0, theta, 41)
    arc = .32*np.column_stack((np.sin(arc_theta), np.cos(arc_theta)))
    ax.plot(*arc.T, color=neutral, lw=.9, zorder=7)
    arrow(ax, arc[-4], arc[-1], neutral, lw=.9, scale=7)
    ax.text(.17, .46, r"$\vartheta$", color=neutral, ha="center", va="center", bbox=box, zorder=9)
    arrow(ax, point, point+.52*er, COLORS[2], lw=1.5)
    ax.text(*(point+.60*er), r"$\mathbf{e}_r$", color=neutral, ha="center", va="center")
    ax.text(-.18, -.13, r"$O$", color=neutral, ha="center")

    local = axis(fig, (.535, .285, .435, .605), (-1.30, 1.36), (-1.35, 1.10))
    s = np.linspace(-2.5, 2.5, 361)
    boundary = s[:, None]*tangent + .13*s[:, None]**2*(-outward)
    gas_patch = np.vstack((boundary, [[4., -4.], [-4., -4.], [-4., 4.]]))
    local.add_patch(Polygon(gas_patch, facecolor=COLORS[1], alpha=.10, edgecolor="none", zorder=0))
    local.plot(*boundary.T, color=COLORS[0], lw=1.6, zorder=2)
    local.scatter([0], [0], s=12, color=neutral, zorder=8)
    local.text(.71, .75, "Liquid\n" + r"$\Omega^+(t)$", color=neutral, ha="center", va="center")
    local.text(-.74, -.77, "Gas\n" + r"$\Omega^-(t)$", color=neutral, ha="center", va="center")

    ut, un = .84, .47
    vt, vn = ut*tangent, un*outward
    velocity = vt+vn
    local.plot([vt[0], velocity[0]], [vt[1], velocity[1]],
                color=neutral, lw=.8, ls=(0, (3, 2)), alpha=.65)
    local.plot([vn[0], velocity[0]], [vn[1], velocity[1]],
                color=neutral, lw=.8, ls=(0, (3, 2)), alpha=.65)
    arrow(local, (0, 0), vt, COLORS[1], lw=1.6)
    arrow(local, (0, 0), vn, COLORS[2], lw=1.6)
    arrow(local, (0, 0), velocity, neutral, lw=1.6)
    local.text(vt[0]+.23, vt[1]-.03, r"$u_t\mathbf{t}_{\Sigma}$", color=neutral,
                ha="center", va="center", bbox=box, zorder=9)
    local.text(vn[0]-.26, vn[1]+.11, r"$u_n\mathbf{n}_{\Sigma}$", color=neutral,
                ha="center", va="center", bbox=box, zorder=9)
    local.text(velocity[0]+.05, velocity[1]-.22, r"$\mathbf{u}_{\mathrm{rel}}$", color=neutral,
                ha="center", va="center", bbox=box, zorder=9)
    local.text(-.55, .65, r"$\Sigma(t)$", color=neutral, ha="center", bbox=box)
    fig.text(.255, .255, r"$\vartheta$ increases from $+\mathbf{e}_z$; $0\leq\vartheta\leq\pi$",
             color=neutral, ha="center", fontsize=10)
    fig.text(.755, .255, r"$\mathbf{t}_{\Sigma}$: increasing $\vartheta$; $\mathbf{n}_{\Sigma}$: into liquid",
             color=neutral, ha="center", fontsize=10)
    fig.text(.5, .155,
             r"$\mathbf{u}_{\mathrm{rel}}=\mathbf{u}_{\mathrm{lab}}|_{\Sigma}-U_b(t)\mathbf{e}_z"
             r"=u_t\mathbf{t}_{\Sigma}+u_n\mathbf{n}_{\Sigma}$",
             color=neutral, ha="center")
    fig.text(.5, .085,
             r"$u_n=(\partial_t r\,\mathbf{e}_r)\cdot\mathbf{n}_{\Sigma}$"
             + ": normal motion comes from the shape",
             color=neutral, ha="center", fontsize=10)
    fig.text(.5, .028, "Conceptual axisymmetric sketch in the translating bubble frame",
             color=neutral, ha="center", fontsize=9)
    return fig


def smoke_test(output, use_tex):
    with plt.style.context("default"):
        apply_style(use_tex)
        fig, ax = plt.subplots(figsize=(4.5, 1.5))
        ax.axis("off")
        ax.text(.03, .55, r"$r(t,\vartheta),\ U_b(t),\ \mathbf{n}_{\Sigma},\ \mathbf{t}_{\Sigma}$"
                + "\n" + r"$u_n=(\partial_t r\,\mathbf{e}_r)\cdot\mathbf{n}_{\Sigma}$")
        for ext in ("svg", "png"):
            fig.savefig(output/f"smoke.{ext}", dpi=DPI)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/media"))
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    args = parser.parse_args()
    use_tex = resolve_renderer("mathtext" if args.no_tex else args.renderer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="hybrid-renderer-") as directory:
        smoke_test(Path(directory), use_tex)
    geometry = vof_geometry()
    radius, er, tangent, outward = polar_geometry(.80)
    assert abs(np.dot(tangent, outward)) < 1e-14
    assert np.dot(er, outward) > 0
    assert np.dot(tangent, np.array([np.cos(.80), -np.sin(.80)])) > 0
    for stem, draw in (("bubble-hybrid-workflow", lambda: draw_workflow(geometry, use_tex)),
                       ("bubble-interface-coordinates", lambda: draw_coordinates(use_tex))):
        for ext in ("svg", "pdf", "png"):
            save_theme_variants(draw, args.output_dir/f"{stem}.{ext}", dpi=DPI)
    print(json.dumps({"renderer": "MathText" if not use_tex else "LaTeX",
                      "width_mm": WIDTH_MM["double"], "height_mm": HEIGHT_MM,
                      "tangent_dot_outward_normal": float(np.dot(tangent, outward)),
                      "radial_dot_outward_normal": float(np.dot(er, outward)),
                      "vof_geometry": geometry[-1]}, indent=2))


if __name__ == "__main__":
    main()
