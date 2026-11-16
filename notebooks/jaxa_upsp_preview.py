"""Plot the 200-frame upper-wing uPSP record from JAXA DPW-8 Case 3.

Run python notebooks/jaxa_upsp_preview.py from the course checkout.
"""

import argparse
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.tri import Triangulation

from figure_style import DPI, WIDTH_MM, add_renderer_argument, figure_style, resolve_renderer
from figure_utils import save_theme_variants
from jaxa_upsp_data import STATISTICS_MEMBER, fetch_jaxa_upsp, load_upper_wing_snapshots


def read_upper_wing(archive_path):
    """Select animation frames and compute statistics on the complete sequence."""
    xyz, triangles, times, pressure = load_upper_wing_snapshots(archive_path)
    pressure[~np.isfinite(pressure).all(axis=1)] = np.nan
    std = pressure.std(axis=1)
    mask = ~np.isfinite(std[triangles]).all(axis=1)
    mesh = Triangulation(*xyz[:, :2].T, triangles, mask=mask)
    return mesh, 1000 * times[::5], pressure[:, ::5].copy(), std


def read_reference_mean(archive_path, mesh):
    """Load the separately published mean pressure coefficient on the same grid."""
    with ZipFile(archive_path) as archive, archive.open(STATISTICS_MEMBER) as stream:
        list(islice(stream, 8))
        data = np.loadtxt(islice(stream, len(mesh.x)))
    if not np.allclose(data[:, :2], np.column_stack([mesh.x, mesh.y]), atol=5e-5, rtol=0):
        raise ValueError("The JAXA mean and snapshot meshes differ.")
    mean = data[:, 3]
    mean[mean == -999] = np.nan
    return mean


def make_figure(panels, heading):
    width = WIDTH_MM["double"] / 25.4 if len(panels) == 2 else 4.2
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.3), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.15, top=0.84, wspace=0.25)
    fig.suptitle(heading, y=0.96)
    artists = []
    for index, (ax, panel) in enumerate(zip(axes[0], panels)):
        mesh, values, title, label, cmap, limits, extend = panel
        artist = ax.tripcolor(mesh, values, cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set(aspect="equal", xlabel=r"$x/c_{\mathrm{MAC}}$", title=title)
        if index == 0:
            ax.set_ylabel(r"$y/c_{\mathrm{MAC}}$")
        colorbar = fig.colorbar(
            artist, ax=ax, location="bottom", pad=0.24, aspect=25, extend=extend
        )
        colorbar.set_label(label)
        artists.append(artist)
    return fig, artists


def render(output, renderer):
    style = figure_style(use_tex=resolve_renderer(renderer))
    style["axes.prop_cycle"] = mpl.rcParamsDefault["axes.prop_cycle"]
    archive_path = fetch_jaxa_upsp()
    mesh, times, pressure, std = read_upper_wing(archive_path)
    mean = read_reference_mean(archive_path, mesh)
    mean_mesh = Triangulation(
        mesh.x, mesh.y, mesh.triangles,
        mask=~np.isfinite(mean[mesh.triangles]).all(axis=1),
    )
    print(f"Upper wing: {std.size} nodes, {np.isfinite(std).sum()} valid across the record")
    cp_limits = np.nanpercentile(mean, [1, 99])
    std_limit = np.nanpercentile(std, 99)
    fluctuation_limit = np.nanpercentile(np.abs(pressure), 99)
    case = r"JAXA CRM upper wing, $\alpha=4.71^\circ$"

    def statistics():
        with mpl.rc_context(style):
            return make_figure([
                (mean_mesh, mean, "JAXA temporal mean", r"$\overline{C_p}$",
                 "viridis", cp_limits, "both"),
                (mesh, std, "Temporal standard deviation", r"$\sigma(C_p)$", "magma",
                 (0, std_limit), "max"),
            ], case)[0]

    save_theme_variants(statistics, output / "jaxa-upsp-statistics.png", dpi=DPI)
    triangles = mesh.get_masked_triangles()
    for theme in ("light", "dark"):
        theme_style = ["default"]
        if theme == "dark":
            theme_style.extend(["dark_background", {
                "figure.facecolor": "#111111", "axes.facecolor": "#111111"
            }])
        with plt.style.context(theme_style), mpl.rc_context(style):
            fig, artists = make_figure([
                (mesh, pressure[:, 0], "Pressure fluctuation", r"$C_p'$",
                 "RdBu_r", (-fluctuation_limit, fluctuation_limit), "both"),
            ], case)
            writer = FFMpegWriter(fps=12, codec="libx264", extra_args=[
                "-pix_fmt", "yuv420p", "-movflags", "+faststart"
            ])
            with writer.saving(fig, output / f"jaxa-upsp-snapshots-{theme}.mp4", dpi=120):
                for frame, time in enumerate(times):
                    fig.suptitle(case + f", {time:.0f} ms", y=0.96)
                    artists[0].set_array(pressure[triangles, frame].mean(axis=1))
                    writer.grab_frame()
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/jaxa_upsp_preview"))
    add_renderer_argument(parser)
    args = parser.parse_args()
    render(args.output, args.renderer)
