#!/usr/bin/env python3
"""Plot the fresh raster digitization of Tripathi et al. (2015), Fig. 1.

See datasets/path_shape_regimes/tripathi2015_digitization/README.md for the
source's inconsistent II/III labels and the independently extracted pixel data.
No coordinates are rounded to a presumed simulation grid or interpolated.

Reproduce: .venv-quarto/bin/python scripts/create_bubble_regime_map.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from figure_style import DPI, WIDTH_MM, figure_style
from figure_utils import save_theme_variants

# Physical names/numbering follow the caption and main Results text, rather
# than the inconsistent II/III numerals printed inside the original image.
REGIMES = (
    ("I", "Axisymmetric", "o"),
    ("II", "Skirted, non-oscillatory", "^"),
    ("III", "Oscillatory path", "v"),
    ("IV", "Peripheral breakup", "s"),
    ("V", "Central breakup", "D"),
)
COLORS = tuple(plt.colormaps["tab10"].colors)
HEIGHT_MM = 130.0


def load_data():
    data_file = ROOT / "datasets/path_shape_regimes/tripathi2015_digitization/points.csv"
    with data_file.open() as handle:
        rows = list(csv.DictReader(handle))
    data = [np.array([[float(row["Ga"]), float(row["Eo"])] for row in rows
                      if row["regime"] == numeral]) for numeral,_,_ in REGIMES]
    for row, values in zip(REGIMES, data):
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"Expected Ga,Eo columns in regime {row[0]}")
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Nonpositive or nonfinite coordinates in regime {row[0]}")
    return data


def draw(data, *, use_tex=True):
    # Apply the categorical cycle AFTER the light/dark styles in the helper.
    style = {**figure_style(use_tex=use_tex),
             "axes.prop_cycle": cycler(color=COLORS),
             "svg.hashsalt": "bubble-regime-map"}
    # save_theme_variants already isolates each callback in a style context.
    # Keep TeX's font settings active through savefig, not only artist creation.
    plt.rcParams.update(style)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4,
                                        HEIGHT_MM / 25.4))
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.265, top=0.91)
        for index, ((numeral, name, marker), values) in enumerate(zip(REGIMES, data)):
            ax.scatter(values[:, 0], values[:, 1], s=28, marker=marker,
                       facecolors="none", edgecolors=COLORS[index], linewidths=1.1,
                       label=f"{numeral}: {name}", zorder=3)
        ax.set(xscale="log", yscale="log", xlim=(6, 600), ylim=(0.07, 280))
        # Small padding retains approximate digitized values beyond nominal limits.
        ax.set_xticks([10, 100, 500], labels=[r"$10$", r"$100$", r"$500$"])
        ax.set_yticks([0.1, 1, 10, 100], labels=[r"$0.1$", r"$1$", r"$10$", r"$100$"])
        ax.set_xlabel(r"$Ga$")
        ax.set_ylabel(r"$Eo$")
        ax.grid(which="major", color=plt.rcParams["text.color"], alpha=0.14, linewidth=0.6)
        ax.tick_params(which="both", direction="out")
        ax.set_title("Digitized simulation points: Tripathi et al. (2015)", pad=12)
        fig.legend(loc="lower center", bbox_to_anchor=(0.54, 0.025), ncol=2,
                   frameon=False, fontsize=10, columnspacing=2.0,
                   handletextpad=0.5, borderaxespad=0, labelspacing=0.65)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks" / "media")
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    parser.add_argument("--png", action="store_true", help="Also export 300-DPI preview PNGs.")
    args = parser.parse_args()
    use_tex = resolve_renderer("mathtext" if args.no_tex else args.renderer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    # A real SVG export containing representative equations tests the full TeX path.
    with plt.rc_context(figure_style(use_tex=use_tex)):
        fig, ax = plt.subplots(figsize=(3, 1))
        ax.text(0.05, 0.5, r"$Ga=\rho_l\sqrt{gR^3}/\mu_l$, $Eo=\rho_l gR^2/\sigma$")
        ax.set_axis_off()
        try:
            fig.savefig(args.output_dir / ".regime-map-renderer-check.svg")
        finally:
            plt.close(fig)
    (args.output_dir / ".regime-map-renderer-check.svg").unlink()
    for extension in (["svg", "png"] if args.png else ["svg"]):
        paths = save_theme_variants(lambda: draw(data, use_tex=use_tex),
                                    args.output_dir / f"bubble-regime-map.{extension}", dpi=DPI)
        for path in paths.values():
            print(path)
    print(f"Plotted {sum(len(values) for values in data)} digitized points; "
          f"renderer={'MathText' if not use_tex else 'LaTeX'}.")


if __name__ == "__main__":
    main()
