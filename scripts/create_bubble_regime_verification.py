#!/usr/bin/env python3
"""Overlay freshly digitized data on the original raster using saved calibration.

The original image remains unchanged in both themes for source verification.
This is an audit figure, not the restyled course regime map.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from figure_style import DPI, WIDTH_MM, figure_style, add_renderer_argument, resolve_renderer
from figure_utils import save_theme_variants


def load_source():
    path = ROOT / "datasets/path_shape_regimes/tripathi2015_digitization"
    image = plt.imread(path / "figure1-source.png")
    with (path / "points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    data = np.array([[float(row["Ga"]),float(row["Eo"])] for row in rows])
    audit = json.loads((path / "audit.json").read_text())
    calibration = audit["log10_coordinate_coefficients"]
    height,width = image.shape[:2]
    # Pixel coordinates index centers; imshow extent describes outside edges.
    extent = (np.polyval(calibration["Ga"],-0.5),
              np.polyval(calibration["Ga"],width-0.5),
              np.polyval(calibration["Eo"],height-0.5),
              np.polyval(calibration["Eo"],-0.5))
    # Independently confirm that CSV values invert to their measured pixels.
    pixels = np.array([[float(row["pixel_x"]),float(row["pixel_y"])] for row in rows])
    back = np.column_stack([(np.log10(data[:,i])-calibration[key][1])/calibration[key][0]
                            for i,key in enumerate(("Ga","Eo"))])
    if np.max(np.abs(back-pixels)) > 0.001:
        raise ValueError("CSV coordinates do not match saved pixel calibration.")
    return image,data,extent


def draw(image,data,extent, *, use_tex=True):
    # The paired-export helper isolates this callback and its export in a context.
    plt.rcParams.update({**figure_style(use_tex=use_tex),
                         "axes.prop_cycle":cycler(color=plt.colormaps["tab10"].colors),
                         "svg.hashsalt":"bubble-regime-verification"})
    fig,ax = plt.subplots(figsize=(WIDTH_MM["double"]/25.4,160/25.4))
    # Native raster aspect ratio is preserved; coordinate axes show physical values.
    fig.subplots_adjust(left=0.12,right=0.98,bottom=0.23,top=0.91)
    ax.set_box_aspect(image.shape[0]/image.shape[1])
    ax.imshow(image,extent=extent,origin="upper",aspect="auto",interpolation="none")
    logdata = np.log10(data)
    # Thin open C3 rings leave the original marker centers and outlines visible.
    ax.scatter(logdata[:,0],logdata[:,1],s=75,facecolors="none",edgecolors="#d62728",
               linewidths=0.7,zorder=4,label="Fresh extraction (104 points)")
    ax.set_xlim(extent[0],extent[1]); ax.set_ylim(extent[2],extent[3])
    ax.set_xticks(np.log10([10,100,500]),labels=[r"$10$",r"$100$",r"$500$"])
    ax.set_yticks(np.log10([0.1,1,10,100]),labels=[r"$0.1$",r"$1$",r"$10$",r"$100$"])
    ax.set_xlabel(r"Galilei number $Ga$")
    ax.set_ylabel(r"E\"otv\"os number $Eo$")
    ax.set_title("Verification against the published Figure 1",pad=12)
    fig.legend(loc="lower center",bbox_to_anchor=(0.55,0.115),frameon=False,fontsize=10)
    fig.text(0.55,0.08,"Background: Tripathi, Sahu and Govindarajan (2015), Figure 1.",
             ha="center",fontsize=9)
    fig.text(0.55,0.05,"Original image retained unchanged; rings use the saved axis calibration.",
             ha="center",fontsize=9)
    fig.text(0.55,0.02,r"Obscured location near $(Ga,Eo)=(50,20)$ is excluded.",
             ha="center",fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"notebooks/media")
    parser.add_argument("--png",action="store_true")
    add_renderer_argument(parser)
    args = parser.parse_args()
    use_tex = resolve_renderer(args.renderer)
    source = load_source()
    for extension in (["svg","png"] if args.png else ["svg"]):
        paths = save_theme_variants(lambda:draw(*source, use_tex=use_tex),
                                    args.output_dir/f"bubble-regime-verification.{extension}",dpi=DPI)
        for path in paths.values():
            print(path)


if __name__ == "__main__":
    main()
