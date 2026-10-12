"""Shared figure settings for the course's theory notebooks."""
WIDTH_MM = {"single": 89.0, "double": 180.0}
DPI = 300
USE_TEX = True  # Explicit opt-out only, e.g. when running without a TeX installation.


def figure_style(*, use_tex=USE_TEX):
    return {
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
    }
