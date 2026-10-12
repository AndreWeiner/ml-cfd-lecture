"""Save Matplotlib figures for Quarto's light and dark themes."""
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_theme_variants(
    draw: Callable[[], Figure],
    filename: Union[str, Path],
    *,
    dpi: int = 160,
    dark_background: Optional[str] = "#111111",
) -> Dict[str, Path]:
    """Call draw twice and save <stem>-light.<ext> and <stem>-dark.<ext>.

    draw must create and return a fresh Figure on each call, without calling
    show() or savefig(). Compute data, random samples, and model predictions
    before calling this function so both drawings use the same results.

    Each drawing starts from Matplotlib defaults. The dark version additionally
    applies dark_background. Its figure/axes backgrounds match the course site
    unless dark_background=None is passed (the unmodified style uses black).
    The filename extension selects the format; PNG is used when omitted.
    Existing output files are replaced. Figures created by draw are closed,
    including on failure; pre-existing pyplot figures and rcParams are retained.
    """
    filename = Path(filename)
    if not filename.suffix:
        filename = filename.with_suffix(".png")
    filename.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for mode in ("light", "dark"):
        styles = ["default"]
        if mode == "dark":
            styles.append("dark_background")
            if dark_background is not None:
                styles.append({"figure.facecolor": dark_background,
                               "axes.facecolor": dark_background})
        existing = set(plt.get_fignums())
        fig = None
        with plt.style.context(styles):
            try:
                fig = draw()
                if not isinstance(fig, Figure):
                    raise TypeError("draw() must return a new matplotlib.figure.Figure")
                if getattr(fig, "number", None) in existing:
                    raise ValueError("draw() must create a fresh figure for each theme")
                path = filename.with_name(f"{filename.stem}-{mode}{filename.suffix}")
                fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(),
                            edgecolor=fig.get_edgecolor(), transparent=False,
                            bbox_inches=None)
                paths[mode] = path
            finally:
                for number in set(plt.get_fignums()) - existing:
                    plt.close(number)
    return paths
