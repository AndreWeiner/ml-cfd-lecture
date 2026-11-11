"""Shared figure settings for the course's theory notebooks."""
WIDTH_MM = {"single": 89.0, "double": 180.0}
DPI = 300


def figure_style(*, use_tex=True):
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


def resolve_renderer(mode, *, preamble=None):
    """Return whether to use LaTeX, after checking the Agg render path."""
    if mode == "mathtext":
        return False
    if mode not in ("auto", "latex"):
        raise ValueError("Renderer must be auto, latex, or mathtext.")

    import shutil
    import warnings

    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    style = figure_style(use_tex=True)
    if preamble is not None:
        style["text.latex.preamble"] = preamble
    try:
        if shutil.which("latex") is None:
            raise RuntimeError("The latex executable could not be found on PATH.")
        with mpl.rc_context(style):
            figure = Figure(figsize=(2, 1))
            figure.text(0.1, 0.5, r"Plot label $\alpha = \frac{1}{2}$")
            FigureCanvasAgg(figure).draw()
    except (RuntimeError, OSError) as error:
        if mode == "latex":
            raise RuntimeError(
                f"LaTeX plot rendering failed. Select mathtext to use Matplotlib's "
                f"built-in renderer.\n{error}"
            ) from error
        warnings.warn(
            f"LaTeX plot rendering is unavailable. Using MathText.\n{error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def add_renderer_argument(parser, *, default="latex", legacy_no_tex=False):
    """Add the shared renderer option and optionally retain the old spelling."""
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--renderer", choices=("auto", "latex", "mathtext"), default=default,
        help=f"Plot text renderer (default: {default}).",
    )
    if legacy_no_tex:
        options.add_argument("--no-tex", action="store_true", help="Alias for --renderer mathtext.")
