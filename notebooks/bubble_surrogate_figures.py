"""Paired course diagnostics, using identical data and limits in both themes."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from figure_style import WIDTH_MM, DPI, figure_style
from figure_utils import save_theme_variants


def save_diagnostics(models, histories, times, interface, stem):
    sample = interface[interface.snapshot == 5]
    angle = np.linspace(0, np.pi, 201)
    time = np.linspace(0, times.t.max(), 301)
    with torch.no_grad():
        ub = models["ub"](torch.tensor(time[:, None])).numpy().ravel()
        features = torch.tensor(np.column_stack((np.full_like(angle, sample.t.iloc[0]), angle)))
        radius = models["r"](features).numpy().ravel()
        ut = models["ut"](features).numpy().ravel()

    def draw():
        plt.rcParams.update(figure_style(use_tex=True))
        plt.rcParams["axes.prop_cycle"] = cycler(color=plt.colormaps["tab10"].colors)
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM["double"]/25.4, 135/25.4))
        ax = axes[0, 0]
        ax.scatter(times.t, times.ub, s=8, color="C0", label="Reference")
        ax.plot(time, ub, color="C1", label="Prediction")
        ax.set(xlabel=r"$\tilde{t}$", ylabel=r"$\tilde{U}_b$", title="Rise speed")
        ax.legend(fontsize=9)
        ax = axes[0, 1]
        for kind, history in histories.items():
            ax.plot(np.arange(1, len(history)+1), history[:, 1], label=kind)
        ax.set(xlabel="Validation checkpoint", ylabel="Scaled validation MSE", yscale="log", title="Validation histories")
        ax.legend(fontsize=9)
        for ax, target, values, ylabel in zip(axes[1], ["r", "ut"], [radius, ut], [r"$\tilde{r}$", r"$\tilde{u}_t$"]):
            ax.scatter(sample.theta, sample[target], s=10, color="C0", label="Reference")
            ax.plot(angle, values, color="C1", label="Prediction")
            ax.set(xlabel=r"$\vartheta$", ylabel=ylabel, xlim=(0, np.pi),
                   title=r"Held out: $\tilde{t}="+f"{sample.t.iloc[0]:.2f}"+"$")
            ax.set_xticks([0, np.pi/2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
        fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.93, hspace=.62, wspace=.42)
        return fig

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        save_theme_variants(draw, stem.with_suffix("."+extension), dpi=DPI)
