"""Generate lecture time-factor plots: python scripts/plot_bubble_time_factors.py OUTPUT."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from figure_style import WIDTH_MM, figure_style

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
# The reference archive has snapshot indices 0 through 99, spaced by 0.15.
T = 99 * 0.15
t = np.linspace(0, T, 400)
for theme, background, foreground in [
    ("light", "#ffffff", "#222222"),
    ("dark", "#111111", "#eeeeee"),
]:
    with plt.rc_context({
        **figure_style(),
        "axes.prop_cycle": cycler(color=plt.get_cmap("tab10").colors),
        "figure.facecolor": background,
        "axes.facecolor": background,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "text.color": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
    }):
        fig, (ax, angular) = plt.subplots(1, 2, figsize=(WIDTH_MM["double"] / 25.4, 84 / 25.4),
                               layout="constrained")
        ax.plot(t, np.log1p(t) / np.log1p(T), label=r"$q(\tilde{t}) = \log(1+\tilde{t})/\log(1+T)$")
        ax.plot(t, t / (1 + t), "--", label=r"$s(\tilde{t}) = \tilde{t}/(1+\tilde{t})$")
        ax.set(xlabel=r"$\tilde{t}$", ylabel=r"$q(\tilde{t}),\ s(\tilde{t})$",
               xlim=(0, T), ylim=(0, 1.05), xticks=[0, 3, 6, 9, 12, T])
        ax.set_xticklabels(["0", "3", "6", "9", "12", "14.85"])
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=False)
        angle = np.linspace(-np.pi / 8, 2 * np.pi + np.pi / 8, 400)
        angular.plot(angle, np.sin(angle), label=r"$\sin(\vartheta)$")
        angular.plot(angle, np.cos(angle), "--", label=r"$\cos(\vartheta)$")
        angular.set(xlabel=r"$\vartheta$", xlim=(angle[0], angle[-1]),
                    ylim=(-1.12, 1.12), xticks=np.arange(5) * np.pi / 2,
                    xticklabels=[r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
        angular.grid(alpha=0.25)
        for boundary in (0, 2 * np.pi):
            angular.axvline(boundary, color=foreground, lw=0.8, ls=":", alpha=0.65)
        angular.legend(loc="lower left", frameon=False)
        fig.savefig(output / f"bubble-time-factors-{theme}.svg")
        plt.close(fig)
