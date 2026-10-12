"""Run `python notebooks/figure_example.py` to regenerate the theme demo."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_utils import save_theme_variants


def main():
    # Compute once; only the drawing function runs twice.
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    def draw():
        fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        ax.plot(x, y, label="sin(x)")
        ax.set(xlabel="x", ylabel="Amplitude", xlim=(0, 2 * np.pi), ylim=(-1.2, 1.2))
        ax.legend()
        return fig

    destination = Path(__file__).resolve().parent / "media" / "theme-example.png"
    for mode, path in save_theme_variants(draw, destination).items():
        print(f"{mode}: {path}")


if __name__ == "__main__":
    main()
