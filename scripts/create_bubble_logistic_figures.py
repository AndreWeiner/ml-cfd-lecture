#!/usr/bin/env python3
"""Audited logistic-regression example using the fresh I/III bubble digitization.

Reproduce from the repository root into an initially empty directory::

    MPLCONFIGDIR=/tmp/bubble-logistic-mpl .venv-quarto/bin/python \
        scripts/create_bubble_logistic_figures.py --output-dir /tmp/bubble-logistic-new

Copy inspected SVGs to notebooks/media and docs/notebooks/media. The default
results file is figures/bubble-classification/logistic-results.json. Importing
this module performs no training, file writes, or plotting configuration.

The fit uses all 47 selected training rows, with no held-out evaluation and no
probability-calibration claim. Unregularized logistic regression on this strictly
separable sample has no finite maximum-likelihood estimate. The 2000-step fit
is a deliberately finite teaching example, not an optimizer-convergence claim.
LaTeX is the default renderer; glyph outlines in SVG retain editable geometry.
Shared 180-mm width and 11-pt typography are used at approximately 620-px placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from create_bubble_classification_figures import (
    COLORS, DATA_FILE, ROOT, fit_log_minmax, load_bubble_data, transform_features,
)


def sigmoid(values):
    """Overflow-safe sigmoid, preserving the input shape (including scalars)."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def mean_bce_and_gradient(augmented, labels, weights):
    """Mean BCE and its full-batch gradient; the last X column contains ones."""
    z = augmented @ weights
    # This standard logaddexp expression avoids log(0) and exp overflow.
    loss = np.mean(np.logaddexp(0.0, z) - labels * z)
    gradient = augmented.T @ (sigmoid(z) - labels) / len(labels)
    return float(loss), gradient


def train_logistic(features, labels, *, learning_rate=1.0, steps=2000):
    """Run a fixed number of mean-BCE gradient steps from exactly zero weights."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 2 or len(features) == 0
            or not np.isfinite(features).all()):
        raise ValueError("Expected a nonempty finite (n, 2) feature matrix.")
    if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected one binary label for each feature row.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("The learning rate must be finite and positive.")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("The number of gradient steps must be a positive integer.")
    augmented = np.column_stack((features, np.ones(len(features))))
    weights = np.zeros(3, dtype=np.float64)
    states, losses = [weights.copy()], []
    for _ in range(steps):
        loss, gradient = mean_bce_and_gradient(augmented, labels, weights)
        losses.append(loss)
        weights -= learning_rate * gradient
        states.append(weights.copy())
    loss, gradient = mean_bce_and_gradient(augmented, labels, weights)
    losses.append(loss)
    probabilities = sigmoid(augmented @ weights)
    log_likelihood = -len(labels) * np.array(losses)
    hard_errors = np.count_nonzero((augmented @ np.array(states).T > 0)
                                  != labels[:, None], axis=0)
    first_zero = np.flatnonzero(hard_errors == 0)
    return {
        "learning_rate": learning_rate, "steps": steps,
        "stopping_rule": "fixed number of full-batch updates; no convergence claim",
        "objective": "mean binary cross-entropy; no regularization",
        "initial_weights": states[0].tolist(), "final_weights": weights.tolist(),
        "weight_states": np.array(states).tolist(), "mean_bce": losses,
        "log_likelihood_sum": log_likelihood.tolist(),
        "likelihood_product": np.exp(log_likelihood).tolist(),
        "final_gradient_norm": float(np.linalg.norm(gradient)),
        "training_errors": int(np.count_nonzero((probabilities > 0.5) != labels)),
        "training_error_trace": hard_errors.tolist(),
        "first_zero_training_error_step": int(first_zero[0]) if len(first_zero) else None,
        "training_probabilities": probabilities.tolist(),
        "minimum_signed_margin": float(np.min((2 * labels - 1) * (augmented @ weights))),
    }


def numerical_checks(features, labels, fit):
    """Independent formulas, central differences, and analytic reference values."""
    augmented = np.column_stack((features, np.ones(len(features))))
    weights = np.array([0.7, -1.2, 0.3])
    loss, gradient = mean_bce_and_gradient(augmented, labels, weights)
    logits = augmented @ weights
    reference_p = 0.5 * (1.0 + np.tanh(logits / 2.0))
    reference_loss = -np.mean(labels * np.log(reference_p)
                              + (1.0 - labels) * np.log1p(-reference_p))
    np.testing.assert_allclose(loss, reference_loss, rtol=1e-13, atol=1e-14)
    epsilon = 1e-5
    numerical_gradient = np.array([
        (mean_bce_and_gradient(augmented, labels, weights + epsilon * unit)[0]
         - mean_bce_and_gradient(augmented, labels, weights - epsilon * unit)[0])
        / (2 * epsilon) for unit in np.eye(3)
    ])
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sigmoid(np.array([-1000., 0., 1000.])), [0., .5, 1.])
    np.testing.assert_allclose(fit["mean_bce"][0], np.log(2.), rtol=1e-14)
    np.testing.assert_allclose(fit["log_likelihood_sum"][0], -len(labels)*np.log(2.), rtol=1e-14)
    np.testing.assert_allclose(fit["likelihood_product"][0], 2.**(-len(labels)), rtol=1e-13, atol=0)
    bound = np.linalg.eigvalsh(augmented.T @ augmented / (4 * len(labels)))[-1]
    assert fit["learning_rate"] < 2 / bound
    assert np.max(np.diff(fit["mean_bce"])) <= 1e-13
    assert fit["minimum_signed_margin"] > 0
    # Strict positive margins supply a separating direction: scaling these
    # coefficients improves every observed-label probability toward one.
    final_weights = np.array(fit["final_weights"])
    scaled_loss = mean_bce_and_gradient(augmented, labels, 2 * final_weights)[0]
    assert scaled_loss < fit["mean_bce"][-1]
    return {
        "central_difference_step": epsilon,
        "gradient_max_absolute_error": float(np.max(abs(gradient - numerical_gradient))),
        "independent_probability_formula_loss_error": float(abs(loss - reference_loss)),
        "gradient_lipschitz_upper_bound": float(bound),
        "largest_mean_bce_increment": float(np.max(np.diff(fit["mean_bce"]))),
        "mean_bce_at_twice_final_weights": scaled_loss,
        "strict_separability_witness": "all final signed training margins are positive",
        "finite_mle_exists": False,
    }


def coefficient_slice(features, labels):
    """Same 181-by-181 coefficient grid for L and log L; fixed bias exactly zero."""
    axis = np.linspace(-8.0, 12.0, 181)
    w1, w2 = np.meshgrid(axis, axis)
    logits = w1[..., None] * features[:, 0] + w2[..., None] * features[:, 1]
    # Equivalent to y*z-logaddexp(0,z), without subtracting large positives.
    log_likelihood = -np.logaddexp(0., (1 - 2 * labels) * logits).sum(axis=-1)
    return w1, w2, np.exp(log_likelihood), log_likelihood


def _configure(use_tex):
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style
    plt.rcParams.update({**figure_style(use_tex=use_tex),
                         "text.latex.preamble": r"\usepackage{amsmath}",
                         "axes.prop_cycle": cycler(color=COLORS),
                         "svg.hashsalt": "bubble-logistic-figures"})


def draw_landscapes(slice_values, *, use_tex=True):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib.ticker import ScalarFormatter
    from figure_style import WIDTH_MM
    _configure(use_tex)
    w1, w2, likelihood, log_likelihood = slice_values
    likelihood_levels = likelihood.max() * np.array([.01, .1, .3, .6, .9])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM["double"] / 25.4, 108 / 25.4))
    fig.subplots_adjust(left=.095, right=.98, bottom=.32, top=.87, wspace=.34)
    for ax, field, title, label in zip(
            axes, (likelihood, log_likelihood),
            ("Likelihood", "Log-likelihood"), (r"$L(\mathbf{w})$", r"$\log(L(\mathbf{w}))$")):
        mesh = ax.pcolormesh(w1, w2, field, shading="nearest", cmap="viridis", rasterized=True)
        # Contour the common log grid in both panels so the discrete geometry
        # is identical too; these are exactly the corresponding L-level sets.
        contours = ax.contour(w1, w2, log_likelihood, levels=np.log(likelihood_levels),
                              colors="white", linewidths=.5, linestyles="solid")
        contours.set_path_effects([effects.Stroke(linewidth=1.0, foreground="#111111"),
                                  effects.Normal()])
        ax.set(xlabel=r"$w_1$", ylabel=r"$w_2$", title=title, xlim=(-8, 12), ylim=(-8, 12))
        ax.set_xticks([-8, 0, 12]); ax.set_yticks([-8, 0, 12])
        ax.set_aspect("equal")
        position = ax.get_position()
        color_ax = fig.add_axes([position.x0, .10, position.width, .035])
        cb = fig.colorbar(mesh, cax=color_ax, orientation="horizontal")
        cb.set_label(label, labelpad=2)
        cb.ax.tick_params(labelsize=9, pad=2)
        if field is likelihood:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0)); cb.formatter = formatter; cb.update_ticks()
    fig.suptitle(r"Same coefficient slice: bias $b=0$", y=.98)
    return fig


def draw_probabilities(features, labels, fit, *, use_tex=True):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from figure_style import WIDTH_MM
    _configure(use_tex)
    fig, (ax, history) = plt.subplots(1, 2, figsize=(WIDTH_MM["double"] / 25.4, 115 / 25.4))
    fig.subplots_adjust(left=.10, right=.98, bottom=.35, top=.87, wspace=.38)
    axis = np.linspace(-1.08, 1.08, 201)
    x1, x2 = np.meshgrid(axis, axis)
    weights = np.array(fit["final_weights"])
    probabilities = sigmoid(weights[0] * x1 + weights[1] * x2 + weights[2])
    mesh = ax.pcolormesh(x1, x2, probabilities, shading="nearest", cmap="viridis",
                         vmin=0., vmax=1., rasterized=True)
    # Stagger one inline value per contour along the steep transition. The
    # positions depend only on the common trained model, never on the theme.
    label_heights = {.1: .92, .25: .55, .5: -.04, .75: -.47, .9: -.87}
    for level, height in label_heights.items():
        if level == .5:
            continue
        minor = ax.contour(x1, x2, probabilities, levels=[level], colors="white",
                           linewidths=.55, linestyles="dotted", alpha=.8)
        horizontal = (np.log(level/(1-level))-weights[2]-weights[1]*height)/weights[0]
        labels_inline = ax.clabel(minor, levels=[level], fmt={level:rf"${level:g}$"},
                                 manual=[(horizontal,height)], fontsize=9, inline_spacing=3)
        for label in labels_inline:
            label.set_color("#111111")
            label.set_alpha(1.)
            label.set_bbox({"facecolor":"#eeeeee", "alpha":.85,
                            "edgecolor":"none", "boxstyle":"round,pad=0.1,rounding_size=0.15"})
    boundary = ax.contour(x1, x2, probabilities, levels=[.5], colors="white", linewidths=1.4)
    boundary.set_path_effects([effects.Stroke(linewidth=2.5, foreground="#111111"), effects.Normal()])
    height = label_heights[.5]
    horizontal = (-weights[2]-weights[1]*height)/weights[0]
    boundary_labels = ax.clabel(boundary, levels=[.5], fmt={.5:r"$0.5$"},
                                manual=[(horizontal,height)], fontsize=9, inline_spacing=3)
    for label in boundary_labels:
        label.set_color("#111111")
        label.set_alpha(1.)
        label.set_bbox({"facecolor":"#eeeeee", "alpha":.85,
                        "edgecolor":"none", "boxstyle":"round,pad=0.1,rounding_size=0.15"})
    for value, marker, color, name in [(0, "o", COLORS[0], r"I: $y=0$"),
                                      (1, "v", COLORS[2], r"III: $y=1$")]:
        selected = labels == value
        ax.scatter(*features[selected].T, s=25, marker=marker, edgecolors=color,
                   facecolors=fig.get_facecolor(), linewidths=1.1, zorder=4, label=name)
    ax.set(xlabel=r"$\overset{\sim}{Ga}$", ylabel=r"$\overset{\sim}{Eo}$",
           title=r"Model probability $y_{\mathbf{w}}(\mathbf{x})$", xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    position = ax.get_position()
    color_ax = fig.add_axes([position.x0, .15, position.width, .035])
    cb = fig.colorbar(mesh, cax=color_ax, orientation="horizontal", ticks=[0, .5, 1])
    cb.set_label(r"$y_{\mathbf{w}}(\mathbf{x})$", labelpad=2, fontsize=9)
    cb.ax.tick_params(labelsize=9, pad=2)
    history.plot(np.arange(fit["steps"] + 1), fit["mean_bce"], color=COLORS[0])
    history.set(xlabel="Gradient steps", ylabel="Mean binary cross-entropy",
                title="Training objective", xlim=(0, fit["steps"]), ylim=(0, .75))
    history.set_xticks([0, 1000, 2000]); history.set_yticks([0, .2, .4, .6])
    history.grid(alpha=.16)
    history.text(.97, .95, r"$\eta=1$, $N=47$" + "\n2000 fixed steps",
                 transform=history.transAxes, ha="right", va="top", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1,
               bbox_to_anchor=(.77, .055), frameon=False, fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks/media")
    parser.add_argument("--results", type=Path,
                        default=ROOT / "figures/bubble-classification/logistic-results.json")
    parser.add_argument("--no-tex", action="store_true",
                        help="Explicit MathText opt-out; record the reason when used.")
    parser.add_argument("--png", action="store_true", help="Also export 300-DPI PNG previews.")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "notebooks"))
    from figure_style import DPI, figure_style
    from figure_utils import save_theme_variants
    data = load_bubble_data()
    selected = np.isin(data.regimes, ["I", "III"])
    raw = data.raw[selected]
    labels = (data.regimes[selected] == "III").astype(np.float64)
    scaling = fit_log_minmax(raw)
    features = transform_features(raw, scaling)
    assert len(labels) == 47 and np.count_nonzero(labels) == 27
    fit = train_logistic(features, labels)
    checks = numerical_checks(features, labels, fit)
    slices = coefficient_slice(features, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(figure_style(use_tex=not args.no_tex)):
        fig, ax = plt.subplots(figsize=(3, 1))
        ax.text(.01, .5, r"$\log(L)=\sum_i\log(q_i)$, $p=\sigma(\mathbf{w}^T\mathbf{x})$")
        ax.set_axis_off()
        smoke_path = args.output_dir / ".logistic-renderer-check.svg"
        try:
            fig.savefig(smoke_path)
        finally:
            plt.close(fig)
        smoke_path.unlink()
    for extension in (["svg", "png"] if args.png else ["svg"]):
        for stem, draw in [
                ("bubble-logistic-landscapes", lambda: draw_landscapes(slices, use_tex=not args.no_tex)),
                ("bubble-logistic-probabilities", lambda: draw_probabilities(features, labels, fit, use_tex=not args.no_tex))]:
            for path in save_theme_variants(draw, args.output_dir / f"{stem}.{extension}", dpi=DPI).values():
                print(path)
    result = {
        "input": str(DATA_FILE.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "renderer": "MathText explicitly selected (--no-tex)" if args.no_tex else "LaTeX; SVG glyph outlines",
        "training_rows": len(labels), "class_counts": {"I": 20, "III": 27},
        "point_ids": [identifier for identifier, keep in zip(data.point_ids, selected) if keep],
        "features": ["Ga", "Eo"], "feature_transform": "log10 then train-fitted min-max to [-1,1]",
        "log10_minimum": scaling.minimum.tolist(), "log10_maximum": scaling.maximum.tolist(),
        "training_features": features.tolist(), "training_labels": labels.astype(int).tolist(),
        "coefficient_order": ["scaled log10 Ga", "scaled log10 Eo", "bias"],
        "fit": fit, "numerical_checks": checks,
        "landscape_slice": {"fixed_bias": 0.0, "w1_limits": [-8, 12], "w2_limits": [-8, 12],
                            "grid_shape": list(slices[0].shape), "log_likelihood_range": [float(slices[3].min()), float(slices[3].max())],
                            "likelihood_contour_levels": (slices[2].max() * np.array([.01, .1, .3, .6, .9])).tolist(),
                            "interpretation": "same two-dimensional slice of the three-parameter objective; maximize both L and log L"},
        "limitations": ["training example only; no generalization or probability-calibration evidence",
                        "strictly separable data; no finite unregularized maximum-likelihood estimate"],
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {args.results}; mean BCE={fit['mean_bce'][-1]:.10g}; errors={fit['training_errors']}/47")


if __name__ == "__main__":
    main()
