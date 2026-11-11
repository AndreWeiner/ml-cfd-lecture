#!/usr/bin/env python3
"""Staged, two-hidden-unit bubble classifier and reproducible course figures.

Run: MPLCONFIGDIR=/tmp/bubble-mlp-mpl .venv-quarto/bin/python \
    scripts/create_bubble_mlp_figures.py --output-dir /tmp/bubble-mlp-new

The two logistic branches and their frozen-probability output head each use
2000 full-batch mean-BCE steps (eta=1), initialized at zero. All three stages
use the same log/min-max map fitted to the 59 I/II/III observations. No joint
fine-tuning, holdout evaluation, or probability calibration is performed.
Importing this module does not train, write files, or configure plotting.
SVG text is rendered by LaTeX as editable glyph outlines; dense fields are
embedded at 300 dpi. Both figures use the shared 180-mm width preset.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np

from create_bubble_classification_figures import (
    COLORS, DATA_FILE, ROOT, convex_hull, fit_log_minmax, load_bubble_data,
    transform_features,
)
from create_bubble_logistic_figures import sigmoid, mean_bce_and_gradient, train_logistic


def nonseparability_certificate(features, labels, point_ids):
    """Certify intersecting convex hulls using explicit convex combinations.

    Search hull-edge intersections, then contained vertices using triangles.
    Only a verified common point warrants the nonseparability claim; a failed
    search is deliberately inconclusive. No optimization failure is evidence.
    """
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    hulls = [convex_hull(features[group]) for group in groups]
    tolerance = 1e-10

    def record(vertices, coefficients):
        indices = [int(np.flatnonzero(np.all(features == vertex, axis=1))[0])
                   for vertex in vertices]
        return {"point_ids": [point_ids[index] for index in indices],
                "vertices": np.asarray(vertices).tolist(),
                "coefficients": np.asarray(coefficients).tolist()}

    witness = None
    for a, b in zip(hulls[0], np.roll(hulls[0], -1, axis=0)):
        for c, d in zip(hulls[1], np.roll(hulls[1], -1, axis=0)):
            matrix = np.column_stack((b-a, c-d))
            if abs(np.linalg.det(matrix)) <= tolerance:
                continue
            t, s = np.linalg.solve(matrix, c-a)
            if 0 <= t <= 1 and 0 <= s <= 1:
                witness = [record([a, b], [1-t, t]), record([c, d], [1-s, s])]
                break
        if witness is not None:
            break
    if witness is None:
        for first in (0, 1):
            for point in hulls[first]:
                for triangle in itertools.combinations(hulls[1-first], 3):
                    matrix = np.vstack((np.asarray(triangle).T, np.ones(3)))
                    if abs(np.linalg.det(matrix)) <= tolerance:
                        continue
                    coefficients = np.linalg.solve(matrix, np.r_[point, 1.])
                    if np.all(coefficients >= 0):
                        witness = [None, None]
                        witness[first] = record([point], [1.])
                        witness[1-first] = record(triangle, coefficients)
                        break
                if witness is not None:
                    break
            if witness is not None:
                break
    if witness is None:
        return {"certified_nonseparable": False, "note": "No common-point witness found."}
    points = []
    for group, item in zip(groups, witness):
        coefficients = np.asarray(item["coefficients"])
        assert coefficients.min() >= 0 and abs(coefficients.sum()-1) < tolerance
        assert all(identifier in [point_ids[i] for i in group] for identifier in item["point_ids"])
        points.append(coefficients @ np.asarray(item["vertices"]))
    residual = float(np.max(abs(points[0]-points[1])))
    assert residual < tolerance
    return {"certified_nonseparable": True,
            "method": "Explicit common point in the two convex hulls",
            "claim": "No affine threshold strictly separates all I from all II+III observations.",
            "common_point": points[0].tolist(), "maximum_coordinate_residual": residual,
            "tolerance": tolerance, "class_zero_combination": witness[0],
            "class_one_combination": witness[1]}


def joint_loss_gradient(features, labels, parameters):
    """Reference chain rule for a 2-2-1 network; used for checking, not fitting.

    Flatten two [w1,w2,b] rows, then output [alpha1,alpha2,b_out].
    """
    branches, output = parameters[:6].reshape(2, 3), parameters[6:]
    augmented = np.column_stack((features, np.ones(len(features))))
    hidden = sigmoid(augmented @ branches.T)
    hidden_augmented = np.column_stack((hidden, np.ones(len(features))))
    loss, output_gradient = mean_bce_and_gradient(hidden_augmented, labels, output)
    output_error = sigmoid(hidden_augmented @ output)-labels
    hidden_error = output_error[:, None]*output[:2]*hidden*(1-hidden)
    branch_gradient = hidden_error.T @ augmented / len(features)
    return loss, np.r_[branch_gradient.ravel(), output_gradient]


def numerical_checks(features, labels, stages, hidden):
    # Nonzero, asymmetric coefficients prevent zero-gradient/symmetry false passes.
    parameters = np.array([.7, -.3, .2, -.4, 1.1, -.5, 1.3, -.8, .1])
    loss, gradient = joint_loss_gradient(features, labels, parameters)
    step = 1e-5
    finite = np.array([(joint_loss_gradient(features, labels, parameters+step*unit)[0]
                       - joint_loss_gradient(features, labels, parameters-step*unit)[0])/(2*step)
                      for unit in np.eye(9)])
    np.testing.assert_allclose(gradient, finite, atol=2e-10, rtol=1e-7)
    assert hidden.shape == (59, 2) and np.all((hidden > 0) & (hidden < 1))
    for stage in stages:
        assert np.isfinite(stage["weight_states"]).all()
        assert len(stage["mean_bce"]) == 2001
        assert max(np.diff(stage["mean_bce"])) < 1e-13
        np.testing.assert_allclose(stage["mean_bce"][0], np.log(2.))
    return {"joint_chain_rule_central_difference_max_error": float(max(abs(gradient-finite))),
            "joint_gradient_test_parameters": parameters.tolist(),
            "joint_gradient_test_loss": loss, "central_difference_step": step,
            "all_stage_bce_nonincreasing": True, "hidden_shape": list(hidden.shape)}


def run_example():
    data = load_bubble_data()
    selected = np.isin(data.regimes, ["I", "II", "III"])
    raw, regimes = data.raw[selected], data.regimes[selected]
    ids = [identifier for identifier, keep in zip(data.point_ids, selected) if keep]
    counts = {name: int(np.sum(regimes == name)) for name in ("I", "II", "III")}
    assert counts == {"I": 20, "II": 12, "III": 27}
    scaling = fit_log_minmax(raw)
    features = transform_features(raw, scaling)
    stages = []
    for other in ("III", "II"):
        mask = np.isin(regimes, ["I", other])
        stage = train_logistic(features[mask], (regimes[mask] == other).astype(int))
        stage.update(class_zero="I", class_one=other,
                     point_ids=[identifier for identifier, keep in zip(ids, mask) if keep])
        stages.append(stage)
    branch_weights = np.array([stage["final_weights"] for stage in stages])
    hidden = sigmoid(np.column_stack((features, np.ones(len(features)))) @ branch_weights.T)
    labels = (regimes != "I").astype(int)
    output = train_logistic(hidden, labels)
    output.update(class_zero="I", class_one="II or III", point_ids=ids)
    stages.append(output)
    # Independently evaluate the nested composition using scalar arithmetic.
    alpha = np.array(output["final_weights"])
    scalar = []
    for x1, x2 in features:
        h = [1/(1+np.exp(-(a*x1+b*x2+c))) for a, b, c in branch_weights]
        scalar.append(1/(1+np.exp(-(alpha[0]*h[0]+alpha[1]*h[1]+alpha[2]))))
    np.testing.assert_allclose(scalar, output["training_probabilities"], rtol=1e-13)
    checks = numerical_checks(features, labels, stages, hidden)
    checks["scalar_nested_composition_max_error"] = float(max(abs(np.array(scalar)-output["training_probabilities"])))
    certificate = nonseparability_certificate(features, labels, ids)
    assert certificate["certified_nonseparable"]
    result = {"schema_version": 1,
              "input": {"path": str(DATA_FILE.relative_to(ROOT)),
                        "sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                        "total_rows": len(data.raw), "selected_rows": len(features), "counts": counts},
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "protocol": {"training": "staged; freeze both logistic branches before fitting output head",
                           "features": "one common log10/minmax map to [-1,1], fitted to all 59 training rows",
                           "hidden": "h1=sigmoid(a1 dot [x1,x2,1]); h2=sigmoid(a2 dot [x1,x2,1])",
                           "output": "sigmoid(alpha dot [h1,h2,1]); combine probabilities, not logits",
                           "positive_label": "II or III", "threshold": "p>0.5; ties assigned I",
                           "stage_settings": "zero initialization, eta=1, 2000 full-batch mean-BCE updates each",
                           "scope": "training demonstration; no held-out evaluation or probability-calibration claim"},
              "feature_scaling": {"log10_minimum": scaling.minimum.tolist(), "log10_maximum": scaling.maximum.tolist()},
              "training_rows": {"point_ids": ids, "regimes": regimes.tolist(), "features": features.tolist(),
                                "hidden_probabilities": hidden.tolist(), "labels": labels.tolist()},
              "branch_I_vs_III": stages[0], "branch_I_vs_II": stages[1], "output_head": output,
              "union_nonseparability_certificate": certificate, "checks": checks}
    return features, regimes, branch_weights, alpha, result


def configure(use_tex):
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style
    plt.rcParams.update({**figure_style(use_tex=use_tex), "text.latex.preamble": r"\usepackage{amsmath}",
                         "axes.prop_cycle": cycler(color=COLORS),
                         "svg.hashsalt": "bubble-mlp"})


def draw_architecture(*, use_tex=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch
    from figure_style import WIDTH_MM
    configure(use_tex)
    fig, ax = plt.subplots(figsize=(WIDTH_MM["double"]/25.4, 155/25.4))
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    foreground = plt.rcParams["text.color"]
    def text(x, y, value, **kw):
        ax.text(x, y, value, ha="center", va="center", **kw)
    def arrow(a, b, color=foreground):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color}, zorder=1)
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=.007", lw=1.3,
                                   edgecolor=color, facecolor=fig.get_facecolor(), zorder=2))
        text(x, y, label, fontsize=11, zorder=3)
    text(.5, .96, "Staged construction: learn branches, then freeze them", fontsize=12)
    # Explicit overset avoids the exported CMEX wide-tilde glyph becoming a
    # tiny, nearly flat accent at 620 px; the notation remains a tilde over Ga/Eo.
    box(.10, .76, .15, .13, r"$\overset{\sim}{Ga},\;\overset{\sim}{Eo}$"+"\nscaled features", COLORS[0])
    box(.43, .83, .27, .13, "I vs III\n"+r"$h_1=\sigma((\mathbf{w}_1^{(1)})^T\mathbf{x})$", COLORS[2])
    box(.43, .65, .27, .13, "I vs II\n"+r"$h_2=\sigma((\mathbf{w}_2^{(1)})^T\mathbf{x})$", COLORS[1])
    box(.83, .74, .27, .17, "I vs (II or III)\n"+r"$y_\theta(\mathbf{x})=\sigma((\mathbf{w}^{(2)})^T\mathbf{h})$", COLORS[0])
    arrow((.182, .79), (.286, .83)); arrow((.182, .72), (.286, .65))
    arrow((.573, .83), (.687, .78), COLORS[2]); arrow((.573, .65), (.687, .70), COLORS[1])
    text(.5, .53, r"$\mathbf{x}=[\overset{\sim}{Ga},\overset{\sim}{Eo},1]^T,\qquad \mathbf{h}=[h_1,h_2,1]^T$", fontsize=12)
    text(.5, .445, "The same computation: a 2--2--1 multilayer perceptron", fontsize=12)
    inputs = [(.15, .33), (.15, .20)]
    hidden = [(.5, .33), (.5, .20)]
    input_bias, hidden_bias = (.15, .065), (.5, .065)
    output = (.83, .20)
    radius = .034
    for i, inp in enumerate(inputs):
        for j, node in enumerate(hidden):
            arrow((inp[0]+radius, inp[1]), (node[0]-radius, node[1]), COLORS[2-j])
    for j, node in enumerate(hidden):
        arrow((node[0]+radius, node[1]), (output[0]-radius, output[1]), COLORS[2-j])
        ax.annotate("", xy=(node[0]-radius, node[1]),
                    xytext=(input_bias[0]+radius, input_bias[1]),
                    arrowprops={"arrowstyle": "->", "lw": 1.3, "linestyle": "dashed",
                                "color": COLORS[2-j]}, zorder=1)
    ax.annotate("", xy=(output[0]-radius, output[1]),
                xytext=(hidden_bias[0]+radius, hidden_bias[1]),
                arrowprops={"arrowstyle": "->", "lw": 1.3, "linestyle": "dashed",
                            "color": foreground}, zorder=1)
    for (x, y), label, color in [(inputs[0], r"$\overset{\sim}{Ga}$", COLORS[0]),
                                (inputs[1], r"$\overset{\sim}{Eo}$", COLORS[0]),
                                (input_bias, "$1$", foreground),
                                (hidden[0], r"$\sigma$", COLORS[2]),
                                (hidden[1], r"$\sigma$", COLORS[1]),
                                (hidden_bias, "$1$", foreground),
                                (output, r"$\sigma$", COLORS[0])]:
        # Compensate unequal physical axes width/height so neurons are circles.
        height = 2*radius*(180*.95)/(155*.955)
        ax.add_patch(Ellipse((x,y), 2*radius, height, facecolor=fig.get_facecolor(), edgecolor=color, lw=1.6, zorder=2))
        text(x, y, label, fontsize=13, zorder=3)
    text(.31, .386, r"$w_{j1}^{(1)},w_{j2}^{(1)}$", fontsize=11)
    text(.65, .318, r"$w_1^{(2)}$", fontsize=11)
    text(.65, .227, r"$w_2^{(2)}$", fontsize=11)
    text(.50, .386, r"$h_1$", fontsize=11)
    text(.50, .255, r"$h_2$", fontsize=11)
    label_background = {"facecolor": fig.get_facecolor(), "edgecolor": "none", "pad": 1}
    text(.30, .175, r"$b_1^{(1)}$", fontsize=11, bbox=label_background, zorder=4)
    text(.30, .095, r"$b_2^{(1)}$", fontsize=11, bbox=label_background, zorder=4)
    text(.65, .10, r"$b^{(2)}$", fontsize=11, bbox=label_background, zorder=4)
    arrow((.87, .20), (.96, .20)); text(.955, .25, r"$y_\theta$", fontsize=12)
    return fig


def draw_probabilities(features, regimes, branches, alpha, *, use_tex=True):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib.lines import Line2D
    from figure_style import WIDTH_MM
    configure(use_tex)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM["double"]/25.4, 112/25.4))
    fig.subplots_adjust(left=.075, right=.985, bottom=.39, top=.90, wspace=.14)
    axis = np.linspace(-1.08, 1.08, 241)
    x1, x2 = np.meshgrid(axis, axis)
    grid = np.column_stack((x1.ravel(), x2.ravel(), np.ones(x1.size)))
    hidden = sigmoid(grid @ branches.T)
    final = sigmoid(np.column_stack((hidden, np.ones(len(grid)))) @ alpha)
    fields = [hidden[:, 0], hidden[:, 1], final]
    titles = [r"$h_1$: I vs III", r"$h_2$: I vs II", r"$y_\theta$: I vs (II or III)"]
    displayed = [("I", "III"), ("I", "II"), ("I", "II", "III")]
    for k, (ax, field, title, names) in enumerate(zip(axes, fields, titles, displayed)):
        field = field.reshape(x1.shape)
        mesh = ax.pcolormesh(x1, x2, field, shading="nearest", cmap="viridis", vmin=0, vmax=1, rasterized=True)
        contours = ax.contour(x1, x2, field, levels=[.1, .9], colors="white", linewidths=.5, linestyles="dotted", alpha=.7)
        boundary = ax.contour(x1, x2, field, levels=[.5], colors="white", linewidths=1.4)
        boundary.set_path_effects([effects.Stroke(linewidth=2.4, foreground="#111111"), effects.Normal()])
        # Three labeled levels stay readable in these narrow panels.
        positions = [((-.5, .75), (-.4, .1), (-.1, -.75)),
                     ((-.8, .35), (-.1, .2), (.7, 0)),
                     ((-.65, .5), (-.2, -.1), (0, -.75))][k]
        for level, position in zip((.1, .5, .9), positions):
            collection = boundary if level == .5 else contours
            texts = ax.clabel(collection, levels=[level], fmt="%g", fontsize=10,
                              inline=True, inline_spacing=3, manual=[position])
            for label in texts:
                label.set_color("#111111")
                label.set_bbox({"facecolor": "#eeeeee", "alpha": .85,
                                "edgecolor": "none",
                                "boxstyle": "round,pad=0.1,rounding_size=0.15"})
        for name, marker, color in zip(("I", "II", "III"), ("o", "^", "v"), COLORS):
            if name in names:
                ax.scatter(*features[regimes == name].T, marker=marker, s=21, facecolors=fig.get_facecolor(),
                           edgecolors=color, linewidths=1.1, zorder=4)
        ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), title=title, xlabel=r"$\overset{\sim}{Ga}$", aspect="equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        if k == 0:
            ax.set_ylabel(r"$\overset{\sim}{Eo}$")
        else:
            ax.tick_params(labelleft=False)
    colorax = fig.add_axes([.27, .23, .49, .035])
    colorbar = fig.colorbar(mesh, cax=colorax, orientation="horizontal", ticks=[0, .5, 1])
    colorbar.set_label("Model probability", labelpad=2)
    handles = [Line2D([], [], marker=marker, color=color, markerfacecolor=fig.get_facecolor(),
                      linestyle="none", markersize=5, label=name)
               for name, marker, color in zip(("I", "II", "III"), ("o", "^", "v"), COLORS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .065), ncol=3, frameon=False)
    fig.text(.5, .025, "Common log/min-max features; each panel shows its training classes", ha="center", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"notebooks/media")
    parser.add_argument("--results", type=Path, default=ROOT/"figures/bubble-classification/mlp-results.json")
    sys.path.insert(0, str(ROOT/"notebooks"))
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    args = parser.parse_args()
    use_tex = resolve_renderer(
        "mathtext" if args.no_tex else args.renderer, preamble=r"\usepackage{amsmath}",
    )
    import matplotlib
    matplotlib.use("Agg")
    from figure_utils import save_theme_variants
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, regimes, branches, alpha, results = run_example()
    # Representative equations exported first: no silent fallback on TeX failure.
    import matplotlib.pyplot as plt
    configure(use_tex)
    smoke = plt.figure(figsize=(3, 1))
    smoke.text(.05, .5, r"$y_\theta(\mathbf{x})=\sigma(w_1^{(2)}h_1+w_2^{(2)}h_2+b),\quad \partial J/\partial b$")
    smoke.savefig(args.output_dir/"mlp-tex-smoke.svg"); plt.close(smoke)
    for stem, draw in [("bubble-mlp-architecture", lambda: draw_architecture(use_tex=use_tex)),
                       ("bubble-mlp-probabilities", lambda: draw_probabilities(features, regimes, branches, alpha, use_tex=use_tex))]:
        save_theme_variants(draw, args.output_dir/f"{stem}.svg", dpi=300)
    results["rendering"] = {"text_usetex": use_tex, "width_mm": 180,
                             "height_mm": {"architecture": 155, "probabilities": 112},
                             "dense_field_dpi": 300, "placement_width_px": 620}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({name: {"training_errors": results[name]["training_errors"],
                             "mean_bce": results[name]["mean_bce"][-1]}
                      for name in ("branch_I_vs_III", "branch_I_vs_II", "output_head")}, indent=2))


if __name__ == "__main__":
    main()
