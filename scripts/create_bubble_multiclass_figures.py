#!/usr/bin/env python3
"""One reproducible five-class MLP trajectory and an early-stopping illustration.

Reproduce in an empty output directory (NumPy, Matplotlib and LaTeX)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bubble-multiclass-mpl \
      .venv-quarto/bin/python scripts/create_bubble_multiclass_figures.py \
      --output-dir /tmp/bubble-multiclass-new

The validation set selects a checkpoint; there is no independent test set.
All views come from one initialized model and one continued training run.
The early-stopping decision is simulated; training continues for illustration.
Importing performs no training, file writes or plotting configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from create_bubble_classification_figures import (
    COLORS, DATA_FILE, ROOT, REGIMES, fit_log_minmax, load_bubble_data, transform_features,
)

ARCHITECTURE = (2, 32, 32, 5)
EPOCHS = 10000
CHECKPOINTS = (0, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000)


def stratified_split(labels, *, seed=42, validation_fraction=.2):
    """Round 20% of each class to its nearest integer; preserve input row order."""
    rng = np.random.default_rng(seed)
    validation = []
    for label in range(5):
        indices = np.flatnonzero(labels == label)
        count = int(np.rint(validation_fraction * len(indices)))
        validation.extend(rng.permutation(indices)[:count])
    validation = np.sort(validation)
    training = np.setdiff1d(np.arange(len(labels)), validation)
    return training, validation


def initialize(architecture=ARCHITECTURE, *, seed=42):
    """Glorot-normal matrices; zero biases; one independent initialization RNG."""
    rng = np.random.default_rng(seed)
    parameters = []
    for incoming, outgoing in zip(architecture[:-1], architecture[1:]):
        parameters.extend([rng.normal(size=(incoming, outgoing)) * np.sqrt(2 / (incoming + outgoing)),
                           np.zeros(outgoing)])
    return parameters


def forward(features, parameters):
    activations = [np.asarray(features, dtype=np.float64)]
    for layer in range(len(parameters)//2 - 1):
        activations.append(np.tanh(activations[-1] @ parameters[2*layer] + parameters[2*layer+1]))
    logits = activations[-1] @ parameters[-2] + parameters[-1]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return log_probabilities, activations


def cross_entropy_and_gradient(features, labels, parameters, *, gradient=True):
    log_probabilities, activations = forward(features, parameters)
    loss = float(-np.mean(log_probabilities[np.arange(len(labels)), labels]))
    if not gradient:
        return loss, np.exp(log_probabilities)
    delta = np.exp(log_probabilities)
    delta[np.arange(len(labels)), labels] -= 1
    delta /= len(labels)
    gradients = [None] * len(parameters)
    for layer in reversed(range(len(parameters)//2)):
        gradients[2*layer] = activations[layer].T @ delta
        gradients[2*layer+1] = delta.sum(axis=0)
        if layer:
            delta = (delta @ parameters[2*layer].T) * (1 - activations[layer]**2)
    return loss, gradients


def numerical_checks():
    """Check every parameter of a small network, including every bias layer."""
    parameters = initialize((2, 4, 3, 5), seed=7)
    features = np.array([[-.7, .3], [.2, -.8], [.9, .1], [-.4, -.2]])
    labels = np.array([0, 3, 1, 4])
    loss, gradients = cross_entropy_and_gradient(features, labels, parameters)
    epsilon = 1e-5
    largest_error = 0.
    for parameter, gradient in zip(parameters, gradients):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            positive = cross_entropy_and_gradient(features, labels, parameters, gradient=False)[0]
            parameter[index] = original - epsilon
            negative = cross_entropy_and_gradient(features, labels, parameters, gradient=False)[0]
            parameter[index] = original
            largest_error = max(largest_error, abs((positive-negative)/(2*epsilon)-gradient[index]))
    assert largest_error < 1e-8
    log_p, activations = forward(features, parameters)
    # Independent direct-exp normalization is safe for these moderate toy logits.
    logits = activations[-1] @ parameters[-2] + parameters[-1]
    reference = np.exp(logits)/np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(np.exp(log_p), reference, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(loss, -np.log(reference[np.arange(len(labels)), labels]).mean(), rtol=1e-13)
    zeros = [np.zeros_like(value) for value in parameters]
    np.testing.assert_allclose(cross_entropy_and_gradient(features, labels, zeros, gradient=False)[0], np.log(5.))
    shifted = [value.copy() for value in parameters]
    shifted[-1] += 10000
    np.testing.assert_allclose(forward(features, shifted)[0], log_p, rtol=1e-10, atol=1e-11)
    return {"all_parameter_central_difference_max_error": largest_error,
            "central_difference_step": epsilon, "toy_parameter_count": sum(p.size for p in parameters),
            "softmax_common_logit_shift_invariance": True, "zero_logits_cross_entropy": float(np.log(5.))}


def train(features, labels, training, validation, *, epochs=EPOCHS, learning_rate=.01,
          patience=500, seed=42):
    """Full-batch Adam; record state zero and each completed parameter update."""
    parameters = initialize(seed=seed)
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    train_loss, val_loss, train_error, val_error = [], [], [], []
    checkpoints = {}
    best_value, best_epoch, best_parameters = np.inf, 0, None
    stop_epoch, stop_best_epoch, restored = None, None, None
    for epoch in range(epochs + 1):
        training_value, gradients = cross_entropy_and_gradient(features[training], labels[training], parameters)
        validation_value, probabilities = cross_entropy_and_gradient(features[validation], labels[validation], parameters, gradient=False)
        train_loss.append(training_value); val_loss.append(validation_value)
        train_error.append(int(np.count_nonzero(np.argmax(forward(features[training], parameters)[0], axis=1) != labels[training])))
        val_error.append(int(np.count_nonzero(probabilities.argmax(axis=1) != labels[validation])))
        if epoch in CHECKPOINTS:
            checkpoints[epoch] = [p.copy() for p in parameters]
        if validation_value < best_value:
            best_value, best_epoch = validation_value, epoch
            best_parameters = [p.copy() for p in parameters]
        if stop_epoch is None and epoch - best_epoch >= patience:
            stop_epoch, stop_best_epoch = epoch, best_epoch
            restored = [p.copy() for p in best_parameters]
        if epoch == epochs:
            break
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first[index] = .9*first[index] + .1*gradient
            second[index] = .999*second[index] + .001*gradient**2
            parameter -= learning_rate * (first[index]/(1-.9**(epoch+1))) / (np.sqrt(second[index]/(1-.999**(epoch+1))) + 1e-8)
    if restored is None:
        stop_best_epoch = best_epoch
        restored = best_parameters
    checkpoints[stop_best_epoch] = restored
    checkpoints[best_epoch] = best_parameters
    return {"training_mean_cross_entropy": train_loss, "validation_mean_cross_entropy": val_loss,
            "training_errors": train_error, "validation_errors": val_error,
            "best_validation_epoch_full_run": best_epoch,
            "early_stopping": {"patience": patience, "min_delta": 0., "stop_epoch": stop_epoch,
                               "restored_epoch": stop_best_epoch,
                               "rule": f"stop after {patience} consecutive updates with no strictly smaller validation mean cross-entropy; restore the smallest observed value"}}, checkpoints


def validate_run(features, labels, training, validation, scaling, raw, result, checkpoints):
    """Audit saved states and the split independently of the recorded trace."""
    assert not np.intersect1d(training, validation).size
    np.testing.assert_array_equal(np.sort(np.r_[training, validation]), np.arange(len(labels)))
    np.testing.assert_allclose(scaling.minimum, np.log10(raw[training]).min(axis=0), atol=0, rtol=0)
    np.testing.assert_allclose(scaling.maximum, np.log10(raw[training]).max(axis=0), atol=0, rtol=0)
    max_sum_error = 0.
    for epoch, parameters in checkpoints.items():
        for indices, key in [(training, "training"), (validation, "validation")]:
            loss, probabilities = cross_entropy_and_gradient(features[indices], labels[indices], parameters, gradient=False)
            assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            max_sum_error = max(max_sum_error, float(abs(probabilities.sum(axis=1)-1).max()))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1., rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(loss, result[f"{key}_mean_cross_entropy"][epoch], rtol=1e-13)
            assert np.count_nonzero(probabilities.argmax(axis=1) != labels[indices]) == result[f"{key}_errors"][epoch]
    stopping = result["early_stopping"]
    if stopping["stop_epoch"] is not None:
        assert stopping["restored_epoch"] == int(np.argmin(result["validation_mean_cross_entropy"][:stopping["stop_epoch"]+1]))
        assert stopping["stop_epoch"] - stopping["restored_epoch"] == stopping["patience"]
    return {"checkpoint_probability_sum_max_error": max_sum_error,
            "checkpoint_metrics_recomputed": True, "split_disjoint_and_complete": True,
            "scaler_fitted_on_training_rows_only": True}


def _configure(use_tex):
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style
    plt.rcParams.update({**figure_style(use_tex=use_tex), "axes.prop_cycle": cycler(color=COLORS),
                         "text.latex.preamble": r"\usepackage{amsmath}",
                         "svg.hashsalt": "bubble-multiclass-figures"})


def _draw_map(ax, features, labels, training, validation, parameters, *, ylabel=True):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
    axis = np.linspace(-1.09, 1.09, 181)
    grid_x, grid_y = np.meshgrid(axis, axis)
    prediction = forward(np.column_stack((grid_x.ravel(),grid_y.ravel())), parameters)[0].argmax(axis=1).reshape(grid_x.shape)
    cmap = ListedColormap([to_rgba(color,.20) for color in COLORS[:5]])
    ax.pcolormesh(grid_x, grid_y, prediction, shading="nearest", cmap=cmap,
                  norm=BoundaryNorm(np.arange(-.5,5.5),5), rasterized=True)
    # Exact boundaries of the displayed argmax grid; no binary p=0.5 assumption.
    step = axis[1]-axis[0]
    rows, cols = np.nonzero(prediction[:,:-1] != prediction[:,1:])
    segments = [((axis[c]+step/2,axis[r]-step/2),(axis[c]+step/2,axis[r]+step/2)) for r,c in zip(rows,cols)]
    rows, cols = np.nonzero(prediction[:-1,:] != prediction[1:,:])
    segments += [((axis[c]-step/2,axis[r]+step/2),(axis[c]+step/2,axis[r]+step/2)) for r,c in zip(rows,cols)]
    ax.add_collection(LineCollection(segments, colors=plt.rcParams["text.color"], linewidths=.7, alpha=.65))
    for category, (_, _, marker) in enumerate(REGIMES):
        for indices, is_training in [(training,True),(validation,False)]:
            selected = indices[labels[indices] == category]
            ax.scatter(*features[selected].T, marker=marker, s=15 if is_training else 29,
                       facecolors=COLORS[category] if is_training else ax.get_facecolor(),
                       edgecolors=ax.get_facecolor() if is_training else COLORS[category],
                       linewidths=.35 if is_training else 1.2, zorder=4)
    ax.set(xlim=(-1.09,1.09), ylim=(-1.09,1.09), xlabel=r"$\overset{\sim}{Ga}$")
    if ylabel:
        ax.set_ylabel(r"$\overset{\sim}{Eo}$")
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_aspect("equal")


def _legends(fig, *, class_y=.085, split_y=.005):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    neutral = plt.rcParams["text.color"]
    class_handles = [Line2D([],[],color=COLORS[i],marker=marker,linestyle="none",markersize=5,
                            label=numeral) for i,(numeral,_,marker) in enumerate(REGIMES)]
    fig.legend(handles=class_handles,loc="lower center",bbox_to_anchor=(.53,class_y),ncol=5,
               frameon=False,fontsize=10,columnspacing=1.6,handletextpad=.3)
    split_handles = [Line2D([],[],color=neutral,marker="o",linestyle="none",markersize=4,label="Training (84)"),
                     Line2D([],[],color=neutral,marker="o",markerfacecolor=fig.get_facecolor(),
                            linestyle="none",markersize=6,label="Validation (20)")]
    fig.legend(handles=split_handles,loc="lower center",bbox_to_anchor=(.53,split_y),ncol=2,
               frameon=False,fontsize=10,handletextpad=.4)


def draw_snapshots(features, labels, training, validation, result, checkpoints, *, use_tex=True):
    import matplotlib.pyplot as plt
    from figure_style import WIDTH_MM
    _configure(use_tex)
    fig, axes = plt.subplots(1,3,figsize=(WIDTH_MM["double"]/25.4,110/25.4))
    fig.subplots_adjust(left=.085,right=.985,bottom=.41,top=.85,wspace=.20)
    for ax, epoch, name in zip(axes,result["figure_snapshot_epochs"],["Early learning","Best validation","Late training"]):
        _draw_map(ax,features,labels,training,validation,checkpoints[epoch],ylabel=ax is axes[0])
        ax.set_title(f"{name}\nEpoch {epoch}",fontsize=10)
        if ax is not axes[0]:
            ax.set_yticklabels([])
        train_ce = result["training_mean_cross_entropy"][epoch]
        val_ce = result["validation_mean_cross_entropy"][epoch]
        if train_ce >= .001:
            train_label = f"{train_ce:.3f}"
        else:
            exponent = int(np.floor(np.log10(train_ce)))
            train_label = rf"${train_ce / 10**exponent:.1f}\times10^{{{exponent}}}$"
        ax.text(.5,-.43,f"Train CE: {train_label}\nVal. CE: {val_ce:.3f}",transform=ax.transAxes,
                ha="center",va="top",fontsize=9)
    _legends(fig)
    return fig


def draw_early_stopping(features, labels, training, validation, result, checkpoints, *, use_tex=True):
    import matplotlib.pyplot as plt
    from figure_style import WIDTH_MM
    _configure(use_tex)
    fig,(curve,model)=plt.subplots(1,2,figsize=(WIDTH_MM["double"]/25.4,115/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.33,top=.86,wspace=.37)
    epochs=np.arange(result["epochs"]+1)
    curve.plot(epochs,result["training_mean_cross_entropy"],color=COLORS[0],label="Training")
    curve.plot(epochs,result["validation_mean_cross_entropy"],color=COLORS[1],label="Validation")
    curve.set_xscale("symlog",linthresh=10)
    curve.set(xlim=(0,result["epochs"]),ylim=(0,2.65),xlabel="Epoch",ylabel="Mean cross-entropy",
              title="One continued training run")
    curve.set_xticks([0,10,100,1000,10000],labels=["0","10",r"$10^2$",r"$10^3$",r"$10^4$"])
    curve.grid(alpha=.15)
    best=result["early_stopping"]["restored_epoch"]
    stop=result["early_stopping"]["stop_epoch"]
    neutral=plt.rcParams["text.color"]
    curve.axvline(stop,color=neutral,linestyle=":",linewidth=1)
    curve.scatter([best],[result["validation_mean_cross_entropy"][best]],facecolors="none",edgecolors=neutral,s=32,zorder=5)
    curve.annotate(f"Best: {best}",xy=(best,result["validation_mean_cross_entropy"][best]),xytext=(12,.95),
                   arrowprops={"arrowstyle":"->","color":neutral,"lw":.8},fontsize=9)
    curve.text(stop*1.13,2.5,f"Stop: {stop}",ha="left",va="top",fontsize=9)
    curve.legend(loc="upper left",frameon=False,fontsize=9)
    curve.text(.5,-.24,"Linear to 10; logarithmic thereafter",transform=curve.transAxes,
               ha="center",va="top",fontsize=8)
    _draw_map(model,features,labels,training,validation,checkpoints[best])
    model.set_title(f"Restored checkpoint: epoch {best}")
    _legends(fig,class_y=.07,split_y=.0)
    return fig


def render_figures(output_dir, features, labels, training, validation, result, checkpoints, *, use_tex=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT/"notebooks"))
    from figure_style import DPI,figure_style
    from figure_utils import save_theme_variants
    output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(figure_style(use_tex=use_tex)):
        fig,ax=plt.subplots(figsize=(3,1));ax.text(.01,.5,r"$p_k=\exp(z_k)/\sum_j\exp(z_j)$, $\widetilde{Ga}$")
        ax.set_axis_off();path=output_dir/".multiclass-renderer-check.svg"
        try:
            fig.savefig(path)
        finally:
            plt.close(fig)
        path.unlink()
    for stem,draw in [("bubble-multiclass-snapshots",draw_snapshots),("bubble-multiclass-early-stopping",draw_early_stopping)]:
        callback=lambda:draw(features,labels,training,validation,result,checkpoints,use_tex=use_tex)
        for path in save_theme_variants(callback,output_dir/f"{stem}.svg",dpi=DPI).values():
            print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"notebooks/media")
    parser.add_argument("--results-dir", type=Path, default=ROOT/"figures/bubble-classification")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-tex", action="store_true", help="Explicit MathText renderer opt-out.")
    args = parser.parse_args()
    checks = numerical_checks()
    data = load_bubble_data()
    labels = np.array([tuple(r[0] for r in REGIMES).index(r) for r in data.regimes])
    training, validation = stratified_split(labels)
    scaling = fit_log_minmax(data.raw[training])
    features = transform_features(data.raw, scaling)
    started = time.perf_counter()
    result, checkpoints = train(features, labels, training, validation)
    elapsed = time.perf_counter() - started
    checks.update(validate_run(features,labels,training,validation,scaling,data.raw,result,checkpoints))
    result.update({"input": str(DATA_FILE.relative_to(ROOT)),
                   "input_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
                   "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "numpy": np.__version__, "python": platform.python_version(),
                   "architecture": list(ARCHITECTURE), "hidden_activation": "tanh", "output_activation": "softmax",
                   "optimizer": {"name": "full-batch Adam", "learning_rate": .01, "beta1": .9, "beta2": .999, "epsilon": 1e-8},
                   "epochs": EPOCHS, "initialization_seed": 42, "split_seed": 42,
                   "split_rule": "round 0.2 times each class count to nearest integer; split before fitting features",
                   "training_indices": training.tolist(), "validation_indices": validation.tolist(),
                   "training_point_ids": [data.point_ids[i] for i in training],
                   "validation_point_ids": [data.point_ids[i] for i in validation],
                   "class_counts": {r[0]: {"training": int(sum(labels[training] == i)), "validation": int(sum(labels[validation] == i))} for i, r in enumerate(REGIMES)},
                   "log10_minimum_training_only": scaling.minimum.tolist(), "log10_maximum_training_only": scaling.maximum.tolist(),
                   "numerical_checks": checks, "training_seconds": elapsed,
                   "checkpoint_epochs": sorted(checkpoints), "figure_snapshot_epochs": [10, result["early_stopping"]["restored_epoch"], EPOCHS],
                   "snapshot_selection": f"Epoch 10 is an early predefined checkpoint, epoch {result['early_stopping']['restored_epoch']} is selected by validation loss, and epoch {EPOCHS} is the predefined final budget; all belong to one trajectory.",
                   "renderer": "MathText explicitly selected (--no-tex)" if args.no_tex else "LaTeX; SVG glyph outlines",
                   "grid": {"limits": [-1.09,1.09],"shape": [181,181],"encoding": "argmax of five softmax probabilities"},
                   "limitations": ["Validation data are used for checkpoint selection; no independent test set.",
                                   "This is one illustrative trajectory, not a comparison across datasets or random seeds."]})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = args.results_dir/"multiclass-checkpoints.npz"
    np.savez_compressed(checkpoint_file, **{f"epoch_{epoch}_parameter_{j}": p for epoch, parameters in checkpoints.items() for j, p in enumerate(parameters)})
    with np.load(checkpoint_file) as saved:
        for epoch, parameters in checkpoints.items():
            for j, parameter in enumerate(parameters):
                np.testing.assert_array_equal(saved[f"epoch_{epoch}_parameter_{j}"], parameter)
    result["numerical_checks"]["saved_checkpoint_arrays_verified"] = True
    (args.results_dir/"multiclass-results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"runtime_seconds": elapsed, "split": [len(training),len(validation)],
                      "early_stopping": result["early_stopping"], "best_epoch_full_run": result["best_validation_epoch_full_run"],
                      "best_train_loss": result["training_mean_cross_entropy"][result["early_stopping"]["restored_epoch"]],
                      "best_val_loss": result["validation_mean_cross_entropy"][result["early_stopping"]["restored_epoch"]],
                      "final_train_loss": result["training_mean_cross_entropy"][-1],
                      "final_val_loss": result["validation_mean_cross_entropy"][-1],
                      "final_errors": [result["training_errors"][-1],result["validation_errors"][-1]]}, indent=2))
    if not args.no_figures:
        render_figures(args.output_dir, features, labels, training, validation, result, checkpoints, use_tex=not args.no_tex)


if __name__ == "__main__":
    main()
