#!/usr/bin/env python3
"""Reproducible feature and online-perceptron examples for the bubble lecture.

Importing this module does not train, export files, or change plotting settings.
Core functions use NumPy float64. The CLI validates tiny known cases, runs the
47-point I/III training example, records every presentation, and exports figures.
All reported errors concern the training set; there is no held-out evaluation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "datasets/path_shape_regimes/tripathi2015_digitization/points.csv"
REGIMES = (
    ("I", "Axisymmetric", "o"),
    ("II", "Skirted, non-oscillatory", "^"),
    ("III", "Oscillatory path", "v"),
    ("IV", "Peripheral breakup", "s"),
    ("V", "Central breakup", "D"),
)
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


@dataclass(frozen=True)
class BubbleData:
    raw: np.ndarray  # Columns: radius-based, dimensionless Ga and Eo.
    regimes: np.ndarray
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class LogMinMax:
    minimum: np.ndarray
    maximum: np.ndarray


def _matrix(values, *, positive=False):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError("Expected a nonempty (n, 2) feature matrix.")
    if not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError("Features must be finite and, before logarithms, positive.")
    return array


def load_bubble_data(path=DATA_FILE):
    """Read the fresh digitization, preserving CSV order and physical classes."""
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    raw = _matrix([[float(row["Ga"]), float(row["Eo"])] for row in rows], positive=True)
    regimes = np.array([row["regime"] for row in rows])
    if not set(regimes).issubset({row[0] for row in REGIMES}):
        raise ValueError("Unknown regime in input CSV.")
    ids = tuple(row["point_id"] for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError("Point identifiers must be unique.")
    return BubbleData(raw, regimes, ids)


def fit_log_minmax(raw):
    """Fit only on the supplied training rows; retain base-ten log extrema."""
    logarithms = np.log10(_matrix(raw, positive=True))
    minimum, maximum = logarithms.min(axis=0), logarithms.max(axis=0)
    if np.any(maximum <= minimum):
        raise ValueError("Both training features need a nonzero log range.")
    return LogMinMax(minimum, maximum)


def transform_features(raw, scaling):
    """Apply a fitted log10/min-max map; new values are not refitted or clipped."""
    values = np.log10(_matrix(raw, positive=True))
    span = np.asarray(scaling.maximum) - np.asarray(scaling.minimum)
    if span.shape != (2,) or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Invalid fitted feature ranges.")
    return 2.0 * (values - scaling.minimum) / span - 1.0


def predict(features, weights):
    """H(z)=1 for z>0, otherwise 0; weights are [w_Ga, w_Eo, bias]."""
    features = _matrix(features)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (3,) or not np.isfinite(weights).all():
        raise ValueError("Expected three finite weights, including the bias.")
    return (features @ weights[:2] + weights[2] > 0).astype(np.int64)


def classification_loss(features, labels, weights):
    """Half the number of hard-label errors: 0.5 * sum_i (y_i-yhat_i)^2."""
    labels = np.asarray(labels)
    predictions = predict(features, weights)
    if labels.shape != predictions.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected one binary label for every feature row.")
    return float(0.5 * np.sum((labels - predictions) ** 2))


def train_perceptron(features, labels, *, seed=42, learning_rate=1.0, max_epochs=500):
    """Deterministic shuffled online updates, with every presentation recorded.

    Start at zero weights. For each sample use w += eta*(y-H(z))*[x1,x2,1].
    Stop only after a complete epoch without an update. This is the online
    perceptron rule, not a gradient of the hard-label squared-error function.
    State 0 is before training; state n follows the nth presented sample,
    including presentations that cause no update. An epoch reshuffles all rows.
    """
    features = _matrix(features)
    labels = np.asarray(labels)
    if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected a binary label for each feature row.")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or max_epochs < 1:
        raise ValueError("Learning rate and epoch budget must be positive.")
    augmented = np.column_stack((features, np.ones(len(features))))
    weights = np.zeros(3, dtype=np.float64)
    rng = np.random.default_rng(seed)
    states = [weights.copy()]
    losses = [classification_loss(features, labels, weights)]
    events = []
    updates = 0
    converged = False
    for epoch in range(1, max_epochs + 1):
        epoch_updates = 0
        for index in rng.permutation(len(features)):
            index = int(index)
            predicted = int(augmented[index] @ weights > 0)
            delta = int(labels[index]) - predicted
            if delta:
                weights += learning_rate * delta * augmented[index]
                epoch_updates += 1
                updates += 1
            states.append(weights.copy())
            losses.append(classification_loss(features, labels, weights))
            events.append({"epoch":epoch, "sample_index":index,
                           "prediction_before":predicted, "delta":delta,
                           "cumulative_updates":updates})
        if epoch_updates == 0:
            converged = True
            break
    zeros = np.flatnonzero(np.asarray(losses) == 0)
    return {"seed":seed, "learning_rate":learning_rate, "max_epochs":max_epochs,
            "initial_weights":[0.0, 0.0, 0.0], "final_weights":weights.tolist(),
            "epochs_completed":epoch, "presentations":len(events), "updates":updates,
            "converged":converged, "first_zero_error_presentation":int(zeros[0]) if len(zeros) else None,
            "loss_increases":int(np.count_nonzero(np.diff(losses) > 0)),
            "initial_loss":losses[0], "final_loss":losses[-1],
            "minimum_loss":min(losses), "maximum_loss":max(losses),
            "weight_states":[state.tolist() for state in states],
            "full_data_loss":losses, "events":events}


def convex_hull(points):
    """2-D monotone-chain hull; no SciPy dependency is needed for this tiny check."""
    points = sorted(set(map(tuple, _matrix(points))))
    if len(points) <= 1:
        return np.array(points)
    def cross(origin, a, b):
        return ((a[0]-origin[0])*(b[1]-origin[1]) -
                (a[1]-origin[1])*(b[0]-origin[0]))
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1])


def separation_certificate(features, labels, *, tolerance=1e-12):
    """Find an independently constructed separating line from convex-hull edges.

    Each candidate is a unit normal to a hull edge. A positive projection gap
    supplies a directly verifiable strict-separation certificate. A failed
    search is reported as uncertified, not inferred from perceptron convergence.
    This does not claim a maximum-margin separator.
    """
    features = _matrix(features)
    labels = np.asarray(labels)
    if labels.shape != (len(features),) or set(labels) != {0, 1}:
        raise ValueError("The certificate needs both binary classes.")
    negative, positive = features[labels == 0], features[labels == 1]
    hulls = [convex_hull(negative), convex_hull(positive)]
    normals = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for hull in hulls:
        for a,b in zip(hull, np.roll(hull, -1, axis=0)):
            edge = b-a
            if np.linalg.norm(edge) > tolerance:
                normals.append(np.array([-edge[1], edge[0]]) / np.linalg.norm(edge))
    best = None
    for normal in normals:
        for direction in (normal, -normal):
            upper_negative = float((negative @ direction).max())
            lower_positive = float((positive @ direction).min())
            gap = lower_positive-upper_negative
            if best is None or gap > best[0]:
                best = gap, direction, upper_negative, lower_positive
    gap,normal,upper_negative,lower_positive = best
    if gap <= tolerance:
        return {"certified":False, "best_projection_gap":float(gap),
                "note":"No positive-gap certificate from the convex-hull edge test."}
    bias = -(upper_negative+lower_positive)/2.0
    weights = np.r_[normal, bias]
    margins = (2*labels-1) * (features @ normal+bias)
    if not np.all(margins > tolerance):
        raise AssertionError("Candidate certificate failed direct signed-margin validation.")
    return {"certified":True, "method":"Convex-hull edge-normal projection separation",
            "weights":weights.tolist(), "projection_gap":float(gap),
            "negative_projection_maximum":upper_negative,
            "positive_projection_minimum":lower_positive,
            "minimum_signed_margin":float(margins.min()),
            "signed_margins":margins.tolist(), "tolerance":tolerance,
            "class_hulls":[hull.tolist() for hull in hulls]}


def histogram_record(data, *, bins=10):
    """Equal-width bins within each raw/log representation, using all 104 rows."""
    record = {}
    for column,name in enumerate(("Ga", "Eo")):
        record[name] = {}
        for transform in ("raw", "log10"):
            values = data.raw[:,column]
            if transform == "log10":
                values = np.log10(values)
            edges = np.linspace(values.min(), values.max(), bins+1)
            counts = [np.histogram(values[data.regimes == regime], edges)[0]
                      for regime,_,_ in REGIMES]
            total = np.sum(counts, axis=0)
            if total.sum() != len(values):
                raise AssertionError("A histogram dropped samples.")
            record[name][transform] = {"edges":edges.tolist(), "counts_by_regime":np.asarray(counts).tolist(),
                                      "total_counts":total.tolist(), "samples":len(values)}
    return record


def validate_small_cases():
    """Analytic transform/tie cases and separable/contradictory training fixtures."""
    scaler = fit_log_minmax([[1,10], [100,1000]])
    np.testing.assert_allclose(transform_features([[1,10], [10,100], [100,1000]], scaler),
                               [[-1,-1], [0,0], [1,1]], atol=1e-14)
    np.testing.assert_allclose(transform_features([[1000,10000]], scaler), [[2,2]], atol=1e-14)
    assert predict([[2,-3]], [0,0,0]).tolist() == [0]
    x = np.array([[-1,-1], [-1,1], [1,-1], [1,1]], dtype=float)
    y = np.array([0,0,1,1])
    separable = train_perceptron(x, y, max_epochs=50)
    assert separable["converged"] and separable["final_loss"] == 0
    assert separation_certificate(x, y)["certified"]
    contradictory = train_perceptron([[0,0], [0,0]], [0,1], max_epochs=10)
    assert not contradictory["converged"] and contradictory["epochs_completed"] == 10
    assert contradictory["final_loss"] == 0.5
    for invalid in ([[0,1], [1,2]], [[1,1], [1,2]]):
        try:
            fit_log_minmax(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid logarithm/range was not rejected.")
    return {"analytic_transform":True, "prediction_tie_zero":True,
            "separable_fixture":True, "contradictory_fixture":True,
            "invalid_features_rejected":True}


def validate_training_trace(features, labels, training):
    """Independently tally each error with scalar arithmetic, then audit changes."""
    for state, stored_loss in zip(training["weight_states"], training["full_data_loss"]):
        errors = sum(int((float(a)*state[0]+float(b)*state[1]+state[2] > 0) != int(label))
                     for (a,b),label in zip(features, labels))
        if stored_loss != 0.5*errors:
            raise AssertionError("Recorded full-data loss disagrees with scalar tally.")
    states = np.asarray(training["weight_states"])
    changed = np.any(states[1:] != states[:-1], axis=1)
    events = training["events"]
    assert int(changed.sum()) == training["updates"]
    assert np.array_equal(changed, [event["delta"] != 0 for event in events])
    for epoch in range(1, training["epochs_completed"]+1):
        indices = [event["sample_index"] for event in events if event["epoch"] == epoch]
        assert sorted(indices) == list(range(len(features)))
    if training["converged"]:
        assert not changed[-len(features):].any()
    return {"all_states_scalar_error_tally":True, "update_count":True,
            "each_epoch_is_permutation":True, "complete_error_free_stopping_pass":training["converged"]}


def run_example(data_path=DATA_FILE):
    data = load_bubble_data(data_path)
    checks = validate_small_cases()
    mask = np.isin(data.regimes, ["I", "III"])
    raw, labels = data.raw[mask], (data.regimes[mask] == "III").astype(np.int64)
    scaling = fit_log_minmax(raw)
    features = transform_features(raw, scaling)
    training = train_perceptron(features, labels)
    checks.update(validate_training_trace(features, labels, training))
    certificate = separation_certificate(features, labels)
    histograms = histogram_record(data)
    result = {
        "schema_version":1,
        "input":{"path":str(Path(data_path).relative_to(ROOT)),
                 "sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                 "total_samples":len(data.raw),
                 "counts_by_regime":{r:int(np.sum(data.regimes == r)) for r,_,_ in REGIMES}},
        "environment":{"python":platform.python_version(), "numpy":np.__version__},
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol":{"class_zero":"I", "class_one":"III", "fit_samples":int(mask.sum()),
                    "fit_counts":{"I":int(np.sum(labels == 0)), "III":int(np.sum(labels == 1))},
                    "features":"log10, then training-only min-max to [-1,1] per column; bias 1 appended",
                    "threshold":"H(z)=1 if z>0, else 0", "update":"w += eta*(y-H(z))*[x1,x2,1] on errors",
                    "initial_state":"all weights zero", "shuffle":"NumPy default_rng(42), fresh permutation each epoch",
                    "stop":"complete epoch with no updates, or 500 epochs",
                    "trace":"state 0 before training; subsequent states after every sample presentation, including no-update steps",
                    "loss":"L=0.5*sum_i(y_i-H(z_i))^2 on all 47 training rows after each presentation",
                    "scope":"Illustrative full-training-set fit; no validation/test split or generalization claim"},
        "feature_scaling":{"log10_minimum":scaling.minimum.tolist(), "log10_maximum":scaling.maximum.tolist()},
        "training_rows":{"point_ids":[pid for pid,keep in zip(data.point_ids, mask) if keep],
                         "features":features.tolist(), "labels":labels.tolist()},
        "training":training, "separation_certificate":certificate,
        "histograms":{"bin_count":10, "bin_convention":"left-closed/right-open, except final bin includes right edge",
                      "data":histograms}, "checks":checks,
    }
    return data, features, labels, result


def _plot_style(use_tex):
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style
    plt.rcParams.update({**figure_style(use_tex=use_tex),
                         "text.latex.preamble":r"\usepackage{amsmath}",
                         "axes.prop_cycle":cycler(color=COLORS),
                         "svg.hashsalt":"bubble-classification"})


def draw_linear(data, *, use_tex=True):
    import matplotlib.pyplot as plt
    from figure_style import WIDTH_MM
    _plot_style(use_tex)
    fig, ax = plt.subplots(figsize=(WIDTH_MM["double"]/25.4, 130/25.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.265, top=0.91)
    for i,(regime,name,marker) in enumerate(REGIMES):
        points = data.raw[data.regimes == regime]
        ax.scatter(*points.T, s=28, marker=marker, facecolors="none", edgecolors=COLORS[i],
                   linewidths=1.1, label=f"{regime}: {name}", zorder=3)
    # Slight negative lower-y display padding prevents clipping small positive Eo.
    ax.set(xlim=(0,525), ylim=(-5,210), xlabel=r"$Ga$", ylabel=r"$Eo$")
    ax.set_xticks(np.arange(0,501,100)); ax.set_yticks(np.arange(0,201,50))
    ax.grid(color=plt.rcParams["text.color"], alpha=0.14, linewidth=0.6)
    ax.set_title("Same digitized points on linear axes", pad=12)
    fig.legend(loc="lower center", bbox_to_anchor=(0.54,0.025), ncol=2, frameon=False,
               fontsize=10, columnspacing=2.0, handletextpad=0.5, borderaxespad=0, labelspacing=0.65)
    return fig


def draw_histograms(result, *, use_tex=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from figure_style import WIDTH_MM
    _plot_style(use_tex)
    fig, axes = plt.subplots(2,2,figsize=(WIDTH_MM["double"]/25.4,145/25.4), sharey="row")
    fig.subplots_adjust(left=0.11,right=0.98,bottom=0.15,top=0.88,hspace=0.58,wspace=0.25)
    histogram = result["histograms"]["data"]
    for row,name in enumerate(("Ga", "Eo")):
        max_count = max(max(histogram[name][mode]["total_counts"]) for mode in ("raw","log10"))
        for column,mode in enumerate(("raw","log10")):
            ax = axes[row,column]
            record = histogram[name][mode]
            edges = np.array(record["edges"])
            bottom = np.zeros(len(edges)-1)
            for i,(regime,_,_) in enumerate(REGIMES):
                counts = np.array(record["counts_by_regime"][i])
                ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", bottom=bottom,
                       color=COLORS[i], edgecolor=plt.rcParams["axes.facecolor"], linewidth=0.4,
                       label=regime)
                bottom += counts
            ax.set_ylim(0,max_count*1.13)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_xlabel(f"${name}$" if mode == "raw" else rf"$\log_{{10}}({name})$")
            if column == 0:
                ax.set_ylabel("Count")
            if row == 0:
                ax.set_title("Raw feature" if mode == "raw" else r"$\log_{10}$ feature", pad=10)
    fig.suptitle("Feature distributions (104 samples)", y=0.97, fontsize=11)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Regime", loc="lower center", bbox_to_anchor=(0.54,0.005),
               ncol=5, frameon=False, fontsize=10, title_fontsize=10)
    return fig


def draw_perceptron(features, labels, result, *, use_tex=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from figure_style import WIDTH_MM
    _plot_style(use_tex)
    fig, (ax,loss_ax) = plt.subplots(1,2,figsize=(WIDTH_MM["double"]/25.4,100/25.4))
    fig.subplots_adjust(left=0.11,right=0.98,bottom=0.27,top=0.87,wspace=0.43)
    for label,name,color,marker in ((0,"I",COLORS[0],"o"),(1,"III",COLORS[2],"v")):
        points = features[labels == label]
        ax.scatter(*points.T, s=25, marker=marker, facecolors="none", edgecolors=color,
                   linewidths=1.1, label=name, zorder=3)
    weights = np.array(result["training"]["final_weights"])
    neutral = plt.rcParams["text.color"]
    if abs(weights[0]) > 1e-12:
        y = np.array([-1.1,1.1])
        x = -(weights[1]*y+weights[2])/weights[0]
        ax.plot(x,y,color=neutral,linestyle="--",linewidth=1.2,label=r"$z=0$")
    else:
        ax.axhline(-weights[2]/weights[1], color=neutral, linestyle="--", linewidth=1.2,label=r"$z=0$")
    ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),xlabel=r"$\overset{\sim}{Ga}$",ylabel=r"$\overset{\sim}{Eo}$")
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    ax.set_title("Final decision boundary",pad=10,fontsize=11)
    ax.legend(loc="upper right",frameon=False,fontsize=9,handlelength=1.8)
    trace = np.array(result["training"]["full_data_loss"])
    loss_ax.step(np.arange(len(trace)),trace,where="post",color=COLORS[7],linewidth=1.0)
    # Small left padding makes the initial error spike visible beside the spine.
    loss_ax.set(xlim=(-0.03*(len(trace)-1),len(trace)-1),ylim=(0,max(trace)*1.1),
                xlabel="Sample presentations",ylabel=r"$L=\frac{1}{2}\sum_i (y_i-\hat y_i)^2$")
    loss_ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    loss_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    loss_ax.set_title("Full training-set error",pad=10,fontsize=11)
    loss_ax.grid(color=neutral, alpha=0.14, linewidth=0.6)
    fig.text(0.54,0.12,"47 training points; fixed shuffle seed 42",ha="center",fontsize=10)
    summary = (f"{result['training']['updates']} updates; "
               f"{result['training']['presentations']} presentations; "
               f"{int(2*result['training']['final_loss'])} final training errors")
    fig.text(0.54,0.06,summary,ha="center",fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"notebooks/media")
    parser.add_argument("--results",type=Path,default=ROOT/"figures/bubble-classification/results.json")
    sys.path.insert(0,str(ROOT/"notebooks"))
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    parser.add_argument("--png",action="store_true",help="Also export 300-DPI PNG previews.")
    args = parser.parse_args()
    use_tex = resolve_renderer(
        "mathtext" if args.no_tex else args.renderer, preamble=r"\usepackage{amsmath}",
    )
    data,features,labels,result = run_example()
    # Plotting imports/configuration are deliberately deferred until CLI execution.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_style import DPI, figure_style
    from figure_utils import save_theme_variants
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({**figure_style(use_tex=use_tex),
                         "text.latex.preamble":r"\usepackage{amsmath}"}):
        fig,ax = plt.subplots(figsize=(3,1))
        ax.text(0.05,0.5,r"$\overset{\sim}{Ga},\ \log_{10}(Eo),\ L=\frac{1}{2}\sum_i(y_i-\hat y_i)^2$")
        ax.set_axis_off()
        fig.savefig(args.output_dir/".bubble-classification-renderer-check.svg")
        plt.close(fig)
    (args.output_dir/".bubble-classification-renderer-check.svg").unlink()
    draws = {
        "bubble-regime-linear":lambda:draw_linear(data,use_tex=use_tex),
        "bubble-feature-histograms":lambda:draw_histograms(result,use_tex=use_tex),
        "bubble-perceptron-results":lambda:draw_perceptron(features,labels,result,use_tex=use_tex),
    }
    for stem,draw in draws.items():
        for extension in (["svg","png"] if args.png else ["svg"]):
            for path in save_theme_variants(draw,args.output_dir/f"{stem}.{extension}",dpi=DPI).values():
                print(path)
    result["environment"]["matplotlib"] = matplotlib.__version__
    result["rendering"] = {"use_tex":use_tex, "width_mm":180,
                           "height_mm":{"linear":130,"histograms":145,"perceptron":100}}
    args.results.parent.mkdir(parents=True,exist_ok=True)
    args.results.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:result["training"][key] for key in
                      ("epochs_completed","presentations","updates","initial_loss","final_loss",
                       "first_zero_error_presentation","loss_increases","final_weights")},indent=2))
    print(f"Results: {args.results}")


if __name__ == "__main__":
    main()
