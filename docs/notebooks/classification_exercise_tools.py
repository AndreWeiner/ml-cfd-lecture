"""Small, inspectable NumPy helpers for the bubble-classification exercise.

The tanh/softmax forward pass, backpropagation and Adam update follow the audited
lecture generator scripts/create_bubble_multiclass_figures.py. This standalone
teaching copy avoids framework setup during the data-curation exercise. Importing
this module performs no fitting, plotting, downloads or filesystem writes.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

REGIMES = ("I", "II", "III", "IV", "V")
MARKERS = ("o", "^", "v", "s", "D")


def load_points(path, *, lecture=False):
    """Read positive radius-based Ga/Eo values and explicitly curated labels.

    This checks the file schema, not the physical validity of a label mapping.
    External rows require provenance; scientific compatibility is reviewed by
    students before calling this function. No values are converted implicitly.
    """
    required = {"point_id", "Ga", "Eo", "regime"}
    if not lecture:
        required |= {"source_id", "group_id", "source_url", "source_location"}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required-set(reader.fieldnames or []))}")
        rows = []
        for number, row in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in row.items() if k is not None}
            row["Ga"], row["Eo"] = float(row["Ga"]), float(row["Eo"])
            if not np.isfinite([row["Ga"], row["Eo"]]).all() or min(row["Ga"], row["Eo"]) <= 0:
                raise ValueError(f"Row {number}: Ga and Eo must be finite and positive")
            if row["regime"] not in REGIMES:
                raise ValueError(f"Row {number}: an explicitly mapped I–V label is required")
            if lecture:
                row.update(source_id="tripathi2015", group_id=row["point_id"],
                           source_url="https://doi.org/10.1038/ncomms7268",
                           source_location="Figure 1; fresh course digitization")
            if any(not str(row[key]) for key in required):
                raise ValueError(f"Row {number}: empty required value")
            rows.append(row)
    if not rows:
        raise ValueError("The file contains no observations; fill the template with sourced data")
    keys = [(r["source_id"], r["point_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (source_id, point_id); a case must not be counted twice")
    return rows


def arrays(rows):
    return (np.array([[r["Ga"], r["Eo"]] for r in rows], dtype=float),
            np.array([REGIMES.index(r["regime"]) for r in rows], dtype=int))


def split_lecture(rows, *, seed=42):
    """Fixed class-stratified 80/20 split, before fitting any transformation."""
    _, labels = arrays(rows)
    rng = np.random.default_rng(seed)
    validation = []
    for label in range(5):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError("Each class needs at least two original observations")
        count = max(1, min(len(indices)-1, int(np.rint(.2*len(indices)))))
        validation.extend(rng.permutation(indices)[:count])
    validation = np.array(sorted(validation))
    training = np.setdiff1d(np.arange(len(rows)), validation)
    train_groups = {(rows[i]["source_id"], rows[i]["group_id"]) for i in training}
    val_groups = {(rows[i]["source_id"], rows[i]["group_id"]) for i in validation}
    if train_groups & val_groups:
        raise ValueError("Related cases cross the split; use a group-based split instead")
    return training, validation


def fit_scaler(raw_training):
    raw = np.asarray(raw_training, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Expected finite, positive Ga/Eo training pairs")
    log_features = np.log10(raw)
    lower, upper = log_features.min(0), log_features.max(0)
    if np.any(upper <= lower):
        raise ValueError("Both training features must vary")
    return lower, upper


def transform(raw, scaler):
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Logarithmic features require finite, positive inputs")
    lower, upper = scaler
    return 2*(np.log10(raw)-lower)/(upper-lower)-1


def merge_training(original_training, additional, original_validation):
    """Add independent external rows without changing or copying validation cases."""
    if not additional or any(r["source_id"] == "tripathi2015" for r in additional):
        raise ValueError("At least one additional source of independent observations is required")
    training = [dict(r) for r in original_training] + [dict(r) for r in additional]
    keys = [(r["source_id"], r["point_id"]) for r in training]
    validation_keys = {(r["source_id"], r["point_id"]) for r in original_validation}
    if len(set(keys)) != len(keys) or set(keys) & validation_keys:
        raise ValueError("Duplicate cases or overlap with validation")
    groups = {(r["source_id"], r["group_id"]) for r in training}
    if groups & {(r["source_id"], r["group_id"]) for r in original_validation}:
        raise ValueError("A related case occurs in both training and validation")
    return training


def write_points(path, rows):
    fields = ["point_id", "group_id", "Ga", "Eo", "regime", "source_id", "source_url", "source_location"]
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def initialize(*, width=16, seed=42, n_classes=5, hidden_sizes=None):
    """Initialize a tanh/softmax MLP; hidden_sizes overrides the legacy width."""
    sizes = (width,) if hidden_sizes is None else tuple(hidden_sizes)
    if not sizes or any(int(v) != v or v < 1 for v in sizes) or n_classes < 2:
        raise ValueError("Positive hidden-layer sizes and at least two classes are needed")
    rng = np.random.default_rng(seed)
    parameters = []
    dimensions = (2,) + sizes + (n_classes,)
    for incoming, outgoing in zip(dimensions[:-1], dimensions[1:]):
        parameters.extend([rng.normal(size=(incoming, outgoing))*np.sqrt(2/(incoming+outgoing)),
                           np.zeros(outgoing)])
    return parameters


def log_probabilities(features, parameters):
    hidden = np.asarray(features)
    for layer in range(0, len(parameters)-2, 2):
        hidden = np.tanh(hidden @ parameters[layer] + parameters[layer+1])
    logits = hidden @ parameters[-2] + parameters[-1]
    shifted = logits-logits.max(1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(1, keepdims=True))


def loss_and_gradient(features, labels, parameters):
    activations = [features]
    for layer in range(0, len(parameters)-2, 2):
        activations.append(np.tanh(activations[-1] @ parameters[layer] + parameters[layer+1]))
    log_p = log_probabilities(features, parameters)
    loss = float(-log_p[np.arange(len(labels)), labels].mean())
    delta = np.exp(log_p)
    delta[np.arange(len(labels)), labels] -= 1
    delta /= len(labels)
    gradients = [None] * len(parameters)
    for layer in reversed(range(len(parameters)//2)):
        gradients[2*layer] = activations[layer].T @ delta
        gradients[2*layer+1] = delta.sum(0)
        if layer:
            delta = (delta @ parameters[2*layer].T)*(1-activations[layer]**2)
    return loss, gradients


def metrics(fit, features, labels):
    labels = np.asarray(labels, dtype=int)
    log_p = log_probabilities(features, fit["parameters"])
    prediction = log_p.argmax(1)
    n_classes = fit["parameters"][-1].size
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(confusion, (labels, prediction), 1)
    return {"cross_entropy": float(-log_p[np.arange(len(labels)), labels].mean()),
            "errors": int((prediction != labels).sum()), "samples": len(labels),
            "confusion": confusion}


def fit_classifier(x_train, y_train, x_validation, y_validation, *,
                   width=16, seed=42, learning_rate=.01, epochs=500, patience=100, n_classes=5, hidden_sizes=None):
    """One full-batch Adam run, with strict-improvement early stopping and restore.

    Mean categorical CE is calculated after each update for both data sets.
    The fixed validation set selects the checkpoint; no test data are used.
    With patience=None, training continues to the epoch budget while retaining
    both the selected and final parameter states for overfitting diagnostics.
    """
    x_train, x_validation = np.asarray(x_train, dtype=float), np.asarray(x_validation, dtype=float)
    y_train, y_validation = np.asarray(y_train), np.asarray(y_validation)
    for x, y in [(x_train, y_train), (x_validation, y_validation)]:
        if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0 or y.shape != (len(x),):
            raise ValueError("Expected nonempty (N,2) inputs and (N,) labels")
        if not np.isfinite(x).all() or y.dtype.kind not in "iu" or np.any((y < 0) | (y >= n_classes)):
            raise ValueError("Expected finite inputs and integer class indices within the model output range")
    if epochs < 1 or (patience is not None and patience < 1) or learning_rate <= 0:
        raise ValueError("Training budget, patience and learning rate must be positive")
    parameters = initialize(width=width, seed=seed, n_classes=n_classes, hidden_sizes=hidden_sizes)
    first = [p.copy() for p in parameters]
    moment = [np.zeros_like(p) for p in parameters]
    variance = [np.zeros_like(p) for p in parameters]
    train_loss, val_loss = [], []
    best_loss, best_epoch, saved = np.inf, 0, None
    for epoch in range(epochs+1):
        if epoch:
            _, gradients = loss_and_gradient(x_train, y_train, parameters)
            for j, gradient in enumerate(gradients):
                moment[j] = .9*moment[j]+.1*gradient
                variance[j] = .999*variance[j]+.001*gradient**2
                parameters[j] -= learning_rate*(moment[j]/(1-.9**epoch))/(np.sqrt(variance[j]/(1-.999**epoch))+1e-8)
        train_loss.append(float(-log_probabilities(x_train, parameters)[np.arange(len(y_train)), y_train].mean()))
        val_loss.append(float(-log_probabilities(x_validation, parameters)[np.arange(len(y_validation)), y_validation].mean()))
        if val_loss[-1] < best_loss:
            best_loss, best_epoch = val_loss[-1], epoch
            saved = [p.copy() for p in parameters]
        if patience is not None and epoch-best_epoch >= patience:
            break
    return {"parameters": saved, "last_parameters": [p.copy() for p in parameters], "initial_parameters": first, "best_epoch": best_epoch,
            "stop_epoch": epoch, "train_loss": np.array(train_loss), "validation_loss": np.array(val_loss),
            "protocol": {"width": width, "seed": seed, "learning_rate": learning_rate,
                         "max_epochs": epochs, "patience": patience, "n_classes": n_classes,
                         "hidden_sizes": list((width,) if hidden_sizes is None else hidden_sizes)}}


def save_model(path, fit, scaler):
    """Save selected weights together with feature scaling and class order."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"parameter_{i}": p for i, p in enumerate(fit["parameters"])},
                        log_minimum=scaler[0], log_maximum=scaler[1], regimes=np.array(REGIMES),
                        best_epoch=fit["best_epoch"])


def save_comparison(output, baseline, enriched, training, validation, additional, scaler, *, use_tex=True):
    """Export matched light/dark loss and argmax maps; no model is fitted here.

    TeX is the default. An intentional use_tex=False opt-out is recorded in the
    accompanying rendering.json file; there is no automatic renderer fallback.
    """
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
    from cycler import cycler
    from figure_style import figure_style, WIDTH_MM
    from figure_utils import save_theme_variants
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    raw, _ = arrays(training+validation+additional)
    all_features = transform(raw, scaler)
    lower, upper = all_features.min(0)-.15, all_features.max(0)+.15
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 201), np.linspace(lower[1], upper[1], 201))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    maps = [log_probabilities(grid, fit["parameters"]).argmax(1).reshape(xx.shape)
            for fit in [baseline, enriched]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    def configure():
        plt.rcParams.update({**figure_style(use_tex=use_tex),
                             "text.latex.preamble": r"\usepackage{amsmath}",
                             "axes.prop_cycle": cycler(color=colors)})
    # Exercise the selected text renderer before creating all exports.
    with plt.rc_context():
        configure(); smoke, ax = plt.subplots()
        ax.set_xlabel(r"$\overset{\sim}{Ga},\ \log_{10}(Eo)$")
        smoke.savefig(output/"renderer-check.svg"); plt.close(smoke)
    def draw_losses():
        configure(); fig, ax = plt.subplots(figsize=(WIDTH_MM["double"]/25.4, 90/25.4), layout="constrained")
        for color, name, fit in zip(colors, ["Original", "With additional data"], [baseline, enriched]):
            ax.plot(fit["train_loss"], color=color, label=name+": training")
            ax.plot(fit["validation_loss"], color=color, linestyle="--", label=name+": validation")
            ax.scatter([fit["best_epoch"]], [fit["validation_loss"][fit["best_epoch"]]], color=color, marker="o")
        ax.set(xlabel="Epoch (one full-batch update)", ylabel="Mean categorical cross entropy")
        ax.legend(fontsize=9); ax.grid(alpha=.15)
        return fig
    def draw_maps():
        configure(); fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM["double"]/25.4, 120/25.4))
        fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.24, wspace=.28)
        cmap = ListedColormap([to_rgba(c, .20) for c in colors[:5]])
        for ax, name, prediction in zip(axes, ["Original training data", "With additional data"], maps):
            ax.pcolormesh(xx, yy, prediction, shading="nearest", cmap=cmap,
                          norm=BoundaryNorm(np.arange(-.5, 5.5), 5), rasterized=True)
            for rows, size, filled, outline in [(training, 14, True, False),
                                               (validation, 26, False, False),
                                               (additional, 35, True, True)]:
                if not rows: continue
                raw_rows, labels = arrays(rows); points = transform(raw_rows, scaler)
                for k, marker in enumerate(MARKERS):
                    mask = labels == k
                    ax.scatter(*points[mask].T, s=size, marker=marker, zorder=3,
                               facecolors=colors[k] if filled else ax.get_facecolor(),
                               edgecolors=plt.rcParams["text.color"] if outline else colors[k], linewidths=.8)
            ax.set(xlim=(lower[0],upper[0]), ylim=(lower[1],upper[1]), title=name,
                   xlabel=r"$\overset{\sim}{Ga}$", ylabel=r"$\overset{\sim}{Eo}$", aspect="equal")
        from matplotlib.lines import Line2D
        class_handles = [Line2D([],[],marker=m,color=c,linestyle="none",label=r)
                         for m,c,r in zip(MARKERS,colors,REGIMES)]
        fig.legend(handles=class_handles, loc="lower center", bbox_to_anchor=(.5,.10), ncol=5, frameon=False)
        fig.text(.5,.055,"Filled: original training; hollow: validation; outlined: additional",ha="center",fontsize=9)
        return fig
    paths = {}
    for stem, draw in [("loss-comparison", draw_losses), ("regime-comparison", draw_maps)]:
        paths[stem] = save_theme_variants(draw, output/(stem+".svg"), dpi=300)
    (output/"rendering.json").write_text(json.dumps({"text_usetex":use_tex,
        "renderer_choice":"course default" if use_tex else "explicit user opt-out",
        "width_mm":180,"map_grid":[201,201]},indent=2)+"\n")
    return paths


def split_by_source_and_class(rows, *, seed=42):
    """80/20 split of case groups, stratified by source and harmonized class.

    A group cannot cross partitions. Each source/class stratum needs >=2 groups.
    Returned indices refer to the supplied merged row order.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for i, row in enumerate(rows):
        key = (row['source_id'], row['group_id'])
        groups.setdefault(key, []).append(i)
    strata = {}
    for key, indices in groups.items():
        labels = {rows[i]['regime'] for i in indices}
        if len(labels) != 1:
            raise ValueError('Related cases have conflicting harmonized labels')
        strata.setdefault((key[0], next(iter(labels))), []).append(indices)
    validation = []
    for key, cases in sorted(strata.items()):
        if len(cases) < 2:
            raise ValueError(f'Not enough independent groups in stratum {key}')
        count = max(1, min(len(cases)-1, int(np.rint(.2*len(cases)))))
        for j in rng.permutation(len(cases))[:count]:
            validation.extend(cases[j])
    validation = np.array(sorted(validation), dtype=int)
    training = np.setdiff1d(np.arange(len(rows)), validation)
    return training, validation


def learning_curve(rows, training, validation, class_names, *,
                   fractions=(.2,.4,.6,.8,1.), seeds=(11,22,33), **configuration):
    """Nested, source/class-stratified subsets of the training pool.

    Fixed validation rows never enter a subset or its scaler. All rows in a
    selected case group travel together. Each subset fits its own scaler.
    Seeds repeat subset selection and model initialization; their spread is
    not a confidence interval and does not capture extraction uncertainty.
    """
    fractions = tuple(fractions)
    if sorted(set(fractions)) != list(fractions) or not all(0 < f <= 1 for f in fractions):
        raise ValueError('Fractions must increase uniquely within (0,1]')
    names = list(class_names)
    raw = np.array([[r['Ga'], r['Eo']] for r in rows], dtype=float)
    labels = np.array([names.index(r['regime']) for r in rows], dtype=int)
    strata = {}
    for i in training:
        r = rows[i]
        strata.setdefault((r['source_id'], r['regime']), {}).setdefault(r['group_id'], []).append(int(i))
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ordered = [[groups[k] for k in rng.permutation(list(groups))] for groups in strata.values()]
        for fraction in fractions:
            subset = np.array(sorted(i for groups in ordered
                for group in groups[:max(1, int(np.ceil(fraction*len(groups))))]
                for i in group), dtype=int)
            scaler = fit_scaler(raw[subset])
            x, xv = transform(raw[subset], scaler), transform(raw[validation], scaler)
            fit = fit_classifier(x, labels[subset], xv, labels[validation],
                                 seed=seed, n_classes=len(names), **configuration)
            score = metrics(fit, xv, labels[validation])
            results.append(dict(seed=seed, fraction=fraction, training_samples=len(subset),
                validation_cross_entropy=score['cross_entropy'],
                validation_accuracy=1-score['errors']/score['samples'],
                best_epoch=fit['best_epoch'], stop_epoch=fit['stop_epoch'],
                training_indices=subset.tolist()))
    return results


def save_learning_curve(output, results, *, use_tex=True):
    """Plot training size versus fixed-validation CE and accuracy, in both themes."""
    import json
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style, WIDTH_MM, DPI
    from figure_utils import save_theme_variants
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    colors = plt.colormaps['tab10'].colors
    def configure():
        plt.rcParams.update({**figure_style(use_tex=use_tex),
                            'axes.prop_cycle': cycler(color=colors)})
    with plt.rc_context():
        configure(); f, ax = plt.subplots(); ax.set_ylabel(r'$\log(\widehat y)$')
        f.savefig(output/'renderer-check.svg'); plt.close(f)
    def draw():
        configure()
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM['double']/25.4, 85/25.4))
        fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.93, wspace=.35)
        for seed in sorted({r['seed'] for r in results}):
            series = [r for r in results if r['seed']==seed]
            for ax, key in zip(axes, ['validation_cross_entropy','validation_accuracy']):
                ax.plot([r['training_samples'] for r in series], [r[key] for r in series],
                        marker='o', markersize=3, alpha=.8, label=f'Seed {seed}')
                ax.set_xlabel('Number of training points'); ax.grid(alpha=.18)
        axes[0].set_ylabel('Validation cross entropy')
        axes[1].set_ylabel('Validation accuracy'); axes[1].set_ylim(0,1)
        axes[0].legend(frameon=False, fontsize=8)
        return fig
    paths=save_theme_variants(draw,output/'learning-curve.svg',dpi=DPI)
    (output/'learning-curve.json').write_text(json.dumps(results,indent=2)+'\n')
    (output/'rendering.json').write_text(json.dumps({'use_tex':use_tex,
        'renderer':'LaTeX' if use_tex else 'MathText'},indent=2)+'\n')
    return paths


def compare_architectures(rows, training, validation, class_names, *,
                          architectures=((8,), (16,), (32,), (16,16), (16,16,16), (32,32)),
                          seeds=(11,22,33), epochs=2000, learning_rate=.01):
    """Fixed data/scaler and epoch budget; continue beyond the validation minimum.

    Fits retain both the best-validation model and the final model. No patience
    trigger truncates these diagnostic histories; no new test data are introduced.
    """
    raw = np.array([[r['Ga'],r['Eo']] for r in rows], dtype=float)
    labels = np.array([list(class_names).index(r['regime']) for r in rows],dtype=int)
    scaler = fit_scaler(raw[training])
    x,xv=transform(raw[training],scaler),transform(raw[validation],scaler)
    records=[]
    for sizes in architectures:
        for seed in seeds:
            fit=fit_classifier(x,labels[training],xv,labels[validation],
                hidden_sizes=sizes,seed=seed,n_classes=len(class_names),
                epochs=epochs,patience=None,learning_rate=learning_rate)
            best=metrics(fit,xv,labels[validation])
            last=metrics({'parameters':fit['last_parameters']},xv,labels[validation])
            records.append(dict(hidden_sizes=list(sizes),seed=seed,
                parameters=sum(p.size for p in fit['parameters']),best_epoch=fit['best_epoch'],
                final_epoch=fit['stop_epoch'],best_validation_ce=best['cross_entropy'],
                final_validation_ce=last['cross_entropy'],
                best_validation_accuracy=1-best['errors']/len(validation),
                final_validation_accuracy=1-last['errors']/len(validation),
                final_training_ce=float(fit['train_loss'][-1]),
                train_loss=fit['train_loss'].tolist(),validation_loss=fit['validation_loss'].tolist()))
    return records


def save_architecture_comparison(output, records, *, use_tex=True):
    """Export selected/final accuracies and seed-one train/validation histories."""
    import json
    import matplotlib.pyplot as plt
    from cycler import cycler
    from figure_style import figure_style,WIDTH_MM,DPI
    from figure_utils import save_theme_variants
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sizes=list(dict.fromkeys(tuple(r['hidden_sizes']) for r in records))
    seeds=sorted({r['seed'] for r in records});colors=plt.colormaps['tab10'].colors
    def configure():
        plt.rcParams.update({**figure_style(use_tex=use_tex),'axes.prop_cycle':cycler(color=colors)})
    with plt.rc_context():
        configure();f,ax=plt.subplots();ax.set_xlabel(r'$\log(\widehat y)$')
        f.savefig(output/'architecture-renderer-check.svg');plt.close(f)
    def accuracy():
        configure();fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,105/25.4))
        fig.subplots_adjust(left=.10,right=.98,bottom=.36,top=.84,wspace=.30)
        for seed in seeds:
            rr=[next(r for r in records if tuple(r['hidden_sizes'])==s and r['seed']==seed) for s in sizes]
            for ax,key in zip(axes,['best_validation_accuracy','final_validation_accuracy']):
                ax.plot(range(len(sizes)),[r[key] for r in rr],marker='o',markersize=3,label=f'Seed {seed}')
        for ax,title in zip(axes,['Validation-selected model','Final-epoch model']):
            ax.set(ylim=(0,1),ylabel='Validation accuracy',title=title)
            ax.set_xticks(range(len(sizes)),['--'.join(map(str,s)) for s in sizes],rotation=60, fontsize=8)
            ax.set_xlabel('Neurons in each hidden layer');ax.grid(alpha=.18)
        axes[0].legend(frameon=False,fontsize=8)
        return fig
    def histories():
        configure(); nrows=(len(sizes)+1)//2
        fig,axes=plt.subplots(nrows,2,squeeze=False,figsize=(WIDTH_MM['double']/25.4,55*nrows/25.4))
        fig.subplots_adjust(left=.10,right=.98,bottom=.08,top=.93,hspace=.67,wspace=.32)
        for ax,s in zip(axes.flat,sizes):
            r=next(r for r in records if tuple(r['hidden_sizes'])==s and r['seed']==seeds[0])
            ax.plot(r['train_loss'],color=colors[0],label='Training')
            ax.plot(r['validation_loss'],color=colors[1],linestyle='--',label='Validation')
            ax.plot(r['best_epoch'],r['best_validation_ce'],'o',color=colors[1],markersize=4)
            ax.set(title='Hidden layers: '+', '.join(map(str,s)),xlabel='Epoch',ylabel='Cross entropy')
            ax.grid(alpha=.18)
        for ax in list(axes.flat)[len(sizes):]:ax.set_visible(False)
        axes.flat[0].legend(frameon=False,fontsize=8)
        return fig
    paths={}
    for name,draw in [('architecture-accuracy',accuracy),('architecture-losses',histories)]:
        paths[name]=save_theme_variants(draw,output/(name+'.svg'),dpi=DPI)
        save_theme_variants(draw,output/(name+'.png'),dpi=DPI)
    (output/'architecture-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    return paths
