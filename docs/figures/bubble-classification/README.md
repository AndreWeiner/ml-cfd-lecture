# Bubble feature and perceptron figures

Reproduce from the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-bubble-classification .venv-quarto/bin/python scripts/create_bubble_classification_figures.py
```

The script exports paired light/dark SVGs to `notebooks/media/` and saves
`results.json` here. `--output-dir`, `--results`, and `--png` support isolated
checks and 300-DPI previews. LaTeX is the default and was used for the delivered
figures. `--no-tex` is an explicit alternative, not an automatic fallback.

## Input and scope

The input is the fresh, approximate 104-point digitization in
`datasets/path_shape_regimes/tripathi2015_digitization/points.csv`.
Its provenance explains the original paper's inconsistent II/III labels.
This example uses the caption/main Results convention: I is axisymmetric,
II is skirted/non-oscillatory, and III has an oscillatory path. The input CSV
is not modified, rounded, or replaced by the older manual digitization.

The linear scatter and histograms use all 104 points. The binary example uses
only I (20 points, label 0) and III (27 points, label 1). It illustrates fitting
these 47 training points; it contains no test split or generalization estimate.

## Feature transformation and histograms

The reusable `fit_log_minmax` and `transform_features` functions apply base-ten
logarithms followed by a separate affine map to [-1,1] for each feature. The
minimum and maximum are fitted only on the 47 binary training points:

| Feature | Minimum log10 value | Maximum log10 value |
|---|---:|---:|
| Ga | 0.8458333601844122 | 2.7000000003229805 |
| Eo | -1.0014660440781047 | 1.3001048540441884 |

Applying the transformation to new points reuses these extrema without
refitting or clipping. A bias input of 1 is appended inside the trainer.

Each histogram uses 10 equal-width bins in its displayed representation:
raw Ga/Eo on the left and log10(Ga)/log10(Eo) on the right. Bin limits cover all
104 samples; right edges are excluded except in the final bin. Bars are stacked
by physical regime, with C0-C4 colors matching the regime map. Count limits are
shared across each row, so raw and transformed marginal counts can be compared.
The log transformation spreads small values apart; it does not make either
marginal distribution, or the joint distribution, uniform. Exact bin edges
and class counts are retained in `results.json`.

## Deterministic online perceptron

Weights are ordered `[w_Ga, w_Eo, bias]` and start at zero. The classifier uses
H(z)=1 for z>0 and 0 otherwise, including H(0)=0. For each presented sample,
the update is `w += eta * (y-H(z)) * [x1,x2,1]`, with eta=1. Each epoch uses
a fresh NumPy `default_rng(42)` permutation of the 47 rows. Training stops
after a complete epoch without an update or after 500 epochs.

This is the online perceptron rule. It is not gradient descent on the plotted
hard-label squared error. With t=2y-1, its mistake updates can be viewed as
chosen subgradient steps for the per-sample perceptron loss max(0,-t*z).
At z=0 the convention updates a positive example and leaves a negative example
unchanged; these are valid choices at the nondifferentiable point.

The plotted quantity is the full-training-set error
`L = 0.5 * sum((y-H(z))**2)`, half the number of incorrect labels. State n=0
is before training; state n is after the nth sample presentation, including
presentations without an update. The figure shows every recorded state, without
smoothing or decimation. It is not a trace of individual-example losses.

Observed result:

- 29 completed epochs, 1,363 sample presentations, and 125 updates.
- Final weights `[8.804494450082567, 1.3294112306272647, 3.0]`.
- Initial L=13.5 (27 errors); final L=0 (no errors).
- First zero full-data error at presentation 1,290, during epoch 28.
- The error increases on 56 presentations; it remains between 0 and 13.5.
- Epoch 29 supplies the complete error-free stopping pass.

Every weight state, presented sample index, epoch, pre-update prediction,
update direction, cumulative update count, and full-data loss is retained in
`results.json`, along with input/source hashes and runtime versions.

## Independent separability check and validation

A separate convex-hull calculation constructs a separating direction from the
two-dimensional hull edges. SciPy is not installed in the rendering environment;
the small NumPy monotone-chain and projection calculation avoids a new dependency.
It does not use the trained perceptron weights. The certificate's unit normal
and bias are:

`[0.9892680924256866, 0.14611174254125936, 0.3239563884201291]`.

The class projection gap is 0.03976457032305347; the minimum signed margin of
all 47 points is 0.019882285161526736, well above the 1e-12 numerical tolerance.
Direct signed-margin evaluation validates strict linear separation of these
digitized, transformed training points. This is a feasible separator, not a
claim of maximum margin or of separation for unseen physical conditions.

Validation includes analytic transformation and threshold-tie cases, a known
separable sample, contradictory identical inputs, and invalid logarithm/constant
range rejection. Every recorded training error is checked again by an independent
scalar tally; weight-change counts, epoch permutations, and the stopping pass
are checked. A separate parent-agent implementation reproduced the training
counts and weights. Repeated runs produced identical training traces.

## Figure contract and inspection

The three generators share `notebooks/figure_style.py` and the paired-export
helper. All figures use a 180-mm canvas width, 11-pt main text and 10-pt ticks.
Heights are 130 mm for the linear scatter, 145 mm for the 2-by-2 histograms, and
100 mm for the two-panel perceptron result. The compact binary legend uses 9 pt.
Colors, bin edges, data, model, axes and label positions are identical across
each light/dark pair; backgrounds are white and #111111. A small negative lower
Eo display margin in the linear scatter prevents clipping the smallest positive
points; no negative data are introduced. Small left padding in the error trace
keeps the initial spike visible beside the vertical axis.

A representative LaTeX SVG export passed before generation. All six actual
SVGs were rasterized and visually inspected at the intended 620-pixel display
width in both themes. Page-level theme switching and resource paths are verified
by the integrating lecture build, separately from these figure checks.
