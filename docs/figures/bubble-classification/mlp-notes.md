# Two-hidden-unit classifier: reproduction and checks

The editable generator is `scripts/create_bubble_mlp_figures.py`. It imports
the common data/feature functions from `create_bubble_classification_figures.py`
and the stable sigmoid, mean binary cross-entropy (BCE), and full-batch gradient
descent from `create_bubble_logistic_figures.py`. Importing it does not train,
write artifacts, or change plotting configuration.

From the repository root, use a new, initially empty output directory:

```sh
MPLCONFIGDIR=/tmp/bubble-mlp-mpl .venv-quarto/bin/python scripts/create_bubble_mlp_figures.py --output-dir /tmp/bubble-mlp-new
```

The default numeric record is `figures/bubble-classification/mlp-results.json`.
It contains the input and generator SHA-256 hashes, selected point IDs, every
weight state and objective value, final predictions, feature scaling, and
numerical checks. SVGs belong in `notebooks/media` and `docs/notebooks/media`.

## Controlled training protocol

Only regimes I (20), II (12), and III (27) are selected from the fresh 104-point
digitization. Both physical features are transformed by one shared base-ten
logarithm/min-max map fitted on all 59 selected rows, mapping extrema to -1 and
1. The retained log minima are `[0.8437500094224538, -1.0014660440781047]`, and
the maxima are `[2.7000000003229805, 2.298620948364928]`, in Ga, Eo order. This
is a new common scaling for the staged example; coefficients from a different
subset-specific scaler must not be reused without conversion.

Each stage starts with zero coefficients and uses exactly 2000 full-batch
mean-BCE gradient steps with learning rate 1 and no regularization. The first
branch predicts III versus I; the second predicts II versus I. Their weights
are then frozen. The final stage takes their two output probabilities and a
constant bias feature, predicting II-or-III versus I. No branch logits are
substituted for the hidden probabilities, and no joint fine-tuning is used.

| Stage | Final coefficients, including last-entry bias | Final mean BCE | Training errors |
|:--|:--|--:|--:|
| I vs III | `[17.806421545038656, 2.89393516715089, 6.3994892144716635]` | 0.07661558836607342 | 0 / 47 |
| I vs II | `[6.226110771773957, 17.80964416738222, -2.6215527001760353]` | 0.04356422168698512 | 0 / 32 |
| I vs (II or III) | `[12.612219784247042, 11.113414794127474, -6.34210652985504]` | 0.028649865487468625 | 0 / 59 |

The threshold is strictly greater than 0.5; ties are assigned I. These are
finite-budget training results, with neither a held-out evaluation nor a
probability-calibration claim. A branch is not trained on the third regime.
The probability maps show only each branch's own training classes and all
three classes in the final panel.

## Independent numerical evidence

The two convex hulls intersect in the common transformed input plane, which
certifies that I and the union of II and III cannot all be strictly separated
by an affine threshold. The explicit witness is

```text
0.38109834574351964 * component102 (I)
 + 0.6189016542564804 * component101 (I)
 = 0.2110927756598796 * split23b (II)
 + 0.7889072243401204 * component105 (III)
 = [-0.2721866082268062, -0.6670212022134627].
```

Both convex combinations have nonnegative coefficients summing to one. The
largest coordinate residual is `5.551115123125783e-17`. This geometric witness,
not an unsuccessful optimizer run, supports the nonseparability claim.

The joint nine-parameter chain rule was checked at deliberately asymmetric,
nonzero parameters by centered differences with step `1e-5`; the maximum
absolute gradient discrepancy was `1.1637683872134374e-11`. This checks the
backpropagation expression even though the plotted fit uses staged training.
Independent scalar evaluation of the nested sigmoid composition agreed with
the vectorized output to `2.220446049250313e-16`. All three objective traces
are finite, begin at log(2), and are nonincreasing over 2000 updates. The hidden
feature matrix has shape (59, 2).

## Figure and reproduction checks

The architecture is a conceptual computational graph, not a measured physical
diagram. Visible inputs are the scaled Galileo and Eotvos numbers. An explicit
constant-one input node connects to both hidden neurons, and a constant-one
hidden-layer node connects to the output. Dashed bias connections carry
`b_j^(1)` and `b^(2)`; ordinary connections use the same layer-indexed `w`
notation as the lecture. The probability map uses identical grids, limits, data, parameters,
and viridis 0--1 normalization across light/dark versions. Regimes use C0
circles, C1 upward triangles, and C2 downward triangles. The solid isoline is
0.5; dotted isolines are 0.1 and 0.9. Each contour carries its own inline numeric
label, positioned deliberately to keep the three narrow panels readable.
The colorbar title does not repeat those levels. All are learned model contours,
not independently established physical regime boundaries.

The shared 180-mm width and 11-pt text style are used. Figure heights are
155 mm (architecture) and 112 mm (three probability panels), chosen to keep
labels and the shared colorbar separate at approximately 620-pixel placement.
Dense probability fields are rasterized at 300 dpi; all other SVG elements
retain editable vector geometry. Text uses actual LaTeX with the minimal
`amsmath` preamble and SVG glyph outlines. The first attempt exposed the
missing `amsmath` preamble for bold alpha; this was corrected without falling
back to another renderer. `--no-tex` remains an explicit optional opt-out.

The generator was rerun into a second empty directory. The JSON result matched
exactly, and every SVG matched exactly after removing the creation-date
metadata. Both actual SVG theme variants were rendered by Inkscape and
visually inspected at 620 pixels wide. An initial axis-label/colorbar spacing
issue was repaired and both probability variants were reinspected. The later
bias-node, scaled-input, layer-indexed-weight, and inline-contour-label revision
was generated into a new empty directory and inspected in both themes at the
same 620-pixel width; training arrays, objective traces, and coefficients were
checked to remain unchanged. Circles,
arrows, titles, model contours, marker shapes, and labels are legible. The
page-theme switching and rendered Quarto source-download checks are performed
as part of the enclosing lecture integration, not by this generator.
