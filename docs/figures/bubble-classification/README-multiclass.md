# Five-class MLP: one continued training trajectory

The generator is `scripts/create_bubble_multiclass_figures.py`. Input coordinates
and class labels come from
`datasets/path_shape_regimes/tripathi2015_digitization/points.csv`; the data
source and digitization limitations are documented in that directory. The JSON
records the input and generator SHA-256 hashes, split point IDs, fitted feature
extrema, numerical checks, full loss/error traces and checkpoint protocol.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bubble-multiclass-mpl \
  .venv-quarto/bin/python scripts/create_bubble_multiclass_figures.py \
  --output-dir /tmp/bubble-multiclass-new
```

Start with an empty output directory. The default audit directory is
`figures/bubble-classification`; `--results-dir` can redirect it. SVGs use LaTeX
glyph outlines, editable vector axes/markers/boundaries, and rasterized region
fills at 300 DPI. Both themes use the same categorical C0–C4 colors, predictions,
data, coordinates and layout; only backgrounds and neutral strokes change.
The shared width is 180 mm; heights are 110 mm for the three snapshots and
115 mm for the loss/restored-model pair. Inspect the exported SVGs at the course's
approximately 620-pixel display width. `--no-tex` is an explicit MathText opt-out;
it is not selected automatically. The delivered assets use LaTeX.

## Fixed protocol

- All 104 points are split before scaling. Independent RNGs with seed 42 create
  the stratified split and initialize the one model. Validation counts are the
  nearest integers to 20% of each class count: I 4, II 2, III 5, IV 3, V 6.
  Training therefore contains 84 points and validation 20. There is no test set.
- Log10/min-max feature scaling is fitted only to the 84 training rows and then
  applied unchanged to both sets. The input order is Ga, Eo.
- Architecture: 2–32–32–5, tanh hidden layers, softmax output, Glorot-normal
  weights and zero biases. The mean categorical cross-entropy is optimized by
  full-batch Adam with learning rate 0.01, beta1 0.9, beta2 0.999 and epsilon 1e-8.
  There is no weight decay or other explicit regularization.
- Exactly 10,000 updates are performed. Epoch 0 is initialization; epoch n is
  the state after n updates. All 10,001 train/validation loss and error samples
  come from this same trajectory. No independent fits are used for the maps.
- Early stopping is simulated with patience 500 and min_delta 0: any strictly
  lower validation loss resets patience. At epoch 557 the rule would stop and
  restore epoch 57. The actual run continues to illustrate the later behavior.
- Maps show epochs 10, 57 and 10,000. Epoch 10 comes from the predefined
  checkpoint list and illustrates early learning; epoch 57 is validation-selected;
  epoch 10,000 is the fixed final budget. These illustrative choices are not an
  unbiased model-performance comparison.

## Results and interpretation

| Epoch | Training mean CE | Validation mean CE | Training errors | Validation errors |
|---|---:|---:|---:|---:|
| 10 | 0.696594640 | 0.733127884 | 22/84 | 4/20 |
| 57 | 0.248676838 | 0.383469238 | 8/84 | 2/20 |
| 10,000 | 7.23207583e-7 | 2.444898904 | 0/84 | 2/20 |

The later validation cross-entropy increases while training cross-entropy falls.
The number of validation errors at the best and final checkpoints is identical:
cross-entropy also responds to the confidence assigned to the observed labels.
Validation is used for checkpoint selection, so it does not supply independent
test performance, calibrated-probability evidence or physical regime boundaries.

The maps show the argmax of all five softmax probabilities on the same 181×181
grid, with boundaries at changes in the predicted class. They are not five
independent p=0.5 boundaries. Filled small class markers denote training rows;
larger hollow class markers denote validation rows. The loss figure uses a
linear epoch axis through epoch 10 and logarithmic spacing thereafter, explicitly
marked below the plot, so the early minimum and the full late run are visible.

`multiclass-checkpoints.npz` stores only the predefined checkpoints and the
selected best state, with keys `epoch_<n>_parameter_<j>`. Parameter order is
W1, b1, W2, b2, W3, b3; matrices have shape (incoming, outgoing). Every saved
array is checked against its in-memory state, and all checkpoint metrics are
recomputed independently of the recorded trace. The small-network central
finite-difference audit covers all weights and biases in all layers; additional
checks cover softmax normalization, common-logit-shift invariance, the analytic
log(5) loss of zero logits, and the disjoint split/training-only scaler.
