# Computed bubble-concentration and mesh animations

The movies show OpenFOAM v2606 cell fields and moving mesh positions. The
boundary conditions evaluate AOTInductor packages through LibTorch 2.14.1 CPU.

The species fields share one transient flow, with Eo=0.4361, Ga=238.2 and
nu=1/Ga. Their Schmidt numbers are 100, 1000, and 10000. All panels therefore
share the same instantaneous Re(t)=Ga Ub(t); Re changes as the bubble accelerates
and is about 511 at dimensionless time 2. They are not three Reynolds-number
cases. The full view is an axisymmetric meridional slice, reflected about the
rise axis. The void is excluded gas, not zero-concentration liquid.

## Displayed interval and checks

The movies use **200 saved snapshots from t=0.05 to t=10**, spaced by 0.05.
These are from the stability run on a 13,584-cell mesh with 132 bubble faces
and surface grading 0.1. No flow solve was repeated to make the visuals.
The three scalar fields share one flow and were computed without clipping.
All displayed pressure, velocity, and scalar fields were finite. Standard
`checkMesh` passed at all 200 times. The extended check at t=10 flagged
51 low-determinant cells, with positive volumes. The highest-Schmidt layer
approaches the grid scale, so the results remain qualitative.

The run used four MPI ranks from rest. After the t=1 checkpoint, the outer
velocity tolerance was changed from 0.0001 to 0.001. The pressure tolerance
remained 0.0001. See `results/run-summary.json` for field checks and
`stability-investigation.md` for the matched-checkpoint comparison.

## Reproduce the computation in stages

The unchanged Basilisk image supplies hydrodynamic data; the code notebook fits
and exports the revised three-layer, 64-unit tanh models. The training refinement
uses Adam followed by L-BFGS and selects weights by validation loss. The
retrospective model-error comparison and volume/derivative diagnostics live in
`test_cases/single_phase_approach/validation/`.

With the course container/runtime setup and exported models available:

```bash
source setup-env
python scripts/prepare_bubble_animation_case.py \
  --models notebooks/output/bubble-reference \
  --case experiments/bubble_transport/runs/reference
apptainer run "$ML_CFD_IMAGE" bash -c \
  'cd experiments/bubble_transport/runs/reference && ./compile.sh'
```

The preparation script applies `animation-case.json`, starts with a pilot
end time of 0.1, and disables the scalar limiter. The student case already
uses the same mesh. For this run, the outer velocity tolerance was 0.0001
through t=1. We set it to 0.001 for the continuation. The pressure tolerance
is 0.0001 throughout.

From the prepared `simulation/` folder in the container, use a
`system/decomposeParDict` with four subdomains and the `scotch` method.

```bash
export LD_LIBRARY_PATH="$PWD/../build/lib:$TORCH_LIBRARIES/lib:$LD_LIBRARY_PATH"
blockMesh
checkMesh
decomposePar
mpirun -np 4 ../build/bin/scalarPimpleFoam -parallel
# Check the pilot. Set startFrom to latestTime and endTime to 1, then continue.
mpirun -np 4 ../build/bin/scalarPimpleFoam -parallel
# Set outer U tolerance to 0.001 and endTime to 1.05, then run.
# Repeat with endTime 5 and then 10, always starting from latestTime.
mpirun -np 4 ../build/bin/scalarPimpleFoam -parallel
reconstructPar -newTimes -time '0.05:10'
checkMesh -time '0.05:10'
```

Reconstruction and the following post-processing steps can be run directly
on an existing trajectory without solving the flow again.

## Extract and render

With the installed ParaView `pvpython` executable (6.1.1 was used):

```bash
pvpython scripts/extract_bubble_animation.py \
  --case experiments/bubble_transport/runs/reference/simulation \
  --output experiments/bubble_transport/frames/reference --max-time 10
python scripts/render_bubble_animations.py \
  --frames experiments/bubble_transport/frames/reference \
  --models notebooks/output/bubble-reference --output /tmp/bubble-movies
```

Extraction starts with the actual z=0 slice and retains cell-associated values.
Triangles are padded by a repeated vertex for plotting; no geometry is changed.
The interface is recovered from actual slice boundary edges. Roundoff-level
signed zeros on the symmetry axis are snapped to positive zero before angular
sorting, preventing a spurious line through the gas interior. There is no
cell-to-point interpolation or smoothing. Raw fields and extracted arrays are
kept in ignored `runs/` and `frames/` folders locally, outside website resources.

The editable renderer uses shared course styling, explicitly tested LaTeX,
180 mm width, 11 pt main text, 10 pt supporting labels, and 9 pt footer text as
a deliberate animation exception. Both themes use the same coordinates, values,
limits, cameras, time samples, label positions and viridis [0,1] normalization.
Only neutral styling/backgrounds differ (white / #111111); interface and marker
accents retain C1 and C3. Video dimensions are 1280 x 900 (even), H.264/yuv420p,
200 frames at 8 fps, 25 seconds per loop. PNG posters use the final displayed
time. Mathematical labels use TeX glyph rendering, with no fallback.

The concentration movie fixes the near-bubble camera for all three Schmidt
numbers. The mesh movie shows actual cell edges plus an explicitly moving local
view near theta=pi/4. The local basis follows the interface tangent and outward
normal; translation/rotation do not stretch either direction. Its colors show
Sc=1000. This is a view transformation, not a remeshed or interpolated field.

Production exports were generated in a fresh directory. Both theme posters and
intermediate frames were inspected. The two movies were embedded with matching
light/dark classes and equal widths. `results/movie-checks.json` records stream
properties and SHA256 values; `field-checks.json` records all shown frames.

## Local Sherwood extraction

From the simulation folder in the course container, run `./Allpost -time '0.05:10'`.
OpenFOAM evaluates patch `snGrad` for the three concentration fields and writes
local Sherwood values, face centres, and area vectors as raw tables.
`surfaceFieldValue` calculates the area-weighted means directly. To prepare the
course CSV files and plots, use the generated `surfaceFieldValue.dat` path:

```bash
python scripts/extract_bubble_sherwood.py \
  --surface CASE/postProcessing/sherwood/surface \
  --mean-file CASE/postProcessing/sherwood/TIME/surfaceFieldValue.dat \
  --output EMPTY_DATA_DIR --max-time 10
python scripts/plot_bubble_sherwood.py --data EMPTY_DATA_DIR --output EMPTY_FIGURE_DIR
```

The profiles show the local values at t=.05, .5, and 10 alongside the OpenFOAM
mean histories. The CSV export checks the direct means against the local tables
and retains the OpenFOAM values. It preserves all 132 faces at each of the 200
times. The highest-Schmidt concentration layer approaches the grid scale.
