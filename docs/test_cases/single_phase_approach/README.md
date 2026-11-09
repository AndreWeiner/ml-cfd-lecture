# Liquid-only bubble transport with AOTInductor / LibTorch

The complementary code notebook exports CPU float64 compiled `.pt2` packages
using `torch.export.export()` and `torch._inductor.aoti_compile_and_package()`.
The boundary conditions use LibTorch's `AOTIModelPackageLoader`, already present
in the unchanged OpenFOAM.com v2606 course container (LibTorch 2.14.1 CPU).
Python export uses the matching PyTorch 2.14.1 CPU and a C++ compiler. Generated
native packages require a compatible platform, C++ runtime and CPU; they are not
universal exchange files. AVX2 generated kernels and one thread are used.

## Environment and build

```bash
source setup-env
export MODEL_DIR="$ML_CFD_BASE/notebooks/output/bubble-reference"
mkdir -p exercises
cp -r test_cases/single_phase_approach exercises/bubble-single-phase
apptainer run "$ML_CFD_IMAGE" bash -c \
  'cd exercises/bubble-single-phase && ./compile.sh && cd simulation && ./Allrun'
```

`TORCH_LIBRARIES=/opt/libtorch` is supplied by the existing container.
`wmake libso` builds the three `aoti*` boundary libraries with local C++20 flags,
then `wmake` builds the existing solver. OpenFOAM itself retains its
installed build settings. Build products stay in `build/bin` and `build/lib`.
No ONNX Runtime download, Python interpreter in OpenFOAM, global compiler-rule
change or container rebuild is required. The Basilisk workflow is unchanged.

The old alternative is TorchScript export and `torch::jit::load()` (deprecated).
ONNX export plus ONNX Runtime is another alternative, requiring a separate
runtime. This example uses neither. A raw `torch.export.save()` archive must
first be compiled/packaged by AOTInductor before this C++ loader can execute it.

## Exchange contract and boundary semantics

Each graph has input `features` and output `prediction`, both float64. The
compiled package expects contiguous row-major input, which the C++ helper provides. Rise-speed
input is `[N,1]` (time); radius and tangential-speed inputs are `[N,2]` (time,
angle in radians). Output is always `[N,1]`. Dynamic batch sizes include N=1.
All normalization, startup factors, and angular symmetry factors are in the
graph. Files are `ub.pt2`, `r.pt2`, and `ut.pt2`; `contract.json` records the
training domain. These models are for one bubble case, not a parameterized
family of Eötvös/Galilei numbers.

Coordinates use vertical y and nonnegative radial x in the wedge case.
`axis`/`direction` and `normal` are orthogonal unit vectors; `normal` denotes
the meridional-plane normal `(0 0 1)`, not an interfacial normal. Radius is
measured from the specified origin/center. The model angle is computed by
`atan2(radial distance, axial distance)`, with no unsafe inverse cosine.

`common/AotiModel.H` uses the official LibTorch AOTInductor package-loader API. A package
is loaded once per constructed model, shared by patch copies/mappings, and
uses one intra/inter-op thread. Every patch update is batched; empty local
patches return without running a zero-sized AOTInductor tensor. There is no per-face
package construction or Python interpreter in the CFD solver.

The liquid-domain patch normal points into the excluded gas region. The
increasing-angle tangent has the orientation of the training data. Surface
normal velocity uses `fvc::meshPhi(U)/magSf`, reusing the v2606
`movingWallVelocityFvPatchVectorField::Uwall()` construction in
`src/finiteVolume/fields/fvPatchFields/derived/movingWallVelocity/`.
A stationary mesh supplies zero normal flux during initial construction.
The surface boundary condition respects supplied initial field values.
Displacement comes from the predicted radius along the fixed radial ray;
normal motion is never fitted independently. Restart uses the same graph and
time, with no recurrent hidden state; rays remain invariant under radial motion.
The teaching case uses Euler time discretization and runs in serial through
t=3 by default. Its 13,584-cell wedge has 132 bubble faces, 80 layers in the
interface-normal blocks, and surface grading 0.1. The outer velocity and
pressure tolerances are 0.001 and 0.0001, respectively. The animation run
used this mesh on four MPI ranks.

All numerical lengths, velocities, times, viscosity, and diffusion coefficients
use the original Basilisk dimensionless scales (d=1, r0=0.5, nu=1/Ga). Frame
acceleration is absorbed into reduced pressure. Validation of transferred
hydrodynamics and appropriate outer-boundary treatment remains part of the
scientific comparison.

## Reproduction and checks

Python reproduction from the repository root with the course environment active:

```bash
python scripts/validate_bubble_workflow.py \
  --data datasets/bubble_reference --output /tmp/bubble-check
```

To compare the **actual C++ helper** as well, compile the small driver and pass
its path to that script:

```bash
g++ -std=c++20 -O2 -Itest_cases/single_phase_approach/common \
  -isystem "$TORCH_LIBRARIES/include" \
  -isystem "$TORCH_LIBRARIES/include/torch/csrc/api/include" \
  test_cases/single_phase_approach/validation/predict.C \
  -L"$TORCH_LIBRARIES/lib" -Wl,-rpath,"$TORCH_LIBRARIES/lib" \
  -ltorch -ltorch_cpu -lc10 -o /tmp/bubble-predict
python scripts/validate_bubble_workflow.py \
  --data datasets/bubble_reference --output /tmp/bubble-check --cpp /tmp/bubble-predict
```

The selected container uses OpenFOAM v2606, `linux64GccDPInt32Opt`, GCC 15.2,
and LibTorch 2.14.1 CPU. See `validation/aoti-runtime.json` for actual checks of
this integration. The animations use 200 saved snapshots through t=10 from a separate
four-rank stability run with the same 13,584-cell, 132-face mesh. The
settings and checks are in `experiments/bubble_transport/README.md` and
`experiments/bubble_transport/results/run-summary.json`.

The reference reproduction retains 100 snapshots and 6400 sampled points;
227 nodata velocity records are removed explicitly. Entire snapshots are held
out. The file `validation/reference.json` records the physical test errors.
`validation/aoti-runtime.json` records the AOTInductor C++ agreement checks
at batch sizes 1, 7, and 101. Startup,
pole values, even/odd symmetry, and zero initial radius derivative are checked.
The compact data archive reproduces the full local archive's selected training
tables exactly. Its `provenance.json` records the extraction; it is a teaching
subset, not an independent CFD simulation.

Run `./Allpost -noZero` from the simulation folder in the course container
to post-process saved results. `setExprFields` creates derived Sherwood fields.
`setExprBoundaryFields` evaluates each concentration's `snGrad` on the bubble
patch. Both d and the concentration difference are unity in this case.
The `sherwood` function object writes local face-centre tables and computes
area-weighted means in OpenFOAM. The liquid-domain outward normal points into
the bubble. For dimensional cases, scale the gradient by d/(c_interface-c_bulk).
The default limiter can hide overshoots; inspect unclipped diagnostic fields for
accuracy studies. The computed profiles and CSV tables are described in
`experiments/bubble_transport/README.md`; highest-Sc transport is not converged.

The model does not guarantee constant volume, positive radius at arbitrary
extrapolated inputs, or an acceptable deformed mesh. These are assessed in the
code notebook and exercise before drawing transport conclusions.

Official API reference:
[PyTorch AOTInductor deployment](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html).

### Surrogate training

The reference models use three 64-unit tanh layers and Adam followed by
L-BFGS refinement. Validation loss selects the weights. The reported test
comparison is retrospective after inspection of the original diagnostics.
`snapshot-errors.csv`, `geometry.csv`, and `geometry-summary.json` record
temporal errors, radius and volume checks, and finite-difference radius speed.
