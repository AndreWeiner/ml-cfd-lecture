#!/usr/bin/env bash
set -e
: "${WM_PROJECT_DIR:?Activate OpenFOAM.com v2606 first}"
: "${TORCH_LIBRARIES:?Set TORCH_LIBRARIES to the container LibTorch directory}"
cd -- "$(dirname -- "$0")"
export FOAM_USER_APPBIN="$PWD/build/bin"
export FOAM_USER_LIBBIN="$PWD/build/lib"
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
for component in aotiInletVelocity aotiSurfaceVelocity aotiBoundaryDisplacement; do
    (cd "$component" && wmake libso)
done
for component in scalarPimpleFoam; do
    (cd "$component" && wmake)
done
