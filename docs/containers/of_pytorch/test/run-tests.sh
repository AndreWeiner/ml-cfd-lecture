#!/usr/bin/env bash
set -euo pipefail
: "${TORCH_LIBRARIES:?Source the container environment or set TORCH_LIBRARIES}"
: "${WM_PROJECT_DIR:?Source the OpenFOAM environment first}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/of-pytorch-tests.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
cp -R "$script_dir/tensorCreation" "$script_dir/simpleMLP" "$work_dir/"
printf 'OpenFOAM: %s; configuration: %s\n' "$WM_PROJECT_VERSION" "$WM_OPTIONS"
printf 'LibTorch: %s\n' "$(cat "$TORCH_LIBRARIES/build-version")"
cd "$work_dir/tensorCreation"
wmake
./tensorCreation
cd "$work_dir/simpleMLP"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "${BUILD_JOBS:-2}"
./build/simpleMLP
printf 'Both wmake and CMake examples compiled and passed.\n'
