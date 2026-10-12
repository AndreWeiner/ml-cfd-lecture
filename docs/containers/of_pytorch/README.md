# OpenFOAM.com + LibTorch containers

The default image contains Ubuntu 26.04, **OpenFOAM.com v2606**, and **CPU LibTorch 2.14.1**. These are the current OpenFOAM.com release and latest stable PyTorch release checked on 12 October 2026. The image contains the C++ library, not the Python PyTorch package. For Python-generated models, use the matching Python PyTorch release.

Sources: [OpenFOAM v2606](https://www.openfoam.com/news/main-news/openfoam-v2606), [PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1), and the [official CPU archive index](https://download.pytorch.org/libtorch/cpu/).

LibTorch now requires **C++20**. OpenFOAM v2606 uses C++17 by default; the example's `Make/options` selects C++20 for that application, and the CMake example selects C++20 explicitly. OpenFOAM's installed compiler rules are not modified. The CPU archive uses the C++11 ABI; the archive naming no longer includes `cxx11-abi`. Its checksum is recorded in both recipes.

## Pull the published Apptainer container

The tested image is hosted in [GitHub Container Registry](https://github.com/users/AndreWeiner/packages/container/package/of-pytorch). It contains **Ubuntu 26.04.1 LTS**, **OpenFOAM.com v2606**, and **CPU LibTorch 2.14.1**.

```bash
apptainer pull of2606-py2.14.1-cpu.sif \
  oras://ghcr.io/andreweiner/of-pytorch:of2606-libtorch2.14.1-ubuntu26.04-cpu
apptainer run of2606-py2.14.1-cpu.sif simpleFoam -help
apptainer run of2606-py2.14.1-cpu.sif /opt/of-pytorch-tests/run-tests.sh
```

Use **`oras://`** for this image: it is an Apptainer SIF artifact, rather than a Docker image. It includes LibTorch, the C++ frontend; install Python PyTorch separately in your Python environment. You can alternatively build a Docker image locally using the recipe below.

The SHA256 of the published SIF is:

```text
5753c889670fbc305a42c264b493e2d737682512709041eff948a88b2db05988
```

Check a download using `sha256sum of2606-py2.14.1-cpu.sif`. The examples passed during the build and again inside the finished image; the OpenFOAM startup check also passed. These checks were performed on 12 October 2026.

## Build an Apptainer image directly

From the repository root:

```bash
apptainer build --fakeroot of2606-py2.14.1-cpu.sif Singularity.def
```

This builds from Ubuntu, installs OpenFOAM from its official package repository, downloads LibTorch, and compiles and runs both examples. It does not require an image to have been published to Docker Hub. Fakeroot requires suitable host configuration; see the [Apptainer build guide](https://apptainer.org/docs/user/latest/build_a_container.html). If your system does not permit fakeroot, a privileged build is an alternative:

```bash
sudo apptainer build of2606-py2.14.1-cpu.sif Singularity.def
```

Run a command, passing its arguments normally:

```bash
apptainer run of2606-py2.14.1-cpu.sif simpleFoam -help
apptainer run of2606-py2.14.1-cpu.sif /opt/of-pytorch-tests/run-tests.sh
```

The runscript activates OpenFOAM and preserves the command's arguments. For an interactive shell:

```bash
apptainer shell of2606-py2.14.1-cpu.sif
source /usr/lib/openfoam/openfoam2606/etc/bashrc
```

`apptainer exec` bypasses the runscript; activate OpenFOAM explicitly when using it:

```bash
apptainer exec of2606-py2.14.1-cpu.sif bash -c \
  'source "$FOAM_PATH/etc/bashrc" && /opt/of-pytorch-tests/run-tests.sh'
```

## Build a Docker image

```bash
docker build -t of_pytorch:of2606-py2.14.1-cpu .
docker run --rm of_pytorch:of2606-py2.14.1-cpu bash -c \
  'source "$FOAM_PATH/etc/bashrc" && /opt/of-pytorch-tests/run-tests.sh'
```

Both compilation examples run during the image build. The default `OPENFOAM_VERSION`, `TORCH_VERSION`, and `LIBTORCH_SHA256` are build arguments. When selecting another LibTorch version, provide its archive checksum too. Other release combinations require their own compatibility tests.

To convert a locally built Docker image into Apptainer:

```bash
apptainer build of2606-py2.14.1-cpu.sif docker-daemon://of_pytorch:of2606-py2.14.1-cpu
```

The convenience scripts `create_openfoam_container.sh` and `start_openfoam.sh` use the locally built Docker image `of_pytorch:of2606-py2.14.1-cpu`. Pass a different Docker image name as the first argument to `create_openfoam_container.sh` if desired. The GHCR artifact above is a SIF image and cannot be used with `docker run`.

The files `Dockerfile.no_abi` and `Dockerfile.abi_source` document historical build approaches. They retain their older releases and are not current supported recipes.

## Compilation and runtime checks

`test/run-tests.sh` copies the examples into a temporary directory, builds them, runs them, and removes the temporary build artifacts. It prints the active OpenFOAM version/build settings and LibTorch version. The `tensorCreation` example links OpenFOAM as well as LibTorch, so it exercises the combination rather than only standalone LibTorch compilation.

- **tensorCreation** uses `wmake`, initializes OpenFOAM's argument handling, compares the autograd derivative of sine with cosine, checks tensor values, and checks a save/load round trip.
- **simpleMLP** uses CMake and LibTorch's compiler flags, runs a deterministic forward pass, and checks the output shape and finite values.

The build fails if compilation or any of these checks fails. A successful run verifies these examples; it does not establish compatibility for every downstream solver, model, CUDA configuration, or MPI application.

To run against an existing installation:

```bash
source /usr/lib/openfoam/openfoam2606/etc/bashrc
export TORCH_LIBRARIES=/path/to/libtorch
./test/run-tests.sh
```

Use `BUILD_JOBS=1` if compilation exceeds available memory. The default is two CMake compilation jobs; the small wmake example contains one source file.
