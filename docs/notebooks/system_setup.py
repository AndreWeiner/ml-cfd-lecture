# %% [markdown]
# ---
# title: Course overview and motivation
# subtitle: Exercise 1 — theory and practical lab
# jupyter: python3
# ---
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# ## Learning objectives
#
# After this exercise, you should be able to:
#
# - decompose a fluid-mechanical application into suitable learning tasks and explain how they connect;
# - identify the required data, propose meaningful checks, and explain which steps can use analytical or numerical methods;
# - set up the course tools on Ubuntu or Windows 11 with WSL 2 and verify the main components.
#
# ## Part 1 — Theory
#
# ### Task 1 — Decompose an application
#
# Choose **two** of the following situations. Draw a workflow and explain your design; several reasonable approaches are possible.
#
# 1. **Find leaks in a water-distribution network.** Pressure and flow-rate sensors record several locations in a pipe network. Water demand changes throughout the day, and some sensors occasionally fail. You have records from normal operation and a few controlled leak tests at known locations. You want to detect a leak and narrow down its location without mistaking ordinary changes in demand for a fault.
# 2. **Choose a cyclone separator design.** A cyclone removes solid particles from a gas stream. Simulations and experiments are available for a limited set of geometries, flow rates, and particle-size distributions. The measurements include pressure loss and the fraction of particles collected. You want to select a geometry that meets a specified collection requirement with low pumping power, while limiting the number of additional simulations or experiments.
# 3. **Operate ventilation in a changing environment.** A room has adjustable supply fans and dampers. Sensors measure temperature and carbon-dioxide concentration at several locations, while occupancy and outdoor conditions vary. You have historical operating records and can test candidate strategies in a room-flow simulation. You want to maintain acceptable air quality and comfort while reducing energy consumption.
# 4. **Plan a large CFD parameter study.** Pilot simulations of flow through a porous structure have produced solver logs, numerical settings, mesh statistics, and a few physical output quantities. Some runs converged quickly, some stalled, and some crashed. You want to identify recurring kinds of simulation behaviour and choose settings and computing resources for the remaining cases. A short run should help you decide whether to continue, change the settings, or stop.
#
# For each chosen situation:
#
# 1. Define the physical question and the desired output.
# 2. Identify the available data and any additional data you would need. Distinguish snapshots, derived quantities, and secondary data where relevant.
# 3. Divide the problem into subproblems. Decide which require regression, classification, dimensionality reduction, clustering, or reinforcement learning, and justify your choices. You do not need to use every category.
# 4. Specify the inputs and outputs of each step. For supervised tasks, distinguish features, feature values, targets, and individual samples. For control, specify the state, action, and reward.
# 5. Explain where domain knowledge, analytical models, or classical numerical methods could help. Identify information that must survive any reduction step.
# 6. Sketch how the steps connect using the extract–learn–use workflows from Lecture 1.
# 7. Propose checks that could reveal failure. Consider data coverage, independent test cases, physical consistency, and the consequences of an incorrect prediction or action.
#
# ### Task 2 — Challenge your proposed workflow
#
# Exchange one design with a partner. Identify one assumption that could fail, one simpler alternative, and one experiment or numerical check that would distinguish the alternatives. Revise your design and state what remains uncertain.
#
# No reference solution is provided. Your aim is to make a credible case for the workflow, rather than merely name an algorithm. Explain your reasoning and use the learning and verification guidance from [Lecture 0](lecture_00.qmd).
#
# ## Part 2 — Practical lab: setting up your system
#
# The course uses Linux for Python, CFD tools, and containers. Follow the common Ubuntu steps after choosing either native Ubuntu or Windows 11 with WSL 2. The commands below assume an **x86-64/AMD64 computer**; the legacy course container images and PyTorch wheels may require a different setup on ARM-based devices.
#
# Use **Python 3.10 or newer**, with **Python 3.12 recommended** for this course. Ubuntu **26.04 LTS** provides Python 3.14 as its default, so Step 4 installs a separate Python 3.12 environment. Python 3.12 is the tested course baseline; Ubuntu's system Python remains unchanged. A CPU-only installation is sufficient to start; a GPU is not required. Keep space for the Linux distribution, Python packages, container images, and later datasets. The old estimate of 8 GB for the tools alone is not a reliable allowance for the full course.
#
# ### Step 1 — Get access to Ubuntu
#
# #### Windows 11: install WSL 2
#
# Install Windows updates, then open **PowerShell as administrator**:
#
# ```powershell
# wsl --list --online
# wsl --install -d Ubuntu-26.04
# ```
#
# Restart if requested, open Ubuntu, and create your Linux username and password. Check from PowerShell:
#
# ```powershell
# wsl --update
# wsl --list --verbose
# ```
#
# Ubuntu should show **VERSION 2**. To convert an existing version-1 installation, use its exact distribution name:
#
# ```powershell
# wsl --set-version Ubuntu-26.04 2
# ```
#
# These commands follow [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install). From this point onward, run Linux commands **in the Ubuntu terminal**, not PowerShell.
#
# #### Native Linux
#
# If you already use Ubuntu 26.04, continue below. Otherwise, follow the [Ubuntu installation guide](https://ubuntu.com/tutorials/install-ubuntu-desktop). Installing a second operating system involves disk partitioning; WSL 2 avoids that step on Windows 11.
#
# ### Step 2 — Install basic tools, including the compiler
#
# In Ubuntu:
#
# ```bash
# sudo apt update
# sudo apt upgrade
# sudo apt install git build-essential python3 python3-venv python3-pip python3-dev software-properties-common cmake wget unzip
# ```
#
# A minimal WSL installation may lack a compiler. **Install `build-essential`; you do not need to compile GCC yourself.** It supplies GCC, G++, Make, and supporting development packages. Check:
#
# ```bash
# gcc --version
# g++ --version
# make --version
# python3 --version
# ```
#
# The system Python should report 3.14 on Ubuntu 26.04. The course environment created below uses Python 3.12 instead. See [Ubuntu's build-essential package](https://packages.ubuntu.com/resolute/build-essential).
#
# ### Step 3 — Clone the course into the Linux home directory
#
# ```bash
# mkdir -p ~/Development
# cd ~/Development
# git clone https://github.com/AndreWeiner/ml-cfd-lecture.git
# cd ml-cfd-lecture
# mkdir -p exercises
# ```
#
# If you already have a checkout, use it rather than cloning again. Keep your own work in `exercises/`, which is ignored by Git. Run `git pull` from the repository root to receive updates; inspect your local changes before resolving any conflicts.
#
# With WSL, keep the checkout under `~/Development`, rather than `/mnt/c/...`, for Linux filesystem performance and permissions. Open the current folder in Windows Explorer with `explorer.exe .`. See [Microsoft's filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).
#
# ### Step 4 — Create the Python environment
#
# From the repository root, install `uv` in a small tools environment and use it to download Python 3.12 and create the course environment:
#
# ```bash
# python3 -m venv ~/.venvs/course-tools
# ~/.venvs/course-tools/bin/python -m pip install --upgrade uv
# ~/.venvs/course-tools/bin/uv venv --python 3.12 --seed ml-cfd
# source ml-cfd/bin/activate
# python -m pip install --upgrade pip
# python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# python -m pip install -r requirements.txt
# python -m pip check
# ```
#
# `uv` downloads a managed Python interpreter when the requested version is missing. See its [Python installation guide](https://docs.astral.sh/uv/guides/install-python/). No Python compilation or replacement of Ubuntu's system Python is required. After activation, check `python --version`; it should report 3.12.
#
# The CPU installation follows [PyTorch's version-specific instructions](https://docs.pytorch.org/get-started/previous-versions/). We retain PyTorch **2.14.1** because the course's OpenFOAM container and manual setup use LibTorch **2.14.1**. Do not independently upgrade one side of that pairing. Most other packages remain unpinned, so successful dependency resolution and the checks below are important. The course uses `torch.export` for model export. Older exercises may still contain TorchScript calls until they are revised. Later examples will need their own compatibility checks.
#
# Install the latest available **FlowTorch** release from PyPI. The distribution is named `flowtorch-fluid`, while Python imports remain `flowtorch`. The latest published release checked for this revision is **1.9.0**; install the `all` extra to include the data loaders, reduced-order modelling, and pressure-sensitive-paint dependencies:
#
# ```bash
# python -m pip install --upgrade "flowtorch-fluid[all]"
# ```
#
# See [FlowTorch's package instructions](https://pypi.org/project/flowtorch-fluid/). The removed `aweiner` Git branch is no longer used. Newer releases may change behavior, so record the installed version and check each later application with its required data.
#
# In each new terminal, reactivate the environment:
#
# ```bash
# cd ~/Development/ml-cfd-lecture
# source ml-cfd/bin/activate
# ```
#
# Use `python -m pip` inside this environment; do not install course Python packages with `sudo pip`. Leave the environment with `deactivate`.
#
# ### Step 5 — Verify Python and run a small script
#
# Save the following as `exercises/setup_check.py`:
#
# ```python
# import sys
# import warnings
# from importlib.metadata import version
# from pathlib import Path
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import pandas
# import sklearn
# import torch
# # Some dependencies currently import legacy TorchScript code internally.
# # Hide only its known deprecation warning; this check uses torch.export.
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore", message=r"`torch\.jit\.script` is deprecated\..*",
#         category=FutureWarning, module=r"torch\.jit\._script",
#     )
#     import botorch
#     import gpytorch
#     import flowtorch
#     from flowtorch import DATASETS
#     from flowtorch.data import FOAMDataloader, mask_box
#
# print("Python:", sys.version.split()[0])
# print("Interpreter:", sys.executable)
# print("PyTorch:", torch.__version__)
# print("FlowTorch:", version("flowtorch-fluid"))
# assert torch.__version__.split("+")[0] == "2.14.1"
# x = torch.tensor([1.0, 2.0, 3.0])
# assert torch.isclose(x.mean(), torch.tensor(2.0))
#
# class ExampleModel(torch.nn.Module):
#     def forward(self, x):
#         return x.square() + 1.0
#
# model = ExampleModel().eval()
# exported = torch.export.export(model, (x,))
# torch.testing.assert_close(exported.module()(x), torch.tensor([2.0, 5.0, 10.0]))
# export_path = Path(__file__).with_name("setup-check.pt2")
# torch.export.save(exported, export_path)
# loaded = torch.export.load(export_path)
# test_input = torch.tensor([0.0, -2.0, 4.0])
# torch.testing.assert_close(loaded.module()(test_input), torch.tensor([1.0, 5.0, 17.0]))
# print("PyTorch export, save/load, and inference passed:", export_path)
#
# t = np.linspace(0, 2 * np.pi, 100)
# fig, ax = plt.subplots()
# ax.plot(t, np.sin(t))
# ax.set(xlabel="t", ylabel="sin(t)")
# path = Path(__file__).with_name("setup-check.png")
# fig.savefig(path)
# plt.close(fig)
# print("Saved:", path)
# print("Core package imports, PyTorch export, and basic calculations passed.")
# ```
#
# Run it from the repository root:
#
# ```bash
# python exercises/setup_check.py
# ```
#
# Inspect the saved sine curve. The script also exports a small model, saves it as `setup-check.pt2`, reloads it, and checks its predictions against known values. Some third-party dependencies still import legacy TorchScript code internally, so the script suppresses only that specific warning during those imports. These checks confirm imports, model export, and basic execution, not the correctness of every later course example. FlowTorch's example datasets are separate from the Python package; later chapters explain which data are needed.
#
# The course sources are Python scripts with `# %%` cells. You can run your own scripts from the terminal or use an editor that supports individual cells. JupyterLab is optional:
#
# ```bash
# cd exercises
# jupyter lab --no-browser
# ```
#
# Open the printed localhost URL, including its token, in your Windows or Linux browser. Keep the server terminal open; stop it with Ctrl+C. The browser token is private. You do not need JupyterLab to run ordinary Python scripts.
#
# ### Step 6 — Install the latest stable ParaView
#
# Use the latest stable binary from the [official ParaView download page](https://www.paraview.org/download/), rather than Ubuntu's potentially older `apt` package. The latest stable release checked on 12 October 2026 is **6.2.0**; avoid nightly builds and release candidates for the course.
#
# On Windows 11, install the Windows AMD64 package and open case files through the WSL filesystem in Explorer. This keeps the visualization application on Windows while the CFD tools run in Ubuntu.
#
# On native Ubuntu, download the Linux x86-64 archive. For the verified 6.2.0 archive, extraction and launch are:
#
# ```bash
# mkdir -p ~/Applications
# tar -xzf ~/Downloads/ParaView-6.2.0-MPI-Linux-Python3.12-x86_64.tar.gz -C ~/Applications
# ~/Applications/ParaView-6.2.0-MPI-Linux-Python3.12-x86_64/bin/paraview --version
# ~/Applications/ParaView-6.2.0-MPI-Linux-Python3.12-x86_64/bin/paraview
# ```
#
# If the download page offers a newer stable release, use its corresponding archive and directory names. ParaView includes its own Python runtime; do not add its Python packages to the course environment's `PYTHONPATH`.
#
# Linux GUI applications can also run through WSLg on supported Windows 11 installations. See [Microsoft's WSL GUI guidance](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).
#
# ### Step 7 — Install Apptainer
#
# In Ubuntu, install Apptainer from its official PPA:
#
# ```bash
# sudo add-apt-repository -y ppa:apptainer/ppa
# sudo apt update
# sudo apt install apptainer
# apptainer --version
# apptainer exec docker://alpine cat /etc/alpine-release
# ```
#
# This uses prebuilt packages and avoids compiling Go or Apptainer. The final command should print the container's Alpine release. The [Apptainer installation guide](https://apptainer.org/docs/admin/latest/installation.html#install-ubuntu-packages) documents this installation route and WSL 2 support.
#
# ### Step 8 — Set up OpenFOAM and LibTorch
#
# Choose **Route A (container)** or **Route B (manual installation)** for OpenFOAM/LibTorch. Both use OpenFOAM.com **v2606** and CPU LibTorch **2.14.1**. Do not mix the OpenFOAM.com and OpenFOAM Foundation distributions when following these commands.
#
# #### Route A — Pull the course Apptainer container
#
# The container uses **Ubuntu 26.04.1 LTS** as its base, with OpenFOAM.com **v2606** and CPU LibTorch **2.14.1**. Its Ubuntu version is independent of the host operating system; the Ubuntu 26.04/WSL host setup above can run this container. Download the tested image from [GitHub Container Registry](https://github.com/users/AndreWeiner/packages/container/package/of-pytorch), then run the acceptance checks:
#
# ```bash
# apptainer pull of2606-py2.14.1-cpu.sif \
#   oras://ghcr.io/andreweiner/of-pytorch:of2606-libtorch2.14.1-ubuntu26.04-cpu
# apptainer run of2606-py2.14.1-cpu.sif simpleFoam -help
# apptainer run of2606-py2.14.1-cpu.sif /opt/of-pytorch-tests/run-tests.sh
# ```
#
# Use `oras://` for this registry artifact; it is an Apptainer SIF image. The runscript activates OpenFOAM automatically. You do not need to build the container yourself.
#
# If you prefer to build it locally, the definition and compilation examples are included in `containers/of_pytorch/`. From the course repository root:
#
# ```bash
# cd containers/of_pytorch
# apptainer build --fakeroot of2606-py2.14.1-cpu.sif Singularity.def
# cd ../..
# cp containers/of_pytorch/of2606-py2.14.1-cpu.sif .
# ```
#
# The build compiles and runs the C++ examples. If fakeroot is unavailable on your computer, use `sudo apptainer build` instead.
#
# The container contains **LibTorch**, the C++ frontend; your Python virtual environment remains on the host. For an interactive container shell, activate OpenFOAM explicitly:
#
# ```bash
# apptainer shell of2606-py2.14.1-cpu.sif
# source /usr/lib/openfoam/openfoam2606/etc/bashrc
# ```
#
# #### Route B — Install OpenFOAM and LibTorch manually
#
# Run these commands in Ubuntu, including Ubuntu inside WSL 2. Download and inspect the official repository setup script before running it:
#
# ```bash
# wget -O /tmp/add-openfoam-repo.sh https://dl.openfoam.com/add-debian-repo.sh
# less /tmp/add-openfoam-repo.sh
# sudo bash /tmp/add-openfoam-repo.sh
# sudo apt update
# sudo apt install openfoam2606-dev
# source /usr/lib/openfoam/openfoam2606/etc/bashrc
# simpleFoam -help
# wmake -help
# ```
#
# This installs precompiled OpenFOAM packages and their development files; it does not compile OpenFOAM from source. See the [OpenFOAM.com release and installation links](https://www.openfoam.com/news/main-news/openfoam-v2606).
#
# Install the CPU LibTorch archive into your home directory:
#
# ```bash
# mkdir -p ~/Applications
# cd ~/Applications
# wget -O libtorch-2.14.1-cpu.zip 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.14.1%2Bcpu.zip'
# echo '0f02329a5266153cffa9dc12926ff2221fa626207b894d40bd61905cc2079821  libtorch-2.14.1-cpu.zip' | sha256sum -c -
# mkdir -p libtorch-2.14.1
# unzip libtorch-2.14.1-cpu.zip -d libtorch-2.14.1
# export TORCH_LIBRARIES="$HOME/Applications/libtorch-2.14.1/libtorch"
# cat "$TORCH_LIBRARIES/build-version"
# ```
#
# The checksum records the official archive used for this revision. Avoid unpacking over an existing installation. See [PyTorch's LibTorch installation guide](https://docs.pytorch.org/cppdocs/installing.html).
#
# For each terminal used to compile OpenFOAM/LibTorch code, source OpenFOAM's `etc/bashrc` and set `TORCH_LIBRARIES` as above. LibTorch 2.14.1 requires **C++20**. The examples select it per application without modifying OpenFOAM's global compiler rules. Include LibTorch headers before OpenFOAM headers to avoid the `TypeName` macro clash.
#
# Verify the combined toolchain from the course root:
#
# ```bash
# cd ~/Development/ml-cfd-lecture
# source /usr/lib/openfoam/openfoam2606/etc/bashrc
# export TORCH_LIBRARIES="$HOME/Applications/libtorch-2.14.1/libtorch"
# bash containers/of_pytorch/test/run-tests.sh
# ```
#
# This compiles and runs both the `wmake` tensor example and the CMake neural-network example. The checks include an analytical derivative comparison, tensor serialization, and output shape and finite-value checks.
#
# #### Basilisk for later exercises
#
# The existing Basilisk exercise still uses its separate container:
#
# ```bash
# apptainer pull basilisk.sif docker://andreweiner/basilisk:20-11-19
# apptainer exec basilisk.sif sh -c 'command -v qcc; gcc --version'
# ```
#
# The first CFD exercise will compile and run the provided Basilisk solver. Listing the compiler does not yet validate that simulation. Record download or execution failures and contact the teaching team rather than substituting an arbitrary image.
#
# ### Step 9 — Optional editor setup
#
# On Windows, install **Visual Studio Code on Windows**, add the **WSL** extension, and open the course from an Ubuntu terminal with `code .`. Install the Python extension in the WSL environment and select `ml-cfd/bin/python` as the interpreter. On native Ubuntu, install VS Code using its official instructions. See [VS Code's WSL guide](https://code.visualstudio.com/docs/remote/wsl).
#
# ### Step 10 — Optional: OpenCode with ScaDS.AI models (TUD:AI)
#
# This step is **optional**. You can complete the practical lab without a coding agent.
#
# Before using a coding agent, read [Topic 0 — How to learn with generative AI](lecture_00.qmd) for guidance on using AI tools while developing your own understanding and checking their output.
#
# #### Install OpenCode Desktop
#
# - **Windows 11:** On the [official OpenCode download page](https://opencode.ai/download), select **OpenCode Desktop → Windows (x64)**. Run the downloaded installer and start OpenCode from the Start menu.
# - **Ubuntu Linux:** On the same page, download the **Linux (.deb)** package. Open a terminal in the download folder and install that file with `sudo apt install ./NAME.deb`, replacing `NAME.deb` with its actual filename. Start OpenCode from the applications menu.
#
# Open your `ml-cfd-lecture` repository folder as the project in OpenCode. Keep your own exercise scripts and generated results in `exercises`.
#
# #### Create a TU Dresden API key
#
# 1. Open the [TUD:AI LLM service in the TU Dresden Self-Service Portal](https://selfservice.tu-dresden.de/services/scads-llm-api/).
# 2. Sign in with your TU Dresden account and select **Manage API access** (German: **API-Zugang verwalten**).
# 3. Follow the portal's prompts to create a key and accept the service's usage terms.
# 4. Keep the key private. Enter it only in the agent's provider connection; do not put it in exercise scripts, shared screenshots, or a Git repository.
#
# The public service description currently lists **employees** as the eligible group, while the TUD:AI documentation describes access for TU Dresden affiliates more broadly. If API access is unavailable for your student account, ask the lecturer or tutor about course access and continue without the optional agent. See the [TUD:AI documentation](https://llm.scads.ai/docs/) for service details.
#
# #### Connect OpenCode to ScaDS.AI models
#
# Create `opencode.json` in the repository root. This local configuration is excluded from Git tracking. It sets **GLM 5.3 Flash** as the default and includes GLM 5.3 and Qwen3.8 27B as alternatives for course work.
#
# ```json
# {
#   "$schema": "https://opencode.ai/config.json",
#   "model": "scads/zai-org/GLM-5.3-Flash",
#   "provider": {
#     "scads": {
#       "name": "TUD:AI",
#       "npm": "@ai-sdk/openai-compatible",
#       "options": {
#         "baseURL": "https://llm.scads.ai/v1"
#       },
#       "models": {
#         "zai-org/GLM-5.3-Flash": {
#           "name": "GLM 5.3 Flash",
#           "limit": {
#             "context": 1048576,
#             "output": 32768
#           }
#         },
#         "zai-org/GLM-5.3": {
#           "name": "GLM 5.3",
#           "limit": {
#             "context": 524288,
#             "output": 32768
#           }
#         },
#         "Qwen/Qwen3.8-27B": {
#           "name": "Qwen3.8 27B",
#           "limit": {
#             "context": 262144,
#             "output": 32768
#           }
#         }
#       }
#     }
#   }
# }
# ```
#
# Models can change or become unavailable. Check the [TUD:AI model availability page](https://llm.scads.ai/status/) to see which models are currently available, and update the model identifiers in this configuration if necessary. The default `model` uses the format `provider/model-id`.
#
# Restart the project after saving the configuration. Open the provider connection dialog (`/connect` in the command interface), choose **TUD:AI** (provider ID `scads`), and enter your portal key. If you need to add it as a custom provider, use the same provider ID, `scads`. Use the model picker (`/models` in the command interface) to switch between the configured models.
#
# The configuration contains no API key. Requests use the TUD:AI endpoint. See the [OpenCode custom-provider guide](https://opencode.ai/docs/providers/#custom-provider) for further configuration details.
#
# On Windows, course Python commands must run inside Ubuntu/WSL, rather than with a Windows Python interpreter. Check the command and interpreter path before allowing the agent to run it.
#
# For a first check, ask the agent to explain `exercises/setup_check.py`. Tell it to use the Python interpreter inside `ml-cfd` when running course code. Read its suggestions and verify the result yourself.
#
# ### Troubleshooting Windows 11 and Ubuntu
#
# | Symptom | What to check |
# |:--------|:--------------|
# | WSL installation remains at 0%. | Try `wsl --install --web-download -d Ubuntu-26.04` in administrator PowerShell. |
# | Ubuntu reports version 1. | Convert the exact distribution name using `wsl --set-version NAME 2`. |
# | A required virtualization feature is missing, for example error `0x80370102` or `0x80370114`. | Check that Virtual Machine Platform is enabled, hardware virtualization is enabled in the firmware, and Windows has restarted. Follow the manufacturer's firmware instructions. |
# | Error `0x8007019e` reports that WSL is not enabled. | Enable the Windows Subsystem for Linux component in Windows features and restart. |
# | A kernel update is requested. | Run `wsl --update`, then `wsl --shutdown`, and reopen Ubuntu. The manual kernel MSI belongs to the older installation route. |
# | `gcc`, `g++`, or `make` is missing. | Install `build-essential` inside Ubuntu; a Windows compiler does not supply Linux build tools. |
# | `venv` creation fails or pip reports an externally managed environment. | Install `python3-venv` for the tools environment, then create `ml-cfd` with `~/.venvs/course-tools/bin/uv venv --python 3.12 --seed ml-cfd` as in Step 4 and activate it. |
# | Python cannot import a course package. | Check `which python`, reactivate `ml-cfd`, and run `python -m pip check`. Preserve any dependency-conflict messages. |
# | Linux file operations are slow or permissions behave unexpectedly. | Move the checkout and simulation cases into the Linux home directory. |
# | Jupyter does not open a browser automatically. | Use `jupyter lab --no-browser` and open its printed localhost URL manually. |
#
# For WSL errors, use [Microsoft's troubleshooting guide](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting). Diagnose the reported problem rather than disabling Windows security features or reinstalling the distribution.
#
#
# ### Completion record
#
# Record your operating system, WSL version if applicable, Python and PyTorch versions, compiler version, Apptainer version, and the outcome of each check. Include the plot and any unresolved errors. Be prepared to explain which tools run on Windows, which run in Ubuntu, and which run inside a container.
