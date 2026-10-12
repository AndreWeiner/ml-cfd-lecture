# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Reduced-order modeling of the flow past a cylinder in flowTorch — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](cylinder_rom_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="cylinder_rom_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# **Note: this exercise still needs to be updated for the winter term 2022/2023.**
#
# In this exercise, we build a reduced-order model (ROM) of the flow past a cylinder using the singular value decomposition (SVD) for dimensionality reduction and a cluster-based network model (CNM) to advance the reduced state in time. For both SVD and CNM, we use the implementation available in the *flowTorch* package.
#
# ## Performing the simulation
#
# The starting point for this exercise is the 2D cylinder simulation available in the *test_cases* folder. The goal is to double the Reynolds number in the modified workflow, so let's create a copy of the base simulation and give it a sensible name:
#
# ```
# cp -r test_cases/cylinder2D/ exercises/cylinder2D_Re200
# ```
#
# Before we can run the simulation, some input dictionaries need to be modified:
#
# - adjust the inlet velocity vector such that the new Reynolds number is $Re=U_{in}d/\nu = 200$
# - in the function object computing the lift and drag coefficients, adjust the reference speed
#
# To start the simulation, execute the *Allrun.singularity* script.
#
# ## Inspecting lift and drag coefficients
#
# As in exercise 2, load and plot lift and drag coefficients. Compared to exercise 1, what differences do you notice?
#
# ## Creating a ROM
#
# The main aid to build a ROM in flowTorch is the tutorial *CNM of the flow past a circular cylinder* available in the [flowTorch documentation](https://flowmodelingcontrol.github.io/flowtorch-docs/1.0/notebooks/cnm_cylinder.html). Following the tutorial,
#
# - create a datamatrix based on the masked magnitude of the velocity field ([torch.linalg.norm()](https://pytorch.org/docs/stable/generated/torch.linalg.norm.html)); use only snapshots with $t>1s$
# - create an *SVDEncoder* instance; use a rank truncation of 30
# - create a CNM model; use 40 clusters and a model order of 4
# - create a prediction of the reduced state's temporal evolution; plot the first two dimensions of the visited centroids
# - reconstruct selected snapshots and compare them visually against the simulation data
#
# **Congratulations! This completes the ninth exercise session.**

# %% [markdown]
#
