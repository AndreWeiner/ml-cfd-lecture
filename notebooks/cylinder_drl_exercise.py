# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Controlling the flow past a cylinder with OpenFOAM and PyTorch — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](cylinder_drl_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="cylinder_drl_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# This exercise consists of two parts, which are meant to be solved in two individual sessions. We will implement two active flow control approaches to reduce drag and lift forces acting on the surface of a circular cylinder in a stream. The base simulation is the same as in exercise 2. In both control approaches, the flow is actuated by rotating the cylinder. The rotation is easily implemented in a CFD simulation by modifying the velocity boundary condition applied to the *cylinder* patch. Say that $u_\parallel$ is the signed cross-radial speed of the cylinder's surface. Then the surface velocity vector field $\mathbf{u}_\parallel$ is obtained by multiplying the speed with the tangential unit vector $\mathbf{t}_\parallel (\mathbf{x})$:
# $$
#   \mathbf{u}_\parallel = u_\parallel \mathbf{t}_\parallel (\mathbf{x}),
# $$
# where $\mathbf{t}_\parallel (\mathbf{x})$ is tangential to the cylinder's surface (note that the simulation is 2D, so there is only one sensible tangent). The two control approaches vary in the way how they determine $u_\parallel$. In open-loop control, the control law is a function of time, e.g., $u_\parallel = u_\parallel (t)$. The closed-loop control law is a function of pressure sensors $\mathbf{p}$, e.g., $u_\parallel = u_\parallel (\mathbf{p})$.

# %% [markdown]
# ## Open-loop active flow control
#
# As observed in exercise 2, the uncontrolled flow past the cylinder at $Re=100$ yields almost perfectly sinusoidal lift and drag curves in the quasi-steady state (neglecting the non-zero mean). Therefore, we could try to interrupt the natural vortex shedding by rotating the cylinder periodically in alternating directions. A suitable control law might be:
# $$
#   u_\parallel (t) = A \mathrm{sin}(2\pi f t),
# $$
# where the amplitude $A$ and the frequency $f$ are adjustable parameters. To implement this new velocity boundary condition, we use OpenFOAM's [inline expressions for dictionaries](https://www.openfoam.com/news/main-news/openfoam-v1912/pre-processing#pre-processing-expressions-syntax). To get started, create a dedicated folder and make a copy of the base simulation:
# ```
# mkdir exercises/openloop_control
# cp -r test_cases/cylinder2D/ exercises/openloop_control/A2f6
# ```
# The velocity boundary condition is specified in the *0.org/U* dictionary. Open the dictionary and make the following modifications:
#
# 1. currently, there is one entry for the *cylinder*, *top*, and *bottom* patches, because the same boundary condition is applied to these patches (no-slip); remove the *cylinder* patch from the joint entry and create a new separate entry only for *cylinder*
# 2. use the *exprFixedValue* type boundary condition as follows:
# ```
# cylinder
# {
#   type        exprFixedValue;
#   value       $internalField;
#   amplitude   2.0;
#   frequency   6.0;
#   valueExpr   "(vector(0, 0, 1) ^ face()) / area() * ???";
# }
# ```
# Variables can be accessed using the syntax `$variable`. E.g., `$frequency` becomes 6.0 in the example above. The expression `(vector(0, 0, 1) ^ face()) / area()` computes the tangential unit vector. The only missing piece is the expression for $u_\parallel (t)$ in place of the three question marks. You can find the names for the required mathematical functions in the [documentation](https://www.openfoam.com/documentation/guides/latest/doc/openfoam-guide-expression-syntax.html).
#
# To run the actuated simulation, follow the usual steps:
# ```
# # top-level folder of repository
# source setup-env --container
# cd exercises/openloop_control
# ```
# Once the simulation is complete, open the case in ParaView:
# ```
# paraview post.foam
# ```
# and apply the following steps to check if the control works:
# 1. set *Case Type* to *Decomposed Case*
# 2. under *Mesh Regions*, select only the cylinder patch
# 3. click *Apply*
# 4. create a *Glyph* filter and set the *Scale Array* to *U*
# 5. click the play button and observe the surface velocity field
#
# If the control is implemented correctly, we can now check its effect on the forces acting on the cylinder. To do so, create a new Jupyter notebook, load the drag and lift coefficients as in previous exercises, and plot the controlled against the un-controlled coefficients. 
#
# Finally, create one or two additional parameter variations in terms of $f$ and $A$, and repeat the process of evaluating the force coefficients.

# %% [markdown]
# ## Closed-loop active flow control
#
# The agent's aim in the DRL-based closed-loop control is learning a control law $u_\parallel = u_\parallel (\mathbf{p})$ by trail and error. The state vector $\mathbf{p}$ consists of pressure probes from 12 different locations in the cylinder's wake. To train the agent, we employ the [drlFoam](https://github.com/OFDataCommittee/drlfoam) library. This library manages the execution of various parallel simulations and the update of the agent for us. To get started, go to the drlFoam repository, follow the installation instructions, and execute the tutorial as described in the README file.
#
# The output created during the training consists of three types of files (*###* indicates the training episode):
# - *checkpoint_###.pt*: checkpoint with training details and snapshots of networks and optimizers; used to evaluate and restart a training
# - *observations_###.pt:* one or more trajectories of experience tuples based on which the agent learns
# - *policy_trace_###.pt:* snapshot of the traced policy in TorchScript format; used to evaluate a policy
#
# Meanwhile the training is running, create a Jupyter notebook that processes the *observations_###.pt* files and:
#
# 1. visualize the **expected** angular velocity for various episodes
# 2. visualize the drag and lift curves for various episodes
# 3. visualize the mean reward over the episodes
#
# The *observations* are stored as simple Pickle files and can be loaded with `torch.load("file_name.pt")`. Each file contains a list of trajectories. Individual trajectories are organized as dictionaries. The following lines of code demonstrate how to assess the available trajectory content:
#
# ```
# import torch as pt
# # zeroth episode
# obs = pt.load("observations_0.pt")
# # second trajectory
# tr_1 = obs[1]
# # trajectory content
# print(tr_1.keys())
# ```
#
# Once the training is complete, evaluate the best policy according to the mean rewards:
# - create a copy of the *base* simulation (inside the training folder)
# - copy the best policy trace into the copy of *base* and rename it to *policy.pt*
# - in the *U* file, set *train* to *false*
# - in the *controlDict*  
#   - set the *endTime* to 8  
#   - set *purgeWrite* to 0  
#   - set *writeInterval* to 0.01  
#   - set *timeStart* to 0 for both *forces* and *probes* function object
# - execute the simulation with *Allrun.pre* and *Allrun*
# - once the simulation is complete, inspect the results in ParaView
#
# **Congratulations! This completes the final two exercise session.**

# %% [markdown]
#
