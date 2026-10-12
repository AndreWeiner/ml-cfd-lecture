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
# title: "End-to-end ML project with OpenFOAM and PyTorch — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](ml_intro_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="ml_intro_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# ## Part I: creating a parameter study
#
# ### Inspecting and running the base simulation
#
# The base simulation for data generation is a 1D channel flow. The simulation folder is located at *test_cases/boundary_layer_1D*. Create a copy of the test case in the exercise folder and run the simulation:
# ```
# # starting from the repository's top level
# source setup-env
# cp -r test_cases/boundary_layer_1D/ exercises/
# cd exercises/boundary_layer_1D
# ./Allrun
# ```
# The simulation completes within a few minutes. In the meantime, try answering the following questions about the setup:
#
# - How many cells does the mesh consist of? Tip: use `source $ML_CFD_BASE/RunFunctions` and `runApplication checkMesh`.
# - What are the boundary conditions for $U$?
# - What is the driving force for this flow? Tip: check the dictionary *system/fvOptions* and consult the [documentation](https://www.openfoam.com/documentation/guides/v2112/doc/index.html).
#
# Once the simulation is finished, open the case in ParaView and complete the following tasks:
#
# - load the final flow state and visualize the velocity profile using the *Glyph* filter
# - visualize individual patches by unselecting the *internalMesh* and selecting only individual patches under *Mesh Regions* in the left properties panel; check again the boundary condition defined for each patch in *0/U*
#
# Close ParaView and reset the simulation by running `./Allclean`. The next goal is to perform the same simulation with increased Reynolds number. The following steps guide you to the modified setup:
#
# - double the Reynolds number by doubling the mean velocity along the channel
# - in *system/controlDict*, adjust the time step such that the Courant number remains roughly constant
# - considering a dimensionless time of $\tilde{t} = t\bar{U}/(2\delta)$, where $\bar{U}$ is the mean velocity along the channel and $2\delta$ is the channel height, modify the end time in *system/controlDict* such that the same amount of dimensionless time units as before is simulated
# - re-run the simulation and inspect the results in ParaView
#
# ## Performing the parameter variation
#
# The script *parameter_variation_1d.py* in *test_cases* automates the manual modifications of the simulation setup conducted in the previous step. To perform the parameter variation, make a copy of the script in the exercise folder:
# ```
# # assuming you are at the repository's top level
# source setup-env
# cp test_cases/parameter_variation_1d.py exercises/
# ```
# Now open the script and inspect the implemented functions. Try answering the following questions:
#
# - How many simulations are performed in total?
# - How many simulations are performed at the same time?
# - Which parameter(s) in which file(s) of the base setup is/are modified?
# - Where are the modified simulations stored?
#
# Your workstation or laptop might be equipped with fewer compute cores than the script assumes. Running multiple simulations at the same time on shared resources slows down the computations unnecessarily. To determine the number of CPU cores available on your machine, run the command `lscpu` and search for the line *Core(s) per socket ...* in the output. You should not run more simulations simultaneously than cores are available (each simulation runs only on a single core). Modify the script accordingly, and divide the parameter space into 10 to 30 sections (this number determines how many simulations are performed). To start the parameter study, start the Python environment, make sure the script is executable, and run the script:
# ```
# # assuming you are at the repository's top level
# source ml-cfd/bin/activate
# cd exercises
# chmod +x parameter_variation_1d.py
# python parameter_variation_1d.py
# ```
# Depending on the available resources and the overall number of simulations, this computation should take about 10-30min. Once all simulations are complete, open a Jupyter notebook and use the following code snippet to load and visualize the velocity profiles:
# ```
# from glob import glob
# from os.path import join
# import torch as pt
# import matplotlib.pyplot as plt
# from flowtorch.data import FOAMDataloader
#
# #
# # adjust the path if necessary
# #
# cases = glob("./boundary_layer_1D_variation/Ub_*")
#
# cases = sorted(cases, key=lambda case: float(case.split("_")[-1]))
# loader = FOAMDataloader(cases[0])
# y = loader.vertices[:, 1]
# u_x = pt.zeros((y.shape[0], len(cases)))
# for i, case in enumerate(cases):
#     loader = FOAMDataloader(case)
#     u_x[:, i] = loader.load_snapshot("U", loader.write_times[-1])[:, 0]
#
# Ubar = pt.tensor([float(case.split("_")[-1]) for case in cases])
# print("Shape of data matrix: ", u_x.shape)
#
# # creating a plot
# delta, nu = 0.5, 1.0e-5
# Re = pt.tensor([Ub.item()*2*delta/nu for Ub in Ubar])
# for i, Ub in enumerate(Ubar):
#     plt.plot(u_x[:, i], y, label=r"$Re={:1.0f}$".format(round(Re[i].item(), 0)))
# plt.xlabel(r"$u_x$")
# plt.ylabel(r"$y$")
# plt.xlim(0.0, 1.1)
# plt.ylim(-0.01, 0.5)
# plt.legend(loc="upper center", ncol=4, bbox_to_anchor=[0.5, 1.3])
# plt.show()
# ```

# %% [markdown]
# ## Part II: creating a model for the streamwise velocity
#
# ## Direct learning approach
#
# Following the lecture notebook:
#
# - compare the velocity profiles against Spalding's function
# - split, reshape, and normalize the data
# - train a baseline model and visualize the $L_2$ norm computed on training, validation and testing data; tip: start with a rather simple model and training routine for the baseline model and only add more complex techniques or architectures once the simple workflow is established
# - plot the predictions against the original data
#
# ## Hyperparameter tuning
#
# In the next step, we try to tune the ML model. Vary the following hyperparameters and try to minimize the prediction error:
#
# - number of neurons per layer
# - number of hidden layers
#
# Take the best model you found, compare the prediction against the original data.
#
# ## Leveraging Spalding's function
#
# The good agreement of our data with Spalding's function might have triggered already the idea that we should be able to use this relation to simplify the modeling. The following steps guide you through the approach:
#
# - for each Reynolds number, extract the friction velocity $u_\tau$ and plot $\tilde{u}_\tau = u_\tau/\bar{U}$ against $Re$; compare the simulation results against the empirical formula $\tilde{u}_\tau = \frac{0.169}{Re^{0.115}}$
# - transform the original data using the $\tilde{u}_\tau$ formula above as follows:  
#   - transform $u_x$ to $u^+$
#   - transform the distance $y$ to $\tilde{y} = \mathrm{log}(y^+)$  
#   - select profiles for training, validation, and testing
#   - reshape and normalize the data  
#
# Since the new model has one feature less, the modified reshape function should look as follows:
#
# ```
# def reshape_data(u_plus, y_plus):
#     data = pt.zeros((u_plus.shape[0]*u_plus.shape[1], 2))
#     for i in range(u_plus.shape[1]):
#         start, end = i*u_plus.shape[0], (i+1)*u_plus.shape[0]
#         data[start:end, 0] = u_plus[:, i]
#         data[start:end, 1] = y_plus[:, i]
#     return data
# ```
#
# PyTorch models expect the input to have at least two dimensions. Therefore, both the feature and label tensors should have the shape $N_s\times 1$ rather than $N_s$ ($N_s$ is the number of samples). This additional reshaping is easily done with `Tensor.unsqueeze(-1)`:
# ```
# train_dataset = pt.utils.data.TensorDataset(
#     feature_scaler.scale(train_tensor[:, 1]).unsqueeze(-1), label_scaler.scale(train_tensor[:, 0]).unsqueeze(-1)
# )
# ```
#
# Now we are ready to train and evaluate the new model:
# - create and train a model $u^+ = f_{\mathbf{\theta}}(\tilde{y})$
# - make predictions for all Reynolds numbers:  
#   - compute $u_\tau$ based on $Re$ and $\bar{U}$
#   - compute $\tilde{y}$ and scale
#   - make a prediction for the scaled $u^+$, re-scale, and multiply by $u_\tau$
# - compare the predictions against the true velocity profiles
#
# The additional transformations and scaling increase the inference complexity. To avoid prediction errors resulting from missing scaling or normalization, it is possible to hide these steps in a top-level model. The following exercises will demonstrate how such composite models are created.
#
# **Congratulations! This completes the fourth and fifth exercise sessions.**

# %% [markdown]
#
