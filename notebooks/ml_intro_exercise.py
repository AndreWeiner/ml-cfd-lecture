# %% [markdown]
# ---
# title: Introduction to machine learning
# subtitle: Exercise 2 — theory and practical project
# jupyter: python3
# ---
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# ## Learning objectives
#
# After this exercise, you should be able to:
#
# - formulate a regression task in terms of data, features, targets, and performance measures;
# - design a small CFD parameter study and separate training, validation, and test cases;
# - run and inspect OpenFOAM and Basilisk simulations and assess whether their output is suitable for further analysis;
# - train and evaluate a velocity-profile surrogate, with and without physical feature transformations;
# - distinguish numerical simulation error, surrogate error, and variation between training runs.
#
# Read [Topic 2 — Introduction to machine learning](ml_intro.py). The practical project combines CFD data generation with regression. The cylinder and bubble simulations introduce the tools and data checks; the channel-flow simulations provide the regression dataset. This exercise can span several practical sessions.
#
# ## Part 1 — Theory
#
# Answer the questions before starting the practical project. Justify your choices rather than listing definitions. You will revisit your answers using your own simulation and training results.
#
# ### Task 1 — Define the learning problem
#
# We want to predict a fully developed mean streamwise velocity profile in a turbulent channel for different prescribed bulk velocities. The geometry, fluid viscosity, mesh, and turbulence model are initially fixed.
#
# 1. State the experience/data, task, and performance measure in Mitchell's definition of machine learning for this problem.
# 2. Propose inputs and outputs for a model that predicts the velocity at one wall-normal location. Distinguish a feature, a feature value, a target, one point sample, and one complete simulation case.
# 3. Is this regression, classification, or unsupervised learning? Explain whether a fixed, trained multilayer perceptron is parametric or non-parametric and deterministic or probabilistic.
# 4. Propose a simple baseline, an intended use, and measurable acceptance criteria. What evidence would be needed to claim that the surrogate saves time after accounting for data generation and training?
#
# ### Task 2 — Design the simulation campaign
#
# Assume a budget of 16 simulations and a bulk-velocity range of $0.1\leq\bar{U}\leq1.0\,\mathrm{m\,s^{-1}}$.
#
# 1. Sketch a regular grid, independent uniform random sampling, and Latin hypercube sampling in this one-dimensional interval. What does stratification guarantee, and what does it leave random?
# 2. How does the cost of a regular grid change if viscosity and channel height also vary, with 16 levels per parameter? Contrast this with a fixed total budget of 16 cases.
# 3. Explain why sampling uniformly in a dimensional parameter need not produce uniform coverage in a transformed feature space, such as $\log Re$.
# 4. Propose an adaptive sampling criterion based on validation errors. Why should the untouched test set not guide the next sample locations?
# 5. List checks required before treating a final CFD field as a reference label. Can a small training loss compensate for unconverged or poorly resolved reference simulations?
#
# ### Task 3 — Split and normalize without leakage
#
# Each simulation produces 200 closely related wall-normal samples.
#
# 1. Compare randomly splitting all point samples with splitting entire simulations. Which evaluates generalization to an unseen Reynolds number more directly?
# 2. Define interpolation and extrapolation tests for this dataset. Sketch a split with training cases on both sides of an unseen interior test case.
# 3. Write the min–max and standardization formulas. Which subset supplies their statistics? Explain how to transform an unseen value outside the training range and how to handle a constant feature.
# 4. Explain why normalizing all data before splitting leaks information. Why is repeatedly selecting architectures using test errors also leakage?
# 5. State what must be saved alongside trained network weights so that a new input receives exactly the same preprocessing and output transformation.
#
# ### Task 4 — Use physical knowledge
#
# Let $\delta$ be the channel half-height, $\nu$ the kinematic viscosity, $y$ the wall distance, and $u_\tau=\sqrt{|\tau_w|/\rho}$ the friction velocity. Define
#
# $$
# Re_b=\frac{2\delta\bar{U}}{\nu},\qquad
# u^+=\frac{u_x}{u_\tau},\qquad
# y^+=\frac{yu_\tau}{\nu}.
# $$
#
# Here $u^+$ denotes dimensionless velocity; it is distinct from the kinematic viscosity $\nu$.
#
# 1. Compare the mappings $(y,\bar{U})\mapsto u_x$, $(y/\delta,Re_b)\mapsto u_x/\bar{U}$, and $\log(y^+)\mapsto u^+$. What assumptions allow the final mapping to omit an explicit Reynolds-number input?
# 2. Why is $\log(y^+)$ undefined at the wall? Which wall-normal region might fail to collapse onto a universal curve even when near-wall scaling works well?
# 3. Suppose $u_\tau$ is extracted from a completed CFD solution. Is it available when the surrogate is used to avoid that simulation? Propose an alternative and explain the additional error it introduces.
# 4. Would permutation importance or a correlation heatmap prove that a feature is physically causal? Discuss correlated inputs and availability during deployment.
#
# ### Task 5 — Train and evaluate
#
# 1. For scalar regression, write the mean squared error and derive the gradient-descent update for network parameters. Distinguish parameters from architecture and optimizer hyperparameters.
# 2. Explain why an MLP with only affine layers cannot represent an arbitrary nonlinear profile, regardless of its depth.
# 3. Propose a fair comparison of two architectures using validation data and multiple random seeds. Distinguish training variability from predictive uncertainty.
# 4. Choose plots that reveal near-wall errors, errors at different Reynolds numbers, and variation between runs. Explain what an aggregate error alone can conceal.
# 5. Explain why agreement with a RANS dataset establishes an approximation of that dataset, rather than experimental validation of the physical flow.
#
# ## Part 2 — Practical project
#
# ### Preparation and working folders
#
# Complete [Exercise 1 — System setup](system_setup.py), including the OpenFOAM and Basilisk container setup. Use the Ubuntu course environment on native Linux, WSL, or VirtualBox. Run ParaView outside the containers. Keep your scripts and results in `exercises`; keep the supplied cases in `test_cases` unchanged.
#
# Start a Bash terminal in the repository root:
#
# ```bash
# source ml-cfd/bin/activate
# source setup-env
# mkdir -p exercises
# ```
#
# Repeat the environment activation and `source setup-env` in each new terminal. The latter selects the course OpenFOAM.com v2606 image. The `Allrun` scripts use the repository's `RunFunctions` helpers to execute applications and write logs. Existing logs can cause an application to be skipped on a repeated run. Use a fresh working copy for a new experiment; preserve results before using a case's `Allclean` script, which deletes generated output.
#
# The commands below assume the named destination folders do not already exist. If you worked with these cases before, choose new folder names and adjust the subsequent paths. Record image names, settings, runtime, and any failures. Runtime depends on the machine, mesh, and output frequency; measure a pilot run before committing to a parameter study.
#
# ### Task 6 — OpenFOAM: flow past a cylinder
#
# #### Inspect and execute the case
#
# From the repository root:
#
# ```bash
# cp -r test_cases/cylinder2D exercises/cylinder2D
# cd exercises/cylinder2D
# ```
#
# Inspect `Allrun`, `system`, `constant`, and `0.org` before running:
#
# 1. Identify the solver, viscosity, turbulence treatment, mesh generation utility, and required initial fields.
# 2. Read the inlet profile and cylinder diameter. Compute Reynolds numbers using both the maximum and the mean inlet speed. State the convention whenever you report $Re$.
# 3. Determine the number of MPI ranks from `system/decomposeParDict`; compare it with the resources available to you. If you change the decomposition, explain how its settings remain consistent.
# 4. Identify time-step size, end time, write interval, pressure solver, and momentum-convection discretization.
# 5. Inspect the `forceCoeffs` configuration: reference speed, reference area, reference length, and lift/drag directions. Why must these be known when comparing coefficients?
#
# Execute the case and check the mesh:
#
# ```bash
# ./Allrun
# source "$ML_CFD_BASE/RunFunctions"
# runApplication checkMesh
# ```
#
# Inspect `log.blockMesh`, `log.checkMesh`, and the solver log (`log.pimpleFoam`). Check for errors, achieved final time, Courant numbers, residuals, continuity errors, and execution time. Explain what the `processor*` directories contain. A script reaching its last line is not sufficient evidence that every application succeeded.
#
# #### Visualize fields and force histories
#
# ```bash
# paraview post.foam
# ```
#
# Select **Decomposed Case** before applying the OpenFOAM reader. Inspect the mesh, pressure, velocity magnitude, and vortex shedding. If using a reconstructed case instead, use the corresponding reconstructed-case reader setting. The reconstruction and shared-folder route is described in Exercise 1.
#
# Locate the force output:
#
# ```bash
# find postProcessing -type f -name 'coefficient.dat'
# ```
#
# Create your own Python script to load and plot drag and lift coefficients over time. Inspect the header before choosing columns and separators: column positions must follow the actual file, rather than an assumed layout. Label axes and units, identify an initial transient, and state the time interval used for any averages or fluctuation measures. Compare the reference speed in the coefficient definition with your Reynolds-number convention.
#
# **Keep:** a case-settings summary, mesh/log checks, one flow-field image, and force-history plots. Explain which parts could provide inputs, targets, or quality checks for a later ML problem.
#
# ### Task 7 — Basilisk: an axisymmetric rising bubble
#
# #### Compile the solver
#
# Return to the repository root. The solver source is `bubble_axis_symmetric.c`; the supplied Makefile builds the executable `bubble` using the Basilisk compiler `qcc` and MPI.
#
# ```bash
# cd "$ML_CFD_BASE"
# cp -r test_cases/basilisk_solver exercises/basilisk_solver
# apptainer shell "$ML_CFD_BASE/basilisk.sif"
# ```
#
# Inside the container, run:
#
# ```bash
# cd exercises/basilisk_solver
# make > log.make 2>&1
# cat log.make
# ```
#
# Check that compilation succeeded and the `bubble` executable exists. Inspect the includes, initial condition, acceleration, adaptive refinement, and output events in the source. Contrast this compiled, event-based setup with OpenFOAM's case dictionaries.
#
# #### Execute and inspect the simulation
#
# Still inside the container:
#
# ```bash
# mkdir run
# cd run
# mpirun -np 2 ../bubble 14 10 115 134 > log.solver 2>&1
# exit
# ```
#
# The arguments are maximum adaptive refinement level, end time, Bond/Eötvös number, and Galilei number. Use an MPI rank count supported by your machine. For a first execution check, you may use a shorter integer end time in a separate pilot directory; this does not replace the full physical run. The source parses end time as an integer and separately sets an initial refinement level, so lowering the maximum adaptive level alone does not necessarily make initialization inexpensive.
#
# Inspect the numerical rows of `log.solver`. The source writes time, gas-volume measure, three center-of-mass coordinates, three center-of-mass velocity components, and interface area. Identify the rising direction from the source and plot rise speed over time. Assess volume drift and whether the late-time behavior appears settled. Distinguish the axisymmetric calculation from a fully three-dimensional bubble-path simulation.
#
# From an Ubuntu terminal outside the container:
#
# ```bash
# cd "$ML_CFD_BASE/exercises/basilisk_solver/run/vtu"
# paraview
# ```
#
# Open the `bubble_*.pvtu` time series and the `plic_*.pvtu` interface series. Inspect adaptive cell sizes and volume fraction $f$: this solver uses $f=0$ in gas and $f=1$ in liquid. Visualize the interface and velocity. To examine the wake in a translating frame, subtract the measured instantaneous rise velocity from the corresponding velocity component; do not assume a fixed rise speed from an earlier run.
#
# **Keep:** compilation/execution checks, a rise-speed history, a volume-conservation diagnostic, and a labeled flow/interface image. Explain why changing numerical resolution can change the labels generated by a simulation.
#
# ### Task 8 — Boundary-layer regression: generate channel profiles
#
# The case in `boundary_layer_1D` computes a wall-normal profile in a fully developed turbulent plane channel: a no-slip lower wall and symmetry at the half-height, periodic streamwise boundaries, and a mean-velocity forcing term. It is not a spatially developing flat-plate boundary layer.
#
# #### Inspect the base case and vary one input manually
#
# From the repository root:
#
# ```bash
# cd "$ML_CFD_BASE"
# cp -r test_cases/boundary_layer_1D exercises/boundary_layer_1D
# cd exercises/boundary_layer_1D
# bash ./Allrun
# source "$ML_CFD_BASE/RunFunctions"
# runApplication checkMesh
# ```
#
# The base mesh has 200 wall-normal cells with grading toward the wall. Verify this using the mesh dictionary and mesh check. Read the velocity boundary conditions, `system/fvOptions`, viscosity, and turbulence model. Identify the prescribed bulk speed $\bar{U}$ and explain how `meanVelocityForce` drives the flow. Here $\delta=0.5\,\mathrm{m}$, $\nu=10^{-5}\,\mathrm{m^2\,s^{-1}}$, and $Re_b=2\delta\bar{U}/\nu$.
#
# Inspect the final velocity profile in ParaView. Check the solver log and late-time profile changes before accepting it as a reference. `purgeWrite` retains only a limited number of output times; if more snapshots are needed to assess stationarity, change that setting in your working copy before running.
#
# Make a second fresh copy of the supplied base case with a different destination name. Double $\bar{U}$ in `system/fvOptions`, halve `deltaT`, and halve `endTime` to preserve $\tilde{t}=t\bar{U}/(2\delta)$. Run it and compare the profiles. Explain why equal dimensionless run duration does not itself prove convergence.
#
# #### Automate a small parameter study
#
# ```bash
# cd "$ML_CFD_BASE"
# cp test_cases/parameter_variation_1d.py exercises/parameter_variation_1d.py
# ```
#
# Inspect your copy before execution. It samples bulk velocity using Latin hypercube sampling, modifies `Ubar`, `deltaT`, and `endTime`, creates `Ub_*` case folders under `exercises/boundary_layer_1D_variation`, and launches serial simulations using a worker pool.
#
# 1. Start with two pilot cases and one worker, using `--samples 2 --workers 1`. Select 10–30 cases for the full study. The script defaults to 16 cases, up to two workers (limited by available CPUs), and seed 2026; change the seed with `--seed` and record it.
# 2. Choose the worker count based on your actual CPU and memory allocation. Each OpenFOAM case runs serially. Keep the total workload within that allocation.
# 3. Inspect the generated dictionaries. The script scales end time and time step inversely with speed, records sampled speeds in `study.json`, and retains three written time directories for late-time comparisons. Verify the configured speeds and dimensionless duration.
# 4. Use a new output directory for each study. The script refuses to reuse an existing directory, preventing results from different configurations from being mixed. Select another destination with `--output`.
# 5. Check every solver log, final time, and profile. The script checks process failures and whether the solver reached its configured end time; these checks do not establish convergence. Compare the last saved profiles and extend the run if they still change appreciably.
#
# From the repository root, run a pilot:
#
# ```bash
# source ml-cfd/bin/activate
# source setup-env
# python exercises/parameter_variation_1d.py --samples 2 --workers 1 --output exercises/channel_pilot
# ```
#
# After checking the pilot, run the full study in a separate, new directory:
#
# ```bash
# python exercises/parameter_variation_1d.py --samples 16 --workers 2
# ```
#
# #### Assemble and inspect the dataset
#
# Write a script in `exercises` using `pathlib`, PyTorch, Matplotlib, and `flowtorch.data.FOAMDataloader`. The following is a starting pattern for reading one serial case:
#
# ```python
# from pathlib import Path
# from flowtorch.data import FOAMDataloader
#
# case = Path("exercises/boundary_layer_1D")  # run Python from the repository root
# loader = FOAMDataloader(str(case))
# y = loader.vertices[:, 1]
# time = loader.write_times[-1]
# u_x = loader.load_snapshot("U", time)[:, 0]
# order = y.argsort()
# y, u_x = y[order], u_x[order]
# ```
#
# Extend this to every successful `Ub_*` case. Check the solver end time and the selected snapshot time: the solver can finish between scheduled field writes. Record both, and check that the selected snapshot is suitable for your analysis. Check that coordinates are consistent across cases, values are finite, and units and shapes are correct. Sort each profile by wall distance and cases by their configured bulk velocity. Use a matrix with shape `(number of cells, number of cases)` for the profiles, plus a case table containing speed, Reynolds number, final time, and quality checks. Plot profiles with a color or legend encoding $Re_b$.
#
# Because the wall-normal mesh is graded, a plain arithmetic mean over cells does not represent the bulk speed. Use cell-volume weighting when checking the computed mean velocity. Likewise, state whether your regression metric weights cell samples equally or approximates a spatial integral.
#
# ### Task 9 — Train a direct regression model
#
# Implement the workflow in your own script; use the lecture's definitions and the [optional Python refresher](python_intro.py) as needed. The optional reading [Training neural networks](neural_network_training.py) provides a reusable supervised training loop and controlled comparisons of batch sizes, learning rates, schedules, architectures, and losses.
#
# 1. Define the pointwise mapping $(y/\delta,Re_b)\mapsto u_x/\bar{U}$. Write down input and target units and tensor shapes: features `(N, 2)` and targets `(N, 1)` for a scalar-output MLP.
# 2. Split **entire simulation profiles** into training, validation, and test subsets before fitting scalers. Keep the test case IDs fixed. Include unseen interior Reynolds numbers for interpolation; label a separate extrapolation evaluation if you reserve cases outside the training range.
# 3. Fit normalization statistics using training data only. Save the split and scalers. Check zero-variance features and confirm the transformation can be inverted.
#
# **Compare training with and without normalization.** Train the direct model twice: first with the feature and target values from step 1 unchanged, and then with statistical normalization of both features and targets. The dimensionless definitions in step 1 remain the same in both runs; only the statistical scaling changes. Fit scaling statistics using training profiles only. For this comparison, keep the network architecture, initialization seed, case split, optimizer, learning rate, and training budget identical. Repeat the paired comparison for several seeds.
#
# Plot training and validation histories, compare convergence and stability, and evaluate dimensional velocity errors after undoing all scaling. Raw losses in different target scales are not directly comparable; also track the same dimensional error measure for both runs. Explain any differences using the feature ranges, activation functions, and gradient-based optimization. If you tune the learning rate separately afterward, report that as an additional comparison and give both approaches the same tuning budget.
#
# 4. Establish a **piecewise linear baseline** using only training profiles on the common wall-normal grid. At each wall distance, interpolate linearly in bulk velocity between the two bracketing training profiles. Below or above the training interval, extend the segment defined by the lowest or highest two training velocities, respectively. Use dimensional velocity profiles for this calculation. Do not clamp predictions to an endpoint profile: `numpy.interp` does this by default outside its input range, so implement the linear extension explicitly. The baseline reproduces its training profiles exactly; zero training error is therefore not evidence of generalization.
# 5. Train a small PyTorch MLP with nonlinear hidden activations and a scalar output. Use an MSE loss, monitor training and validation losses, and retain the checkpoint selected by validation performance. State the sample weighting: explain how much each cell sample contributes to the loss. Equal weights give every sample the same influence and emphasize the densely sampled near-wall region on this graded mesh. Cell-volume weights instead approximate a spatially averaged error. State which choice you use for training and evaluation.
# 6. Test different hidden-layer activation functions, for example **tanh, ReLU, and SiLU**. First keep layer count, width, normalization, optimizer, learning rate, case split, and training budget fixed so that you can isolate the effect of the activation function. Repeat with several initialization seeds. Compare training and validation histories, convergence, and dimensional prediction errors. Then vary hidden-layer count and width with a modest, recorded budget. Use validation data for selecting the activation and architecture; reserve the test set for evaluating the selected procedure.
# 7. Undo target normalization and dimensional scaling before comparing dimensional predictions. Plot predicted and reference profiles, residuals versus wall distance, and errors versus Reynolds number. Report RMSE and maximum absolute error with units, and variation between runs.
#
# Discuss near-wall and centerline behavior as well as aggregate accuracy. If a good average score conceals a poor profile in part of the domain, revisit the validation analysis rather than tuning against the test set.
#
# ### Task 10 — Compare physical feature transformations
#
# #### Wall scaling and Spalding's relation
#
# Estimate friction velocity for each reference case. One option is
#
# $$
# u_\tau=\sqrt{\nu\left|\frac{\partial u_x}{\partial y}\right|_w},
# $$
#
# using the no-slip wall and resolved near-wall gradient. Explain how you estimate that gradient, assess sensitivity to the near-wall discretization, and distinguish this estimate from a solver-reported wall shear stress. OpenFOAM wall-stress outputs may be expressed per unit density; check dimensions before converting them.
#
# Plot $u^+=u_x/u_\tau$ against $y^+=yu_\tau/\nu$. Compare with Spalding's relation,
#
# $$
# y^+=u^++\frac{1}{E}\left[e^{\kappa u^+}-1-\kappa u^+-\frac{(\kappa u^+)^2}{2}-\frac{(\kappa u^+)^3}{6}\right],
# $$
#
# using $\kappa=0.41$ and $E=9.8$ as reference values. Generate the comparison curve parametrically in $u^+$; a nonlinear inverse is not required for plotting. Treat it as an empirical near-wall reference, not an exact solution throughout the channel.
#
# Compare the reference-case estimates of $u_\tau/\bar{U}$ against the empirical correlation
#
# $$
# \frac{\widehat{u_\tau}}{\bar{U}}=0.169\,Re_b^{-0.115}.
# $$
#
# Assess its discrepancy over your sampled range. This is a candidate empirical approximation for this exercise, not a universal friction law. If you fit a replacement, use training cases for fitting and validation cases for choosing its form.
#
# #### Train and compare a transformed model
#
# 1. Use the deployable estimate $\widehat{u_\tau}$ to form $\widehat{y^+}=y\widehat{u_\tau}/\nu$ and the target $\widehat{u^+}=u_x/\widehat{u_\tau}$. Use the same estimate in training and inference. Reserve CFD-derived $u_\tau$ for diagnostic plots, rather than silently providing unavailable information at prediction time.
# 2. Train a scalar mapping $\log(\widehat{y^+})\mapsto\widehat{u^+}$ using the same case splits. Cell centers have positive wall distance; reject nonpositive inputs before taking a logarithm. Treat the wall value separately using the no-slip condition.
# 3. Fit statistical scalers on training data only. Predict using the full chain: compute $Re_b$ and $\widehat{u_\tau}$, transform and scale the input, evaluate the model, invert target scaling, and multiply by $\widehat{u_\tau}$.
# 4. Compare against the direct model and baseline in **dimensional velocity units** on identical held-out profiles. Do not compare raw losses from differently scaled target spaces as if they measured the same error.
# 5. Check where profile collapse works and where omitting Reynolds number loses information. Separate errors from the friction-velocity approximation and errors from the learned mapping as far as your diagnostics allow.
#
# ### Task 11 — Test extrapolation to unseen Reynolds numbers
#
# The interior test profiles in Task 9 measure interpolation. Now examine predictions outside the training interval, using additional CFD reference cases that remain separate from all model-development data.
#
# 1. Record the minimum and maximum training Reynolds numbers. Select at least one Reynolds number outside this interval before inspecting its reference profile. For example, $\bar{U}=1.2\,\mathrm{m\,s^{-1}}$ gives $Re_b=120{,}000$ for the geometry and viscosity in Task 8, above the full sampling range specified there. Optionally add a lower-Reynolds-number case; discuss whether turbulent-flow assumptions still apply as Reynolds number decreases.
# 2. Create a fresh copy of the channel case and set its bulk velocity, time step, and end time consistently with Task 8. Preserve a comparable dimensionless duration as a starting point, then check late-time profile changes and extend the run if necessary. Check mesh adequacy, near-wall resolution, solver logs, final time, and the volume-weighted bulk speed. Record both the solver end time and the time of the reference field. A new Reynolds number can require different numerical resolution or run duration.
# 3. **Freeze the models before evaluation:** keep network parameters, architecture, activation, statistical scalers, friction-velocity correlation, and baseline training profiles unchanged. Do not refit normalization or use the new CFD solution to supply friction velocity to the wall-scaled model. If any model-development choices are changed after inspecting these results, classify this case as development data and generate another untouched case for testing.
# 4. Predict the new profile with the piecewise linear baseline, direct MLP, and wall-scaled MLP. For the baseline, use the linear extension from the nearest two training profiles. For the MLPs, apply the complete input and output transformations from Tasks 9 and 10 without clipping inputs to the training range.
# 5. Plot the dimensional velocity profiles and residuals versus wall distance. Report RMSE and maximum absolute error with the same sample weighting used for the interior tests. Include the achieved bulk speed of each predicted profile and variation across the frozen training seeds.
# 6. Compare training, validation, interior-test, and extrapolation errors in one table. Explain why the linear baseline's zero training error and a low interpolation error do not establish extrapolation capability. Identify where the direct MLP's inputs and the wall-scaled model's $\log(\widehat{y^+})$ leave their respective training ranges. Discuss the near-wall and outer-channel regions separately, including limitations of the friction-velocity correlation and wall scaling.
#
# Treat the result as evidence for the tested flow conditions and distance from the training interval. One successful extrapolation case does not establish reliable predictions at arbitrary Reynolds numbers.
