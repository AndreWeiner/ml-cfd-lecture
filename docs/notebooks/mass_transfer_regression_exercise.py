# %% [markdown]
# ---
# title: "Surrogate models for continuous predictions"
# subtitle: "Exercise 5 — theory and bubble-transport project"
# jupyter: python3
# execute:
#   eval: false
# ---
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
#
#
# ## Learning objectives
#
# The [lecture](mass_transfer_regression.py) explains the physics and coupling. The [complementary code notebook](mass_transfer_regression_code.py) provides the complete reference workflow, including data extraction, constrained regression, AOTInductor export, and OpenFOAM compilation. We first reproduce that workflow, then investigate a different bubble and test where the approach remains useful.
#
# After this exercise, we should be able to explain which information passes between the two simulations, export a constrained surrogate without changing its preprocessing, assess interpolation on withheld interface snapshots, and identify limitations in the predicted shape and transport.
#
# ## Files and preparation
#
# [Download exercise source](mass_transfer_regression_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="mass_transfer_regression_exercise.py"}
#
# The complementary notebook links the reference dataset and the files for running the case. There is no submission for this exercise.
#
# ## Part 1 — Theory
#
# ### Task 1 — Scales and resolution
#
# For a bubble with $d_b=1\,\mathrm{mm}$, $U_b=0.25\,\mathrm{m\,s^{-1}}$, $\nu=10^{-6}\,\mathrm{m^2\,s^{-1}}$, and $D=2\times10^{-9}\,\mathrm{m^2\,s^{-1}}$:
#
# 1. Compute $Re$, $Sc$, and $Pe$ and explain which transport processes they compare.
# 2. Using $\delta_h/d_b=(2Re)^{-1/2}$ and $\overline\delta_c/\delta_h=\sqrt{\pi/(2Sc)}$, estimate both thicknesses. What interface-normal cell width gives ten cells across the mean concentration layer?
# 3. Why does this estimate not determine the total cell count or prove that the mass-transfer rate is converged? What changes when $D$ is reduced by a factor of ten?
#
# ### Task 2 — What do the surrogates replace?
#
# 1. List the learned functions and the boundary data they provide in the bubble-fixed liquid domain. Which fields are still obtained from conservation equations?
# 2. Explain why learning the radius and normal velocity independently can make the boundary conditions inconsistent. What does the discrete mesh flux contribute in the implementation?
# 3. With the normal pointing from gas into liquid, discuss normal-velocity continuity, tangential no-slip, and the pressure jump. Which statements are conservation laws, and which require additional assumptions?
# 4. Explain the separation of dissolution and boundary-layer time scales. Give two situations in which ignoring size loss or azimuthal asymmetry would become questionable.
#
# ### Task 3 — Structure, validation, and exchange
#
# 1. Show why a cosine angular feature produces an even radius extension, while a sine output factor produces an odd tangential-velocity extension. Where must the tangential velocity vanish on $[0,2\pi]$?
# 2. Show that $\widehat r=r_0+S_r s(t)^2f_r(q(t),\cos(\vartheta))$ gives the initial sphere and zero initial deformation velocity. Use $s(t)=t/(1+t)$ and $q(t)=\log(1+t)/\log(1+T)$, and assume that $f_r$ is continuously differentiable.
# 3. Which constraints guarantee positive radius and constant volume? If neither is guaranteed, propose a check or an improved model form.
# 4. Why are entire snapshots withheld rather than randomly splitting neighboring PLIC samples? Does this split establish generalization to a new Eötvös number?
# 5. Write down the input order, shape, memory layout, dtype, units, and output of each AOTInductor model. Why do correct tensor shapes alone not protect against a time/angle swap or dimensional-time mismatch?
#
# ## Part 2 — Practical project
#
# ### Task 4 — Reproduce the reference workflow
#
# 1. Download the compact reference dataset linked in the code notebook, open the [code notebook](mass_transfer_regression_code.py), and run its cells in order. The supplied code can be used directly: start by adjusting only `DATA` and `OUTPUT`. There is no need to implement a new parser or training loop.
# 2. Record the number of snapshots, discarded nodata records, and training/validation/test samples. Check the sampling interval against the Basilisk run duration and inspect the coordinate/velocity transformations.
# 3. Save the three physical test errors, prediction plots, initial/pole/symmetry checks, and radius/volume diagnostics. Identify the worst time/angle region. Avoid selecting a new architecture by repeatedly looking at the final test errors; reserve another final holdout if the test set becomes part of development.
# 4. Export AOTInductor, inspect `contract.json`, and run the export agreement checks. Explain why these errors should be much smaller than the errors against CFD targets.
# 5. Compile in the current course OpenFOAM container and perform a short liquid-only startup run in a fresh copied case. Confirm that all three model files and all three boundary libraries load. Then run the intended time interval and inspect mesh quality, velocity, and scalar layers. A successful startup is not the final validation.
#
# **Expected evidence:** a compact table of software versions, model settings, held-out errors, export discrepancies, volume drift, and the outcome of the CFD run. We keep a separate folder for every configuration.
#
# ### Task 5 — Change the bubble properties
#
# 1. Choose a nearby Eötvös number or Galileo number and generate a second Basilisk trajectory using the same container and extraction procedure. Start with a low-cost pilot. Keep axisymmetric, straight-rising, star-shaped bubbles for the main comparison; record the selected properties and numerical resolution.
# 2. Preserve the reference data and exports. Change `DATA`, `OUTPUT`, and `SNAPSHOT_DT` for the new run and repeat the complete procedure. A different bubble requires retraining all three functions; these networks are not parameterized by Eötvös or Galilei number.
# 3. Compare rise speed, deformation, tangential surface velocity, fit errors, and volume drift. Which differences are explained by the physics, and which could result from sampling, fit capacity, or simulation resolution?
# 4. Update the liquid viscosity consistently with $\nu=1/Ga$ in the dimensionless liquid-only case. If the desired Schmidt numbers are fixed, also update $D_i=\nu/Sc_i$. Eötvös number influences the recorded motion and shape, rather than appearing as an independently solved surface-tension balance at the void boundary.
# 5. Compare time-dependent Sherwood numbers and concentration layers at the same dimensionless times. Verify at least one meaningful mesh refinement before interpreting a transfer-rate difference as physical.
#
# A more deformed star-shaped bubble need not be convex. A radius function fails when a ray intersects the interface more than once. Breakup, strong non-axisymmetric motion, large dissolution-induced volume change, or invalid fitted radii lie outside this example's model class. A failed pilot with an explanation can be a useful result; forcing such data through the pipeline is not.
#
# ### Task 6 — Find a limitation and improve one component
#
# Choose **one** controlled variation:
#
# - reduce temporal coverage and compare interpolation with extrapolation beyond the training interval;
# - change network width or the number of retained surface samples, keeping the split fixed;
# - compare a constrained ansatz with a plain MLP at the initial time and the poles;
# - reduce the species diffusion coefficient while refining the interface-normal grid;
# - compare a random point split with a snapshot split to demonstrate how correlated samples affect apparent error.
#
# State a prediction before running the comparison. Change one factor, retain the same data and seeds where appropriate, and report both an aggregate error and a local diagnostic. For extrapolation, inspect the predicted shape before attempting mesh motion. Explain whether the limiting error comes from the surrogate, the two-phase data, the liquid-only approximation, or numerical transport. Exporting successfully is not a useful accuracy criterion on its own.
