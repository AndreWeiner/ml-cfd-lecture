# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ml-cfd
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "End-to-end simulations in OpenFOAM and Basilisk — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](cfd_intro_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="cfd_intro_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# ## Part I: flow past a circular cylinder at low Reynolds number
#
# For a description of the mathematical problem, refer to the article by [Schäfer and Turek (1996)](http://www.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf).
#
# ### Inspecting and running the simulation test-case
#
# Every OpenFOAM simulation is a hierarchical structure of folders and files. The top-level simulation folder always contains the following three sub-folders:
#
# 1. *system* - control dictionaries for solver execution, equation discretization, linear solvers, meshing, and others  
# 2. *constant* - transport and thermophysical properties, the mesh (*polyMesh* folder), and geometry files  
# 3. a time folder; fields are written to time folders, where the folder name reflects the physical time at which the snapshot was taken; typically, there is a *0* (zero) folder to define initial conditions for each field; a *0.org* folder indicates that one or more of the fields will be modified before starting the simulation, e.g, to set more complex, nonuniform initial conditions  
#
# The simulation folder for this exercise is located at *test_cases/cylinder2D*. Before inspecting and running the simulation, we create a copy in the *exercises* folder. If you haven't created the exercises folder yet, run:
# ```
# # assuming you are at the repository's top level
# mkdir exercises
# ```
# For all OpenFOAM simulations in this lecture, we employ a dedicated Apptainer image. The logic of modifying the usual OpenFOAM commands is hidden in the *RunFunctions* script at the repositories top level. The script contains shell functions like `runApplication` and `runParallel`, which simplify the execution of OpenFOAM applications and write regular output as well as error messages to log files. These functions in turn rely on environment variables, which define the Apptainer image name and the location of the lecture repository on *your* system. The script *setup-env* defines all required shell variables:
# ```
# # run at the repository's top level
# source setup-env
# ```
# You can check the newly defined variables by typing `$ML_CFD` and pressing the TAB key twice (quickly). **Note that you have to source the environment variables whenever you open a new terminal.** Alternatively, you can add the line above to your `$HOME/.bashrc` file.
#
# Now let's copy the simulation folder recursively (*-r* option), and navigate into the new folder:
# ```
# cp -r test_cases/cylinder2D/ exercises/
# cd exercises/cylinder2D/
# ```
# You can use the *ls* command to inspect the files and folders forming the simulation case. There are two scripts at the test case's top level. The *Allrun* script executes all necessary pre-processing steps and starts the simulation using the Apptainer container. The *Allclean* script re-sets the folder to its initial state.
#
# Before starting the simulation, try to answer the following questions by browsing through all files and folders inside the *cylinder2D* folder:
# - With how many MPI ranks (how many processor cores) is the simulation executed? Tip: parallelization in most CFD codes is based on domain decomposition.  
# - For which fields do we need initial and boundary conditions? Tip: check the initial time folder.  
# - Write down the function defining the inlet velocity. Tip: for which field do we define this boundary condition?  
# - Which utility is used for the domain discretization? Tip: check the section about mesh generation in the official [user guide](https://www.openfoam.com/documentation/user-guide).  
# - What is the Reynolds number based on the maximum inlet velocity and the cylinder's diameter? Tip: try to answer this question in four steps:  
#   - determine the kinematic viscosity  
#   - determine the cylinder's diameter  
#   - determine the maximum inlet velocity
#   - plug these numbers into the definition of $Re$  
# - Is the numerical solver compressible or incompressible? Tip: search for the solver name in the [code documentation](https://doc.openfoam.com/2312/).  
# - What linear solver is used the solve the pressure Poisson equation? Tip: check the *fvSolution* dictionary.  
# - How is the convective term of the momentum equation discretized? Tip: check the *fvSchemes* dictionary.  
# - Which turbulence model is employed?  
# - How many time folders would you expect to be created? Tip: check the *controlDict*.  
# - Which file contains a dictionary to compute force coefficients?  
#
# Now, let's actually run the simulation:
# ```
# ./Allrun
# ```
# The simulation takes about 20-30 minutes to complete. In the meantime, try to answer the following questions:
#
# - Which new folders are/were created?  
# - What do the processor[0-1] folders contain?  
# - How many cells does the mesh consist of? Let the following steps guide you:  
#   - source the function to use the container: `source $ML_CFD_BASE/RunFunctions`  
#   - run the *checkMesh* utility: `runApplication checkMesh`  
#   - inspect the *log.checkMesh* file  
# - The force coefficients are written to the file *coefficient.dat*; where is the file located? Tip: use `find . -type f -name coefficient.dat`  
# - What do the first four columns of *coefficient.dat* contain (after the header)? Tip: use `head -n 20 path/to/coefficient.dat` and refer to the [documentation](https://doc.openfoam.com/2312/tools/post-processing/function-objects/forces/forceCoeffs/) for the acronyms  
#
# Before we start to inspect and visualize the flow field, it is always a good idea to browse through the solver's log file. Open the *log.pimpleFoam* file, preferably using a command line editor like *vim*, *nano*, or *emacs*, and try answering the following questions:
#
# - How many seconds were needed to complete the simulation? Tip: go to the very end of the log file  
# - How many iterations on average did the *p*-*U* coupling take per time step?  
# - What was the maximum Courant number (roughly)?  
#
# Inspecting the log file(s) should always be the first action in case your simulation did not execute as expected (crashes, unexpected results, extremely slow execution). There is an extremely useful utility to extract information about residuals and the iterative solution from the log files called *foamLog*. It can be used as follows:
# ```
# source $ML_CFD_BASE/RunFunctions
# runApplication foamLog log.pimpleFoam
# ```
# A new *logs* folder will be created. To have a quick look at some residuals and iteration counts, we use a small tool called *Gnuplot*. If Gnuplot is not installed on your system, run:
# ```
# sudo apt install gnuplot
# ```
# Then, navigate into the *logs* folder, and plot some of the available files, e.g.:
# ```
# cd logs
# gnuplot
# # now we are in a gnuplot shell
# # continuity error after the first p-U-coupling iteration over time
# gnuplot> plot 'contCumulative_0'
# # sometimes, it's helpful to use a logarithmic scale on the ordinate
# gnuplot> set logscale y
# # final residual for p and Ux after fist p-U-coupling iteration
# gnuplot> plot 'pFinalRes_0', 'UxFinalRes_0'
# # iterations to solve for p in the first p-U-coupling iteration
# gnuplot> unset logscale y
# gnuplot> plot 'pIters_0'
# # exit Gnuplot -> q + Enter
# ```
#
# ### Visualizing the flow in ParaView
#
# Let's have a look at the flow fields. To open ParaView, run:
# ```
# # run this command at the simulation folder's top level
# paraview post.foam
# ```
# *post.foam* is an empty dummy file to indicate ParaView that we are providing an OpenFOAM simulation. **Before clicking on Apply,** set the *Case Type* to **Decomposed Case**. To inspect the mesh, select *Surface With Edges* as representation. Then, return to the *Surface* representation and animate the velocity's magnitude:
#
# - select *U* in the fields drop down menu
# - go to the last time step and click on the button *Rescale To Data Range* (a text is displayed when hovering with the mouse pointer over buttons)
# - go back to the first time step and click on the play button
# - enjoy the vortex street
#
# You can also save an animation by going to *File->Save Animation*.
#
# The domain is split into two sub-domains, one for each MPI rank (core). To visualize the decomposition, close ParaView, and execute the following steps:
# ```
# touch processor0/post.foam
# touch processor1/post.foam
# # inspect the first processor domain
# paraview processor0/post.foam
# # inspect the second processor domain
# paraview processor1/post.foam
# ```
#
# ### Evaluating lift and drag coefficients
#
# Typical target quantities for the flow past a cylinder are drag and lift forces acting on the cylinder. As the last step of this tutorial, we will plot drag and lift coefficients over time in a Jupyter notebook. Open up a new Jupyter notebook, give the file a sensible name, and add some description in the first cell, e.g.:
#
# ```
# # Flow past a circular cylinder at low Reynolds number
# ## Drag and lift coefficients
#
# The Reynolds number based on the cylinder diameter $d$, the average inlet velocity $\bar{U}$ and the kinematic viscosity $\nu$ is $Re=d\bar{U}/\nu = ...$.
# ```
# The Pandas module provides a powerful function to read comma-separated-value (CSV) files. The coefficients may be visualized as simple line plots with Matplotlib.
# ```
# import matplotlib.pyplot as plt
# from pandas import read_csv
#
# # increase plot resolution
# plt.rcParams["figure.dpi"] = 160
# ```
# In the next cell, use the following gist as starting point and determine the missing parameters by inspecting the *coefficient.dat* file and consulting the [read_csv documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).
# ```
# path = "path/to/coefficient.dat"
# names = ["t", "cd", "cl"]
# data = read_csv(path, skiprows=???, header=0, sep=???, usecols=[0, 1, 4], names=names)
# data.head()
# ```
# Finally, we create a figure with two axes and plot both drag and lift coefficients as lines. Replace the indicated arguments of the plot function.
# ```
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
#
# ax1.plot(data.t, ???)
# ax2.plot(data.t, ???)
# ax2.set_xlabel(r"$t$ in $s$")
# ax2.set_ylabel(r"$c_l$")
# ax1.set_ylabel(r"$c_d$")
# ax1.set_ylim(2.8, 3.3)
# plt.savefig("drag_and_lift.pdf", bbox_inches="tight")
# plt.plot()
# ```

# %% [markdown]
# ## Part II: rise of an axis-symmetric bubble in stagnant liquid
#
# ### The *bubble* solver
#
# Basilisk is a highly specialized library for the simulation of mulTiphase flows with large differences in spatial scales. In contrast to OpenFOAM, simulations in Basilisk are not controlled by input dictionaries. Instead, a customized solver is compiled based on modular library components. The solver present in the *test_cases* folder performs the simulation of an axis-symmetric bubble (symmetric in azimuth direction around the rise direction) rising in a liquid tank. The solver itself is defined in the *bubble.c* source file. Additional header files write output files in the VTK format, which are useful for post-processing in ParaView.
#
# ### Compiling the solver
#
# Compiling and running the simulation requires the *basilisk.sif* Apptainer container. We could use functions similar to *runApplication* and *runParallel* to automate running commands inside the container. However, this time, we walk manually through the process to understand a little bit better the interaction with the container. Compiling the solver is simplified by the *Makefile* located in the source folder. From the repository's top-level, run the following commands:
#
# ```
# # create a copy for the exercise
# cp -r test_cases/basilisk_solver/ exercises/
# # open a new shell in the Singularity container
# apptainer shell basilisk.sif
# # now we are operating inside the container
# Apptainer> cd exercises/basilisk_solver/
# Apptainer> make &> log.make
# ```
#
# Inspect the log file to see if any problems were encountered during the compilation by running `cat log.make`. In the same folder, you should now see a new binary file called *bubble*.
#
# ### Performing a simulation
#
# The solver takes 4 arguments to control the simulation:
#
# 1. the maximum refinement level; the solver employs adaptive mesh refinement; a denser mesh is used in some regions of the mesh to reduce the numerical error; the maximum refinement level limits the mesh refinement such that the smallest cells do not become too small (with every refinement, the time step drops and the cell count increases)  
# 2. the end time; the simulation is stopped after reaching this physical time  
# 3. the [Eötvös](https://en.wikipedia.org/wiki/E%C3%B6tv%C3%B6s_number) or Bond number  
# 4. the [Galilei number](https://en.wikipedia.org/wiki/Galilei_number)  
#
# We will discuss the definition and meaning of the dimensionless numbers in lecture 5. For now, it is sufficient to know that an Eötvös number of $Eo=115$ combined with a Galilei number of $Ga=134$ leads to a bubble in the so-called spherical cap regime. The following simulations completes in about 10-15 minutes. If your machine has more than 2 processor cores available, you can reduce the runtime by passing a higher value to `mpirun`, e.g., `mpirun -np 4` to run the simulation on 4 cores.
# ```
# # create a dedicated folder for this test simulation
# # in the exercises/basilisk_solver/ folder
# Apptainer> mkdir run
# Apptainer> cd run
# Apptainer> mpirun -np 2 ../bubble 14 10 115 134 &> log.solver
# # once the solver is done, leave the container
# exit
# ```
#
# You can follow the solver's progress by opening a new terminal and using `tail -f log.solver`. To stop tracing the output, press *Ctrl+c*. Note that this key combination only stops the *tail* command but not the solver execution.
#
# ### Visualizing the flow in ParaView
#
# VTK files of the flow variables and the gas-liquid interface are stored in the *vtu* folder. To open the snapshots, navigate to *exercises/basilisk_solver/run/vtu/* and open ParaView.
#
# ```
# # Note: you have to exit the Apptainer container before opening ParaView
# # if you still see the 'Apptainer>' prompt, type 'exit' and press enter
# cd exercises/basilisk_solver/run/vtu/
# paraview
# ```
# To open the entire set of snapshots, click on *File->Open->bubble_..pvtu*. There are several interesting options to post-process this mulTiphase simulation. Here are some suggestions:
#
# - visualize the mesh by selecting the *Surface With Edges* representation; advance the simulation time and observe, how the mesh changes  
# - select the volume fraction field *f*; the volume fraction is $f=0$ inside the bubble, $f=1$ outside the bubble, and $0<f<1$ in cells containing the gas-liquid interface  
# - the reconstructed interface is stored as line segments in files named *plic_..pvtu*; they can be opened the same way as the field snapshots; the line segments are a little bit hard to see; to improve their visibility,  
#   - chose the *Wireframe* representation  
#   - set the *Line Width* value in the *Properties* panel to 3  
#   - select a different color under *Edit Color Map*  
# - the *u.x* field holds the absolute velocity field; to depict potential recirculation zones, it is more suitable to switch to a moving reference frame by subtracting the rise velocity from the x-component of the velocity field; this computation can be done in ParaView:  
#   - the (dimensionless) rise velocity is approximately $\tilde{U}_b=0.735$ after 10 time units; refer to the 6th column of the solver log file  
#   - in ParaView, apply the *Calulator* filter; refer to the image below for the computation  
#   - apply the *Glyph* filter to visualize the recirculation zone in the bubble's wake  
#
# <img src="media/calculator.png" style="width:300px">
#
# **Congratulations! This completes the second and third exercise sessions.**

# %% [markdown]
#
