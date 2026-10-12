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
# title: "Analyzing coherent structures in flows displaying transonic buffets — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](coherent_structures_dim_reduction_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="coherent_structures_dim_reduction_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# This exercise consists of two parts, which are meant to be solved in two individual exercise sessions. In the first part, we analyze coherent structures in the surface pressure coefficient of a NACA0012 airfoil in transonic flow conditions by means of a principal component analysis (PCA). In the second part, we analyze the same data using dynamic mode decomposition (DMD). The specific flow conditions in terms of Reynolds number, Mach number, and angle of attack are $Re=10^7$, $Ma=0.75$, and $\alpha = 4^\circ$, respectively. The pressure coefficient is defined as:
#
# $$
#   c_p = \frac{p}{0.5\rho_\infty U_\infty},
# $$
#
# where $p$ is the total pressure, $U_\infty$ is the free-stream speed and $\rho_\infty$ is the free-stream density. The dominant frequency associated with the shock buffet is $f\approx 38Hz$.
#
# To execute this exercise, create an empty script or Jupyter notebook in the *exercise* folder.
#
# ## Principal component analysis
#
# ### Loading and masking the dataset
#
# The simulation was conducted using the OpenFOAM solver *rhoCentralFoam*. The [pressure](https://www.openfoam.com/documentation/guides/latest/doc/guide-fos-field-pressure.html) function object combined with [surface sampling](https://www.openfoam.com/documentation/guides/latest/doc/guide-fos-sampling-surfaces.html) was used to compute and write the pressure coefficient. Such data are typically written to the directory *postProcessing/surfaces/time/field.csv*. If you download the latest lecture dataset, you find the *surfaces* folder under *datasets/naca0012_buffet/surfaces/*. To access the data, we use the flowTorch [CSVDataloader](https://flowmodelingcontrol.github.io/flowtorch-docs/1.0/flowtorch.data.html#module-flowtorch.data.csv_dataloader). More specifically, we create a loader object using the `CSVDataloader.from_foam_surface()` class method, which indicates the dataloader how the data are organized:
# ```
# from flowtorch.data import CSVDataloader
#
# data = "../datasets/naca0012_buffet/surface/"
# loader = CSVDataloader.from_foam_surface(data, "total(p)_coeff_airfoil.raw")
# ```
# The loader provides access to several attributes of the dataset:
#
# - `loader.vertices` returns the x/y/z coordinates of the sample points located on the surface
# - `loader.write_times` returns a list of available snapshot times formatted as strings, e.g., "0.01"
# - `loader.load_snapshot(field, time)` loads a *field* at the specified *time*; the *time* must be given as string
#
# As a first step, load the vertices, normalize them with the chord length $c=0.60105m$, and visualize the $x$ and $y$ components (first and second colum of *loader.vertices*) as scatter plot.
#
# The dataset contains both lower and upper surface (pressure and suction side), but we will analyze only the upper side. Therefore, we create a boolean mask to distinguish between points on the lower and upper surface. flowTorch provides [selection tools](https://flowmodelingcontrol.github.io/flowtorch-docs/1.0/flowtorch.data.html#module-flowtorch.data.selection_tools) to create such masks. Use the *mask_box()* function to select only points on the upper surface. Hint: the airfoil is symmetric with respect to the plane $(x, y=0, z)$.
# ```
# from flowtorch.data import mask_box
# mask = mask_box(normalized_vertices, lower=[?, ?, -1.0], upper=[?, ?, 1.0])
# ```
# To apply the mask to a tensor, we can use the [masked_select()](https://pytorch.org/docs/stable/generated/torch.masked_select.html) function. For example, the commands
# ```
# x = pt.masked_select(normalized_vertices[:, 0], mask)
# z = pt.masked_select(normalized_vertices[:, 2], mask)
# ```
# could be used to sub-select the $x$ and $z$ component of points on the upper surface. The number of selected points is equal to the sum over all elements in *mask*, e.g., `n_points = mask.sum().item()`.
#
# Next, we analyze the available write times and select a window for the PCA analysis. To do so,
#
# - print the first and the last write time; the elements of `loader.write_times` are sorted
# - compute the time step between the first two snapshots by converting the strings into floats
# - create a new list of string-encoded times selecting only the elements of `loader.write_times` with $0.0326s \le t \le 0.1015s$
#
# Based on the number of selected points and snapshots, we can now allocate an empty data matrix, in which individual snapshots will be organized as column vectors:
# ```
# data_matrix = pt.zeros((?, ?))
# ```
# To fill up the data matrix, execute the following steps:
#
# - loop over the selected times
# - load the snapshot at time *t* using loader.load_snapshot("f", t)
# - apply the mask to the snapshot
# - save the selected data as the corresponding column of the data matrix
#
# After the data matrix is complete, compute the temporal mean and variance of each point in the data matrix and plot both fields using [tricontourf](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tricontourf.html). The function call might look similar to the following line:
# ```
# plt.tricontourf(x, z, data_matrix.mean(dim=1))
# ```
#
# Finally, subtract the mean from each column. Hint: use the *dim* keyword of the [mean()](https://pytorch.org/docs/stable/generated/torch.mean.html) function and [unsqueeze()](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html) the result to convert the row into a column vector, which can be more easily subtracted from each column of the data matrix.
#
# ### Computing the singular value decomposition
#
# We compute the PCA be means of a singular value decomposition (SVD) of the mean-subtracted data matrix. The [svd()](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html?highlight=svd#torch.linalg.svd) function returns three tensors, namely the left singular vectors, the singular values, and the right singular vectors (in that order).
# ```
# U, s, VH = pt.linalg.svd(data_matrix, full_matrices=False)
# # the right singular vectors are usually returned transposed
# V = VH.T
# ```
#
# The singular values tell us about the importance of each mode. The elements of *s* are already organized in descending order. Create the following two plots to visualize the importance of the first 50 modes:
#
# - a [bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html) plot of the singular values
# - a bar plot of the normalized singular values, i.e., `s_rel = [si/s.sum().item() for si in s]`
#
# ### Visualizing modes and mode coefficients
#
# The column vectors of *U* correspond to the co-called principal components or proper orthogonal modes. Visualize the first five modes (the first five columns of *U*) as you visualized mean and variance in the first sub-task. What can you say about the first mode?
#
# Convert the selected snapshot times into floating point values and plot the first five columns of *V* over time. To identify the frequency associated with each mode, use the [welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html) function as follows:
# ```
# from scipy.signal import welch
#
# # times is a list containing the selected write times as floats
# dt = times[1] - times[0]
# for i in range(5):
#     f, power = welch(V[:, i], 1/dt, nfft=len(times)*2)
#     plt.plot(f, power, label=f"mode {i+1}")
#     print(f"Maximum amplitude at f={f[np.argmax(power)]:2.4f}Hz")
#     
# plt.legend()
# plt.xlim(0, 300)
# plt.xlabel(r"$f$ in $Hz$")
# plt.ylabel("PSD")
# plt.show()
# ```
#
# Which mode is most representative of the shock buffet?

# %% [markdown]
# ## Dynamic mode decomposition
#
# In the second part of this exercise, we use the same surface pressure data as before but apply dynamic mode decomposition (DMD) to obtain spatial modes that have only a single frequency associated with them. To get started, assemble the data matrix as in the first part of the exercise but do not extract the temporal mean from the snapshots.
#
# ### DMD implementation
#
# Computing the DMD essentially consists of four steps:
#
# 1. compute the SVD of the data matrix omitting the last snapshot: $ \mathbf{X} = \mathbf{U\Sigma V}^H $
# 2. compute the reduced linear operator $\tilde{\mathbf{A}} = \mathbf{U}^H\mathbf{X}^\prime\mathbf{V\Sigma}^{-1}$
# 3. compute the eigen decomposition of the reduced linear operator: $\tilde{\mathbf{A}} = \mathbf{W\Lambda W}^{-1}$
# 4. reconstruct the eigenvectors (modes) of the full linear operator: $\mathbf{\Phi} = \mathbf{X}^\prime\mathbf{V\Sigma}^{-1}\mathbf{W}$
#
# Apply the DMD algorithm to the pressure surface coefficient data matrix. For the implementation of this and the next tasks, the following hints might be helpful:
#
# - [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html); use `full_matrices=False`
# - [torch.linalg.eig](https://pytorch.org/docs/stable/generated/torch.linalg.eig.html#torch.linalg.eig)
# - [torch.diag](https://pytorch.org/docs/stable/generated/torch.diag.html)
# - to convert a tensor of floats into a complex tensor, use the *type* method, e.g., `matrix.type(pt.cfloat)`
# - to access real and imaginary parts of a complex tensor, use the properties `matrix.real` and `matrix.imag`
# - compute the complex conjugate transpose of a matrix with the syntax `matrix.conj().T`
# - compute matrix multiplications using the @ operator
#
# To see if the implementation yields sensibe results, plot the eigenvalues in the complex plane. Most eigenvalues should be located on the unit circle. The following code provides most of the lines to create such a plot.
#
# ```
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# t = pt.linspace(0, 2 * np.pi, 100)
# ax.plot(pt.cos(t), pt.sin(t), ls="--", color="k", lw=2)
# ax.scatter(???, ???, marker="x", lw=1, s=20, zorder=7)
# ax.set_xlim(-1.3, 1.3)
# ax.set_ylim(-1.3, 1.3)
# ax.set_xlabel(r"$Re(\lambda)$")
# ax.set_ylabel(r"$Im(\lambda)$")
# plt.show()
# ```
#
# ### Plotting the spectrum
#
# We can now determine the frequency associated with each mode and assign a mode amplitude/importance. The frequency corresponds to the imaginary part of the eigenvalues $\omega_i$ of the time-continuous linear operator. The eigenvalues of discrete and continuous operator are connected by the relation $\omega_i = \mathrm{log}(\lambda_i)/\Delta t$. Deviding the imaginary part of $\omega_i$ by $2\pi$ yields the frequency of the $i$th mode in $Hz$. Use the initial conditions $ \mathbf{b} = \mathbf{\Phi}^{-1}\mathbf{x}_0 $ as mode amplitudes (ignore the imaginary part).
#
# A common way to visualize a spectrum is a [stem plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.stem.html). The eigenvalues come as complex conjugate pairs because our data matrix contains only real values. The following code snippet shows how to plot only the positive frequencies:
#
# ```
# amplitude = b.abs()
# freq = omega.imag / (2.0*np.pi)
# freq_i_pos = (freq > 0).nonzero().flatten()
#
# fig, ax= plt.subplots()
# ax.stem(freq[freq_i_pos].numpy(), amplitude[freq_i_pos].numpy(), basefmt="none")
# ax.set_xlim(0.0, 1000)
# ax.axvline(38, ls="--", c="k")
# ax.set_xlabel(r"$f$ in $Hz$")
# ax.set_ylabel("amplitude")
# plt.show()
# ```
#
# **Congratulations! This completes the seventh and eighth exercise session.**

# %%
