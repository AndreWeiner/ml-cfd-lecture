# %% [markdown]
# ---
# title: "From PIV snapshots to wake modes"
# subtitle: "Complementary code for Analysis of coherent structures"
# jupyter: python3
# bibliography: ../references.bib
# execute:
#   eval: false
# ---
#
# This notebook accompanies the [lecture](coherent_structures.qmd). We fetch the
# measured cylinder-wake data, find the two probes, and compute the time
# coefficients and spatial modes. The lecture figures and animations are
# reproduced by the plotting script below.
#
# [Download Python source](coherent_structures_code.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="coherent_structures_code.py"}
# [Download dataset helper](cylinder_piv_data.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="cylinder_piv_data.py"}
# [Download figure generator](coherent_structures_figures.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="coherent_structures_figures.py"}
# [Download shared figure style](figure_style.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="figure_style.py"}
#
# ## Load the measurements
#
# The cells use NumPy, SciPy, and Matplotlib and run from the repository's
# **notebooks** directory. We fetch the original MATLAB file [@shang2026piv].
# The first fetch transfers about 1.1 GB. The helper checks the Zenodo checksum
# and stores the file at the Git-ignored path
# datasets/cylinder_piv/cylinder_vel.mat. Later calls verify and reuse it.
#
# The loader returns coordinates in millimeters and both velocity components
# in m/s. Each component has shape (135, 80, 8000), with time along the final
# axis. The static mask excludes points with both velocities zero over the
# entire record, corresponding to the source's unilluminated regions.

# %%
import numpy as np

from coherent_structures_figures import analyze_wake, read_wake, wake_probes

x, y, fields, valid = read_wake()
u, v = fields
print("Velocity array shape:", u.shape, "valid points:", valid.sum())

# %% [markdown]
# ## Two-probe PCA
#
# We find the streamwise standard-deviation peak in each near-wake lobe.
# The upper peak fixes the common streamwise station, and the lower peak
# supplies the cross-stream coordinate of the other probe. The returned rows
# are the centered streamwise signals. The sample covariance is 2 by 2.
# Its eigenvectors define the projection axes, and projection gives one row
# of time coefficients per axis.

# %%
probes, fluctuations = wake_probes(x, y, u, valid)
covariance = fluctuations @ fluctuations.T / (fluctuations.shape[1] - 1)
variances, axes = np.linalg.eigh(covariance)
variances, axes = variances[::-1], axes[:, ::-1]
if axes[0, 0] < 0:
    axes[:, 0] *= -1
time_coefficients = axes.T @ fluctuations
print("Probe positions (mm):", [(x[i], y[j]) for i, j in probes])
print("Covariance (m²/s²):", covariance)
print("Variance fractions:", variances / variances.sum())

# %% [markdown]
# ## Full-field POD
#
# The generator's analysis function computes a mean and a sample standard
# deviation over time at each grid point. For the POD, it removes the mean
# at each point, flattens the valid streamwise and transverse samples in the
# same order, and stacks the streamwise block above the transverse block.
# The resulting snapshot matrix has shape (20870, 8000).
#
# The full calculation uses several gigabytes of memory. The truncated SVD
# computes the 20 leading singular triplets directly from the snapshot
# matrix. The variance fractions use the sum of squares of the entire
# centered matrix as their denominator. The spatial modes are returned on
# the original grid with component order (u, v).

# %%
_, _, means, standard_deviations, modes, coefficients, fractions = analyze_wake(
    x, y, fields, valid
)
print("Mode and coefficient shapes:", modes.shape, coefficients.shape)
print("Cumulative variance through mode 5:", np.cumsum(fractions[:5]))

# %% [markdown]
# ## Reconstruct one snapshot
#
# Each column of the basis contains both velocity blocks in matching order.
# We form a matrix–vector product for the selected modes, then add the mean.
# The animation illustrates weaker patterns by multiplying the contributions
# of mode 3 and of modes 4–5 by three. The function below gives the
# reconstruction at its original amplitude.

# %%
mean_field = np.stack(means)
mean_vector = mean_field[:, valid].reshape(-1)
basis = modes[:, :, valid].reshape(len(modes), -1).T


def reconstruct(snapshot, mode_indices):
    indices = np.asarray(mode_indices, dtype=int)
    samples = mean_vector + basis[:, indices] @ coefficients[indices, snapshot]
    field = np.full(mean_field.shape, np.nan)
    field[:, valid] = samples.reshape(2, -1)
    return field


mean_only = reconstruct(0, [])
mean_plus_modes_1_and_2 = reconstruct(0, [0, 1])
mean_plus_mode_3 = reconstruct(0, [2])
mean_plus_modes_4_and_5 = reconstruct(0, [3, 4])

# %% [markdown]
# ## Projected Fourier transform
#
# The POD time coefficients already have zero temporal mean. We apply
# flowTorch's DFT to their rows, then lift only the three selected frequency
# modes through the POD basis. This is the same projection used in PDFT.
# It reuses the truncated POD above rather than asking PDFT to fit another SVD.
# The transform uses the complete record at 20 Hz and a boxcar window.
# The combined spectrum is one-sided and normalized by total retained
# variance. For the individual coefficients, we plot variance per bin
# on a shared scale so that we can compare amplitudes across modes 1, 3, 4,\n# and 6. The generator also animates one phase cycle of the leading Fourier\n# mode by rotating its complex coefficient and plotting the real part.

# %%
from coherent_structures_figures import INFLOW_SPEED, coefficient_spectra, wake_fourier

normalized_coefficients = coefficients / INFLOW_SPEED
coefficient_figure = coefficient_spectra(normalized_coefficients)
transform, frequency_indices, fourier_vectors = wake_fourier(
    modes, normalized_coefficients, valid
)
print("Selected frequencies (Hz):", transform.frequency[frequency_indices].tolist())

# %% [markdown]
# ## Reproduce the lecture figures and animations
#
# The Fourier example requires PyTorch and flowTorch. The figure generator also requires LaTeX and FFmpeg with H.264 support.
# From the repository root, the following command generates the paired SVG
# figures, MP4 videos, and poster images. The generator uses the same cached
# dataset and computes the POD once for both themes.
#
# ~~~bash
# python notebooks/coherent_structures_figures.py --output notebooks/media
# ~~~
#
# ## References
#
# ::: {#refs}
# :::
