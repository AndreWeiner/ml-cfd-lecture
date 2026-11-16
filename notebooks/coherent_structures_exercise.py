# %% [markdown]
# ---
# title: "Coherent structures in transonic buffet"
# subtitle: "Exercise 6 — reading and PAMSPOD of JAXA pressure measurements"
# jupyter: python3
# execute:
#   eval: false
# ---
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
#
# We use projected adaptive multitaper spectral proper orthogonal decomposition (PAMSPOD) to look for coherent pressure patterns on a swept wing. The [lecture](coherent_structures.qmd) introduced spatial POD and the projected DFT. Here we use the same idea of a compact spatial basis to estimate several ranked spatial modes at each frequency.
#
# ## Part 1 — Reading
#
# ### Task 1 — From shock motion to buffet cells
#
# Read the abstract and the first two paragraphs of [Section I of Lusher & Sansica (2026)](https://arxiv.org/html/2601.11137v1#S1). Focus on the nearly spanwise-uniform shock oscillation, the spanwise modulation of separation, and how sweep affects the motion of the cells in their simulations. The [abstract of Sugioka et al. (2021)](https://link.springer.com/article/10.1007/s00348-020-03118-y) gives the corresponding experimental context for the JAXA CRM. The two studies use different wing configurations.
#
# 1. How would a nearly two-dimensional chordwise shock oscillation and a travelling buffet cell differ in a top view of the wing? Sketch the pressure pattern at two instants for each case.
# 2. According to Lusher & Sansica, which mode is more sensitive to sweep? What changes about the frequency and spanwise motion of the cells? Where did Sugioka et al. observe the disturbances first, and in which direction did they propagate?
#
# ### Task 2 — Why use an adaptive number of tapers?
#
# Read the abstract and opening paragraph of [Yeung & Schmidt (2024)](https://flowphysics.ucsd.edu/assets/papers/YeungSchmidt_2024_TCFD.pdf). Then look at **Fig. 8(a,b)** and its caption in Section 3.2. The two panels show the SPOD eigenvalue spectra and the corresponding counts of sine tapers. The example is a cavity flow.
#
# 1. Where does the algorithm use relatively few tapers in Fig. 8? Explain why averaging with more tapers can smooth a broadband spectrum and also broaden a narrow spectral peak.
# 2. What quantity would you compare across two parameter settings to decide if a selected *spatial mode* is robust?
#
# ## Part 2 — Practical analysis
#
# ### Files and preparation
#
# [Download exercise source](coherent_structures_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="coherent_structures_exercise.py"}
#
# With Jupytext installed (`pip install jupytext`), open this percent-format script as a notebook or convert it from the course root with `jupytext --to ipynb notebooks/coherent_structures_exercise.py`. For standalone use, keep [the data helper](jaxa_upsp_data.py){data-noresolveinput="true" download="jaxa_upsp_data.py"} and [the figure style](figure_style.py){data-noresolveinput="true" download="figure_style.py"} next to the notebook. The [course Python environment](../requirements.txt) includes the dependencies. The animation plays in the notebook and uses its browser player.
#
# We use the upper surface of the right wing at the corrected angle of attack $\alpha=4.71^\circ$ and $Ma_\infty=0.848$. The first call downloads the 567 MB archive into a local cache. We read the 200 pressure-fluctuation snapshots at their original spacing of $0.2\,\mathrm{ms}$. The archive includes separate statistics and spectra. [JAXA DPW-8 Test Case 3](https://cfdws.chofu.jaxa.jp/apc/dpw/upsp.html)
#
# The starter code masks incomplete measurements and gives each valid node one third of the area of each adjacent valid triangle. This approximates an area integral on the wing. Each column of the input matrix contains one snapshot. FlowTorch subtracts the temporal mean of this short record when it computes the POD basis.
#
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from flowtorch.analysis import PAMSPOD, SVD
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from matplotlib.tri import Triangulation

from figure_style import WIDTH_MM, figure_style, resolve_renderer
from jaxa_upsp_data import fetch_jaxa_upsp, load_upper_wing_snapshots

xyz, triangles, times, pressure = load_upper_wing_snapshots(fetch_jaxa_upsp())
dt = float(np.diff(times).mean())
assert np.allclose(np.diff(times), dt)

complete = np.isfinite(pressure).all(axis=1)
valid_triangles = triangles[complete[triangles].all(axis=1)]
a, b, c = np.moveaxis(xyz[valid_triangles], 1, 0)
areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
node_areas = np.bincount(
    valid_triangles.ravel(), weights=np.repeat(areas / 3, 3), minlength=len(xyz)
)
valid = node_areas > 0
data = pt.from_numpy(pressure[valid])
weights = pt.as_tensor(node_areas[valid] / node_areas.sum(), dtype=data.dtype)
mesh = Triangulation(*xyz[:, :2].T, valid_triangles)
print(f"{data.shape[0]:,} analysed nodes, {data.shape[1]} snapshots, dt = {dt:.4f} s")

# %% [markdown]
# ### Task 3 — Compare spectra and taper counts
#
# In [flowTorch](https://flowtorch.readthedocs.io/en/latest/flowtorch.analysis.html#flowtorch.analysis.spod.PAMSPOD), PAMSPOD applies AMSPOD to the POD time coefficients and projects selected modes back onto the wing. The implementation uses sine tapers over the record. Both settings below reuse the same area-weighted POD, so we can compare the spectral settings directly.
#
# 1. Run the baseline and the fixed-taper case. Plot the leading two eigenvalues and the number of tapers against $St=f c_{\mathrm{MAC}}/U_\infty$. Compare the spectra and taper counts around $St=0.2$–$0.5$, where the CRM study reports buffet cells.
# 2. Change **one** setting in the adaptive case, for example `max_tapers` from 8 to 4 or `tolerance` from $10^{-3}$ to $10^{-5}$. Rerun the comparison. Which features are stable? Which become smoother, weaker or wider? Use the same axes for the comparisons.
#
# We estimate $U_\infty$ from the corrected Mach number and JAXA's total temperature of $328.25\,\mathrm{K}$ using the perfect-gas relation. The record duration is about $40\,\mathrm{ms}$, giving a Fourier-bin spacing of about $25\,\mathrm{Hz}$ before taper smoothing. We interpret bands and spatial patterns with this resolution in mind.
#
# %%
POD_RANK = 60
MAX_TAPERS = 8
TOLERANCE = 1.0e-3
FIXED_TAPERS = 3

chord = 0.15131
mach, total_temperature = 0.848, 328.25
gamma, gas_constant = 1.4, 287.05
temperature = total_temperature / (1 + (gamma - 1) * mach**2 / 2)
u_inf = mach * np.sqrt(gamma * gas_constant * temperature)
convective_time = chord / u_inf

pod = SVD(data, rank=POD_RANK, weight=weights, subtract_mean=True)
adaptive = PAMSPOD.from_svd(
    pod, dt=dt, adaptive=True, max_tapers=MAX_TAPERS, tolerance=TOLERANCE, keep_n_modes=2
)
fixed = PAMSPOD.from_svd(
    pod, dt=dt, adaptive=False, max_tapers=FIXED_TAPERS, keep_n_modes=2
)
st = adaptive.frequency.numpy() * convective_time

# %%
THEME = "light"  # Set to "dark" for a dark notebook.
RENDERER = "auto"  # Check LaTeX and warn before using the local built-in fallback.
styles = ["default"]
if THEME == "dark":
    styles.extend(["dark_background", {"figure.facecolor": "#111111", "axes.facecolor": "#111111"}])
styles.append(figure_style(use_tex=resolve_renderer(RENDERER)))
styles.append({"axes.prop_cycle": mpl.rcParamsDefault["axes.prop_cycle"]})

with plt.style.context(styles):
    fig, (ax_spectrum, ax_tapers) = plt.subplots(
        2, sharex=True, figsize=(WIDTH_MM["double"] / 25.4, 5),
        height_ratios=[2, 1], layout="constrained"
    )
    for result, label, color in ((adaptive, "Adaptive", "C0"), (fixed, "Fixed", "C1")):
        for mode, linestyle in enumerate(("-", "--")):
            ax_spectrum.semilogy(
                st, result.eigvals[:, mode].numpy(), linestyle,
                color=color, label=f"{label}, mode {mode + 1}"
            )
        ax_tapers.step(st, result.log["n_tapers"].numpy(), where="mid", color=color)
    for ax in (ax_spectrum, ax_tapers):
        ax.axvspan(0.2, 0.5, color="C7", alpha=0.15)
    ax_spectrum.set_ylabel("SPOD eigenvalue")
    ax_spectrum.legend(ncols=2)
    ax_tapers.set(xlabel=r"$St=f c_{\mathrm{MAC}}/U_\infty$", ylabel="Tapers", xlim=(0, 1.2))
    plt.show()

# %% [markdown]
# ### Task 4 — Animate and compare spatial modes
#
# 1. Animate the leading mode at two frequencies. The starter code chooses the largest leading eigenvalue in $0.2\le St\le0.5$ and the largest one in $0.55\le St\le1.0$. We can replace these choices with two interesting bins from our spectra. For each mode, describe where the pressure pattern is strongest and whether its phase appears to travel toward the tip, toward the root, or has little systematic spanwise change. The tip is toward increasing $y/c_{\mathrm{MAC}}$.
# 2. At one of those **same** frequency bins, animate the fixed-taper or modified adaptive result. Compare the spatial pattern and apparent direction of motion. An arbitrary phase offset between modes can shift the starting frame, so compare the motion over a full period.
#
# Each animation shows the real oscillation of one complex mode over one period, at a playback speed chosen for inspection. For each animation, the symmetric color limits are fixed at the 99th percentile of the absolute plotted amplitudes across the cycle. The ends of the colorbar mark saturation. Compare the animations for spatial structure and motion, using the eigenvalue spectra to compare magnitudes between frequencies.
#
# %%
def strongest_bin(result, lower, upper):
    candidates = np.flatnonzero((st >= lower) & (st <= upper))
    return int(candidates[result.eigvals[candidates, 0].argmax().item()])


def animate_mode(result, frequency_index, label):
    cycle = result.mode_reconstruction(frequency_index, eig_idx=0, N=25).numpy()[:, :-1]
    amplitudes = np.full((len(xyz), cycle.shape[1]), np.nan)
    amplitudes[valid] = cycle
    face_values = amplitudes[valid_triangles].mean(axis=1)
    scale = np.percentile(np.abs(face_values), 99)
    face_values /= scale if scale > 0 else np.abs(face_values).max() or 1

    with plt.style.context(styles):
        fig, ax = plt.subplots(figsize=(4.4, 4.4), layout="constrained")
        field = ax.tripcolor(mesh, facecolors=face_values[:, 0], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(
            aspect="equal", xlabel=r"$x/c_{\mathrm{MAC}}$", ylabel=r"$y/c_{\mathrm{MAC}}$",
            title=f"{label}, mode 1 at $St={st[frequency_index]:.3f}$"
        )
        fig.colorbar(field, ax=ax, label="Relative modal amplitude", extend="both")

        def update(frame):
            field.set_array(face_values[:, frame])

        animation = FuncAnimation(fig, update, frames=cycle.shape[1], interval=85)
        player = HTML(animation.to_jshtml())
        plt.close(fig)
    return player


buffet_bin = strongest_bin(adaptive, 0.2, 0.5)
comparison_bin = strongest_bin(adaptive, 0.55, 1.0)
display(animate_mode(adaptive, buffet_bin, "Adaptive"))
display(animate_mode(adaptive, comparison_bin, "Adaptive"))
display(animate_mode(fixed, buffet_bin, "Fixed"))

# %% [markdown]
# Summarise which conclusions about the JAXA pressure field are supported by both parameter settings. Use one spectrum comparison and one observation from the animations.
