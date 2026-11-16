"""Generate the paired figures and movies for the coherent-structures lecture.

From the repository root:
    python notebooks/coherent_structures_figures.py --output notebooks/media
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from scipy.io import loadmat
from scipy.sparse.linalg import svds

from cylinder_piv_data import fetch_cylinder_piv

from figure_style import WIDTH_MM, add_renderer_argument, figure_style, resolve_renderer


FPS = 20
CLIP_FRAMES = 160
SPECTRUM_MODES = 20
RECONSTRUCTION_MODES = 5
CYLINDER_CENTER = (-45.0, -26.6735714674)  # mm, approximate patch center
PROBE_SEARCH_CENTER_Y = -23.5  # mm, reference for the original peak search
CYLINDER_DIAMETER = 9.5  # mm, Tu et al. (2014)
INFLOW_SPEED = 0.0435  # m/s, Tu et al. (2014)
CONVECTIVE_TIME = CYLINDER_DIAMETER * 1e-3 / INFLOW_SPEED  # s
MASK_COLOR = "#8a8a8a"


def canvas(height):
    return plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, height / 25.4),
                        layout="constrained")


def schematic(height, xlim, ylim):
    fig, ax = canvas(height)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal")
    ax.set_axis_off()
    return fig, ax


def arrow(ax, start, end, color, style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": style, "color": color, "lw": 1.5,
                            "shrinkA": 0, "shrinkB": 0})


def buffet(neutral):
    """Adapt the geometry, incidence and visual style of CFD Lecture 2."""
    fig, ax = schematic(94, (-0.40, 1.66), (-0.43, 0.62))
    x = np.linspace(0, 1, 400)
    y = 0.6 * (0.2969*np.sqrt(x) - 0.126*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    rotation = Affine2D().rotate_deg_around(0.25, 0, -5)
    transform = rotation + ax.transData
    xx = np.linspace(-0.30, 1.56, 220)
    for yy in (0.16, 0.24, 0.33, -0.15, -0.24, -0.33):
        offset = (0.075 if yy > 0 else -0.035) * np.exp(-((xx - 0.38)/0.4)**2)
        line = yy + offset - 0.018 * (np.tanh(5*(xx - 0.65)) + 1)
        ax.plot(xx, line, color=neutral, alpha=0.17, lw=0.8)
    states = ((0.55, 1.0, 0.070, "--"), (0.68, 0.90, 0.035, "-"))
    for shock, end, height, style in states:
        pocket = np.linspace(0.045, shock, 100)
        surface = np.interp(pocket, x, y)
        cap = surface + 0.21 * np.sin(np.pi*(pocket - 0.045)/0.87)
        if style == "-":
            ax.fill_between(pocket, surface, cap, transform=transform, color="C1", alpha=0.25)
        ax.plot([shock, shock], [surface[-1], cap[-1]], style,
                transform=transform, color="C3", lw=1.8)
        sep = np.linspace(shock, end, 120)
        surface = np.interp(sep, x, y)
        cap = surface + height * np.sin(np.pi * (sep - shock) / (end - shock))
        ax.plot(sep, cap, style, transform=transform, color="C2")
        if style == "-":
            ax.fill_between(sep, surface, cap, transform=transform, color="C2", alpha=0.20)
    ax.fill_between(x, -y-0.025*np.sqrt(x), -y, transform=transform, color="C2", alpha=0.45)
    ax.add_patch(Polygon(np.c_[np.r_[x,x[::-1]], np.r_[y,-y[::-1]]],
                         transform=transform, facecolor=ax.get_facecolor(),
                         edgecolor=neutral, lw=1.5, zorder=5))
    trailing_edge = rotation.transform((1.0, 0.0))
    wake_x = np.linspace(trailing_edge[0], 1.61, 200)
    distance = wake_x - trailing_edge[0]
    wake_y = -0.05 - 0.021 * np.cos(np.pi * (wake_x - 1.08) / 0.145)
    wake_y += (trailing_edge[1] - wake_y[0]) * np.exp(-distance / 0.035)
    ax.plot(wake_x, wake_y, color="C2", lw=1.2, alpha=0.85)
    for k in range(4):
        cx = 1.08 + 0.145 * k
        sign = -1 if k % 2 == 0 else 1
        cy = -0.05 - sign * 0.042
        theta = sign * np.linspace(0.25 * np.pi, 1.95 * np.pi, 100)
        radius = np.linspace(0.045, 0.029, len(theta))
        vx = cx + radius * np.cos(theta)
        vy = cy + radius * np.sin(theta)
        ax.plot(vx, vy, color="C2", lw=1.2)
        arrow(ax, (vx[-8], vy[-8]), (vx[-1], vy[-1]), "C2", "-|>")
    arrow(ax, (1.03, 0.10), (1.56, 0.10), "C2")
    arrow(ax, (0.54, 0.34), (0.68, 0.34), "C3", "<->")
    arrow(ax, (-0.32, 0.08), (-0.09, 0.08), neutral)
    ax.text(-0.21, 0.115, r"$U_\infty$", ha="center")
    leader = {"arrowstyle": "-", "color": neutral, "lw": 0.8}
    box = {"facecolor": ax.get_facecolor(), "edgecolor": "none", "pad": 1.5, "alpha": 0.94}
    ax.text(0.60, 0.39, "Shock oscillation", ha="center", bbox=box)
    ax.annotate("Supersonic pocket", xy=rotation.transform((0.26, 0.14)), xytext=(0.17, 0.49),
                ha="center", arrowprops=leader, bbox=box)
    ax.text(1.12, 0.44, "Subsonic outer flow", ha="center", bbox=box)
    ax.annotate("Separation region\ngrows and shrinks", xy=rotation.transform((0.82, np.interp(0.82, x, y) + 0.070 * np.sin(np.pi * (0.82 - 0.55) / (1.0 - 0.55)))),
                xytext=(0.94, 0.23), ha="center",
                arrowprops={**leader, "arrowstyle": "->"}, bbox=box)
    ax.text(1.38, 0.29, "Vortices shed and\nconvect downstream", ha="center", bbox=box)
    return fig


def read_wake():
    data = loadmat(fetch_cylinder_piv(), variable_names=["x", "y", "u", "v"])
    x = data["x"][::-1, 0] * 1000
    y = data["y"][0, :] * 1000
    fields = tuple(data[name][::-1] for name in ("u", "v"))
    assert np.allclose(data["x"][::-1] * 1000, x[:, None])
    assert np.allclose(data["y"] * 1000, y[None, :])
    valid = ~np.logical_and.reduce([np.all(field == 0, axis=-1) for field in fields])
    return x, y, fields, valid


def wake_probes(x, y, streamwise, valid):
    streamwise_std = streamwise.std(axis=-1, ddof=1)
    # Compare the two lobes in the same near-wake band, 2–3 diameters aft.
    downstream = (x - CYLINDER_CENTER[0]) / CYLINDER_DIAMETER
    lateral = (y - PROBE_SEARCH_CENTER_Y) / CYLINDER_DIAMETER
    band = (downstream >= 2) & (downstream <= 3)
    probes = []
    for side in (-1, 1):
        lobe = valid & band[:, None] & (side * lateral > 0) & (abs(lateral) < 1.5)
        peak = np.argmax(np.where(lobe, streamwise_std, -np.inf))
        probes.append(np.unravel_index(peak, valid.shape))
    # Use the upper peak as the common streamwise station.
    probes[0] = (probes[1][0], probes[0][1])
    assert all(valid[i, j] for i, j in probes)
    probe = np.stack([streamwise[i, j, :] for i, j in probes])
    probe -= probe.mean(axis=1, keepdims=True)
    return probes, probe


def analyze_wake(x, y, fields, valid):
    probes, probe = wake_probes(x, y, fields[0], valid)
    count = probe.shape[1]
    samples = int(valid.sum())
    snapshot = np.empty((2 * samples, count), dtype=np.float32)
    means, stds = [], []
    for field, block in zip(fields, np.split(snapshot, 2)):
        mean = field.mean(axis=-1)
        means.append(mean)
        stds.append(field.std(axis=-1, ddof=1))
        block[:] = field[valid] - mean[valid, None]

    # Iterative truncated SVD uses all samples without forming either Gram matrix.
    vectors, values, right = svds(snapshot, k=SPECTRUM_MODES, which="LM", tol=1e-6,
                                  v0=np.random.default_rng(41).normal(size=count).astype(np.float32))
    vectors, values, right = vectors[:, ::-1], values[::-1], right[::-1]
    signs = np.sign(vectors[np.argmax(abs(vectors), axis=0), np.arange(SPECTRUM_MODES)])
    vectors = vectors * signs
    scores = values[:, None] * right * signs[:, None]
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(SPECTRUM_MODES), atol=5e-5)
    np.testing.assert_allclose(vectors.T @ snapshot, scores, atol=3e-4)
    total = np.einsum("ij,ij->", snapshot, snapshot, dtype=np.float64)
    fractions = values.astype(float)**2 / total
    modes = np.zeros((vectors.shape[1], 2, *valid.shape), dtype=vectors.dtype)
    for i, block in enumerate(np.split(vectors, 2)):
        modes[:, i, valid] = block.T

    covariance = np.cov(probe)
    estimate = np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axis = axes[:, -1] * np.sign(axes[0, -1])
    print("Probes (mm):", [(float(x[i]), float(y[j])) for i, j in probes])
    print("Probe covariance:\n", covariance, "\nRotated:\n", estimate.T @ covariance @ estimate)
    print("Probe leading axis:", axis, "angle (deg):", np.degrees(np.arccos(axis @ estimate[:, 0])))
    print("Probe eigenvalues:", eigenvalues[::-1], "correlation:", np.corrcoef(probe)[0, 1],
          "leading fraction:", eigenvalues[-1] / eigenvalues.sum())
    print("Full field:", snapshot.shape, "variance fractions:", fractions)
    print("Cumulative fractions:", np.cumsum(fractions)[:RECONSTRUCTION_MODES])
    freq = np.fft.rfftfreq(count, 1 / FPS)
    print("Leading score peak frequencies:", freq[np.argmax(abs(np.fft.rfft(scores)), axis=1)])
    return probes, probe, means, stds, modes, scores, fractions


def field_axes(ax, x, y, *, show_cylinder=True):
    ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect="equal",
           xlabel=r"$x/D$", ylabel=r"$y/D$")
    if show_cylinder:
        ax.add_patch(Circle(np.asarray(CYLINDER_CENTER) / CYLINDER_DIAMETER, 0.5,
                            facecolor=plt.rcParams["axes.facecolor"],
                            edgecolor=plt.rcParams["text.color"], lw=1.3, zorder=5))


def field_image(ax, x, y, data, valid, cmap, limits, *, rasterized=True, show_cylinder=True):
    field_axes(ax, x, y, show_cylinder=show_cylinder)
    cmap = mpl.colormaps[cmap].with_extremes(bad=MASK_COLOR)
    return ax.pcolormesh(x, y, np.ma.array(data.T, mask=~valid.T), cmap=cmap,
                         vmin=limits[0], vmax=limits[1], shading="nearest",
                         rasterized=rasterized)


def probe_map(x, y, std, valid, probes):
    fig, ax = canvas(92)
    image = field_image(ax, x, y, std, valid, "magma", (0, std[valid].max()))
    fig.colorbar(image, ax=ax, label=r"$s_u/U_\infty$")
    for k, (i, j) in enumerate(probes):
        ax.scatter(x[i], y[j], marker=("o", "s")[k], color=f"C{k}",
                   edgecolor="white", linewidth=0.9, s=45, zorder=6)
        ax.annotate(f"Probe {k+1}", (x[i], y[j]), xytext=(8, (-11, 6)[k]),
                    textcoords="offset points", color="white", fontsize=10)
    return fig


def field_panels():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM["double"] / 25.4, 5.0),
                             layout="constrained", sharex=True, sharey=True)
    return fig, axes


def clean_axes(axes):
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")


def statistics(x, y, means, stds, valid):
    fig, axes = field_panels()
    sigma_max = max(field[valid].max() for field in stds)
    for i, symbol in enumerate(("u", "v")):
        mean = means[i]
        bound = abs(mean[valid]).max()
        limits = (min(0, mean[valid].min()), mean[valid].max()) if i == 0 else (-bound, bound)
        for row, field, cmap, bounds, label in (
            (0, mean, "viridis" if i == 0 else "RdBu_r", limits, fr"Mean ${symbol}/U_\infty$"),
            (1, stds[i], "magma", (0, sigma_max), fr"Std. dev. $s_{symbol}/U_\infty$"),
        ):
            ax = axes[row, i]
            image = field_image(ax, x, y, field, valid, cmap, bounds, show_cylinder=False)
            ax.set_title(label)
            fig.colorbar(image, ax=ax, shrink=0.8, pad=0.025)
    clean_axes(axes)
    return fig


def component_modes(x, y, modes, valid, fractions):
    selected = (0, 2, 3)
    fig, axes = plt.subplots(len(selected), 2, figsize=(WIDTH_MM["double"] / 25.4, 6.7),
                             layout="constrained", sharex=True, sharey=True)
    limit = abs(modes[list(selected)]).max()
    for k, row in zip(selected, axes):
        for i, ax in enumerate(row):
            image = field_image(ax, x, y, modes[k, i], valid, "RdBu_r", (-limit, limit))
            symbol = ("$u$", "$v$")[i]
            ax.set_title(fr"Mode {k+1} ({100*fractions[k]:.1f}\%), {symbol} block")
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.colorbar(image, ax=axes, label="Mode entry", shrink=0.78)
    return fig


def wake_streamlines(x, y, means, modes, valid):
    fig, axes = field_panels()
    neutral = plt.rcParams["text.color"]
    selected = modes[[0, 2, 3]]
    mode_limit = np.hypot(selected[:, 0, valid], selected[:, 1, valid]).max()
    for ax, field, label in zip(axes.flat, (means, *selected),
                                ("Mean flow", "Mode 1", "Mode 3", "Mode 4")):
        u, v = (np.ma.array(a.T, mask=~valid.T) for a in field)
        speed = np.ma.hypot(u, v)
        field_axes(ax, x, y)
        ax.contourf(x, y, (~valid).T, levels=[0.5, 1.5], colors=[MASK_COLOR])
        if label == "Mean flow":
            center = np.asarray(CYLINDER_CENTER) / CYLINDER_DIAMETER
            near_y = center[1] + np.linspace(-1.25, 1.25, 11)
            outer_y = np.r_[np.linspace(y[0] + 0.2, near_y[0] - 0.35, 5),
                            np.linspace(near_y[-1] + 0.35, y[-1] - 0.2, 5)]
            seed_y = np.r_[near_y, outer_y]
            seeds = np.column_stack((np.full(seed_y.size, center[0] + 1.25), seed_y))
            stream = ax.streamplot(x, y, u, v, start_points=seeds, color=speed,
                                   cmap="viridis", norm=mpl.colors.Normalize(0, speed.max()),
                                   linewidth=1.0, density=1.6, broken_streamlines=False,
                                   arrowsize=0.65)
            fig.colorbar(stream.lines, ax=ax, label=r"$|\overline{\mathbf u}|/U_\infty$",
                         shrink=0.72, pad=0.025)
        else:
            ax.streamplot(x, y, u, v, color=neutral, density=0.75,
                          linewidth=(0.15 + 1.1 * speed / mode_limit).filled(0),
                          arrowsize=0.65)
        ax.set_title(label)
    clean_axes(axes)
    return fig


def mode_scores(t, scores, fractions):
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True,
                             figsize=(WIDTH_MM["double"] / 25.4, 4.0), layout="constrained")
    for ax, group in zip(axes, ((0, 1), (2,), (3, 4))):
        for index, k in enumerate(group):
            ax.plot(t, scores[k, :len(t)], ("-", "--")[index], color=f"C{k}",
                    label=fr"Mode {k+1} ({100*fractions[k]:.1f}\%)")
        ax.set_xlim(0, len(t) / (FPS * CONVECTIVE_TIME))
        ax.legend(ncols=len(group), loc="upper center", fontsize=9)
        ax.margins(y=0.46)
        ax.grid(alpha=0.18)
    fig.supylabel(r"$y_{k,n}/U_\infty$")
    axes[-1].set_xlabel(r"$tU_\infty/D$")
    return fig


def wake_movie(output, x, y, fields, valid, theme):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained", dpi=160)
    clips = [field[..., :CLIP_FRAMES] for field in fields]
    speed, transverse = [clip[valid] for clip in clips]
    umin, umax = np.percentile(speed, (0.5, 99.5))
    umin = np.floor(umin * 100) / 100
    umax = np.ceil(umax * 100) / 100
    vlim = np.ceil(np.percentile(abs(transverse), 99.5) * 100) / 100
    print(f"Normalized movie color limits: u=[{umin:.3f}, {umax:.3f}], v=±{vlim:.3f}")
    images = []
    for ax, clip, cmap, limits, name in zip(
        axes, clips, ("viridis", "RdBu_r"), ((umin, umax), (-vlim, vlim)),
        (r"Streamwise $u/U_\infty$", r"Transverse $v/U_\infty$"),
    ):
        image = field_image(ax, x, y, clip[..., 0], valid, cmap, limits, rasterized=False)
        ax.set_title(name)
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.025, extend="both")
        images.append(image)
    axes[1].set_ylabel("")
    timestamp = fig.suptitle("")
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "4"])
    with writer.saving(fig, str(output / f"coherent-piv-{theme}.mp4"), fig.dpi):
        for n in range(CLIP_FRAMES):
            for image, clip in zip(images, clips):
                image.set_array(np.ma.array(clip[..., n].T, mask=~valid.T))
            timestamp.set_text(fr"$t={n / FPS:.2f}$ s")
            if n == 0:
                fig.savefig(output / f"coherent-piv-{theme}.png", dpi=fig.dpi)
            writer.grab_frame()
    plt.close(fig)


def pod_spectrum(fractions):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM["double"] / 25.4, 2.6),
                             layout="constrained")
    k = np.arange(1, len(fractions) + 1)
    axes[0].semilogy(k, 100 * fractions, "o-", color="C0", ms=3)
    axes[1].plot(k, 100 * np.cumsum(fractions), "s-", color="C1", ms=3)
    axes[0].set_ylabel(r"Variance per mode (\%)")
    axes[1].set_ylabel(r"Cumulative variance (\%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set(xlabel="Mode number", xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=0.18)
    return fig


def coefficient_spectra(coefficients):
    indices = (0, 2, 3, 5)
    signals = coefficients[list(indices)]
    n = signals.shape[1]
    st = np.fft.rfftfreq(n, d=1 / FPS) * CONVECTIVE_TIME
    energy = abs(np.fft.rfft(signals, axis=1) / n)**2
    energy[:, 1:(-1 if n % 2 == 0 else None)] *= 2
    np.testing.assert_allclose(energy[:, 1:].sum(axis=1), signals.var(axis=1), rtol=2e-6)

    fig, ax = canvas(72)
    for k, bins, style in zip(indices, energy, ("-", "--", ":", "-.")):
        ax.semilogy(st[1:], bins[1:], style, color=f"C{k}", label=f"Mode {k+1}", lw=1.1)
        peak = 1 + bins[1:].argmax()
        ax.plot(st[peak], bins[peak], "o", color=f"C{k}", ms=4)
    ax.set(xlabel=r"$St=fD/U_\infty$", ylabel="Variance per bin",
           xlim=(0, 0.8))
    ax.legend(ncols=4, loc="lower left")
    ax.grid(alpha=0.18)
    return fig


def wake_fourier(modes, coefficients, valid):
    """Transform the retained POD coefficients and lift selected bins only."""
    import torch
    from flowtorch.analysis.dft import DFT

    # The POD coefficients already have zero temporal mean.
    transform = DFT(torch.from_numpy(coefficients), dt=1 / FPS,
                    subtract_mean=False, window="boxcar")
    frequency = transform.frequency.numpy()
    energy = transform.bin_energy.numpy()
    np.testing.assert_allclose(energy.sum(), np.square(coefficients).sum() / coefficients.shape[1],
                               rtol=2e-6)
    band = np.flatnonzero((frequency >= 0.5) & (frequency <= 1.2))
    fundamental = frequency[band[np.argmax(energy[band])]]
    indices = []
    for harmonic in (1, 2, 3):
        target = harmonic * fundamental
        candidates = np.flatnonzero(abs(frequency - target) < 0.1 * fundamental)
        indices.append(candidates[np.argmax(energy[candidates])])
    basis = modes[:, :, valid].reshape(len(modes), -1).T
    lifted = np.stack([basis @ transform.get_mode(int(k)).numpy() for k in indices])
    np.testing.assert_allclose(np.linalg.norm(lifted, axis=1), 1, atol=5e-5)
    return transform, np.asarray(indices), lifted


def fourier_spectrum(transform, selected):
    from flowtorch.visualization import plot_spod_spectrum

    fig, ax = canvas(67)
    # Omit the residual DC bin of the already centered data.
    plot_spod_spectrum(transform.frequency[1:], transform.eigvals[1:],
                       n_show=1, show_sum=False, ax=ax,
                       reference_timescale=CONVECTIVE_TIME, normalize_energy=True,
                       show_legend=False)
    st = transform.frequency.numpy() * CONVECTIVE_TIME
    fractions = transform.bin_energy.numpy()
    fractions = fractions / fractions[1:].sum()
    visible = fractions[(st > 0) & (st <= 0.8)]
    ax.set(xlabel=r"$St=fD/U_\infty$", ylabel="Variance fraction per bin",
           yscale="log", xlim=(0, 0.8), ylim=(visible.min() / 2, visible.max() * 2))
    ax.scatter(st[selected], fractions[selected], color="C1", marker="o", s=25, zorder=3)
    ax.grid(alpha=0.18)
    return fig


def align_fourier_phases(lifted):
    """Rotate each mode so its largest-magnitude entry is positive real."""
    anchor = lifted[np.arange(len(lifted)), abs(lifted).argmax(axis=1)]
    return lifted * np.exp(-1j * np.angle(anchor))[:, None]


def fourier_movie(output, x, y, lifted, valid, theme):
    leading = align_fourier_phases(lifted)[0]
    field = np.zeros((2, *valid.shape), dtype=leading.dtype)
    field[:, valid] = leading.reshape(2, -1)
    limit = np.percentile(abs(leading), 98)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained",
                             dpi=160, sharex=True, sharey=True)
    images = []
    for ax, component, label in zip(axes, field, ("Streamwise", "Transverse")):
        images.append(field_image(ax, x, y, component.real, valid, "RdBu_r",
                                  (-limit, limit), rasterized=False))
        ax.set_title(label)
    axes[1].set_ylabel("")
    fig.colorbar(images[0], ax=axes, label="Mode entry", extend="both", shrink=0.8)
    header = fig.suptitle("")
    stem = output / f"coherent-piv-fourier-leading-{theme}"
    frames = 48
    writer = FFMpegWriter(fps=24, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "4"])
    with writer.saving(fig, str(stem.with_suffix(".mp4")), fig.dpi):
        for n in range(frames):
            phase = n / frames
            for image, component in zip(images, field):
                values = (component * np.exp(2j * np.pi * phase)).real
                image.set_array(np.ma.array(values.T, mask=~valid.T))
            header.set_text(fr"Leading Fourier mode, phase $t/T={phase:.2f}$")
            if n == 0:
                fig.savefig(stem.with_suffix(".png"), dpi=fig.dpi)
            writer.grab_frame()
    plt.close(fig)


def fourier_modes(x, y, transform, selected, lifted, valid):
    fig, axes = plt.subplots(len(selected), 2, figsize=(WIDTH_MM["double"] / 25.4, 6.7),
                             layout="constrained", sharex=True, sharey=True)
    phased = align_fourier_phases(lifted).real
    components = np.zeros((len(selected), 2, *valid.shape), dtype=phased.dtype)
    components[:, :, valid] = phased.reshape(len(selected), 2, -1)
    limit = np.percentile(abs(phased), 98)
    for k, (index, row) in enumerate(zip(selected, axes)):
        st = float(transform.frequency[index]) * CONVECTIVE_TIME
        for i, (ax, symbol) in enumerate(zip(row, ("u", "v"))):
            image = field_image(ax, x, y, components[k, i], valid, "RdBu_r", (-limit, limit))
            ax.set_title(fr"$St={st:.3f}$, {symbol} block")
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.colorbar(image, ax=axes, label="Mode entry", shrink=0.78, extend="both")
    return fig


def reconstruction_movies(output, x, y, means, modes, scores, valid, theme):
    for i, symbol in enumerate(("u", "v")):
        mean = means[i][..., None]
        contributions = (modes[:RECONSTRUCTION_MODES, i, ..., None]
                         * scores[:RECONSTRUCTION_MODES, None, None, :CLIP_FRAMES])
        gain = 3
        levels = np.stack((np.broadcast_to(mean, contributions.shape[1:]),
                           mean + contributions[:2].sum(axis=0),
                           mean + gain * contributions[2],
                           mean + gain * contributions[3:5].sum(axis=0)))
        factor = rf"$\times {gain}$"
        titles = ("Mean", "Mean + modes 1 and 2",
                  f"Mean + {factor} mode 3", f"Mean + {factor} (modes 4 and 5)")
        values = levels[:, valid, :]
        bound = abs(values).max()
        limits = (min(0, values.min()), values.max()) if i == 0 else (-bound, bound)
        cmap = ("viridis", "RdBu_r")[i]
        fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), dpi=160,
                                 layout="constrained", sharex=True, sharey=True)
        images = []
        for k, ax in enumerate(axes.flat):
            images.append(field_image(ax, x, y, levels[k, ..., 0], valid,
                                      cmap, limits, rasterized=False))
            ax.set_title(titles[k])
        for ax in axes[:-1].flat:
            ax.set_xlabel("")
        for ax in axes[:, 1]:
            ax.set_ylabel("")
        fig.colorbar(images[0], ax=axes, label=fr"${symbol}/U_\infty$", shrink=0.8)
        timestamp = fig.suptitle("")
        stem = output / f"coherent-piv-reconstruction-{symbol}-{theme}"
        writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=[
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "4"])
        with writer.saving(fig, str(stem.with_suffix(".mp4")), fig.dpi):
            for n in range(CLIP_FRAMES):
                for image, frame in zip(images, levels[..., n]):
                    image.set_array(np.ma.array(frame.T, mask=~valid.T))
                timestamp.set_text(fr"$t={n / FPS:.2f}$ s")
                if n == 0:
                    fig.savefig(stem.with_suffix(".png"), dpi=fig.dpi)
                writer.grab_frame()
        plt.close(fig)


def time_series(t, fluctuations):
    fig, ax = canvas(62)
    for i, (color, linestyle) in enumerate([("C0", "-"), ("C1", "--")]):
        ax.plot(t, fluctuations[i], color=color, ls=linestyle,
                label=fr"$u_{i+1}^\prime$")
    ax.set(xlabel=r"$tU_\infty/D$", ylabel=r"$u_i^\prime/U_\infty$", xlim=(0, len(t) / (FPS * CONVECTIVE_TIME)))
    ax.margins(y=0.24)
    ax.legend(loc="upper right", ncols=2)
    ax.grid(alpha=0.18)
    return fig


def probe_density(fluctuations, neutral):
    fig, (scatter_ax, density_ax) = plt.subplots(
        1, 2, figsize=(WIDTH_MM["double"] / 25.4, 3.2), layout="constrained",
        sharex=True, sharey=True)
    limit = np.ceil(np.percentile(abs(fluctuations), 99.8) * 200) / 200
    visible = np.all(abs(fluctuations) <= limit, axis=0)
    fluctuations = fluctuations / INFLOW_SPEED
    limit /= INFLOW_SPEED
    print(f"Normalized density plot zoom: ±{limit:.3f}, "
          f"{np.count_nonzero(~visible)} points outside axes")
    scatter_ax.scatter(*fluctuations[:, visible], color="C0", s=5, alpha=0.22,
                       linewidths=0, rasterized=True)
    scatter_ax.grid(alpha=0.18)
    counts = density_ax.hexbin(*fluctuations[:, visible], gridsize=38, mincnt=1,
                       extent=(-limit, limit, -limit, limit),
                       cmap="viridis", linewidths=0)
    assert counts.get_array().sum() == visible.sum()
    fig.colorbar(counts, ax=density_ax, label="Pairs per hexagon")
    for ax, title in zip((scatter_ax, density_ax), ("Scatter plot", "Hexagonal counts")):
        for sign, style, label in [(-1, "--", r"$\phi_1$ (visual estimate)"),
                                   (1, ":", r"$\phi_2$ (perpendicular)")]:
            ax.plot([-limit, limit], [-sign * limit, sign * limit], style, color=neutral,
                    label=label)
        ax.set(title=title, xlabel=r"$u_1^\prime/U_\infty$",
               xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    scatter_ax.set_ylabel(r"$u_2^\prime/U_\infty$")
    fig.legend(*scatter_ax.get_legend_handles_labels(), loc="outside upper center", ncols=2)
    return fig


def svd_shapes(neutral):
    fig, ax = schematic(110, (0, 9.0), (-0.15, 5.3))
    n, m = 5, 10  # Illustrative matrix dimensions.
    cell = 0.17
    centers = (1.60, 3.85, 5.95, 7.95)
    for full, y, title in ((False, 3.79, "Economy SVD"), (True, 1.28, "Full SVD")):
        ax.text(0.08, y + 1.02, title, fontsize=12, weight="bold")
        blocks = (
            (m, n, r"$\mathbf X$", r"$M\times N$", "C0"),
            (m, m if full else n, r"$\mathbf U_{\mathrm f}$" if full else r"$\mathbf U$",
             r"$M\times M$" if full else r"$M\times N$", "C1"),
            (m if full else n, n, r"$\Sigma_{\mathrm f}$" if full else r"$\Sigma$",
             r"$M\times N$" if full else r"$N\times N$", "C2"),
            (n, n, r"$\mathbf V^\mathsf{T}$", r"$N\times N$", "C3"),
        )
        for j, (cx, (rows, cols, label, shape, color)) in enumerate(zip(centers, blocks)):
            width, height = cols * cell, rows * cell
            left, bottom = cx - width / 2, y - height / 2
            ax.add_patch(Rectangle((left, bottom), width, height, facecolor=color,
                                   alpha=0.16, edgecolor="none"))
            ax.add_patch(Rectangle((left, bottom), width, height, fill=False,
                                   edgecolor=neutral, lw=0.8))
            ax.text(cx, y + 1.02, label, ha="center", fontsize=13)
            ax.text(cx, y - 1.10, shape, ha="center", fontsize=11)
            if j == 2:
                for k in range(n):
                    xy = (left + k * cell, bottom + (rows - 1 - k) * cell)
                    ax.add_patch(Rectangle(xy, cell, cell, color=color, linewidth=0))
                for k in range(1, cols):
                    xx = left + k * cell
                    ax.plot([xx, xx], [bottom, bottom + height], color=neutral, alpha=0.35, lw=0.5)
                for k in range(1, rows):
                    yy = bottom + k * cell
                    ax.plot([left, left + width], [yy, yy], color=neutral, alpha=0.35, lw=0.5)
                if full:
                    ax.text(cx, bottom + n * cell / 2, r"$\mathbf 0$", ha="center", va="center")
            if full and j == 1:
                ax.plot([cx, cx], [bottom, bottom + height],
                        "--", color=neutral, lw=0.8)
        for xx, symbol in ((2.57, "$=$"), (5.00, r"$\times$"), (6.95, r"$\times$")):
            ax.text(xx, y, symbol, ha="center", va="center", fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_renderer_argument(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    use_tex = resolve_renderer(args.renderer)
    x, y, fields, valid = read_wake()
    probes, probe, means, stds, modes, scores, fractions = analyze_wake(x, y, fields, valid)
    # Retain dimensional calculations for the worked covariance example.
    x, y = x / CYLINDER_DIAMETER, y / CYLINDER_DIAMETER
    for field in (*fields, *means, *stds):
        field /= INFLOW_SPEED
    scores /= INFLOW_SPEED
    transform, selected, lifted = wake_fourier(modes, scores, valid)
    t = np.arange(CLIP_FRAMES) / (FPS * CONVECTIVE_TIME)
    palette = mpl.rcParamsDefault["axes.prop_cycle"]
    for theme, background, neutral in [("light", "#ffffff", "#222222"),
                                       ("dark", "#111111", "#ededed")]:
        with plt.style.context("dark_background" if theme == "dark" else "default"):
            with mpl.rc_context({
                **figure_style(use_tex=use_tex), "axes.prop_cycle": palette,
                "figure.facecolor": background, "axes.facecolor": background,
                "text.color": neutral, "axes.labelcolor": neutral,
                "axes.edgecolor": neutral, "xtick.color": neutral,
                "ytick.color": neutral, "svg.hashsalt": "coherent-structures",
            }):
                figures = {
                    "buffet": buffet(neutral),
                    "cylinder": probe_map(x, y, stds[0], valid, probes),
                    "signals": time_series(t, probe[:, :len(t)] / INFLOW_SPEED),
                    "scatter": probe_density(probe, neutral),
                    "piv-statistics": statistics(x, y, means, stds, valid),
                    "piv-mode-components": component_modes(x, y, modes, valid, fractions),
                    "piv-mode-vectors": wake_streamlines(x, y, means, modes, valid),
                    "piv-scores": mode_scores(t, scores, fractions),
                    "piv-spectrum": pod_spectrum(fractions),
                    "piv-coefficient-spectra": coefficient_spectra(scores),
                    "piv-fourier-spectrum": fourier_spectrum(transform, selected),
                    "piv-fourier-modes": fourier_modes(x, y, transform, selected, lifted, valid),
                    "svd": svd_shapes(neutral),
                }
                for name, fig in figures.items():
                    fig.savefig(args.output / f"coherent-{name}-{theme}.svg",
                                metadata={"Date": None}, dpi=300)
                    plt.close(fig)
                fourier_movie(args.output, x, y, lifted, valid, theme)
                wake_movie(args.output, x, y, fields, valid, theme)
                reconstruction_movies(args.output, x, y, means, modes, scores, valid, theme)


if __name__ == "__main__":
    main()
