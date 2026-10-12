"""Regenerate the introductory lecture's light/dark Venn diagram."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from figure_utils import save_theme_variants


def draw_requirements():
    # Plain text matches the existing website figures; no mathematical typesetting.
    with plt.rc_context({"text.usetex": False, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for center, color in [((-0.65, 0.35), "C0"), ((0.65, 0.35), "C1"), ((0, -0.65), "C2")]:
            ax.add_patch(Circle(center, 1.12, facecolor=color, alpha=0.25, edgecolor="none"))
            ax.add_patch(Circle(center, 1.12, facecolor="none", edgecolor=color, linewidth=1.8))
        ax.text(-0.98, 0.68, "Domain Knowledge", ha="center", va="center", fontsize=12, rotation=45)
        ax.text(0.95, 0.68, "Mathematics", ha="center", va="center", fontsize=12, rotation=-45)
        ax.text(0, -1.15, "Computer science", ha="center", va="center", fontsize=12)
        ax.text(0, 0.28, "ML in Fluid Mechanics", ha="center", va="center", fontsize=11, fontweight="bold", rotation=90)
        ax.set(xlim=(-1.9, 1.9), ylim=(-1.85, 1.55), aspect="equal")
        ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        return fig



def draw_regression():
    import numpy as np
    from figure_style import figure_style, WIDTH_MM
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 3.6), layout="constrained")
        reynolds = np.array([100, 180, 300, 450, 650, 850, 1000])
        drag = 0.65 + 11 / np.sqrt(reynolds) + np.array([0.03, -0.04, 0.02, 0.035, -0.025, 0.01, -0.015])
        # Synthetic observations and a least-squares illustrative regression.
        design = np.column_stack([np.ones_like(reynolds), 1 / np.sqrt(reynolds)])
        coefficients = np.linalg.lstsq(design, drag, rcond=None)[0]
        query = np.linspace(100, 1000, 300)
        ax.scatter(reynolds, drag, color="C0", marker="o", label="Training data", zorder=3)
        ax.plot(query, coefficients[0] + coefficients[1] / np.sqrt(query), color="C1", linestyle="--", label="Regression model")
        ax.set(xlabel=r"Reynolds number $\mathrm{Re}$ (feature)", ylabel=r"Drag coefficient $c_D$ (target)", xlim=(70, 1030), ylim=(0.85, 1.9))
        ax.legend(loc="upper right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        return fig


def draw_classification():
    import numpy as np
    from figure_style import figure_style, WIDTH_MM
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 3.6), layout="constrained")
        rng = np.random.default_rng(42)
        reynolds = rng.uniform(150, 1000, 44)
        angle = rng.uniform(0.5, 11.5, 44)
        boundary = lambda re: 10.5 - 0.007 * re
        turbulent = angle > boundary(reynolds)
        # Give selected observations labels opposite to the model prediction.
        # Choose points near the boundary from both classes for visible errors.
        distance = np.abs(angle - boundary(reynolds))
        laminar_indices = np.flatnonzero(~turbulent)
        turbulent_indices = np.flatnonzero(turbulent)
        mismatch_indices = np.concatenate([
            laminar_indices[np.argsort(distance[laminar_indices])[:2]],
            turbulent_indices[np.argsort(distance[turbulent_indices])[:2]],
        ])
        turbulent[mismatch_indices] = ~turbulent[mismatch_indices]
        # Keep these mismatches visibly separated from the boundary.
        angle[mismatch_indices] = boundary(reynolds[mismatch_indices]) + np.where(turbulent[mismatch_indices], -0.7, 0.7)
        for mask, color, marker, label in [(~turbulent, "C0", "o", "Laminar"), (turbulent, "C1", "^", "Turbulent")]:
            ax.scatter(reynolds[mask], angle[mask], color=color, marker=marker, label=label, zorder=3)
        query = np.linspace(100, 1050, 300)
        ax.fill_between(query, 0, boundary(query), color="C0", alpha=0.13, linewidth=0, zorder=0)
        ax.fill_between(query, boundary(query), 12, color="C1", alpha=0.13, linewidth=0, zorder=0)
        ax.plot(query, boundary(query), color="C2", linestyle="--", label="Decision boundary")
        ax.set(xlabel=r"Reynolds number $\mathrm{Re}$ (feature)", ylabel=r"Angle of attack $\alpha$ (degrees; feature)", xlim=(100, 1050), ylim=(0, 12))
        ax.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.17))
        ax.spines[["top", "right"]].set_visible(False)
        return fig



def draw_dimensionality_reduction():
    import numpy as np
    from figure_style import figure_style, WIDTH_MM
    rng = np.random.default_rng(12)
    along = np.linspace(-1.6, 1.6, 45)
    across = rng.normal(0, 0.07, along.size)
    theta = np.deg2rad(35)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    probes = np.column_stack([along, across]) @ rotation.T + [5, 4]
    centered = probes - probes.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    if axes[0, 0] < 0:
        axes[0] *= -1
    if np.linalg.det(axes) < 0:
        axes[1] *= -1
    transformed = centered @ axes.T
    with plt.rc_context(figure_style()):
        fig, panels = plt.subplots(1, 2, figsize=(WIDTH_MM["double"] / 25.4, 3.3), layout="constrained")
        ax = panels[0]
        ax.scatter(*probes.T, s=16, color="C0", zorder=3)
        mean = probes.mean(axis=0)
        for direction, length, label, color in [(axes[0], 1.65, r"$U'_1$", "C1"), (axes[1], 0.85, r"$U'_2$", "C2")]:
            end = mean + length * direction
            ax.annotate("", xy=end, xytext=mean, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6})
            ax.annotate(label, xy=end, xytext=(5, 5), textcoords="offset points", color=color)
        ax.set(xlabel=r"Probe velocity $U_1$ (m/s)", ylabel=r"Probe velocity $U_2$ (m/s)", title="Original coordinates", xlim=(3.3, 6.7), ylim=(2.3, 5.7), aspect="equal")
        ax = panels[1]
        for x, y in transformed:
            ax.plot([x, x], [0, y], color="C0", alpha=0.3, lw=0.7)
        ax.scatter(*transformed.T, s=16, color="C0", label="Transformed data", zorder=3)
        ax.scatter(transformed[:, 0], np.zeros(len(along)), s=13, marker="x", color="C1", label="1D representation", zorder=4)
        ax.axhline(0, color="C1", lw=1, alpha=0.5)
        ax.set(xlabel=r"First component $U'_1$ (m/s)", ylabel=r"Second component $U'_2$ (m/s)", title="Centered and rotated coordinates", xlim=(-1.85, 1.85), ylim=(-1.85, 1.85), aspect="equal")
        ax.legend(loc="upper center", fontsize=8, frameon=False)
        for ax in panels:
            ax.spines[["top", "right"]].set_visible(False)
        return fig


def draw_clustering():
    import numpy as np
    from figure_style import figure_style, WIDTH_MM
    rng = np.random.default_rng(19)
    centers = np.array([[3.4, 3.3], [5.8, 5.3]])
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 3.6), layout="constrained")
        for i, (center, count) in enumerate(zip(centers, [14, 30])):
            points = rng.normal(center, [0.30, 0.25], size=(count, 2))
            centroid = points.mean(axis=0)
            radius = np.linalg.norm(points - centroid, axis=1).max() + 0.10
            ax.scatter(*points.T, color="C0", marker="o", s=25, zorder=3)
            ax.add_patch(Circle(centroid, radius, facecolor="none", edgecolor=f"C{i+1}", linewidth=1.6, zorder=2))
            ax.scatter(*centroid, color=f"C{i+1}", marker="x", s=70, linewidths=2, zorder=4)
            ax.annotate(f"Cluster {i+1}", xy=centroid + [0, radius], xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", color=f"C{i+1}")
        ax.set(xlabel=r"Probe velocity $U_1$ (m/s)", ylabel=r"Probe velocity $U_2$ (m/s)", xlim=(2.1, 7.6), ylim=(2.1, 7.2), aspect="equal")
        ax.spines[["top", "right"]].set_visible(False)
        return fig



def draw_reinforcement():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from figure_style import figure_style, WIDTH_MM
    with plt.rc_context(figure_style()):
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, 3.8))
        foreground = plt.rcParams["text.color"]
        for x, color in [(0.1, "C0"), (8.9, "C1")]:
            ax.add_patch(FancyBboxPatch((x, 0.35), 3, 3.35, boxstyle="round,pad=0.05", facecolor=color, alpha=0.22, edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x, 0.35), 3, 3.35, boxstyle="round,pad=0.05", facecolor="none", edgecolor=foreground, linewidth=1.3, linestyle="--"))
            for y in [1.47, 2.59]:
                ax.plot([x,x+3],[y,y],color=foreground,ls="--",lw=1.1)
        ax.text(1.6,3.92,"Agent",ha="center",fontsize=11,fontweight="bold",color="C0")
        ax.text(10.4,3.92,"Environment",ha="center",fontsize=11,fontweight="bold",color="C1")
        entries=[
            (1.6,3.15,"Action",r"$a=\omega$","Set cylinder rotation speed"),
            (1.6,2.03,"Policy",r"$\pi(a\mid s)$","Wake pressures to speed distribution"),
            (1.6,0.91,"Improvement",r"$\pi^{\mathrm{old}}\rightarrow\pi^{\mathrm{new}}$","Update the controller"),
            (10.4,3.15,"State",r"$s$","Pressure measurements in the wake"),
            (10.4,2.03,"Transition function",r"$p(s'\mid s,a)$","Flow responds to cylinder rotation"),
            (10.4,0.91,"Reward function",r"$r=-c_D$","Lower drag coefficient: higher reward"),
        ]
        for x,y,title,formula,example in entries:
            ax.text(x,y+0.25,title,ha="center",va="center",fontsize=9)
            ax.text(x,y,formula,ha="center",va="center",fontsize=10)
            ax.text(x,y-0.30,example,ha="center",va="center",fontsize=7)
        # Two successive interactions, following the supplied reference.
        for y,step,state,action in [(3.13,r"Step $n$",r"$S_n$",r"$A_n$"),(1.08,r"Step $n+1$",r"$S_{n+1},\ R_{n+1}$",r"$A_{n+1}$")]:
            ax.add_patch(Circle((6,y),0.57,fill=False,edgecolor=foreground,linewidth=1))
            ax.text(6,y,step,ha="center",va="center",fontsize=9)
            ax.add_patch(FancyArrowPatch((8.83,y+0.27),(3.18,y+0.27),connectionstyle="arc3,rad=0.20",arrowstyle="->",mutation_scale=12,color=foreground,lw=1.3))
            ax.add_patch(FancyArrowPatch((3.18,y-0.27),(8.83,y-0.27),connectionstyle="arc3,rad=0.20",arrowstyle="->",mutation_scale=12,color=foreground,lw=1.3))
            ax.text(6,y+0.84,state,ha="center",va="center",fontsize=10)
            ax.text(6,y-0.84,action,ha="center",va="center",fontsize=10)
        ax.text(6,2.10,r"$\vdots$",ha="center",va="center",fontsize=12)
        ax.set(xlim=(-0.1,12.1),ylim=(-0.05,4.25))
        ax.axis("off")
        fig.subplots_adjust(left=0.02,right=0.98,bottom=0.03,top=0.97)
        return fig



def draw_workflow(kind):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from figure_style import figure_style, WIDTH_MM
    with plt.rc_context(figure_style()):
        height = 3.4 if kind == "stacked" else (2.6 if kind == "circular" else 1.65)
        fig, ax = plt.subplots(figsize=(WIDTH_MM["double"] / 25.4, height))
        foreground = plt.rcParams["text.color"]
        def box(x,y,label,color):
            ax.add_patch(FancyBboxPatch((x-1,y-.30),2,.60,boxstyle="round,pad=0.05",facecolor=color,alpha=.18,edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x-1,y-.30),2,.60,boxstyle="round,pad=0.05",facecolor="none",edgecolor=color,lw=1.4))
            ax.text(x,y,label,ha="center",va="center",fontsize=10)
        def arrow(start,end,label,position,rad=-.35):
            ax.add_patch(FancyArrowPatch(start,end,connectionstyle=f"arc3,rad={rad}",arrowstyle="->",mutation_scale=12,color=foreground,lw=1.4))
            ax.text(*position,label,ha="center",va="center",fontsize=10)
        if kind in ["extract","use","circular"]:
            labels = ["Data","Models","CFD or\nexperiment"] if kind=="use" else ["CFD or\nexperiment","Data","Models"]
            colors = ["C1","C2","C0"] if kind=="use" else ["C0","C1","C2"]
            for x,label,color in zip([0,3,6],labels,colors):
                box(x,0,label,color)
            operations = ["Learn","Use"] if kind=="use" else ["Extract","Learn"]
            for x,operation in zip([0,3],operations):
                arrow((x+1.07,0),(x+1.93,0),operation,(x+1.5,.55))
            if kind=="circular":
                arrow((6,-.37),(0,-.37),"Use",(3,-1.65),rad=-.38)
                ax.set(ylim=(-1.95,1))
            else:
                ax.set(ylim=(-.65,1))
        else:
            for x,label,color in [(0,"CFD or experiment\nType 1","C0"),(3,"Data","C1"),(6,"Models","C2")]:
                box(x,2,label,color)
            arrow((1.07,2),(1.93,2),"Extract",(1.5,2.55))
            arrow((4.07,2),(4.93,2),"Learn",(4.5,2.55))
            for x,label,color in [(6,"CFD or experiment\nType 2","C0"),(3,"Data","C1"),(0,"Models","C2")]:
                box(x,0,label,color)
            arrow((6,1.63),(6,.37),"Use",(7.0,1),rad=-.35)
            arrow((4.93,0),(4.07,0),"Extract",(4.5,.55),rad=.35)
            arrow((1.93,0),(1.07,0),"Learn",(1.5,.55),rad=.35)
            arrow((0,-.37),(0,-1.05),"Use",(.7,-.7),rad=.2)
            ax.text(0,-1.3,"Further stages",ha="center",fontsize=9)
            ax.set(ylim=(-1.6,3))
        ax.set(xlim=(-1.25,7.65))
        ax.axis("off")
        fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.96)
        return fig


def draw_workflow_extract():
    return draw_workflow("extract")


def draw_workflow_use():
    return draw_workflow("use")


def draw_workflow_stacked():
    return draw_workflow("stacked")


def draw_workflow_circular():
    return draw_workflow("circular")


if __name__ == "__main__":
    from figure_style import DPI
    directory = Path(__file__).resolve().parent / "media"
    for stem, draw in [("ml-fluid-mechanics-venn", draw_requirements), ("supervised-regression", draw_regression), ("supervised-classification", draw_classification), ("unsupervised-reduction", draw_dimensionality_reduction), ("unsupervised-clustering", draw_clustering), ("reinforcement-learning", draw_reinforcement), ("workflow-extract", draw_workflow_extract), ("workflow-use", draw_workflow_use), ("workflow-stacked", draw_workflow_stacked), ("workflow-circular", draw_workflow_circular)]:
        for extension in ("png", "svg"):
            save_theme_variants(draw, directory / (stem + "." + extension), dpi=DPI)
