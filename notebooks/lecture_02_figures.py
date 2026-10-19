"""Reproduce Lecture 2's sampling sketches: ml-cfd/bin/python notebooks/lecture_02_figures.py."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from figure_style import figure_style, WIDTH_MM, DPI


def sampling_designs():
    centers = (np.arange(4) + 0.5) / 4
    grid = np.array(np.meshgrid(centers, centers)).reshape(2, -1).T
    return {
        'grid': grid,
        'random': np.random.default_rng(17).random((16, 2)),
        'lhs': qmc.LatinHypercube(d=2, rng=np.random.default_rng(23)).random(16),
        'sobol': qmc.Sobol(d=2, scramble=True, rng=np.random.default_rng(31)).random_base2(4),
    }


def draw_design(points, method, *, use_tex=True):
    with plt.rc_context(figure_style(use_tex=use_tex)):
        fig, ax = plt.subplots(figsize=(WIDTH_MM['single']/25.4, 3.5), layout='constrained')
        if method in ('grid', 'lhs'):
            divisions = 4 if method == 'grid' else 16
            for edge in np.linspace(0, 1, divisions + 1)[1:-1]:
                ax.axvline(edge, color='0.5', alpha=0.35, linewidth=0.6, zorder=0)
                ax.axhline(edge, color='0.5', alpha=0.35, linewidth=0.6, zorder=0)
        ax.scatter(points[:, 0], points[:, 1], s=32, color='C0', zorder=3, clip_on=False)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal',
               xlabel=r'$X_1$ (normalized)', ylabel=r'$X_2$ (normalized)',
               xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        return fig


def draw_lhs_process():
    # Four samples keep the pairing legible at the course's double-column width.
    values = np.array([[0.10, 0.34, 0.69, 0.88], [0.18, 0.42, 0.57, 0.94]])
    permutation = np.array([2, 0, 3, 1])
    with plt.rc_context(figure_style()):
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH_MM['double']/25.4, 3.0), layout='constrained')
        for stage, ax in enumerate(axes[:2]):
            for j, y in enumerate([0.7, 0.25]):
                ax.plot([0, 1], [y, y], color='0.5', linewidth=1)
                for edge in np.linspace(0, 1, 5):
                    ax.plot([edge, edge], [y-0.07, y+0.07], color='0.5', linewidth=1)
                ax.text(-0.08, y, rf'$X_{j+1}$', ha='right', va='center')
                if stage == 1:
                    ax.scatter(values[j], np.full(4,y), color=f'C{j}', s=22, zorder=3)
                    labels = ['a','b','c','d'] if j == 0 else ['u','v','w','z']
                    for x, label in zip(values[j], labels):
                        ax.text(x,y+0.10,rf'${label}$',ha='center',fontsize=10)
            ax.set(xlim=(-0.2,1.05),ylim=(0,1))
            ax.axis('off')
        axes[0].set_title('1. Divide each range\ninto four segments',fontsize=10)
        axes[1].set_title('2. Draw one value\nper segment',fontsize=10)
        ax=axes[2]
        for edge in np.linspace(0,1,5)[1:-1]:
            ax.axvline(edge,color='0.5',alpha=0.35,linewidth=0.6)
            ax.axhline(edge,color='0.5',alpha=0.35,linewidth=0.6)
        points=np.column_stack([values[0], values[1,permutation]])
        ax.scatter(*points.T,s=24,color='C0',zorder=3)
        for (x,y),label in zip(points,['(a,w)','(b,u)','(c,z)','(d,v)']):
            ax.annotate(label,(x,y),xytext=(0,-12 if y>0.85 else 7),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel=r'$X_1$',ylabel=r'$X_2$',xticks=[0,1],yticks=[0,1])
        ax.set_title('3. Shuffle and pair\nthe sampled values',fontsize=10)
        return fig


def draw_sobol_process():
    points=qmc.Sobol(d=2,scramble=False).random_base2(4)
    assert np.allclose(points[2],[0.75,0.25])
    with plt.rc_context(figure_style()):
        fig=plt.figure(figsize=(WIDTH_MM['double']/25.4,3.0),layout='constrained')
        layout=fig.add_gridspec(1,3)
        for i,n in enumerate([4,8,16]):
            ax=fig.add_subplot(layout[0,i])
            previous=0 if n == 4 else n//2
            for edge in [0.25,0.5,0.75]:
                ax.axvline(edge,color='0.5',alpha=0.45,linewidth=0.9 if edge==0.5 else 0.5)
                ax.axhline(edge,color='0.5',alpha=0.45,linewidth=0.9 if edge==0.5 else 0.5)
            if previous:
                ax.scatter(*points[:previous].T,s=22,color='C0',marker='o',label='Earlier',clip_on=False,zorder=3)
            ax.scatter(*points[previous:n].T,s=26,color='C1',marker='^',label='Added' if previous else 'Initial',clip_on=False,zorder=3)
            ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel=r'$X_1$',ylabel=r'$X_2$',xticks=[0,1],yticks=[0,1])
            ax.set_title(rf'3. Extend: $N={n}$',fontsize=10)
            ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.24),ncol=2,fontsize=8,frameon=False,handletextpad=0.3,columnspacing=0.6)
        return fig


def draw_aliasing():
    n = 11
    regular = np.linspace(0, 20.2*np.pi, n)
    lhs = qmc.LatinHypercube(d=1, rng=np.random.default_rng(51)).random(n).ravel()*20.2*np.pi
    x = np.linspace(0, 20.2*np.pi, 2400)
    with plt.rc_context(figure_style()):
        fig, axes = plt.subplots(2, 1, figsize=(WIDTH_MM['double']/25.4, 4.7), layout='constrained', sharex=True, sharey=True)
        for ax, sample, title in zip(axes, [regular, lhs], ['Regular grid: samples suggest a slowly increasing trend', 'Randomized LHS: samples reveal variation']):
            ax.plot(x/np.pi, np.sin(x), color='C0', alpha=0.65, linewidth=1.2, label=r'True function $y=\sin(x)$')
            ax.scatter(sample/np.pi, np.sin(sample), color='C1', marker='o', s=30, zorder=3, label=r'$N=11$ observations')
            ax.axhline(0,color='0.5',linewidth=0.6,alpha=0.4)
            ax.set(ylabel=r'$y$',ylim=(-1.2,1.2),xlim=(-0.2,20.4),yticks=[-1,0,1])
            ax.set_title(title,fontsize=11)
        axes[0].legend(loc='upper right', ncol=2,fontsize=9,frameon=True,framealpha=1,edgecolor='none')
        axes[1].set(xlabel=r'$x/\pi$',xticks=np.arange(0,21,4))
        return fig


def draw_sigmoid():
    z=np.linspace(-8,8,500)
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.1),layout='constrained')
        ax.plot(z,1/(1+np.exp(-z)),color='C0',label=r'$\sigma(z)=1/(1+\exp(-z))$')
        for bound in [0,1]:
            ax.axhline(bound,color='0.5',linestyle='--',linewidth=0.8)
        ax.axvline(0,color='0.5',linewidth=0.6,alpha=0.4)
        ax.set(xlabel=r'$z$',ylabel=r'$\sigma(z)$',xlim=(-8,8),ylim=(-0.06,1.06),yticks=[0,.5,1])
        ax.legend(loc='upper left',frameon=False)
        return fig


def draw_mlp():
    from matplotlib.patches import Circle
    with plt.rc_context(figure_style()):
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.6),layout='constrained')
        columns=[np.array([[x,y] for y in np.linspace(-0.8,0.8,n)]) for x,n in [(0,3),(1.5,4),(3,4),(4.5,2)]]
        for left,right in zip(columns[:-1],columns[1:]):
            for a in left:
                for b in right:
                    ax.plot([a[0],b[0]],[a[1],b[1]],color='0.5',alpha=0.38,linewidth=0.7,zorder=0)
        for k,points in enumerate(columns):
            for x,y in points:
                ax.add_patch(Circle((x,y),0.115,facecolor=plt.rcParams['axes.facecolor'],edgecolor=f'C{k%3}',linewidth=1.5,zorder=3))
        for i,(x,y) in enumerate(columns[0]):
            ax.text(x-.2,y,rf'$x_{i+1}$',ha='right',va='center')
        for i,(x,y) in enumerate(columns[-1]):
            ax.text(x+.2,y,rf'$y_{{\theta,{i+1}}}(\mathbf{{x}})$',ha='left',va='center')
        for x,title in [(0,'Features'),(1.5,'Hidden layer 1'),(3,'Hidden layer 2'),(4.5,'Prediction')]:
            ax.text(x,1.18,title,ha='center',fontsize=10)
        for x,label in [(.75,r'$W^{(1)},\,\mathbf{b}^{(1)}$'),(2.25,r'$W^{(2)},\,\mathbf{b}^{(2)}$'),(3.75,r'$W^{(3)},\,\mathbf{b}^{(3)}$')]:
            ax.text(x,-1.2,label,ha='center',fontsize=10)
        ax.annotate('',xy=(4.5,-1.62),xytext=(0,-1.62),arrowprops={'arrowstyle':'->','color':plt.rcParams['text.color'],'lw':1})
        ax.text(2.25,-1.78,'Forward evaluation',ha='center',va='top',fontsize=10)
        ax.set(xlim=(-.6,5.5),ylim=(-2,1.45),aspect='equal')
        ax.axis('off')
        return fig


def error_visualization_data():
    rng=np.random.default_rng(20261019)
    features=rng.random((1200,2))
    target=np.sin(2*np.pi*features[:,0])+0.3*features[:,1]
    scale=0.045+0.26*np.exp(-features[:,0]/0.13)
    error=0.035+rng.normal(size=1200)*scale
    outcomes=[rng.lognormal(np.log(median),spread,24) for median,spread in [(0.025,.3),(.018,.5),(.022,.2)]]
    return features,target,error,outcomes


def draw_error_visualization(kind):
    from scipy.stats import binned_statistic_2d
    features,target,error,outcomes=error_visualization_data()
    with plt.rc_context(figure_style()):
        # Compress the fixed-aspect heat-map axes and colorbar as a centered group.
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,3.8),
                            layout='compressed' if kind == 'map' else 'constrained')
        if kind == 'box':
            box=ax.boxplot(outcomes,tick_labels=['A','B','C'],patch_artist=True,
                           medianprops={'color':'C1','linewidth':1.6},
                           boxprops={'facecolor':'C0','alpha':.28},
                           whiskerprops={'color':plt.rcParams['text.color']},
                           capprops={'color':plt.rcParams['text.color']},
                           flierprops={'marker':'o','markersize':3,'markeredgecolor':'C0'})
            ax.set(xlabel='Illustrative architecture',ylabel='Evaluation MSE')
        elif kind == 'residual':
            ax.scatter(target,error,s=7,alpha=.35,color='C0',rasterized=True)
            ax.axhline(0,color='C1',linestyle='--',linewidth=1)
            ax.set(xlabel=r'Test target $y$',ylabel=r'$e=y_\theta(\mathbf{x})-y$')
        elif kind == 'hist':
            ax.hist(error,bins=35,color='C0',alpha=.8)
            ax.axvline(0,color='C1',linestyle='--',linewidth=1,label='Zero error')
            ax.set(xlabel=r'Signed error $e$',ylabel='Number of samples')
        elif kind == 'map':
            binned,xedges,yedges,_=binned_statistic_2d(features[:,0],features[:,1],np.abs(error),statistic='max',bins=8,range=[[0,1],[0,1]])
            cmap=plt.get_cmap('viridis').copy()
            cmap.set_bad('0.5')
            mesh=ax.pcolormesh(xedges,yedges,np.ma.masked_invalid(binned.T),cmap=cmap,shading='flat',vmin=0)
            ax.set(xlabel=r'Normalized feature $X_1$',ylabel=r'Normalized feature $X_2$',aspect='equal')
            fig.colorbar(mesh,ax=ax,label=r'Maximum $|e|$ in bin',shrink=.9)
        return fig


if __name__ == '__main__':
    import argparse
    import subprocess
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', choices=['grid', 'random', 'lhs', 'sobol', 'aliasing', 'sigmoid', 'mlp', 'errors-box', 'errors-residual', 'errors-hist', 'errors-map'])
    parser.add_argument('--theme', choices=['light', 'dark'])
    args = parser.parse_args()
    designs = sampling_designs()
    lhs_bins = np.floor(16 * designs['lhs']).astype(int)
    assert all(len(np.unique(lhs_bins[:, j])) == 16 for j in range(2))
    if args.design is None:
        # Separate renderer processes prevent TeX labels disappearing between themes.
        for method in [*designs, 'aliasing', 'sigmoid', 'mlp', 'errors-box', 'errors-residual', 'errors-hist', 'errors-map']:
            for mode in ('light', 'dark'):
                subprocess.run([sys.executable, __file__, '--design', method,
                                '--theme', mode], check=True)
    else:
        points = designs.get(args.design)
        if points is not None:
            assert points.shape == (16, 2) and np.all((points >= 0) & (points <= 1))
        styles = ['default']
        if args.theme == 'dark':
            styles += ['dark_background', {'figure.facecolor': '#111111',
                                           'axes.facecolor': '#111111'}]
        with plt.style.context(styles):
            fig = (draw_error_visualization(args.design.removeprefix('errors-')) if args.design.startswith('errors-') else
                   draw_mlp() if args.design == 'mlp' else
                   draw_sigmoid() if args.design == 'sigmoid' else
                   draw_aliasing() if args.design == 'aliasing' else
                   draw_lhs_process() if args.design == 'lhs' else
                   draw_sobol_process() if args.design == 'sobol' else
                   draw_design(points, args.design))
            output = Path(__file__).parent / 'media'
            output.mkdir(parents=True, exist_ok=True)
            fig.savefig(output / f'doe-{args.design}-{args.theme}.png', dpi=DPI,
                        facecolor=fig.get_facecolor(), transparent=False)
            plt.close(fig)
