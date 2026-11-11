"""Compare Tripathi digitization with the discernible Figure 4 Sharaf markers.
Uses shared course styles, LaTeX and identical geometry in both themes.
"""
from pathlib import Path
import argparse,csv,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
from figure_style import figure_style,WIDTH_MM,DPI, add_renderer_argument, resolve_renderer
from figure_utils import save_theme_variants

def read(relative):
    with (ROOT/relative).open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    add_renderer_argument(p)
    args = p.parse_args()
    use_tex = resolve_renderer(args.renderer)
    args.output.mkdir(parents=True,exist_ok=True)
    sets=[read('datasets/path_shape_regimes/tripathi2015_digitization/points.csv'),read('datasets/path_shape_regimes/sharaf2017_digitization/points.csv')]
    palette=plt.colormaps['tab10'].colors
    style={**figure_style(use_tex=use_tex),'axes.prop_cycle':cycler(color=palette)}
    # Real SVG export verifies the TeX renderer before the paired exports.
    with plt.rc_context(style):
        f,a=plt.subplots();a.set_xlabel(r'$Ga$');a.set_ylabel(r'$Eo$');f.savefig(args.output/'tex-smoke.svg');plt.close(f)
    names=['I: Axisymmetric','II: Skirted','III: Oscillatory','IV: Peripheral breakup','V: Central breakup']
    markers=['o','^','v','s','D']
    def draw():
        plt.rcParams.update(style)
        fig,axes=plt.subplots(1,2,figsize=(WIDTH_MM['double']/25.4,125/25.4),sharex=True,sharey=True)
        fig.subplots_adjust(left=.09,right=.98,top=.83,bottom=.31,wspace=.13)
        titles=['Tripathi et al. (2015)\n104 digitized markers','Sharaf et al. (2017)\n183 discernible Figure 4 markers']
        for panel,(ax,rows,title) in enumerate(zip(axes,sets,titles)):
            for k,reg in enumerate(['I','II','III','IV','V'] if panel==0 else ['I','II','III','IV']):
                key='regime' if panel==0 else 'source_regime'
                r=[v for v in rows if v[key]==reg]
                label=names[k] if panel==0 else ['I: Axisymmetric','II: Skirted','III: Oscillatory','IV: Breakup (source label)'][k]
                ax.scatter([float(v['Ga']) for v in r],[float(v['Eo']) for v in r],marker=markers[k],s=22,facecolors='none',edgecolors=[palette[k]],linewidths=1.0,label=label)
            ax.set(xscale='log',yscale='log',xlim=(4,800),ylim=(.04,500),xlabel=r'$Ga$',title=title)
            ax.grid(alpha=.17,linewidth=.5)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1,frameon=False,fontsize=8)
        axes[0].set_ylabel(r'$Eo$')
        return fig
    def draw_overlay():
        from matplotlib.lines import Line2D
        plt.rcParams.update(style)
        fig,ax=plt.subplots(figsize=(WIDTH_MM['double']/25.4,145/25.4))
        fig.subplots_adjust(left=.11,right=.97,top=.84,bottom=.30)
        handles=[]
        for panel,rows in enumerate(sets):
            for k,reg in enumerate(['I','II','III','IV','V'] if panel==0 else ['I','II','III','IV']):
                key='regime' if panel==0 else 'source_regime'
                r=[v for v in rows if v[key]==reg]
                marker='X' if panel==1 and reg=='IV' else markers[k]
                # Open larger Tripathi symbols and smaller filled Sharaf symbols
                # remain distinguishable when observations coincide. No jitter.
                artist=ax.scatter([float(v['Ga']) for v in r],[float(v['Eo']) for v in r],
                    marker=marker,s=48 if panel==0 else 21,
                    facecolors='none' if panel==0 else [palette[k]],
                    edgecolors=[palette[k]],linewidths=1.2 if panel==0 else .6,zorder=3+panel)
                if panel==0:
                    handles.append(Line2D([],[],marker=marker,linestyle='none',color=palette[k],
                        markerfacecolor='none',label=names[k]))
                elif reg=='IV':
                    handles.append(Line2D([],[],marker='X',linestyle='none',color=palette[k],
                        label='IV: Breakup (Sharaf)'))
        ax.set(xscale='log',yscale='log',xlim=(4,800),ylim=(.04,500),xlabel=r'$Ga$',ylabel=r'$Eo$')
        ax.grid(alpha=.17,linewidth=.5)
        neutral=plt.rcParams['text.color']
        source_handles=[Line2D([],[],marker='o',markersize=8,linestyle='none',color=neutral,
            markerfacecolor='none',label='Tripathi (2015): open, 104 markers'),
            Line2D([],[],marker='o',markersize=5,linestyle='none',color=neutral,
            label='Sharaf (2017): filled, 183 markers')]
        fig.legend(handles=source_handles,loc='upper center',bbox_to_anchor=(.5,.995),frameon=False,fontsize=9)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False,fontsize=9)
        return fig
    save_theme_variants(draw_overlay,args.output/'bubble-source-overlay.svg',dpi=DPI)
    save_theme_variants(draw_overlay,args.output/'bubble-source-overlay.png',dpi=DPI)
    save_theme_variants(draw,args.output/'bubble-source-comparison.svg',dpi=DPI)
    save_theme_variants(draw,args.output/'bubble-source-comparison.png',dpi=DPI)
if __name__=='__main__':main()
