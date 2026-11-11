"""Render paired movies of cell-associated concentration and the actual mesh.

python scripts/render_bubble_animations.py --frames DIR --models DIR --output DIR
The extracted half-plane is reflected only for the axisymmetric full-bubble
view. Local mesh coordinates rotate/translate with a surface point near 45 deg.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter
from cycler import cycler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from figure_style import figure_style,WIDTH_MM, add_renderer_argument, resolve_renderer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frames',type=Path,required=True)
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--preview-only',action='store_true')
add_renderer_argument(parser)
a = parser.parse_args()
use_tex = resolve_renderer(a.renderer)
a.output.mkdir(parents=True, exist_ok=True)
paths=sorted(a.frames.glob('frame-*.npz'))
checks=json.loads((a.frames/'frames.json').read_text())
torch.set_num_threads(1)
rise = torch._inductor.aoti_load_package(str(a.models / 'ub.pt2'))
times = np.array([frame['time'] for frame in checks], dtype=np.float64)
with torch.inference_mode():
    ub = rise(torch.from_numpy(times[:, None])).numpy().ravel()
norm=Normalize(0,1);width=WIDTH_MM['double']/25.4;dpi=1280/width


def full_interface(interface):
    return np.vstack((interface,interface[::-1]*[-1,1]))


def local_frame(interface):
    angle=np.arctan2(interface[:,0],interface[:,1]);i=np.argmin(np.abs(angle-np.pi/4))
    tangent=interface[min(i+1,len(interface)-1)]-interface[max(i-1,0)]
    tangent/=np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]])
    return interface[i],np.column_stack((tangent,normal))


def setup(mode,kind):
    plt.rcParams.update(figure_style(use_tex=use_tex))
    plt.rcParams['axes.prop_cycle']=cycler(color=plt.colormaps['tab10'].colors)
    fig=plt.figure(figsize=(width,width*900/1280))
    neutral=plt.rcParams['text.color'];background=fig.get_facecolor()
    axes=[];items=[]
    n=3 if kind=='concentration' else 2
    positions=[(.09+i*.30,.26,.26,.60) for i in range(3)] if n==3 else [(.10,.26,.35,.60),(.57,.26,.37,.60)]
    for i,rect in enumerate(positions):
        ax=fig.add_axes(rect);ax.set_aspect('equal');axes.append(ax)
        if kind=='concentration' or i==0:
            ax.set(xlim=(-.85,.85),ylim=(-1.2,.75),xlabel=r'$x/d_b$')
        else:
            ax.set(xlim=(-.008,.008),ylim=(-.003,.013),xlabel=r'$s/d_b$',ylabel=r'$n/d_b$')
            ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=False)
        if i==0:ax.set_ylabel(r'$y/d_b$')
        elif kind=='concentration':ax.tick_params(labelleft=False)
        if kind=='concentration':
            ax.set_title(r'$Sc='+str([100,1000,10000][i])+'$',pad=10)
            artists=[ax.add_collection(PolyCollection([],cmap='viridis',norm=norm,edgecolors='none',antialiased=False)) for _ in range(2)]
            grid=None
        elif i==0:
            ax.set_title('Deforming liquid mesh',pad=10)
            artists=[]
            grid=ax.add_collection(PolyCollection([],facecolors='none',edgecolors=neutral,linewidths=.22,alpha=.60))
        else:
            ax.set_title(r'Interface close-up: $Sc=1000$',pad=10)
            artists=[ax.add_collection(PolyCollection([],cmap='viridis',norm=norm,edgecolors=neutral,linewidths=.35,antialiased=False))]
            grid=None
        void=Polygon(np.zeros((3,2)),closed=True,facecolor=background,edgecolor='C1',lw=1.3,zorder=5)
        ax.add_patch(void)
        marker,=ax.plot([],[],marker='o',color='C3',markersize=4,zorder=6)
        items.append((artists,grid,void,marker))
    header=fig.text(.5,.96,'',ha='center',va='center',fontsize=12)
    footer='Computed cell values; shared flow, three diffusivities' if kind=='concentration' else 'Actual mesh; local view follows the surface near polar angle 45 degrees'
    fig.text(.5,.065,footer,ha='center',fontsize=10)
    fig.text(.5,.025,'Bubble-fixed frame; full view mirrored by axisymmetry',ha='center',fontsize=9)
    cax=fig.add_axes((.29,.16,.42,.020))
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cb.set_label(r'Normalized dissolved concentration $c/c_\Sigma$',fontsize=10,labelpad=1)
    return fig,items,header


def update(items,header,kind,data,index):
    vertices=data['vertices'];c=data['concentration'];interface=data['interface']
    outline=full_interface(interface)
    if kind=='concentration':
        for i,(artists,grid,void,marker) in enumerate(items):
            for sign,artist in zip([1,-1],artists):
                artist.set_verts(vertices*np.array([sign,1]));artist.set_array(c[:,i])
            void.set_xy(outline)
    else:
        origin,basis=local_frame(interface)
        artists,grid,void,marker=items[0]
        grid.set_verts(np.concatenate((vertices,vertices*np.array([-1,1]))))
        void.set_xy(outline);marker.set_data([origin[0]],[origin[1]])
        local=(vertices-origin)@basis
        lo=local.min(axis=1);hi=local.max(axis=1)
        mask=(lo[:,0]<.008)&(hi[:,0]>-.008)&(lo[:,1]<.013)&(hi[:,1]>-.003)
        artists,grid,void,marker=items[1]
        artists[0].set_verts(local[mask]);artists[0].set_array(c[mask,1])
        void.set_xy((outline-origin)@basis)
    header.set_text(r'$\tilde{t}='+f'{times[index]:.2f}'+r'\qquad Re(t)='+f'{238.2*ub[index]:.0f}'+'$')


for mode in ('light','dark'):
    styles=['default'] if mode=='light' else ['default','dark_background',{'figure.facecolor':'#111111','axes.facecolor':'#111111'}]
    for kind in ('concentration','mesh'):
        with plt.style.context(styles):
            fig,items,header=setup(mode,kind)
            last=np.load(paths[-1]);update(items,header,kind,last,len(paths)-1)
            fig.savefig(a.output/f'bubble-{kind}-{mode}.png',dpi=dpi)
            if not a.preview_only:
                writer=FFMpegWriter(fps=8,codec='libx264',extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart'])
                with writer.saving(fig,str(a.output/f'bubble-{kind}-{mode}.mp4'),dpi=dpi):
                    for index,path in enumerate(paths):
                        with np.load(path) as data:update(items,header,kind,data,index)
                        writer.grab_frame(facecolor=fig.get_facecolor())
            plt.close(fig)
(a.output/'animation-display.json').write_text(json.dumps({'frames':len(paths),'times':times.tolist(),'fps':8,'pixels':[1280,900],'width_mm':180,'renderer':'Matplotlib with LaTeX labels; libx264','field_association':'cell','interpolation':'none; flat cell colors','normalization':[0,1],'mirrored':'axisymmetric full view only','local_camera':'moves and rotates with mesh interface near theta=pi/4','Schmidt':[100,1000,10000],'Ga':238.2},indent=2)+'\n')
