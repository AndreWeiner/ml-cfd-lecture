"""Actual Basilisk samples before/after frame and interface-basis transforms."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from figure_style import figure_style,WIDTH_MM,DPI, add_renderer_argument, resolve_renderer
from figure_utils import save_theme_variants
from bubble_surrogate_tools import load_log
COLORS=['#1f77b4','#ff7f0e','#2ca02c']
def save_data_transforms(data, output, *, snapshot_dt=.15, use_tex=True):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    log=load_log(data/'log.solver');samples=[]
    for index in (1,10,90):
        t=snapshot_dt*index
        parts=[np.loadtxt(p,ndmin=2) for p in sorted((data/'plic').glob(f'points_{index:06d}_n*.txt')) if p.read_text().strip()]
        raw=np.concatenate(parts);raw=raw[(np.abs(raw[:,4:6])<1e20).all(1)]
        center=np.interp(t,log.t,log.x);speed=np.interp(t,log.t,log.ux)
        pos=raw[:,:2].copy();pos[:,0]-=center
        vel=raw[:,4:6].copy();vel[:,0]-=speed
        normal=raw[:,2:4].copy();normal/=np.linalg.norm(normal,axis=1)[:,None]
        normal*=np.where((normal*pos).sum(1)<0,-1.,1.)[:,None]
        tangent=np.column_stack((-normal[:,1],normal[:,0]))
        theta=np.arctan2(pos[:,1],pos[:,0]);order=np.argsort(theta)
        # Same angle-ordered, at-most-64 sample rule as the regression helper.
        sel=order[np.linspace(0,len(order)-1,min(64,len(order))).astype(int)]
        samples.append(dict(t=t,center=center,speed=speed,raw=raw[sel],pos=pos[sel],vel=vel[sel],theta=theta[sel],radius=np.linalg.norm(pos[sel],axis=1),un=(vel*normal).sum(1)[sel],ut=(vel*tangent).sum(1)[sel]))
    def style():
        plt.rcParams.update(figure_style(use_tex=use_tex));plt.rcParams['axes.prop_cycle']=cycler(color=COLORS)
    def cartesian():
        style();f,axes=plt.subplots(2,3,figsize=(WIDTH_MM['double']/25.4,140/25.4),layout='constrained')
        for j,s in enumerate(samples):
            for row in (0,1):
                ax=axes[row,j];pos=s['raw'][:,:2] if row==0 else s['pos'];vel=s['raw'][:,4:6] if row==0 else s['vel']
                ax.plot(pos[:,1],pos[:,0],color=COLORS[j],marker='o',ms=2,lw=.8)
                q=ax.quiver(pos[::5,1],pos[::5,0],vel[::5,1],vel[::5,0],angles='xy',scale_units='xy',scale=20,color=COLORS[j],width=.008)
                c=s['center'] if row==0 else 0
                ax.set(xlim=(-.06,.98),ylim=(c-.67,c+.72),aspect='equal',xlabel=r'Radial $y$',ylabel=r'Lab $x$' if row==0 else r'$x-x_c(t)$')
                ax.ticklabel_format(useOffset=False,axis='y');ax.grid(alpha=.15)
                if row==0:ax.set_title(rf'$\tilde t={s["t"]:g}$'+'\n'+rf'$U_b={s["speed"]:.2f}$',fontsize=10)
                if row==1 and j==0:ax.quiverkey(q,.35,.93,1,r'$u=1$',coordinates='axes',labelpos='E',fontproperties={'size':9})
        return f
    def polar():
        style();f,axes=plt.subplots(1,3,figsize=(WIDTH_MM['double']/25.4,73/25.4),layout='constrained')
        for j,(key,label) in enumerate([('radius',r'$r(\theta)$'),('un',r'$u_n^{\mathrm{rel}}(\theta)$'),('ut',r'$u_t^{\mathrm{rel}}(\theta)$')]):
            for i,s in enumerate(samples):
                axes[j].plot(np.rad2deg(s['theta']),s[key],color=COLORS[i],ls=['-','--','-.'][i],marker=['o','s','^'][i],markevery=10,ms=3,label=rf'$\tilde t={s["t"]:g}$')
            axes[j].set(xlim=(0,180),xticks=[0,90,180],xlabel=r'$\theta\;(\mathrm{degrees})$',ylabel=label);axes[j].grid(alpha=.18)
        axes[0].legend(fontsize=9,frameon=False,loc='best')
        return f
    for name,draw in [('bubble-data-cartesian',cartesian),('bubble-data-polar',polar)]:
        for ext in ('svg','pdf','png'):save_theme_variants(draw,output/f'{name}.{ext}',dpi=DPI)
    return samples
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot-dt',type=float,default=.15)
    add_renderer_argument(p, default="auto")
    a = p.parse_args()
    if a.output.exists():raise FileExistsError('Use an initially empty output directory')
    save_data_transforms(a.data,a.output,snapshot_dt=a.snapshot_dt, use_tex=resolve_renderer(a.renderer))
