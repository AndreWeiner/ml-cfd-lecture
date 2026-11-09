"""Paired local Sherwood profiles and mean histories from actual surface exports.

python scripts/plot_bubble_sherwood.py --data DIR --output EMPTY_DIR
"""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from figure_style import figure_style,WIDTH_MM,DPI
from figure_utils import save_theme_variants
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
local=pd.read_csv(a.data/'local-sherwood.csv');mean=pd.read_csv(a.data/'mean-sherwood.csv')
times = [mean.time.min(), .5, mean.time.max()]
profile_fields = ['Sh100', 'Sh1000', 'Sh10000']
profiles = [local[np.isclose(local.time, time)].sort_values('theta') for time in times]
values = np.concatenate([profile[profile_fields].to_numpy().ravel() for profile in profiles])
local_limits = (min(0, 1.05 * values.min()), 1.05 * values.max())
colors=['#1f77b4','#ff7f0e','#2ca02c'];styles=['-','--','-.'];markers=['o','s','^']
# Test a real TeX export before the full paired set; no renderer fallback.
with plt.rc_context(figure_style()):
 f,ax=plt.subplots();ax.set_xlabel(r'$\theta\;(\mathrm{degrees})$');ax.set_ylabel(r'$Sh_{\mathrm{local}}$');f.savefig(a.output/'tex-check.svg');plt.close(f)
def draw():
 plt.rcParams.update(figure_style());plt.rcParams['axes.prop_cycle']=cycler(color=colors)
 f,axes=plt.subplots(2,2,figsize=(WIDTH_MM['double']/25.4,140/25.4),layout='constrained')
 for ax, t, d in zip(axes.flat[:3], times, profiles):
  for i,sc in enumerate((100,1000,10000)):
   ax.plot(np.rad2deg(d.theta),d[f'Sh{sc}'],color=colors[i],ls=styles[i],marker=markers[i],markevery=11,ms=3,label=rf'$Sc={sc}$')
  ax.set(title=rf'$\tilde t={t:g}$',xlabel=r'$\theta\;(\mathrm{degrees})$',ylabel=r'$Sh_{\mathrm{local}}$',xlim=(0,180),ylim=local_limits,xticks=[0,45,90,135,180]);ax.grid(alpha=.18)
 ax=axes[1,1]
 for i,sc in enumerate((100,1000,10000)):
  ax.plot(mean.time,mean[f'Sh{sc}'],color=colors[i],ls=styles[i],label=rf'$Sc={sc}$')
 ax.set(title='Area-weighted mean',xlabel=r'$\tilde t$',ylabel=r'$\overline{Sh}$',yscale='log',xlim=(mean.time.min(),mean.time.max()));ax.grid(alpha=.18)
 axes[0,0].legend(fontsize=9,loc='lower center',frameon=False)
 return f
for ext in ('svg','pdf','png'):save_theme_variants(draw,a.output/f'bubble-local-sherwood.{ext}',dpi=DPI)
