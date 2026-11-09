"""Compare exported predictions with each snapshot and audit shape motion.

python scripts/audit_bubble_surrogates.py --data DATA --models MODELS --output DIR
These are retrospective diagnostics, not a fresh independent test campaign.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from bubble_surrogate_tools import load_data,split_masks
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
log,times,surface=load_data(a.data)
sessions={key:torch._inductor.aoti_load_package(str(a.models/f'{key}.pt2')) for key in ['ub','r','ut']}
def predict(key,x):
 with torch.no_grad():
  return sessions[key](torch.from_numpy(np.asarray(x,dtype=np.float64).copy())).numpy().ravel()

records=[]
for key,frame in [('ub',times),('r',surface),('ut',surface)]:
 columns=['t'] if key=='ub' else ['t','theta'];frame=frame.copy();frame['error']=predict(key,frame[columns].to_numpy())-frame[key]
 tr,va,te=split_masks(frame.snapshot);frame['split']=np.where(tr,'train',np.where(va,'validation','test'))
 for index,group in frame.groupby('snapshot'):
  records.append({'target':key,'snapshot':int(index),'time':float(group.t.iloc[0]),'split':group.split.iloc[0], 'rmse':float(np.sqrt(np.mean(group.error**2))),'max_abs':float(group.error.abs().max())})
pd.DataFrame(records).to_csv(a.output/'snapshot-errors.csv',index=False)
angle=np.linspace(0,np.pi,801);geometry=[]
for t in times.t:
 x=np.column_stack((np.full_like(angle,t),angle));r=predict('r',x)
 # Finite difference of the exported graph, not a second independently fitted normal model.
 h=1e-4
 if t==0:rdot=(predict('r',np.column_stack((np.full_like(angle,h),angle)))-r)/h
 else:rdot=(predict('r',x+np.array([h,0]))-predict('r',x-np.array([h,0])))/(2*h)
 volume=2*np.pi/3*np.trapezoid(r**3*np.sin(angle),angle)
 geometry.append({'time':float(t),'min_radius':float(r.min()),'max_radius':float(r.max()),'volume':float(volume),'reference_volume':float(2*np.pi*np.interp(t,log.t,log.v)),'max_radius_speed':float(np.abs(rdot).max())})
g=pd.DataFrame(geometry);g['relative_volume_error']=g.volume/g.reference_volume-1;g.to_csv(a.output/'geometry.csv',index=False)
summary={'max_abs_relative_volume_error':float(g.relative_volume_error.abs().max()),'min_radius':float(g.min_radius.min()),'initial_radius_derivative_fd':float(g.max_radius_speed.iloc[0]),'notes':'FD startup derivative has O(h) error; exact zero checked separately by PyTorch autodiff. Snapshot tests are retrospective; no new-bubble generalization claim.'}
(a.output/'geometry-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
