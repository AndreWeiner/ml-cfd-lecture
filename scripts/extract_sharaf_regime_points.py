"""Reproduce calibrated Figure 4 digitization from reviewed pixel centers.
Input: embedded JPEG from PDF page 6, extracted without resampling by pdfimages.
Reproduce: .venv-quarto/bin/python scripts/extract_sharaf_regime_points.py --output /tmp/sharaf-extract
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/path_shape_regimes/sharaf2017_digitization'
CAL={'Ga':{'pixel':[632.,1787.],'value':[10.,100.]},'Eo':{'pixel':[315.,750.,1183.,1617.],'value':[100.,10.,1.,.1]}}
META={'black_circle':('I','spherical'),'red_circle':('I','oblate'),'purple_circle':('I','dimpled'),'blue_triangle':('II','skirted'),'black_triangle':('III','oscillatory'),'brown_square':('IV','skirted breakup'),'magenta_square':('IV','satellite or toroidal breakup; subtype unresolved')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DATA);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 fits={k:np.polyfit(v['pixel'],np.log10(v['value']),1) for k,v in CAL.items()}
 points=json.loads((DATA/'marker_centers.json').read_text());rows=[]
 image=Image.open(DATA/'figure4-source.jpg').convert('RGB');draw=ImageDraw.Draw(image)
 for i,v in enumerate(points,1):
  x,y=v['pixel_x'],v['pixel_y'];reg,behavior=META[v['source_marker']]
  # All centers are approximate figure digitizations, including apparently isolated ones.
  row=dict(point_id=f'sharaf-f4-{i:03}',Ga=f'{10**np.polyval(fits["Ga"],x):.6g}',Eo=f'{10**np.polyval(fits["Eo"],y):.6g}',source_regime=reg,behavior=behavior,source_marker=v['source_marker'],pixel_x=x,pixel_y=y,source_id='sharaf2017',source_url='https://doi.org/10.1063/1.5006726',source_location='Figure 4',note=v['note'])
  rows.append(row);draw.ellipse((x-8,y-8,x+8,y+8),outline='#17becf',width=3);draw.text((x+10,y-10),str(i),fill='#006060',stroke_width=0)
 with (a.output/'points.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 image.save(a.output/'extraction-overlay.png')
 audit=dict(marker_count=len(rows),counts={k:sum(v['source_marker']==k for v in points) for k in META},calibration=CAL,fits={k:v.tolist() for k,v in fits.items()},image_sha256=hashlib.sha256((DATA/'figure4-source.jpg').read_bytes()).hexdigest(),scope='All individually discernible markers selected from Figure 4; cannot recover fully occluded markers or underlying repeat runs.',mapping='Source IV deliberately not mapped to Tripathi IV/V.')
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(audit['counts']))
if __name__=='__main__':main()
