#!/usr/bin/env python3
"""Fresh, auditable pixel digitization of Tripathi et al. (2015), Figure 1.

Input is the unaltered 517 x 364 image extracted with Poppler pdfimages from
page 3 of https://www.nature.com/articles/ncomms7268.pdf.
Point selection and physical classes were inspected on
the source image; calibration uses the labeled logarithmic axis ticks.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/path_shape_regimes/tripathi2015_digitization"

# Component IDs follow raster scan order, with the fixed threshold/filter below.
# Green and blue source regions use the caption/main Results numbering II/III.
SELECTED = {
    "I": [43,44,60,61,62,74,75,76,87,88,89,98,99,100,101,102,111,112,113,114],
    "II": [2,3,13,14,41,42],
    "III": [56,57,58,59,70,71,72,73,81,83,84,85,86,92,93,94,95,96,97,
            103,104,105,106,107,108,109,110],
    "IV": [5,6,15,16,26,32,51,45,63,64,67,77,79,80,90,91],
    "V": [7,8,9,10,11,17,18,19,20,21,27,28,29,30,31,33,34,35,36,
          46,47,48,49,53,65,68,69,78],
}
# Manually inspected centers of markers joined to a boundary/overlaid by a shape.
# Coordinates refer to pixel centers, with (0,0) at upper left. Triangle positions
# are the geometric centroid, not the bounding-box center (see below).
OVERRIDES = {
    32: (194.5,100.5,"Square joined to dashed boundary; center from visible square edges."),
    45: (313.0,127.5,"Square partly occluded by bubble illustration; visible right/bottom edges."),
    51: (276.5,127.5,"Square partly occluded by bubble illustration; visible side/bottom edges."),
    81: (355.5,217.7,"Triangle joined to boundary; visible apex at y213 and base at y220."),
    83: (194.5,245.3,"Triangle joined to boundary; position from visible triangle outline."),
    90: (385.5,245.0,"Square joined to boundary; position from visible square outline."),
    101:(194.0,272.0,"Circle joined to boundary; position from visible circular outline."),
    102:(203.5,308.0,"Circle joined to boundary; position from visible circular outline."),
}
ADDITIONS = [
    ("split23a","II",35.5,91.3,"Upper triangle of touching pair in component 23."),
    ("split23b","II",35.5,100.3,"Lower triangle of touching pair in component 23."),
    ("split24a","II",73.0,91.3,"Upper triangle of touching pair in component 24."),
    ("split24b","II",73.0,100.3,"Lower triangle of touching pair in component 24."),
    ("split25a","II",157.0,91.3,"Upper triangle joined to boundary in component 25."),
    ("split25b","II",157.0,100.3,"Lower triangle joined to boundary in component 25."),
    ("partialV","V",276.5,100.5,"Square partially covered by bubble cap; center from visible top and sides."),
]
CALIBRATION = {
    "Ga": {"pixel": [73.0,313.0], "value": [10.0,100.0]},
    "Eo": {"pixel": [64.0,155.0,245.0,335.0], "value": [100.0,10.0,1.0,0.1]},
}


def components(image):
    array = np.asarray(image.convert("RGB"))
    mask = array.max(axis=2) < 90
    mask[:10,:] = mask[-12:,:] = False
    mask[:,:12] = mask[:,-12:] = False
    seen = np.zeros(mask.shape, dtype=bool)
    results = []
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:
            continue
        points, stack = [], [(x,y)]
        seen[y,x] = True
        while stack:
            xx,yy = stack.pop()
            points.append((xx,yy))
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
                nx,ny = xx+dx,yy+dy
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny,nx] and not seen[ny,nx]):
                    seen[ny,nx] = True
                    stack.append((nx,ny))
        p = np.asarray(points)
        low, high = p.min(axis=0), p.max(axis=0)
        width,height = high-low+1
        if len(p) > 9 and width >= 5 and height >= 5:
            results.append({"id":len(results)+1, "x":float((low[0]+high[0])/2),
                            "y":float((low[1]+high[1])/2), "width":int(width),
                            "height":int(height), "area":len(p)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(DATA / "figure1-source.png")
    if image.size != (517,364):
        raise ValueError("Source raster dimensions changed; calibration must be re-inspected.")
    candidates = components(image)
    by_id = {item["id"]: item for item in candidates}
    rows = []
    for regime,ids in SELECTED.items():
        for cid in ids:
            c = by_id[cid]
            x,y = c["x"],c["y"]
            note = "Isolated marker; center measured from connected component."
            if regime in ("II","III"):
                # Equilateral upright triangle centroid lies 2/3 down from apex.
                y += (c["height"]-1)/6
            if cid in OVERRIDES:
                x,y,note = OVERRIDES[cid]
            rows.append((f"component{cid}",regime,x,y,note))
    rows.extend(ADDITIONS)
    coefficients = {key: np.polyfit(values["pixel"], np.log10(values["value"]),1)
                    for key,values in CALIBRATION.items()}
    output = []
    for point_id,regime,x,y,note in rows:
        ga = 10 ** np.polyval(coefficients["Ga"],x)
        eo = 10 ** np.polyval(coefficients["Eo"],y)
        output.append(dict(point_id=point_id,regime=regime,pixel_x=x,pixel_y=y,
                           Ga=f"{ga:.6f}",Eo=f"{eo:.6f}",note=note))
    output.sort(key=lambda r: (["I","II","III","IV","V"].index(r["regime"]),
                                r["pixel_x"],r["pixel_y"]))
    with (args.output_dir / "points.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    audit = {"source_image_size": image.size, "calibration": CALIBRATION,
             "log10_coordinate_coefficients": {k:v.tolist() for k,v in coefficients.items()},
             "counts": dict(Counter(r["regime"] for r in output)),
             "unresolved": [{"pixel_x":240.5,"pixel_y":127.5,
                             "reason":"Component 50: bubble illustration, dashed boundary and Morton line obscure the marker; omitted."}],
             "candidate_components":candidates}
    (args.output_dir / "audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    # Source overlay supports manual checking; it is not the lecture figure.
    overlay = image.resize((1551,1092))
    painter = ImageDraw.Draw(overlay)
    for row in output:
        x,y = row["pixel_x"]*3,row["pixel_y"]*3
        painter.ellipse((x-13,y-13,x+13,y+13),outline="red",width=2)
    overlay.save(args.output_dir / "extraction-overlay.png")
    print(f"Extracted {len(output)} points: {audit['counts']}")
    print(f"Wrote fresh digitization and audit to {args.output_dir}")


if __name__ == "__main__":
    main()
