"""Package the same 64 angle-ordered surface points used by the worked example.

The numeric solver log and all nodata records are retained. No values are fitted
or synthesized. Run from the repository root with the bubble Python environment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"notebooks"))
from bubble_surrogate_tools import load_log

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
log = load_log(args.source/"log.solver")
groups = {}
for path in sorted((args.source/"plic").glob("points_*_n*.txt")):
    groups.setdefault(int(path.stem.split("_")[1]), []).append(path)
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(args.source/"log.solver", "bubble_reference/log.solver")
    for index, paths in groups.items():
        parts = [np.loadtxt(p, ndmin=2) for p in paths if p.read_text().strip()]
        points = np.concatenate(parts)
        valid = (np.abs(points[:, 4:6]) < 1e20).all(axis=1)
        actual = points[valid]
        center = np.interp(.15*index, log.t, log.x)
        order = np.argsort(np.arctan2(actual[:, 1], actual[:, 0]-center))
        selected = order[np.linspace(0, len(order)-1, 64).astype(int)]
        retained = np.vstack((actual[selected], points[~valid]))
        content = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in retained)+"\n"
        archive.writestr(f"bubble_reference/plic/points_{index:06d}_n000.txt", content)
    metadata = {"source": "Existing course ellipsoidal_bubble dataset; original solver unchanged",
                "kind": "Teaching subset: 64 valid angle-ordered points per snapshot plus nodata records",
                "snapshots": len(groups), "snapshot_dt": .15, "Eo": .4361, "Ga": 238.2,
                "log_sha256": hashlib.sha256((args.source/"log.solver").read_bytes()).hexdigest(),
                "columns": ["x_vertical", "y_radial", "normal_x", "normal_y", "velocity_x", "velocity_y"],
                "units": "Basilisk dimensionless; d=1, initial radius=.5",
                "note": "All MPI records are merged; n000 denotes the packaged file, not an original rank."}
    archive.writestr("bubble_reference/provenance.json", json.dumps(metadata, indent=2)+"\n")
print(args.output, args.output.stat().st_size, "bytes")
