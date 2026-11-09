"""Convert native OpenFOAM Sherwood tables to CSV for the course plots."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np

FIELDS = ("Sh100", "Sh1000", "Sh10000")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--mean-file", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-time", type=float, default=2.0)
args = parser.parse_args()

means = np.loadtxt(args.mean_file, ndmin=2)
means = means[(means[:, 0] > 0) & (means[:, 0] <= args.max_time)]
rows = []
max_mean_error = 0.0
face_counts = set()
time_dirs = {float(path.name): path for path in args.surface.iterdir() if path.is_dir()}
for time, total_area, *mean in means:
    folder = time_dirs[time]
    tables = [np.loadtxt(folder / f"{field}_patch_bubble.raw", ndmin=2) for field in FIELDS]
    xyz = tables[0][:, :3]
    area_vectors = tables[0][:, 4:7]
    for table in tables[1:]:
        np.testing.assert_array_equal(table[:, :3], xyz)
        np.testing.assert_array_equal(table[:, 4:7], area_vectors)
    area = np.linalg.norm(area_vectors, axis=1)
    sh = np.column_stack([table[:, 3] for table in tables])
    assert np.isfinite(sh).all() and (area > 0).all()
    np.testing.assert_allclose(area.sum(), total_area, rtol=1e-10)
    check = np.average(sh, axis=0, weights=area)
    np.testing.assert_allclose(check, mean, rtol=1e-10, atol=1e-10)
    max_mean_error = max(max_mean_error, float(np.max(np.abs(check - mean))))
    theta = np.arctan2(np.hypot(xyz[:, 0], xyz[:, 2]), xyz[:, 1])
    for face in np.argsort(theta):
        rows.append([time, theta[face], *xyz[face], area[face], *sh[face]])
    face_counts.add(len(xyz))

assert len(means) and len(face_counts) == 1
args.output.mkdir(parents=True, exist_ok=False)
tables = (
    ("local-sherwood.csv", ["time", "theta", "x", "y", "z", "area", *FIELDS], rows),
    ("mean-sherwood.csv", ["time", *FIELDS, "area"], means[:, [0, 2, 3, 4, 1]]),
)
for name, header, data in tables:
    with (args.output / name).open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(data)

report = {
    "source": "OpenFOAM patch snGrad and surfaceFieldValue areaAverage",
    "frames": len(means),
    "faces_per_frame": face_counts.pop(),
    "diameter": 1.0,
    "delta_c": 1.0,
    "normal": "outward from liquid into excluded gas",
    "theta": "0 at top, pi at bottom; radians",
    "max_abs_mean_check_error": max_mean_error,
}
(args.output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
