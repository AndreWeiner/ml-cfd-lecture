"""Reproduce reference fits and check export/C++ inference without running CFD.

PYTHONPATH=notebooks python scripts/validate_bubble_workflow.py \
    --data datasets/ellipsoidal_bubble --output /tmp/bubble-check
Optionally --cpp /tmp/bubble-aoti-predict checks the actual boundary helper.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"notebooks"))
from bubble_surrogate_tools import load_data, fit_model, export_models, check_constraints
from bubble_surrogate_figures import save_diagnostics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cpp", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
log, times, surface = load_data(args.data)
models, histories, metrics = {}, {}, []
for kind, frame in (("ub", times), ("r", surface), ("ut", surface)):
    models[kind], histories[kind], result = fit_model(kind, frame, times.t.max())
    metrics.append(result)
    torch.save(models[kind].state_dict(), args.output/f"{kind}.pth")
    np.save(args.output/f"{kind}-history.npy", histories[kind])
check_constraints(models, times.t.max())
export = export_models(models, args.output, times.t.max())
save_diagnostics(models, histories, times, surface, args.output/"bubble-regression-diagnostics")
cpp_errors = []
if args.cpp:
    for kind, model in models.items():
        for batch in (1, 7, 101):
            n_features = 1 if kind == "ub" else 2
            x = np.column_stack((np.linspace(0, times.t.max(), batch),
                                 np.linspace(0, 2*np.pi, batch)))[:, :n_features]
            run = subprocess.run([str(args.cpp), str(args.output/f"{kind}.pt2"), str(n_features)],
                                 input=" ".join(map(str, x.ravel())), text=True,
                                 capture_output=True, check=True)
            prediction = np.fromstring(run.stdout, sep="\n")
            with torch.no_grad(): reference = model(torch.from_numpy(x)).numpy().ravel()
            np.testing.assert_allclose(prediction, reference, rtol=1e-9, atol=1e-10)
            cpp_errors.append(float(np.abs(prediction-reference).max()))
report = {"torch": torch.__version__, "snapshots": len(times),
          "nodata_rows_removed": surface.attrs["nodata_rows_removed"],
          "training": metrics, "max_aoti_error": float(export.max_export_error.max()),
          "max_cpp_error": max(cpp_errors) if cpp_errors else None}
(args.output/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
