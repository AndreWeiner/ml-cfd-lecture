#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
from shutil import copytree
import subprocess
from scipy.stats import qmc


def replace_entry(path, name, value):
    text, count = re.subn(
        rf"(?m)^(\s*){re.escape(name)}\s+[^;]+;",
        lambda match: f"{match[1]}{name}    {value};",
        path.read_text(),
    )
    if count != 1:
        raise ValueError(f"Expected one {name} entry in {path}, found {count}")
    path.write_text(text)


def run_simulation(case):
    with (case / "log.Allrun").open("w") as log:
        subprocess.run(["bash", "./Allrun"], cwd=case, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    text = (case / "log.pimpleFoam").read_text()
    times = re.findall(r"^Time = (\S+)", text, re.MULTILINE)
    control = (case / "system/controlDict").read_text()
    end = float(re.search(r"^endTime\s+([^;]+);", control, re.MULTILINE)[1])
    step = float(re.search(r"^deltaT\s+([^;]+);", control, re.MULTILINE)[1])
    if ("FOAM FATAL" in text or "\nEnd\n" not in text or not times
            or abs(float(times[-1]) - end) > max(1e-6, 1.1 * step)):
        raise RuntimeError(f"Incomplete simulation: {case}")
    print(f"Finished {case.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.workers < 1:
        parser.error("samples and workers must be positive")
    if not os.environ.get("ML_CFD_BASE"):
        parser.error("ML_CFD_BASE is not set; source setup-env first")
    base = Path(os.environ["ML_CFD_BASE"])
    output = args.output or base / "exercises/boundary_layer_1D_variation"
    samples = qmc.LatinHypercube(d=1, rng=args.seed).random(args.samples)[:, 0]
    speeds = [round(0.1 + 0.9 * float(sample), 8) for sample in samples]
    if len(set(speeds)) != len(speeds):
        raise ValueError("Sampled speeds collide at the case naming precision")
    output.mkdir(parents=True, exist_ok=False)
    (output / "study.json").write_text(json.dumps({
        "seed": args.seed, "workers": args.workers, "bulk_speeds_m_s": speeds,
    }, indent=2))
    cases = []
    for speed in speeds:
        case = output / f"Ub_{speed:.8f}"
        copytree(base / "test_cases/boundary_layer_1D", case)
        replace_entry(case / "system/controlDict", "endTime", f"{50 / speed:.12g}")
        replace_entry(case / "system/controlDict", "deltaT", f"{1e-3 / speed:.12g}")
        replace_entry(case / "system/controlDict", "purgeWrite", "3")
        replace_entry(case / "system/fvOptions", "Ubar", f"({speed:.8f} 0 0)")
        cases.append(case)
    with ThreadPoolExecutor(max_workers=min(args.workers, args.samples)) as pool:
        list(pool.map(run_simulation, cases))
    print("Parameter variation finished", flush=True)


if __name__ == "__main__":
    main()
