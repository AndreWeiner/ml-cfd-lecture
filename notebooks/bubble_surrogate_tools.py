"""Small, CPU-only building blocks for the continuous-bubble code notebook.

Basilisk coordinates are (vertical x, radial y); all quantities are dimensionless.
Exported models accept float64 arrays [N, 1] (rise) or [N, 2] (time, angle),
and return [N, 1]. Scaling and physical constraints are inside each compiled package.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch
from torch import nn


def load_log(path):
    """Ignore solver messages; retain complete, numeric nine-column rows."""
    frame = pd.read_csv(path, sep=r"\s+", names=list("tvxyz") + ["ux", "uy", "uz", "area"],
                        on_bad_lines="skip")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    return frame.sort_values("t").drop_duplicates("t").reset_index(drop=True)


def load_data(directory, snapshot_dt=.15, points_per_snapshot=64):
    directory = Path(directory)
    log = load_log(directory / "log.solver")
    groups = {}
    for path in sorted((directory / "plic").glob("points_*_n*.txt")):
        groups.setdefault(int(path.stem.split("_")[1]), []).append(path)
    rows, rejected = [], 0
    for index, paths in sorted(groups.items()):
        time = index * snapshot_dt
        if time > log.t.iloc[-1] + np.median(np.diff(log.t)):
            raise ValueError("Snapshot time exceeds log: check snapshot_dt")
        parts = [np.loadtxt(path, ndmin=2) for path in paths if path.read_text().strip()]
        points = np.concatenate(parts)
        if not np.isfinite(points).all() or points.shape[1] != 6:
            raise ValueError(f"Invalid six-column interface snapshot {index}")
        # Basilisk nodata is ~1e30, not a physical velocity outlier.
        valid = (np.abs(points[:, 4:6]) < 1e20).all(axis=1)
        rejected += int((~valid).sum())
        points = points[valid]
        position = points[:, :2].copy()
        position[:, 0] -= np.interp(time, log.t, log.x)
        velocity = points[:, 4:6].copy()
        velocity[:, 0] -= np.interp(time, log.t, log.ux)
        radius = np.linalg.norm(position, axis=1)
        normal = points[:, 2:4].copy()
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal *= np.where((normal*position).sum(1) < 0, -1., 1.)[:, None]
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        theta = np.arctan2(position[:, 1], position[:, 0])
        order = np.argsort(theta)
        select = order[np.linspace(0, len(order)-1, min(len(order), points_per_snapshot)).astype(int)]
        rows.append(pd.DataFrame({"snapshot": index, "t": time, "theta": theta[select],
                                  "r": radius[select], "ut": (velocity*tangent).sum(1)[select]}))
    interface = pd.concat(rows, ignore_index=True)
    interface.attrs["nodata_rows_removed"] = rejected
    times = interface.groupby("snapshot").t.first().reset_index()
    times["ub"] = np.interp(times.t, log.t, log.ux)
    return log, times, interface


class BubbleModel(nn.Module):
    """Constrained physical prediction; preprocessing travels with the model."""
    def __init__(self, kind, time_scale, output_scale, r0=.5, width=64, depth=3):
        super().__init__()
        self.kind = kind
        self.register_buffer("time_scale", torch.tensor(float(time_scale), dtype=torch.float64))
        self.register_buffer("output_scale", torch.tensor(float(output_scale), dtype=torch.float64))
        self.register_buffer("r0", torch.tensor(float(r0), dtype=torch.float64))
        layers = [nn.Linear(1 if kind == "ub" else 2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))
        self.double()

    def forward(self, features):
        t = features[:, :1]
        q = torch.log1p(t) / torch.log1p(self.time_scale)
        start = t / (1+t)
        if self.kind == "ub":
            return self.output_scale * start * self.net(q)
        angle = features[:, 1:2]
        value = self.net(torch.cat((q, torch.cos(angle)), dim=1))
        if self.kind == "r":
            return self.r0 + self.output_scale * start.square() * value
        return self.output_scale * start * torch.sin(angle) * value


def split_masks(snapshot):
    """Hold out entire snapshots, never individual neighboring surface points."""
    test = np.asarray(snapshot) % 10 == 5
    validation = (np.asarray(snapshot) % 5 == 0) & ~test
    return ~(validation | test), validation, test


def fit_model(kind, frame, time_scale, *, epochs=2500, width=64, depth=3,
              lbfgs_steps=40, seed=7):
    torch.manual_seed(seed)
    columns = ["t"] if kind == "ub" else ["t", "theta"]
    x = torch.tensor(frame[columns].to_numpy(), dtype=torch.float64)
    y = torch.tensor(frame[[kind]].to_numpy(), dtype=torch.float64)
    train, validation, test = split_masks(frame.snapshot)
    scale = max(float(y[train].max()-y[train].min()), .05)
    model = BubbleModel(kind, time_scale, scale, width=width, depth=depth)
    best_loss, best, history = float("inf"), None, []
    start = perf_counter()

    def record():
        nonlocal best_loss, best
        with torch.no_grad():
            loss = ((model(x[train])-y[train])/scale).square().mean().item()
            val_loss = ((model(x[validation])-y[validation])/scale).square().mean().item()
        history.append((loss, val_loss))
        if val_loss < best_loss:
            best_loss, best = val_loss, deepcopy(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ((model(x[train])-y[train])/scale).square().mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            record()
    model.load_state_dict(best)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=20,
                                  history_size=50, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-13)

    def closure():
        optimizer.zero_grad()
        loss = ((model(x[train])-y[train])/scale).square().mean()
        loss.backward()
        return loss

    for _ in range(lbfgs_steps):
        optimizer.step(closure)
        record()
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        error = (model(x[test])-y[test]).numpy()
        val_error = (model(x[validation])-y[validation]).numpy()
    metrics = {"target": kind, "test_rmse": float(np.sqrt(np.mean(error**2))),
               "test_max_abs": float(np.abs(error).max()),
               "validation_rmse": float(np.sqrt(np.mean(val_error**2))), "scale": scale,
               "training_seconds": perf_counter()-start, "adam_steps": epochs,
               "lbfgs_blocks": lbfgs_steps, "width": width, "depth": depth,
               "train_samples": int(train.sum()), "validation_samples": int(validation.sum()),
               "test_samples": int(test.sum())}
    return model, np.asarray(history), metrics


def export_models(models, directory, time_max):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for kind, model in models.items():
        n_features = 1 if kind == "ub" else 2
        example = torch.ones((8, n_features), dtype=torch.float64)
        program = torch.export.export(
            model.eval(), (example,),
            dynamic_shapes={"features": {0: torch.export.Dim("batch", min=1)}})
        path = directory / f"{kind}.pt2"
        # C++ constants avoid an executable-stack marker from generated assembly.
        torch._inductor.aoti_compile_and_package(
            program, package_path=str(path),
            inductor_configs={"cpp.simdlen": 256, "cpp.threads": 1,
                              "aot_inductor.use_consts_asm_build": False})
        compiled = torch._inductor.aoti_load_package(str(path))
        for batch in (1, 7, 101):
            features = np.column_stack((np.linspace(0, time_max, batch),
                                        np.linspace(0, 2*np.pi, batch)))[:, :n_features]
            tensor = torch.from_numpy(features.copy())
            with torch.no_grad():
                reference = model(tensor).numpy()
                prediction = compiled(tensor).numpy()
            np.testing.assert_allclose(prediction, reference, rtol=1e-9, atol=1e-10)
            results.append({"model": kind, "batch": batch,
                            "max_export_error": float(np.abs(prediction-reference).max())})
    (directory / "contract.json").write_text(json.dumps({
        "format": "AOTInductor compiled PT2 package", "device": "cpu", "cpu_isa": "AVX2",
        "dtype": "float64", "input_layout": "contiguous row-major",
        "inputs": {"ub": ["t"], "r": ["t", "theta"], "ut": ["t", "theta"]},
        "output_shape": ["N", 1], "units": "Basilisk dimensionless units",
        "time_range": [0., float(time_max)], "theta_range": [0., float(np.pi)],
        "theta_extension": "radius even; signed tangential velocity odd; 2*pi periodic",
        "r0": .5, "torch": torch.__version__}, indent=2)+"\n")
    return pd.DataFrame(results)


def check_constraints(models, time_max):
    theta = torch.linspace(0, 2*torch.pi, 129, dtype=torch.float64)
    zero = torch.column_stack((torch.zeros_like(theta), theta))
    sample = torch.column_stack((torch.full_like(theta, time_max/2), theta))
    reflected = sample.clone(); reflected[:, 1] *= -1
    with torch.no_grad():
        torch.testing.assert_close(models["ub"](zero[:, :1]), torch.zeros((129, 1), dtype=torch.float64))
        torch.testing.assert_close(models["r"](zero), torch.full((129, 1), .5, dtype=torch.float64))
        torch.testing.assert_close(models["ut"](zero), torch.zeros((129, 1), dtype=torch.float64))
        torch.testing.assert_close(models["r"](sample), models["r"](reflected))
        torch.testing.assert_close(models["ut"](sample), -models["ut"](reflected))
        torch.testing.assert_close(models["ut"](sample[[0, 64, 128]]), torch.zeros((3, 1), dtype=torch.float64), atol=1e-12, rtol=0)
    initial = zero.clone().requires_grad_()
    derivative = torch.autograd.grad(models["r"](initial).sum(), initial)[0][:, 0]
    torch.testing.assert_close(derivative, torch.zeros_like(derivative))
