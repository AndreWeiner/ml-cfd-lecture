"""Small supervised-training components for the optional neural-network reading."""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values):
        scale = values.std(dim=0, unbiased=False)
        return cls(values.mean(dim=0), torch.where(scale > 0, scale, torch.ones_like(scale)))

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean


class HiddenBlock(nn.Module):
    def __init__(self, width, normalization, activation, dropout, residual):
        super().__init__()
        norm = {'none': nn.Identity, 'layer': lambda: nn.LayerNorm(width),
                'batch': lambda: nn.BatchNorm1d(width)}[normalization]
        self.branch = nn.Sequential(nn.Linear(width, width), norm(), activation(),
                                    nn.Dropout(dropout))
        self.residual = residual

    def forward(self, x):
        out = self.branch(x)
        return x + out if self.residual else out


class ProfileMLP(nn.Module):
    def __init__(self, n_in=2, n_out=1, width=32, depth=3, normalization='none',
                 residual=False, concatenate=False, dropout=0., activation='silu'):
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be positive')
        act = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'silu': nn.SiLU}[activation]
        self.input = nn.Sequential(nn.Linear(n_in, width), act())
        self.blocks = nn.ModuleList([
            HiddenBlock(width, normalization, act, dropout, residual)
            for _ in range(depth - 1)
        ])
        self.concatenate = concatenate
        self.output = nn.Linear(width + (n_in if concatenate else 0), n_out)

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        if self.concatenate:
            h = torch.cat((h, x), dim=-1)
        return self.output(h)


@torch.no_grad()
def evaluate(model, loader, loss_fn, metric_fn=None, device='cpu'):
    """loss_fn and metric_fn return means over equally weighted samples."""
    model.eval()
    loss_sum = metric_sum = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        metric = loss if metric_fn is None else metric_fn(pred, y)
        if not torch.isfinite(loss) or not torch.isfinite(metric):
            raise FloatingPointError('Non-finite evaluation value')
        count += len(x)
        loss_sum += len(x) * loss.item()
        metric_sum += len(x) * metric.item()
    if not count:
        raise ValueError('Empty evaluation loader')
    return {'loss': loss_sum / count, 'metric': metric_sum / count}


def fit(model, train_loader, validation_loader, loss_fn, optimizer, *, epochs=100,
        metric_fn=None, scheduler=None, scheduler_interval='epoch',
        max_updates=None, patience=20, device='cpu'):
    """Return validation-selected state and history; never access test data."""
    if patience is not None and patience < 1:
        raise ValueError('patience must be positive')
    if epochs < 1 or (max_updates is not None and max_updates < 1):
        raise ValueError('Training budgets must be positive')
    if scheduler_interval not in ('epoch', 'update', 'validation'):
        raise ValueError('Unknown scheduler interval')
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if scheduler is not None and plateau != (scheduler_interval == 'validation'):
        raise ValueError('ReduceLROnPlateau requires validation stepping')
    model.to(device)
    history, best, best_value, stale, updates, samples = [], None, float('inf'), 0, 0, 0
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        online_sum, online_n, lr_first, lr_last = 0., 0, None, None
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if len(x) < 2 and any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
                raise ValueError('BatchNorm training requires at least two samples per batch')
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if loss.ndim or not torch.isfinite(loss):
                raise FloatingPointError('Loss must be a finite scalar mean')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite gradient')
            lr_last = optimizer.param_groups[0]['lr']
            lr_first = lr_last if lr_first is None else lr_first
            optimizer.step()
            updates += 1
            samples += len(x)
            online_sum += len(x) * loss.item()
            online_n += len(x)
            if scheduler is not None and scheduler_interval == 'update':
                scheduler.step()
            if max_updates is not None and updates >= max_updates:
                break
        if not online_n:
            raise ValueError('Empty training loader')
        validation = evaluate(model, validation_loader, loss_fn, metric_fn, device)
        # Fixed-order evaluation makes training and validation curves comparable.
        training = evaluate(model, DataLoader(train_loader.dataset, batch_size=1024),
                            loss_fn, metric_fn, device)
        if scheduler is not None:
            if scheduler_interval == 'validation':
                scheduler.step(validation['metric'])
            elif scheduler_interval == 'epoch':
                scheduler.step()
        improved = validation['metric'] < best_value
        if improved:
            best_value, stale = validation['metric'], 0
            best = {'model': deepcopy(model.state_dict()),
                    'optimizer': deepcopy(optimizer.state_dict()),
                    'scheduler': None if scheduler is None else deepcopy(scheduler.state_dict()),
                    'epoch': epoch, 'updates': updates}
        else:
            stale += 1
        history.append({'epoch': epoch, 'updates': updates, 'samples_seen': samples,
                        'online_loss': online_sum / online_n,
                        'train_loss': training['loss'], 'validation_loss': validation['loss'],
                        'train_metric': training['metric'], 'validation_metric': validation['metric'],
                        'lr_first': lr_first, 'lr_last': lr_last,
                        'seconds': perf_counter() - start})
        if (max_updates is not None and updates >= max_updates) or (patience is not None and stale >= patience):
            break
    model.load_state_dict(best['model'])
    model.eval()
    return {'history': history, 'best': best, 'best_validation_metric': best_value}


def make_profile_problem(seed=2026, outliers=False):
    """Synthetic channel-like profiles; not CFD data or a turbulence solution."""
    ub = torch.linspace(.1, 1., 12)
    faces = torch.linspace(0, 1, 129).square()
    eta = .5 * (faces[:-1] + faces[1:])
    weights = faces[1:] - faces[:-1]
    re = ub / 1e-5
    a = re.pow(.8) / 30.
    g = torch.log1p(a[:, None] * (2 * eta - eta.square()))
    clean = ub[:, None] * g / (g * weights).sum(dim=1, keepdim=True)
    x = torch.stack((eta.expand(12, -1), re[:, None].expand(-1, 128)), dim=-1)
    split = {'train': [0, 1, 3, 4, 6, 7, 10, 11], 'validation': [2, 8], 'test': [5, 9]}
    generator = torch.Generator().manual_seed(seed)
    observed = clean + .002 * ub[:, None] * torch.randn(clean.shape, generator=generator)
    xs = Standardizer.fit(x[split['train']].reshape(-1, 2))
    ys = Standardizer.fit(observed[split['train']].reshape(-1, 1))
    datasets = {}
    for name, indices in split.items():
        labels = observed[indices] if name == 'train' else clean[indices]
        scaled_y = ys.transform(labels.reshape(-1, 1))
        if name == 'train' and outliers:
            scaled_y = scaled_y.clone()
            # Fixed target corruption; scalers and all other splits stay unchanged.
            scaled_y[::20] += 3.
        datasets[name] = TensorDataset(xs.transform(x[indices].reshape(-1, 2)), scaled_y)
    return {'datasets': datasets, 'x_scaler': xs, 'y_scaler': ys, 'split': split,
            'ub': ub, 'eta': eta, 'weights': weights, 'profiles': clean}
