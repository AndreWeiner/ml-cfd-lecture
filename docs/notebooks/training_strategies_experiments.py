"""Run the controlled comparisons used by the optional training reading."""
import argparse
import hashlib
import json
import math
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from training_tools import ProfileMLP, evaluate, fit, make_profile_problem


def configurations():
    base = {'batch': 64, 'lr': .003, 'schedule': 'constant', 'normalization': 'none',
            'residual': False, 'concatenate': False, 'dropout': 0., 'activation': 'silu',
            'loss': 'mse', 'outliers': False, 'max_updates': None}
    def case(label, **changes):
        return {'label': label, **base, **changes}
    reference = case('Plain MLP (batch 64)')
    return {
        'batch': [case('Batch 16', batch=16), case('Batch 64'), case('Batch 256', batch=256), case('Full batch', batch=1024)],
        'update_budget': [case('Batch 16', batch=16, max_updates=1600), case('Batch 64', max_updates=1600), case('Full batch', batch=1024, max_updates=1600)],
        'learning_rate': [case('LR 0.0001', lr=.0001), case('LR 0.003'), case('LR 0.03', lr=.03)],
        'schedule': [case('Constant LR 0.003'), case('Step decay', schedule='step'), case('Cosine decay', schedule='cosine'), case('Validation plateau', schedule='plateau')],
        'architecture': [reference, case('LayerNorm', normalization='layer'), case('BatchNorm', normalization='batch'), case('Residual', residual=True), case('Concatenated skip', concatenate=True), case('Dropout 0.1', dropout=.1)],
        'activation': [case('SiLU'), case('Tanh', activation='tanh'), case('ReLU', activation='relu')],
        'combined': [case(f'LayerNorm + {activation} + {label}', normalization='layer', activation=activation, schedule=schedule)
                     for activation in ('silu', 'tanh', 'relu')
                     for schedule, label in (('constant', 'constant LR'), ('plateau', 'plateau LR'))],
        'loss': [case('MSE + outliers', outliers=True), case('MAE + outliers', loss='mae', outliers=True), case('Huber + outliers', loss='huber', outliers=True)],
    }


def run_case(config, seed, epochs):
    torch.manual_seed(seed)
    problem = make_profile_problem(outliers=config['outliers'])
    data = problem['datasets']
    generator = torch.Generator().manual_seed(seed + 1000)
    train = DataLoader(data['train'], batch_size=config['batch'], shuffle=True, generator=generator)
    validation = DataLoader(data['validation'], batch_size=1024)
    model = ProfileMLP(**{key: config[key] for key in ('normalization', 'residual', 'concatenate', 'dropout', 'activation')})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = {'mse': nn.MSELoss(), 'mae': nn.L1Loss(), 'huber': nn.HuberLoss(delta=.2)}[config['loss']]
    interval, scheduler = 'epoch', None
    if config['schedule'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=.3)
    elif config['schedule'] == 'cosine':
        interval = 'update'
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train), eta_min=1e-5)
    elif config['schedule'] == 'plateau':
        interval = 'validation'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=10, min_lr=1e-5)
    scale = problem['y_scaler'].scale
    def dimensional_mse(pred, target):
        return ((pred - target) * scale).square().mean()
    actual_epochs = epochs if config['max_updates'] is None else math.ceil(config['max_updates'] / len(train))
    result = fit(model, train, validation, loss_fn, optimizer, epochs=actual_epochs,
                 metric_fn=dimensional_mse, scheduler=scheduler, scheduler_interval=interval,
                 max_updates=config['max_updates'])
    result['best'].pop('optimizer')
    result['best'].pop('scheduler')
    result['config'], result['seed'] = config, seed
    result['parameters'] = sum(p.numel() for p in model.parameters())
    result['train_samples'] = len(data['train'])
    result['validation_rmse_m_s'] = result['best_validation_metric'] ** .5
    return result, model, problem


def available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


def execute_case(config, seed, epochs, threads, output, run_id):
    torch.set_num_threads(threads)
    result, model, problem = run_case(config, seed, epochs)
    torch.save({'model': result['best'].pop('model'), 'config': config,
                'x_mean': problem['x_scaler'].mean, 'x_scale': problem['x_scaler'].scale,
                'y_mean': problem['y_scaler'].mean, 'y_scale': problem['y_scaler'].scale},
               output / f'{run_id}.pt')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--groups', nargs='+', default=list(configurations()))
    parser.add_argument('--output', type=Path, default=Path('exercises/training_strategies'))
    parser.add_argument('--threads', type=int, default=1, help='CPU threads per training job')
    parser.add_argument('--workers', type=int, default=None, help='Concurrent jobs; defaults to available CPUs divided by threads')
    args = parser.parse_args()
    if args.threads < 1 or (args.workers is not None and args.workers < 1):
        parser.error('threads and workers must be positive')
    if args.workers is None:
        args.workers = max(1, available_cpus() // args.threads)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    start, runs, groups, jobs = perf_counter(), {}, {}, {}
    for group in args.groups:
        groups[group] = []
        for config in configurations()[group]:
            key = json.dumps({k: v for k, v in config.items() if k != 'label'}, sort_keys=True)
            identity = hashlib.sha256(key.encode()).hexdigest()[:12]
            groups[group].append({'label': config['label'], 'id': identity})
            for seed in args.seeds:
                jobs.setdefault(f'{identity}_{seed}', (config, seed))
    workers = min(args.workers, len(jobs))
    print(f'Using {workers} workers, {args.threads} threads per job ({available_cpus()} CPUs available)', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(execute_case, config, seed, args.epochs, args.threads, args.output, run_id): run_id
                   for run_id, (config, seed) in jobs.items()}
        for future in as_completed(futures):
            run_id = futures[future]
            result = future.result()
            runs[run_id] = result
            print(result['config']['label'], result['seed'], f"validation RMSE={result['validation_rmse_m_s']:.5g}", flush=True)
            (args.output / 'results.json').write_text(json.dumps({'groups': groups, 'runs': runs}, indent=2))
    # Select on mean validation error across seeds, excluding corrupted-target and fixed-update experiments.
    eligible = {entry['id'] for group, entries in groups.items() if group not in ('loss', 'update_budget') for entry in entries}
    winner = min(eligible, key=lambda key: sum(runs[f'{key}_{s}']['best_validation_metric'] for s in args.seeds) / len(args.seeds)) if eligible else None
    test_results = []
    for seed in args.seeds if winner is not None else []:
        saved = torch.load(args.output / f'{winner}_{seed}.pt', weights_only=True)
        problem = make_profile_problem()
        model = ProfileMLP(**{k: saved['config'][k] for k in ('normalization', 'residual', 'concatenate', 'dropout', 'activation')})
        model.load_state_dict(saved['model'])
        scale = saved['y_scale']
        metric = lambda pred, target: ((pred - target) * scale).square().mean()
        test = evaluate(model, DataLoader(problem['datasets']['test'], batch_size=1024), nn.MSELoss(), metric)
        test_results.append({'seed': seed, 'rmse_m_s': test['metric'] ** .5})
    output = {'groups': groups, 'runs': runs, 'epochs': args.epochs, 'seeds': args.seeds,
              'workers': workers, 'available_cpus': available_cpus(), 'threads': args.threads, 'torch': torch.__version__, 'seconds': perf_counter() - start,
              'selected_id': winner, 'selected_config': None if winner is None else saved['config'], 'test': test_results,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('training_tools.py', 'training_strategies_experiments.py')}}
    (args.output / 'results.json').write_text(json.dumps(output, indent=2))
    print('Finished', output['seconds'], 'seconds', flush=True)


if __name__ == '__main__':
    main()
