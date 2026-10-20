"""Compare training-profile counts with fixed validation profiles and settings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import torch
from torch.utils.data import TensorDataset
import training_strategies_experiments as experiments
from training_tools import Standardizer, make_profile_problem

SUBSETS = [[0, 11], [0, 4, 7, 11], [0, 1, 4, 6, 7, 11], [0, 1, 3, 4, 6, 7, 10, 11]]


def subset_problem(indices):
    original = make_profile_problem()
    original_ids = original['split']['train']
    x_all = original['x_scaler'].inverse(original['datasets']['train'].tensors[0]).reshape(8, 128, 2)
    y_all = original['y_scaler'].inverse(original['datasets']['train'].tensors[1]).reshape(8, 128, 1)
    positions = [original_ids.index(i) for i in indices]
    x, y = x_all[positions].reshape(-1, 2), y_all[positions].reshape(-1, 1)
    xs, ys = Standardizer.fit(x), Standardizer.fit(y)
    datasets = {'train': TensorDataset(xs.transform(x), ys.transform(y))}
    for name in ('validation', 'test'):
        a, b = original['datasets'][name].tensors
        datasets[name] = TensorDataset(xs.transform(original['x_scaler'].inverse(a)),
                                      ys.transform(original['y_scaler'].inverse(b)))
    return {**original, 'datasets': datasets, 'x_scaler': xs, 'y_scaler': ys,
            'split': {**original['split'], 'train': indices}}


def run(config, seed, indices, epochs):
    torch.set_num_threads(1)
    problem = subset_problem(indices)
    experiments.make_profile_problem = lambda **kwargs: problem
    result, model, _ = experiments.run_case(config, seed, epochs)
    result.pop('best')
    result['training_profiles'] = indices
    result['stopped_early'] = len(result['history']) < epochs
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50000)
    args = parser.parse_args()
    previous = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    workers = min(experiments.available_cpus(), len(SUBSETS)*len(previous['seeds']))
    start, results = perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run, previous['selected_config'], seed, indices, args.epochs)
                   for indices in SUBSETS for seed in previous['seeds']]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(len(result['training_profiles']), result['seed'], result['validation_rmse_m_s'], len(result['history']), flush=True)
    record = {'config': previous['selected_config'], 'seeds': previous['seeds'], 'epochs': args.epochs,
              'patience': 20, 'subsets': SUBSETS, 'validation_profiles': [2, 8], 'test_profiles': [5, 9],
              'workers': workers, 'threads': 1, 'runs': results, 'seconds': perf_counter()-start,
              'source_sha256': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                for n in ('training_tools.py', 'training_strategies_experiments.py', 'training_data_scaling.py')}}
    (args.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
