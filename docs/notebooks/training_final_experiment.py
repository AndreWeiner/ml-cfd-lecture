"""Run the selected configuration until patience-based stopping, with a generous cap."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from training_tools import ProfileMLP, evaluate, make_profile_problem
from training_strategies_experiments import available_cpus, execute_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50000)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    config, seeds = selection['selected_config'], selection['seeds']
    identity = selection['selected_id']
    args.output.mkdir(parents=True, exist_ok=False)
    workers = min(available_cpus(), len(seeds))
    start, runs = perf_counter(), {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(execute_case, config, seed, args.epochs, 1, args.output, f'{identity}_{seed}'): seed for seed in seeds}
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            if len(result['history']) >= args.epochs:
                raise RuntimeError(f'Seed {seed} reached the cap; increase --epochs to observe early stopping')
            assert result['history'][-1]['epoch'] - result['best']['epoch'] == 20
            runs[f'{identity}_{seed}'] = result
    torch.set_num_threads(1)
    problem = make_profile_problem()
    test_results = []
    for seed in seeds:
        saved = torch.load(args.output/f'{identity}_{seed}.pt', weights_only=True)
        model = ProfileMLP(**{k: config[k] for k in ('normalization', 'residual', 'concatenate', 'dropout', 'activation')})
        model.load_state_dict(saved['model'])
        metric = lambda pred, target: ((pred-target)*saved['y_scale']).square().mean()
        test = evaluate(model, DataLoader(problem['datasets']['test'], batch_size=1024), nn.MSELoss(), metric)
        test_results.append({'seed': seed, 'rmse_m_s': test['metric']**.5})
    record = {'groups': {'final': [{'label': 'LayerNorm + SiLU + plateau LR', 'id': identity}]},
              'runs': runs, 'epochs': args.epochs, 'patience': 20, 'seeds': seeds,
              'workers': workers, 'threads': 1, 'seconds': perf_counter()-start,
              'selected_config': config, 'selected_id': identity, 'test': test_results,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('training_tools.py', 'training_strategies_experiments.py', 'training_final_experiment.py')}}
    (args.output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(test_results), flush=True)


if __name__ == '__main__':
    main()
