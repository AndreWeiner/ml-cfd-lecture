"""Render the optional reading from recorded training histories; no retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import figure_style
from figure_utils import save_theme_variants
from training_tools import ProfileMLP, make_profile_problem

PALETTE = [f'C{i}' for i in range(10)]
DARK_CYCLE = plt.style.library['dark_background']['axes.prop_cycle']
STYLES = ['-', '--', '-.', ':', '--', '-.']


def render(record_path, output):
    record = json.loads(record_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    def save(draw, name):
        for suffix in ['.png', '.pdf']:
            save_theme_variants(draw, output / f'training-strategies-{name}{suffix}', dpi=300)
    problem = make_profile_problem()
    def prototype():
        with plt.rc_context({**figure_style(), 'axes.prop_cycle': DARK_CYCLE}):
            fig, ax = plt.subplots(figsize=(180/25.4, 85/25.4), layout='constrained')
            colors = [PALETTE[i % 10] for i in range(12)]
            for ub, u, color in zip(problem['ub'], problem['profiles'], colors):
                ax.plot(u, .5*problem['eta'], color=color)
            cycle_colors = DARK_CYCLE.by_key()['color']
            cmap = matplotlib.colors.ListedColormap([cycle_colors[i % 10] for i in range(12)])
            ub = problem['ub'].numpy()
            bounds = np.r_[ub[0] - (ub[1]-ub[0])/2, (ub[:-1]+ub[1:])/2, ub[-1] + (ub[-1]-ub[-2])/2]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r'$\bar U$ (m/s)')
            ax.set(xlabel=r'$u_x$ (m/s)', ylabel='$y$ (m)', ylim=(0, .5))
            return fig
    save(prototype, 'prototype')
    for group, entries in record['groups'].items():
        entries = list(entries)
        if group == 'loss' and 'architecture' in record['groups']:
            control = record['groups']['architecture'][0]
            entries = [{'label': 'MSE, no outliers', 'id': control['id']}] + entries
        def curves(entries=entries, group=group):
            with plt.rc_context({**figure_style(), 'axes.prop_cycle': DARK_CYCLE}):
                fig, axs = plt.subplots(1, 2, figsize=(180/25.4, (120 if group == 'combined' else 100)/25.4), layout='constrained')
                for i, entry in enumerate(entries):
                    runs = [record['runs'][f"{entry['id']}_{seed}"] for seed in record['seeds']]
                    histories = [r['history'] for r in runs]
                    for ax, key, scale in [(axs[0], 'updates', 1), (axs[1], 'samples_seen', runs[0]['train_samples'])]:
                        x = np.unique([h[key] / scale for history in histories for h in history])
                        error = np.array([np.interp(x, [h[key] / scale for h in history],
                                          np.sqrt([h['validation_metric'] for h in history]),
                                          left=np.nan, right=np.nan) for history in histories])
                        ax.semilogy(x, np.nanmean(error, axis=0), color=PALETTE[i], ls=STYLES[i], label=entry['label'])
                        ax.fill_between(x, np.nanmin(error, axis=0), np.nanmax(error, axis=0), color=PALETTE[i], alpha=.13)
                        ax.set_ylabel('Validation RMSE (m/s)')
                axs[0].set_xlabel('Optimizer updates');axs[1].set_xlabel('Data exposure (epochs)')
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='outside lower center', ncol=2 if len(entries)>4 else 2, fontsize=8)
                return fig
        save(curves, group)
    def rates():
        with plt.rc_context({**figure_style(), 'axes.prop_cycle': DARK_CYCLE}):
            fig, ax = plt.subplots(figsize=(180/25.4, 75/25.4), layout='constrained')
            for i, entry in enumerate(record['groups']['schedule']):
                histories = [record['runs'][f"{entry['id']}_{seed}"]['history'] for seed in record['seeds']]
                epochs = np.unique([h['epoch'] for history in histories for h in history])
                lr = np.array([np.interp(epochs, [h['epoch'] for h in history], [h['lr_last'] for h in history],
                                        left=np.nan, right=np.nan) for history in histories])
                ax.semilogy(epochs, np.nanmean(lr, axis=0), color=PALETTE[i], ls=STYLES[i], label=entry['label'])
                ax.fill_between(epochs, np.nanmin(lr, axis=0), np.nanmax(lr, axis=0), color=PALETTE[i], alpha=.15)
            ax.set(xlabel='Completed epoch', ylabel='Learning rate')
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=2, fontsize=8)
            return fig
    if 'schedule' in record['groups']:
        save(rates, 'rates')
    (output/'training-strategies-results.json').write_text(record_path.read_text())
    config = record['selected_config']
    if config is None:
        return
    prediction = []
    for seed in record['seeds']:
        checkpoint = torch.load(record_path.parent / f"{record['selected_id']}_{seed}.pt", weights_only=True)
        model = ProfileMLP(**{k: config[k] for k in ('normalization', 'residual', 'concatenate', 'dropout', 'activation')})
        model.load_state_dict(checkpoint['model']);model.eval()
        with torch.no_grad():
            pred = model(problem['datasets']['test'].tensors[0])
            prediction.append(problem['y_scaler'].inverse(pred).reshape(2, 128).numpy())
    prediction = np.array(prediction)
    def profiles():
        with plt.rc_context({**figure_style(), 'axes.prop_cycle': DARK_CYCLE}):
            fig, axs = plt.subplots(1, 2, figsize=(180/25.4, 95/25.4), layout='constrained')
            for i, ax in enumerate(axs):
                index = problem['split']['test'][i]
                ax.plot(problem['profiles'][index], .5*problem['eta'], color=PALETTE[0], label='Synthetic reference')
                ax.plot(prediction[:,i].mean(0), .5*problem['eta'], '--', color=PALETTE[1], label='Selected MLP, mean')
                ax.fill_betweenx(.5*problem['eta'], prediction[:,i].min(0), prediction[:,i].max(0), color=PALETTE[1], alpha=.15)
                ax.set(xlabel=r'$u_x$ (m/s)', ylabel='$y$ (m)', title=rf'Test profile {index}: $\bar U={problem["ub"][index]:.3f}$ m/s')
            for ax in axs:
                ax.title.set_fontsize(9)
            fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, fontsize=8)
            return fig
    save(profiles, 'test')
    (output/'training-strategies-results.json').write_text(record_path.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'media')
    args = parser.parse_args()
    torch.set_num_threads(1)
    render(args.record, args.output)
