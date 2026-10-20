"""Plot validation error versus the number of training profiles."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figure_style import figure_style
from figure_utils import save_theme_variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'media')
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    counts = [len(ids) for ids in record['subsets']]
    values = np.array([[next(r['validation_rmse_m_s'] for r in record['runs']
                            if len(r['training_profiles']) == count and r['seed'] == seed)
                        for seed in record['seeds']] for count in counts])
    def draw():
        with plt.rc_context({**figure_style(), 'axes.prop_cycle': plt.style.library['dark_background']['axes.prop_cycle']}):
            fig, ax = plt.subplots(figsize=(180/25.4, 95/25.4), layout='constrained')
            mean = values.mean(1)
            ax.errorbar(counts, mean, yerr=np.stack([mean-values.min(1), values.max(1)-mean]),
                        fmt='o-', capsize=4, color='C0', label='Mean and range over seeds')
            for j, seed in enumerate(record['seeds']):
                ax.scatter(np.array(counts)+(j-1)*.08, values[:, j], marker=['s','^','d'][j],
                           color=f'C{j+1}', s=23, label=f'Seed {seed}')
            ax.set(xticks=counts, xlabel='Training profiles (128 samples per profile)',
                   ylabel='Best validation RMSE (m/s)', yscale='log')
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=2, fontsize=8)
            return fig
    for suffix in ('.png', '.pdf'):
        save_theme_variants(draw, args.output/f'training-strategies-data-scaling{suffix}', dpi=300)
    (args.output/'training-strategies-data-scaling-results.json').write_text(args.record.read_text())


if __name__ == '__main__':
    main()
