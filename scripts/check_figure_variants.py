"""Check paired styles, image dimensions, and Matplotlib state cleanup."""
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from figure_utils import save_theme_variants


def main():
    with tempfile.TemporaryDirectory() as directory:
        plt.rcParams['axes.facecolor'] = 'pink'
        original = plt.rcParams.copy()
        keep = plt.figure()
        observations = []

        def draw():
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot([0, 1], [0, 1])
            observations.append((ax.get_facecolor(), ax.xaxis.label.get_color()))
            return fig

        paths = save_theme_variants(draw, Path(directory) / 'plot', dpi=100)
        assert len(observations) == 2
        assert observations[0][1] == 'black' and observations[1][1] == 'white'
        for mode, rgb in [('light', (255, 255, 255)), ('dark', (17, 17, 17))]:
            with Image.open(paths[mode]) as image:
                assert image.size == (400, 300)
                assert image.convert('RGB').getpixel((0, 0)) == rgb
        paths = save_theme_variants(draw, Path(directory) / 'pure.png', dark_background=None)
        with Image.open(paths['dark']) as image:
            assert image.convert('RGB').getpixel((0, 0)) == (0, 0, 0)
        assert all(plt.rcParams[key] == value for key, value in original.items())
        assert plt.get_fignums() == [keep.number]

        def fail():
            plt.figure()
            raise RuntimeError('deliberate drawing failure')

        try:
            save_theme_variants(fail, Path(directory) / 'failure.png')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Drawing failures must propagate')
        assert plt.get_fignums() == [keep.number]
        assert all(plt.rcParams[key] == value for key, value in original.items())
        plt.close(keep)
    print('Verified both styles, matching dimensions, backgrounds, rcParams restoration, and failure cleanup.')


if __name__ == '__main__':
    main()
