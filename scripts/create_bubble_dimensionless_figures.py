"""Original conceptual Eotvos/Galilei sketches with editable paired SVG geometry.

Reproduce from repository root into a fresh output directory::

    MPLCONFIGDIR=/tmp/bubble-dimensionless-mpl .venv-quarto/bin/python \
        scripts/create_bubble_dimensionless_figures.py --output-dir /tmp/bubble-dimensionless-new

Source/provenance: original explanatory geometry based on the dimensional
interpretations and radius-based definitions already in
notebooks/bubble_path_classification.py. These are not digitized observations,
solutions, regime boundaries, quantitative silhouettes, or predictive sequences.
The Eotvos comparison holds Ga, property ratios, and initial conditions comparable;
cap-like shapes are possibilities, not an inevitable large-Eo limit. The Galilei
comparison holds Eo, property ratios, and initial conditions comparable; weaker
viscous damping does not guarantee instability or larger path oscillations.

Styling: reuse the guarded methods generator's SVG/LaTeX helpers and course
figure_style. The web-width preset is 700 px, placed at approximately 620 px;
17--22 px text and 2--2.5 px main strokes deliberately prioritize page readability.
Both themes share exact geometry, C0/C1 accents, and text placement. All curved
wake lines carry blue arrowheads; dashed orange tracks denote prior bubble-center
positions, never forces. Neutral rise arrows indicate direction, not force size.
Text uses LaTeX glyph outlines, while shapes remain native editable SVG paths.
No random samples, numerical fields, measurements, or fitted thresholds are used.
"""
from pathlib import Path
import argparse

import matplotlib as mpl
import create_bubble_methods_figures as svg
from create_bubble_methods_figures import text, path, rect, ellipse, arrow, group, save

C0, C1 = svg.C0, svg.C1


def definitions():
    return (f'<defs><marker id="wake-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L9 5L1 9Z" fill="{C0}"/>'
            '</marker>'
            f'<marker id="path-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto"><path d="M1 1L9 5L1 9Z" fill="{C1}"/>'
            '</marker></defs>')


def bubble(x, y, rx, ry, tilt=0):
    return (f'<g transform="rotate({tilt} {x} {y})">'
            + ellipse(x, y, rx, ry, 'BG', C0, 2.5) + '</g>')


def wake(d):
    return path(d, C0, width=2.2, marker_end='url(#wake-arrow)')


def history(d):
    return path(d, C1, width=2.3, stroke_dasharray='6 5', marker_end='url(#path-arrow)')


def eotvos():
    """Compact qualitative Eo comparison without numerical regime boundaries.

    Labels mean much less than one, order one, and much greater than one using
    the lecture's radius-based convention. They are approximate scale ranges,
    not universal shape classifications: the unchanged silhouettes remain
    illustrative possibilities at comparable Ga, ratios and initial conditions.
    The mathematical O(1) label deliberately avoids an unsupported numeric
    interval. The title, formula and qualifications remain in the lecture prose
    and generator documentation rather than in this simplified artwork.
    """
    b = definitions()
    b += arrow(57, 41, 642, 41, 'FG', 1.8)
    b += text(350, 28, r'Increasing $Eo$', 20, 'middle')
    for x in (16, 244, 472):
        b += rect(x, 62, 212, 144, 'PANEL', 'BORDER', 1, rx=9)
    for x, label in [(122, r'$Eo\ll1$'), (350, r'$Eo=\mathcal{O}(1)$'),
                     (578, r'$Eo\gg1$')]:
        b += text(x, 90, label, 21, 'middle')
    # Preserve every silhouette coordinate; translate as one group to compact
    # the removed title/definition/footer space without rescaling the geometry.
    shapes = bubble(122, 234, 36, 36)
    shapes += bubble(350, 234, 54, 27)
    shapes += path('M518 252C520 215 544 193 578 193C612 193 636 215 638 252'
                   'Q578 235 518 252Z', C0, 'BG', 2.5)
    b += group(0, -90, shapes)
    return b, 220


def galileo():
    # Retained panel/legend geometry is translated together after removing the
    # title, formula and footer; no physical silhouettes or paths are rescaled.
    b = ''
    b += arrow(57, 113, 642, 113, 'FG', 1.8)
    b += text(350, 101, r'Increasing $Ga$: less relative viscous damping', 19, 'middle')
    b += rect(16, 136, 246, 304, 'PANEL', 'BORDER', 1, rx=9)
    b += rect(278, 136, 406, 304, 'PANEL', 'BORDER', 1, rx=9)
    b += text(139, 165, 'Damped example', 20, 'middle', bold=True)
    b += text(481, 165, 'Possible inertial response', 20, 'middle', bold=True)
    # Weak steady near-wake example and straight bubble-center history, separated.
    b += bubble(147, 231, 39, 27)
    b += wake('M128 260C116 284 121 315 131 333')
    b += wake('M166 260C178 284 173 315 163 333')
    b += history('M70 338L70 210')
    b += arrow(215, 238, 215, 198, 'FG', 1.8)
    b += text(215, 190, 'Rise', 15.5, 'middle')
    b += text(139, 377, 'Weak, steady wake', 18, 'middle')
    b += text(139, 406, 'Straight rise', 18, 'middle')
    # Illustrative alternating vortical wake; every curved blue segment has an arrowhead.
    b += bubble(387, 221, 46, 26, -12)
    b += wake('M367 252C331 261 336 292 359 289C377 287 376 272 364 274')
    b += wake('M406 278C441 287 434 318 411 313C395 310 399 295 410 298')
    b += wake('M374 307C340 319 346 347 368 342C383 339 379 326 370 329')
    # A separate projected trajectory inset, not a streamline through the wake.
    b += path('M552 195V344', 'GRID', width=1, stroke_dasharray='3 5')
    b += history('M552 344C507 320 510 294 552 281C594 268 592 243 552 230'
                 'C523 221 522 207 543 195')
    b += text(552, 365, 'Example path history', 16, 'middle')
    b += text(386, 377, 'Asymmetric wake', 18, 'middle')
    b += text(481, 406, 'Unsteady wake; zigzag or spiral rise may occur.', 17, 'middle')
    # Legend explicitly separates local fluid-motion arrows and path history.
    b += wake('M27 475Q44 458 62 475')
    b += text(77, 479, 'Wake motion (bubble frame)', 16)
    b += history('M337 475L375 475')
    b += text(390, 479, 'Prior bubble-center positions', 16)
    return definitions() + group(0, -72, b), 429


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'notebooks/media')
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    parser.add_argument('--no-tex-reason')
    args = parser.parse_args()
    use_tex = resolve_renderer("mathtext" if args.no_tex else args.renderer)
    if args.no_tex and not args.no_tex_reason:
        parser.error('--no-tex requires --no-tex-reason')
    svg.USE_TEX = use_tex
    svg.glyphs.cache_clear()
    mpl.rcParams.update(svg.figure_style(use_tex=svg.USE_TEX))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=svg.PALETTE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = text(5, 28, r'$Ga=\sqrt{gR^3}/\nu_l$, $Eo=\rho_l gR^2/\sigma$', 18)
    save(args.output_dir, '_dimensionless-smoke', probe, 40, 'Representative equation export test')
    for p in args.output_dir.glob('_dimensionless-smoke-*.svg'):
        p.unlink()
    for stem, draw in [('bubble-eotvos-trend', eotvos), ('bubble-galileo-trend', galileo)]:
        body, height = draw()
        save(args.output_dir, stem, body, height, draw.__name__ +
             ': original conceptual schematic, not numerical data or a universal regime sequence.')
    print(f'Created four SVGs in {args.output_dir}; LaTeX={svg.USE_TEX}')


if __name__ == '__main__':
    main()
