"""Generate editable, paired conceptual numerical / experimental method diagrams.

Reproduce from the repository root (LaTeX is the default and was export-tested)::

    MPLCONFIGDIR=/tmp/bubble-methods-mpl .venv-quarto/bin/python \
        scripts/create_bubble_methods_figures.py --output-dir /tmp/bubble-methods-new

Copy the four SVGs to notebooks/media and docs/notebooks/media after inspection.
The native SVG geometry is editable; LaTeX text is stored as glyph outlines.
Shared web-width preset: 700 logical pixels (placed at approximately 620 px),
17--22 logical-pixel text, 1.7--2.2 px main strokes. This deliberate web-layout
exception to the 180 mm preset preserves readable multi-panel explanations.
These original schematics illustrate standard method distinctions; geometry,
particle positions, arrows and scalar fields are illustrative, not measured or
computed data. No quantitative resolution/cost ordering or accuracy claim is made.
Euler--Lagrange markers are smaller than cells; they are not resolved interfaces.
Euler--Euler uses averaged gas fraction, with no individual bubble interfaces.
PIV shows seeded-liquid velocity in an illuminated plane; LIF shows concentration
in a calibrated fluorescence measurement. Tomography depicts X-ray projections
around a reactor section and a reconstructed gas distribution, not single-view
imaging. Scalar fields use a fixed sequential viridis map in both themes.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import html
import math
import sys

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from figure_style import figure_style

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
C0, C1, C2, C3, C4 = PALETTE[:5]
WIDTH = 700
USE_TEX = True


def path(d, stroke='FG', fill='none', width=2, **kwargs):
    attrs = ' '.join(f'{key.replace("_", "-")}="{value}"' for key, value in kwargs.items())
    return f'<path d="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{width}" {attrs}/>'


def rect(x, y, w, h, fill='none', stroke='FG', width=1.5, **kwargs):
    attrs = ' '.join(f'{key.replace("_", "-")}="{value}"' for key, value in kwargs.items())
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" {attrs}/>'


def ellipse(x, y, rx, ry, fill='none', stroke=C0, width=2, **kwargs):
    attrs = ' '.join(f'{key.replace("_", "-")}="{value}"' for key, value in kwargs.items())
    return f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" {attrs}/>'


@lru_cache(maxsize=None)
def glyphs(value, size, bold):
    if bold and USE_TEX:
        value = r'\textbf{' + value + '}'
    p = TextPath((0, 0), value, size=size,
                 prop=FontProperties(family='serif', weight='bold' if bold else 'normal'),
                 usetex=USE_TEX)
    bounds = p.get_extents()
    commands = []
    for points, code in p.iter_segments(curves=True, simplify=False):
        op = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q',
              MplPath.CURVE4: 'C', MplPath.CLOSEPOLY: 'Z'}[code]
        commands.append(op + (' '.join(f'{v:.4f}' for v in points) if op != 'Z' else ''))
    return ' '.join(commands), bounds.x0, bounds.x1


def text(x, y, value, size=18, anchor='start', bold=False, color='FG'):
    d, left, right = glyphs(value, size, bold)
    shift = -(left + right) / 2 if anchor == 'middle' else -left
    return (f'<g aria-label="{html.escape(value, quote=True)}" '
            f'transform="translate({x+shift:.4f} {y}) scale(1 -1)">'
            f'<path fill="{color}" d="{d}"/></g>')


def arrow(x1, y1, x2, y2, color='FG', width=2):
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    ux, uy = dx/length, dy/length
    s = 6
    return (path(f'M{x1} {y1}L{x2} {y2}', color, width=width)
            + path(f'M{x2-s*ux+s*.45*uy} {y2-s*uy-s*.45*ux}'
                   f'L{x2} {y2}L{x2-s*ux-s*.45*uy} {y2-s*uy+s*.45*ux}', color, width=width))


def group(x, y, body):
    return f'<g transform="translate({x} {y})">{body}</g>'


def grid(x, y, w, h, step):
    b = rect(x, y, w, h, 'PANEL', 'GRID', 1)
    for xx in range(x+step, x+w, step):
        b += path(f'M{xx} {y}V{y+h}', 'GRID', width=.7)
    for yy in range(y+step, y+h, step):
        b += path(f'M{x} {yy}H{x+w}', 'GRID', width=.7)
    return b


def field(x, y, cols, rows, cell, kind='plume'):
    b = ''
    for j in range(rows):
        for i in range(cols):
            xx, yy = (i+.5)/cols, (j+.5)/rows
            if kind == 'plume':
                v = .15 + .72*math.exp(-((xx-.5)/(.18+.25*(1-yy)))**2)*(.7+.3*yy)
            else:
                v = .12 + .75*math.exp(-((xx-.53)/.24)**2-((yy-.5)/.45)**2)
            color = mpl.colors.to_hex(mpl.colormaps['viridis'](v))
            b += rect(x+i*cell, y+j*cell, cell+.15, cell+.15, color, 'none', 0)
    return b


def numerical():
    """Three slide-friendly panels at the shared 700 px web-width preset.

    The 420 px height replaces the previous stacked layout without shrinking
    text below 16 px. These conceptual arrays are not CFD solutions. Orange
    arrows are bubble velocity directions; dashed orange lines are path history.
    Curved blue arrows indicate local wake motion. No carrier-liquid velocity
    vectors or force vectors are drawn in the Euler--Lagrange panel.
    The Euler--Euler field is a synthetic widening gas-rich plume, with no
    individual interfaces; the sequential color key has no quantitative values.
    """
    b = ('<defs><marker id="numerical-wake-arrow" viewBox="0 0 10 10" '
         'refX="8.5" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
         f'<path d="M1 1L9 5L1 9Z" fill="{C0}"/></marker></defs>')
    b += text(24, 30, 'Three ways to represent a bubbly flow', 22, bold=True)
    for x in (16, 244, 472):
        b += rect(x, 52, 212, 290, 'PANEL', 'BORDER', 1, rx=9)
    for x, title in [(122, 'Interface-resolved'), (350, 'Euler--Lagrange'), (578, 'Euler--Euler')]:
        b += text(x, 80, title, 20, anchor='middle', bold=True)

    # One oblate bubble. Its velocity starts at its center and follows its
    # vertical minor axis; wake curves are below the rear of the rising bubble.
    a = grid(0, 0, 180, 180, 10)
    a += ellipse(90, 54, 39, 26, 'BG', C0, 2.7)
    a += arrow(90, 54, 90, 17, C1, 2.4)
    for d in [
        'M71 84C40 93 42 123 67 120C84 118 81 102 69 105',
        'M106 112C139 120 137 151 112 147C95 144 99 129 110 132',
    ]:
        a += path(d, C0, width=2.3, marker_end='url(#numerical-wake-arrow)')
    b += group(32, 98, a)
    b += text(122, 305, 'Resolved interface + wake', 17, anchor='middle')
    b += text(122, 329, 'Fine computational grid', 17, anchor='middle')

    # Euler--Lagrange: bubble centers smaller than the 30 px cells. For each
    # quadratic path the endpoint derivative is exactly parallel to (dx, dy).
    # Bubble velocity starts at the current center, with varied rise directions.
    a = grid(0, 0, 180, 180, 30)
    particles = [
        (21, 133, 9, -17), (61, 105, -9, -15),
        (102, 142, 13, -11), (151, 103, -6, -18),
        (40, 49, -12, -12), (111, 38, 8, -17),
    ]
    for x, y, dx, dy in particles:
        a += path(f'M{x-1.5*dx-5} {y-1.5*dy}'
                  f'Q{x-.55*dx} {y-.55*dy} {x} {y}', C1,
                  width=1.7, stroke_dasharray='3 3')
        a += arrow(x, y, x+dx, y+dy, C1, 1.8)
        a += ellipse(x, y, 3.3, 3.3, C1, C1, .8)
    b += group(260, 98, a)
    b += text(350, 305, 'Tracked bubble centers', 17, anchor='middle')
    b += text(350, 329, 'Subgrid effects modeled', 17, anchor='middle')

    # Euler--Euler: smooth synthetic concentration widening upward, sampled
    # on an editable vector grid. A narrow base and low-gas outer region make
    # the plume explicit without depicting separate individual bubbles.
    a = ''
    for j in range(30):
        for i in range(30):
            xx = (i+.5)/30-.5
            yy = (j+.5)/30  # y increases downward in SVG
            plume_width = .075 + .255*(1-yy)
            v = .045 + (.91-.12*(1-yy))*math.exp(-(xx/plume_width)**4)
            col = mpl.colors.to_hex(mpl.colormaps['viridis'](v))
            a += rect(i*6, j*6, 6.1, 6.1, col, 'none', 0)
    for q in range(0, 181, 30):
        a += path(f'M{q} 0V180M0 {q}H180', 'GRID', width=.55)
    a += rect(0, 0, 180, 180, 'none', 'GRID', .8)
    # Fixed neutral shaft on the bright plume core indicates upward gas motion;
    # category colors remain unchanged and the scalar map remains quantitative-free.
    a += arrow(90, 132, 90, 70, '#252525', 2.1)
    b += group(488, 98, a)
    b += text(578, 305, r'Gas volume fraction $\alpha_g$', 17, anchor='middle')
    b += text(578, 329, 'Interpenetrating phases', 17, anchor='middle')

    # Short keys sit directly below their corresponding panels.
    b += arrow(27, 365, 51, 365, C1, 1.8)
    b += text(61, 370, 'Bubble velocity', 16)
    b += path('M255 369Q264 355 279 365', C0, width=1.8,
              marker_end='url(#numerical-wake-arrow)')
    b += text(289, 370, 'Wake motion', 16)
    b += text(483, 370, 'low', 15)
    for i in range(75):
        b += rect(516+i*1.6, 359, 1.7, 10,
                  mpl.colors.to_hex(mpl.colormaps['viridis'](i/74)), 'none', 0)
    b += text(645, 370, 'high', 15)
    b += text(24, 405, 'Conceptual; dashed lines show paths. Arrows indicate motion, not forces.', 16.5)
    return b, 420


def camera(x, y, direction='right'):
    b = rect(0, 0, 34, 25, 'BG', 'FG', 1.8)
    b += ellipse(12, 12, 5, 5, 'none', 'FG', 1.3)
    b += path('M34 6L46 1V24L34 19Z', 'FG', 'BG', 1.8)
    b += path('M14 25V35M7 35H24', 'FG', width=1.5)
    if direction == 'left':
        b = f'<g transform="translate(46 0) scale(-1 1)">{b}</g>'
    elif direction == 'up':
        b = f'<g transform="rotate(-90 23 12)">{b}</g>'
    return group(x, y, b)


def tank(x, y, w=117, h=126):
    return (rect(x, y, w, h, 'none', 'FG', 1.8)
            + path(f'M{x+4} {y+12}H{x+w-4}', C0, width=1))


def experimental():
    b = text(24, 30, 'Experimental approaches: what becomes observable?', 22, bold=True)
    b += text(24, 56, 'An apparatus and its reconstructed or recorded quantity.', 18)
    for x, y in [(16, 76), (358, 76), (16, 418), (358, 418)]:
        b += rect(x, y, 326, 325, 'PANEL', 'BORDER', 1, rx=9)
    # Camera and backlight with successive observed outlines.
    a = text(15, 30, 'High-speed imaging', 21, bold=True)
    a += tank(112, 68, 113, 139)
    a += rect(248, 72, 15, 131, C1, C1, 1, fill_opacity=.12)
    for yy in (88, 135, 182):
        a += arrow(246, yy, 229, yy, C1, 1.4)
    a += camera(37, 119)
    a += path('M83 120L111 86M83 141L111 184', 'GRID', width=1, stroke_dasharray='4 3')
    for xx, yy, rr, tilt in [(158, 181, 12, 0), (179, 141, 16, -12), (156, 100, 19, 12)]:
        a += f'<g transform="rotate({tilt} {xx} {yy})">' + ellipse(xx, yy, rr, 9, 'none', C0, 2) + '</g>'
    a += path('M158 181Q191 149 169 124T156 100', C0, width=1.3, stroke_dasharray='4 4')
    a += text(58, 231, 'Camera', 16, anchor='middle')
    a += text(256, 231, 'Backlight', 16, anchor='middle')
    a += text(15, 278, 'Records shape and projected path.', 18)
    a += text(15, 305, 'Outlines shown at successive times.', 16)
    b += group(16, 76, a)
    # PIV: laser sheet, tracers and spatial velocity samples.
    a = text(15, 30, 'Particle image velocimetry', 20, bold=True)
    a += tank(109, 69, 151, 139)
    a += rect(112, 82, 145, 122, C2, 'none', 0, fill_opacity=.10)
    a += rect(19, 120, 43, 25, 'BG', C2, 1.8)
    a += path('M63 127L108 83L108 202L63 138Z', C2, C2, 1, fill_opacity=.08)
    for j in range(4):
        for i in range(4):
            x, y = 127+i*32, 100+j*29
            a += ellipse(x, y, 1.8, 1.8, 'FG', 'none', 0)
            a += arrow(x+3, y-3, x+10+3*math.sin(j), y-17, C0, 1.5)
    a += camera(170, 222, direction='up')
    a += text(42, 166, 'Laser', 16, anchor='middle')
    a += text(15, 278, 'Measures seeded-liquid velocity.', 18)
    a += text(15, 305, 'Tracer displacement in a light sheet.', 16)
    b += group(358, 76, a)
    # LIF: fluorescent scalar map in a measurement plane.
    a = text(15, 30, 'Laser-induced fluorescence', 20, bold=True)
    a += tank(109, 69, 151, 139)
    a += field(116, 83, 10, 8, 13.7, 'concentration')
    a += ellipse(184, 115, 20, 12, 'BG', C0, 2)
    a += rect(19, 120, 43, 25, 'BG', C2, 1.8)
    a += path('M63 127L108 83L108 202L63 138Z', C2, C2, 1, fill_opacity=.08)
    a += camera(170, 222, direction='up')
    a += text(42, 166, 'Laser', 16, anchor='middle')
    a += text(15, 278, 'Infers concentration from fluorescence.', 17)
    a += text(15, 305, 'Calibration links intensity to concentration.', 15.5)
    b += group(16, 418, a)
    # X-ray tomography, side projection plus rotation, reconstructed section.
    a = text(15, 30, 'X-ray tomography', 21, bold=True)
    a += ellipse(125, 138, 53, 53, 'BG', 'FG', 1.8)
    a += ellipse(125, 138, 37, 37, 'none', 'GRID', 1)
    a += path('M28 130L39 138L28 146Z', C4, C4, 1)
    a += rect(208, 89, 10, 98, C4, C4, 1, fill_opacity=.12)
    for y in (98, 118, 138, 158, 178):
        a += path(f'M40 138L207 {y}', C4, width=.85, opacity=.6)
    a += path('M83 86A69 69 0 0 1 185 97', 'FG', width=1.7)
    a += arrow(176, 87, 185, 97, 'FG', 1.7)
    a += text(33, 214, 'Source', 15, anchor='middle')
    a += text(212, 214, 'Detector', 15, anchor='middle')
    a += text(125, 238, 'Views around a reactor section', 16, anchor='middle')
    a += field(247, 106, 5, 5, 11, 'plume')
    a += arrow(222, 136, 241, 136, 'FG', 1.5)
    a += text(15, 278, 'Reconstructs spatial gas distribution.', 17)
    a += text(15, 305, 'Many projections form a cross-section.', 16)
    b += group(358, 418, a)
    b += text(24, 777, 'Conceptual apparatus and signals; fields and arrows are illustrative.', 17)
    b += text(24, 801, 'Imaging / PIV / LIF need optical access; layouts are simplified.', 16)
    return b, 821


def save(out, stem, body, height, description):
    for theme in ('light', 'dark'):
        colors = ({'BG': '#ffffff', 'FG': '#252525', 'PANEL': '#ffffff',
                   'GRID': '#c4c4c4', 'BORDER': '#cecece'} if theme == 'light'
                  else {'BG': '#111111', 'FG': '#e8e8e8', 'PANEL': '#111111',
                        'GRID': '#565656', 'BORDER': '#606060'})
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
               f'viewBox="0 0 {WIDTH} {height}" role="img">'
               f'<title>{html.escape(stem.replace("-", " "))}</title>'
               f'<desc>{html.escape(description)}</desc>'
               f'<rect width="{WIDTH}" height="{height}" fill="BG"/>{body}</svg>')
        for token, color in colors.items():
            svg = svg.replace(f'"{token}"', f'"{color}"')
        (out / f'{stem}-{theme}.svg').write_text(svg)


def main():
    global USE_TEX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'notebooks/media')
    from figure_style import add_renderer_argument, resolve_renderer
    add_renderer_argument(parser, legacy_no_tex=True)
    parser.add_argument('--no-tex-reason', help='Required record for explicit non-LaTeX export.')
    args = parser.parse_args()
    use_tex = resolve_renderer("mathtext" if args.no_tex else args.renderer)
    if args.no_tex and not args.no_tex_reason:
        parser.error('--no-tex requires --no-tex-reason; no silent rendering fallback is permitted.')
    USE_TEX = use_tex
    mpl.rcParams.update(figure_style(use_tex=USE_TEX))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Actual TeX path-generation smoke test, including representative math.
    probe = text(8, 28, r'Gas fraction $\alpha_g$; velocity $\mathbf{u}_l$', 18)
    save(args.output_dir, '_renderer-smoke', probe, 40, 'LaTeX renderer smoke test')
    for p in args.output_dir.glob('_renderer-smoke-*.svg'):
        p.unlink()
    body, height = numerical()
    save(args.output_dir, 'bubble-numerical-methods', body, height,
         'Conceptual interface-resolved, Euler-Lagrange and Euler-Euler representations; no numerical data.')
    body, height = experimental()
    save(args.output_dir, 'bubble-experimental-methods', body, height,
         'Conceptual high-speed imaging, PIV, LIF and X-ray tomography apparatus and observables; no measured data.')
    print(f'Wrote four paired SVGs to {args.output_dir}; renderer: ' +
          ('LaTeX glyph outlines' if USE_TEX else f'MathText ({args.no_tex_reason or args.renderer})'))


if __name__ == '__main__':
    main()
