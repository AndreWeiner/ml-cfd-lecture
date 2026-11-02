"""Create conceptual Topic 3 SVGs with shared geometry, tab10, and LaTeX.

Reproduce: MPLCONFIGDIR=/tmp/ml-mpl .venv-quarto/bin/python scripts/create_bubble_schematics.py
Native SVG shapes remain editable; text is exported as LaTeX glyph outlines.
Web-layout exception: 1120 x 435 and 900 x 450 logical SVG units preserve the
existing panel geometry, with shared 20/22/24-unit typography and 2.5-unit strokes.
Both are placed at full manuscript width so their text remains readable.
No quantitative data or fitted regime boundaries are represented.
"""
from pathlib import Path
import argparse
import sys
import matplotlib as mpl
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from figure_style import figure_style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'notebooks/media')
parser.add_argument('--no-tex', action='store_true', help='Explicitly disable LaTeX; document the reason when using this option.')
args = parser.parse_args()
mpl.rcParams.update(figure_style(use_tex=not args.no_tex))
# Fixed default palette, unaffected by the dark-background Matplotlib style.
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.colormaps['tab10'].colors)


OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
STYLE = '''<style>text{font-family:Arial,sans-serif;fill:#17324d;font-size:18px}.small{font-size:15px}.title{font-size:20px;font-weight:bold}.line{fill:none;stroke:#1f77b4;stroke-width:2.5}.bubble{fill:#1f77b4;fill-opacity:0.16;stroke:#1f77b4;stroke-width:2.5}</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#1f77b4"/></marker></defs>'''
def text(x,y,value,cls='',anchor='middle'):
    # Keep the native SVG geometry; embed LaTeX glyph outlines for consistent
    # mathematical typography without requiring client-side fonts or TeX.
    value = value.replace('→', r'$\rightarrow$').replace('ρl − ρg', r'$\rho_l-\rho_g$')
    size = {'small': 20, 'title': 24}.get(cls, 22)
    if cls == 'title' and not args.no_tex:
        value = r'\textbf{' + value + '}'
    prop = FontProperties(family='serif', weight='bold' if cls == 'title' else 'normal')
    path = TextPath((0, 0), value, size=size, prop=prop, usetex=not args.no_tex)
    bounds = path.get_extents()
    shift = -(bounds.x0+bounds.x1)/2 if anchor == 'middle' else -bounds.x0
    commands = []
    for points, code in path.iter_segments(curves=True, simplify=False):
        op = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q',
              MplPath.CURVE4: 'C', MplPath.CLOSEPOLY: 'Z'}[code]
        commands.append(op + (' '.join(f'{v:.4f}' for v in points) if op != 'Z' else ''))
    return f'<g transform="translate({x+shift:.4f} {y}) scale(1 -1)"><path fill="#17324d" d="{" ".join(commands)}"/></g>'
def line(x1,y1,x2,y2,arrow=False):
    return f'<path class="line" d="M{x1} {y1}L{x2} {y2}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>'
def bubble(x,y,rx=18,ry=24):
    return f'<ellipse class="bubble" cx="{x}" cy="{y}" rx="{rx}" ry="{ry}"/>'
# Explicit shared palette for both schematics, matching the course dark background.
DARK_COLORS = {
    'white': '#111111', '#17324d': '#e8edf2', '#1f77b4': '#1f77b4',
    '#f4f8fb': '#19242e',
    '#b5c8d5': '#617888', '#ff7f0e': '#ff7f0e',
}
def save(name,w,h,body):
    light = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'viewBox="0 0 {w} {h}" role="img"><rect width="100%" height="100%" '
             'fill="white"/>'+STYLE+body+'</svg>')
    for mode in ('light', 'dark'):
        svg = light
        if mode == 'dark':
            for color, replacement in DARK_COLORS.items():
                svg = svg.replace(color, replacement)
        (OUT / name.replace('.svg', f'-{mode}.svg')).write_text(svg)
# Scale-up geometry is deliberately shared by the two themes. These are
# conceptual examples of possible flow/transport patterns, not measured fields.
# The blue outlines represent bubbles and qualitative wakes; neutral arrows
# indicate motion. The orange shell indicates a concentration boundary layer.
def scale_bubble(x, y, rx, ry, tilt=0, asymmetry=0):
    """An editable, mildly asymmetric oblate bubble (all lengths in SVG units)."""
    return (f'<g transform="translate({x} {y}) rotate({tilt})">'
            f'<path class="bubble" d="M{-rx} 0 '
            f'C{-rx} {-0.63*ry} {-0.45*rx} {-ry} {asymmetry*rx} {-ry} '
            f'C{0.61*rx} {-ry} {rx} {-0.52*ry} {rx} 0 '
            f'C{rx} {0.63*ry} {0.48*rx} {0.88*ry} 0 {0.88*ry} '
            f'C{-0.54*rx} {0.88*ry} {-rx} {0.48*ry} {-rx} 0Z"/></g>')


def motion_arrow(path):
    return (f'<path d="{path}" fill="none" stroke="#17324d" '
            'stroke-width="2.2" marker-end="url(#motion-arrow)"/>')


s = ('<defs><marker id="motion-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
     'markerWidth="5" markerHeight="5" orient="auto">'
     '<path d="M0 0L10 5L0 10z" fill="#17324d"/></marker>'
     '<marker id="wake-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
     'markerWidth="4" markerHeight="4" orient="auto">'
     '<path d="M0 0L10 5L0 10z" fill="#1f77b4"/></marker></defs>')
labels = [
    ('Molecular', 'Molecular motion', 'Diffusion / reactions'),
    ('Micrometers', 'Boundary layers', 'Local transport'),
    ('Millimeters', 'Single bubble', 'Unsteady wake'),
    ('Centimeters', 'Bubble interactions', 'Swarm dynamics'),
    ('Meters', 'Whole reactor', 'Liquid return at sides'),
]
for k, (title, sub, foot) in enumerate(labels):
    x = 20 + 220*k
    cx = x + 100
    s += f'<rect x="{x}" y="20" width="200" height="290" rx="12" fill="#f4f8fb" stroke="#b5c8d5"/>'
    s += text(cx, 53, title, 'title') + text(cx, 82, sub, 'small') + text(cx, 284, foot, 'small')
    if k == 0:
        # Arrows show independent displacement directions. Each starts outside
        # its own molecule and points into free space, never toward a collision
        # partner or a product: no chemical reaction mechanism is implied.
        for dx, dy, col, arrow in [
            (-43, -33, '#ff7f0e', 'M-61 138L-76 123'),
            (34, -20, '#1f77b4', 'M53 158L77 151'),
            (-28, 37, '#1f77b4', 'M-45 222L-62 240'),
            (47, 52, '#ff7f0e', 'M63 237L78 250'),
        ]:
            s += f'<circle cx="{cx+dx}" cy="{175+dy}" r="12" fill="{col}"/>'
            s += f'<g transform="translate({cx} 0)">' + motion_arrow(arrow) + '</g>'
        s += text(cx, 193, 'Random motion', 'small')
    elif k == 1:
        # For a rising bubble, this illustrative layer is thin on the upstream
        # cap and thicker toward the rear. The even-odd cutout leaves gas clear.
        outer = (f'M{cx-57} 166 C{cx-57} 137 {cx-29} 124 {cx} 124 '
                 f'C{cx+29} 124 {cx+57} 137 {cx+57} 166 '
                 f'C{cx+57} 197 {cx+31} 223 {cx} 229 '
                 f'C{cx-31} 223 {cx-57} 197 {cx-57} 166Z')
        hole = (f'M{cx-49} 165 A49 32 0 1 0 {cx+49} 165 '
                f'A49 32 0 1 0 {cx-49} 165Z')
        s += f'<path d="{outer} {hole}" fill="#ff7f0e" fill-opacity="0.16" fill-rule="evenodd"/>'
        s += f'<path d="{outer}" fill="none" stroke="#ff7f0e" stroke-width="2.5" stroke-dasharray="5 4"/>'
        s += bubble(cx, 165, 49, 32)
        s += motion_arrow(f'M{cx+76} 175L{cx+76} 136')
        s += text(cx, 255, 'Concentration layer', 'small')
    elif k == 2:
        s += scale_bubble(cx, 143, 43, 28, -11, 0.08)
        s += motion_arrow(f'M{cx} 108L{cx} 92')
        # Alternating offset vortical structures suggest shedding; these are
        # neither computed streamlines nor a prescribed symmetric steady wake.
        for d in [
            f'M{cx-18} 174 C{cx-52} 182 {cx-48} 207 {cx-24} 205 C{cx-9} 203 {cx-11} 191 {cx-23} 190',
            f'M{cx+15} 196 C{cx+49} 203 {cx+45} 230 {cx+21} 228 C{cx+6} 226 {cx+8} 213 {cx+21} 213',
            f'M{cx-14} 224 C{cx-43} 235 {cx-35} 255 {cx-15} 251 C{cx-4} 249 {cx-7} 239 {cx-16} 240',
        ]:
            s += f'<path class="line" d="{d}" stroke-width="2.2" marker-end="url(#wake-arrow)"/>'
    elif k == 3:
        # Wakes from leading bubbles bend into a shared region and around a
        # following bubble. Draw behind the bubbles to avoid line/interface
        # crossings. Sizes, tilt and placement vary without implying a PDF.
        for d in [
            f'M{cx-54} 144 C{cx-67} 168 {cx-31} 174 {cx-31} 190 C{cx-32} 213 {cx-58} 212 {cx-60} 225',
            f'M{cx-31} 145 C{cx-24} 165 {cx-6} 166 {cx-9} 180',
            f'M{cx+12} 163 C{cx+2} 174 {cx-3} 174 {cx-9} 180',
            f'M{cx+45} 161 C{cx+65} 181 {cx+32} 190 {cx+22} 205 C{cx+19} 219 {cx+17} 239 {cx+34} 251',
            f'M{cx-16} 209 C{cx-23} 226 {cx+5} 229 {cx+4} 245 C{cx+3} 254 {cx-8} 255 {cx-13} 249',
        ]:
            s += f'<path class="line" d="{d}" stroke-dasharray="5 4" opacity="0.72"/>'
        for dx, y, rx, ry, tilt in [
            (-44, 133, 21, 12, -10), (28, 150, 24, 14, 12),
            (-7, 195, 23, 13, -8), (48, 224, 18, 11, 15),
            (-49, 238, 18, 10, -6),
        ]:
            s += scale_bubble(cx+dx, y, rx, ry, tilt, 0.06)
    else:
        s += f'<path d="M{cx-75} 105V242H{cx+75}V105" fill="none" stroke="#17324d" stroke-width="2.5"/>'
        # A widening dispersed plume, not a chain of bubbles. No descending
        # bubbles are drawn: the side arrows refer explicitly to liquid return.
        s += (f'<path d="M{cx-17} 233 Q{cx-44} 180 {cx-45} 112 '
              f'L{cx+43} 112 Q{cx+42} 182 {cx+17} 233Z" '
              'fill="#1f77b4" fill-opacity="0.07"/>')
        for dx, y, rx, ry, tilt in [
            (-7, 227, 5, 3.8, -8), (10, 215, 6, 4, 10),
            (-15, 202, 7, 4.5, -8), (20, 190, 6, 4, 8),
            (-23, 180, 7, 4.5, 7), (13, 166, 7, 4.5, -12),
            (-18, 152, 8, 5, -10), (29, 142, 7, 4.5, 12),
            (-31, 125, 7, 4.5, -8), (7, 120, 8, 5, 7),
        ]:
            s += scale_bubble(cx+dx, y, rx, ry, tilt)
        s += motion_arrow(f'M{cx} 197Q{cx-3} 176 {cx} 155')
        s += motion_arrow(f'M{cx-34} 108Q{cx-65} 120 {cx-58} 178L{cx-55} 216')
        s += motion_arrow(f'M{cx+34} 108Q{cx+65} 120 {cx+58} 178L{cx+55} 216')
        s += text(cx, 262, 'Gas-rich rising plume', 'small')
for k in range(4):
    s += line(125+220*k, 340, 325+220*k, 340, True)
s += text(560, 380, 'Scale-up: resolved information → models for unresolved effects', 'title')
s += text(560, 409, 'Representative scales; conceptual patterns, not quantitative flow fields.', 'small')
save('bubble-multiscale.svg', 1120, 435, s)
# Single-bubble mechanics. Web preset: 900 x 650 logical units, displayed at
# approximately 620 px wide (20/22/24-unit type becomes 13.8/15.2/16.5 px).
# This conceptual cross-section is not a complete free-body diagram or a
# computed flow field. Forces are shown separately from motion and local stress.
# Surface-tension convention: tension acts tangentially on the edges of a
# selected interfacial element; curvature gives its local normal resultant.
# Reference: J. W. M. Bush, MIT 18.357 Interfacial Phenomena, Chapter 2:
# https://thales.mit.edu/bush/wp-content/uploads/2018/02/18.357-LecNotes-2010.pdf
# Uniform surface tension has zero resultant over the entire closed interface.
FORCE_COLORS = {'buoyancy': '#2ca02c', 'viscous': '#d62728',
                'capillary': '#ff7f0e', 'flow': '#1f77b4', 'motion': '#17324d'}

def force_path(d, kind, *, width=3, arrow=False, dash=None, opacity=1):
    return (f'<path d="{d}" fill="none" stroke="{FORCE_COLORS[kind]}" '
            f'stroke-width="{width}" stroke-linecap="round" opacity="{opacity}"'
            + (f' marker-end="url(#force-{kind})"' if arrow else '')
            + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

s = '<defs>'
for kind, color in FORCE_COLORS.items():
    s += (f'<marker id="force-{kind}" viewBox="0 0 10 10" refX="8" refY="5" '
          'markerWidth="5" markerHeight="5" orient="auto">'
          f'<path d="M0 0L10 5L0 10z" fill="{color}"/></marker>')
s += '</defs>'
s += '<title>A rising bubble: forces, interfacial stress, and flow response</title>'
s += ('<desc>An oblate tilted gas bubble rises relative to the liquid. Net buoyancy '
      'points upward and the viscous contribution to drag points downward. '
      'A magnified interface segment shows tangential surface tension. Dashed '
      'curves suggest an unsteady wake; they are not force vectors or computed '
      'streamlines. Arrow lengths are arbitrary.</desc>')
s += text(450, 37, 'A rising bubble: forces, stress, and flow', 'title')
s += text(450, 67, 'Conceptual cross-section; upward motion relative to the liquid', 'small')

# Qualitative wake trails below the rising bubble. Its asymmetric structures
# indicate that inertia can affect the surrounding flow, without identifying
# inertia as an additional force independent of pressure and viscous stresses.
for d in [
    'M367 331 C343 360 357 381 383 385 C409 390 413 414 393 421 C373 429 359 414 371 404',
    'M529 323 C558 355 551 380 526 392 C500 404 507 432 533 434 C553 435 562 417 548 410',
    'M392 445 C361 454 368 485 398 483 C417 481 419 467 407 461',
    'M531 460 C562 474 550 505 526 503 C508 501 507 487 519 483',
]:
    s += force_path(d, 'flow', width=2.7, arrow=True, dash='6 5')
s += scale_bubble(450, 270, 120, 74, -9, 0.08)
s += text(450, 271, 'Gas', 'title')
s += text(450, 300, r'$\rho_g$', 'small')
s += text(708, 314, r'Liquid, $\rho_l$', 'small')

# Net hydrostatic buoyancy minus gas weight, and the viscous part of drag.
# Separate exterior arrows do not imply a location at which a point load acts.
s += force_path('M450 185L450 99', 'buoyancy', width=4, arrow=True)
s += text(480, 118, 'Net buoyancy', 'title', anchor='start')
s += text(480, 149, r'$F_b=(\rho_l-\rho_g)gV$', anchor='start')
s += text(480, 176, 'Hydrostatic force minus gas weight', 'small', anchor='start')
s += force_path('M450 344L450 434', 'viscous', width=4, arrow=True)
s += text(600, 383, 'Viscous resistance', 'title', anchor='start')
s += text(600, 413, 'Opposes relative rise', 'small', anchor='start')
s += force_path('M583 395L469 395', 'motion', width=1.4)

# A dashed neutral kinematic arrow distinguishes velocity from force.
s += force_path('M624 270L624 215', 'motion', width=2.2, arrow=True, dash='5 4')
s += text(650, 239, 'Relative rise', 'small', anchor='start')
s += text(650, 265, r'$U_{\mathrm{rel}}$', anchor='start')

# Orange marks only the selected interface portion. The small open segment
# below it is magnified schematically; the orange arrows pull tangentially
# away from the segment's two ends, not inward toward the bubble centroid.
s += ('<g transform="translate(450 270) rotate(-9)">'
      '<path d="M-120 0 C-120 -46.62 -54 -74 9.6 -74" fill="none" '
      'stroke="#ff7f0e" stroke-width="3.8" stroke-linecap="round"/></g>')
s += text(35, 178, 'Surface tension', 'title', anchor='start')
s += text(35, 208, 'Local interfacial tension;', 'small', anchor='start')
s += text(35, 235, 'resists deformation.', 'small', anchor='start')
s += force_path('M100 343Q147 300 192 330', 'capillary', width=3.5)
s += force_path('M100 343L76 365', 'capillary', width=2.7, arrow=True)
s += force_path('M192 330L220 349', 'capillary', width=2.7, arrow=True)
s += text(76, 329, r'$\sigma$', anchor='start')
s += text(217, 323, r'$\sigma$', anchor='start')
s += force_path('M175 300L335 250', 'motion', width=1.3, dash='3 5')
s += text(148, 393, 'Magnified interface segment', 'small')
s += text(148, 421, r'Uniform $\sigma$: no net force', 'small')
s += text(148, 448, 'on the whole closed interface', 'small')

s += text(600, 484, 'Inertial flow response', 'title', anchor='start')
s += text(600, 514, 'Liquid acceleration, pressure', 'small', anchor='start')
s += text(600, 541, 'redistribution, and wake', 'small', anchor='start')
s += force_path('M582 492L552 483', 'motion', width=1.4)

# Keep pressure drag explicit: the red arrow is the viscous contribution,
# not the total hydrodynamic resistance at arbitrary Reynolds number.
s += text(450, 593, 'Pressure stresses also contribute to hydrodynamic resistance.', 'small')
s += text(450, 624, 'Schematic, not a complete force balance; arrow lengths are not magnitudes.', 'small')
save('bubble-forces.svg', 900, 650, s)
