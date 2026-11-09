# Conceptual representations of a gas–liquid interface

The editable generator is `scripts/create_interface_model_figures.py`. It uses
the shared `notebooks/figure_style.py` and `notebooks/figure_utils.py` settings.
These are original conceptual sketches, not CFD results, measured geometries,
or reproductions of figures from the references below.

## Reproduction

From the repository root:

```bash
MPLCONFIGDIR=/tmp/ml-interface-mpl .venv-quarto/bin/python \
  scripts/create_interface_model_figures.py --output-dir notebooks/media
```

For an independent reproduction, a new, initially empty directory can replace
`notebooks/media`. NumPy and Matplotlib are the only Python dependencies beyond
the standard library and the course's shared helpers. The default renderer is
LaTeX (`text.usetex=True`), with a minimal empty preamble. An actual representative
SVG and PNG smoke export runs before the complete set; an export failure is not
caught or silently converted to MathText. The optional `--no-tex` explicitly
selects MathText for environments where that is intended. It was tested separately
but was **not** used for the delivered assets. TeX glyphs are paths in SVG;
the generator retains their editable source strings.

Each of these stems has `-light` and `-dark` versions in SVG, PDF, and 300 DPI PNG:

| Stem | Physical size |
| --- | --- |
| `sharp-interface` | 180 × 111 mm |
| `vof-interface` | 180 × 133 mm |
| `fitted-interface` | 180 × 117 mm |

All use the shared 180 mm width preset, 11 pt main text, and 10 pt ancillary
labels. Different heights accommodate the geometry and the VOF definition.
Light backgrounds are white; dark backgrounds are `#111111`. Category colors
retain the standard C0–C9 palette in both themes. Scalar fractions use the same
perceptually uniform sequential `viridis` map with limits [0, 1]. All geometry,
positions, dimensions, fractions, and reconstruction segments are shared between
the themes. Diagram coordinates are arbitrary sketch units, with equal spatial
aspect ratio and no physical scale claim.

## Geometry and interpretation

All three sketches start from the same mildly oblate ellipse, with semiaxes
`A = 1.38` and `B = 1.04`. The gas is the interior `Omega^-(t)` and the liquid is
the exterior `Omega^+(t)`. The interface normal is the normalized **positive**
ellipse gradient: it starts on the interface and points from gas into liquid, matching the manuscript's
sign convention.

The sharp-interface sketch identifies separate bulk balances and coupling
conditions. A sharp interface is geometrically of zero thickness. A massless
interface is an additional modeling assumption; the drawing does not claim
that sharpness implies zero interfacial mass or absence of interfacial transport.

The VOF sketch uses a 12 × 9 square-cell grid with spacing 0.35. Its dimensionless
gas volume fraction is the cell-average gas indicator. The 2D occupied area is
interpreted as a unit-depth volume. An 8192-edge polygon samples the analytic
ellipse and is clipped against each cell to calculate the gas fraction; colors
are **not** point samples or arbitrary mixed-cell values. The interface passes
through 26 mixed cells. For each mixed cell, an analytic ellipse-gradient
direction evaluated at the cell center supplies a local plane normal. Bisection
positions a straight line so that its gas-side polygon occupies exactly the
computed gas fraction. Only this cell's line segment is drawn. Adjacent segments
are not forced to meet, as local PLIC reconstruction need not be globally
continuous. The known analytic direction is an illustrative input, not a claim
that a VOF solver recovers the exact normal. No transport step or flow solution
is computed. The dark core with a white keyline makes segments visible over the
full scalar color range; the keyline has no physical width interpretation.

The fitted-mesh sketch draws 24 angular sectors and five curvilinear
quadrilateral bands. Two bands are inside the interface and three are outside.
The band edges have ellipse scale factors 0.61, 0.80, 1.00, 1.20, 1.42, and 1.64;
both domains share the **same** scale-1 ellipse and its interface vertices.
The gas interior and far liquid mesh are omitted. Curved faces deliberately
emphasize exact fitting to the conceptual ellipse; this is not a claim about
the face order or connectivity of a particular finite-volume implementation.
The upward arrow illustrates interface motion and a following mesh, with no
quantitative velocity magnitude or computed mesh-deformation field implied.
The separate normal arrow starts on the interface and follows the exact ellipse
gradient into the liquid. The sharp-interface sketch uses the page background
without a tinted rectangular region or phase fills.

## Source context

The manuscript supplies the governing-equation definitions and discussion.
Relevant primary references are:

- [Bothe (2022), sharp-interface continuum modeling](https://doi.org/10.1016/j.ijengsci.2022.103731).
- [Roenby et al. (2016), geometric volume-of-fluid representation and advection](https://doi.org/10.1098/rsos.160405).
- [Tukovic and Jasak (2012), moving-mesh interface tracking](https://doi.org/10.1016/j.compfluid.2011.11.003).

## Checks performed

The complete LaTeX set was generated in initially empty temporary directories,
then copied into `notebooks/media`. Both theme variants of all three exported
PNGs were visually inspected. A fitted-mesh leader initially crossed the normal
label; it was moved and the resulting exports were inspected again. The VOF
labels explicitly identify pure liquid (`alpha_g=0`) and gas (`alpha_g=1`).

The geometric checks give:

- Minimum and maximum cell fraction: 0 and 1.
- Number of mixed cells and PLIC segments: 26.
- Largest reconstructed-fraction discrepancy: `1.89e-15`.
- Relative total ellipse-area error from polygon sampling: `9.80e-8`.

These checks establish the consistency of the illustration's geometric encoding,
not the accuracy of a flow solver, advection scheme, or physical model. Actual
Quarto-page theme switching, asset paths, source downloads, and apparent label
size after placement are checked as part of the manuscript integration; the
standalone generator does not perform those browser checks.

## Integrated page QA (2026-11-05)

The affected chapter was explicitly rendered to refresh its frozen result, then
a scoped site build and local-link check passed. All six figure pairs loaded
from their actual relative page paths. Browser theme toggling showed exactly
one matching asset per pair (six visible, six hidden), at equal widths of
675 px in the narrow preview and about 763 px in the wider preview. Both actual
themes were inspected, including the workflow and coordinate sketches and
rendered model equations; no MathJax error elements were present. The final
dark-page background was verified as rgb(17,17,17). All source numerical code
cells remain unchanged and were not executed during this writing-stage build.

## Outward-normal and workflow revision (2026-11-05)

The normal convention is now gas-to-liquid throughout the revised manuscript
and sketches. The sharp-interface phase tints have been removed, and the
normal arrows start exactly on the interface. The manuscript defines the jump
as liquid minus gas, giving positive spherical curvature and the usual Laplace
pressure difference. The analytic tangent/normal dot product is zero; the radial
projection of the outward normal is +0.9785321866 at the illustrated point.

All revised exports were reproduced with LaTeX in empty output directories.
Both actual page themes were visually checked at about 763 px figure width,
including the sharp interface, fitted mesh, angular profiles, transport wake,
and coordinate inset. All six matching theme assets loaded, exactly one per
pair was visible, and no MathJax errors occurred. Figure URLs include a revision
query to prevent stale browser image caches. The refreshed frozen result and
scoped site build passed the local-link check. The lecture now contains theory
and references only: practical code and subsequent placeholders were removed
at the author's request; no code blocks remain in the rendered lecture.
