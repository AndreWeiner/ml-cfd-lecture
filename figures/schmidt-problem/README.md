# Lecture 5: the high-Schmidt-number problem

The canonical manuscript is `notebooks/mass_transfer_regression.py`. Its new
opening covers the physical motivation, dimensionless groups, illustrative
water–oxygen calculation, resolution estimate, and transition to regression.
The data-processing/training sections and subsequent practical placeholders
have been removed from the lecture at the author’s request; that material is
intended for a separate exercise script. The lecture now contains theory and
references only.

## Reproduction

From the repository root, starting with an empty output directory:

```bash
mkdir /tmp/schmidt-figures
MPLCONFIGDIR=/tmp/ml-schmidt-mpl .venv-quarto/bin/python \
  scripts/create_schmidt_problem_figure.py --output-dir /tmp/schmidt-figures
```

Copy the named light/dark pairs to `notebooks/media/`. SVG, PDF, and 300 DPI PNG
are exported from the same deterministic geometry. The figure uses the shared
`notebooks/figure_style.py` and `notebooks/figure_utils.py`, 180 mm width and a
120 mm height chosen to keep the labels readable. Text is rendered by LaTeX;
SVG glyphs are paths. An SVG+PNG export of representative labels/equations was
successfully smoke-tested before generation. `--no-tex` is an explicit opt-out,
not an automatic fallback. The opt-out was smoke-tested in a separate empty
output directory; production assets use TeX.

## Meaning of the sketch

This is an original conceptual meridional drawing, not measured data or a CFD
result. The bubble is mildly oblate, with semiaxes 1.30 and 1.04 in arbitrary
schematic units. Exterior curves are generated from sphere potential-flow
streamfunction contours and affinely transformed to the ellipse. This does not
produce a physical spheroid flow solution. Internal ovals merely show the
opposite circulation directions in the two halves of a toroidal circulation:
downward near the outer surface, upward near the axis. Arrows indicate direction,
not velocity magnitude. The bubble center is stationary in the translating frame;
the upward rise arrow is explicitly identified as a laboratory-frame velocity.

The liquid-sided layer widths are exaggerated. The local surface layers thicken
from front to rear. The orange species region continues downstream between the
innermost exterior streamlines, showing dissolved species carried into the wake.
Its outline is the outer envelope of the rear local layer and those streamlines;
this is a conceptual transport region, not a computed concentration contour.
The legend is placed beside the drawing. The
dashed constant-offset contour represents an effective film thickness, not an
arithmetic average of the drawn offsets. Colors are unmodified Matplotlib C0
(flow) and C1 (species) in both themes; neutral strokes and page backgrounds change.

## Scientific references and corrections

- Levich, *Physicochemical Hydrodynamics*, Prentice-Hall (1962). The hydrodynamic
  estimate on p. 441 is cited indirectly through Weiner & Bothe (2017); the book
  page itself was not inspected.
- Weiner & Bothe (2017), *Journal of Computational Physics* 347, 261–289,
  https://doi.org/10.1016/j.jcp.2017.06.040. Provides the clean-bubble scaling context.
- Weiner, Gründing & Bothe (2021), *Chemie Ingenieur Technik* 93, 81–90,
  https://doi.org/10.1002/cite.202000214. The reported resolved Pe=10^4 range is a
  method-specific historical example, not a universal or current upper bound.
- Weiner et al. (2022), *AIChE Journal* 68(7), e17641,
  https://doi.org/10.1002/aic.17641, Section 2.3. The surrogates supply rise speed,
  shape, and surface velocity; the liquid velocity is solved again on the
  single-phase mesh rather than copied wholesale from the two-phase mesh.

The existing slides reversed a thickness ratio. The consistent clean-bubble
film estimate is delta_c/delta_h = sqrt(pi/(2 Sc)), or equivalently
 delta_h/delta_c = sqrt(2 Sc/pi). The slides have not been edited in this step.
For the supplied inputs, an independent arithmetic check gives Re=250, Sc=500,
Pe=125000, delta_h=44.7214 micrometers, delta_c=2.50663 micrometers,
delta_h/delta_c=17.8412 and d_b/delta_c=398.942. The requested planning rule is
at least 10 cells across the layer, giving Delta_n approximately 0.250663
micrometers and about 3989 equivalent cells across the diameter. This is not a
mesh-convergence result.

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

### Wake and layout revision (2026-11-05)

Regenerated all six exports into a fresh empty directory using LaTeX. The
concentration region now extends between the innermost downstream streamlines;
the dashed effective film is unchanged. Canvas height is reduced from 145 to
120 mm, with the legend on the right. The refreshed chapter and scoped site
build passed the repository check (including local links). Both actual browser
themes were visually inspected at 763 px figure width: the matching asset alone
was visible and loaded, with readable labels and the new wake shading.
