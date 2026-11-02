# Fresh digitization of Tripathi et al. (2015), Figure 1

Source: M. K. Tripathi, K. C. Sahu and R. Govindarajan,
*Dynamics of an initially spherical bubble rising in quiescent liquid*,
Nature Communications **6**, 6268 (2015),
[doi:10.1038/ncomms7268](https://doi.org/10.1038/ncomms7268).
The [publisher PDF](https://www.nature.com/articles/ncomms7268.pdf), page 3,
was downloaded and inspected on 2026-10-28.

These are **approximate coordinates freshly read from the published figure**,
not original simulation output. A bounded search did not
identify downloadable original numerical coordinates; see
[the source-search record](../TRIPATHI_SOURCE_SEARCH.md).

## Source and calibration

`figure1-source.png` is the unaltered 517-by-364-pixel raster extracted from the
publisher PDF with Poppler:

```bash
pdfimages -f 3 -l 3 -png tripathi2015.pdf source
# source-000.png is Figure 1's plot area; source-001.png is Figure 2.
```

The PDF stores the plot as a JPEG image, so the markers are not available as
vector coordinates. Axis labels and Roman numerals are separate PDF text;
the labeled axes and caption were checked on the complete PDF page.
The source plot image is retained solely for attribution, reproducibility and
inspection; it is not the newly styled lecture figure. Credit for the original
image belongs to Tripathi, Sahu and Govindarajan / Nature Communications.

`scripts/extract_tripathi_regime_points.py` thresholds dark pixels, finds
connected components, and applies a visually checked selection and class list.
The exact threshold and component filters are in the script. Boundary-connected,
touching, and partly occluded markers have explicit split/center records rather
than undocumented changes. Circles and squares use bounding-box centers;
upright triangles use their geometric centroid. `points.csv` retains every
selected pixel position and its measurement note. `audit.json` retains the
candidate components, calibration coefficients and unresolved location.

Pixel coordinates have origin at the upper-left pixel center. The horizontal
calibration uses the labeled Ga=10 and Ga=100 ticks at x=73 and x=313.
The vertical calibration is a least-squares line through log10(Eo) at four
labeled tick positions: (y,Eo)=(64,100), (155,10), (245,1), (335,0.1).
The transformations are linear in log10(Ga) and log10(Eo), not in Ga and Eo.
No extracted value is snapped to a presumed simulation grid.

One pixel corresponds to approximately 0.96% in Ga and 2.6% in Eo. This is
pixel-scale sensitivity, **not a measured confidence interval**. JPEG artifacts,
finite marker size, calibration, and partial occlusion add uncertainty; decimal
places retained in the CSV do not establish equivalent physical precision.

## Regime names and the source's internal inconsistency

The Roman numerals **inside the original Figure 1 image** label the low-Ga,
high-Eo green area III and the high-Ga, low-Eo blue area II. Its **caption and
main Results discussion** instead call the green skirted/non-oscillatory region
II and the blue oscillatory region III. Parts of the source's Methods also use
the conflicting numbering. This reproduction explicitly follows the caption
and main Results convention, using descriptive physical names:

| Course regime | Physical behavior | Original image area | Points |
|---|---|---|---:|
| I | Axisymmetric | Pink, circles | 20 |
| II | Skirted, non-oscillatory path | Green, triangles; printed III | 12 |
| III | Oscillatory path | Blue, triangles; printed II | 27 |
| IV | Peripheral breakup | Pale yellow, squares | 16 |
| V | Central breakup | Darker yellow, squares | 29 |

The source uses radius R: Ga = rho_l sqrt(g R^3)/mu_l and
Eo = rho_l g R^2/sigma, with rho_g/rho_l=0.001 and mu_g/mu_l=0.01.
Both axes are dimensionless. These conditions and the initially spherical,
quiescent-release setup limit the map's applicability.

## Completeness and validation

The extraction retains **104 visible or recoverable markers**, not all the
approximately 130 simulations mentioned by the paper. At pixel (240.5,127.5),
approximately (Ga,Eo)=(50,20), a bubble illustration and two lines obscure a
possible marker. Its identity and position are not sufficiently clear, so it
is explicitly excluded. Completely hidden markers cannot be recovered.
Several other partial markers retain enough outline to locate them; each is
flagged in the CSV. The overlay was visually checked against the source by
both the extraction agent and an independent reviewing agent.

All 104 coordinate pairs are finite, positive and unique. The source-image
overlay provides a visual audit of the retained marker centers. Each partially
obscured or boundary-connected point has an explicit measurement note in
`points.csv`; the calibration and excluded location are recorded in `audit.json`.

## Reproduction and styling

From the repository root:

```bash
.venv-quarto/bin/python scripts/extract_tripathi_regime_points.py
.venv-quarto/bin/python scripts/create_bubble_regime_map.py
```

For an empty-directory reproducibility check and PNG previews:

```bash
.venv-quarto/bin/python scripts/extract_tripathi_regime_points.py --output-dir /tmp/tripathi-fresh-check
.venv-quarto/bin/python scripts/create_bubble_regime_map.py --output-dir /tmp/tripathi-plot-check --png
```

The single plotting generator uses the shared course style, 180 mm width and
130 mm height, 11 pt main text, and 10 pt tick labels and legend. The axis labels
are Ga and Eo; there is no footer note. SVG labels are rendered through LaTeX; editable generator text is retained,
while TeX glyphs in the SVG are vector paths. Both themes use identical data,
positions, limits, dimensions, and default Matplotlib C0-C4 colors. Backgrounds
are white and #111111. The axes are padded beyond the measured extrema so no
markers are clipped. The default TeX path passed a representative SVG export;
no fallback renderer was used for the delivered pair. `--no-tex` is an explicit
optional renderer for environments without TeX.

The SVG and PNG exports were inspected for label, legend and marker legibility
in both themes. Actual Quarto page theme verification belongs to the integrating
lecture build; successful figure generation alone does not establish that check.

PDF SHA-256: `05ef2cac64bb7d6894cd78bc4e39bee39204ea108b19b90480abb4e463de6ec1`.
Source PNG SHA-256: `ac58b9f125328859560b350b2e80943841b03bef044dbd12d476678d942b6dab`.

## Separate source-background verification figure

At the user's request, `scripts/create_bubble_regime_verification.py` produces
`bubble-regime-verification-light.svg` and `bubble-regime-verification-dark.svg`
as a separate inspection aid. Run:

```bash
.venv-quarto/bin/python scripts/create_bubble_regime_verification.py --png
```

The source raster is placed on axes linear in log10(Ga) and log10(Eo), with tick
labels showing the dimensionless physical values. Its extent is calculated
from the calibrated outside pixel edges. Open C3 rings mark log10 of the
**CSV coordinate values**, with no manual alignment shift. An inverse
calibration check requires the rounded CSV values to return to their measured
pixel centers within 0.001 pixel. The source raster aspect ratio is preserved.
The 180-by-160-mm canvas provides room for the source attribution and exclusions.

For this verification view only, the original image's colors and background
remain unchanged in both themes, as requested; the surrounding page, labels
and axes follow the theme. This deliberate reference-image exception is stated
on the figure. Both exports were visually checked. The overlay checks graphical
alignment, not whether the underlying numerical simulations are valid.
