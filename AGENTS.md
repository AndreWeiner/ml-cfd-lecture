# Course authoring instructions

These rules apply throughout this repository to lecture manuscripts, exercises,
and their generated sketches, plots, diagrams, and animations. Follow explicit
user instructions when they override these defaults. Do not rewrite unrelated
existing figures merely because this file was added; apply the rules to new or
modified figures.

## Required visualization workflow

Use the **scientific-visualization** skill for creating, modifying, or reviewing
course figures. Read its `SKILL.md` from the available skill catalog before doing
that work. In the current environment it is located at
`/home/andre/.codex/skills/scientific-visualization/SKILL.md`.
The requirements below are the persistent course figure contract; also follow
the skill's guidance on scientific fidelity, exports, and visual inspection.

## Light and dark versions are mandatory

- Generate both `<stem>-light.<ext>` and `<stem>-dark.<ext>` from one editable
  generator and shared style configuration. This applies to conceptual sketches,
  scientific plots, and generated animations. Keep data, geometry, random samples,
  predictions, axis limits, dimensions, and label placement identical across the
  pair; change only the theme presentation.
- Match the page background: white for light mode and `#111111` for dark mode,
  unless the course theme explicitly changes. Adapt text, axes, neutral strokes,
  and gridlines for contrast. Do not put a light-only drawing on a white rectangle
  in the dark page. Transparent backgrounds are acceptable only after checking
  all foreground elements in both themes; still provide the named pair.
- Include both assets in the canonical Quarto/Jupytext source using the existing
  `.light-content` and `.dark-content` classes, with equal display widths:

  ```markdown
  ![Meaningful description.](media/example-light.svg){.light-content width=100%}
  ![Meaningful description.](media/example-dark.svg){.dark-content width=100%}
  ```

  In a percent-format Python notebook, prefix each Markdown line with `# `.
  Use the same theme classes on HTML video elements for paired animations.
  Keep any visible caption shared rather than duplicating it across themes.
- Register both assets, or a suitable paired wildcard, in `_quarto.yml` resources
  where needed. Verify their actual paths relative to the rendered HTML. Update
  source, generated assets, and cached/frozen output together. With `freeze: true`,
  a full site render can reuse stale results: explicitly refresh the affected
  notebook's frozen result before relying on the final site build.

## Colors

- Use Matplotlib's standard default categorical palette, **`C0` through `C9`**
  (not `Z0` through `Z9`), for categorical distinctions and sketch accents:

  | Color | Hex | Color | Hex |
  |:--|:--|:--|:--|
  | C0 | `#1f77b4` | C5 | `#8c564b` |
  | C1 | `#ff7f0e` | C6 | `#e377c2` |
  | C2 | `#2ca02c` | C7 | `#7f7f7f` |
  | C3 | `#d62728` | C8 | `#bcbd22` |
  | C4 | `#9467bd` | C9 | `#17becf` |

- Preserve each category's color in light and dark versions. Set the default
  categorical cycle explicitly **after** theme styles if those styles replace
  it. In SVG or another vector format, use the corresponding hex values. Do not
  invent a separate accent palette or automatically brighten category colors in
  dark mode. Theme-specific neutral colors and transparent categorical fills
  are allowed. Use line styles, markers, and labels as well as color.
- Continuous scalar fields require an appropriate sequential, diverging, or
  cyclic colormap, as specified by the visualization skill; do not force a
  categorical cycle onto continuous data. Preserve physically meaningful color
  limits across comparisons.

## Typography, proportions, and strokes

- Reuse `notebooks/figure_style.py` and `notebooks/figure_utils.py` where suitable.
  Inspect their behavior: using a helper alone does not establish compliance.
  Apply one shared configuration within a generator rather than styling each
  figure independently. Keep equivalent figures consistent across both courses.
- For Matplotlib, explicitly use `text.usetex=True` by default, with consistent
  fonts and a minimal preamble. Test an actual export containing representative
  equations and labels before producing the full set. An installed `latex`
  executable alone is insufficient verification.
- Never silently fall back to MathText or plain text. If TeX is unavailable,
  report the concrete failure and use only an explicit, documented option such
  as `--no-tex` / `use_tex=False`, recording the reason. Do not install system
  packages or alter global fonts automatically. For native SVG text, document
  the renderer; render mathematical notation through LaTeX where supported.
- Design at intended page placement size. Use shared named width presets;
  existing starting points are 89 mm / 180 mm, 11 pt text, 10 pt tick labels,
  1.6 pt main lines, and 0.8 pt axes. These are starting points, not a reason to
  shrink a multi-panel diagram until its text is unreadable. Set height to suit
  the content, preserve geometric aspect ratios, and record deliberate exceptions.
- Verify apparent text size after Quarto scaling. Avoid crowded labels, clipped
  arrows, excessive empty margins, distorted geometry, and decorative 3D effects.

## Export and completion checks

- Prefer editable SVG/PDF for vector drawings; use raster or mixed output for
  dense fields. Default raster exports to 300 DPI at intended placement size.
  For animations, use suitable browser formats and even video pixel dimensions.
- Retain the generator, shared style, input/source references, and reproduction
  command. Clearly label conceptual schematics; do not invent numerical regime
  boundaries or present illustrative geometry as measured results.
- Generate into an initially empty output directory to check reproducibility.
  Render the affected Quarto page and visually inspect **both** actual themes
  at intended display size. Toggle the page theme and confirm that exactly the
  matching asset is visible, that it loads without a broken path, and that text,
  equations, strokes, captions, and colors remain legible.
- Check dimensions, labels, units, limits, and source-download consistency as
  applicable. Run the repository's relevant checks. Report any unverified visual
  QA or renderer limitations explicitly; successful rendering alone is not visual
  verification. Do not claim underlying numerical validity from appearance.

## Mathematical notation

- Always enclose the arguments of standard mathematical functions in parentheses,
  including logarithms, exponentials, and trigonometric functions. Write
  `\log(x)`, `\log_{10}(Ga)`, `\exp(z)`, and `\sin(\theta)` rather than
  leaving the argument ungrouped. Use appropriately sized parentheses for
  fractions and longer expressions. Apply this to prose equations and figure labels.

## Course prose and preferred wording

- Keep the wording approachable, relatively informal, and explanatory. Explain
  what we are doing and why, rather than presenting the manuscript as a sequence
  of commands to the reader.
- Use the we form or passive voice for derivations and descriptions. Avoid
  imperative instructions such as “Define the feature vector”; write “We define
  the feature vector” or “The feature vector is defined”. Apply this preference
  to lectures, explanatory exercise text, captions, and other course prose.
- Use direct instructions only where the text explicitly assigns a task, such
  as numbered exercise steps or clearly identified practical instructions.
  Preserve genuine discussion questions as questions.
