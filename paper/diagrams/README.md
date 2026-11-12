# Manual method diagrams

The paper uses five SVG sources and their vector PDFs. The benchmark overview,
SSL objectives, and Protocol A/B use the revised native Figma exports integrated
on 7 September 2026. The data-pipeline and encoder diagrams are unchanged.
`SOURCE_MANIFEST.json` pins every source, PDF, and conversion dependency.

Paper compilation uses the supplied PDFs. The numerical renderer regenerates
separate result tables and plots, not these manual diagrams.

Optional conversion requires Typst 0.14.2. From the extracted code archive's root:

```sh
uv run --no-project python paper/diagrams/convert.py --out converted-diagrams
```

The revised sources use the bundled Inter 3.19 fonts, including their license,
and produce PDFs at 6.5 inches wide. The two unchanged sources retain the earlier
Arial fallback conversion. The converter checks source and dependency hashes
and reports whether each PDF matches its supplied version. All five match on
the validated host. Other renderer versions or fallback fonts can change bytes.

The three revised SVGs retain native editable text. Inter 3.19 lacks the gradient
glyph in the protocol's stop-gradient label, so PDF conversion substitutes that
label's exact native vector outline in a temporary SVG. The delivered SVG stays
unchanged. The two older SVGs contain outlined labels.

Captions distinguish fixed settings from objective-specific masks and modules,
full JEPA target input from visible online input, and maximum training budgets
from validation-selected downstream checkpoints. Combined is pooled data.
