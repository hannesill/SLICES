# SLICES Result Artifacts

This directory contains local result packages for the SLICES thesis benchmark.
Patient-level source data is not included.

## Packages

- `slices-thesis-v1/` - primary frozen thesis metric export. Use this for thesis tables, figures, aggregate comparisons, and statistical tests.
- `slices-thesis-encoder-audit-v1/` - small checkpoint-provenance audit for the 45 core SSL pretraining encoder exports. The full copied encoder archive is stored outside `results/` under `outputs/archives/`.
- `slices-thesis-jepa-collapse-diagnostic-v1/` - post-hoc JEPA anti-collapse diagnostic from W&B history. This is supporting diagnostic evidence, not part of the primary metric export.

Historical source project: W&B `hannes-ill/slices-final`, revision `thesis-v1`.
