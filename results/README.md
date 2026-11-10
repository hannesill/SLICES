# SLICES Result Artifacts

Result packages for the SLICES thesis benchmark. Every package holds run-level
aggregates only; no patient-level data and no model weights are included.

Verify all packages with:

```bash
uv run python scripts/audit_results.py
```

## Packages

- `slices-thesis-v1/` - primary frozen thesis metric export (2,310 evaluated runs). Use this for thesis tables, figures, aggregate comparisons, and statistical tests.
- `slices-thesis-encoder-audit-v1/` - checkpoint-provenance audit for the 45 core SSL pretraining encoder exports. The encoder archive it was computed from is not distributed.
- `slices-thesis-jepa-collapse-diagnostic-v1/` - post-hoc JEPA anti-collapse diagnostic from W&B history.
- `slices-thesis-random-probe-diagnostic-v1/` - random-encoder linear-probe sanity check against the frozen SSL probes.

The three diagnostic packages are supporting evidence, not part of the primary
metric export.

Historical source project: W&B `hannes-ill/slices-final` (private), revision
`thesis-v1`. Launch code: tag `thesis-v1-launch`.
