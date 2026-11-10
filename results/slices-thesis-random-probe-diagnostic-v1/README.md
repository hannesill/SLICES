# SLICES Random-Probe Diagnostic v1

Post-hoc sanity check: a linear probe on a randomly initialized, frozen
`TransformerEncoder`, compared with the frozen MAE, JEPA, and contrastive
linear-probe rows of `results/slices-thesis-v1/`.

This package is supporting diagnostic evidence. It is not part of the primary
thesis metric export and must not be merged into it.

## Scope

- Datasets: `miiv`, `eicu`
- Tasks: `mortality_24h`, `mortality_hospital`, `aki_kdigo`, `los_remaining`
- Seeds: 42, 123, 456, 789, 1011 (the thesis seeds; each seed sets its own patient split)
- Metric: `test/auprc` for classification, `test/mae` for LOS
- Diagnostic revision: `random-probe-cached-v1`; compared against `thesis-v1`

Result: the random-encoder probe is worse than the SSL probe in 118 of 120
matched dataset/task/seed/objective pairs.

## Files

- `random_probe_per_seed.parquet` - one row per random-probe run (40 rows).
- `random_probe_ssl_comparison_per_seed.parquet` - matched random vs SSL rows (120 rows).
- `random_probe_task_summary.parquet` - per dataset/task summary (8 rows).
- `random_probe_overall.json` - compact summary and interpretation.
- `checksum_manifest.json` - SHA256, row and column counts.

## Provenance

Runs: `scripts/diagnostics/run_cached_random_transformer_probe.py`, logged
locally without W&B (see `docs/random_transformer_probe.md`). The launch commit
was not recorded. Paths in the package are repo-relative; the cached
embeddings and probe checkpoints under `outputs/` are not distributed.

Regenerate from a local cache into a new directory:

```bash
uv run python scripts/diagnostics/export_random_probe_diagnostic.py \
  --input outputs/diagnostics/random_transformer_probe_cached \
  --out results/<new-diagnostic-dir>
```
