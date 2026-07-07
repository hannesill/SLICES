# SLICES Thesis v1

Primary frozen metric export for the SLICES thesis benchmark.

Source: W&B project `hannes-ill/slices-final`, revision `thesis-v1`.

Use this package for reported thesis result values, aggregate tables, figures,
and statistical comparisons.

## Files

- `per_seed_results.parquet` - one row per evaluated run/seed.
- `aggregated_results.parquet` - grouped means, standard deviations, ranges, and confidence intervals.
- `statistical_tests.parquet` - paired comparison tests and correction flags.
- `metrics_long.parquet` - long-form metric values for auditing and plotting.
- `run_configs.parquet` - exported run configuration/provenance records.
- `label_efficiency_curves.parquet` - label-efficiency analysis rows.
- `capacity_study_comparison.parquet` - model-capacity study rows.
- `classical_context.parquet` - classical and supervised baseline context rows.
- `ts2vec_vs_core_contrastive.parquet` - TS2Vec extension comparison rows.
- `evidence_manifest.json` - row counts, column counts, SHA256 checksums, source W&B project, revision, and generation note.

Do not mix exploratory diagnostics into this package.
