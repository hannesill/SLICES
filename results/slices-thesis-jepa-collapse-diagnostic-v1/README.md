# SLICES Thesis JEPA Collapse Diagnostic v1

Post-hoc diagnostic checking whether JEPA target representations collapsed
during core SSL pretraining.

This is supporting diagnostic evidence. It is separate from the primary thesis
metric export in `results/slices-thesis-v1/`.

## Files

- `jepa_collapse_summary.csv` - one row per JEPA pretraining run.
- `jepa_collapse_history.parquet` - per-run logged history for target representation standard deviation and mean pairwise cosine.
- `jepa_collapse_overall.json` - compact corpus-level summary.
- `checksum_manifest.json` - row counts where applicable and SHA256 checksums for diagnostic artifacts.

Regenerate with:

```bash
uv run python scripts/diagnostics/export_jepa_collapse_diagnostics.py \
  --entity hannes-ill \
  --project slices-final \
  --revision thesis-v1 \
  --out results/slices-thesis-jepa-collapse-diagnostic-v1
```
