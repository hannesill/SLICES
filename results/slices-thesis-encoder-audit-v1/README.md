# SLICES Thesis Encoder Audit v1

Checkpoint-provenance audit for the 45 core SSL pretraining encoder exports.

This package supports the thesis method claim that downstream SSL evaluations
used the fixed final pretraining export `encoder.pt`, not `encoder_best_val.pt`.
It does not contain model weights.

## Files

- `encoder_manifest.csv` - inventory of copied core SSL encoder exports.
- `encoder_epoch_audit.csv` - audit comparing retained Lightning checkpoints, best-validation checkpoint metadata, and final encoder exports.
- `checksum_manifest.json` - row counts, column counts, and SHA256 checksums for the public CSV files.

The full local encoder archive is stored under
`outputs/archives/slices-thesis-encoders-thesis-v1/` and is not part of the
publishable result table package.

Regenerate with:

```bash
uv run python scripts/diagnostics/analyze_encoder_epochs.py --strict
```
