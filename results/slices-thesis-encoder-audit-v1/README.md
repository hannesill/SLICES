# SLICES Thesis Encoder Audit v1

Checkpoint-provenance audit for the 45 core SSL pretraining encoder exports.

This package supports the thesis method claim that downstream SSL evaluations
used the fixed final pretraining export `encoder.pt`, not `encoder_best_val.pt`.
It does not contain model weights.

## Files

- `encoder_manifest.csv` - inventory of copied core SSL encoder exports.
- `encoder_epoch_audit.csv` - audit comparing retained Lightning checkpoints, best-validation checkpoint metadata, and final encoder exports.
- `checksum_manifest.json` - row counts, column counts, and SHA256 checksums for the public CSV files.

## Reading the audit

- `encoder.pt` is written at the end of the fixed 100-epoch schedule. It
  differs from `encoder_best_val.pt` in 44 of 45 runs.
- The retained Lightning `last.ckpt` is only refreshed when a top-k
  validation checkpoint is saved, so its epoch reflects the last top-k save,
  not the end of training: epochs 3-5 for JEPA, 80-98 for MAE, 92-99 for
  contrastive. That is why `final_encoder_equals_retained_last_ckpt` is false in
  42 of 45 runs while every row has `status=ok`. The JEPA W&B history shows
  full-length training.
- Pretraining runs are keyed by run name, not W&B run ID. TS2Vec, hp-robustness,
  and capacity-study encoders are not covered.

## Regenerating

The audit reads the local encoder archive
`outputs/archives/slices-thesis-encoders-thesis-v1/`, which is not distributed.
With that archive in place:

```bash
uv run python scripts/diagnostics/analyze_encoder_epochs.py --strict
```
