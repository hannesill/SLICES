# Provenance of `ml4h-rebuttal-v1`

- Copied on 2026-11-05 from `results/ml4h-rebuttal-v1/` of the rebuttal
  worktree at commit `378b474` (`378b474f851ae9ac4f9f0f2adfd2a63ba3e65be2`).
  Every file listed in `checksum_manifest.json` was checked against its SHA256,
  byte count and row/column count after copying; the copied files are
  byte-identical to the source and were not edited.
- `scripts/*.py.txt` are the 12 diagnostic scripts named in the README
  "Scripts" section, copied from `scripts/diagnostics/` of that worktree, where
  they were uncommitted. They are the executed source kept as text and are not
  runnable from this artifact.
- The package is post-hoc rebuttal diagnostics, separate from the frozen
  `slices-thesis-v1` export (`results/per_seed.csv`, W&B `slices-final`),
  which it only reads. It holds aggregate run-level tables only: no stay or
  patient identifiers, embeddings or predictions.
- `render_rebuttal.py` at the artifact root renders the rebuttal paper tables
  from this package and `results/per_seed.csv`.
