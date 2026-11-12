# Camera-ready v1: training-label support in the label-efficiency study

Computed 2026-11-05 for the ML4H 2026 camera-ready appendix. This answers the
reviewer question on how many positive training examples each seed has in the
reduced-label runs. The labeled subsets were drawn without class
stratification.

## Files

| File | Content |
|---|---|
| `low_label_train_positives.csv` | One row per dataset x task x label fraction (including the 1.0 full-label reference) x seed: 225 rows (3 datasets x 15 task-fraction cells x 5 seeds). Aggregate counts only. |
| `executed_low_label_positives.py.txt` | Byte-identical copy of the script that produced the CSV. Stored as text so it is not run from this directory. |

CSV columns: `dataset,task,label_fraction,seed,n_train_labeled,n_train_positive,train_prevalence`.
`n_train_labeled` is the size of the labeled training subset used for
optimisation. `n_train_positive` counts label == 1 in that subset.
`train_prevalence` is `n_train_positive / n_train_labeled`, rounded to 6
decimals. For `los_remaining` (regression) the last two columns are empty.
Validation and test sets are never reduced.

Grid, taken from `results/per_seed.csv` (`experiment_class == label_efficiency`):
`mortality_hospital` 0.01/0.05/0.1/0.25/0.5, `mortality_24h` 0.05/0.1/0.25/0.5,
`aki_kdigo` 0.1, `los_remaining` 0.1. Each task also has a 1.0 row. Datasets
are `miiv`, `eicu` and `combined`. Seeds are 42, 123, 456, 789 and 1011. The
classical baselines (XGBoost, GRU-D) and the capacity study use reduced-label
cells from the same grid and go through the same `ICUDataModule` path, so the
counts apply to them too.

## Code used for the split and the subset

- Launch commit of the frozen runs: `ffd6dc52c2ed224da5290cfb5d1a6634b5ba6ab6`.
  The functions were loaded from a checkout at
  `378b474f851ae9ac4f9f0f2adfd2a63ba3e65be2` (the rebuttal-diagnostics
  commit). `src/slices/data/splits.py` (blob `b3ad2fb1`) and
  `src/slices/data/datamodule.py` (blob `6504a2a4`) are identical at both
  commits. The working-tree `splits.py` has sha256 `8597fb04…801e`.
- Split: `slices.data.splits.compute_patient_level_splits`. It assigns
  patients to a 70/15/15 split over the full cohort with
  `np.random.RandomState(seed)`. Rows with a missing task label are dropped
  only afterwards. For seed 42 the function reads the cached `splits.yaml`;
  the script checked that this cache equals a fresh computation for all three
  datasets.
- Subset: `slices.data.splits.subsample_train_indices`, called by
  `ICUDataModule.setup` when `label_fraction < 1.0`. It takes
  `n = max(1, int(n_full * fraction))`, where `int()` truncates (a floor for
  positive values). It then draws
  `np.random.RandomState(seed).choice(n_full, n, replace=False)` over the
  task-filtered training rows and sorts the result. The seed is the run seed,
  so one seed fixes both the split and the subset.
- The script loads `splits.py` by file path. It does not import the training
  package or build the DataModule, and it writes nothing next to the data.

## Data

Credentialed RICU-derived processed parquet (`static.parquet`,
`timeseries.parquet` with only the `stay_id` column read, `labels.parquet`,
`splits.yaml`) for `miiv`, `eicu` and `combined`. These files are not
included. The data directory listing (path, size, mtime) was identical before
and after the run.

## Command

`RW` is a checkout at `378b474`. `PROCESSED` is the processed-data root, with
one subdirectory per dataset. Run from the script's directory with the
project environment of `RW` (uv 0.9.5, Python 3.12.12, numpy 2.3.5,
polars 1.35.2):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 uv run --no-sync --project "$RW" \
  python low_label_positives.py \
  --splits-py "$RW/src/slices/data/splits.py" \
  --processed-root "$PROCESSED" \
  --per-seed-csv results/per_seed.csv \
  --rebuttal-parity-csv "$RW/results/ml4h-rebuttal-v1/p1_validity_split_label_parity.csv" \
  --out results/camera-ready-v1/low_label_train_positives.csv
```

A second, independent run wrote a byte-identical CSV.

## Validation (all passed)

1. Test support against the frozen export: test rows and test positives
   matched `test/n_samples` and `test/n_positive` of `core_ssl_benchmark` in
   `results/per_seed.csv` for **60/60** dataset x task x seed cells (45
   classification, 15 LOS). Those cells contain 60 rows with recorded test
   support. The test support also matched **165/165** `label_efficiency`
   rows that record it (990 rows record none).
2. Rebuttal ranges: **9/9** reproduced exactly, both min-max and the rounded
   mean: 24h mortality at 5%, hospital mortality at 1%, and AKI at 10%, each
   for MIMIC-IV, eICU-CRD and Combined.
3. Subset size: `n_train_labeled == max(1, int(n_full * fraction))` in
   **165/165** reduced-label cells. The largest gap between
   `n_train_labeled / n_full` and the fraction is 2.3e-5. Full-label train
   rows and positives matched the frozen run values in the rebuttal parity
   table in **45/45** classification cells. LOS train rows matched the
   rebuttal recomputation in 15/15 cells, since the frozen LOS runs recorded
   no train count.

Within a seed, the subsets are nested across increasing fractions (165/165).
The reason is that `RandomState.choice(..., replace=False)` takes a prefix of
a single permutation of the same training list. The design does not require
this nesting.

## Checksums (sha256)

- `low_label_train_positives.csv`: `b92225be3869e12fc941396e4c64f606fa59379daa1e3266aa82eb7ecc6480e5`
- `executed_low_label_positives.py.txt`: `183baeb3fd3d67bf5a0e749254746c4490c2c12b6145e8c3b6be61bccb61b259`
