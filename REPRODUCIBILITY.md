# Reproducibility

SLICES has three reproducibility paths with different access requirements.

| Path | Needs | Status |
|---|---|---|
| 1. Audit the bundled results | This repository | Available |
| 2. Synthetic CPU smoke rerun | This repository | Planned |
| 3. Full ICU rerun | Credentialed MIMIC-IV/eICU, R, W&B, GPU compute | Documented below |

## 1. Audit The Bundled Thesis Export

This path is public and local. It checks the committed result packages under
`results/` without raw ICU data, W&B credentials, or encoder weights.

| Package | Contents |
|---|---|
| `results/slices-thesis-v1/` | Primary frozen thesis metric export (2,310 evaluated runs) |
| `results/slices-thesis-encoder-audit-v1/` | Final-encoder provenance for the 45 core pretraining runs |
| `results/slices-thesis-jepa-collapse-diagnostic-v1/` | JEPA anti-collapse signals from W&B history |
| `results/slices-thesis-random-probe-diagnostic-v1/` | Random-encoder linear-probe sanity check |

Source of all thesis packages: W&B `hannes-ill/slices-final` (private), revision
`thesis-v1`.

Run from the repository root:

```bash
uv run python scripts/audit_results.py
```

It checks every listed file's SHA256, row count, and column count against the
package manifest, fails on unlisted files, and scans text files and parquet
string columns for local absolute paths and retired package names. CI runs the
same command.

Source checks for the export and diagnostics code:

```bash
uv run ruff check scripts/export_results.py scripts/audit_results.py scripts/diagnostics
uv run pytest tests/test_export_results.py tests/test_audit_results.py
```

The thesis tables and figures were rendered from these packages by the thesis
repository (`scripts/viz/render_all.py` in `SLICES-thesis`), which is not public.
The ML4H 2026 paper artifact ships its own standalone audit and renderer.

## 2. Public Smoke Rerun (Planned)

A small synthetic RICU-shaped fixture that runs preprocessing, one-epoch MAE
pretraining, fine-tuning, and export on CPU without ICU data or W&B. Synthetic
metrics will be functional checks only, not scientific evidence.

## 3. Full Credentialed ICU Rerun

This path rebuilds data, reruns training, and exports a new metric package. It
does not reproduce bitwise: training ran with `deterministic: false`, and GPU
kernels are nondeterministic. Expect seed-level metrics to differ; compare
five-seed means and spreads rather than single runs.

Requirements:

- Credentialed MIMIC-IV v2.2 and eICU-CRD v2.0 access.
- R with `ricu` 0.6.3, `arrow`, `yaml`, `data.table`, `optparse`, `units`.
- A W&B entity and project you control. The runner requires online W&B.
- GPU compute for 2,455 runs (145 pretraining, 2,310 downstream).

### Code state

The thesis runs record launch commit `ffd6dc52c2ed224da5290cfb5d1a6634b5ba6ab6`.
That commit predates a history cleanup. Tag `thesis-v1-launch` (commit
`d1b86f6`) has identical code, configs, and `uv.lock`; only two internal
planning documents are absent. Check it out for an exact-code rerun:

```bash
git checkout thesis-v1-launch
uv sync --dev --frozen
```

Current `main` refactors parts of the SSL, encoder, and launcher code. Its
downstream configs resolve to the same values as the recorded runs, but it is
not the launch code.

### Steps

1. Build the processed data (see the README, "Build Processed Data"). Check that
   `data/processed/{miiv,eicu,combined}/metadata.yaml` report 74,829, 132,686,
   and 207,515 stays and 84 features. A mismatch means a different cohort; do
   not compare its results to the frozen export as a replication.

2. Warm the tensor caches:

   ```bash
   uv run python scripts/internal/run_experiments.py warmup \
     --experiment-class core_ssl_benchmark label_efficiency cross_dataset_transfer \
       hp_robustness capacity_study classical_baselines ts2vec_extension
   ```

3. Preview, then launch the matrix. Use a new revision name so reruns never
   mix with `thesis-v1`. The launcher refuses a dirty tree or a commit that
   differs from `--launch-commit`.

   ```bash
   uv run python scripts/internal/run_experiments.py run \
     --experiment-class core_ssl_benchmark label_efficiency cross_dataset_transfer \
       hp_robustness capacity_study classical_baselines ts2vec_extension \
     --project <project> --entity <entity> --revision <new-revision> \
     --launch-commit "$(git rev-parse HEAD)" --dry-run
   ```

   Drop `--dry-run` to launch. `scripts/internal/launch_thesis_tmux.sh` wraps
   warmup, training, fairness, and export in a tmux session; it reads
   `WANDB_PROJECT`, `WANDB_ENTITY`, and `REVISION` from the environment.

4. Run fairness evaluation, then export into a new directory:

   ```bash
   uv run python scripts/eval/evaluate_fairness.py \
     --project <project> --entity <entity> --revision <new-revision> --force
   uv run python scripts/export_results.py \
     --project <project> --entity <entity> --revision <new-revision> \
     --output-dir results/export-<new-revision>
   ```

   The export refuses to write into a directory that already holds an
   `evidence_manifest.json`, so it cannot overwrite `results/slices-thesis-v1/`
   by accident.

5. Compare `per_seed_results.parquet` and `aggregated_results.parquet` with the
   frozen package by experiment class, dataset, task, protocol, and paradigm.
   There is no automated comparison tool yet.

### Known deviations in the frozen export

- Run `9wmuq5jv` (core MAE, eICU, LOS, seed 42, linear probe) was assembled
  from W&B runs `zx225mz4` and `8qmc2n7r` after a resumed training job. Its
  `config_json` records the stitch policy; no metric values are averaged.
- 84 classical-baseline runs (31 GRU-D, 53 XGBoost) were launched on
  2026-05-24/25 to fill missing matrix rows. They record
  `rerun_reason="complete missing thesis-final matrix rows"` and the same
  launch commit.
- The SMART external reference (135 runs) is outside the exported corpus. The
  tmux launcher now includes it only with `INCLUDE_SMART_REFERENCE=1`, which
  was the default at launch time.
- Pretraining runs are not exported. Downstream rows link to their encoder
  through the checkpoint path in `config_json`.
- Encoder weights are not distributed. `results/slices-thesis-encoder-audit-v1/`
  records their SHA256 values for the 45 core pretraining runs.
