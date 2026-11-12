# SLICES ML4H Rebuttal Diagnostics v1

Post-hoc diagnostics answering ML4H 2026 reviewer points on the SLICES
benchmark. This package is **not** part of the frozen thesis export
(`results/slices-thesis-v1/`), which was only read. It holds aggregate,
run-level metric tables; no patient-level data, embeddings or predictions.

Generated 2026-10-06/07 from worktree HEAD `378b474f851ae9ac4f9f0f2adfd2a63ba3e65be2`
plus the uncommitted scripts listed under "Scripts". Frozen reference numbers
come from W&B `hannes-ill/slices-final`, revision `thesis-v1`, launch commit
`ffd6dc52c2ed224da5290cfb5d1a6634b5ba6ab6`.

Extended 2026-10-10 with the two items that need the training VM (same
worktree HEAD and scripts): the checkpoint-based per-source Combined evaluation
(`p3_checkpoint_*`) and the exclusive-L4 step timing (`p2_l4_exclusive_*`).
The tables of 2026-10-06/07 are unchanged (same SHA256).

## Status

| Item | Reviewer point | Status |
|---|---|---|
| P1 converged frozen probe | probes under-trained; LOS loss mismatch | Done locally, all 45 archived encoders x 4 tasks |
| P2 compute table | compute per objective | W&B shared-GPU runtimes and FLOPs done locally. Exclusive-L4 step timing **done on the VM** (2026-10-10). The earlier exclusive timing on the Mac CPU is kept as a separate local table |
| P3 per-source Combined results | Combined split by source | Checkpoint-based evaluation **done on the VM** (2026-10-10): all 140 frozen Combined core runs, linear probe, full fine-tune and supervised. The per-source converged frozen probe (local, Mac) is kept as a separate table |
| P4 maintained-framework reproduction | HEAD vs frozen numbers | `miiv`, seeds 42 and 123, all 3 objectives: 6 pretraining and 48 downstream runs, plus 24 downstream runs from the frozen thesis encoders. All run **locally on the Mac (Apple M3 Pro, CPU)**, not on the L4 VM |

VM access. First attempt (2026-10-06): `slices-perm-us-west4-a` started (no
stockout), but SSH was not possible because `~/.ssh/google_compute_engine` is
passphrase-protected and was not loaded in an agent or the keychain. The VM
was stopped again after 11 min 15 s of uptime with nothing run on it.
Everything dated 2026-10-06/07 was therefore computed on the Mac.

Second attempt (2026-10-10, key loaded in the ssh-agent): the VM started on
the first attempt in `us-west4-a` and the P3 and P2 items ran on it (see
"VM run of 2026-10-10"). No other credential was used or created, no
temporary cloud resource (VM, disk, snapshot) was created, and the VM was
stopped afterwards.

## P1. Converged frozen probe

### Protocol (fixed before any test metric was computed)

- Embeddings: benchmark `FineTuneModule` in `protocol=linear_probe`
  configuration, final-epoch `encoder.pt`, eval mode, valid-timestep padding
  mask, mean pooling, 64 dimensions. Computed once per encoder for the full
  cohort under the seed's train-split normalisation.
- Splits and label filtering: `slices.data.splits.compute_patient_level_splits`
  (patient-level 70/15/15; rows with a missing task label, e.g. AKI, dropped
  after the global patient assignment).
- Embeddings standardised with train-split mean/sd.
- Classification (`mortality_24h`, `mortality_hospital`, `aki_kdigo`): L2
  logistic regression, L-BFGS (`tol=1e-6`, `max_iter=20000`), class weights
  `sqrt(n / (2 n_class))`, i.e. the benchmark probe's resolution of
  `class_weight=balanced`. The benchmark probe's label smoothing (0.1) is not
  applied. L2 strength on 17 log-spaced values (`1e-7 .. 10`, mean-loss scale),
  selected on validation AUPRC. Test AUPRC/AUROC through the benchmark's
  torchmetrics collection.
- `los_remaining` (raw remaining days, no target transform, as in the
  benchmark):
  - `ridge_valmse`: exact ridge least squares, 21 strengths (`1e-6 .. 1e4`),
    selected on validation MSE (loss-matched to squared error).
  - `ridge_valmae`: same ridge path, selected on validation MAE (the benchmark's
    checkpoint-selection metric for LOS).
  - `huber_valmae`: absolute-error-type probe, Huber loss with
    `delta = 0.01` days (L-BFGS with delta continuation `1 -> 0.1 -> 0.01`),
    8 strengths (`1e-6 .. 10`), selected on validation MAE.
  - Constants on the same test rows: training-median MAE, training-mean MSE.
- Test rows scored once per selected setting. All 2295 logistic, 945 ridge and
  360 Huber fits met their convergence criterion. Selected strengths were never
  at the strongest end of a grid; selections at the weakest end (6/135
  logistic, 15/45 ridge-MSE, 21/45 Huber) sit on a flat part of the path.

### Validity checks

| Check | Outcome |
|---|---|
| A. Local `encoder.pt` SHA256 vs `slices-thesis-encoder-audit-v1` | 45/45 match |
| B. Train rows, train positives, test rows, test positives vs frozen export | 45/45 classification cells and 60/60 test sizes identical; resolved class weights equal to the frozen run configs (max abs diff 2e-16); LOS train mean/median and constant-baseline MSE/MAE equal to the frozen values (max abs diff 2.5e-6) |
| C. Cached embeddings vs benchmark `FineTuneModule` forward on `ICUDataModule` batches (3 cells, full test split) | max abs diff 9.5e-7; stay order and labels identical |
| D. Benchmark capped AdamW recipe re-run on the cached embeddings vs frozen same-seed probe (15 cells) | Where the rerun trained as long as the frozen run (8 cells at the 50-epoch cap, 3 early-stopped cells with the same stopping epoch): abs AUPRC diff <= 0.008, abs LOS MAE diff <= 0.010. Where early stopping fired at a different epoch (4 cells): AUPRC diff up to 0.039, LOS MAE diff up to 0.090, LOS MSE diff up to 6.8 |
| E. Benchmark `finetune.py` linear probe re-run at HEAD from the archived frozen encoders (miiv, seed 42, 12 runs; `p4b_*`) | all 24 test metrics equal to the frozen values within 7.6e-6; same stopping epoch in 12/12 |

Check E reproduces the frozen probe numbers through the benchmark's own
pipeline on this machine, so data, splits, normalisation and encoder forward
are the frozen ones. Check D differs from it only in head initialisation and
batch order (the cached recipe consumes the random stream differently), and
shows that the capped probe is not a stable function of the embedding for
`mortality_24h` and `los_remaining`: early stopping (patience 10 on a noisy
validation metric) decides the outcome. Repeating the capped recipe with five
head-initialisation seeds on fixed embeddings
(`p1_validity_capped_probe_seed_sensitivity.csv` plus the seed-42 run of
check D) gives test AUPRC 0.253-0.269 for JEPA/miiv/`mortality_24h` (frozen
0.217, best checkpoint at epoch index 2), 0.287-0.304 for MAE/miiv/`mortality_24h`
(frozen 0.294) and test MAE 2.644-2.698 for MAE/miiv/`los_remaining`
(frozen 2.700).

### Results: classification

Test AUPRC, mean over 5 seeds (sd). "Frozen" is the capped probe in
`results/slices-thesis-v1`.

| Dataset | Task | MAE conv. | MAE frozen | JEPA conv. | JEPA frozen | Contr. conv. | Contr. frozen |
|---|---|---|---|---|---|---|---|
| miiv | mortality_24h | 0.357 (0.035) | 0.313 (0.042) | 0.298 (0.018) | 0.236 (0.028) | 0.322 (0.037) | 0.254 (0.035) |
| miiv | mortality_hospital | 0.520 (0.008) | 0.510 (0.014) | 0.440 (0.016) | 0.415 (0.020) | 0.471 (0.018) | 0.458 (0.023) |
| miiv | aki_kdigo | 0.510 (0.008) | 0.504 (0.009) | 0.433 (0.017) | 0.410 (0.017) | 0.492 (0.011) | 0.478 (0.009) |
| eicu | mortality_24h | 0.287 (0.017) | 0.276 (0.015) | 0.219 (0.019) | 0.199 (0.017) | 0.215 (0.003) | 0.195 (0.009) |
| eicu | mortality_hospital | 0.439 (0.007) | 0.439 (0.006) | 0.359 (0.005) | 0.355 (0.004) | 0.356 (0.009) | 0.350 (0.010) |
| eicu | aki_kdigo | 0.366 (0.010) | 0.361 (0.009) | 0.275 (0.004) | 0.268 (0.008) | 0.293 (0.006) | 0.286 (0.004) |
| combined | mortality_24h | 0.309 (0.033) | 0.284 (0.031) | 0.208 (0.012) | 0.188 (0.018) | 0.239 (0.023) | 0.205 (0.016) |
| combined | mortality_hospital | 0.470 (0.012) | 0.470 (0.013) | 0.363 (0.006) | 0.357 (0.006) | 0.395 (0.010) | 0.392 (0.010) |
| combined | aki_kdigo | 0.451 (0.005) | 0.448 (0.006) | 0.332 (0.004) | 0.327 (0.005) | 0.402 (0.006) | 0.396 (0.005) |

- The converged probe is better than the capped probe in 132 of 135 runs
  (mean AUPRC change +0.017, range -0.003 to +0.086). The change is largest
  for `mortality_24h` (+0.027 MAE, +0.034 JEPA, +0.041 contrastive on average),
  where 82% of frozen probes had early-stopped, and small for
  `mortality_hospital` and `aki_kdigo` (+0.004 to +0.012), where 93-96% of
  frozen probes ended at the 50-epoch cap.
- Objective ranking by mean AUPRC and by mean AUROC is identical under both
  probes in all 9 dataset x task cells; MAE is first in all 9.
- Paired per seed, MAE beats contrastive in 45/45 and JEPA in 45/45
  dataset x task x seed comparisons under the converged probe (same under the
  frozen probe), for AUPRC and for AUROC.
- The MAE advantage shrinks on MIMIC-IV and is nearly unchanged elsewhere.
  Mean paired AUPRC delta, converged (frozen):

| Dataset | Task | MAE - contrastive | MAE - JEPA |
|---|---|---|---|
| miiv | mortality_24h | +0.035 (+0.059) | +0.059 (+0.077) |
| miiv | mortality_hospital | +0.049 (+0.052) | +0.080 (+0.095) |
| miiv | aki_kdigo | +0.018 (+0.026) | +0.077 (+0.094) |
| eicu | mortality_24h | +0.072 (+0.081) | +0.068 (+0.076) |
| eicu | mortality_hospital | +0.083 (+0.089) | +0.081 (+0.084) |
| eicu | aki_kdigo | +0.074 (+0.075) | +0.091 (+0.093) |
| combined | mortality_24h | +0.070 (+0.079) | +0.101 (+0.096) |
| combined | mortality_hospital | +0.075 (+0.078) | +0.107 (+0.112) |
| combined | aki_kdigo | +0.049 (+0.051) | +0.119 (+0.120) |

Per-seed sd, min/max and sign counts are in
`p1_paired_deltas_mae_vs_others.csv`.

### Results: `los_remaining`

Test metrics, mean over 5 seeds. Constants: training-median MAE and
training-mean MSE on the same test rows.

| Dataset | Objective | Median MAE | Mean MSE | Frozen MAE / MSE / R2 | Ridge (val-MSE) MAE / MSE / R2 | Huber (val-MAE) MAE / MSE / R2 |
|---|---|---|---|---|---|---|
| miiv | MAE | 2.733 | 31.01 | 2.755 / 29.24 / 0.062 | 2.774 / 25.28 / 0.184 | 2.439 / 28.34 / 0.086 |
| miiv | JEPA | 2.733 | 31.01 | 2.679 / 30.78 / 0.007 | 2.750 / 25.30 / 0.184 | 2.420 / 28.28 / 0.088 |
| miiv | contrastive | 2.733 | 31.01 | 2.592 / 30.58 / 0.014 | 2.757 / 25.25 / 0.185 | 2.416 / 28.34 / 0.086 |
| eicu | MAE | 2.176 | 22.26 | 2.191 / 19.73 / 0.119 | 2.228 / 19.55 / 0.127 | 1.992 / 21.25 / 0.048 |
| eicu | JEPA | 2.176 | 22.26 | 2.249 / 20.70 / 0.071 | 2.283 / 20.03 / 0.105 | 2.024 / 21.79 / 0.022 |
| eicu | contrastive | 2.176 | 22.26 | 2.139 / 23.05 / -0.034 | 2.272 / 19.97 / 0.108 | 2.018 / 21.72 / 0.025 |
| combined | MAE | 2.403 | 29.16 | 2.425 / 25.70 / 0.121 | 2.469 / 25.44 / 0.130 | 2.193 / 27.78 / 0.048 |
| combined | JEPA | 2.403 | 29.16 | 2.475 / 26.23 / 0.102 | 2.506 / 25.84 / 0.116 | 2.215 / 28.27 / 0.031 |
| combined | contrastive | 2.403 | 29.16 | 2.309 / 29.47 / -0.011 | 2.486 / 25.65 / 0.123 | 2.200 / 28.03 / 0.040 |

- Converged squared-error probes (ridge, either selection rule) are worse than
  the training-median constant on MAE in every dataset x objective cell (they
  beat it in 1/5 seeds on miiv and 0/5 on eicu and combined), while beating the
  training-mean constant on MSE in 5/5 seeds everywhere (MSE reduction 18% on
  miiv, 10-13% on eicu and combined). The target is right-skewed (training
  mean 2.7-3.4 days, median 1.3-1.5 days), and a squared-error probe targets
  the conditional mean.
- The frozen capped probe sits in between. For contrastive (all datasets) and
  JEPA (miiv) it beats the median constant on MAE in 5/5 seeds with R2 near or
  below zero; for MAE and JEPA on eicu and combined it is worse than the median
  constant in 5/5 seeds.
- The absolute-error probe beats the training-median constant on MAE in 5/5
  seeds for every objective and dataset (MAE reduction 10.7-11.6% on miiv,
  7.0-8.4% on eicu, 7.8-8.7% on combined).
- The frozen LOS ranking does not survive. In the frozen export contrastive
  has the best MAE on all three datasets with R2 near or below zero: its
  capped probes stop at epoch 11-13 with patience 10, so the checkpoint
  selected on validation MAE comes from the first one to three epochs. With
  loss-matched probes: on eicu and
  combined the MAE objective is best on MAE (Huber; 5/5 seeds vs JEPA, 5/5 and
  4/5 seeds vs contrastive) and on MSE (ridge; 5/5 seeds); on miiv the three
  objectives are within 0.023 days of MAE (MAE objective worst in 5/5 seeds)
  and within 0.05 day^2 of MSE (no consistent ordering).

Full table with sd, skill scores and seed counts: `p1_los_loss_matched.csv`.

## P2. Compute

`p2_pretrain_compute_by_paradigm_dataset.csv` (5 seeds each, all runs
completed the fixed 100-epoch schedule, batch 256, fp32, one NVIDIA L4):

| Objective | Dataset | W&B runtime, h (sd) | Last logged step | s/step, shared GPU | Mean concurrent runs | Runtime / concurrency, h |
|---|---|---|---|---|---|---|
| MAE | miiv | 1.99 (0.60) | 20419 | 0.350 | 3.46 | 0.57 |
| MAE | eicu | 3.94 (0.04) | 36219 | 0.391 | 3.99 | 0.99 |
| MAE | combined | 6.16 (0.04) | 56739 | 0.391 | 4.00 | 1.54 |
| JEPA | miiv | 3.00 (0.02) | 20419 | 0.529 | 4.00 | 0.75 |
| JEPA | eicu | 5.34 (0.03) | 36219 | 0.531 | 3.99 | 1.34 |
| JEPA | combined | 8.32 (0.06) | 56739 | 0.528 | 4.00 | 2.08 |
| Contrastive | miiv | 2.67 (0.01) | 20419 | 0.470 | 4.00 | 0.67 |
| Contrastive | eicu | 4.76 (0.05) | 36219 | 0.473 | 3.99 | 1.19 |
| Contrastive | combined | 7.41 (0.03) | 56739 | 0.470 | 3.99 | 1.86 |

Caveat: W&B `_runtime` is wall-clock on one L4 shared by several runs (the
thesis launcher ran with `PARALLEL_MAIN=16`, pretraining runs take 4 slots, so
up to 4 pretraining runs at once). "Mean concurrent runs" is the time-weighted
number of `thesis-v1` runs of the project active during each pretraining run
(including itself), from run start times and durations of all 2455 runs.
"Runtime / concurrency" is a rough attribution, not a measurement. One run
shows the size of the effect: MAE/miiv/seed 42 ran with 1.32 concurrent runs
on average and took 3299 s, the other four MAE/miiv seeds ran 4-concurrent
and took 8092-8149 s for the same schedule.

Forward FLOPs per sample (`scripts/measure_flops.py`, T=24, D=84, 70% missing;
`p2_flops_per_objective.csv`): MAE 4.78 M, JEPA 10.60 M, contrastive 6.32 M
(batch 256), i.e. JEPA 2.2x and contrastive 1.3x MAE per step. Trainable
parameters during pretraining: MAE 269k, JEPA 189k (349k with the EMA target
encoder), contrastive 193k.

### Exclusive L4 step timing (VM, 2026-10-10)

`time_ssl_training_steps.py --device cuda` on `slices-perm-us-west4-a`
(g2-standard-8: one NVIDIA L4, 8 vCPU Intel Xeon 2.2 GHz; torch 2.9.1+cu128)
with nothing else on the GPU (`nvidia-smi` before each run: 0 MiB used, no
compute process). The microbenchmark runs real optimisation steps with the
thesis pretraining configuration (same encoder, SSL config, optimiser and
gradient clipping) on real training batches, seed 42: batch 256, fp32 (no
autocast), 20 warm-up steps, then 300 timed steps. Training steps only: no
validation, checkpointing or logging.

`p2_l4_exclusive_step_timing.csv` (`miiv`):

| Objective | ms/step | steps/s | s per training epoch (205 steps) | Projected 100-epoch training time, h | Step cost vs MAE |
|---|---|---|---|---|---|
| MAE | 119.6 | 8.36 | 24.5 | 0.68 | 1.00 |
| JEPA | 179.7 | 5.56 | 36.8 | 1.02 | 1.50 |
| Contrastive | 97.6 | 10.24 | 20.0 | 0.56 | 0.82 |

Checks of that table, each a separate invocation on the idle GPU:

| Variant | File suffix | MAE ms/step | JEPA ms/step | Contrastive ms/step | Cost vs MAE (JEPA / contrastive) |
|---|---|---|---|---|---|
| `miiv`, 2000 timed steps | `_miiv_2000steps` | 120.9 | 181.4 | 97.9 | 1.50 / 0.81 |
| `miiv`, 2000 timed steps, 8 data-loader workers | `_miiv_workers8_2000steps` | 117.3 | 177.2 | 97.5 | 1.51 / 0.83 |
| `eicu`, 300 timed steps | `_eicu` | 118.5 | 179.9 | 98.2 | 1.52 / 0.83 |
| `combined`, 300 timed steps | `_combined` | 119.8 | 181.1 | 99.1 | 1.51 / 0.83 |

Over all five invocations the step time is 117-121 ms (MAE), 177-181 ms
(JEPA) and 98-99 ms (contrastive). It does not depend on the dataset, and the
data-loader setting changes it by at most 3% (the frozen pretraining runs
used `data.num_workers=8`, read from their `.hydra/config.yaml`; the
microbenchmark default is 0).

Projected exclusive-L4 training time for the 100-epoch schedule (measured
ms/step x steps per epoch of seed 42 x 100), next to the frozen W&B
wall-clock on the shared GPU from the table above:

| Objective | miiv (205 steps/epoch) | eicu (362) | combined (568) | W&B shared runtime, h (miiv / eicu / combined) |
|---|---|---|---|---|
| MAE | 0.68 h | 1.19 h | 1.89 h | 1.99 / 3.94 / 6.16 |
| JEPA | 1.02 h | 1.81 h | 2.86 h | 3.00 / 5.34 / 8.32 |
| Contrastive | 0.56 h | 0.99 h | 1.56 h | 2.67 / 4.76 / 7.41 |

Applied to the gradient steps of the 45 frozen pretraining runs, the measured
ms/step give a projected 18.8 (MAE), 28.4 (JEPA) and 15.5 (contrastive)
exclusive L4 training hours for the 15 runs of each objective, against 60.4,
83.3 and 74.2 h of summed W&B wall-clock on the shared GPU.

Reading:

- JEPA costs 1.5x the MAE step on an exclusive L4, contrastive 0.8x. JEPA is
  the most expensive objective in every L4 measurement.
- The order of MAE and contrastive differs between measurements: contrastive
  is cheaper than MAE on the exclusive L4 (0.82) and on the Mac CPU (0.63),
  and more expensive in FLOPs (1.3) and in the shared-GPU W&B seconds per step
  (1.20 on eicu and combined). The frozen shared runs were 2.9-3.3x (MAE),
  2.9x (JEPA) and 4.7-4.8x (contrastive) slower per step than the exclusive
  timing, at 3.5-4.0 mean concurrent runs and including validation,
  checkpointing and logging. Why contrastive lost more under sharing was not
  investigated.
- The step is limited by the host process rather than by GPU compute. In a
  separate, untimed rerun with `nvidia-smi` sampled once per second, GPU
  utilisation during the steps was 16% (MAE), 10% (JEPA) and 18%
  (contrastive) with 0.4 GiB of GPU memory in use, while the Python process
  used one full CPU core (samples under the gitignored
  `outputs/ml4h-rebuttal/compute/l4_logs/util/`). Accordingly the exclusive L4
  is not faster than the Mac CPU for this model size: 1.02x (MAE), 1.49x
  (JEPA) and 1.32x (contrastive) the Mac CPU step time.
- The frozen run with the least sharing is consistent with the exclusive
  timing: MAE/miiv/seed 42 took 3299 s for 20500 steps at 1.32 mean concurrent
  runs (0.161 s/step including validation and checkpointing), against 0.120
  s/step for training steps alone here.

### Exclusive Mac CPU step timing (local, 2026-10-07)

`p2_local_cpu_step_timing.csv` is the same microbenchmark (300 timed
optimisation steps on real `miiv` batches, thesis config, batch 256) run on
the Mac with no other job of this task running (Apple M3 Pro, CPU, 5 threads).
It was made as a substitute before the VM was reachable and is kept for
comparison:

| Objective | ms/step | steps/s | s per training epoch (205 steps) | Projected 100-epoch training time, h | Step cost vs MAE |
|---|---|---|---|---|---|
| MAE | 117.5 | 8.51 | 24.1 | 0.67 | 1.00 |
| JEPA | 120.9 | 8.27 | 24.8 | 0.69 | 1.03 |
| Contrastive | 74.1 | 13.50 | 15.2 | 0.42 | 0.63 |

These are CPU-hours on a laptop, not L4 GPU-hours, and exclude validation and
checkpointing. The cost ordering depends on the device: FLOPs rank
MAE < contrastive < JEPA, the shared L4 ranks MAE < contrastive < JEPA
(0.39 / 0.47 / 0.53 s/step), the exclusive L4 ranks contrastive < MAE < JEPA,
this CPU ranks contrastive < MAE ~ JEPA.

## P3. Per-source Combined results

Two sets of tables, from different machines and protocols:

| Tables | Where | What |
|---|---|---|
| `p3_checkpoint_per_source_*` | training VM, 2026-10-10 | the 140 frozen Combined core runs (linear probe, full fine-tune, supervised), re-evaluated from their evaluated checkpoints |
| `p3_local_converged_probe_per_source_*` | Mac, 2026-10-06/07 | the P1 converged frozen probes of the Combined-regime encoders (frozen-encoder protocol only) |

### P3a. Checkpoint-based evaluation of the frozen runs (VM)

`scripts/diagnostics/eval_combined_per_source.py` reads run selection, configs
and checkpoint provenance from the local frozen export (no W&B access),
verifies each checkpoint's SHA256 against `_eval_artifact_sha256`, re-runs
test inference and reports overall and per-`source_dataset` metrics. It was
run on `slices-perm-us-west4-a` (NVIDIA L4, torch 2.9.1+cu128) from a separate
copy of this worktree (`~/SLICES-rebuttal`); `~/SLICES` was only read.

- Runs: 140 = 3 SSL objectives x 2 protocols x 4 tasks x 5 seeds, plus the
  supervised Transformer x 4 tasks x 5 seeds. All 140 evaluated checkpoints
  exist on the VM disk and all 140 match the recorded SHA256 (checked by the
  script and independently with `sha256sum -c`). All 140 evaluated
  (`status=ok`).
- Reproduction of the frozen export: the overall (not per-source) test
  metrics of the re-run equal the frozen values within 2.4e-5 over all runs
  and metrics (max abs diff AUPRC 2.9e-6, AUROC 1.8e-7, LOS absolute error
  1.8e-6, LOS MSE 2.4e-5, R2 1.6e-6).
- Test rows and positives per task x seed x portion equal those of the local
  converged-probe table in 60/60 cells.
- Integrity: file listings (path, size, mtime) of `~/SLICES/data/processed`
  and `~/SLICES/outputs` were identical before and after; the data was read
  through a symlink view and nothing was written to the source directories.

Mean over 5 seeds (sd). "MAE" in column headers is the masked-autoencoder
objective; the LOS metric is written "abs. error". Rows per portion: the
MIMIC-IV and eICU stays of the Combined test split (`source_dataset`).

Full fine-tune and supervised, classification:

| Task | Portion | n (mean) | Prevalence | Metric | MAE | JEPA | Contrastive | Supervised |
|---|---|---|---|---|---|---|---|---|
| mortality_24h | MIMIC-IV | 11149 | 0.016 | AUPRC | 0.448 (0.033) | 0.423 (0.051) | 0.429 (0.053) | 0.424 (0.032) |
| mortality_24h | MIMIC-IV | 11149 | 0.016 | AUROC | 0.939 (0.008) | 0.937 (0.007) | 0.936 (0.011) | 0.935 (0.010) |
| mortality_24h | eICU | 19714 | 0.016 | AUPRC | 0.392 (0.026) | 0.377 (0.027) | 0.384 (0.024) | 0.373 (0.029) |
| mortality_24h | eICU | 19714 | 0.016 | AUROC | 0.926 (0.006) | 0.923 (0.008) | 0.923 (0.010) | 0.917 (0.009) |
| mortality_hospital | MIMIC-IV | 11155 | 0.118 | AUPRC | 0.576 (0.010) | 0.570 (0.013) | 0.563 (0.017) | 0.566 (0.012) |
| mortality_hospital | MIMIC-IV | 11155 | 0.118 | AUROC | 0.890 (0.005) | 0.887 (0.005) | 0.885 (0.003) | 0.886 (0.005) |
| mortality_hospital | eICU | 19714 | 0.093 | AUPRC | 0.498 (0.014) | 0.485 (0.008) | 0.479 (0.016) | 0.484 (0.012) |
| mortality_hospital | eICU | 19714 | 0.093 | AUROC | 0.869 (0.003) | 0.864 (0.003) | 0.863 (0.005) | 0.863 (0.004) |
| aki_kdigo | MIMIC-IV | 8456 | 0.204 | AUPRC | 0.697 (0.002) | 0.695 (0.006) | 0.685 (0.006) | 0.720 (0.008) |
| aki_kdigo | MIMIC-IV | 8456 | 0.204 | AUROC | 0.879 (0.005) | 0.876 (0.004) | 0.874 (0.004) | 0.887 (0.004) |
| aki_kdigo | eICU | 13268 | 0.139 | AUPRC | 0.553 (0.010) | 0.556 (0.007) | 0.541 (0.005) | 0.582 (0.006) |
| aki_kdigo | eICU | 13268 | 0.139 | AUROC | 0.848 (0.004) | 0.845 (0.005) | 0.843 (0.003) | 0.854 (0.006) |

Full fine-tune and supervised, `los_remaining`:

| Portion | n (mean) | Label mean / median, days | Metric | MAE | JEPA | Contrastive | Supervised |
|---|---|---|---|---|---|---|---|
| MIMIC-IV | 11199 | 3.39 / 1.49 | Abs. error, days | 2.387 (0.045) | 2.408 (0.049) | 2.422 (0.057) | 2.468 (0.031) |
| MIMIC-IV | 11199 | 3.39 / 1.49 | MSE, days^2 | 27.88 (1.03) | 28.52 (1.57) | 27.83 (1.85) | 28.16 (1.57) |
| eICU | 19893 | 2.75 / 1.30 | Abs. error, days | 1.967 (0.025) | 1.982 (0.011) | 2.001 (0.024) | 2.034 (0.039) |
| eICU | 19893 | 2.75 / 1.30 | MSE, days^2 | 22.88 (6.90) | 22.93 (7.14) | 22.78 (6.98) | 22.93 (6.74) |

Linear probe (the frozen capped probe), classification:

| Task | Portion | n (mean) | Prevalence | Metric | MAE | JEPA | Contrastive |
|---|---|---|---|---|---|---|---|
| mortality_24h | MIMIC-IV | 11149 | 0.016 | AUPRC | 0.298 (0.044) | 0.227 (0.008) | 0.236 (0.011) |
| mortality_24h | MIMIC-IV | 11149 | 0.016 | AUROC | 0.912 (0.014) | 0.887 (0.009) | 0.901 (0.010) |
| mortality_24h | eICU | 19714 | 0.016 | AUPRC | 0.282 (0.037) | 0.175 (0.025) | 0.195 (0.021) |
| mortality_24h | eICU | 19714 | 0.016 | AUROC | 0.906 (0.009) | 0.863 (0.018) | 0.878 (0.017) |
| mortality_hospital | MIMIC-IV | 11155 | 0.118 | AUPRC | 0.502 (0.013) | 0.394 (0.009) | 0.432 (0.016) |
| mortality_hospital | MIMIC-IV | 11155 | 0.118 | AUROC | 0.864 (0.005) | 0.811 (0.009) | 0.832 (0.005) |
| mortality_hospital | eICU | 19714 | 0.093 | AUPRC | 0.446 (0.014) | 0.332 (0.008) | 0.363 (0.008) |
| mortality_hospital | eICU | 19714 | 0.093 | AUROC | 0.851 (0.004) | 0.792 (0.005) | 0.813 (0.001) |
| aki_kdigo | MIMIC-IV | 8456 | 0.204 | AUPRC | 0.520 (0.007) | 0.388 (0.008) | 0.480 (0.012) |
| aki_kdigo | MIMIC-IV | 8456 | 0.204 | AUROC | 0.823 (0.005) | 0.720 (0.004) | 0.779 (0.006) |
| aki_kdigo | eICU | 13268 | 0.139 | AUPRC | 0.371 (0.009) | 0.259 (0.006) | 0.304 (0.009) |
| aki_kdigo | eICU | 13268 | 0.139 | AUROC | 0.786 (0.007) | 0.661 (0.002) | 0.716 (0.005) |

Linear probe (the frozen capped probe), `los_remaining`:

| Portion | n (mean) | Label mean / median, days | Metric | MAE | JEPA | Contrastive |
|---|---|---|---|---|---|---|
| MIMIC-IV | 11199 | 3.39 / 1.49 | Abs. error, days | 2.801 (0.052) | 2.793 (0.052) | 2.643 (0.081) |
| MIMIC-IV | 11199 | 3.39 / 1.49 | MSE, days^2 | 29.62 (1.66) | 29.83 (1.60) | 34.97 (1.86) |
| eICU | 19893 | 2.75 / 1.30 | Abs. error, days | 2.214 (0.020) | 2.296 (0.021) | 2.122 (0.020) |
| eICU | 19893 | 2.75 / 1.30 | MSE, days^2 | 23.49 (6.89) | 24.21 (6.88) | 26.37 (7.04) |

Objective ranking per portion (seed counts are out of 5, computed per
protocol x task x portion x seed from `p3_checkpoint_per_source_per_run.csv`):

- Linear probe, classification: MAE > contrastive > JEPA on both portions for
  all three tasks, by AUPRC and by AUROC. The MAE objective beats both others
  in 5/5 seeds in all 12 task x portion x metric cells; the complete order
  holds in 5/5 seeds in 11 cells and in 4/5 in one (`mortality_24h` AUPRC on
  the MIMIC-IV portion). Mean paired AUPRC advantage of the MAE objective:
  +0.040 to +0.087 over contrastive and +0.071 to +0.132 over JEPA.
- Full fine-tune, classification: the objectives are close. By mean AUPRC the
  MAE objective is first in 5 of the 6 task x portion cells; JEPA is first for
  `aki_kdigo` on the eICU portion (0.556 vs 0.553, JEPA ahead in 4/5 seeds).
  By mean AUROC the MAE objective is first in all 6. The best-to-worst spread
  is 0.012-0.025 AUPRC and 0.003-0.006 AUROC. Per seed, the MAE objective has
  the best AUPRC in 4/5 seeds (`mortality_24h` on both portions,
  `mortality_hospital` on MIMIC-IV), 5/5 (`mortality_hospital` on eICU) and
  3/5 (`aki_kdigo` on MIMIC-IV); the complete order of the means is reproduced
  by 2 to 4 of 5 seeds. Second place by mean AUPRC is contrastive for
  `mortality_24h` and JEPA for `mortality_hospital` and `aki_kdigo`, on both
  portions.
- Supervised Transformer: on `aki_kdigo` it is ahead of all three fine-tuned
  SSL objectives in 5/5 seeds on both portions, by AUPRC and by AUROC. On the
  two mortality tasks the fine-tuned MAE objective is ahead of it by AUPRC in
  5/5 seeds on both portions (+0.010 to +0.024), while JEPA and contrastive
  are ahead of it in 1 to 4 of 5 seeds.
- `los_remaining`, full fine-tune: absolute error orders MAE < JEPA <
  contrastive < supervised on both portions (MAE objective better than
  contrastive in 4/5 seeds on MIMIC-IV and 5/5 on eICU, better than JEPA in
  5/5 and 4/5). MSE differences between the four are within one seed sd on
  both portions.
- `los_remaining`, linear probe: contrastive has the lowest absolute error on
  both portions (5/5 seeds) together with the highest MSE and R2 below zero
  (-0.015 on MIMIC-IV, -0.013 on eICU); the MAE objective has the lowest MSE
  on both portions (5/5 seeds vs contrastive; 4/5 and 5/5 vs JEPA). The
  capped-probe LOS pattern described under P1 is present on both portions.
- Source effect: mean AUROC is higher on the MIMIC-IV portion than on the eICU
  portion in all 21 objective x protocol x task cells (101 of 105 runs). For
  `mortality_24h`, where prevalence is 0.016 on both portions, AUPRC is higher
  on the MIMIC-IV portion in 29 of 35 runs. AUPRC is not comparable across
  portions with different prevalence (`mortality_hospital`, `aki_kdigo`).

### P3b. Converged frozen probe per source (local)

Computed on the Mac (frozen-encoder protocol only): the P1 converged probes of
the Combined-regime encoders, scored separately on the two portions of the
Combined test split (`p3_local_converged_probe_per_source_*.csv`). Test AUPRC,
mean over 5 seeds (sd):

| Task | Portion | n (mean) | Prevalence | MAE | JEPA | Contrastive |
|---|---|---|---|---|---|---|
| mortality_24h | MIMIC-IV | 11149 | 0.016 | 0.334 (0.037) | 0.250 (0.019) | 0.276 (0.031) |
| mortality_24h | eICU | 19714 | 0.016 | 0.302 (0.048) | 0.195 (0.018) | 0.225 (0.026) |
| mortality_hospital | MIMIC-IV | 11155 | 0.118 | 0.507 (0.011) | 0.401 (0.008) | 0.442 (0.016) |
| mortality_hospital | eICU | 19714 | 0.093 | 0.443 (0.014) | 0.337 (0.008) | 0.362 (0.008) |
| aki_kdigo | MIMIC-IV | 8456 | 0.204 | 0.525 (0.007) | 0.396 (0.007) | 0.486 (0.013) |
| aki_kdigo | eICU | 13268 | 0.139 | 0.373 (0.010) | 0.261 (0.005) | 0.309 (0.007) |

MAE is first on both portions for all three classification tasks and beats
both other objectives in 5/5 seeds on each portion. LOS (Huber probe, MAE in
days): eICU portion MAE 2.013, contrastive 2.036, JEPA 2.049 (MAE objective
best in 5/5 seeds); MIMIC-IV portion contrastive 2.493, JEPA 2.509, MAE 2.514
(MAE objective behind contrastive in 5/5 seeds). AUPRC is not comparable
across portions with different prevalence.

### P3c. Do the two per-source tables agree?

- Classification, frozen encoder: yes. The capped probe checkpoints (P3a) and
  the converged probes (P3b) give the same order, MAE > contrastive > JEPA, on
  both portions for all three tasks, by AUPRC and by AUROC (12/12 cells), with
  the MAE objective ahead of both others in 5/5 seeds in every cell of both
  tables.
- Classification, full fine-tune (P3a only): the MAE objective is still first
  by mean in 5 of 6 cells by AUPRC and 6 of 6 by AUROC, but the best-to-worst
  spread is 0.012-0.025 AUPRC against 0.071-0.132 for the capped probe, and
  JEPA is ahead of contrastive (the reverse of the frozen-encoder order) in 4
  of 6 cells by AUPRC and 6 of 6 by AUROC.
- `los_remaining`, frozen encoder: no. On the MIMIC-IV portion both tables
  order absolute error as contrastive, JEPA, MAE objective. On the eICU
  portion the capped probe has contrastive first and the converged
  absolute-error probe has the MAE objective first. This is the P1 finding
  (the frozen LOS probe ranking does not survive a loss-matched converged
  probe), now seen per portion.

Zero-shot cross-database evaluation (stretch): not attempted and not in scope
of the VM run of 2026-10-10. `ICUDataset` normalises each database with its
own train-split statistics, so applying a source model's normalisation to the
other database needs a preprocessing path that does not exist in the
benchmark.

## P4. Maintained-framework reproduction slice

Run on the Mac (Apple M3 Pro, CPU, fp32) because the VM was not reachable at
the time (2026-10-06/07); the frozen runs used an NVIDIA L4. CPU and CUDA draw different random streams
for SSL masks and dropout, so a rerun with the same seed is a different
realisation of the same configuration, not a bitwise repeat. The yardstick is
the frozen 5-seed standard deviation of the same paradigm x task x protocol
cell.

Code state: the tree diff between the launch commit and HEAD touches 30 files
under `src/`, `configs/`, `scripts/training/` and `pyproject.toml`
(202 insertions, 123 deletions): validation refactors in the SSL objectives
and checkpoint loading, config-schema validation, removal of `los_days` from
model-facing static batch fields, import order and comments. `uv.lock` is
unchanged.

Commands are the ones the HEAD launcher generates (`protocol=linear_probe` /
`protocol=full_finetune`, downstream from the final `encoder.pt`) plus
`data.processed_dir`, `data.num_workers=0`, `training.accelerator=cpu`,
`training.devices=1` and the rebuttal W&B target. The resolved configuration
of every rerun equals the frozen run's exported configuration after removing
provenance and location keys (78/78 runs, 0 training-config differences).
20 of the frozen `miiv` seed-123 downstream runs carry
`ckpt_path=.../last.ckpt`, i.e. they had been resumed after an interruption;
the reruns ran uninterrupted.

### P4a. Full pipeline at HEAD: pretraining, then probe and fine-tune

`miiv`, seeds 42 and 123, 3 objectives: 6 pretraining runs and 48 downstream
runs (`p4_head_reproduction_*`).

Pretraining: all 6 runs completed the 100-epoch schedule with the same number
of optimiser steps as the frozen runs. Final validation loss is within 0.63
frozen seed sd of the frozen same-seed value in 6/6 runs. Over the last 50
epochs the rerun and frozen loss curves differ by 0.5-2.8% of the loss level
on average for 11 of 12 curves (7.2% for the MAE seed-123 validation loss).
The final-epoch training loss is lower than the frozen value by 5.0 seed sd
for contrastive seed 42 (0.0666 vs 0.0689; the seed sd is 0.0005) and by 2.3
seed sd for MAE seed 42; the other four runs are within 1.2 seed sd.

Downstream, primary metric (AUPRC for the three classification tasks, MAE for
`los_remaining`), rerun vs frozen same-seed value:

| Protocol | Runs | Within 1 seed sd | Within 2 seed sd | Median abs diff / sd | Max abs diff / sd | Same stopping epoch |
|---|---|---|---|---|---|---|
| Linear probe | 24 | 23 | 24 | 0.23 | 1.08 | 18 |
| Full fine-tune | 24 | 19 | 24 | 0.43 | 1.82 | 3 |
| All | 48 | 42 | 48 | 0.28 | 1.82 | 21 |

- AUPRC difference (rerun - frozen): linear probe mean -0.001, mean abs
  0.008, max abs 0.031; full fine-tune mean +0.004, mean abs 0.010, max abs
  0.048.
- LOS MAE difference: linear probe mean abs 0.007 days (max 0.016); full
  fine-tune +0.044 days on average, higher than the frozen value in 6/6 runs
  (max 0.074).
- Including the secondary metrics (AUROC, LOS MSE): 84 of 96 metrics within
  1 seed sd, 95 within 2 (exception: contrastive/`aki_kdigo`/probe AUROC,
  seed 123, -0.010, 2.1 sd).
- Objective order within seed x task x protocol equals the frozen same-seed
  order in 13/16 cells (8/8 for the linear probe). The three changes are full
  fine-tune cells: `mortality_24h` for both seeds and `los_remaining` for
  seed 42.

### P4b. Downstream only at HEAD, from the frozen thesis encoders

`miiv`, seed 42, 12 linear probes and 12 full fine-tunes started from the
archived `encoder.pt` files (`p4b_*`).

- Linear probes: all 24 test metrics equal the frozen values within 7.6e-6
  (AUPRC within 1.5e-6), with the same stopping epoch in 12/12 runs. The
  frozen-encoder probe path is deterministic given the seed and is reproduced
  at HEAD on different hardware.
- Full fine-tunes: primary metric within 1 seed sd in 12/12 runs (median 0.35
  sd, max 0.90 sd); max abs AUPRC difference 0.040, max abs LOS MAE difference
  0.045 days; all 24 fine-tune metrics within 1 seed sd. These runs share
  encoder, head initialisation and batch order with the frozen runs, so the
  remaining differences come from running the trainable encoder on CPU
  instead of the L4.

### Reading

HEAD reproduces the frozen `miiv` numbers within seed-to-seed variation, and
the frozen-encoder linear probe exactly. One pattern is not explained: LOS
full fine-tune MAE is higher in the rerun than in the frozen run in all 9
comparisons (6 in P4a, 3 in P4b; +0.03 to +0.04 days on average, each within
1.5 seed sd). This slice cannot separate chance from a small CPU-vs-GPU
effect. No L4 rerun was made.

## VM run of 2026-10-10

`slices-perm-us-west4-a` (`us-west4-a`, g2-standard-8, one NVIDIA L4, driver
580.173.02; Python 3.12.12, torch 2.9.1+cu128, the same torch build as the
frozen `~/SLICES/.venv`). Two sessions: 09:39:06-10:28:11 UTC (P3, then P2)
and 10:30:31-10:49:48 UTC (P2 with 8 data-loader workers); 68 min of uptime
in total (49 min 5 s and 19 min 17 s). `sudo shutdown -h` was set as a
dead-man switch right after the first SSH of each session (+840 and +120
minutes).

What was found on the disk before anything ran:

- Repository at `~/SLICES`, HEAD `0c72016` (one commit after the launch commit
  `ffd6dc5`), clean apart from an untracked `.codex` directory.
- Processed data at `~/SLICES/data/processed/{miiv,eicu,combined}` with the
  tensor caches and five hash-keyed normalisation files per dataset.
- `~/SLICES/outputs/core_ssl_benchmark_rev-thesis-v1` holds the run
  directories; all 140 evaluated Combined checkpoints exist and match their
  recorded SHA256.
- The crontab entry for the idle auto-shutdown points at
  `~/SLICES/scripts/auto_shutdown.sh`, which no longer exists (the script is
  at `scripts/internal/auto_shutdown.sh`), so it fails every five minutes and
  cannot power the VM off. It was left as found.

Commands as run (code shipped as a tarball because HEAD is not on `origin`).
Nothing under `~/SLICES` was written: its `git status` and the file listings
(path, size, mtime) of `data/processed` and `outputs` were identical before
and after both sessions; only the mtime of the `.git` directory changed, from
running `git status`.

```bash
# Mac
(git ls-files; git ls-files --others --exclude-standard scripts tests) | sort -u \
  | tar czf slices-rebuttal.tgz -T -
gcloud compute scp slices-rebuttal.tgz hannesill@slices-perm-us-west4-a:~ --zone us-west4-a

# VM (in tmux, WANDB_MODE=disabled)
mkdir -p ~/SLICES-rebuttal && tar xzf ~/slices-rebuttal.tgz -C ~/SLICES-rebuttal
cd ~/SLICES-rebuttal && uv sync
# P3a
uv run python scripts/diagnostics/eval_combined_per_source.py \
  --outputs-root ~/SLICES/outputs --processed-root ~/SLICES/data/processed --device cuda
# P2, after P3 had exited (nvidia-smi: 0 MiB used, no compute process)
T="uv run python -m scripts.diagnostics.time_ssl_training_steps \
  --processed-root $HOME/SLICES/data/processed \
  --data-view-root outputs/ml4h-rebuttal/data_view --device cuda"
C=outputs/ml4h-rebuttal/compute
$T --output $C/l4_exclusive_step_timing.csv
$T --output $C/l4_exclusive_step_timing_miiv_2000steps.csv --timed-steps 2000
$T --output $C/l4_exclusive_step_timing_eicu.csv --dataset eicu
$T --output $C/l4_exclusive_step_timing_combined.csv --dataset combined
$T --output $C/l4_exclusive_step_timing_miiv_workers8_2000steps.csv \
  --num-workers 8 --timed-steps 2000
```

Only the aggregate tables (and the timing logs, which hold `nvidia-smi`
snapshots and the timing tables) were copied back; no data, predictions or
embeddings left the VM. `~/SLICES-rebuttal` (code copy, virtual environment,
symlink views, outputs and logs) was left on the VM disk.

Still not run: the zero-shot cross-database evaluation, and the optional L4
repeat of the P4 reproduction slice. The latter would be, from
`~/SLICES-rebuttal` on the VM (with `WANDB_MODE=offline` if the VM is not
logged in to W&B, then `wandb sync` from the Mac):

```bash
uv run python -m scripts.diagnostics.run_head_reproduction_slice \
  --processed-root ~/SLICES/data/processed --data-view-root outputs/ml4h-rebuttal/data_view \
  --launch-commit 378b474f851ae9ac4f9f0f2adfd2a63ba3e65be2
```

## W&B

Project `hannes-ill/slices-ml4h-rebuttal` (new). Nothing was written to
`slices-final`.

- `experiment_class:converged_frozen_probe`, `revision:ml4h-rebuttal-v1`: 45
  runs `converged_frozen_probe_{dataset}_{paradigm}_seed{seed}`, groups
  `converged_frozen_probe_{dataset}_{paradigm}`; configs and summary metrics.
- `experiment_class:core_ssl_benchmark`, `revision:ml4h-rebuttal-v1`: the 54
  P4a runs, with the benchmark's usual names and groups
  (`core_ssl_benchmark_{pretrain|probe|finetune}_miiv_...`,
  groups `pretrain_miiv_{paradigm}`, `probe_...`, `finetune_...`).
- `experiment_class:core_ssl_benchmark`,
  `revision:ml4h-rebuttal-v1-frozen-encoders`: the 24 P4b runs, same names and
  groups.

## Commands

All from the worktree root. `ARCHIVE` is
`/Users/hannesill/Developer/SLICES/outputs/archives/slices-thesis-encoders-thesis-v1`.
`VIEW` is `outputs/ml4h-rebuttal/data_view`, a directory of symlinks to the
processed parquet/yaml files of the main checkout, so cache files written by
the data module never land in the main checkout.

```bash
# P1
uv run python scripts/diagnostics/extract_frozen_ssl_embeddings.py \
  --encoder-archive $ARCHIVE --processed-root $VIEW --device cpu --skip-existing
uv run python -m scripts.diagnostics.check_frozen_probe_validity \
  --encoder-archive $ARCHIVE --processed-root $VIEW \
  --checks encoders splits embeddings capped capped_seed_sensitivity
uv run python scripts/diagnostics/run_converged_frozen_probe.py --processed-root $VIEW --skip-existing
uv run python scripts/diagnostics/export_ml4h_rebuttal_probe.py
uv run python -m scripts.diagnostics.log_ml4h_rebuttal_probe_wandb --launch-commit $(git rev-parse HEAD)

# P2
uv run python scripts/diagnostics/export_pretrain_compute.py
uv run python scripts/measure_flops.py --no-wandb
uv run python scripts/diagnostics/time_ssl_training_steps.py --processed-root $VIEW --device cpu
# P2 exclusive-L4 timing: on the training VM, see "VM run of 2026-10-10"

# P3b (local converged probe per source)
uv run python -m scripts.diagnostics.run_converged_probe_per_source --processed-root $VIEW
# P3a (checkpoint-based): on the training VM, see "VM run of 2026-10-10"

# P4a (one process per objective and seed; OMP_NUM_THREADS=3; seeds 42 and 123)
uv run python scripts/diagnostics/run_head_reproduction_slice.py \
  --processed-root $VIEW --launch-commit $(git rev-parse HEAD) --paradigms mae --seeds 42 \
  --extra-override data.num_workers=0 --extra-override training.accelerator=cpu \
  --extra-override training.devices=1
uv run python scripts/diagnostics/export_head_reproduction.py
# P4b (downstream only, from the frozen thesis encoders)
uv run python scripts/diagnostics/run_head_reproduction_slice.py \
  --processed-root $VIEW --launch-commit $(git rev-parse HEAD) --paradigms mae \
  --phases probe finetune --encoder-root $ARCHIVE \
  --revision ml4h-rebuttal-v1-frozen-encoders \
  --reason "ml4h rebuttal HEAD downstream rerun from frozen thesis encoders" \
  --extra-override data.num_workers=0 --extra-override training.accelerator=cpu \
  --extra-override training.devices=1
uv run python scripts/diagnostics/export_head_reproduction.py \
  --run-root outputs/ml4h-rebuttal/head_reproduction/core_ssl_benchmark_rev-ml4h-rebuttal-v1-frozen-encoders \
  --revision ml4h-rebuttal-v1-frozen-encoders \
  --output-dir outputs/ml4h-rebuttal/head_downstream_frozen_encoders_tables

# Package
uv run python -m scripts.diagnostics.package_ml4h_rebuttal
```

Tests: `uv run pytest tests/test_converged_frozen_probe.py tests/test_ml4h_rebuttal_diagnostics.py`.

## Scripts

New, uncommitted, under `scripts/diagnostics/`:
`extract_frozen_ssl_embeddings.py`, `run_converged_frozen_probe.py`,
`check_frozen_probe_validity.py`, `export_ml4h_rebuttal_probe.py`,
`log_ml4h_rebuttal_probe_wandb.py`, `export_pretrain_compute.py`,
`time_ssl_training_steps.py`, `eval_combined_per_source.py`,
`run_converged_probe_per_source.py`, `run_head_reproduction_slice.py`,
`export_head_reproduction.py`, `package_ml4h_rebuttal.py`. No existing source
file, config or script was changed.

Change of 2026-10-10: `package_ml4h_rebuttal.py` got six more entries in its
copy plan (`p3_checkpoint_per_source_summary.json` and the five
`p2_l4_exclusive_step_timing*.csv` files). The two scripts that ran on the VM
were used unmodified.

## Caveats

- Five seeds per cell; sds are over seeds and paired counts are out of 5.
- P1, the converged-probe per-source tables (P3b), the Mac CPU step timing and
  all of P4 ran on the Mac, not on a VM. P4 timing on the Mac is not exclusive
  (3 to 6 training processes shared the CPU). Only the checkpoint-based
  per-source evaluation (P3a) and the `p2_l4_exclusive_*` timings ran on the
  L4 VM.
- The exclusive-L4 timing is a microbenchmark of training steps for one seed
  (42); it is not a re-run of pretraining and excludes validation,
  checkpointing and logging.
- 37 of the 420 frozen `core_ssl_benchmark` downstream and supervised runs
  (and 7 classical-baseline runs) carry a `ckpt_path` to `last.ckpt` in their
  exported config, i.e. they were resumed after an interruption: `miiv` seed
  123 (20), `eicu` seed 42 (10), `combined` seed 42 (7).
- The converged probe differs from the benchmark probe in optimiser (L-BFGS to
  convergence vs AdamW for at most 50 epochs), regularisation (tuned L2 vs
  fixed weight decay 0.05), input scaling (standardised vs raw embeddings) and
  label smoothing (none vs 0.1). Class weighting is the same.
- The Huber probe uses a fixed delta of 0.01 days rather than an exact L1 loss.
- Normalisation statistics for the Combined seeds other than 42 were not cached
  in the main checkout and were recomputed from the same parquet files and
  train split.
- `.gitignore` ignores `/results/*` except the three existing packages, so this
  directory is untracked until an allow rule is added.
- Bulky intermediates (embedding caches, probe cell JSON, run directories,
  checkpoints) are under the gitignored `outputs/ml4h-rebuttal/`.

## Files

`checksum_manifest.json` lists every file with row/column counts and SHA256.

- `p1_converged_probe_per_seed.csv`, `p1_converged_probe_aggregated.csv`,
  `p1_objective_ranking.csv`, `p1_paired_deltas_mae_vs_others.csv`,
  `p1_los_loss_matched.csv`, `p1_regularisation_selection.csv`,
  `p1_regularisation_path_validation.csv` (validation metrics only)
- `p1_validity_*.csv`, `p1_validity_summary.json`
- `p2_pretrain_compute_by_paradigm_dataset.csv`, `p2_pretrain_compute_per_run.csv`,
  `p2_pretrain_compute_manifest.json`, `p2_flops_per_objective.csv`,
  `p2_local_cpu_step_timing.csv` (Mac CPU)
- `p2_l4_exclusive_step_timing.csv` (VM; miiv, 300 timed steps),
  `p2_l4_exclusive_step_timing_{miiv_2000steps,eicu,combined}.csv` (VM),
  `p2_l4_exclusive_step_timing_miiv_workers8_2000steps.csv` (VM; 8 data-loader
  workers as in the thesis pretraining config)
- `p3_checkpoint_per_source_per_run.csv` (VM; one row per run and portion
  `all`/`miiv`/`eicu`, with the frozen overall metrics and the difference to
  them), `p3_checkpoint_per_source_aggregated.csv` (5-seed mean and sd),
  `p3_checkpoint_per_source_summary.json`
- `p3_local_converged_probe_per_source_{per_seed,aggregated,paired_deltas}.csv`
  (Mac)
- `p4_head_reproduction_{downstream,pretrain,pretrain_curves,config_parity}.csv`,
  `p4_head_reproduction_summary.json`
- `p4b_head_downstream_from_frozen_encoders{,_config_parity}.csv`,
  `p4b_head_downstream_from_frozen_encoders_summary.json`
