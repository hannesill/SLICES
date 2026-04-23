# SLICES Final Experiment Plan

This plan defines the final thesis rerun corpus for SLICES. The corpus is
organized by scientific experiment class, not by historical launch order.

Final release-ready W&B runs live in the final thesis project, default
`slices-thesis`, and use a mandatory revision tag such as `thesis-v1`.
Exploratory runs remain in older projects/outputs and are not mixed into final
export or fairness jobs.

## Benchmark Contract

SLICES compares SSL paradigms for sparse, irregular ICU time series under
controlled conditions.

Controlled SSL objectives:

- `mae`
- `jepa`
- `contrastive`

Controlled SSL objectives share:

- same RICU-derived data pipeline
- same canonical `TransformerEncoder`
- same obs-aware timestep tokenization
- same masking budget where configs intend parity
- same training budget unless the experiment class is an explicit ablation

Masking strategy is not forced to be identical across objectives. MAE uses
random timestep masking, JEPA uses block timestep masking to avoid the
random-mask interpolation failure observed during development, and Contrastive
uses two masked views. The controlled invariant is the mask budget/interface, not
the exact mask geometry.

`ts2vec_extension` is a temporal-contrastive extension. It provides context for
the contrastive result but is not one of the three core thesis vertices.

`smart_external_reference` is an appendix external reference because it uses
`SMARTEncoder` / MART and element-wise masking, not the shared controlled
encoder contract.

Supervised Transformer, GRU-D, XGBoost, and linear views are baselines or
contextual comparisons, not SSL paradigms.

## Fixed Benchmark Choices

Fixed preprocessing/evaluation choices are benchmark invariants and should live
in `src/slices/constants.py` when code needs them:

- 24 hour observation window and minimum stay
- patient-level splits, 70/15/15
- normalization stats from train split only
- normalize then zero-fill
- preserve observation mask separately from values
- block model-input leakage from `los_hosp`, `los_icu`, and `dur_var`
- default labels: `mortality_24h`, `mortality_hospital`, `aki_kdigo`,
  `los_remaining`

The optional ICU-mortality task `mortality` is not part of the primary final
matrix.

## Metadata Contract

Every final run must carry class-based metadata:

| Field | Meaning |
|---|---|
| `experiment_class` | Scientific experiment family |
| `experiment_subtype` | Optional finer-grained family, e.g. `lr_sensitivity` |
| `revision` | Rerun corpus identifier, e.g. `thesis-v1` |
| `phase` | `pretrain`, `finetune`, `supervised`, or `baseline` |
| `protocol` | `A` for linear probe, `B` for full finetune/full-training where applicable |
| `dataset` | `miiv`, `eicu`, or `combined` |
| `paradigm` | `mae`, `jepa`, `contrastive`, `supervised`, `gru_d`, `xgboost`, `ts2vec`, `smart` |
| `task` | Downstream task where applicable |
| `seed` | Random seed |

Final runs must not require historical launch-group metadata. W&B tags and
exports use `experiment_class:*` as the canonical family key.

Example run names:

```text
core_ssl_benchmark_pretrain_miiv_mae_seed42
core_ssl_benchmark_probe_miiv_mae_mortality_24h_seed42
label_efficiency_finetune_eicu_jepa_mortality_24h_seed123_frac010
hp_robustness_finetune_miiv_mae_mortality_24h_seed456_lr00005
```

Exact suffixes may differ, but the experiment class must be visible without
project-history knowledge.

## Experiment Classes

The final launched run budget keeps the controlled benchmark structure and uses
task-specific low-label fractions where fixed random splits are scientifically
defensible.

| Experiment class | Scope | Launched runs |
|---|---:|---:|
| `core_ssl_benchmark` | MAE, JEPA, contrastive, supervised Transformer; 3 datasets; 4 tasks; 5 seeds; Protocol A/B for SSL | 465 |
| `label_efficiency` | SSL Protocol A/B plus supervised Transformer at low-label fractions; 5 seeds | 1155 |
| `cross_dataset_transfer` | MIMIC-IV to eICU and eICU to MIMIC-IV; SSL Protocol B; 5 seeds | 120 |
| `hp_robustness` | LR robustness plus MAE/JEPA mask-ratio and contrastive view/mask sensitivity on MIMIC-IV `mortality_24h`; 5 seeds | 150 |
| `capacity_study` | Larger MAE and supervised Transformer encoders on MIIV `mortality_24h`; 5 seeds | 100 |
| `classical_baselines` | XGBoost and GRU-D full-label plus label-efficiency context; 5 seeds | 330 |
| `ts2vec_extension` | TS2Vec temporal contrastive extension; 3 datasets; 4 tasks; 5 seeds; Protocol A/B | 135 |
| `smart_external_reference` | SMART external SSL reference; 3 datasets; 4 tasks; 5 seeds; Protocol A/B | 135 |
| **Total** | Includes appendix SMART reference | **2590** |

The formal thesis corpus excluding `smart_external_reference` has 2455 launched
runs.

Low-label policy:

- `mortality_24h` starts at 5% labels. The 1% fixed random subsets can
  undersample positives under the final seeds.
- `mortality_hospital` gets the full 1%, 5%, 10%, 25%, 50% low-label curve.
- `aki_kdigo` and `los_remaining` remain contextual 10% low-label points.
- Capacity and classical-baseline `mortality_24h` slices also start at 5%.

## Derived Comparison Views

Some analysis plots need comparison rows that are not relaunched in the target
class. These are explicit export-time views, not launch-time tags:

- label-efficiency curves include 100 percent endpoints from
  `core_ssl_benchmark`
- capacity plots include default-size baselines from `core_ssl_benchmark` and
  `label_efficiency`
- classical baseline context compares `classical_baselines` against neural
  full-label and low-label rows
- TS2Vec comparison pairs `ts2vec_extension` against core `contrastive`

The canonical per-seed export keeps each run in its true `experiment_class`.

## Launching

Main final rerun:

```bash
uv run python scripts/internal/run_experiments.py run \
  --experiment-class \
    core_ssl_benchmark \
    label_efficiency \
    cross_dataset_transfer \
    hp_robustness \
    capacity_study \
    classical_baselines \
    ts2vec_extension \
    smart_external_reference \
  --project slices-thesis \
  --revision thesis-v1 \
  --launch-commit <reviewed-git-commit> \
  --entity <entity>
```

Use the reviewed git commit hash for `--launch-commit`. The tmux launcher fills
this from `HEAD` and refuses tracked dirty launches by default; direct launcher
invocations should be equally explicit so retries cannot mix same-revision
artifacts produced by different code.

Dry-run count check:

```bash
uv run python scripts/internal/run_experiments.py run \
  --experiment-class \
    core_ssl_benchmark \
    label_efficiency \
    cross_dataset_transfer \
    hp_robustness \
    capacity_study \
    classical_baselines \
    ts2vec_extension \
    smart_external_reference \
  --project slices-thesis \
  --revision thesis-v1 \
  --launch-commit <reviewed-git-commit> \
  --entity dummy \
  --dry-run
```

Expected dry-run count: 2590 launched runs.

## Fairness

Fairness is evaluated only on downstream-producing runs. Default classes:

```text
core_ssl_benchmark
label_efficiency
cross_dataset_transfer
hp_robustness
capacity_study
classical_baselines
ts2vec_extension
smart_external_reference
```

Run fairness after training:

```bash
uv run python scripts/eval/evaluate_fairness.py \
  --project slices-thesis \
  --revision thesis-v1 \
  --entity <entity> \
  --force
```

Revision filtering is mandatory to avoid mixing rerun corpora. Use `--force`
for final publication refreshes so old fairness summaries are replaced with
current artifact/source and evaluation-setting metadata.

## Export

Publication export:

```bash
uv run python scripts/export_results.py \
  --project slices-thesis \
  --revision thesis-v1 \
  --entity <entity> \
  --output-dir results/slices-thesis_thesis-v1
```

The export writes canonical per-run/aggregated tables plus derived comparison
views for label efficiency, capacity, classical context, and TS2Vec comparison.
Aggregated metric columns include seed mean, standard deviation, min/max, and
95% confidence intervals for finite seed values.

Do not use publication escape hatches such as `--allow-incomplete`,
`--allow-extraction-failures`, `--allow-duplicate-fingerprints`, or
`--allow-multiple-revisions` for final tables.

## Checkpoint Policy

SSL downstream runs use `encoder.pt`, the last encoder from the fixed pretraining
schedule. They intentionally do not use `encoder_best_val.pt`, because SSL
validation loss is not a reliable early-stopping criterion for every paradigm.
Downstream finetune and supervised runs still evaluate their best downstream
checkpoint when reporting test metrics, and record checkpoint provenance for
post-hoc fairness evaluation.

## Validation Checklist

Before launching final runs:

- generated matrix has exactly 2590 launched runs including SMART
- generated matrix has exactly 2455 launched runs excluding SMART
- every generated run has `experiment_class`
- no generated command contains historical launch-group overrides
- every generated command for a final rerun contains `revision=<revision>`
- every downstream SSL run has exactly one valid pretrain dependency
- supervised, GRU-D, and XGBoost runs have no pretrain dependency
- transfer runs depend on source-dataset core pretrains
- TS2Vec and SMART downstream runs depend on their own class pretrains
- no duplicate scientific fingerprints exist
- export groups by `experiment_class`
- fairness defaults are class-based and do not fetch pretraining runs
- final launch/export/fairness commands target the final W&B project
- final launch/retry commands include `--launch-commit`

Focused regression suite:

```bash
uv run pytest \
  tests/test_config_schemas.py \
  tests/test_export_results.py \
  tests/test_evaluate_fairness.py \
  tests/test_fixes.py
```

## Archival Note

Earlier internal planning used execution batches as a convenience while the
matrix was still being explored. Those batch labels are retired for the final
rerun corpus. The final system exposes experiment classes as the public API and
derives dependencies from the generated run graph.
