# SLICES Experiment Protocol

This protocol defines the public SLICES thesis benchmark. It describes the
controlled comparison, fixed preprocessing and evaluation choices, run metadata,
result export expectations, and validation checks for publication-scale reruns.

## Benchmark Contract

SLICES compares self-supervised learning (SSL) paradigms for sparse, irregular
ICU time series under controlled conditions.

The controlled SSL objectives are:

- `mae`
- `jepa`
- `contrastive`

These objectives share:

- the same RICU-derived data pipeline
- the same canonical `TransformerEncoder`
- the same obs-aware timestep tokenization
- the same masking budget where configs intend parity
- the same training budget unless the experiment class is an explicit ablation

Masking strategy is reported per objective. MAE uses random timestep masking,
JEPA uses block timestep masking, and Contrastive uses two masked views. The
controlled invariant is the masking budget and encoder interface, not identical
mask geometry.

`ts2vec_extension` is a temporal-contrastive extension. It provides context for
the contrastive result but is not one of the three core thesis vertices.

SMART is outside the controlled thesis corpus because it changes both the
encoder and objective contract: it uses `SMARTEncoder` / MART and element-wise
masking. The implementation remains useful as an example of adding a custom
model/objective to the SLICES framework, but SMART results are not thesis
evidence for the controlled SSL comparison.

Supervised Transformer, GRU-D, XGBoost, and linear views are baselines or
contextual comparisons, not SSL paradigms.

## Fixed Benchmark Choices

Fixed preprocessing and evaluation choices are benchmark invariants. Code-level
invariants belong in `src/slices/constants.py`; experiment knobs remain in Hydra
configs.

- 24 hour observation window and minimum stay
- patient-level splits, 70/15/15
- normalization statistics from the train split only
- normalize then zero-fill
- preserve the observation mask separately from values
- block model-input leakage from `los_hosp`, `los_icu`, and `dur_var`
- default labels: `mortality_24h`, `mortality_hospital`, `aki_kdigo`,
  `los_remaining`

The optional ICU-mortality task `mortality` is not part of the primary thesis
matrix.

## Metadata Contract

Final runs use class-based metadata so launch, export, fairness, and analysis
jobs do not depend on historical launch groups.

| Field | Meaning |
|---|---|
| `experiment_class` | Scientific experiment family |
| `experiment_subtype` | Optional finer-grained family, e.g. `lr_sensitivity` |
| `revision` | Rerun corpus identifier, e.g. `thesis-v1` |
| `phase` | `pretrain`, `finetune`, `supervised`, or `baseline` |
| `protocol` | `linear_probe` or `full_finetune` where applicable |
| `dataset` | `miiv`, `eicu`, or `combined` |
| `paradigm` | `mae`, `jepa`, `contrastive`, `supervised`, `gru_d`, `xgboost`, `ts2vec` |
| `task` | Downstream task where applicable |
| `seed` | Random seed |

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

The thesis run budget keeps the controlled benchmark structure and uses
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
| **Total** | Thesis corpus excluding SMART | **2455** |

Low-label policy:

- `mortality_24h` starts at 5% labels.
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
  --project slices-thesis \
  --revision thesis-v1 \
  --launch-commit <reviewed-git-commit> \
  --entity dummy \
  --dry-run
```

Expected dry-run count: 2455 launched runs.

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

The export writes canonical per-run and aggregated tables plus derived
comparison views for label efficiency, capacity, classical context, and TS2Vec
comparison. Aggregated metric columns include seed mean, standard deviation,
min/max, and 95% confidence intervals for finite seed values.

Do not use publication escape hatches such as `--allow-incomplete`,
`--allow-extraction-failures`, `--allow-duplicate-fingerprints`, or
`--allow-multiple-revisions` for final tables.

## Checkpoint Policy

SSL downstream runs use `encoder.pt`, the last encoder from the fixed
pretraining schedule. They intentionally do not use `encoder_best_val.pt`,
because SSL validation loss is not a reliable early-stopping criterion for every
paradigm. Downstream finetune and supervised runs still evaluate their best
downstream checkpoint when reporting test metrics, and record checkpoint
provenance for post-hoc fairness evaluation.

## Validation Checklist

Before launching final runs:

- generated matrix has exactly 2455 launched thesis runs
- every generated run has `experiment_class`
- no generated command contains historical launch-group overrides
- every generated command for a final rerun contains `revision=<revision>`
- every downstream SSL run has exactly one valid pretrain dependency
- supervised, GRU-D, and XGBoost runs have no pretrain dependency
- transfer runs depend on source-dataset core pretrains
- TS2Vec downstream runs depend on their own class pretrains
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
