# SLICES: Self-Supervised Learning for Intensive Care Embeddings

<p align="center">
  <img src=".github/SLICES-logo.png" alt="SLICES logo" width="320"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776ab?style=flat-square&logo=python" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-e07c24?style=flat-square&logo=pytorch" alt="PyTorch 2.0+"/>
  <img src="https://img.shields.io/badge/config-Hydra-89b4fa?style=flat-square" alt="Hydra"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT license"/>
</p>

<p align="center">
  <code>ICU time series</code> <code>self-supervised learning</code> <code>clinical AI</code> <code>PyTorch Lightning</code> <code>RICU</code> <code>Hydra</code>
</p>

**SLICES** is a benchmark and PyTorch codebase for comparing self-supervised
learning objectives on sparse, irregular ICU time series.

The core question is narrow by design: when the data pipeline, encoder,
tokenization, training budget, and downstream protocols are held fixed, how do
**MAE**, **JEPA**, and **contrastive learning** differ?

<p align="center">
  <img src="docs/assets/benchmark-overview.svg" alt="SLICES benchmark overview" width="860"/>
</p>

## Status

- Master's thesis benchmark code is public.
- Patient data is not included. Reproduction requires credentialed MIMIC-IV and
  eICU access.
- The frozen thesis result export is not bundled in this repository yet.
- The public static result bundle, citation metadata, and paper-style summary are
  planned after thesis finalization.

## What Is Included

| Area | Scope |
|---|---|
| Data pipeline | RICU extraction, hourly ICU tensors, observation masks, train-only normalization |
| Core SSL objectives | MAE, JEPA, contrastive |
| Extension objective | TS2Vec temporal contrastive learning |
| External reference | SMART implementation, outside the controlled thesis comparison |
| Encoders | Observation-aware Transformer, GRU-D, SMART/MART, linear |
| Downstream tasks | 24h mortality, hospital mortality, AKI KDIGO, remaining LOS |
| Protocols | Linear probing and full fine-tuning |
| Baselines | Supervised Transformer, GRU-D, XGBoost |
| Evaluation | AUROC, AUPRC, calibration, regression metrics, fairness summaries, paired tests |

SLICES is not a clinical deployment model, a public patient-data release, or a
replacement for broad ICU benchmarks such as YAIB. It is focused on objective
attribution for SSL in sparse clinical time series.

## Result Snapshot

These claims are from the frozen thesis export
`results/slices-thesis-v1`, which is currently local and not yet bundled:

- In the core SSL benchmark, MAE is the best or tied-best SSL method on most
  classification comparisons, especially under linear probing.
- In pooled classification tests, MAE significantly exceeds JEPA and contrastive
  under several core settings, but MAE is not significantly different from the
  supervised Transformer under full fine-tuning.
- No per-task core SSL comparison is statistically significant after the thesis
  correction rules; task-level claims should be read descriptively.
- LOS behaves differently from classification: contrastive often has the best
  frozen linear-probe `test/mae`.
- In the label-efficiency analysis, MAE has the best mean in `35/45`
  full-finetune cells and `37/45` linear-probe cells.

<p align="center">
  <img src="docs/assets/label-efficiency-curves.png" alt="SLICES label-efficiency curves" width="860"/>
</p>

## Install

Prerequisites:

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- R with access to the source ICU datasets for extraction

```bash
git clone https://github.com/hannesill/SLICES.git
cd SLICES
uv sync --dev
```

Smoke test:

```bash
uv run python -c "from slices.models.pretraining import MAEObjective, JEPAObjective, ContrastiveObjective; print('OK')"
```

## Quick Start

### 1. Build Processed Data

Run the RICU extraction once per credentialed source dataset:

```bash
Rscript scripts/preprocessing/extract_with_ricu.R --dataset miiv
Rscript scripts/preprocessing/extract_with_ricu.R --dataset eicu
```

Convert RICU output into SLICES tensors:

```bash
uv run python scripts/preprocessing/build_processed_data.py --datasets miiv eicu --combined
```

This creates `data/processed/{miiv,eicu,combined}` with:

- `static.parquet`
- `timeseries.parquet`
- `labels.parquet`
- `metadata.yaml`

### 2. Pretrain an SSL Encoder

Pick the objective with `ssl=`. Disable W&B for local smoke runs unless you have
configured an entity.

```bash
# MAE
uv run python scripts/training/pretrain.py dataset=miiv ssl=mae logging.use_wandb=false

# JEPA
uv run python scripts/training/pretrain.py dataset=miiv ssl=jepa logging.use_wandb=false

# Contrastive
uv run python scripts/training/pretrain.py dataset=miiv ssl=contrastive logging.use_wandb=false
```

### 3. Fine-Tune

```bash
# Linear probe
uv run python scripts/training/finetune.py \
  dataset=miiv \
  checkpoint=outputs/2026-07-01/12-00-00/encoder.pt \
  protocol=linear_probe \
  logging.use_wandb=false

# Full fine-tuning
uv run python scripts/training/finetune.py \
  dataset=miiv \
  checkpoint=outputs/2026-07-01/12-00-00/encoder.pt \
  protocol=full_finetune \
  logging.use_wandb=false
```

Switch tasks with `tasks=mortality_hospital`, `tasks=aki_kdigo`, or
`tasks=los_remaining`.

### 4. Run Baselines

```bash
uv run python scripts/training/supervised.py dataset=miiv logging.use_wandb=false
uv run python scripts/training/xgboost_baseline.py dataset=miiv
```

### 5. Validate

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run mypy src
```

## Configuration

SLICES uses Hydra. Main config groups:

| Group | Options |
|---|---|
| `ssl/` | `mae`, `jepa`, `contrastive`, `ts2vec`, `smart` |
| `model/` | `transformer`, `transformer_medium`, `transformer_large`, `gru_d`, `smart`, `linear` |
| `tasks/` | `mortality_24h`, `mortality_hospital`, `aki_kdigo`, `los_remaining` |
| `protocol/` | `linear_probe`, `full_finetune` |
| `data/` | `ricu` |

Examples:

```bash
# Change objective
uv run python scripts/training/pretrain.py dataset=eicu ssl=jepa

# Override objective hyperparameters
uv run python scripts/training/pretrain.py dataset=miiv ssl=jepa ssl.mask_ratio=0.75

# Inspect the resolved config
uv run python scripts/training/pretrain.py dataset=miiv ssl=mae --cfg job
```

## Benchmark Contract

The controlled comparison fixes:

- RICU-derived preprocessing
- 24h observation window
- patient-level splits
- train-only normalization
- normalize-then-zero-fill imputation
- separate observation masks
- observation-aware Transformer encoder
- downstream tasks and primary metrics
- five-seed evaluation where present

MAE, JEPA, and contrastive learning share the encoder interface and masking
budget. They do not use identical mask geometry: MAE uses random timestep masks,
JEPA uses block timestep masks, and contrastive learning uses two masked views.

Full thesis reruns and export rules are documented in
[docs/experiment_protocol.md](docs/experiment_protocol.md).

## Repository Layout

```text
SLICES/
+-- configs/                  # Hydra configs
+-- docs/                     # Public protocol and evaluation docs
+-- scripts/
|   +-- preprocessing/        # RICU extraction and tensor preparation
|   +-- training/             # SSL, supervised, GRU-D, XGBoost entry points
|   +-- eval/                 # Fairness and imputation evaluation
|   +-- internal/             # Thesis-scale launch helpers
+-- src/slices/
|   +-- data/                 # datasets, labels, splits, transforms
|   +-- models/               # encoders, heads, SSL objectives
|   +-- training/             # Lightning modules and training utilities
|   +-- eval/                 # metrics, fairness, statistical tests
+-- tests/                    # pytest suite
```

## Roadmap

- [x] Controlled MAE, JEPA, and contrastive SSL objectives.
- [x] RICU-based MIMIC-IV/eICU preprocessing pipeline.
- [x] Linear-probe and full-finetune downstream protocols.
- [x] Supervised Transformer, GRU-D, XGBoost, TS2Vec, and SMART implementations.
- [x] Fairness summaries and paired statistical export utilities.
- [ ] Publish frozen thesis result bundle and provenance manifest.
- [ ] Add `CITATION.cff` after thesis/paper citation details are stable.
- [ ] Add a small public synthetic-data fixture for end-to-end smoke tests.
- [ ] Add a notebook-style walkthrough for adding a new SSL objective.
- [ ] Package thesis figures/tables as a lightweight result report.
- [ ] Add Docker or Apptainer environment for cluster reproduction.

## References

- Johnson et al. (2023), MIMIC-IV, *Scientific Data*.
- Pollard et al. (2018), eICU Collaborative Research Database, *Scientific Data*.
- Gygax et al. (2023), ricu, *GigaScience*.
- He et al. (2022), Masked Autoencoders, *CVPR*.
- Assran et al. (2023), I-JEPA, *CVPR*.
- Chen et al. (2020), SimCLR, *ICML*.
- Yu et al. (2024), SMART, *NeurIPS*.
- Yeche et al. (2024), YAIB, *ICLR*.

## License

SLICES is released under the MIT License. See [LICENSE](LICENSE).
