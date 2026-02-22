# SLICES: Self-Supervised Learning for Intensive Care Embeddings System

<p align="center">
  <img src=".github/SLICES-logo.png" alt="SLICES Logo" width="320"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyPI--uv-blueviolet?style=flat-square" alt="uv Package Manager"/>
  <img src="https://img.shields.io/badge/pytorch-2.0+-e07c24?style=flat-square&logo=pytorch" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/tested%20with-pytest-6fa8dc?style=flat-square" alt="Tested with pytest"/>
</p>

<p align="center">
  <code>ICU</code> <code>Self-Supervised Learning</code> <code>PyTorch</code> <code>Time Series</code> <code>Benchmark</code> <code>Clinical AI</code> <code>Polars</code> <code>Lightning</code>
</p>

**SLICES** is a benchmark framework for controlled comparison of self-supervised learning (SSL) paradigms on sparse, irregularly-sampled ICU time-series data.

## Research Question

How do the three major SSL paradigm families — **reconstruction** (masked autoencoding), **self-distillation** (JEPA), and **contrastive learning** — compare when applied to clinical time series under controlled conditions?

### The Comparison Triangle

| Comparison | What Varies | What It Tests |
|---|---|---|
| **MAE vs JEPA** | Input-space vs latent-space prediction | Same masking, same encoder input |
| **JEPA vs Contrastive** | Local positional prediction vs global invariance | Both operate in latent space |
| **MAE vs Contrastive** | Reconstruction vs discrimination | Opposite ends of the SSL spectrum |

### Why a New Benchmark?

Fair comparison of SSL objectives for clinical time series is currently impossible because published methods differ in preprocessing, cohort definitions, input modalities, and evaluation setups. SLICES standardizes everything that **can** be shared (data pipeline, encoder architecture, hyperparameter budget, evaluation protocol) and explicitly documents what **must** differ (tokenization strategy, encoder-data interface) as paradigm-intrinsic requirements.

**Key insight**: SSL paradigm choice is not just "swap the loss function" — it imposes structural requirements on how the encoder interfaces with sparse clinical data. This interaction between paradigm and data sparsity is itself a contribution.

## SSL Paradigms

All observation-level objectives share the same `ObservationTransformerEncoder` (one token per observed measurement) and the same masking logic (`masking.py`), differing only in what they predict and how they compute loss:

| Objective | Predicts | Target | Loss |
|---|---|---|---|
| **MAE** | Raw scalar values at masked positions | Input values | MSE |
| **JEPA** | Latent representations at masked positions | EMA target encoder representations | MSE / Cosine |
| **Contrastive** | Global embedding similarity across views | Positive pair agreement (NT-Xent) | Cross-entropy |

**SMART** (NeurIPS 2024) is also included in the codebase as a sanity check and to demonstrate the framework's extensibility — it uses its own MART encoder and element-wise masking, so it is not part of the controlled thesis experiments.

## Pipeline

```
RICU (R) ──→ Parquet ──→ ICUDataset ──→ SSL Pretraining ──→ Downstream Finetuning
  hourly-binned         dense tensors     MAE / JEPA /       mortality, LOS,
  feature extraction    + obs masks       Contrastive          AKI
  across datasets
```

1. **Extraction**: RICU (R package) harmonizes raw ICU data (MIMIC-IV, eICU) into hourly-binned parquet files
2. **Ingestion**: `RicuExtractor` reads RICU output into dense tensors + observation masks + labels
3. **Loading**: `ICUDataset` applies normalize-then-zero-fill imputation with patient-level splits
4. **Pretraining**: Config-driven SSL with any registered objective — switch paradigm by changing one config
5. **Evaluation**: Fine-tune on downstream clinical tasks with fairness metrics

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
git clone <repository-url>
cd SLICES
uv sync --dev
```

Verify:
```bash
uv run python -c "from slices.models.pretraining import JEPAObjective, ContrastiveObjective; print('OK')"
```

## Quick Start

### 1. Extract Features

```bash
uv run python scripts/preprocessing/extract_ricu.py \
    data.parquet_root=/path/to/ricu-output
```

### 2. Pretrain

Each SSL paradigm has its own entry config with matched training budgets:

```bash
# MAE (masked autoencoder — reconstruction baseline)
uv run python scripts/training/pretrain.py data.parquet_root=/path/to/data

# JEPA (latent-space prediction with EMA target encoder)
uv run python scripts/training/pretrain.py --config-name pretrain_jepa

# Contrastive (SimCLR-style with two masked views)
uv run python scripts/training/pretrain.py --config-name pretrain_contrastive

# SMART (sanity check / extensibility demo — not part of thesis experiments)
uv run python scripts/training/pretrain.py --config-name pretrain_smart
```

### 3. Fine-tune

```bash
uv run python scripts/training/finetune.py checkpoint=outputs/encoder.pt
```

### 4. Supervised Baseline

```bash
uv run python scripts/training/supervised.py data.parquet_root=/path/to/data
```

## Project Structure

```
SLICES/
├── src/slices/                     # Main package (src layout)
│   ├── data/
│   │   ├── extractors/             # Dataset-specific extraction
│   │   │   ├── base.py             # Abstract base extractor
│   │   │   └── ricu.py             # RICU-based extractor
│   │   ├── labels/                 # Task label builders (factory pattern)
│   │   ├── dataset.py              # ICUDataset (PyTorch Dataset)
│   │   ├── datamodule.py           # Lightning DataModule (patient-level splits)
│   │   ├── transforms.py           # SSL augmentations
│   │   └── sliding_window.py       # Sliding window utilities
│   ├── models/
│   │   ├── encoders/               # Backbone architectures (factory pattern)
│   │   │   ├── transformer.py      # Timestep-level Transformer
│   │   │   ├── observation_transformer.py  # Observation-level Transformer
│   │   │   └── smart.py            # SMART/MART encoder
│   │   ├── pretraining/            # SSL objectives (factory pattern)
│   │   │   ├── masking.py          # Shared observation masking
│   │   │   ├── mae.py              # Masked Autoencoder
│   │   │   ├── jepa.py             # Joint-Embedding Predictive Architecture
│   │   │   ├── contrastive.py      # SimCLR-style contrastive
│   │   │   └── smart.py            # SMART (sanity check / extensibility demo)
│   │   ├── heads/                  # Task heads (MLP, Linear)
│   │   └── common.py               # Shared utilities
│   ├── training/
│   │   ├── pretrain_module.py      # SSLPretrainModule (Lightning)
│   │   ├── finetune_module.py      # FineTuneModule (Lightning)
│   │   └── utils.py                # Optimizer/scheduler builders
│   └── eval/
│       ├── metrics.py              # AUROC, AUPRC, F1, MSE, etc.
│       ├── fairness.py             # Per-group AUROC, demographic parity
│       └── imputation.py           # SSL reconstruction quality
├── configs/                        # Hydra configs
│   ├── pretrain.yaml               # MAE pretraining
│   ├── pretrain_jepa.yaml          # JEPA pretraining
│   ├── pretrain_contrastive.yaml   # Contrastive pretraining
│   ├── pretrain_smart.yaml         # SMART pretraining
│   ├── finetune.yaml
│   ├── supervised.yaml
│   ├── data/                       # Dataset configs
│   ├── model/                      # Encoder configs
│   ├── ssl/                        # SSL objective configs (mae, jepa, contrastive, smart)
│   └── tasks/                      # Downstream task definitions
├── scripts/                        # Entry point scripts
│   ├── preprocessing/
│   └── training/
└── tests/                          # pytest test suite (940+ tests)
```

## Configuration

SLICES uses [Hydra](https://hydra.cc/) for configuration. Switch SSL paradigm by changing one config default:

```bash
# Override SSL objective on the fly
uv run python scripts/training/pretrain.py ssl=jepa model=observation_transformer

# Override hyperparameters
uv run python scripts/training/pretrain.py --config-name pretrain_jepa ssl.mask_ratio=0.5

# Smoke test any config
uv run python scripts/training/pretrain.py --config-name pretrain_jepa training.overfit_batches=2 --cfg job
```

### Config Groups

| Group | Options | Purpose |
|---|---|---|
| `ssl/` | `mae`, `jepa`, `contrastive`, `smart` | SSL objective hyperparameters |
| `model/` | `transformer`, `observation_transformer`, `smart` | Encoder architecture |
| `data/` | `ricu` | Dataset and paths |
| `tasks/` | `mortality`, `los`, `aki`, ... | Downstream task definitions |

## Data Format

Extracted ICU stays are stored as Parquet files:

- `static.parquet` — Stay-level demographics and admission info
- `timeseries.parquet` — Dense hourly-binned time-series with observation masks (T x D)
- `labels.parquet` — Task labels (mortality, LOS, AKI, etc.)
- `metadata.yaml` — Feature names, sequence length, task definitions

`ICUDataset` returns batches with:
- `timeseries`: `FloatTensor (B, T, D)` — hourly-binned feature values
- `mask`: `BoolTensor (B, T, D)` — True = observed, False = missing
- `label`: `FloatTensor` — task label (when task specified)

## Development

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=slices --cov-report=html --cov-report=term

# Format / lint / type check
black src/ scripts/ tests/
ruff check src/ scripts/ tests/
mypy src/
```

## Key Design Decisions

- **RICU-based extraction**: Data harmonization across datasets handled by RICU (R). Python reads the output.
- **Normalize-then-zero-fill**: Single imputation strategy (z-normalize, fill missing with 0). Eliminates imputation as a confound.
- **Observation masks**: Missingness tracked separately; SSL objectives use this for masking.
- **Shared masking**: MAE, JEPA, and Contrastive share identical masking code (`masking.py`) for fair comparison.
- **Patient-level splits**: No data leakage between train/val/test.
- **Config-driven ablations**: Change one YAML default to switch paradigm, encoder, or task.

## Extending SLICES

<!-- TODO: Create EXTENDING_SLICES.md documenting how to add new downstream tasks, SSL objectives, encoder architectures, and datasets using the existing factory patterns. -->

The framework uses factory patterns throughout, making it straightforward to add new components. See `EXTENDING_SLICES.md` (coming soon) for details on adding new downstream tasks, SSL objectives, encoder architectures, and datasets.

## References

- **MIMIC-IV**: Johnson, A. E. W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*.
- **eICU**: Pollard, T. J., et al. (2018). The eICU Collaborative Research Database. *Scientific Data*.
- **ricu**: Gygax, D. M., et al. (2023). ricu: R's interface to intensive care data. *GigaScience*.
- **MAE**: He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
- **I-JEPA**: Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*.
- **SimCLR**: Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
- **SMART**: Yu, Z., et al. (2024). SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction. *NeurIPS*.
- **YAIB**: Yèche, H., et al. (2024). YAIB: Yet Another ICU Benchmark. *ICLR*.

## License

See [LICENSE](LICENSE) file for details.
