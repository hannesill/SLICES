# SLICES: Self-Supervised Learning for Intensive Care Embeddings System

A benchmark framework for learning universal patient embeddings from unlabeled ICU time-series data using self-supervised learning (SSL). The learned embeddings transfer across clinical prediction tasks (mortality, length of stay, AKI, sepsis) and institutions.

## Overview

SLICES provides a complete pipeline for:

1. **Data Extraction**: Extract ICU time-series data from local MIMIC-IV Parquet files using DuckDB
2. **Preprocessing**: Convert raw events into hourly-binned dense tensors with observation masks
3. **SSL Pretraining**: Train self-supervised models (MAE, JEPA, contrastive) on unlabeled data
4. **Downstream Evaluation**: Fine-tune and evaluate on clinical prediction tasks

**Key Design Choice**: Users only need to specify their `data_dir` path pointing to local MIMIC-IV Parquet files. No cloud credentials or API keys required.

## Installation

This project uses the **src layout** (Python packaging best practice) and **uv** for dependency management.

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SLICES
```

2. Install the package in editable mode with development dependencies:
```bash
uv sync --dev
```

This will:
- Create a virtual environment
- Install all dependencies
- Install the package in editable mode (`-e`)

3. Verify installation:
```bash
python -c "from slices.data.extractors.base import BaseExtractor; print('Import successful!')"
```

## Quick Start

### 1. Extract MIMIC-IV Data

First, ensure you have MIMIC-IV Parquet files locally. Then run:

```bash
python scripts/extract_mimic_iv.py data.data_dir=/path/to/mimic-iv
```

This will extract and preprocess ICU stays into the format needed for training.

### 2. Pretrain SSL Model

```bash
python scripts/pretrain.py data.data_dir=/path/to/mimic-iv
```

### 3. Fine-tune on Downstream Task

```bash
python scripts/finetune.py task=mortality checkpoint=/path/to/pretrained.ckpt
```

## Project Structure

```
slices/                          # Repository root
├── src/
│   └── slices/                  # Main package (src layout)
│       ├── data/
│       │   ├── extractors/       # Dataset-specific extraction
│       │   │   └── base.py       # Abstract base extractor
│       │   ├── concepts/         # Concept dictionary YAMLs
│       │   ├── dataset.py        # PyTorch Dataset
│       │   ├── datamodule.py     # Lightning DataModule
│       │   └── transforms.py     # SSL augmentations
│       ├── models/
│       │   ├── encoders/         # Backbone architectures
│       │   │   └── base.py       # Abstract base encoder
│       │   └── pretraining/      # SSL objectives
│       │       └── base.py       # Abstract SSL objective
│       ├── tasks/                # Downstream task heads
│       │   └── base.py           # Abstract task head
│       └── training/             # Training utilities
│           └── utils.py
├── configs/                      # Hydra configs (outside src/)
│   ├── config.yaml               # Main config
│   ├── data/
│   │   └── mimic_iv.yaml
│   ├── model/
│   │   └── transformer.yaml
│   └── concepts/
│       └── core_features.yaml    # Concept dictionary
├── scripts/                       # Entry points (outside src/)
│   ├── extract_mimic_iv.py
│   ├── pretrain.py
│   └── finetune.py
└── tests/                         # Tests (outside src/)
```

## Data Format

ICU stays are stored as Parquet files with the following schema:

- `stay_id: int64` - Unique ICU stay identifier
- `patient_id: int64` - Patient identifier (for patient-level splits)
- `timeseries: array<float32>` - Shape (T, D), hourly binned features
- `mask: array<bool>` - Shape (T, D), True = observed, False = imputed
- `feature_names: list<str>` - Column names for timeseries
- `static: dict` - Age, gender, admission type, etc.
- `labels: dict` - Task labels (mortality_48h, los_days, aki_stage, etc.)

## Configuration

SLICES uses [Hydra](https://hydra.cc/) for configuration management. All configs are in the `configs/` directory.

### Setting Data Directory

The data directory must be set via command line:

```bash
python scripts/extract_mimic_iv.py data.data_dir=/path/to/mimic-iv
```

Expected MIMIC-IV structure:
```
mimic-iv/
├── hosp/
│   ├── patients.parquet
│   ├── admissions.parquet
│   └── labevents.parquet
├── icu/
│   ├── icustays.parquet
│   ├── chartevents.parquet
│   └── inputevents.parquet
└── ...
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ scripts/ tests/

# Lint
ruff check src/ scripts/ tests/

# Type check
mypy src/
```

## References

- **MIMIC-IV**: Johnson, A. E. W., et al. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data.
- **ricu**: Gygax, D. M., et al. (2023). ricu: R's interface to intensive care data. GigaScience.
- **YAIB**: Yèche, H., et al. (2024). YAIB: Yet Another ICU Benchmark. ICLR.
- **ICareFM**: [Preprint] Self-supervised learning for ICU time-series.

## License

See [LICENSE](LICENSE) file for details.

## Contributing

This is a master's thesis project. Contributions welcome via issues and pull requests.

