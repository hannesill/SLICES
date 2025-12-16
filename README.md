# SLICES (BETA): The SSL Framework for ICU embeddings

<p align="center">
  <img src=".github/SLICES-logo.png" alt="SLICES Logo" width="320"/>
</p>

<!--- Badges and Tags --->

<p align="center">
  <img src="https://img.shields.io/badge/PyPI--uv-blueviolet?style=flat-square" alt="uv Package Manager"/>
  <img src="https://img.shields.io/badge/pytorch-2.0+-e07c24?style=flat-square&logo=pytorch" alt="PyTorch Version"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/tested%20with-pytest-6fa8dc?style=flat-square" alt="Tested with pytest"/>
</p>

<p align="center">
  <b>Tags:</b>
  <code>ICU</code> <code>Self-Supervised Learning</code> <code>PyTorch</code> <code>Time Series</code> <code>Benchmark</code> <code>Healthcare AI</code> <code>DuckDB</code> <code>Polars</code> <code>Lightning</code>
</p>



A benchmark framework for learning universal patient embeddings from unlabeled ICU time-series data using self-supervised learning (SSL). The learned embeddings transfer across clinical prediction tasks (mortality, length of stay, AKI, sepsis) and institutions.

## Overview

SLICES provides a complete pipeline for:

1. **Data Conversion** (optional): Convert CSV.gz files to Parquet format for efficient queries
2. **Feature Extraction**: Extract ICU time-series data from local MIMIC-IV Parquet files using DuckDB
3. **Preprocessing**: Convert raw events into hourly-binned dense tensors with observation masks
4. **SSL Pretraining**: Train self-supervised models (MAE, JEPA, contrastive) on unlabeled data
5. **Downstream Evaluation**: Fine-tune and evaluate on clinical prediction tasks

**Key Design Choice**: Works entirely with local files. Users can start with either CSV or Parquet files. No cloud credentials or API keys required.

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

SLICES supports two starting points: **CSV files** or **Parquet files**.

### Option A: Starting with CSV Files (Two-Step Process)

If you have MIMIC-IV in CSV.gz format from PhysioNet:

**Step 1: Convert CSV to Parquet**
```bash
python scripts/convert_csv_to_parquet.py \
    data.csv_root=/path/to/mimic-iv-3.0 \
    data.parquet_root=/path/to/mimic-iv-parquet
```

**Step 2: Extract Features**
```bash
python scripts/extract_mimic_iv.py \
    data.parquet_root=/path/to/mimic-iv-parquet
```

**Convenience shortcut** (runs both steps):
```bash
python scripts/setup_mimic_iv.py data.csv_root=/path/to/mimic-iv-3.0
```

### Option B: Starting with Parquet Files (Direct Extraction)

If you already have MIMIC-IV in Parquet format:

```bash
python scripts/extract_mimic_iv.py \
    data.parquet_root=/path/to/mimic-iv-parquet
```

### Next Steps

**Pretrain SSL Model**
```bash
python scripts/pretrain.py data.parquet_root=/path/to/mimic-iv-parquet
```

**Fine-tune on Downstream Task**
```bash
uv run python scripts/finetune.py checkpoint=outputs/encoder.pt
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
│       │   │   ├── base.py       # Abstract base encoder
│       │   │   ├── factory.py    # Encoder factory
│       │   │   └── transformer.py # Transformer encoder
│       │   ├── pretraining/      # SSL objectives
│       │   │   ├── base.py       # Abstract SSL objective
│       │   │   ├── factory.py    # SSL objective factory
│       │   │   └── mae.py        # MAE objective
│       │   └── heads/            # Task heads (for finetuning)
│       │       ├── base.py       # Abstract BaseTaskHead
│       │       ├── factory.py    # Task head factory
│       │       └── mlp.py        # MLP and Linear task heads
│       └── training/              # Training utilities and Lightning modules
│           ├── pretrain_module.py # SSLPretrainModule
│           ├── finetune_module.py # FineTuneModule
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
│   ├── convert_csv_to_parquet.py # Convert CSV.gz to Parquet
│   ├── setup_mimic_iv.py         # Convenience: convert + extract
│   ├── extract_mimic_iv.py       # Extract features from Parquet
│   ├── pretrain.py
│   └── finetune.py
└── tests/                         # Tests (outside src/)
```

## Data Format

Extracted ICU stays are stored as **separate Parquet files** in the output directory:

- `static.parquet` - Stay-level metadata (demographics, admission info)
  - Columns: `stay_id`, `patient_id`, `age`, `gender`, `race`, `admission_type`, `los_days`, etc.
- `timeseries.parquet` - Dense hourly-binned time-series with masks
  - Columns: `stay_id`, `timeseries` (nested array<float32>, shape T×D), `mask` (nested array<bool>, shape T×D)
- `labels.parquet` - Task labels
  - Columns: `stay_id`, plus one column per task (e.g., `mortality_24h`, `mortality_48h`)
- `metadata.yaml` - Feature names, sequence length, task names, etc.

The `ICUDataset` loads these files and returns dictionaries with:
- `timeseries`: FloatTensor of shape (seq_length, n_features)
- `mask`: BoolTensor of shape (seq_length, n_features) - True = observed, False = missing/imputed
- `static`: Dict with static features (age, gender, etc.)
- `label`: FloatTensor with task label (if task_name specified)

## Configuration

SLICES uses [Hydra](https://hydra.cc/) for configuration management. All configs are in the `configs/` directory.

### Data Paths Configuration

Edit `configs/data/mimic_iv.yaml` or override via command line:

```yaml
# Optional: Path to raw CSV.gz files (if starting from CSVs)
csv_root: null  # e.g., /data/mimic-iv-3.0

# Required: Path to Parquet files (used by extraction)
parquet_root: data/parquet/mimic-iv-demo

# Output for extracted features
output_dir: data/processed/mimic-iv-demo
```

**Command-line overrides:**
```bash
# Convert CSVs
python scripts/convert_csv_to_parquet.py \
    data.csv_root=/path/to/csv \
    data.parquet_root=/path/to/parquet

# Extract from Parquet
python scripts/extract_mimic_iv.py \
    data.parquet_root=/path/to/parquet
```

### Expected Directory Structures

**CSV format** (from PhysioNet):
```
mimic-iv-3.0/
├── hosp/
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   └── labevents.csv.gz
├── icu/
│   ├── icustays.csv.gz
│   ├── chartevents.csv.gz
│   └── inputevents.csv.gz
└── ...
```

**Parquet format** (after conversion or if pre-converted):
```
mimic-iv-parquet/
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

### Environment Variables for Conversion

Fine-tune CSV-to-Parquet conversion performance:

```bash
# Maximum parallel workers (default: 4)
export SLICES_CONVERT_MAX_WORKERS=8

# DuckDB memory limit per worker (default: 3GB)
export SLICES_DUCKDB_MEM=4GB

# DuckDB threads per worker (default: 2)
export SLICES_DUCKDB_THREADS=4
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
