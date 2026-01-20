"""Entry point for extracting MIMIC-IV data from Parquet files.

Example usage:
    # Use paths from config (defaults from configs/data/mimic_iv.yaml)
    uv run python scripts/preprocessing/extract_mimic_iv.py

    # Override parquet path
    uv run python scripts/preprocessing/extract_mimic_iv.py data.parquet_root=/path/to/parquet

    # Override output path
    uv run python scripts/preprocessing/extract_mimic_iv.py data.processed_dir=/path/to/output

    # Specify different tasks
    uv run python scripts/preprocessing/extract_mimic_iv.py \
        'data.tasks=[mortality_24h,mortality_hospital]'

    # Use full MIMIC-IV with custom output
    uv run python scripts/preprocessing/extract_mimic_iv.py \
        data.parquet_root=/path/to/mimic-iv \
        data.processed_dir=data/processed/mimic-iv-full
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract MIMIC-IV data from local Parquet files.

    Args:
        cfg: Hydra configuration object. Uses cfg.data.* for all settings:
            - data.parquet_root: Path to Parquet files
            - data.processed_dir: Path for processed output
            - data.seq_length_hours: Sequence length in hours
            - data.feature_set: Feature set to extract
            - data.tasks: List of task names for label extraction
    """
    # Validate parquet root exists
    parquet_root = Path(cfg.data.parquet_root)

    if not parquet_root.exists():
        print(f"Error: Parquet root directory not found: {parquet_root}")
        print("\nIf you have CSV files, run conversion first:")
        print(
            "  uv run python scripts/preprocessing/convert_csv_to_parquet.py "
            "data.csv_root=/path/to/csv"
        )
        print("\nOr provide the correct path:")
        print(
            "  uv run python scripts/preprocessing/extract_mimic_iv.py "
            "data.parquet_root=/path/to/parquet"
        )
        sys.exit(1)

    # Build ExtractorConfig from unified data config
    extractor_config = ExtractorConfig(
        parquet_root=str(cfg.data.parquet_root),
        output_dir=str(cfg.data.processed_dir),
        seq_length_hours=cfg.data.seq_length_hours,
        feature_set=cfg.data.feature_set,
        min_stay_hours=cfg.data.min_stay_hours,
        batch_size=cfg.data.extraction_batch_size,
        tasks=list(cfg.data.tasks) if cfg.data.tasks else ["mortality_24h"],
        categories=list(cfg.data.categories) if cfg.data.get("categories") else None,
        concepts_dir=cfg.data.get("concepts_dir"),
        datasets_dir=cfg.data.get("datasets_dir"),
        tasks_dir=cfg.data.get("tasks_dir"),
    )

    # Create and run extractor
    extractor = MIMICIVExtractor(extractor_config)
    extractor.run()


if __name__ == "__main__":
    main()
