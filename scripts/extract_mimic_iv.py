"""Entry point for extracting MIMIC-IV data from Parquet files.

Example usage:
    # Use paths from config
    python scripts/extract_mimic_iv.py

    # Override parquet path
    python scripts/extract_mimic_iv.py data.parquet_root=/path/to/mimic-iv-parquet
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from slices.data.extractors.base import ExtractorConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract MIMIC-IV data from local Parquet files.
    
    Args:
        cfg: Hydra configuration object. Expects:
            - data.parquet_root: Path to Parquet files
            - data.output_dir: Path for processed output
            - data.seq_length_hours: Sequence length
            - data.feature_set: Feature set to extract
    """
    parquet_root = Path(cfg.data.parquet_root)
    
    if not parquet_root.exists():
        print(f"Error: Parquet root directory not found: {parquet_root}")
        print("\nIf you have CSV files, run conversion first:")
        print("  python scripts/convert_csv_to_parquet.py data.csv_root=/path/to/csv")
        print("\nOr provide the correct path:")
        print("  python scripts/extract_mimic_iv.py data.parquet_root=/path/to/parquet")
        sys.exit(1)
    
    print(f"Dataset: {cfg.data.name}")
    print(f"Parquet root: {parquet_root}")
    print(f"Output directory: {cfg.data.output_dir}")
    print(f"Sequence length: {cfg.data.seq_length_hours} hours")
    print(f"Feature set: {cfg.data.feature_set}")
    print()
    
    # TODO: Create MIMICIVExtractor instance and run extraction
    # extractor = MIMICIVExtractor(ExtractorConfig(**cfg.data))
    # extractor.run()
    print("TODO: Implement MIMIC-IV extraction")
    print("(MIMICIVExtractor not yet implemented)")


if __name__ == "__main__":
    main()

