"""Convenience script to setup MIMIC-IV data (convert + extract in one go).

This script automatically:
1. Converts CSV.gz to Parquet (if csv_root is provided)
2. Extracts features from Parquet files

Example usage:
    # Start with CSV files
    python scripts/setup_mimic_iv.py extraction.csv_root=/path/to/mimic-iv-csv

    # Start with Parquet files (skip conversion)
    python scripts/setup_mimic_iv.py extraction.parquet_root=/path/to/mimic-iv-parquet
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from slices.data.data_io import convert_csv_to_parquet
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Setup MIMIC-IV data pipeline (convert + extract).

    Args:
        cfg: Hydra configuration object. Expects:
            - extraction.csv_root: Optional path to CSV files
            - extraction.parquet_root: Path to/for Parquet files
            - extraction.output_dir: Path for processed output
    """
    csv_root_str = cfg.extraction.get("csv_root")
    parquet_root = Path(cfg.extraction.parquet_root)
    dataset_name = cfg.extraction.get("name", "dataset")

    print(f"=== SLICES Data Setup: {dataset_name} ===\n")

    # Step 1: Convert CSV to Parquet if needed
    if csv_root_str is not None:
        csv_root = Path(csv_root_str)

        if not csv_root.exists():
            print(f"Error: CSV root directory not found: {csv_root}")
            sys.exit(1)

        print("Step 1/2: Converting CSV to Parquet...")
        print(f"  CSV root: {csv_root}")
        print(f"  Parquet root: {parquet_root}")
        print()

        success = convert_csv_to_parquet(
            csv_root=csv_root,
            parquet_root=parquet_root,
            dataset_name=dataset_name,
        )

        if not success:
            print("\n✗ Conversion failed. Aborting.")
            sys.exit(1)

        print()
    else:
        # No CSV conversion needed
        if not parquet_root.exists():
            print(f"Error: Parquet root not found: {parquet_root}")
            print("\nPlease provide either:")
            print("  - extraction.csv_root=/path/to/csv (will convert to Parquet)")
            print("  - extraction.parquet_root=/path/to/existing/parquet")
            sys.exit(1)

        print("Step 1/2: Skipping CSV conversion (Parquet files already exist)")
        print(f"  Parquet root: {parquet_root}")
        print()

    # Step 2: Extract features
    print("Step 2/2: Extracting features from Parquet...")
    print(f"  Output directory: {cfg.extraction.output_dir}")
    print(f"  Sequence length: {cfg.extraction.seq_length_hours} hours")
    print(f"  Feature set: {cfg.extraction.feature_set}")
    print()

    # Convert Hydra config to ExtractorConfig
    # Only pass fields that ExtractorConfig accepts
    extractor_fields = {
        "parquet_root",
        "output_dir",
        "seq_length_hours",
        "feature_set",
        "concepts_dir",
        "tasks_dir",
        "tasks",
        "min_stay_hours",
    }

    extraction_config = OmegaConf.to_container(cfg.extraction, resolve=True)
    filtered_config = {k: v for k, v in extraction_config.items() if k in extractor_fields}

    # Handle tasks (may be a list in config)
    if "tasks" not in filtered_config or filtered_config["tasks"] is None:
        filtered_config["tasks"] = ["mortality_24h", "mortality_48h", "mortality_hospital"]

    # Create ExtractorConfig
    try:
        extractor_config = ExtractorConfig(**filtered_config)
    except ValueError as e:
        print(f"Error: Invalid extraction configuration: {e}")
        sys.exit(1)

    # Create and run extractor
    try:
        extractor = MIMICIVExtractor(extractor_config)
        extractor.run()
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

    print("\n=== Setup Summary ===")
    print(f"Parquet files: {parquet_root}")
    print(f"Processed data: {cfg.extraction.output_dir}")


if __name__ == "__main__":
    main()
