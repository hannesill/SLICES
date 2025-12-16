"""Convenience script to setup MIMIC-IV data (convert + extract in one go).

This script automatically:
1. Converts CSV.gz to Parquet (if csv_root is provided)
2. Extracts features from Parquet files

Example usage:
    # Start with CSV files
    python scripts/setup_mimic_iv.py data.csv_root=/path/to/mimic-iv-csv

    # Start with Parquet files (skip conversion)
    python scripts/setup_mimic_iv.py data.parquet_root=/path/to/mimic-iv-parquet
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from slices.data.data_io import convert_csv_to_parquet


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Setup MIMIC-IV data pipeline (convert + extract).

    Args:
        cfg: Hydra configuration object. Expects:
            - data.csv_root: Optional path to CSV files
            - data.parquet_root: Path to/for Parquet files
            - data.output_dir: Path for processed output
    """
    csv_root_str = cfg.data.get("csv_root")
    parquet_root = Path(cfg.data.parquet_root)
    dataset_name = cfg.data.get("name", "dataset")

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
            print("  - data.csv_root=/path/to/csv (will convert to Parquet)")
            print("  - data.parquet_root=/path/to/existing/parquet")
            sys.exit(1)

        print("Step 1/2: Skipping CSV conversion (Parquet files already exist)")
        print(f"  Parquet root: {parquet_root}")
        print()

    # Step 2: Extract features
    print("Step 2/2: Extracting features from Parquet...")
    print(f"  Output directory: {cfg.data.output_dir}")
    print(f"  Sequence length: {cfg.data.seq_length_hours} hours")
    print(f"  Feature set: {cfg.data.feature_set}")
    print()

    # TODO: Implement extraction
    # from slices.data.extractors.mimic_iv import MIMICIVExtractor
    # extractor = MIMICIVExtractor(ExtractorConfig(**cfg.data))
    # extractor.run()

    print("TODO: Implement MIMIC-IV extraction")
    print("(MIMICIVExtractor not yet implemented)")

    print("\n=== Setup Summary ===")
    print(f"Parquet files: {parquet_root}")
    print(f"Processed data: {cfg.data.output_dir} (when extraction is implemented)")


if __name__ == "__main__":
    main()
