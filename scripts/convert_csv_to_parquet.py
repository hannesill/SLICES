"""Entry point for converting CSV.gz files to Parquet format.

Example usage:
    # Using config file paths
    python scripts/convert_csv_to_parquet.py

    # Override paths via command line
    python scripts/convert_csv_to_parquet.py \
        data.csv_root=/path/to/csv data.parquet_root=/path/to/parquet
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from slices.data.data_io import convert_csv_to_parquet


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Convert CSV.gz files to Parquet format.

    Args:
        cfg: Hydra configuration object. Expects:
            - data.csv_root: Path to directory containing CSV.gz files
            - data.parquet_root: Path to output directory for Parquet files
            - data.name: Dataset name for logging
    """
    csv_root_str = cfg.data.get("csv_root")

    if csv_root_str is None:
        print("Error: data.csv_root must be specified in config or via command line")
        print(
            "Example: python scripts/convert_csv_to_parquet.py data.csv_root=/path/to/mimic-iv-csv"
        )
        print("\nAlternatively, set csv_root in configs/data/mimic_iv.yaml")
        sys.exit(1)

    csv_root = Path(csv_root_str)
    parquet_root = Path(cfg.data.parquet_root)
    dataset_name = cfg.data.get("name", "dataset")

    if not csv_root.exists():
        print(f"Error: CSV root directory not found: {csv_root}")
        print("Please provide a valid path via data.csv_root")
        sys.exit(1)

    print(f"Dataset: {dataset_name}")
    print(f"CSV root: {csv_root}")
    print(f"Parquet root: {parquet_root}")
    print()

    success = convert_csv_to_parquet(
        csv_root=csv_root,
        parquet_root=parquet_root,
        dataset_name=dataset_name,
    )

    if success:
        print("\n✓ Conversion completed successfully!")
        print(f"Parquet files are available at: {parquet_root}")
        print("\nNext step: Run extraction")
        print("  python scripts/extract_mimic_iv.py")
    else:
        print("\n✗ Conversion failed. Check logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
