"""Entry point for extracting MIMIC-IV data from Parquet files.

Example usage:
    # Use paths from config (defaults to demo data)
    uv run python scripts/extract_mimic_iv.py

    # Override parquet path
    uv run python scripts/extract_mimic_iv.py extraction.parquet_root=/path/to/mimic-iv-parquet

    # Specify different tasks
    uv run python scripts/extract_mimic_iv.py 'extraction.tasks=[mortality_24h,mortality_hospital]'

    # Use full MIMIC-IV
    uv run python scripts/extract_mimic_iv.py \
        extraction.parquet_root=/path/to/mimic-iv \
        extraction.output_dir=data/processed/mimic-iv-full
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract MIMIC-IV data from local Parquet files.

    Args:
        cfg: Hydra configuration object. Expects:
            - extraction.parquet_root: Path to Parquet files
            - extraction.output_dir: Path for processed output
            - extraction.seq_length_hours: Sequence length
            - extraction.feature_set: Feature set to extract
            - extraction.tasks: List of task names for label extraction
    """
    parquet_root = Path(cfg.extraction.parquet_root)

    if not parquet_root.exists():
        print(f"Error: Parquet root directory not found: {parquet_root}")
        print("\nIf you have CSV files, run conversion first:")
        print("  uv run python scripts/convert_csv_to_parquet.py extraction.csv_root=/path/to/csv")
        print("\nOr provide the correct path:")
        print(
            "  uv run python scripts/extract_mimic_iv.py extraction.parquet_root=/path/to/parquet"
        )
        sys.exit(1)

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
        "categories",
    }

    extraction_config = OmegaConf.to_container(cfg.extraction, resolve=True)
    filtered_config = {k: v for k, v in extraction_config.items() if k in extractor_fields}

    # Handle tasks (may be a list in config)
    if "tasks" not in filtered_config or filtered_config["tasks"] is None:
        filtered_config["tasks"] = ["mortality_24h", "mortality_48h", "mortality_hospital"]

    # Create ExtractorConfig
    extractor_config = ExtractorConfig(**filtered_config)

    # Create and run extractor
    extractor = MIMICIVExtractor(extractor_config)
    extractor.run()


if __name__ == "__main__":
    main()
