"""Entry point for extracting data from RICU parquet output.

Two-step pipeline:
    # Step 1: R extraction (once per dataset)
    Rscript scripts/preprocessing/extract_with_ricu.R \
        --dataset miiv --output_dir data/ricu_output/miiv

    # Step 2: Python processing -> produces final SLICES format
    uv run python scripts/preprocessing/extract_ricu.py \
        data.ricu_output_dir=data/ricu_output/miiv

    # Override output path
    uv run python scripts/preprocessing/extract_ricu.py \
        data.ricu_output_dir=data/ricu_output/miiv \
        data.processed_dir=data/processed/miiv

    # Specify different tasks
    uv run python scripts/preprocessing/extract_ricu.py \
        data.ricu_output_dir=data/ricu_output/miiv \
        'data.tasks=[mortality_24h,mortality_hospital]'
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.ricu import RicuExtractor


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract data from RICU parquet output into SLICES format.

    Args:
        cfg: Hydra configuration object. Uses cfg.data.* for all settings:
            - data.ricu_output_dir: Path to RICU parquet output
            - data.processed_dir: Path for processed output
            - data.feature_set: Feature set name (stored in metadata)
            - data.tasks: List of task names for label extraction
    """
    ricu_dir = Path(cfg.data.ricu_output_dir)

    if not ricu_dir.exists():
        print(f"Error: RICU output directory not found: {ricu_dir}")
        print("\nRun the R extraction first:")
        print(
            "  Rscript scripts/preprocessing/extract_with_ricu.R "
            f"--dataset miiv --output_dir {ricu_dir}"
        )
        sys.exit(1)

    config = ExtractorConfig(
        parquet_root=str(cfg.data.ricu_output_dir),
        output_dir=str(cfg.data.processed_dir),
        feature_set=cfg.data.feature_set,
        tasks=list(cfg.data.get("tasks", [])) if cfg.data.get("tasks") else [],
    )

    extractor = RicuExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
