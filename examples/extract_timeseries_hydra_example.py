"""Example: Extract time-series features using Hydra configuration.

This demonstrates the recommended way to configure the extractor using Hydra,
which makes it robust for both development and production deployments.

Usage:
    # Default config (configs/config.yaml)
    uv run python examples/extract_timeseries_hydra_example.py

    # Override parquet root
    uv run python examples/extract_timeseries_hydra_example.py \
        extraction.parquet_root=/path/to/mimic-iv

    # Explicit concepts directory override
    uv run python examples/extract_timeseries_hydra_example.py \
        extraction.concepts_dir=/opt/slices/concepts
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from slices.data.extractors.mimic_iv import ExtractorConfig, MIMICIVExtractor


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract time-series features from MIMIC-IV."""

    # Convert Hydra config to ExtractorConfig
    # Filter only fields that ExtractorConfig accepts
    extraction_config = OmegaConf.to_container(cfg.extraction, resolve=True)
    extractor_fields = {
        "parquet_root",
        "output_dir",
        "feature_set",
    }
    filtered_config = {k: v for k, v in extraction_config.items() if k in extractor_fields}

    extractor_config = ExtractorConfig(**filtered_config)

    print("Extractor configuration:")
    print(f"  Parquet root: {extractor_config.parquet_root}")
    print(f"  Output dir: {extractor_config.output_dir}")
    print(f"  Feature set: {extractor_config.feature_set}")

    # Create extractor
    extractor = MIMICIVExtractor(extractor_config)

    # Extract stays
    print("\nExtracting ICU stays...")
    stays = extractor.extract_stays()
    print(f"Found {len(stays)} ICU stays")

    # Extract time-series for first 10 stays
    stay_ids = stays["stay_id"].head(10).to_list()
    print(f"\nExtracting time-series for {len(stay_ids)} stays...")

    timeseries = extractor.extract_timeseries(stay_ids)

    print("\nTime-series extraction complete:")
    print(f"  Shape: {timeseries.shape}")
    print(f"  Columns: {timeseries.columns}")
    print(f"  Unique stays: {timeseries['stay_id'].n_unique()}")
    print(f"  Hour range: {timeseries['hour'].min()} to {timeseries['hour'].max()}")

    # Show sample data
    print("\nSample data (first few rows):")
    print(timeseries.head())

    # Show feature availability (how many observations per feature)
    print("\nFeature observation counts:")
    for col in timeseries.columns:
        if col.endswith("_mask"):
            feature_name = col.replace("_mask", "")
            if feature_name in timeseries.columns:
                obs_count = timeseries[col].sum()
                total_hours = len(timeseries)
                pct = (obs_count / total_hours * 100) if total_hours > 0 else 0
                print(f"  {feature_name:15} {obs_count:5} / {total_hours:5} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
