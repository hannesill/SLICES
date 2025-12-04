"""Entry point for extracting MIMIC-IV data.

Example usage:
    python scripts/extract_mimic_iv.py data.data_dir=/path/to/mimic-iv
"""

import hydra
from omegaconf import DictConfig

from slices.data.extractors.base import BaseExtractor, ExtractorConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Extract MIMIC-IV data from local Parquet files.
    
    Args:
        cfg: Hydra configuration object.
    """
    # TODO: Create MIMICIVExtractor instance and run extraction
    # extractor = MIMICIVExtractor(ExtractorConfig(**cfg.data))
    # extractor.run()
    print("TODO: Implement MIMIC-IV extraction")
    print(f"Data directory: {cfg.data.data_dir}")


if __name__ == "__main__":
    main()

