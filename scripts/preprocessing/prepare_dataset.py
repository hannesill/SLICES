"""Prepare dataset splits and normalization statistics."""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from slices.data.preparation import prepare_processed_dataset


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Prepare dataset splits and normalization statistics."""
    prepare_processed_dataset(
        processed_dir=Path(cfg.data.processed_dir),
        seed=cfg.seed,
        dataset_name=cfg.dataset,
    )


if __name__ == "__main__":
    main()
