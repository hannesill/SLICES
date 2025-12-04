"""Entry point for SSL pretraining.

Example usage:
    python scripts/pretrain.py data.data_dir=/path/to/mimic-iv
"""

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run SSL pretraining.
    
    Args:
        cfg: Hydra configuration object.
    """
    # TODO: Implement pretraining pipeline
    # 1. Load DataModule
    # 2. Create SSL model (encoder + SSL objective)
    # 3. Create Lightning Trainer
    # 4. Train
    print("TODO: Implement SSL pretraining")
    print(f"Config: {cfg}")


if __name__ == "__main__":
    main()

