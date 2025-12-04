"""Entry point for downstream task finetuning.

Example usage:
    python scripts/finetune.py task=mortality checkpoint=/path/to/pretrained.ckpt
"""

import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run downstream task finetuning.
    
    Args:
        cfg: Hydra configuration object.
    """
    # TODO: Implement finetuning pipeline
    # 1. Load pretrained encoder
    # 2. Add task head
    # 3. Create Lightning Trainer
    # 4. Train and evaluate
    print("TODO: Implement downstream task finetuning")
    print(f"Config: {cfg}")


if __name__ == "__main__":
    main()

