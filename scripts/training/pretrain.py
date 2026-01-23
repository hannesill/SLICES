"""Entry point for SSL pretraining.

This script orchestrates the full SSL pretraining pipeline using Lightning
and Hydra for configuration management. It's agnostic to encoder architecture and
SSL objective - all components are built from YAML configs.

Example usage:
    # Use default config
    uv run python scripts/pretrain.py

    # Override data path
    uv run python scripts/pretrain.py data.processed_dir=/path/to/processed

    # Override encoder and SSL settings
    uv run python scripts/pretrain.py encoder.d_model=256 ssl.mask_ratio=0.20

    # Use different configs
    uv run python scripts/pretrain.py encoder=transformer ssl=mae
"""

from pathlib import Path
from typing import Optional

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from slices.data.config_schemas import DataConfig
from slices.data.datamodule import ICUDataModule
from slices.training import SSLPretrainModule


def validate_data_config(cfg: DictConfig) -> DataConfig:
    """Validate data configuration using Pydantic schema.

    Args:
        cfg: Hydra configuration object containing 'data' section.

    Returns:
        Validated DataConfig instance.

    Raises:
        pydantic.ValidationError: If data config is invalid.
    """
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    return DataConfig(**data_dict)


def setup_callbacks(cfg: DictConfig) -> list:
    """Set up training callbacks.

    Args:
        cfg: Configuration object.

    Returns:
        List of Lightning callbacks.
    """
    callbacks = []

    # Model checkpointing
    # Note: Lightning sanitizes metric names for filenames (val/loss -> val_loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "checkpoints"),
        filename="ssl-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.get("early_stopping_patience", None):
        early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Set up experiment logger.

    Args:
        cfg: Configuration object.

    Returns:
        Logger instance or None.
    """
    if not cfg.logging.get("use_wandb", False):
        return None

    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.get("wandb_entity", None),
        name=cfg.logging.get("run_name", None),
        group=cfg.logging.get("wandb_group", None),
        save_dir=cfg.output_dir,
        log_model=False,  # Don't save model to wandb (too large)
    )

    # Log configuration
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    return logger


@hydra.main(version_base=None, config_path="../../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    """Run SSL pretraining.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("SSL Pretraining Pipeline")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate data configuration early
    print("\nValidating data configuration...")
    data_config = validate_data_config(cfg)
    print(f"  - Data config validated: processed_dir={data_config.processed_dir}")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # =========================================================================
    # 1. Setup DataModule
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. Setting up DataModule")
    print("=" * 80)

    datamodule = ICUDataModule(
        processed_dir=cfg.data.processed_dir,
        task_name=None,  # No task labels needed for SSL pretraining
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
        train_ratio=cfg.data.get("train_ratio", 0.7),
        val_ratio=cfg.data.get("val_ratio", 0.15),
        test_ratio=cfg.data.get("test_ratio", 0.15),
        seed=cfg.seed,
        normalize=cfg.data.get("normalize", True),
        impute_strategy=cfg.data.get("impute_strategy", "forward_fill"),
        pin_memory=cfg.data.get("pin_memory", True),
        # Sliding window parameters for longer sequences
        enable_sliding_windows=cfg.data.get("enable_sliding_windows", False),
        window_size=cfg.data.get("window_size", None),
        window_stride=cfg.data.get("window_stride", None),
    )

    # Setup data
    datamodule.setup()

    print("\n✓ DataModule initialized")
    print(f"  - Processed dir: {cfg.data.processed_dir}")
    print(f"  - Feature dimension: {datamodule.get_feature_dim()}")
    print(f"  - Sequence length: {datamodule.get_seq_length()}")

    # Get split info
    split_info = datamodule.get_split_info()
    print("\n✓ Data splits (patient-level):")
    print(
        f"  - Train: {split_info['train_patients']} patients, " f"{split_info['train_stays']} stays"
    )
    print(f"  - Val:   {split_info['val_patients']} patients, " f"{split_info['val_stays']} stays")
    print(
        f"  - Test:  {split_info['test_patients']} patients, " f"{split_info['test_stays']} stays"
    )

    # =========================================================================
    # 2. Create SSL Module
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Creating SSL Module")
    print("=" * 80)

    # Inject d_input from data into encoder config
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.encoder.d_input = datamodule.get_feature_dim()
    OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # Create Lightning module
    model = SSLPretrainModule(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n✓ SSL module created")
    print(f"  - Encoder: {cfg.encoder.name}")
    print(f"  - SSL objective: {cfg.ssl.name}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # 3. Setup Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. Setting up Trainer")
    print("=" * 80)

    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", "auto"),
        precision=cfg.training.get("precision", 32),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        deterministic=cfg.get("deterministic", False),
        overfit_batches=cfg.training.get("overfit_batches", 0),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("\n✓ Trainer configured")
    print(f"  - Max epochs: {cfg.training.max_epochs}")
    print(f"  - Accelerator: {cfg.training.get('accelerator', 'auto')}")
    print(f"  - Devices: {cfg.training.get('devices', 'auto')}")
    print(f"  - Precision: {cfg.training.get('precision', 32)}")
    if logger:
        print(f"  - Logger: W&B (project={cfg.logging.wandb_project})")
    else:
        print("  - Logger: None (CSV logs only)")

    # =========================================================================
    # 4. Train
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. Starting Training")
    print("=" * 80)

    trainer.fit(model, datamodule=datamodule)

    # =========================================================================
    # 5. Save Encoder
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. Saving Encoder")
    print("=" * 80)

    # Save encoder weights for downstream tasks
    encoder_path = Path(cfg.output_dir) / "encoder.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_encoder(str(encoder_path))

    print(f"\n✓ Encoder saved to: {encoder_path}")
    print("  - Use this for downstream fine-tuning")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Get best checkpoint
    if hasattr(callbacks[0], "best_model_path"):
        print(f"\n✓ Best checkpoint: {callbacks[0].best_model_path}")
        print(f"  - Best val/loss: {callbacks[0].best_model_score:.4f}")

    print(f"\n✓ Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")
    print(f"  - Encoder weights: {encoder_path}")

    if logger:
        print(
            f"\n✓ View training curves at: https://wandb.ai/{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
