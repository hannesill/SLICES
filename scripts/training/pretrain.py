"""Entry point for SSL pretraining.

This script orchestrates the full SSL pretraining pipeline using Lightning
and Hydra for configuration management. It's agnostic to encoder architecture and
SSL objective - all components are built from YAML configs.

Paradigm and experiment_name are auto-derived from cfg.ssl.name. Pick paradigm
with ssl= override:

    uv run python scripts/training/pretrain.py dataset=miiv              # MAE (default)
    uv run python scripts/training/pretrain.py dataset=miiv ssl=jepa
    uv run python scripts/training/pretrain.py dataset=miiv ssl=contrastive
    uv run python scripts/training/pretrain.py dataset=miiv ssl=smart model=smart
"""

from pathlib import Path

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from slices.data.config_schemas import DataConfig
from slices.data.datamodule import ICUDataModule
from slices.training import SSLPretrainModule
from slices.training.utils import (
    setup_pretrain_callbacks,
    setup_wandb_logger,
    validate_data_prerequisites,
)

# SSL -> compatible model encoder mappings
_SSL_MODEL_COMPAT = {
    "mae": {"observation_transformer"},
    "jepa": {"observation_transformer"},
    "contrastive": {"observation_transformer"},
    "smart": {"smart"},
}


def validate_ssl_model_compatibility(cfg: DictConfig) -> None:
    """Validate that the SSL objective and encoder are compatible."""
    ssl_name = cfg.ssl.name
    model_name = cfg.encoder.name

    compatible = _SSL_MODEL_COMPAT.get(ssl_name)
    if compatible is not None and model_name not in compatible:
        expected = ", ".join(sorted(compatible))
        raise ValueError(
            f"SSL objective '{ssl_name}' requires encoder model in {{{expected}}}, "
            f"but got '{model_name}'. "
            f"Use: model={list(compatible)[0]}"
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    """Run SSL pretraining."""
    # Auto-derive paradigm and experiment_name from ssl.name
    ssl_name = cfg.ssl.name
    OmegaConf.set_struct(cfg, False)
    cfg.paradigm = ssl_name
    cfg.experiment_name = f"{ssl_name}_pretrain"
    OmegaConf.set_struct(cfg, True)

    print("=" * 80)
    print(f"SSL Pretraining Pipeline — {ssl_name.upper()}")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate SSL/model compatibility
    validate_ssl_model_compatibility(cfg)

    # Validate data prerequisites
    validate_data_prerequisites(cfg.data.processed_dir, cfg.dataset)

    # Validate data configuration early
    print("\nValidating data configuration...")
    data_dict = OmegaConf.to_container(cfg.data, resolve=True)
    data_config = DataConfig(**data_dict)
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
        seed=cfg.seed,
        enable_sliding_windows=cfg.data.get("enable_sliding_windows", False),
        window_size=cfg.data.get("window_size", None),
        window_stride=cfg.data.get("window_stride", None),
    )

    datamodule.setup()

    print("\n DataModule initialized")
    print(f"  - Processed dir: {cfg.data.processed_dir}")
    print(f"  - Feature dimension: {datamodule.get_feature_dim()}")
    print(f"  - Sequence length: {datamodule.get_seq_length()}")

    split_info = datamodule.get_split_info()
    print("\n Data splits (patient-level):")
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

    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    cfg.encoder.max_seq_length = datamodule.get_seq_length()
    OmegaConf.set_struct(cfg, True)

    model = SSLPretrainModule(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n SSL module created")
    print(f"  - Encoder: {cfg.encoder.name}")
    print(f"  - SSL objective: {ssl_name}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # 3. Setup Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. Setting up Trainer")
    print("=" * 80)

    callbacks = setup_pretrain_callbacks(cfg)
    logger = setup_wandb_logger(cfg)

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

    print("\n Trainer configured")
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
    # 5. Save Encoder (from best checkpoint)
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. Saving Encoder")
    print("=" * 80)

    checkpoint_callback = callbacks[0]
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"\n  Loading best checkpoint: {best_ckpt}")
        print(f"  Best val/loss: {checkpoint_callback.best_model_score:.4f}")
        model = SSLPretrainModule.load_from_checkpoint(best_ckpt)

    encoder_path = Path(cfg.output_dir) / "encoder.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_encoder(str(encoder_path))

    print(f"\n Encoder saved to: {encoder_path}")
    print("  - Use this for downstream fine-tuning")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    if best_ckpt:
        print(f"\n Best checkpoint: {best_ckpt}")
        print(f"  - Best val/loss: {checkpoint_callback.best_model_score:.4f}")

    print(f"\n Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")
    print(f"  - Encoder weights: {encoder_path}")

    if logger:
        print(
            f"\n View training curves at: https://wandb.ai/{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
