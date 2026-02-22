"""Entry point for downstream task finetuning.

This script orchestrates finetuning a pretrained encoder on downstream clinical
prediction tasks. It supports various training strategies:

1. Linear probing (default): Freeze encoder, train only task head
2. Full finetuning: Train both encoder and task head
3. Gradual unfreezing: Start frozen, unfreeze after warmup epochs

Example usage:
    # Linear probing with encoder weights
    uv run python scripts/training/finetune.py \
        dataset=miiv checkpoint=outputs/encoder.pt

    # Full finetuning
    uv run python scripts/training/finetune.py \
        dataset=miiv checkpoint=outputs/encoder.pt \
        training.freeze_encoder=false

    # Different task
    uv run python scripts/training/finetune.py \
        dataset=miiv checkpoint=outputs/encoder.pt \
        tasks=mortality_hospital

    # Gradual unfreezing
    uv run python scripts/training/finetune.py \
        dataset=miiv checkpoint=outputs/encoder.pt \
        training.unfreeze_epoch=5
"""

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from slices.data.datamodule import ICUDataModule
from slices.training import FineTuneModule
from slices.training.utils import (
    run_fairness_evaluation,
    setup_finetune_callbacks,
    setup_wandb_logger,
    validate_data_prerequisites,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Run downstream task finetuning."""
    print("=" * 80)
    print("Downstream Task Finetuning Pipeline")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate checkpoint
    if cfg.checkpoint is None and cfg.pretrain_checkpoint is None:
        raise ValueError(
            "Must provide either 'checkpoint' (encoder.pt) or "
            "'pretrain_checkpoint' (full .ckpt file)"
        )

    # Auto-detect paradigm from encoder checkpoint metadata
    if cfg.checkpoint is not None:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "ssl_name" in ckpt:
            detected = ckpt["ssl_name"]
            if cfg.paradigm != detected:
                print(f"\n  Auto-detected paradigm from checkpoint: {detected}")
                OmegaConf.set_struct(cfg, False)
                cfg.paradigm = detected
                OmegaConf.set_struct(cfg, True)
        del ckpt

    # Validate data prerequisites
    validate_data_prerequisites(cfg.data.processed_dir, cfg.dataset)

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # =========================================================================
    # 1. Setup DataModule
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. Setting up DataModule")
    print("=" * 80)

    task_name = cfg.task.get("task_name", "mortality_24h")

    datamodule = ICUDataModule(
        processed_dir=cfg.data.processed_dir,
        task_name=task_name,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.get("num_workers", 4),
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
    )

    datamodule.setup()

    print("\n DataModule initialized")
    print(f"  - Processed dir: {cfg.data.processed_dir}")
    print(f"  - Task: {task_name}")
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

    label_stats = datamodule.get_label_statistics()
    if task_name in label_stats:
        stats = label_stats[task_name]
        print(f"\n Label distribution for '{task_name}':")
        print(f"  - Total samples: {stats['total']}")
        print(
            f"  - Positive: {stats.get('positive', 'N/A')} "
            f"({stats.get('prevalence', 0)*100:.1f}%)"
        )
        print(
            f"  - Negative: {stats.get('negative', 'N/A')} "
            f"({(1 - stats.get('prevalence', 0))*100:.1f}%)"
        )

    # =========================================================================
    # 2. Create Finetune Module
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Creating Finetune Module")
    print("=" * 80)

    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    cfg.encoder.max_seq_length = datamodule.get_seq_length()

    # Resolve "balanced" class weights from label distribution
    if cfg.training.get("class_weight") == "balanced":
        if task_name in label_stats:
            stats = label_stats[task_name]
            n_pos = stats.get("positive", 1)
            n_neg = stats.get("negative", 1)
            n_total = n_pos + n_neg
            cfg.training.class_weight = [n_total / (2 * n_neg), n_total / (2 * n_pos)]
            print(f"\n  Balanced class weights: {cfg.training.class_weight}")
        else:
            print(f"\n  Warning: No label stats for '{task_name}', skipping class weighting")
            cfg.training.class_weight = None

    OmegaConf.set_struct(cfg, True)

    model = FineTuneModule(
        config=cfg,
        checkpoint_path=cfg.checkpoint,
        pretrain_checkpoint_path=cfg.pretrain_checkpoint,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n Finetune module created")
    print(f"  - Encoder: {cfg.encoder.name}")
    print(f"  - Task head: {cfg.task.get('head_type', 'mlp')}")
    print(f"  - Task: {cfg.task.task_name} ({cfg.task.task_type})")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Freeze strategy: {'frozen' if cfg.training.freeze_encoder else 'full finetuning'}")
    if cfg.training.get("unfreeze_epoch"):
        print(f"  - Unfreeze epoch: {cfg.training.unfreeze_epoch}")

    # =========================================================================
    # 3. Setup Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. Setting up Trainer")
    print("=" * 80)

    callbacks = setup_finetune_callbacks(cfg, checkpoint_prefix="finetune")
    logger = setup_wandb_logger(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.get("accelerator", "auto"),
        devices=cfg.training.get("devices", "auto"),
        precision=cfg.training.get("precision", 32),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 10),
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
    # 5. Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. Evaluating on Test Set")
    print("=" * 80)

    best_ckpt = callbacks[0].best_model_path if hasattr(callbacks[0], "best_model_path") else None

    if best_ckpt:
        print(f"\n Best checkpoint: {best_ckpt}")
        try:
            checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print("  - Loaded best checkpoint weights")
        except Exception as e:
            print(f"  - Warning: Could not load checkpoint ({e}), using final model")
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        print("\n  No best checkpoint found, testing with final model")
        test_results = trainer.test(model, datamodule=datamodule)

    # =========================================================================
    # 6. Fairness Evaluation (optional)
    # =========================================================================
    run_fairness_evaluation(model, datamodule, cfg, logger)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Finetuning Complete!")
    print("=" * 80)

    if hasattr(callbacks[0], "best_model_path"):
        print(f"\n Best checkpoint: {callbacks[0].best_model_path}")
        print(f"  - Best {callbacks[0].monitor}: {callbacks[0].best_model_score:.4f}")

    if test_results:
        print("\n Test Results:")
        for key, value in test_results[0].items():
            print(f"  - {key}: {value:.4f}")

        if logger:
            logger.experiment.summary.update(test_results[0])

    print(f"\n Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")

    if logger:
        print(
            f"\n View training curves at: https://wandb.ai/{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
