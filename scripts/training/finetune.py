"""Entry point for downstream task finetuning.

This script orchestrates finetuning a pretrained encoder on downstream clinical
prediction tasks. It supports various training strategies:

1. Linear probing (default): Freeze encoder, train only task head
2. Full finetuning: Train both encoder and task head
3. Gradual unfreezing: Start frozen, unfreeze after warmup epochs

Example usage:
    # Linear probing with encoder weights
    uv run python scripts/finetune.py checkpoint=outputs/encoder.pt

    # Linear probing from full pretrain checkpoint
    uv run python scripts/finetune.py pretrain_checkpoint=outputs/ssl-last.ckpt

    # Full finetuning
    uv run python scripts/finetune.py checkpoint=outputs/encoder.pt training.freeze_encoder=false

    # Different task
    uv run python scripts/finetune.py checkpoint=outputs/encoder.pt \
        task.task_name=mortality_hospital

    # Gradual unfreezing
    uv run python scripts/finetune.py checkpoint=outputs/encoder.pt training.unfreeze_epoch=5
"""

from typing import Optional

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from slices.data.datamodule import ICUDataModule
from slices.training import FineTuneModule


def setup_callbacks(cfg: DictConfig) -> list:
    """Set up training callbacks.

    Args:
        cfg: Configuration object.

    Returns:
        List of Lightning callbacks.
    """
    callbacks = []

    # Model checkpointing (based on AUROC for classification tasks)
    monitor = cfg.training.get("early_stopping_monitor", "val/auroc")
    mode = cfg.training.get("early_stopping_mode", "max")

    # Convert metric name for filename (val/auroc -> val_auroc)
    metric_filename = monitor.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "checkpoints"),
        filename=f"finetune-{{epoch:03d}}-{{{metric_filename}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.get("early_stopping_patience", None):
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=cfg.training.early_stopping_patience,
            mode=mode,
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
        log_model=False,
    )

    # Log configuration
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    return logger


@hydra.main(version_base=None, config_path="../../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Run downstream task finetuning.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("Downstream Task Finetuning Pipeline")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate checkpoint
    if cfg.checkpoint is None and cfg.pretrain_checkpoint is None:
        raise ValueError(
            "Must provide either 'checkpoint' (encoder.pt) or "
            "'pretrain_checkpoint' (full .ckpt file)"
        )

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # =========================================================================
    # 1. Setup DataModule
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. Setting up DataModule")
    print("=" * 80)

    task_name = cfg.task.get("task_name", "mortality_24h")

    if task_name == "decompensation":
        from slices.data.decompensation_datamodule import DecompensationDataModule

        datamodule = DecompensationDataModule(
            ricu_parquet_root=cfg.data.parquet_root,
            processed_dir=cfg.data.processed_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            obs_window_hours=cfg.task.get("observation_window_hours", 48),
            pred_window_hours=cfg.task.get("prediction_window_hours", 24),
            stride_hours=cfg.task.get("label_params", {}).get("stride_hours", 6),
            eval_stride_hours=cfg.task.get("label_params", {}).get("eval_stride_hours", 1),
            seed=cfg.seed,
        )
    else:
        datamodule = ICUDataModule(
            processed_dir=cfg.data.processed_dir,
            task_name=task_name,  # Load task labels
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            seed=cfg.seed,
        )

    # Setup data
    datamodule.setup()

    print("\n✓ DataModule initialized")
    print(f"  - Processed dir: {cfg.data.processed_dir}")
    print(f"  - Task: {task_name}")
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

    # Get label statistics
    label_stats = datamodule.get_label_statistics()
    if task_name in label_stats:
        stats = label_stats[task_name]
        print(f"\n✓ Label distribution for '{task_name}':")
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

    # Inject d_input from data into encoder config
    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    OmegaConf.set_struct(cfg, True)

    # Create Lightning module
    model = FineTuneModule(
        config=cfg,
        checkpoint_path=cfg.checkpoint,
        pretrain_checkpoint_path=cfg.pretrain_checkpoint,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n✓ Finetune module created")
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

    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)

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

    print("\n✓ Trainer configured")
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

    # Load best checkpoint for testing
    best_ckpt = callbacks[0].best_model_path if hasattr(callbacks[0], "best_model_path") else None

    if best_ckpt:
        print(f"\n✓ Best checkpoint: {best_ckpt}")
        # Load checkpoint manually with weights_only=False (PyTorch 2.6+ compatibility)
        try:
            checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print("  - Loaded best checkpoint weights")
        except Exception as e:
            print(f"  - Warning: Could not load checkpoint ({e}), using final model")
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        print("\n⚠ No best checkpoint found, testing with final model")
        test_results = trainer.test(model, datamodule=datamodule)

    # =========================================================================
    # 6. Fairness Evaluation (optional)
    # =========================================================================
    fairness_cfg = cfg.get("eval", {}).get("fairness", {})
    if fairness_cfg.get("enabled", False) and task_name != "decompensation":
        print("\n" + "=" * 80)
        print("6. Fairness Evaluation")
        print("=" * 80)

        from slices.eval.fairness_evaluator import FairnessEvaluator

        # Collect test predictions
        model.eval()
        all_preds, all_labels, all_stay_ids = [], [], []
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                outputs = model(
                    batch["timeseries"].to(model.device),
                    batch["mask"].to(model.device),
                )
            probs = outputs["probs"]
            if probs.dim() > 1 and probs.shape[1] == 2:
                all_preds.append(probs[:, 1].cpu())
            else:
                all_preds.append(probs.cpu())
            all_labels.append(batch["label"].cpu())
            all_stay_ids.extend(
                batch["stay_id"].tolist()
                if isinstance(batch["stay_id"], torch.Tensor)
                else batch["stay_id"]
            )

        predictions = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)

        evaluator = FairnessEvaluator(
            static_df=datamodule.dataset.static_df,
            protected_attributes=list(
                fairness_cfg.get("protected_attributes", ["gender", "age_group"])
            ),
            min_subgroup_size=fairness_cfg.get("min_subgroup_size", 50),
        )
        fairness_report = evaluator.evaluate(predictions, labels_tensor, all_stay_ids)
        evaluator.print_report(fairness_report)

        if logger:
            for attr, metrics in fairness_report.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.experiment.summary[f"fairness/{attr}/{metric_name}"] = value

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Finetuning Complete!")
    print("=" * 80)

    # Get best checkpoint info
    if hasattr(callbacks[0], "best_model_path"):
        monitor = cfg.training.get("early_stopping_monitor", "val/auroc")
        print(f"\n✓ Best checkpoint: {callbacks[0].best_model_path}")
        print(f"  - Best {monitor}: {callbacks[0].best_model_score:.4f}")

    # Print test results
    if test_results:
        print("\n✓ Test Results:")
        for key, value in test_results[0].items():
            print(f"  - {key}: {value:.4f}")

        # Log test results to wandb summary for easy retrieval
        if logger:
            logger.experiment.summary.update(test_results[0])

    print(f"\n✓ Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")

    if logger:
        print(
            f"\n✓ View training curves at: https://wandb.ai/{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
