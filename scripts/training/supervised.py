"""Entry point for full supervised training from scratch.

This script trains a model end-to-end on labeled data without any pretraining.
It serves as a baseline for comparing SSL pretraining approaches.

Key differences from finetuning:
- No pretrained checkpoint loading (trains from random initialization)
- All parameters trainable from the start
- Higher learning rate (training from scratch typically needs higher LR)
- Saves encoder weights for potential use as initialization

Example usage:
    # Train supervised baseline on default task (mortality_24h)
    uv run python scripts/training/supervised.py

    # Different task
    uv run python scripts/training/supervised.py task.task_name=mortality_hospital

    # Resume interrupted training
    uv run python scripts/training/supervised.py ckpt_path=outputs/.../checkpoints/last.ckpt

    # With W&B logging
    uv run python scripts/training/supervised.py logging.use_wandb=true
"""

from pathlib import Path
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
from slices.training.utils import save_encoder_checkpoint


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
        filename=f"supervised-{{epoch:03d}}-{{{metric_filename}:.4f}}",
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


def compute_baseline_metrics(datamodule: ICUDataModule, task_name: str) -> dict:
    """Compute baseline metrics for comparison.

    Computes AUROC for random predictions and majority-class predictions
    to provide context for the model's performance.

    Args:
        datamodule: Data module with test set loaded.
        task_name: Name of the task being evaluated.

    Returns:
        Dictionary with baseline metrics.
    """
    baselines = {}

    # Get test labels
    test_dataset = datamodule.test_dataloader().dataset
    labels = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if "label" in sample:
            labels.append(sample["label"])

    if not labels:
        return baselines

    labels_tensor = torch.tensor(labels)

    # Compute class distribution
    n_samples = len(labels_tensor)
    n_positive = labels_tensor.sum().item()
    n_negative = n_samples - n_positive
    positive_ratio = n_positive / n_samples

    baselines["test/n_samples"] = n_samples
    baselines["test/n_positive"] = n_positive
    baselines["test/n_negative"] = n_negative
    baselines["test/positive_ratio"] = positive_ratio

    # Random baseline AUROC is always 0.5
    baselines["baseline/random_auroc"] = 0.5

    # Majority class accuracy
    majority_class_accuracy = max(positive_ratio, 1 - positive_ratio)
    baselines["baseline/majority_accuracy"] = majority_class_accuracy

    # Majority class "AUROC" - if predicting all same class, AUROC is undefined
    # but we can note what the trivial predictor would achieve
    baselines["baseline/trivial_auroc"] = 0.5  # Same as random for AUROC

    return baselines


def save_encoder_weights(model: FineTuneModule, cfg: DictConfig, output_dir: str) -> Path:
    """Save encoder weights to a .pt file in v3 checkpoint format.

    Uses the shared save_encoder_checkpoint helper to ensure consistent
    checkpoint format across all training scripts.

    Args:
        model: Trained model with encoder.
        cfg: Hydra configuration with encoder settings.
        output_dir: Directory to save the weights.

    Returns:
        Path to saved encoder weights.
    """
    from slices.models.encoders import EncoderWithMissingToken

    encoder_path = Path(output_dir) / "encoder.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    encoder_config = {
        "name": cfg.encoder.name,
        **{k: v for k, v in cfg.encoder.items() if k != "name"},
    }

    encoder = model.encoder
    missing_token = None

    if isinstance(encoder, EncoderWithMissingToken):
        missing_token = encoder.missing_token
        inner_encoder = encoder.encoder
    else:
        inner_encoder = encoder

    save_encoder_checkpoint(
        encoder=inner_encoder,
        encoder_config=encoder_config,
        path=encoder_path,
        missing_token=missing_token,
    )

    return encoder_path


@hydra.main(version_base=None, config_path="../../configs", config_name="supervised")
def main(cfg: DictConfig) -> None:
    """Run full supervised training from scratch.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 80)
    print("Full Supervised Training (Baseline)")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate that freeze_encoder is False
    if cfg.training.get("freeze_encoder", False):
        print("\nWarning: freeze_encoder=true in supervised training makes no sense.")
        print("Overriding to freeze_encoder=false")
        OmegaConf.set_struct(cfg, False)
        cfg.training.freeze_encoder = False
        OmegaConf.set_struct(cfg, True)

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
            task_name=task_name,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.get("num_workers", 4),
            seed=cfg.seed,
        )

    # Setup data
    datamodule.setup()

    print("\n DataModule initialized")
    print(f"  - Processed dir: {cfg.data.processed_dir}")
    print(f"  - Task: {task_name}")
    print(f"  - Feature dimension: {datamodule.get_feature_dim()}")
    print(f"  - Sequence length: {datamodule.get_seq_length()}")

    # Get split info
    split_info = datamodule.get_split_info()
    print("\n Data splits (patient-level):")
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
    # 2. Create Model (Training from Scratch)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Creating Model (Training from Scratch)")
    print("=" * 80)

    # Inject d_input from data into encoder config
    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    OmegaConf.set_struct(cfg, True)

    # Create Lightning module WITHOUT pretrained weights
    model = FineTuneModule(
        config=cfg,
        checkpoint_path=None,  # No pretrained weights
        pretrain_checkpoint_path=None,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n Model created (random initialization)")
    print(f"  - Encoder: {cfg.encoder.name}")
    print(f"  - Task head: {cfg.task.get('head_type', 'mlp')}")
    print(f"  - Task: {cfg.task.task_name} ({cfg.task.task_type})")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print("  - Training mode: Full (all parameters trainable)")

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

    print("\n Trainer configured")
    print(f"  - Max epochs: {cfg.training.max_epochs}")
    print(f"  - Accelerator: {cfg.training.get('accelerator', 'auto')}")
    print(f"  - Devices: {cfg.training.get('devices', 'auto')}")
    print(f"  - Learning rate: {cfg.optimizer.lr}")
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

    # Support resuming from checkpoint
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path:
        print(f"\n Resuming from checkpoint: {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # =========================================================================
    # 5. Save Encoder Weights
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. Saving Encoder Weights")
    print("=" * 80)

    # Load best checkpoint before saving encoder
    best_ckpt = callbacks[0].best_model_path if hasattr(callbacks[0], "best_model_path") else None

    if best_ckpt:
        print(f"\n Loading best checkpoint: {best_ckpt}")
        try:
            checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print("  - Loaded best checkpoint weights")
        except Exception as e:
            print(f"  - Warning: Could not load checkpoint ({e}), using final model")

    encoder_path = save_encoder_weights(model, cfg, cfg.output_dir)
    print(f"\n Encoder saved to: {encoder_path}")

    # =========================================================================
    # 6. Test with Baseline Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. Evaluating on Test Set")
    print("=" * 80)

    test_results = trainer.test(model, datamodule=datamodule)

    # Compute baseline metrics
    print("\n Computing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(datamodule, task_name)

    # =========================================================================
    # 7. Fairness Evaluation (optional)
    # =========================================================================
    fairness_cfg = cfg.get("eval", {}).get("fairness", {})
    if fairness_cfg.get("enabled", False) and task_name != "decompensation":
        print("\n" + "=" * 80)
        print("7. Fairness Evaluation")
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
    print("Supervised Training Complete!")
    print("=" * 80)

    # Get best checkpoint info
    if hasattr(callbacks[0], "best_model_path"):
        monitor = cfg.training.get("early_stopping_monitor", "val/auroc")
        print(f"\n Best checkpoint: {callbacks[0].best_model_path}")
        print(f"  - Best {monitor}: {callbacks[0].best_model_score:.4f}")

    # Print test results with baseline comparison
    if test_results:
        print("\n Test Results:")
        for key, value in test_results[0].items():
            print(f"  - {key}: {value:.4f}")

        # Print baseline comparison
        print("\n Baseline Comparison:")
        if "baseline/random_auroc" in baseline_metrics:
            print(f"  - Random AUROC: {baseline_metrics['baseline/random_auroc']:.4f}")
        if "baseline/majority_accuracy" in baseline_metrics:
            print(
                f"  - Majority class accuracy: {baseline_metrics['baseline/majority_accuracy']:.4f}"
            )

        # Compare model to baselines
        if "test/auroc" in test_results[0] and "baseline/random_auroc" in baseline_metrics:
            model_auroc = test_results[0]["test/auroc"]
            random_auroc = baseline_metrics["baseline/random_auroc"]
            improvement = model_auroc - random_auroc
            print(f"\n  Model vs Random: +{improvement:.4f} AUROC")
            if improvement > 0.05:
                print("  -> Model shows meaningful learning above random baseline")
            elif improvement > 0:
                print("  -> Model shows marginal improvement over random")
            else:
                print("  -> WARNING: Model performs at or below random baseline!")

    # Log test results and baseline metrics to W&B summary for easy retrieval
    if logger:
        if test_results:
            logger.experiment.summary.update(test_results[0])
        if baseline_metrics:
            logger.experiment.summary.update(baseline_metrics)

    print(f"\n Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")
    print(f"  - Encoder weights: {encoder_path}")

    if logger:
        print(
            f"\n View training curves at: https://wandb.ai/"
            f"{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
