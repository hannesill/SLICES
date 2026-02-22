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
    uv run python scripts/training/supervised.py dataset=miiv

    # Different task
    uv run python scripts/training/supervised.py dataset=miiv tasks=mortality_hospital

    # Different dataset
    uv run python scripts/training/supervised.py dataset=eicu

    # Resume interrupted training
    uv run python scripts/training/supervised.py \
        dataset=miiv ckpt_path=outputs/.../checkpoints/last.ckpt
"""

from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, OmegaConf
from slices.data.datamodule import ICUDataModule
from slices.models.encoders import EncoderWithMissingToken
from slices.training import FineTuneModule
from slices.training.utils import (
    run_fairness_evaluation,
    save_encoder_checkpoint,
    setup_finetune_callbacks,
    setup_wandb_logger,
    validate_data_prerequisites,
)


def compute_baseline_metrics(datamodule, task_name: str, task_type: str = "binary") -> dict:
    """Compute baseline metrics for comparison.

    For classification tasks, computes AUROC for random predictions and
    majority-class predictions. For regression tasks, computes mean/median
    predictor baselines.
    """
    baselines = {}

    test_dataset = datamodule.test_dataloader().dataset
    labels = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if "label" in sample:
            labels.append(sample["label"])

    if not labels:
        return baselines

    labels_tensor = torch.tensor(labels)
    n_samples = len(labels_tensor)
    baselines["test/n_samples"] = n_samples

    if task_type == "regression":
        mean_label = labels_tensor.mean().item()
        median_label = labels_tensor.median().item()
        baselines["test/label_mean"] = mean_label
        baselines["test/label_std"] = labels_tensor.std().item()
        baselines["baseline/mean_predictor_mse"] = (labels_tensor - mean_label).pow(2).mean().item()
        baselines["baseline/median_predictor_mae"] = (
            (labels_tensor - median_label).abs().mean().item()
        )
    else:
        n_positive = labels_tensor.sum().item()
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples

        baselines["test/n_positive"] = n_positive
        baselines["test/n_negative"] = n_negative
        baselines["test/positive_ratio"] = positive_ratio
        baselines["baseline/random_auroc"] = 0.5
        majority_class_accuracy = max(positive_ratio, 1 - positive_ratio)
        baselines["baseline/majority_accuracy"] = majority_class_accuracy
        baselines["baseline/trivial_auroc"] = 0.5

    return baselines


def save_encoder_weights(model: FineTuneModule, cfg: DictConfig, output_dir: str) -> Path:
    """Save encoder weights to a .pt file in v3 checkpoint format."""
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
    """Run full supervised training from scratch."""
    print("=" * 80)
    print("Full Supervised Training (Baseline)")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate that freeze_encoder is False
    if cfg.training.get("freeze_encoder", False):
        print("\nWarning: freeze_encoder=true in supervised training makes no sense.")
        print("Overriding to freeze_encoder=false")
        OmegaConf.set_struct(cfg, False)
        cfg.training.freeze_encoder = False
        OmegaConf.set_struct(cfg, True)

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
    # 2. Create Model (Training from Scratch)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. Creating Model (Training from Scratch)")
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
        checkpoint_path=None,
        pretrain_checkpoint_path=None,
    )

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

    callbacks = setup_finetune_callbacks(cfg, checkpoint_prefix="supervised")
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

    task_type = cfg.task.get("task_type", "binary")
    print("\n Computing baseline metrics...")
    baseline_metrics = compute_baseline_metrics(datamodule, task_name, task_type)

    # =========================================================================
    # 7. Fairness Evaluation (optional)
    # =========================================================================
    run_fairness_evaluation(model, datamodule, cfg, logger)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Supervised Training Complete!")
    print("=" * 80)

    if hasattr(callbacks[0], "best_model_path"):
        print(f"\n Best checkpoint: {callbacks[0].best_model_path}")
        print(f"  - Best {callbacks[0].monitor}: {callbacks[0].best_model_score:.4f}")

    if test_results:
        print("\n Test Results:")
        for key, value in test_results[0].items():
            print(f"  - {key}: {value:.4f}")

        # Print baseline comparison
        print("\n Baseline Comparison:")
        if task_type == "regression":
            if "baseline/mean_predictor_mse" in baseline_metrics:
                print(
                    f"  - Mean predictor MSE: "
                    f"{baseline_metrics['baseline/mean_predictor_mse']:.4f}"
                )
            if "baseline/median_predictor_mae" in baseline_metrics:
                print(
                    f"  - Median predictor MAE: "
                    f"{baseline_metrics['baseline/median_predictor_mae']:.4f}"
                )
            if (
                "test/mae" in test_results[0]
                and "baseline/median_predictor_mae" in baseline_metrics
            ):
                model_mae = test_results[0]["test/mae"]
                baseline_mae = baseline_metrics["baseline/median_predictor_mae"]
                improvement = baseline_mae - model_mae
                print(f"\n  Model vs Median Predictor: {improvement:+.4f} MAE")
                if improvement > 0:
                    print("  -> Model outperforms trivial baseline")
                else:
                    print("  -> WARNING: Model performs at or below trivial baseline!")
        else:
            if "baseline/random_auroc" in baseline_metrics:
                print(f"  - Random AUROC: {baseline_metrics['baseline/random_auroc']:.4f}")
            if "baseline/majority_accuracy" in baseline_metrics:
                print(
                    f"  - Majority class accuracy: "
                    f"{baseline_metrics['baseline/majority_accuracy']:.4f}"
                )

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

    # Log test results and baseline metrics to W&B summary
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
