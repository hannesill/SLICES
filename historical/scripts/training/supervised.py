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
from slices.eval.fairness_metadata import (
    EVAL_ARTIFACT_PATH_KEY,
    EVAL_ARTIFACT_SHA256_KEY,
    file_sha256,
)
from slices.models.encoders import EncoderWithMissingToken
from slices.training import FineTuneModule
from slices.training.utils import (
    report_and_validate_train_label_support,
    resolve_balanced_class_weights,
    run_fairness_evaluation,
    save_encoder_checkpoint,
    setup_finetune_callbacks,
    setup_wandb_logger,
    train_label_support_summary,
    validate_data_prerequisites,
    WandbEntityNotFoundError,
)


def _evaluated_artifact_metadata(
    cfg: DictConfig,
    eval_checkpoint_source: str,
    best_ckpt: str | None,
) -> dict[str, str]:
    """Return path and digest for the artifact used to produce test metrics."""
    if eval_checkpoint_source == "best" and best_ckpt:
        artifact_path = Path(best_ckpt)
    elif eval_checkpoint_source == "final":
        artifact_path = Path(cfg.get("checkpoint_dir", "checkpoints")) / "last.ckpt"
    else:
        return {EVAL_ARTIFACT_PATH_KEY: "", EVAL_ARTIFACT_SHA256_KEY: ""}

    artifact_sha256 = file_sha256(artifact_path) if artifact_path.exists() else ""
    return {
        EVAL_ARTIFACT_PATH_KEY: str(artifact_path),
        EVAL_ARTIFACT_SHA256_KEY: artifact_sha256,
    }


def _collect_dataset_labels(dataset) -> torch.Tensor:
    """Collect labels from a dataset into a flat tensor."""
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if "label" not in sample:
            continue
        labels.append(torch.as_tensor(sample["label"], dtype=torch.float32).reshape(-1))

    if not labels:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(labels, dim=0)


def compute_baseline_metrics(datamodule, task_name: str, task_type: str = "binary") -> dict:
    """Compute baseline metrics for comparison.

    Baselines are fit on the train split and evaluated on the test split.

    For classification tasks, computes AUROC for random predictions and
    majority-class predictions. For regression tasks, computes train-fit
    mean/median predictor baselines on the test labels.
    """
    del task_name
    baselines = {}

    train_dataset = datamodule.train_dataloader().dataset
    test_dataset = datamodule.test_dataloader().dataset

    train_labels = _collect_dataset_labels(train_dataset)
    test_labels = _collect_dataset_labels(test_dataset)

    if len(train_labels) == 0 or len(test_labels) == 0:
        return baselines

    n_samples = len(test_labels)
    baselines["test/n_samples"] = n_samples

    if task_type == "regression":
        mean_label = train_labels.mean().item()
        median_label = train_labels.median().item()
        baselines["baseline/train_label_mean"] = mean_label
        baselines["baseline/train_label_std"] = train_labels.std(unbiased=False).item()
        baselines["baseline/train_label_median"] = median_label
        baselines["baseline/mean_predictor_mse"] = (test_labels - mean_label).pow(2).mean().item()
        baselines["baseline/median_predictor_mae"] = (
            (test_labels - median_label).abs().mean().item()
        )
    else:
        n_positive = test_labels.sum().item()
        n_negative = n_samples - n_positive
        positive_ratio = n_positive / n_samples
        train_positive_ratio = train_labels.mean().item()
        majority_class = 1.0 if train_positive_ratio >= 0.5 else 0.0

        baselines["test/n_positive"] = n_positive
        baselines["test/n_negative"] = n_negative
        baselines["test/positive_ratio"] = positive_ratio
        baselines["baseline/train_positive_ratio"] = train_positive_ratio
        baselines["baseline/random_auroc"] = 0.5
        majority_class_accuracy = (test_labels == majority_class).float().mean().item()
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
    task_name_for_validation = cfg.task.get("task_name", "mortality_24h")
    validate_data_prerequisites(
        cfg.data.processed_dir,
        cfg.dataset,
        task_names=[task_name_for_validation],
        task_configs=[cfg.task],
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
    train_label_stats = datamodule.get_train_label_statistics()
    task_type = cfg.task.get("task_type", "binary")
    if task_name in label_stats:
        stats = label_stats[task_name]
        print(f"\n Label distribution for '{task_name}':")
        print(f"  - Total samples: {stats['total']}")
        if stats.get("task_type") == "regression":
            print(f"  - Mean: {stats.get('mean', 0.0):.4f}")
            print(f"  - Std:  {stats.get('std', 0.0):.4f}")
            print(f"  - Min:  {stats.get('min', 0.0):.4f}")
            print(f"  - Max:  {stats.get('max', 0.0):.4f}")
        else:
            print(
                f"  - Positive: {stats.get('positive', 'N/A')} "
                f"({stats.get('prevalence', 0)*100:.1f}%)"
            )
            print(
                f"  - Negative: {stats.get('negative', 'N/A')} "
                f"({(1 - stats.get('prevalence', 0))*100:.1f}%)"
            )

    train_support_stats = report_and_validate_train_label_support(
        datamodule=datamodule,
        task_name=task_name,
        task_type=task_type,
        dataset=cfg.dataset,
        seed=cfg.seed,
        label_fraction=cfg.get("label_fraction", 1.0),
        min_train_positives=cfg.get("min_train_positives", 3),
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
        resolved_class_weight = resolve_balanced_class_weights(
            task_name=task_name,
            task_type=task_type,
            train_label_stats=train_label_stats,
        )
        if resolved_class_weight is not None:
            cfg.training.class_weight = resolved_class_weight
            n_total = int(train_label_stats[task_name]["total"])
            print(f"\n Training-split labels used for class weighting: {n_total}")
            print(f"\n  sqrt(balanced) class weights: {cfg.training.class_weight}")
        else:
            if task_type == "regression":
                print(f"\n  class_weight='balanced' ignored for regression task '{task_name}'")
            else:
                print(
                    f"\n  Warning: No train-split label stats for '{task_name}', "
                    "skipping class weighting"
                )
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
    try:
        logger = setup_wandb_logger(cfg)
    except WandbEntityNotFoundError as exc:
        print(f"\nError: {exc}")
        return
    if logger:
        logger.experiment.summary.update(train_label_support_summary(train_support_stats))

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
    eval_checkpoint_source = "none"
    best_ckpt_load_ok = False
    best_ckpt_error = None

    if best_ckpt:
        print(f"\n Loading best checkpoint: {best_ckpt}")
        try:
            checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print("  - Loaded best checkpoint weights")
            eval_checkpoint_source = "best"
            best_ckpt_load_ok = True
        except Exception as e:
            best_ckpt_error = str(e)
            allow_fallback = cfg.training.get("allow_best_ckpt_fallback", False)
            if allow_fallback:
                print(
                    f"  - WARNING: Could not load best checkpoint ({e}),"
                    " falling back to final model"
                )
                eval_checkpoint_source = "final"
            else:
                if logger:
                    logger.experiment.summary.update(
                        {
                            "_eval_checkpoint_source": "failed",
                            "_best_ckpt_path": best_ckpt,
                            "_best_ckpt_load_ok": False,
                            "_best_ckpt_error": best_ckpt_error,
                        }
                    )
                    logger.experiment.finish()
                raise RuntimeError(
                    f"Best checkpoint load failed: {e}. "
                    f"Set training.allow_best_ckpt_fallback=true to use final model instead."
                ) from e
    else:
        eval_checkpoint_source = "final"

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

    # Log test results, baseline metrics, and checkpoint provenance to W&B summary
    if logger:
        if test_results:
            logger.experiment.summary.update(test_results[0])
            logger.experiment.summary.update(
                {
                    "_eval_checkpoint_source": eval_checkpoint_source,
                    **_evaluated_artifact_metadata(cfg, eval_checkpoint_source, best_ckpt),
                    "_best_ckpt_path": best_ckpt or "",
                    "_best_ckpt_load_ok": best_ckpt_load_ok,
                    "_best_ckpt_error": best_ckpt_error or "",
                }
            )
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
