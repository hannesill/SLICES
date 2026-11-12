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
from slices.training import FineTuneModule
from slices.training.utils import (
    report_and_validate_train_label_support,
    resolve_balanced_class_weights,
    run_fairness_evaluation,
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


def _detect_paradigm_from_checkpoint(path: str, *, full_checkpoint: bool) -> str | None:
    """Infer the SSL paradigm recorded in an encoder or Lightning checkpoint."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=not full_checkpoint,
    )

    if not isinstance(checkpoint, dict):
        return None

    if "ssl_name" in checkpoint:
        return checkpoint["ssl_name"]

    if not full_checkpoint:
        return None

    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    config = hyper_parameters.get("config") or {}
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    if not isinstance(config, dict):
        return None

    ssl_config = config.get("ssl") or {}
    if isinstance(ssl_config, DictConfig):
        ssl_config = OmegaConf.to_container(ssl_config, resolve=True)

    if isinstance(ssl_config, dict) and ssl_config.get("name") is not None:
        return str(ssl_config["name"])

    paradigm = config.get("paradigm")
    return str(paradigm) if paradigm is not None else None


@hydra.main(version_base=None, config_path="../../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Run downstream task finetuning."""
    print("=" * 80)
    print("Downstream Task Finetuning Pipeline")
    print("=" * 80)

    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Validate checkpoint source.
    if cfg.checkpoint is not None and cfg.pretrain_checkpoint is not None:
        raise ValueError(
            "Provide exactly one checkpoint source: use 'checkpoint' for encoder.pt "
            "or 'pretrain_checkpoint' for a full Lightning pretrain .ckpt, not both."
        )
    if cfg.checkpoint is None and cfg.pretrain_checkpoint is None:
        raise ValueError(
            "Must provide either 'checkpoint' (encoder.pt) or "
            "'pretrain_checkpoint' (full .ckpt file)"
        )

    # Auto-detect paradigm from checkpoint metadata
    detected_paradigm = None
    if cfg.checkpoint is not None:
        detected_paradigm = _detect_paradigm_from_checkpoint(
            cfg.checkpoint,
            full_checkpoint=False,
        )
    elif cfg.pretrain_checkpoint is not None:
        detected_paradigm = _detect_paradigm_from_checkpoint(
            cfg.pretrain_checkpoint,
            full_checkpoint=True,
        )

    if detected_paradigm and cfg.paradigm != detected_paradigm:
        print(f"\n  Auto-detected paradigm from checkpoint: {detected_paradigm}")
        OmegaConf.set_struct(cfg, False)
        cfg.paradigm = detected_paradigm
        OmegaConf.set_struct(cfg, True)

    # Validate data prerequisites
    task_name = cfg.task.get("task_name", "mortality_24h")
    validate_data_prerequisites(
        cfg.data.processed_dir,
        cfg.dataset,
        task_names=[task_name],
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
        print(f"  Resuming trainer state from: {ckpt_path}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # =========================================================================
    # 5. Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. Evaluating on Test Set")
    print("=" * 80)

    best_ckpt = callbacks[0].best_model_path if hasattr(callbacks[0], "best_model_path") else None
    eval_checkpoint_source = "none"
    best_ckpt_load_ok = False
    best_ckpt_error = None

    if best_ckpt:
        print(f"\n Best checkpoint: {best_ckpt}")
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
                # Log failure to W&B before crashing
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
        test_results = trainer.test(model, datamodule=datamodule)
    else:
        print("\n  No best checkpoint found, testing with final model")
        eval_checkpoint_source = "final"
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
            logger.experiment.summary.update(
                {
                    "_eval_checkpoint_source": eval_checkpoint_source,
                    **_evaluated_artifact_metadata(cfg, eval_checkpoint_source, best_ckpt),
                    "_best_ckpt_path": best_ckpt or "",
                    "_best_ckpt_load_ok": best_ckpt_load_ok,
                    "_best_ckpt_error": best_ckpt_error or "",
                }
            )

    print(f"\n Output directory: {cfg.output_dir}")
    print(f"  - Checkpoints: {cfg.get('checkpoint_dir', 'checkpoints')}")

    if logger:
        print(
            f"\n View training curves at: https://wandb.ai/{cfg.logging.wandb_entity}/{cfg.logging.wandb_project}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
