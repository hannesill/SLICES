"""Training utilities and helpers.

Shared optimizer/scheduler construction for pretrain and finetune modules,
shared callback/logger setup, fairness evaluation, and checkpoint save helper.
"""

import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# =============================================================================
# Optimizer / Scheduler
# =============================================================================


def build_optimizer(
    params: Union[Iterator[nn.Parameter], List[Dict[str, Any]]],
    config: Any,
) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Args:
        params: Model parameters to optimize. Can be an iterator of
                parameters or a list of param group dicts.
        config: Optimizer config with 'name', 'lr', and optional
                'weight_decay', 'momentum' fields.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    name = config.name.lower()
    lr = config.lr
    weight_decay = config.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = config.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Supported: adam, adamw, sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
) -> Optional[Dict[str, Any]]:
    """Build learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Scheduler config with 'name' and scheduler-specific fields.
                If None, returns None.

    Returns:
        Lightning-compatible scheduler dict, or None if no scheduler.

    Raises:
        ValueError: If scheduler name is not recognized.
    """
    if config is None:
        return None

    name = config.name.lower()

    if name == "cosine":
        T_max = config.get("T_max", 100)
        eta_min = config.get("eta_min", 0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif name == "step":
        step_size = config.get("step_size", 30)
        gamma = config.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif name == "plateau":
        mode = config.get("mode", "min")
        factor = config.get("factor", 0.1)
        patience = config.get("patience", 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.get("monitor", "val/loss"),
            },
        }
    elif name == "warmup_cosine":
        warmup_epochs = config.get("warmup_epochs", 10)
        max_epochs = config.get("max_epochs", 100)
        eta_min = config.get("eta_min", 0.0)

        # Get base_lr to make eta_min absolute (consistent with CosineAnnealingLR).
        # LambdaLR multiplies the lambda return by base_lr, so we divide eta_min
        # by base_lr here so the effective minimum LR equals eta_min exactly.
        base_lr = optimizer.defaults["lr"]
        eta_min_ratio = eta_min / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
                return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (
                    1.0 + math.cos(progress * math.pi)
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported: cosine, step, plateau, warmup_cosine"
        )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        },
    }


# =============================================================================
# Callbacks
# =============================================================================


def setup_pretrain_callbacks(cfg: DictConfig) -> list:
    """Set up training callbacks for SSL pretraining.

    Monitors val/loss (minimise).
    """
    callbacks = []

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

    if cfg.training.get("early_stopping_patience", None):
        early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


def setup_finetune_callbacks(cfg: DictConfig, checkpoint_prefix: str = "finetune") -> list:
    """Set up training callbacks for finetuning / supervised training.

    Task-type-aware: monitors val/auprc (max) for classification,
    val/mse (min) for regression.
    """
    callbacks = []

    task_type = cfg.task.get("task_type", "binary")
    if task_type == "regression":
        default_monitor, default_mode = "val/mse", "min"
    else:
        default_monitor, default_mode = "val/auprc", "max"
    monitor = cfg.training.get("early_stopping_monitor", default_monitor)
    mode = cfg.training.get("early_stopping_mode", default_mode)

    metric_filename = monitor.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "checkpoints"),
        filename=f"{checkpoint_prefix}-{{epoch:03d}}-{{{metric_filename}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    if cfg.training.get("early_stopping_patience", None):
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=cfg.training.early_stopping_patience,
            mode=mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    return callbacks


# =============================================================================
# Logger
# =============================================================================


def setup_wandb_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Set up W&B experiment logger.

    Returns None if wandb is disabled.
    """
    if not cfg.logging.get("use_wandb", False):
        return None

    tags = list(cfg.logging.get("wandb_tags", []))
    if cfg.get("sprint") is not None:
        tags.append(f"sprint:{cfg.sprint}")
    tags = tags or None

    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.get("wandb_entity", None),
        name=cfg.logging.get("run_name", None),
        group=cfg.logging.get("wandb_group", None),
        tags=tags,
        save_dir=cfg.output_dir,
        log_model=False,
    )

    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    return logger


# =============================================================================
# Fairness evaluation
# =============================================================================


def run_fairness_evaluation(
    model: nn.Module,
    datamodule: Any,
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
) -> Optional[dict]:
    """Run fairness evaluation on the test set.

    Returns the fairness report dict, or None if fairness eval is disabled.
    """
    fairness_cfg = cfg.get("eval", {}).get("fairness", {})
    if not fairness_cfg.get("enabled", False):
        return None

    print("\n" + "=" * 80)
    print("Fairness Evaluation")
    print("=" * 80)

    from slices.eval.fairness_evaluator import FairnessEvaluator

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
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, (int, float)):
                            logger.experiment.summary[
                                f"fairness/{attr}/{metric_name}/{sub_key}"
                            ] = sub_val

    return fairness_report


# =============================================================================
# Data validation
# =============================================================================


def validate_data_prerequisites(processed_dir: str, dataset: str) -> None:
    """Validate that required data files exist before training.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    path = Path(processed_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed directory not found: {path}\n"
            f"Run first: uv run python scripts/preprocessing/extract_ricu.py dataset={dataset}"
        )

    splits_path = path / "splits.yaml"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.yaml not found in {path}\n"
            f"Run first: uv run python scripts/preprocessing/prepare_dataset.py dataset={dataset}"
        )

    stats_path = path / "normalization_stats.yaml"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"normalization_stats.yaml not found in {path}\n"
            f"Run first: uv run python scripts/preprocessing/prepare_dataset.py dataset={dataset}"
        )


# =============================================================================
# Checkpoint saving
# =============================================================================


def save_encoder_checkpoint(
    encoder: nn.Module,
    encoder_config: Dict[str, Any],
    path: Union[str, Path],
    missing_token: Optional[torch.Tensor] = None,
    d_input: Optional[int] = None,
    ssl_name: Optional[str] = None,
) -> None:
    """Save encoder in v3 checkpoint format.

    Standardized checkpoint saving used across pretrain, finetune, and supervised
    training scripts.

    Args:
        encoder: Encoder module whose state_dict to save.
        encoder_config: Dict with 'name' and encoder architecture params.
        path: Path to save the checkpoint.
        missing_token: Optional learned missing token tensor.
        d_input: Optional input dimension (for token shape validation).
        ssl_name: Optional SSL paradigm name (e.g. 'mae', 'jepa', 'contrastive')
                  for automatic paradigm detection during finetuning.
    """
    checkpoint: Dict[str, Any] = {
        "encoder_state_dict": encoder.state_dict(),
        "encoder_config": encoder_config,
        "version": 3,
    }

    if missing_token is not None:
        checkpoint["missing_token"] = missing_token.data.clone()
        if d_input is not None:
            checkpoint["d_input"] = d_input

    if ssl_name is not None:
        checkpoint["ssl_name"] = ssl_name

    torch.save(checkpoint, path)
