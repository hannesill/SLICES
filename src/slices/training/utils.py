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
        auto_insert_metric_name=False,
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

    # Replace '/' with '_' in monitor name for safe filenames.
    # Lightning interprets '/' as a directory separator in checkpoint filenames,
    # which causes best checkpoints to silently not save (empty directories).
    safe_monitor = monitor.replace("/", "_")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "checkpoints"),
        filename=f"{checkpoint_prefix}-{{epoch:03d}}-{{{safe_monitor}:.4f}}",
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
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
    if cfg.get("revision") is not None:
        tags.append(f"revision:{cfg.revision}")
    if cfg.get("rerun_reason") is not None:
        tag = f"rerun-reason:{cfg.rerun_reason}"
        if len(tag) > 64:
            tag = tag[:61] + "..."
        tags.append(tag)

    # Add protocol tag for finetune runs (Protocol A = frozen, Protocol B = unfrozen)
    freeze_encoder = cfg.get("training", {}).get("freeze_encoder")
    if freeze_encoder is not None:
        protocol = "A" if freeze_encoder else "B"
        tags.append(f"protocol:{protocol}")

    # Add mask_ratio tag for pretrain runs (useful for ablation filtering)
    ssl_cfg = cfg.get("ssl", {})
    if ssl_cfg and ssl_cfg.get("mask_ratio") is not None:
        tags.append(f"mask_ratio:{ssl_cfg.mask_ratio}")

    # Add label_fraction tag when subsampling training data
    label_fraction = cfg.get("label_fraction")
    if label_fraction is not None and label_fraction < 1.0:
        tags.append(f"label_fraction:{label_fraction}")

    tags = tags or None

    # Adjust run name: use "probe" prefix instead of "finetune" for frozen encoder
    run_name = cfg.logging.get("run_name", None)
    if run_name and freeze_encoder is True:
        run_name = run_name.replace("_finetune_", "_probe_", 1)

    # Adjust group to include protocol and label_fraction so that W&B "Group" view
    # aggregates exactly the runs that differ only by seed.
    group = cfg.logging.get("wandb_group", None)
    if group:
        if freeze_encoder is True:
            group = group.replace("finetune_", "probe_", 1)
        if label_fraction is not None and label_fraction < 1.0:
            frac_str = str(label_fraction).replace(".", "")
            group += f"_frac{frac_str}"

    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.get("wandb_entity", None),
        name=run_name,
        group=group,
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
    from slices.eval.inference import run_inference

    predictions, labels_tensor, all_stay_ids = run_inference(
        model,
        datamodule.test_dataloader(),
        device=model.device,
    )

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


def report_and_validate_train_label_support(
    *,
    datamodule: Any,
    task_name: str,
    task_type: str,
    dataset: str,
    seed: int,
    label_fraction: float,
    min_train_positives: int = 3,
) -> Dict[str, Any]:
    """Report effective train-set class support and fail on degenerate subsets.

    This is primarily aimed at label-efficiency runs where aggressive
    subsampling can leave the optimization subset with too few positive
    examples to produce meaningful metrics.
    """
    if task_type != "binary":
        return {}

    subset_stats = datamodule.get_train_label_statistics()
    full_stats = datamodule.get_train_label_statistics(use_full_train=True)

    if task_name not in subset_stats:
        return {}

    subset = subset_stats[task_name]
    full = full_stats.get(task_name, subset)

    subset_total = int(subset.get("total", 0))
    subset_pos = int(subset.get("positive", 0))
    subset_neg = int(subset.get("negative", 0))
    subset_prev = float(subset.get("prevalence", 0.0))

    full_total = int(full.get("total", 0))
    full_pos = int(full.get("positive", 0))
    full_neg = int(full.get("negative", 0))
    full_prev = float(full.get("prevalence", 0.0))

    print("\n Train label support:")
    print(
        "  - Dataset / task / seed / fraction: "
        f"{dataset} / {task_name} / {seed} / {label_fraction}"
    )
    print(
        f"  - Full train split: {full_pos} positive, {full_neg} negative, "
        f"{full_total} total ({full_prev * 100:.2f}% positive)"
    )
    print(
        f"  - Optimization subset: {subset_pos} positive, {subset_neg} negative, "
        f"{subset_total} total ({subset_prev * 100:.2f}% positive)"
    )

    if subset_pos == 0 or subset_neg == 0:
        raise ValueError(
            "Binary training subset lost a class: "
            f"dataset={dataset}, task={task_name}, seed={seed}, "
            f"label_fraction={label_fraction}, positives={subset_pos}, negatives={subset_neg}. "
            "Adjust the fraction or seed before training."
        )

    if label_fraction < 1.0 and subset_pos < min_train_positives:
        raise ValueError(
            "Too few positive training examples in the label-efficiency subset: "
            f"dataset={dataset}, task={task_name}, seed={seed}, "
            f"label_fraction={label_fraction}, positives={subset_pos} < {min_train_positives}. "
            "Increase the fraction or drop this run."
        )

    return {
        "task_name": task_name,
        "dataset": dataset,
        "seed": seed,
        "label_fraction": label_fraction,
        "full_train_total": full_total,
        "full_train_positive": full_pos,
        "full_train_negative": full_neg,
        "full_train_prevalence": full_prev,
        "train_subset_total": subset_total,
        "train_subset_positive": subset_pos,
        "train_subset_negative": subset_neg,
        "train_subset_prevalence": subset_prev,
    }


def validate_data_prerequisites(
    processed_dir: str,
    dataset: str,
    task_names: Optional[List[str]] = None,
) -> None:
    """Validate that required data files exist before training.

    Checks file existence and, if task_names are provided, validates the label
    manifest in metadata.yaml to ensure labels were built with the current
    builder version and task config.

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If label manifest indicates stale labels.
    """

    import yaml

    from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig

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

    # Validate label manifest if task_names are provided
    if task_names:
        metadata_path = path / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.yaml not found in {path} — cannot validate label freshness.\n"
                f"Re-run extraction: uv run python scripts/preprocessing/"
                f"extract_ricu.py dataset={dataset}"
            )

        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        label_manifest = metadata.get("label_manifest")
        if label_manifest is None:
            raise RuntimeError(
                f"metadata.yaml in {path} has no label_manifest. "
                "Labels were extracted before manifest support was added.\n"
                f"Re-run extraction: uv run python scripts/preprocessing/"
                f"extract_ricu.py dataset={dataset}"
            )

        # Load current task configs and compare against manifest
        tasks_path = Path(__file__).parent.parent / "data" / "tasks"
        for task_name in task_names:
            config_file = tasks_path / f"{task_name}.yaml"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Task config not found for '{task_name}': {config_file}. "
                    "Cannot validate label freshness safely."
                )

            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
            current_config = LabelConfig(**config_dict)
            current_hash = LabelBuilder.config_hash(current_config)

            builder = LabelBuilderFactory.create(current_config)
            current_version = builder.SEMANTIC_VERSION

            manifest_entry = label_manifest.get(task_name)
            if manifest_entry is None:
                raise RuntimeError(
                    f"Task '{task_name}' not found in label manifest. "
                    "Labels were extracted without this task.\n"
                    f"Re-run extraction: uv run python scripts/preprocessing/"
                    f"extract_ricu.py dataset={dataset}"
                )

            stored_version = manifest_entry.get("builder_version")
            stored_hash = manifest_entry.get("config_hash")

            if stored_version != current_version:
                raise RuntimeError(
                    f"Label builder version mismatch for task '{task_name}': "
                    f"stored={stored_version}, current={current_version}. "
                    f"Re-run extraction: uv run python scripts/preprocessing/"
                    f"extract_ricu.py dataset={dataset}"
                )

            if stored_hash != current_hash:
                raise RuntimeError(
                    f"Task config changed for '{task_name}': "
                    f"stored_hash={stored_hash}, current_hash={current_hash}. "
                    f"Re-run extraction: uv run python scripts/preprocessing/"
                    f"extract_ricu.py dataset={dataset}"
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
