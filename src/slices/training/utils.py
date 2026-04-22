"""Training utilities and helpers.

Shared optimizer/scheduler construction for pretrain and finetune modules,
shared callback/logger setup, fairness evaluation, and checkpoint save helper.
"""

import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig

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
        filename="ssl-{epoch:03d}",
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

    Defaults to the task's declared primary metric when available; otherwise
    falls back to val/auprc for classification and val/mse for regression.
    """
    callbacks = []

    lower_is_better = {"loss", "mse", "mae", "rmse", "brier_score", "ece"}
    higher_is_better = {
        "auroc",
        "auprc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "r2",
    }

    task_type = cfg.task.get("task_type", "binary")
    default_metric = cfg.task.get("primary_metric", None)
    if default_metric:
        default_monitor = default_metric if "/" in default_metric else f"val/{default_metric}"
    elif task_type == "regression":
        default_monitor = "val/mse"
    else:
        default_monitor = "val/auprc"

    monitor = cfg.training.get("early_stopping_monitor", default_monitor)
    metric_name = monitor.split("/", 1)[-1]
    if metric_name in lower_is_better:
        default_mode = "min"
    elif metric_name in higher_is_better:
        default_mode = "max"
    else:
        raise ValueError(
            f"Cannot infer checkpoint mode for finetune monitor '{monitor}'. "
            "Set training.early_stopping_mode explicitly."
        )

    mode = cfg.training.get("early_stopping_mode", default_mode)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "checkpoints"),
        filename=f"{checkpoint_prefix}-{{epoch:03d}}",
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


def _add_wandb_tag(tags: list[str], tag: str) -> None:
    """Append a W&B tag once, preserving caller order."""
    if tag not in tags:
        tags.append(tag)


def setup_wandb_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    """Set up W&B experiment logger.

    Returns None if wandb is disabled.
    """
    if not cfg.logging.get("use_wandb", False):
        return None

    tags = list(cfg.logging.get("wandb_tags", []))
    if cfg.get("sprint") is not None:
        _add_wandb_tag(tags, f"sprint:{cfg.sprint}")
    if cfg.get("revision") is not None:
        _add_wandb_tag(tags, f"revision:{cfg.revision}")
    if cfg.get("rerun_reason") is not None:
        tag = f"rerun-reason:{cfg.rerun_reason}"
        if len(tag) > 64:
            tag = tag[:61] + "..."
        _add_wandb_tag(tags, tag)
    if cfg.get("launch_commit") is not None:
        _add_wandb_tag(tags, f"commit:{str(cfg.launch_commit)[:12]}")
    model_size = cfg.get("model_size")
    if model_size is not None:
        _add_wandb_tag(tags, f"model_size:{model_size}")

    # Add downstream protocol family tag. Supervised-from-scratch shares the
    # Protocol B optimization budget; the phase tag distinguishes it from SSL.
    freeze_encoder = cfg.get("training", {}).get("freeze_encoder")
    if freeze_encoder is not None:
        protocol = "A" if freeze_encoder else "B"
        _add_wandb_tag(tags, f"protocol:{protocol}")

    # Add mask_ratio tag for pretrain runs (useful for ablation filtering)
    ssl_cfg = cfg.get("ssl", {})
    if ssl_cfg and ssl_cfg.get("mask_ratio") is not None:
        _add_wandb_tag(tags, f"mask_ratio:{ssl_cfg.mask_ratio}")

    # Add label_fraction tag when subsampling training data
    label_fraction = cfg.get("label_fraction")
    if label_fraction is not None and label_fraction < 1.0:
        _add_wandb_tag(tags, f"label_fraction:{label_fraction}")
        _add_wandb_tag(tags, "ablation:label-efficiency")

    if cfg.get("source_dataset") is not None:
        _add_wandb_tag(tags, "ablation:transfer")

    tags = tags or None

    # Adjust run name: use "probe" prefix instead of "finetune" for frozen encoder
    run_name = cfg.logging.get("run_name", None)
    if run_name and freeze_encoder is True:
        run_name = run_name.replace("_finetune_", "_probe_", 1)
    if run_name and model_size is not None:
        run_name += f"_{model_size}"

    # Adjust group to include protocol and label_fraction so that W&B "Group" view
    # aggregates exactly the runs that differ only by seed.
    group = cfg.logging.get("wandb_group", None)
    if group:
        if freeze_encoder is True:
            group = group.replace("finetune_", "probe_", 1)
        if model_size is not None:
            group += f"_{model_size}"
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

    from slices.eval.fairness_evaluator import FairnessEvaluator, flatten_fairness_report
    from slices.eval.inference import run_inference

    predictions, labels_tensor, all_stay_ids = run_inference(
        model,
        datamodule.test_dataloader(),
        device=model.device,
    )
    task_type = cfg.get("task", {}).get("task_type", "binary")

    evaluator = FairnessEvaluator(
        static_df=datamodule.dataset.static_df,
        protected_attributes=list(
            fairness_cfg.get("protected_attributes", ["gender", "age_group"])
        ),
        min_subgroup_size=fairness_cfg.get("min_subgroup_size", 50),
        task_type=task_type,
        dataset_name=getattr(getattr(datamodule, "processed_dir", None), "name", None),
    )
    fairness_report = evaluator.evaluate(predictions, labels_tensor, all_stay_ids)
    evaluator.print_report(fairness_report)

    if logger:
        for key, value in flatten_fairness_report(fairness_report).items():
            logger.experiment.summary[key] = value

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
        f"  - Dataset / task / seed / fraction: {dataset} / {task_name} / {seed} / {label_fraction}"
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


def resolve_balanced_class_weights(
    task_name: str,
    task_type: str,
    train_label_stats: Dict[str, Dict[str, Any]],
) -> Optional[List[float]]:
    """Resolve ``class_weight='balanced'`` for supported task types.

    Only binary classification is supported. Regression returns ``None`` because
    class weights do not apply. Other task types fail closed instead of
    constructing incorrect weights.
    """
    normalized_task_type = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "multilabel_classification": "multilabel",
    }.get(task_type, task_type)

    if normalized_task_type == "regression":
        return None

    if normalized_task_type != "binary":
        raise ValueError(
            f"class_weight='balanced' is only supported for binary tasks, got "
            f"task_type='{task_type}'. Set class_weight=null or provide explicit weights."
        )

    if task_name not in train_label_stats:
        return None

    stats = train_label_stats[task_name]
    n_pos = int(stats.get("positive", 0))
    n_neg = int(stats.get("negative", 0))
    n_total = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Cannot compute balanced class weights for '{task_name}': "
            f"{n_pos} positive, {n_neg} negative. Check label extraction."
        )

    raw = [n_total / (2 * n_neg), n_total / (2 * n_pos)]
    return [w**0.5 for w in raw]


def validate_data_prerequisites(
    processed_dir: str,
    dataset: str,
    task_names: Optional[List[str]] = None,
    task_configs: Optional[List[Union[LabelConfig, DictConfig, Dict[str, Any]]]] = None,
) -> None:
    """Validate that required data files exist before training.

    Checks file existence and, if task definitions are provided, validates the
    label manifest in metadata.yaml to ensure labels were built with the
    current builder version and task config.

    When ``task_configs`` are supplied, they are treated as the source of truth
    because they represent the active Hydra-composed task configuration for the
    run. ``task_names`` remain as a fallback for callers that only know names.

    Raises:
        FileNotFoundError: If required files are missing.
        RuntimeError: If label manifest indicates stale labels.
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

    label_config_fields = {field.name for field in fields(LabelConfig)}

    def coerce_label_config(
        task_config: Union[LabelConfig, DictConfig, Dict[str, Any]],
    ) -> LabelConfig:
        if isinstance(task_config, LabelConfig):
            return task_config

        if isinstance(task_config, DictConfig):
            raw_config = OmegaConf.to_container(task_config, resolve=True)
        else:
            raw_config = dict(task_config)

        if not isinstance(raw_config, dict):
            raise TypeError("Task configuration must resolve to a mapping.")

        label_config_dict = {
            key: value for key, value in raw_config.items() if key in label_config_fields
        }
        return LabelConfig(**label_config_dict)

    def get_training_tasks_path() -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        hydra_tasks_path = repo_root / "configs" / "tasks"
        if hydra_tasks_path.exists():
            return hydra_tasks_path
        return Path(__file__).resolve().parents[1] / "data" / "tasks"

    resolved_task_configs: List[LabelConfig] = []
    if task_configs is not None:
        resolved_task_configs = [coerce_label_config(task_config) for task_config in task_configs]
        if task_names is not None:
            resolved_names = {task_config.task_name for task_config in resolved_task_configs}
            requested_names = set(task_names)
            if resolved_names != requested_names:
                raise ValueError(
                    "task_names does not match task_configs: "
                    f"task_names={sorted(requested_names)}, "
                    f"task_configs={sorted(resolved_names)}"
                )
    elif task_names:
        tasks_path = get_training_tasks_path()
        for task_name in task_names:
            config_file = tasks_path / f"{task_name}.yaml"
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Task config not found for '{task_name}': {config_file}. "
                    "Cannot validate label freshness safely."
                )

            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
            resolved_task_configs.append(coerce_label_config(config_dict))

    # Validate label manifest if task configs are provided or can be resolved.
    if resolved_task_configs:
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

        for current_config in resolved_task_configs:
            task_name = current_config.task_name
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
