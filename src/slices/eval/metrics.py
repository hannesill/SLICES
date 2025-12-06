"""Configurable metrics for evaluation.

This module provides a factory for building torchmetrics collections
based on task type and configuration. Designed to be minimal but extendable.

Example:
    >>> config = MetricConfig(
    ...     task_type="binary",
    ...     metrics=["auroc", "auprc", "accuracy"],
    ... )
    >>> metrics = build_metrics(config, prefix="val")
    >>> metrics.update(probs, labels)
    >>> results = metrics.compute()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score, MetricCollection


# Supported task types
TaskType = Literal["binary", "multiclass", "multilabel", "regression"]

# Available metrics per task type
AVAILABLE_METRICS = {
    "binary": ["auroc", "auprc", "accuracy", "f1"],
    "multiclass": ["auroc", "auprc", "accuracy", "f1"],
    "multilabel": ["auroc", "auprc", "accuracy", "f1"],
    "regression": [],  # TODO: Add MSE, MAE, R2 when needed
}

# Default metrics per task type (minimal set)
DEFAULT_METRICS = {
    "binary": ["auroc", "auprc"],
    "multiclass": ["auroc", "accuracy"],
    "multilabel": ["auroc"],
    "regression": [],
}


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics.
    
    Attributes:
        task_type: Type of prediction task.
        n_classes: Number of classes (for classification tasks).
        metrics: List of metric names to compute. If None, uses defaults.
        threshold: Decision threshold for binary classification (default: 0.5).
    """
    task_type: TaskType
    n_classes: int = 2
    metrics: Optional[List[str]] = None
    threshold: float = 0.5
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.task_type not in AVAILABLE_METRICS:
            raise ValueError(
                f"Unknown task_type '{self.task_type}'. "
                f"Available: {list(AVAILABLE_METRICS.keys())}"
            )
        
        # Use defaults if not specified
        if self.metrics is None:
            self.metrics = DEFAULT_METRICS[self.task_type].copy()
        
        # Validate requested metrics
        available = AVAILABLE_METRICS[self.task_type]
        for metric in self.metrics:
            if metric not in available:
                raise ValueError(
                    f"Metric '{metric}' not available for task_type '{self.task_type}'. "
                    f"Available: {available}"
                )


def _build_metric(
    name: str,
    task_type: TaskType,
    n_classes: int,
) -> torch.nn.Module:
    """Build a single metric instance.
    
    Args:
        name: Metric name (auroc, auprc, accuracy, f1).
        task_type: Task type for metric configuration.
        n_classes: Number of classes.
    
    Returns:
        Configured torchmetrics instance.
    """
    # Map task_type to torchmetrics task parameter
    if task_type == "binary":
        task = "binary"
        kwargs = {"num_classes": n_classes}
    elif task_type == "multiclass":
        task = "multiclass"
        kwargs = {"num_classes": n_classes}
    elif task_type == "multilabel":
        task = "multilabel"
        kwargs = {"num_labels": n_classes}
    else:
        raise ValueError(f"Unsupported task_type for classification: {task_type}")
    
    # Build the metric
    if name == "auroc":
        return AUROC(task=task, **kwargs)
    elif name == "auprc":
        return AveragePrecision(task=task, **kwargs)
    elif name == "accuracy":
        return Accuracy(task=task, **kwargs)
    elif name == "f1":
        return F1Score(task=task, **kwargs)
    else:
        raise ValueError(f"Unknown metric: {name}")


def build_metrics(
    config: MetricConfig,
    prefix: str = "",
) -> MetricCollection:
    """Build a MetricCollection from configuration.
    
    Args:
        config: Metric configuration.
        prefix: Prefix for metric names (e.g., "val" -> "val/auroc").
    
    Returns:
        MetricCollection with configured metrics.
    
    Example:
        >>> config = MetricConfig(task_type="binary", metrics=["auroc", "auprc"])
        >>> val_metrics = build_metrics(config, prefix="val")
        >>> test_metrics = build_metrics(config, prefix="test")
    """
    if config.task_type == "regression":
        # TODO: Add regression metrics when needed
        return MetricCollection({}, prefix=f"{prefix}/" if prefix else "")
    
    metrics = {}
    for name in config.metrics:
        metrics[name] = _build_metric(name, config.task_type, config.n_classes)
    
    return MetricCollection(metrics, prefix=f"{prefix}/" if prefix else "")


def get_default_metrics(
    task_type: TaskType,
    n_classes: int = 2,
    prefix: str = "",
) -> MetricCollection:
    """Get default metrics for a task type.
    
    Convenience function for quick metric setup with sensible defaults.
    
    Args:
        task_type: Type of prediction task.
        n_classes: Number of classes.
        prefix: Prefix for metric names.
    
    Returns:
        MetricCollection with default metrics.
    
    Example:
        >>> val_metrics = get_default_metrics("binary", prefix="val")
        >>> test_metrics = get_default_metrics("binary", prefix="test")
    """
    config = MetricConfig(task_type=task_type, n_classes=n_classes)
    return build_metrics(config, prefix=prefix)
