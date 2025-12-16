"""Task heads for downstream clinical prediction tasks.

This module provides lightweight classification/regression heads that sit on top
of an encoder. The encoder-head composition is handled by the FineTuneModule.

Available task heads:
- MLPTaskHead: Multi-layer perceptron (configurable hidden layers)
- LinearTaskHead: Single linear layer (for linear probing)

Example:
    >>> from slices.models.heads import build_task_head, TaskHeadConfig
    >>>
    >>> config = TaskHeadConfig(
    ...     name="mlp",
    ...     task_name="mortality_24h",
    ...     task_type="binary",
    ...     n_classes=None,
    ...     input_dim=128,
    ...     hidden_dims=[64],
    ... )
    >>> head = build_task_head(config)
"""

from .base import BaseTaskHead, TaskHeadConfig
from .factory import (
    build_task_head,
    build_task_head_from_dict,
    get_available_task_heads,
)
from .mlp import LinearTaskHead, MLPTaskHead

__all__ = [
    # Base classes
    "BaseTaskHead",
    "TaskHeadConfig",
    # Concrete implementations
    "MLPTaskHead",
    "LinearTaskHead",
    # Factory functions
    "build_task_head",
    "build_task_head_from_dict",
    "get_available_task_heads",
]
