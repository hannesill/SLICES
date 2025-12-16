"""Factory for creating task heads.

Provides a unified interface for instantiating different task head architectures
with proper configuration validation.
"""

from typing import Any, Dict, Type

from .base import BaseTaskHead, TaskHeadConfig
from .mlp import LinearTaskHead, MLPTaskHead

# Registry of available task heads
TASK_HEAD_REGISTRY: Dict[str, Type[BaseTaskHead]] = {
    "mlp": MLPTaskHead,
    "linear": LinearTaskHead,
}


def build_task_head(config: TaskHeadConfig) -> BaseTaskHead:
    """Build task head from configuration.

    Args:
        config: Task head configuration.

    Returns:
        Instantiated task head.

    Raises:
        ValueError: If task head name is not registered.

    Example:
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
    if config.name not in TASK_HEAD_REGISTRY:
        available = ", ".join(TASK_HEAD_REGISTRY.keys())
        raise ValueError(f"Unknown task head '{config.name}'. " f"Available heads: {available}")

    head_cls = TASK_HEAD_REGISTRY[config.name]
    return head_cls(config)


def build_task_head_from_dict(config_dict: Dict[str, Any]) -> BaseTaskHead:
    """Build task head from configuration dictionary.

    Convenience function that creates TaskHeadConfig from a dictionary,
    useful for Hydra integration.

    Args:
        config_dict: Dictionary with task head configuration.

    Returns:
        Instantiated task head.

    Example:
        >>> config_dict = {
        ...     "name": "mlp",
        ...     "task_name": "mortality_24h",
        ...     "task_type": "binary",
        ...     "n_classes": None,
        ...     "input_dim": 128,
        ...     "hidden_dims": [64],
        ... }
        >>> head = build_task_head_from_dict(config_dict)
    """
    # Handle hidden_dims if it's not a list (OmegaConf ListConfig)
    if "hidden_dims" in config_dict and not isinstance(config_dict["hidden_dims"], list):
        config_dict["hidden_dims"] = list(config_dict["hidden_dims"])

    config = TaskHeadConfig(**config_dict)
    return build_task_head(config)


def get_available_task_heads() -> list:
    """Get list of available task head names.

    Returns:
        List of registered task head names.
    """
    return list(TASK_HEAD_REGISTRY.keys())
