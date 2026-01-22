"""Factory for creating encoder architectures.

Provides a unified interface for instantiating different encoder architectures
with proper configuration validation.
"""

from typing import Any, Dict, Type

from .base import BaseEncoder, EncoderConfig
from .linear import LinearConfig, LinearEncoder
from .transformer import TransformerConfig, TransformerEncoder

# Registry of available encoders
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    "transformer": TransformerEncoder,
    "linear": LinearEncoder,
}

# Registry of encoder configs
ENCODER_CONFIG_REGISTRY: Dict[str, Type[EncoderConfig]] = {
    "transformer": TransformerConfig,
    "linear": LinearConfig,
}


def build_encoder(name: str, config_dict: Dict[str, Any]) -> BaseEncoder:
    """Build encoder from name and configuration dictionary.

    Args:
        name: Name of encoder architecture (e.g., 'transformer').
        config_dict: Dictionary with encoder configuration parameters.

    Returns:
        Instantiated encoder.

    Raises:
        ValueError: If encoder name is not registered.

    Example:
        >>> config = {
        ...     "d_input": 35,
        ...     "d_model": 128,
        ...     "n_layers": 4,
        ...     "n_heads": 8,
        ...     "pooling": "none"
        ... }
        >>> encoder = build_encoder("transformer", config)
    """
    if name not in ENCODER_REGISTRY:
        available = ", ".join(ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. " f"Available encoders: {available}")

    # Get config class and create config instance
    config_cls = ENCODER_CONFIG_REGISTRY[name]
    config = config_cls(**config_dict)

    # Build encoder
    encoder_cls = ENCODER_REGISTRY[name]
    return encoder_cls(config)


def get_encoder_config_class(name: str) -> Type[EncoderConfig]:
    """Get encoder config class by name.

    Args:
        name: Name of encoder architecture.

    Returns:
        Config class for the encoder.

    Raises:
        ValueError: If encoder name is not registered.
    """
    if name not in ENCODER_CONFIG_REGISTRY:
        available = ", ".join(ENCODER_CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown encoder '{name}'. " f"Available encoders: {available}")

    return ENCODER_CONFIG_REGISTRY[name]
