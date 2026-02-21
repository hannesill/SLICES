"""Factory for creating SSL objectives.

Provides a unified interface for instantiating different SSL objectives
with proper configuration validation.
"""

from typing import Dict, Type

import torch.nn as nn

from .base import BaseSSLObjective, SSLConfig
from .contrastive import ContrastiveConfig, ContrastiveObjective
from .jepa import JEPAConfig, JEPAObjective
from .mae import MAEConfig, MAEObjective
from .smart import SMARTObjective, SMARTSSLConfig

# Registry of available SSL objectives
SSL_REGISTRY: Dict[str, Type[BaseSSLObjective]] = {
    "mae": MAEObjective,
    "smart": SMARTObjective,
    "jepa": JEPAObjective,
    "contrastive": ContrastiveObjective,
}

# Registry of SSL configs
CONFIG_REGISTRY: Dict[str, Type[SSLConfig]] = {
    "mae": MAEConfig,
    "smart": SMARTSSLConfig,
    "jepa": JEPAConfig,
    "contrastive": ContrastiveConfig,
}


def build_ssl_objective(
    encoder: nn.Module,
    config: SSLConfig,
) -> BaseSSLObjective:
    """Build SSL objective from configuration.

    Args:
        encoder: Encoder module to use.
        config: SSL configuration.

    Returns:
        Instantiated SSL objective.

    Raises:
        ValueError: If objective name is not registered.

    Example:
        >>> from slices.models.encoders import TransformerEncoder, TransformerConfig
        >>> from slices.models.pretraining import MAEConfig, build_ssl_objective
        >>>
        >>> enc_config = TransformerConfig(d_input=35, d_model=128, n_layers=4)
        >>> encoder = TransformerEncoder(enc_config)
        >>>
        >>> ssl_config = MAEConfig(mask_ratio=0.15, mask_strategy="block")
        >>> ssl_objective = build_ssl_objective(encoder, ssl_config)
    """
    if config.name not in SSL_REGISTRY:
        available = ", ".join(SSL_REGISTRY.keys())
        raise ValueError(
            f"Unknown SSL objective '{config.name}'. " f"Available objectives: {available}"
        )

    objective_cls = SSL_REGISTRY[config.name]
    return objective_cls(encoder, config)


def get_ssl_config_class(name: str) -> Type[SSLConfig]:
    """Get SSL config class by name.

    Args:
        name: Name of SSL objective.

    Returns:
        Config class for the objective.

    Raises:
        ValueError: If objective name is not registered.
    """
    if name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown SSL objective '{name}'. " f"Available objectives: {available}")

    return CONFIG_REGISTRY[name]
