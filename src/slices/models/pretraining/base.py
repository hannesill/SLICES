"""Abstract base class for self-supervised learning objectives."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, cast

import torch
import torch.nn as nn

from slices.models.encoders.base import SSLTokenizingEncoder


@dataclass
class SSLConfig:
    """Configuration for SSL objective."""

    name: str = "mae"
    # Objective-specific params added in subclasses


class BaseSSLObjective(ABC, nn.Module):
    """Abstract base class for self-supervised learning objectives."""

    def __init__(self, encoder: nn.Module, config: SSLConfig) -> None:
        """Initialize SSL objective with encoder and configuration.

        Args:
            encoder: Encoder module to use for feature extraction.
            config: SSL configuration.
        """
        super().__init__()
        self.encoder = encoder
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) input
        obs_mask: torch.Tensor,  # (B, T, D) observation mask
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute SSL loss.

        Args:
            x: Input tensor of shape (B, T, D).
            obs_mask: Observation mask of shape (B, T, D) where True indicates
                     observed values and False indicates missing/imputed.

        Returns:
            Tuple of:
            - loss: Scalar loss tensor
            - metrics: Dict of additional metrics to log
        """
        pass

    def get_encoder(self) -> nn.Module:
        """Return the encoder for downstream use.

        Returns:
            The encoder module.
        """
        return self.encoder


def require_ssl_tokenizing_encoder(
    encoder: nn.Module,
    objective_name: str,
) -> SSLTokenizingEncoder:
    """Validate and type-narrow the encoder contract used by controlled SSL.

    MAE, JEPA, contrastive, and TS2Vec share a timestep-token interface instead
    of using the pooled downstream encoder forward path. This helper keeps that
    contract in one place while preserving each objective's public behavior.
    """

    missing_methods = [
        method_name
        for method_name in ("tokenize", "encode")
        if not callable(getattr(encoder, method_name, None))
    ]
    if missing_methods:
        missing = ", ".join(missing_methods)
        raise ValueError(
            f"{objective_name} requires an encoder with tokenize()/encode() SSL "
            f"tokenization support, but {type(encoder).__name__} is missing: {missing}"
        )

    encoder_config = getattr(encoder, "config", None)
    if not getattr(encoder_config, "obs_aware", False):
        raise ValueError(
            f"{objective_name} requires an encoder with obs_aware=True "
            "(e.g., TransformerEncoder with obs_aware=True). Got: "
            f"{type(encoder).__name__}"
        )

    encoder_pooling = getattr(encoder_config, "pooling", "none")
    if encoder_pooling != "none":
        raise ValueError(
            f"{objective_name} requires encoder with pooling='none' to get "
            f"per-token representations, but got pooling='{encoder_pooling}'"
        )

    return cast(SSLTokenizingEncoder, encoder)
