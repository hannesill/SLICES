"""Abstract base class for time-series encoders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    """Configuration for encoder architecture."""

    d_input: int = 35  # Number of input features
    d_model: int = 128  # Model dimension
    n_layers: int = 4  # Number of layers
    dropout: float = 0.1
    max_seq_length: int = 168  # 7 days


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for time-series encoders."""

    def __init__(self, config: EncoderConfig) -> None:
        """Initialize encoder with configuration.

        Args:
            config: Encoder configuration.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) input features
        mask: Optional[torch.Tensor] = None,  # (B, T, D) observation mask
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) sequence padding
    ) -> torch.Tensor:
        """Encode input time-series.

        Args:
            x: Input tensor of shape (B, T, D) where B is batch size,
               T is sequence length, and D is feature dimension.
            mask: Optional observation mask of shape (B, T, D) where True
                  indicates observed values and False indicates missing/imputed.
            padding_mask: Optional padding mask of shape (B, T) where True
                         indicates valid timesteps and False indicates padding.

        Returns:
            Tensor of shape (B, T, H) or (B, H) depending on pooling,
            where H is the hidden dimension.
        """
        pass

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder.

        Returns:
            Output dimension (typically d_model).
        """
        return self.config.d_model

    def handles_missingness_intrinsically(self) -> bool:
        """Return whether the encoder consumes observation masks natively.

        Downstream checkpoint loading uses this capability to decide whether an
        encoder still needs EncoderWithMissingToken wrapping.
        """
        return False


class SSLTokenizingEncoder(Protocol):
    """Protocol for encoders that expose the SSL timestep-token interface.

    Controlled SSL objectives operate on timestep tokens directly. Keeping this
    as a protocol makes the required encoder capability explicit without forcing
    every baseline encoder to inherit a second abstract base class.
    """

    config: EncoderConfig

    def tokenize(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Convert dense values and observation masks into timestep tokens."""

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode prebuilt timestep tokens."""

    def get_output_dim(self) -> int:
        """Return the encoder output dimension."""
