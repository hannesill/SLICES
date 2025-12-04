"""Abstract base class for self-supervised learning objectives."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


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

