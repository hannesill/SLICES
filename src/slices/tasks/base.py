"""Abstract base class for downstream task heads."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TaskConfig:
    """Configuration for downstream task head."""

    name: str  # Task name (e.g., 'mortality', 'los', 'aki')
    n_classes: int = 2  # Number of output classes
    dropout: float = 0.1


class BaseTaskHead(ABC, nn.Module):
    """Abstract base class for downstream task heads."""

    def __init__(self, encoder: nn.Module, config: TaskConfig) -> None:
        """Initialize task head with encoder and configuration.
        
        Args:
            encoder: Encoder module to use for feature extraction.
            config: Task configuration.
        """
        super().__init__()
        self.encoder = encoder
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) input
        mask: Optional[torch.Tensor] = None,  # (B, T, D) observation mask
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) sequence padding
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through task head.
        
        Args:
            x: Input tensor of shape (B, T, D).
            mask: Optional observation mask of shape (B, T, D).
            padding_mask: Optional padding mask of shape (B, T).
        
        Returns:
            Dictionary containing task outputs (e.g., 'logits', 'probs').
        """
        pass

