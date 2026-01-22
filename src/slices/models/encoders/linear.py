"""Linear encoder for ICU time-series data.

A simple linear baseline encoder that projects input features directly to
embeddings with optional pooling. No attention, no hidden layers, no
nonlinearities - the simplest possible encoder architecture.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseEncoder, EncoderConfig


@dataclass
class LinearConfig(EncoderConfig):
    """Configuration for linear encoder.

    Extends base encoder config with linear-specific parameters.
    """

    pooling: str = "mean"  # Pooling strategy: mean, max, last, none


class LinearEncoder(BaseEncoder):
    """Linear encoder for ICU time-series.

    The simplest possible encoder architecture:
    - Linear projection from input features to output dimension
    - Observation mask integration (same as transformer)
    - Pooling for sequence-level representation

    No attention, no hidden layers, no nonlinearities.

    Example:
        >>> config = LinearConfig(d_input=35, d_model=128)
        >>> encoder = LinearEncoder(config)
        >>> x = torch.randn(32, 48, 35)  # (batch, seq_len, features)
        >>> mask = torch.rand(32, 48, 35) > 0.3  # 30% missing
        >>> out = encoder(x, mask=mask)  # (32, 128) if pooled
    """

    def __init__(self, config: LinearConfig) -> None:
        """Initialize linear encoder.

        Args:
            config: Linear encoder configuration.
        """
        super().__init__(config)
        self.config: LinearConfig = config

        # Validate pooling
        valid_pooling = {"mean", "max", "last", "none"}
        if config.pooling not in valid_pooling:
            raise ValueError(f"Invalid pooling '{config.pooling}'. Choose from: {valid_pooling}")

        # Compute actual input dimension based on observation mask mode
        if config.use_observation_mask and config.mask_input_mode == "concat":
            actual_d_input = config.d_input * 2
        else:
            actual_d_input = config.d_input

        # Single linear projection: (B, T, actual_d_input) -> (B, T, d_model)
        self.linear = nn.Linear(actual_d_input, config.d_model)

    def _apply_mask_input(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Prepare input with observation mask when use_observation_mask is enabled.

        Args:
            x: Input tensor (B, T, D) - may contain forward-filled values.
            mask: Observation mask (B, T, D) - True = observed, False = imputed.

        Returns:
            Modified input ready for projection.
        """
        if not self.config.use_observation_mask:
            return x

        B, T, D = x.shape

        # Default to all observed if mask not provided
        if mask is None:
            mask = torch.ones(B, T, D, dtype=torch.bool, device=x.device)

        # Optionally zero out imputed values
        if self.config.zero_imputed_values:
            x = x * mask.float()

        if self.config.mask_input_mode == "concat":
            return torch.cat([x, mask.float()], dim=-1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input time-series with a linear projection.

        Args:
            x: Input tensor of shape (B, T, D).
            mask: Optional observation mask of shape (B, T, D).
            padding_mask: Optional padding mask of shape (B, T) where True
                         indicates valid timesteps.

        Returns:
            Encoded tensor:
            - If pooling='none': shape (B, T, d_model)
            - Otherwise: shape (B, d_model)
        """
        # Apply observation mask as explicit input if enabled
        x = self._apply_mask_input(x, mask)

        # Linear projection
        x = self.linear(x)  # (B, T, d_model)

        # Apply pooling
        return self._apply_pooling(x, padding_mask)

    def _apply_pooling(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply pooling strategy to get sequence-level representation.

        Args:
            x: Encoded tensor of shape (B, T, d_model).
            padding_mask: Padding mask of shape (B, T) where True = valid.

        Returns:
            Pooled tensor of shape (B, d_model) or (B, T, d_model) if no pooling.
        """
        if self.config.pooling == "none":
            return x

        elif self.config.pooling == "last":
            if padding_mask is not None:
                lengths = padding_mask.sum(dim=1)
                batch_idx = torch.arange(x.size(0), device=x.device)
                last_idx = (lengths - 1).clamp(min=0)
                return x[batch_idx, last_idx, :]
            else:
                return x[:, -1, :]

        elif self.config.pooling == "mean":
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1)
                x_masked = x * mask_expanded
                sum_valid = x_masked.sum(dim=1)
                count_valid = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
                return sum_valid / count_valid
            else:
                return x.mean(dim=1)

        elif self.config.pooling == "max":
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(-1)
                x_masked = x.masked_fill(~mask_expanded, float("-inf"))
                return x_masked.max(dim=1)[0]
            else:
                return x.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder."""
        return self.config.d_model
