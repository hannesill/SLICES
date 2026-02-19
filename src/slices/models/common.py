"""Shared model utilities: pooling, positional encoding, activation lookup.

Consolidates duplicate implementations from transformer.py, linear.py,
smart.py, and heads/mlp.py into single shared functions and modules.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation function name (relu, gelu, silu, tanh).

    Returns:
        Activation module.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(activations.keys())}")
    return activations[name]()


def apply_pooling(
    x: torch.Tensor,
    pooling: str,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply pooling strategy to get sequence-level representation.

    Supports mean, max, last, cls, and none pooling strategies.

    Args:
        x: Encoded tensor of shape (B, T, d_model).
        pooling: Pooling strategy name.
        padding_mask: Optional padding mask of shape (B, T) where True
                     indicates valid timesteps (our convention).

    Returns:
        Pooled tensor of shape (B, d_model) or (B, T, d_model) if pooling='none'.

    Raises:
        ValueError: If pooling strategy is not recognized.
    """
    if pooling == "none":
        return x

    elif pooling == "cls":
        return x[:, 0, :]

    elif pooling == "last":
        if padding_mask is not None:
            lengths = padding_mask.sum(dim=1)
            batch_idx = torch.arange(x.size(0), device=x.device)
            last_idx = (lengths - 1).clamp(min=0)
            return x[batch_idx, last_idx, :]
        else:
            return x[:, -1, :]

    elif pooling == "mean":
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1)
            x_masked = x * mask_expanded
            sum_valid = x_masked.sum(dim=1)
            count_valid = padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return sum_valid / count_valid
        else:
            return x.mean(dim=1)

    elif pooling == "max":
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1)
            x_masked = x.masked_fill(~mask_expanded, float("-inf"))
            return x_masked.max(dim=1)[0]
        else:
            return x.max(dim=1)[0]

    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Uses fixed sinusoidal encoding as in "Attention is All You Need".
    Supports both 3D (B, T, D) and 4D (B, V, T, D) inputs.
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Model dimension.
            max_seq_length: Maximum sequence length to support.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (B, T, D), (B, V, T, D), or (B*V, T, D).

        Returns:
            Tensor with positional encoding added.
        """
        if x.dim() == 4:
            # (B, V, T, d_model) - broadcast over B and V
            T = x.size(2)
            x = x + self.pe[:T, :].unsqueeze(0).unsqueeze(0)
        else:
            # (B, T, d_model) or (B*V, T, d_model) - broadcast over batch
            T = x.size(1)
            x = x + self.pe[:T, :]
        return self.dropout(x)
