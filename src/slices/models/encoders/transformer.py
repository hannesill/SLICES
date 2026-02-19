"""Transformer encoder for ICU time-series data.

A modular, configurable transformer architecture designed for self-supervised
learning on ICU time-series with missing values and variable-length sequences.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from slices.models.common import PositionalEncoding, apply_pooling, get_activation

from .base import BaseEncoder, EncoderConfig


@dataclass
class TransformerConfig(EncoderConfig):
    """Configuration for transformer encoder.

    Extends base encoder config with transformer-specific parameters.
    """

    n_heads: int = 8  # Number of attention heads
    d_ff: int = 512  # Feedforward dimension (typically 4 * d_model)
    dropout: float = 0.1
    activation: str = "gelu"  # Activation function: gelu, relu, silu
    layer_norm_eps: float = 1e-5
    use_positional_encoding: bool = True
    pooling: str = "mean"  # Pooling strategy: mean, max, cls, last, none
    prenorm: bool = True  # Pre-LN vs Post-LN transformer


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with Pre-LN or Post-LN.

    Supports both pre-normalization (more stable, modern default) and
    post-normalization (original transformer) architectures.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        prenorm: bool = True,
    ) -> None:
        """Initialize transformer layer.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward dimension.
            dropout: Dropout probability.
            activation: Activation function name.
            layer_norm_eps: Layer norm epsilon.
            prenorm: If True, use Pre-LN; if False, use Post-LN.
        """
        super().__init__()
        self.prenorm = prenorm

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # Use (B, T, D) format
        )

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer layer.

        Args:
            x: Input tensor of shape (B, T, D).
            attn_mask: Attention mask of shape (T, T) or (B*n_heads, T, T).
            key_padding_mask: Key padding mask of shape (B, T) where True
                            indicates positions to ignore (padding).

        Returns:
            Output tensor of shape (B, T, D).
        """
        if self.prenorm:
            # Pre-LN: Normalize before attention and FF
            # Self-attention block
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(
                x_norm,
                x_norm,
                x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout1(attn_out)

            # Feedforward block
            x_norm = self.norm2(x)
            ff_out = self.ff(x_norm)
            x = x + self.dropout2(ff_out)
        else:
            # Post-LN: Normalize after attention and FF
            # Self-attention block
            attn_out, _ = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + self.dropout1(attn_out))

            # Feedforward block
            ff_out = self.ff(x)
            x = self.norm2(x + self.dropout2(ff_out))

        return x


class TransformerEncoder(BaseEncoder):
    """Transformer encoder for ICU time-series.

    A modular transformer architecture with:
    - Input projection from raw features to model dimension
    - Optional positional encoding
    - Stacked transformer layers with self-attention
    - Padding mask support (sequence-level masking)
    - Configurable pooling for sequence-level representations

    Note:
        For MAE pretraining, the input should have learned tokens (MISSING_TOKEN
        and MASK_TOKEN) already substituted by the SSL objective. The encoder
        processes these tokens like any other input values.

    Example:
        >>> config = TransformerConfig(
        ...     d_input=35, d_model=128, n_layers=4, n_heads=8
        ... )
        >>> encoder = TransformerEncoder(config)
        >>> x = torch.randn(32, 48, 35)  # (batch, seq_len, features)
        >>> out = encoder(x)  # (32, 48, 128) or (32, 128) if pooled
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize transformer encoder.

        Args:
            config: Transformer configuration.
        """
        super().__init__(config)
        self.config: TransformerConfig = config

        # Validate configuration before creating modules
        self._validate_config()

        # Input projection: (B, T, d_input) -> (B, T, d_model)
        self.input_proj = nn.Linear(config.d_input, config.d_model)

        # Optional CLS token for pooling
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=config.d_model,
                max_seq_length=config.max_seq_length + 1,  # +1 for CLS token
                dropout=config.dropout,
            )
        else:
            self.pos_encoder = nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    layer_norm_eps=config.layer_norm_eps,
                    prenorm=config.prenorm,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm (for Pre-LN architecture)
        if config.prenorm:
            self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        else:
            self.final_norm = nn.Identity()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError(
                f"d_model ({self.config.d_model}) must be divisible by "
                f"n_heads ({self.config.n_heads})"
            )

        valid_pooling = {"mean", "max", "cls", "last", "none"}
        if self.config.pooling not in valid_pooling:
            raise ValueError(
                f"Invalid pooling '{self.config.pooling}'. " f"Choose from: {valid_pooling}"
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input time-series.

        Args:
            x: Input tensor of shape (B, T, D) where B is batch size,
               T is sequence length, and D is feature dimension (d_input).
               For MAE pretraining, this should have MISSING_TOKEN and MASK_TOKEN
               already substituted by the SSL objective.
            mask: Optional observation mask of shape (B, T, D). Reserved for
                  future use (e.g., attention masking). Currently ignored.
            padding_mask: Optional padding mask of shape (B, T) where True
                         indicates valid timesteps and False indicates padding.
                         Note: PyTorch convention is inverted in MultiheadAttention
                         (True = ignore), so we invert it internally.

        Returns:
            Encoded tensor:
            - If pooling='none': shape (B, T, d_model)
            - Otherwise: shape (B, d_model)
        """
        B, T, D = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, T, d_model)

        # Add CLS token if using CLS pooling
        if self.config.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

            # Extend padding mask to include CLS token (always valid)
            if padding_mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, T, d_model) or (B, T+1, d_model)

        # Convert padding mask to PyTorch convention (True = ignore)
        # Our convention: True = valid, False = padding
        # PyTorch convention: True = padding (ignore), False = valid
        if padding_mask is not None:
            key_padding_mask = ~padding_mask  # Invert
        else:
            key_padding_mask = None

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # Final layer norm (for Pre-LN)
        x = self.final_norm(x)  # (B, T, d_model)

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
            padding_mask: Padding mask of shape (B, T) where True indicates
                         valid timesteps (our convention, not PyTorch's).

        Returns:
            Pooled tensor of shape (B, d_model) or (B, T, d_model) if no pooling.
        """
        return apply_pooling(x, self.config.pooling, padding_mask)

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder.

        Returns:
            Output dimension (d_model).
        """
        return self.config.d_model
