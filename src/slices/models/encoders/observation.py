"""Observation-level transformer encoder for ICU time-series data.

Tokenizes each observed measurement as a separate token (timestep, feature, value),
rather than collapsing all features at a timestep into one vector. This avoids the
"mostly zeros" problem with sparse clinical data where 80%+ of values are missing.

Each token is: value_proj(value) + feature_embed(feature_id) + time_pe(timestep).
Only observed values become tokens -- missingness is handled intrinsically.

Shares TransformerEncoderLayer from transformer.py (same block architecture) for
fair comparison with timestep-level models.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from slices.models.common import apply_pooling

from .base import BaseEncoder, EncoderConfig
from .transformer import TransformerEncoderLayer


@dataclass
class ObservationTransformerConfig(EncoderConfig):
    """Configuration for observation-level transformer encoder."""

    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    pooling: str = "mean"  # mean, max, none (use 'none' for SSL pretraining)
    prenorm: bool = True
    layer_norm_eps: float = 1e-5


class ObservationTransformerEncoder(BaseEncoder):
    """Observation-level transformer encoder for ICU time-series.

    Instead of projecting all D features at each timestep into one token,
    this encoder creates one token per observed measurement. Each token
    encodes the scalar value, the feature identity, and the timestep.

    This maps naturally to MAE visible-only encoding: the encoder only
    processes visible observation tokens, never seeing masked ones.

    For finetuning, forward() tokenizes all observations and pools to
    get a patient-level representation.

    Example:
        >>> config = ObservationTransformerConfig(
        ...     d_input=35, d_model=128, n_layers=4, n_heads=8
        ... )
        >>> encoder = ObservationTransformerEncoder(config)
        >>> x = torch.randn(4, 48, 35)
        >>> obs_mask = torch.rand(4, 48, 35) > 0.3
        >>> out = encoder(x, mask=obs_mask)  # (4, 128) with mean pooling
    """

    def __init__(self, config: ObservationTransformerConfig) -> None:
        super().__init__(config)
        self.config: ObservationTransformerConfig = config

        self._validate_config()

        # Tokenization layers
        self.value_proj = nn.Linear(1, config.d_model)
        self.feature_embed = nn.Embedding(config.d_input, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Sinusoidal time positional encoding (buffer, not learned)
        pe = self._build_sinusoidal_pe(config.max_seq_length, config.d_model)
        self.register_buffer("time_pe", pe)  # (max_seq_length, d_model)

        # Transformer layers (reuses TransformerEncoderLayer from transformer.py)
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
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError(
                f"d_model ({self.config.d_model}) must be divisible by "
                f"n_heads ({self.config.n_heads})"
            )

        valid_pooling = {"mean", "max", "last", "none"}
        if self.config.pooling not in valid_pooling:
            raise ValueError(
                f"Invalid pooling '{self.config.pooling}'. Choose from: {valid_pooling}"
            )

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def tokenize(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Convert observed measurements into observation-level tokens.

        Args:
            x: Input values of shape (B, T, D).
            obs_mask: Boolean mask (B, T, D), True = observed.

        Returns:
            tokens: (B, max_obs, d_model) embedded tokens, zero-padded.
            padding_mask: (B, max_obs) True = valid token, False = padding.
            token_info: dict with:
                - timestep_idx: (B, max_obs) long, timestep index per token
                - feature_idx: (B, max_obs) long, feature index per token
                - values: (B, max_obs) float, original scalar values
                - n_obs: (B,) long, number of observed tokens per sample
        """
        B, T, D = x.shape
        device = x.device

        # Count observations per sample
        n_obs = obs_mask.sum(dim=(1, 2))  # (B,)
        max_obs = int(n_obs.max().item())

        # Ensure at least 1 token per sample (for degenerate edge case)
        max_obs = max(max_obs, 1)

        # Flatten spatial dims: (B, T*D)
        flat_mask = obs_mask.reshape(B, T * D)  # (B, T*D)
        flat_values = x.reshape(B, T * D)  # (B, T*D)

        # Build timestep and feature index grids: (T*D,)
        t_grid = torch.arange(T, device=device).unsqueeze(1).expand(T, D).reshape(T * D)
        f_grid = torch.arange(D, device=device).unsqueeze(0).expand(T, D).reshape(T * D)

        # Sort so observed values come first (argsort trick)
        # Sort descending by mask so True (1) comes before False (0)
        sort_idx = flat_mask.float().argsort(dim=1, descending=True, stable=True)  # (B, T*D)

        # Gather sorted values, timesteps, features
        sorted_values = flat_values.gather(1, sort_idx)  # (B, T*D)
        sorted_t = t_grid.unsqueeze(0).expand(B, -1).gather(1, sort_idx)  # (B, T*D)
        sorted_f = f_grid.unsqueeze(0).expand(B, -1).gather(1, sort_idx)  # (B, T*D)

        # Truncate to max_obs
        values = sorted_values[:, :max_obs]  # (B, max_obs)
        timestep_idx = sorted_t[:, :max_obs]  # (B, max_obs)
        feature_idx = sorted_f[:, :max_obs]  # (B, max_obs)

        # Build padding mask
        token_positions = torch.arange(max_obs, device=device).unsqueeze(0)  # (1, max_obs)
        padding_mask = token_positions < n_obs.unsqueeze(1)  # (B, max_obs)

        # Build tokens: value_proj + feature_embed + time_pe
        val_tokens = self.value_proj(values.unsqueeze(-1))  # (B, max_obs, d_model)
        feat_tokens = self.feature_embed(feature_idx)  # (B, max_obs, d_model)
        time_tokens = self.time_pe[timestep_idx]  # (B, max_obs, d_model)

        tokens = val_tokens + feat_tokens + time_tokens  # (B, max_obs, d_model)
        tokens = self.embed_dropout(tokens)

        # Zero out padding positions
        tokens = tokens * padding_mask.unsqueeze(-1)

        token_info = {
            "timestep_idx": timestep_idx,
            "feature_idx": feature_idx,
            "values": values,
            "n_obs": n_obs,
        }

        return tokens, padding_mask, token_info

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers on tokens.

        Args:
            tokens: (B, N, d_model) token embeddings.
            padding_mask: (B, N) True = valid, False = padding.

        Returns:
            (B, N, d_model) encoded tokens.
        """
        # Convert to PyTorch convention: True = ignore
        key_padding_mask = ~padding_mask

        x = tokens
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tokenize, encode, and pool.

        Args:
            x: Input (B, T, D).
            mask: Observation mask (B, T, D), True = observed. If None,
                  all values are treated as observed.
            padding_mask: Sequence-level padding (B, T). Currently unused
                         since observation-level padding is derived from obs_mask.

        Returns:
            If pooling='none': (B, max_obs, d_model)
            Otherwise: (B, d_model)
        """
        B, T, D = x.shape

        if mask is None:
            mask = torch.ones(B, T, D, dtype=torch.bool, device=x.device)

        tokens, tok_padding, token_info = self.tokenize(x, mask)
        encoded = self.encode(tokens, tok_padding)

        return apply_pooling(encoded, self.config.pooling, tok_padding)

    def get_output_dim(self) -> int:
        return self.config.d_model
