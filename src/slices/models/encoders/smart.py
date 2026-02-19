"""SMART (MART) encoder for ICU time-series data.

Implements the Missing-Aware Transformer (MART) architecture from SMART:
"Self-supervised Missing-Aware Reconstruction Transformer" (NeurIPS 2024).

Key features:
- MLPEmbedder: Jointly embeds (value, mask) pairs
- Per-variable query tokens for representation learning
- SeqAttention: Temporal attention within each variable respecting observation masks
- VarAttention: Cross-variable attention using query-based pooling
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from slices.models.common import PositionalEncoding

from .base import BaseEncoder, EncoderConfig


@dataclass
class SMARTEncoderConfig(EncoderConfig):
    """Configuration for SMART (MART) encoder.

    Extends base encoder config with SMART-specific parameters.
    Note: d_input represents the number of variables (V dimension).
    """

    n_heads: int = 4  # Number of attention heads
    d_ff: int = 256  # Feedforward dimension (typically 8 * d_model)
    dropout: float = 0.1
    pooling: str = "query"  # "query", "mean", or "none" (for SSL)


class MLPEmbedder(nn.Module):
    """Embeds (value, observation_mask) pairs jointly.

    Each position combines [value_i, mask_i] into a 2D feature that gets embedded
    to d_model dimensions. This is a key component of MART that allows the model
    to be aware of missingness at the input level.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize MLPEmbedder.

        Args:
            d_model: Output embedding dimension.

        Note:
            Original SMART uses NO activation between the two linear layers.
            This is intentional - the embedding learns a linear projection of
            the (value, mask) pairs.
        """
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed (value, mask) pairs.

        Args:
            x: Values tensor of shape (B, V, T).
            mask: Observation mask tensor of shape (B, V, T) where True = observed.

        Returns:
            Embedded tensor of shape (B, V, T, d_model).
        """
        # Stack value and mask as 2D features
        x = torch.stack((x, mask.float()), dim=-1)  # (B, V, T, 2)
        x = self.embed(x)  # (B, V, T, d_model)
        return x


class SeqAttentionBlock(nn.Module):
    """Temporal attention block within each variable.

    Computes self-attention across the time dimension for each variable,
    using additive observation-based attention biases (SMART's key innovation).

    The attention mask uses an additive scheme where mask[i] + mask[j] creates
    graded attention: 0 (neither observed) → blocked, 1 (one observed) → partial,
    2 (both observed) → full attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """Initialize SeqAttention block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        # Manual QKV projection like original (no bias)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal attention within each variable.

        Args:
            x: Input tensor of shape (B, V, T+1, d_model) where T+1 includes query token.
            obs_mask: Observation mask of shape (B, V, T) for original timesteps.
            padding_mask: Optional padding mask of shape (B, T) where True = valid.

        Returns:
            Output tensor of shape (B, V, T+1, d_model).
        """
        B, V, T_plus_1, d_model = x.shape

        # Reshape for batch processing: (B*V, T+1, d_model)
        x_flat = x.reshape(B * V, T_plus_1, d_model)

        # Pre-norm
        x_norm = self.norm(x_flat)

        # QKV projection
        qkv = self.qkv(x_norm).reshape(B * V, T_plus_1, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*V, n_heads, T+1, d_head)
        q, k, v = qkv.unbind(0)  # Each: (B*V, n_heads, T+1, d_head)

        # Create additive attention mask (SMART's key innovation)
        # Extend obs_mask to include query token (always "observed" = 1)
        query_observed = torch.ones(B, V, 1, dtype=obs_mask.dtype, device=x.device)
        obs_mask_extended = torch.cat([query_observed, obs_mask], dim=2)  # (B, V, T+1)
        obs_mask_flat = obs_mask_extended.reshape(B * V, T_plus_1).float()  # (B*V, T+1)

        # Additive mask: mask[i] + mask[j] gives 0, 1, or 2
        # This creates graded attention bias (higher = more attention)
        attn_bias = obs_mask_flat.unsqueeze(-1) + obs_mask_flat.unsqueeze(-2)  # (B*V, T+1, T+1)

        # Handle padding mask if provided (add -inf for padded positions)
        if padding_mask is not None:
            # Extend padding mask to include query token (always valid)
            query_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            padding_mask_extended = torch.cat([query_valid, padding_mask], dim=1)  # (B, T+1)
            # Expand to (B*V, T+1)
            padding_mask_flat = padding_mask_extended.unsqueeze(1).expand(B, V, T_plus_1)
            padding_mask_flat = padding_mask_flat.reshape(B * V, T_plus_1)
            # Create padding attention mask (outer product)
            padding_attn = padding_mask_flat.unsqueeze(-2) & padding_mask_flat.unsqueeze(-1)
            # Add -inf where padding should be ignored
            attn_bias = attn_bias + (~padding_attn).float() * float("-1e9")

        # Expand attn_bias for heads: (B*V, T+1, T+1) -> (B*V, n_heads, T+1, T+1)
        attn_bias = attn_bias.unsqueeze(1)  # (B*V, 1, T+1, T+1)

        # Scaled dot-product attention with additive bias
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B * V, T_plus_1, d_model)
        attn_out = self.proj(attn_out)

        # Residual connection
        x_flat = x_flat + self.dropout(attn_out)

        # Reshape back to (B, V, T+1, d_model)
        x = x_flat.reshape(B, V, T_plus_1, d_model)
        return x


class VarAttentionBlock(nn.Module):
    """Cross-variable attention block (SMART's key innovation).

    Implements the original SMART variable attention:
    - Queries come from position 0 (query tokens) only
    - Keys are masked-averaged across time dimension
    - Values preserve full temporal structure packed into feature dimension
    - Output has temporal structure unpacked back

    This allows cross-variable attention while preserving temporal information.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """Initialize VarAttention block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model)
        # Manual QKV projection like original (no bias)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """Apply cross-variable attention using query tokens.

        Original SMART implementation:
        - q from position 0 only: (B, n_heads, V, d_head)
        - k averaged across time: (B, n_heads, V, d_head)
        - v packed with temporal info: (B, n_heads, V, T*d_head)
        - Output unpacked back to (B, V, T+1, d_model)

        Args:
            x: Input tensor of shape (B, V, T+1, d_model).
            obs_mask: Observation mask of shape (B, V, T) for original timesteps.

        Returns:
            Output tensor of shape (B, V, T+1, d_model).
        """
        B, V, T_plus_1, d_model = x.shape

        # Pre-norm
        x_norm = self.norm(x)

        # QKV projection: (B, V, T+1, d_model) -> (B, V, T+1, 3*d_model)
        qkv = self.qkv(x_norm).reshape(B, V, T_plus_1, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(3, 0, 2, 4, 1, 5)  # (3, B, T+1, n_heads, V, d_head)
        q, k, v = qkv.unbind(0)  # Each: (B, T+1, n_heads, V, d_head)

        # Query: only use position 0 (query tokens)
        q = q[:, 0]  # (B, n_heads, V, d_head)

        # Key: masked average across time dimension
        # Create mask for averaging: extend obs_mask to include query token
        query_mask = torch.ones(B, V, 1, device=x.device, dtype=obs_mask.dtype)
        mask_extended = torch.cat([query_mask, obs_mask], dim=2)  # (B, V, T+1)
        # Reshape for broadcasting with k: (B, T+1, n_heads, V, d_head)
        mask_for_k = mask_extended.permute(0, 2, 1)  # (B, T+1, V)
        mask_for_k = mask_for_k.unsqueeze(2).unsqueeze(-1)  # (B, T+1, 1, V, 1)
        mask_for_k = mask_for_k.expand(
            -1, -1, self.n_heads, -1, self.d_head
        )  # (B, T+1, n_heads, V, d_head)

        # Masked sum and count for averaging
        k_masked = k.masked_fill(~mask_for_k.bool(), 0)
        k_sum = k_masked.sum(dim=1)  # (B, n_heads, V, d_head)
        mask_count = mask_for_k.float().sum(dim=1).clamp(min=1)  # (B, n_heads, V, d_head)
        k = k_sum / mask_count  # (B, n_heads, V, d_head)

        # Value: pack temporal dimension into feature dimension
        # v: (B, T+1, n_heads, V, d_head) -> (B, n_heads, V, (T+1)*d_head)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.n_heads, V, -1)

        # Scaled dot-product attention (manual implementation for MPS compatibility)
        # q: (B, n_heads, V, d_head)
        # k: (B, n_heads, V, d_head)
        # v: (B, n_heads, V, (T+1)*d_head)
        # Note: F.scaled_dot_product_attention has a bug on MPS when value has
        # different last dimension than key, so we use manual attention here.
        scale = self.d_head**-0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, V, V)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # (B, n_heads, V, (T+1)*d_head)

        # Unpack temporal dimension from output
        # (B, n_heads, V, (T+1)*d_head) -> (B, V, T+1, d_model)
        attn_out = attn_out.view(B, self.n_heads, V, self.d_head, T_plus_1)
        attn_out = attn_out.permute(0, 2, 4, 1, 3).reshape(B, V, T_plus_1, d_model)

        # Project and apply dropout
        attn_out = self.proj(attn_out)

        # Residual connection
        x = x + self.dropout(attn_out)

        return x


class MLPBlock(nn.Module):
    """Feedforward MLP block with pre-normalization."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """Initialize MLP block.

        Args:
            d_model: Model dimension.
            d_ff: Feedforward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feedforward transformation.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Output tensor of same shape.
        """
        x_norm = self.norm(x)
        ff_out = self.ff(x_norm)
        return x + self.dropout(ff_out)


class BasicBlock(nn.Module):
    """Combined attention + MLP block for MART.

    Applies:
    1. SeqAttention: Temporal attention within each variable
    2. VarAttention: Cross-variable attention using query tokens
    3. MLP: Feedforward transformation
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize BasicBlock.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.seq_att = SeqAttentionBlock(d_model, n_heads, dropout)
        self.var_att = VarAttentionBlock(d_model, n_heads, dropout)
        self.mlp = MLPBlock(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through BasicBlock.

        Args:
            x: Input tensor of shape (B, V, T+1, d_model).
            obs_mask: Observation mask of shape (B, V, T).
            padding_mask: Optional padding mask of shape (B, T).

        Returns:
            Output tensor of shape (B, V, T+1, d_model).
        """
        x = self.seq_att(x, obs_mask, padding_mask)  # Temporal attention
        x = self.var_att(x, obs_mask)  # Variable attention
        x = self.mlp(x)  # Feedforward
        return x


class SMARTEncoder(BaseEncoder):
    """SMART (MART) encoder for ICU time-series.

    Implements the Missing-Aware Transformer architecture with:
    - Joint (value, mask) embedding via MLPEmbedder
    - Per-variable query tokens for representation learning
    - Temporal and cross-variable attention
    - Configurable pooling for downstream tasks

    Input Format:
        SLICES uses (B, T, D) format where D is features.
        SMART uses (B, V, T) format where V is variables.
        This encoder handles the transpose internally.

    Example:
        >>> config = SMARTEncoderConfig(d_input=35, d_model=32, n_layers=2)
        >>> encoder = SMARTEncoder(config)
        >>> x = torch.randn(32, 48, 35)  # (batch, seq_len, features)
        >>> mask = torch.rand(32, 48, 35) > 0.3  # observation mask
        >>> out = encoder(x, mask)
    """

    def __init__(self, config: SMARTEncoderConfig) -> None:
        """Initialize SMART encoder.

        Args:
            config: SMART encoder configuration.
        """
        super().__init__(config)
        self.config: SMARTEncoderConfig = config

        # Validate configuration
        self._validate_config()

        # Joint (value, mask) embedding
        self.embedder = MLPEmbedder(config.d_model)

        # One query token per variable
        # Shape: (V, 1, d_model) - will be expanded for batch
        self.query = nn.Parameter(torch.zeros(config.d_input, 1, config.d_model))
        nn.init.normal_(self.query, std=0.02)

        # Positional encoding for temporal dimension
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_seq_length=config.max_seq_length + 1,  # +1 for query token
            dropout=config.dropout,
        )

        # MART blocks
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.d_model % self.config.n_heads != 0:
            raise ValueError(
                f"d_model ({self.config.d_model}) must be divisible by "
                f"n_heads ({self.config.n_heads})"
            )

        valid_pooling = {"query", "mean", "none"}
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
            x: Input tensor of shape (B, T, D) in SLICES format.
            mask: Observation mask of shape (B, T, D) where True = observed.
            padding_mask: Optional padding mask of shape (B, T) where True = valid.

        Returns:
            Encoded tensor:
            - If pooling='none': shape (B, V, T, d_model) for SSL objectives
            - If pooling='query': shape (B, V*d_model) flattened query embeddings
            - If pooling='mean': shape (B, d_model) mean pooled
        """
        B, T, D = x.shape

        # Create default mask if not provided
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)

        # Transpose from SLICES format (B, T, D) to SMART format (B, V, T)
        # where V = D (variables = features)
        x = x.transpose(1, 2)  # (B, D, T) = (B, V, T)
        mask = mask.transpose(1, 2)  # (B, D, T) = (B, V, T)

        # Embed (value, mask) pairs jointly
        x = self.embedder(x, mask)  # (B, V, T, d_model)

        # Add query tokens at position 0 for each variable
        # self.query: (V, 1, d_model) -> expand to (B, V, 1, d_model)
        query = self.query.unsqueeze(0).expand(B, -1, -1, -1)  # (B, V, 1, d_model)
        x = torch.cat([query, x], dim=2)  # (B, V, T+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply MART blocks
        for block in self.blocks:
            x = block(x, mask, padding_mask)

        # Final layer norm
        x = self.final_norm(x)

        # Apply pooling
        return self._apply_pooling(x, mask)

    def _apply_pooling(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling strategy to get final representation.

        Args:
            x: Encoded tensor of shape (B, V, T+1, d_model).
            obs_mask: Observation mask of shape (B, V, T).

        Returns:
            Pooled tensor based on pooling strategy.
        """
        B, V, _, d_model = x.shape

        if self.config.pooling == "none":
            # Return full output without query tokens for SSL
            return x[:, :, 1:, :]  # (B, V, T, d_model)

        elif self.config.pooling == "query":
            # Extract query tokens and flatten
            query_out = x[:, :, 0, :]  # (B, V, d_model)
            return query_out.reshape(B, -1)  # (B, V*d_model)

        elif self.config.pooling == "mean":
            # Mean pool over time and variables
            temporal_values = x[:, :, 1:, :]  # (B, V, T, d_model)

            # Masked mean over time
            mask_expanded = obs_mask.unsqueeze(-1).float()  # (B, V, T, 1)
            masked_sum = (temporal_values * mask_expanded).sum(dim=2)  # (B, V, d_model)
            mask_count = mask_expanded.sum(dim=2).clamp(min=1)  # (B, V, 1)
            var_means = masked_sum / mask_count  # (B, V, d_model)

            # Mean over variables
            return var_means.mean(dim=1)  # (B, d_model)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

    def get_output_dim(self) -> int:
        """Return the output dimension of the encoder.

        Returns:
            Output dimension based on pooling strategy.
        """
        if self.config.pooling == "query":
            return self.config.d_input * self.config.d_model
        else:
            return self.config.d_model
