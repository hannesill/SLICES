"""Masked Autoencoder (MAE) for self-supervised learning on ICU time-series.

Timestep-level tokenization variant: each token = one timestep with all D features.
The encoder only sees visible (unmasked) timestep tokens, matching the original MAE design.

Architecture:
1. TransformerEncoder.tokenize() -> one token per timestep (B, T, d_model)
2. Random mask: 50% of timestep tokens are masked (configurable via mask_ratio)
3. Encoder processes only visible tokens
4. Decoder reassembles full sequence (visible + mask tokens), predicts D features per timestep
5. MSE loss on observed features at masked timesteps only
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from slices.models.common import build_sinusoidal_pe

from .base import BaseSSLObjective, SSLConfig
from .masking import create_timestep_mask, extract_visible_timesteps


@dataclass
class MAEConfig(SSLConfig):
    """Configuration for timestep-level MAE objective.

    The encoder only sees visible (unmasked) timestep tokens.
    Mask ratio is applied to timesteps (not individual observations).
    """

    name: str = "mae"

    # Masking
    mask_ratio: float = 0.5  # Fraction of timesteps to mask

    # Decoder parameters
    decoder_d_model: int = 128
    decoder_n_layers: int = 2
    decoder_n_heads: int = 4
    decoder_d_ff: int = 512
    decoder_dropout: float = 0.1


class MAEDecoder(nn.Module):
    """Lightweight decoder for timestep-level MAE.

    Reassembles encoded visible tokens with learnable mask tokens at masked
    timestep positions. Adds temporal positional information to all tokens
    before running through lightweight transformer layers. Predicts D features
    per timestep.
    """

    def __init__(
        self,
        d_encoder: int,
        n_features: int,
        max_seq_length: int,
        config: MAEConfig,
    ) -> None:
        super().__init__()
        self.config = config
        d_dec = config.decoder_d_model

        # Project encoder output to decoder dimension
        self.encoder_proj = nn.Linear(d_encoder, d_dec)

        # Learnable mask token (in decoder space)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        nn.init.normal_(self.mask_token, std=0.02)

        # Sinusoidal time PE for decoder
        self.register_buffer("time_pe", build_sinusoidal_pe(max_seq_length, d_dec))

        # Decoder transformer layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_dec,
            nhead=config.decoder_n_heads,
            dim_feedforward=config.decoder_d_ff,
            dropout=config.decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.decoder_n_layers,
        )

        # Dropout on input embeddings before decoder transformer
        self.embed_dropout = nn.Dropout(config.decoder_dropout)

        # Output: predict D features per timestep
        self.output_proj = nn.Linear(d_dec, n_features)

    def forward(
        self,
        encoded_visible: torch.Tensor,
        ssl_mask: torch.Tensor,
        token_info: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> torch.Tensor:
        """Decode: reassemble visible + mask tokens, predict feature values.

        Args:
            encoded_visible: (B, n_vis, d_encoder) encoded visible tokens.
            ssl_mask: (B, T) bool, True = visible, False = masked.
            token_info: dict with timestep_idx (B, T).
            n_timesteps: total number of timesteps T.

        Returns:
            (B, T, D) predicted feature values per timestep.
        """
        B = encoded_visible.shape[0]
        d_dec = self.config.decoder_d_model

        # Project visible tokens to decoder space
        vis_proj = self.encoder_proj(encoded_visible)  # (B, n_vis, d_dec)

        # Build full decoder input: mask tokens everywhere, then scatter visible
        full_tokens = self.mask_token.expand(B, n_timesteps, d_dec).clone()

        # Scatter visible tokens to their original positions
        vis_indices = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
        n_vis = vis_proj.shape[1]
        scatter_idx = vis_indices[:, :n_vis]  # (B, n_vis)
        scatter_idx_expanded = scatter_idx.unsqueeze(-1).expand(-1, -1, d_dec)
        full_tokens.scatter_(1, scatter_idx_expanded, vis_proj.to(full_tokens.dtype))

        # Add time PE to all positions
        timestep_idx = token_info["timestep_idx"]  # (B, T)
        full_tokens = full_tokens + self.time_pe[timestep_idx]
        full_tokens = self.embed_dropout(full_tokens)

        # Run decoder transformer (no padding mask needed, all T positions valid)
        decoded = self.decoder(full_tokens)

        # Predict D features per timestep
        predictions = self.output_proj(decoded)  # (B, T, D)

        return predictions


class MAEObjective(BaseSSLObjective):
    """Timestep-level Masked Autoencoder for ICU time-series.

    Flow:
    1. encoder.tokenize(x, obs_mask) -> timestep tokens + padding + info
    2. Random mask: mask_ratio of timesteps masked
    3. encoder.encode(visible_tokens) -> encoded visible
    4. decoder(encoded_visible, mask, info) -> predicted values (B, T, D)
    5. MSE loss on observed features at masked timesteps

    Requires encoder with tokenize()/encode() and pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: MAEConfig) -> None:
        super().__init__(encoder, config)
        self.config: MAEConfig = config

        # Validate encoder has obs-aware tokenization
        if not getattr(getattr(encoder, "config", None), "obs_aware", False):
            raise ValueError(
                "MAE requires an encoder with obs_aware=True "
                "(e.g., TransformerEncoder with obs_aware=True). Got: "
                f"{type(encoder).__name__}"
            )

        # Validate pooling
        encoder_pooling = getattr(encoder.config, "pooling", "none")
        if encoder_pooling != "none":
            raise ValueError(
                "MAE requires encoder with pooling='none' to get per-token "
                f"representations, but got pooling='{encoder_pooling}'"
            )

        d_encoder = encoder.get_output_dim()
        n_features = encoder.config.d_input
        max_seq_length = encoder.config.max_seq_length

        self.missing_token = None

        # Create decoder
        self.decoder = MAEDecoder(
            d_encoder=d_encoder,
            n_features=n_features,
            max_seq_length=max_seq_length,
            config=config,
        )

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute timestep-level MAE loss.

        Args:
            x: Input tensor (B, T, D).
            obs_mask: Observation mask (B, T, D), True = observed.

        Returns:
            (loss, metrics_dict)
        """
        B, T, D = x.shape
        device = x.device

        # 1. Tokenize timesteps
        tokens, padding_mask, token_info = self.encoder.tokenize(x, obs_mask)
        # tokens: (B, T, d_model)

        # 2. Create SSL mask on timesteps
        ssl_mask = create_timestep_mask(B, T, self.config.mask_ratio, device)

        # 3. Extract visible tokens
        visible_tokens, vis_padding = extract_visible_timesteps(tokens, ssl_mask)

        # 4. Encode visible tokens only
        encoded_visible = self.encoder.encode(visible_tokens, vis_padding)

        # 5. Decode full sequence
        predictions = self.decoder(
            encoded_visible=encoded_visible,
            ssl_mask=ssl_mask,
            token_info=token_info,
            n_timesteps=T,
        )  # (B, T, D)

        # 6. Compute loss on observed features at masked timesteps
        loss, metrics = self._compute_loss(predictions, x, ssl_mask, obs_mask)

        return loss, metrics

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        true_values: torch.Tensor,
        ssl_mask: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MSE loss on observed features at masked timesteps.

        Args:
            predictions: (B, T, D) predicted feature values.
            true_values: (B, T, D) original input values.
            ssl_mask: (B, T) True = visible, False = masked.
            obs_mask: (B, T, D) True = observed.

        Returns:
            (loss, metrics_dict)
        """
        # Loss mask: masked timesteps AND observed features -> (B, T, D)
        loss_mask = (~ssl_mask).unsqueeze(-1) & obs_mask

        squared_error = (predictions - true_values) ** 2
        loss = (squared_error * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

        with torch.no_grad():
            B, T, D = true_values.shape
            n_timesteps = T
            n_masked_timesteps = (~ssl_mask).sum().item()
            n_visible_timesteps = ssl_mask.sum().item()
            n_loss_positions = loss_mask.sum().item()

            # Visible reconstruction for monitoring
            visible_loss_mask = ssl_mask.unsqueeze(-1) & obs_mask
            if visible_loss_mask.sum() > 0:
                visible_loss = (
                    squared_error * visible_loss_mask.float()
                ).sum() / visible_loss_mask.float().sum()
            else:
                visible_loss = torch.tensor(0.0, device=loss.device)

            metrics = {
                "mae_loss": loss.detach(),
                "ssl_loss": loss.detach(),
                "mae_recon_loss_masked": loss.detach(),
                "mae_recon_loss_visible": visible_loss,
                "mae_mask_ratio_actual": n_masked_timesteps / max(B * n_timesteps, 1),
                "mae_n_timesteps": n_timesteps,
                "mae_n_visible_per_sample": n_visible_timesteps / B,
                "mae_n_masked_per_sample": n_masked_timesteps / B,
                "mae_n_loss_positions": n_loss_positions,
            }

        return loss, metrics
