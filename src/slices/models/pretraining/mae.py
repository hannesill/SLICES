"""Masked Autoencoder (MAE) for self-supervised learning on ICU time-series.

Observation-level tokenization variant: each token = one observed (timestep, feature, value)
triplet. The encoder only sees visible (unmasked) tokens, matching the original MAE design.

Architecture:
1. ObservationTransformerEncoder.tokenize() → one token per observed measurement
2. Random mask: 50% of observation tokens are masked (configurable via mask_ratio)
3. Encoder processes only visible tokens (50%)
4. Decoder reassembles full sequence (visible + mask tokens), predicts scalar values
5. MSE loss on masked token values only
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import BaseSSLObjective, SSLConfig
from .masking import create_observation_mask, extract_visible


@dataclass
class MAEConfig(SSLConfig):
    """Configuration for observation-level MAE objective.

    The encoder only sees visible (unmasked) observation tokens.
    Mask ratio is applied to observation tokens (not timesteps).
    """

    name: str = "mae"

    # Masking
    mask_ratio: float = 0.5  # Fraction of observation tokens to mask

    # Decoder parameters
    decoder_d_model: int = 128
    decoder_n_layers: int = 2
    decoder_n_heads: int = 4
    decoder_d_ff: int = 512
    decoder_dropout: float = 0.1


class MAEDecoder(nn.Module):
    """Lightweight decoder for observation-level MAE.

    Reassembles encoded visible tokens with learnable mask tokens at masked
    positions. Adds positional information (feature + temporal) to all tokens
    before running through lightweight transformer layers. Predicts a scalar
    value per token.
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

        # Positional information for decoder
        self.feature_embed = nn.Embedding(n_features, d_dec)

        # Sinusoidal time PE for decoder
        pe = self._build_sinusoidal_pe(max_seq_length, d_dec)
        self.register_buffer("time_pe", pe)

        # Decoder transformer layers.
        # PyTorch's TransformerEncoderLayer is used here (not DecoderLayer) because
        # MAE's decoder uses self-attention over all tokens (visible + mask), not
        # cross-attention between encoder output and mask tokens.
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

        # Output: predict scalar value per token
        self.output_proj = nn.Linear(d_dec, 1)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(
        self,
        encoded_visible: torch.Tensor,
        ssl_mask: torch.Tensor,
        token_info: Dict[str, torch.Tensor],
        max_tokens: int,
        token_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode: reassemble visible + mask tokens, predict values.

        Args:
            encoded_visible: (B, n_vis, d_encoder) encoded visible tokens.
            ssl_mask: (B, max_obs) bool, True = visible, False = masked.
            token_info: dict with timestep_idx, feature_idx (B, max_obs).
            max_tokens: total number of token positions (max_obs).
            token_padding_mask: (B, max_obs) True = valid token, False = padding.

        Returns:
            (B, max_tokens) predicted scalar value per token position.
        """
        B = encoded_visible.shape[0]
        d_dec = self.config.decoder_d_model

        # Project visible tokens to decoder space
        vis_proj = self.encoder_proj(encoded_visible)  # (B, n_vis, d_dec)

        # Build full decoder input: place visible tokens and mask tokens
        full_tokens = self.mask_token.expand(B, max_tokens, d_dec).clone()

        # Place visible tokens at their original positions.
        # ssl_mask is (B, max_obs), True at visible positions.
        # Stable descending argsort on ssl_mask.float() puts visible (1.0) positions
        # first. This mirrors extract_visible(), so vis_indices[:, :n_vis] gives
        # the original positions of the n_vis visible tokens in the same order
        # they were fed to the encoder.
        # Padding invariant: padding positions have ssl_mask=True (see
        # create_observation_mask), so they sort among visible positions.
        # But because tokenization places valid tokens before padding and argsort
        # is stable, valid-visible tokens always precede padding tokens in the
        # sorted order. We only scatter the first n_vis entries, which are the
        # actual visible tokens, not padding.
        vis_indices = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
        # vis_indices[:, :n_vis] gives the original positions of visible tokens
        n_vis = vis_proj.shape[1]
        scatter_idx = vis_indices[:, :n_vis]  # (B, n_vis)
        scatter_idx_expanded = scatter_idx.unsqueeze(-1).expand(-1, -1, d_dec)  # (B, n_vis, d_dec)
        full_tokens.scatter_(1, scatter_idx_expanded, vis_proj)

        # Add positional information to ALL token positions
        timestep_idx = token_info["timestep_idx"]  # (B, max_tokens)
        feature_idx = token_info["feature_idx"]  # (B, max_tokens)

        full_tokens = full_tokens + self.feature_embed(feature_idx)
        full_tokens = full_tokens + self.time_pe[timestep_idx]
        full_tokens = self.embed_dropout(full_tokens)

        # Build key_padding_mask for decoder (True = ignore in PyTorch convention)
        key_padding_mask = ~token_padding_mask  # (B, max_tokens)

        # Run decoder transformer
        decoded = self.decoder(full_tokens, src_key_padding_mask=key_padding_mask)

        # Predict scalar value per token
        predictions = self.output_proj(decoded).squeeze(-1)  # (B, max_tokens)

        return predictions


class MAEObjective(BaseSSLObjective):
    """Observation-level Masked Autoencoder for ICU time-series.

    Flow:
    1. encoder.tokenize(x, obs_mask) → observation tokens + padding + info
    2. Random mask: 75% of tokens masked, 25% visible
    3. encoder.encode(visible_tokens) → encoded visible
    4. decoder(encoded_visible, mask, info) → predicted values per token
    5. MSE loss on masked token values

    Requires ObservationTransformerEncoder with pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: MAEConfig) -> None:
        super().__init__(encoder, config)
        self.config: MAEConfig = config

        # Validate encoder type
        if not hasattr(encoder, "tokenize") or not hasattr(encoder, "encode"):
            raise ValueError(
                "MAE requires an encoder with tokenize() and encode() methods "
                "(e.g., ObservationTransformerEncoder). Got: "
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

        # No missing_token needed -- observation encoder handles this intrinsically
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
        """Compute observation-level MAE loss.

        Args:
            x: Input tensor (B, T, D).
            obs_mask: Observation mask (B, T, D), True = observed.

        Returns:
            (loss, metrics_dict)
        """
        B, T, D = x.shape
        device = x.device

        # 1. Tokenize observed measurements
        tokens, padding_mask, token_info = self.encoder.tokenize(x, obs_mask)
        # tokens: (B, max_obs, d_model), padding_mask: (B, max_obs)
        max_obs = tokens.shape[1]
        true_values = token_info["values"]  # (B, max_obs)

        # 2. Create SSL mask on observation tokens
        # ssl_mask: True = visible, False = masked
        ssl_mask = create_observation_mask(padding_mask, self.config.mask_ratio, device)

        # 3. Extract visible tokens
        visible_tokens, vis_padding = extract_visible(tokens, ssl_mask, padding_mask)

        # 4. Encode visible tokens only
        encoded_visible = self.encoder.encode(visible_tokens, vis_padding)

        # 5. Decode full sequence
        predictions = self.decoder(
            encoded_visible=encoded_visible,
            ssl_mask=ssl_mask,
            token_info=token_info,
            max_tokens=max_obs,
            token_padding_mask=padding_mask,
        )

        # 6. Compute loss on masked token values
        loss, metrics = self._compute_loss(predictions, true_values, ssl_mask, padding_mask)

        return loss, metrics

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        true_values: torch.Tensor,
        ssl_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute MSE loss on masked token predictions.

        Args:
            predictions: (B, max_obs) predicted values.
            true_values: (B, max_obs) original observed values.
            ssl_mask: (B, max_obs) True = visible, False = masked.
            padding_mask: (B, max_obs) True = valid token.

        Returns:
            (loss, metrics_dict)
        """
        # Loss only on masked AND valid positions
        loss_mask = (~ssl_mask) & padding_mask  # (B, max_obs)

        squared_error = (predictions - true_values) ** 2
        loss = (squared_error * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

        with torch.no_grad():
            B = padding_mask.shape[0]
            n_total_tokens = padding_mask.sum().item()
            n_masked = loss_mask.sum().item()
            n_visible = (ssl_mask & padding_mask).sum().item()

            # Visible (unmasked) reconstruction for monitoring
            visible_mask = ssl_mask & padding_mask
            if visible_mask.sum() > 0:
                visible_loss = (
                    squared_error * visible_mask.float()
                ).sum() / visible_mask.float().sum()
            else:
                visible_loss = torch.tensor(0.0, device=loss.device)

            metrics = {
                "mae_loss": loss.detach(),
                "ssl_loss": loss.detach(),
                "mae_recon_loss_masked": loss.detach(),
                "mae_recon_loss_visible": visible_loss,
                "mae_mask_ratio_actual": n_masked / max(n_total_tokens, 1),
                "mae_obs_ratio": n_total_tokens
                / max(padding_mask.shape[0] * padding_mask.shape[1], 1),
                "mae_n_tokens_per_sample": n_total_tokens / B,
                "mae_n_visible_per_sample": n_visible / B,
                "mae_n_masked_per_sample": n_masked / B,
            }

        return loss, metrics
