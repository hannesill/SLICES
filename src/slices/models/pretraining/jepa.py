"""Joint-Embedding Predictive Architecture (JEPA) for ICU time-series SSL.

Observation-level tokenization variant: same masking as MAE, but predicts
latent representations of masked tokens instead of raw input values.

Architecture:
1. ObservationTransformerEncoder.tokenize() → one token per observed measurement
2. Random mask: 75% of observation tokens are masked
3. Online encoder processes only visible tokens (25%)
4. EMA target encoder processes ALL tokens → target representations
5. Predictor reassembles (visible encoded + mask tokens), predicts target repr
6. MSE/cosine loss on masked positions in latent space

Key difference from MAE: predicts in latent space (d_model vectors) not input space.
Key difference from Contrastive: local positional prediction, not global invariance.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLObjective, SSLConfig
from .masking import create_observation_mask, extract_visible


@dataclass
class JEPAConfig(SSLConfig):
    """Configuration for observation-level JEPA objective."""

    name: str = "jepa"

    # Masking
    mask_ratio: float = 0.75

    # Predictor parameters (mirrors MAE decoder for fairness)
    predictor_d_model: int = 128
    predictor_n_layers: int = 2
    predictor_n_heads: int = 4
    predictor_d_ff: int = 512
    predictor_dropout: float = 0.1

    # Momentum encoder
    momentum_base: float = 0.996
    momentum_final: float = 1.0

    # Loss
    loss_type: str = "mse"  # "mse" or "cosine"


class JEPAPredictor(nn.Module):
    """Lightweight transformer predictor for JEPA.

    Same architecture as MAEDecoder but output_proj maps to d_encoder
    (representation vectors) instead of scalar values.
    """

    def __init__(
        self,
        d_encoder: int,
        n_features: int,
        max_seq_length: int,
        config: JEPAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        d_pred = config.predictor_d_model

        self.encoder_proj = nn.Linear(d_encoder, d_pred)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.normal_(self.mask_token, std=0.02)

        self.feature_embed = nn.Embedding(n_features, d_pred)

        pe = self._build_sinusoidal_pe(max_seq_length, d_pred)
        self.register_buffer("time_pe", pe)

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=d_pred,
            nhead=config.predictor_n_heads,
            dim_feedforward=config.predictor_d_ff,
            dropout=config.predictor_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=config.predictor_n_layers,
        )

        self.embed_dropout = nn.Dropout(config.predictor_dropout)

        # Output projects to d_encoder (representation space), not scalar
        self.output_proj = nn.Linear(d_pred, d_encoder)

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
        """Predict target representations at all positions.

        Args:
            encoded_visible: (B, n_vis, d_encoder) encoded visible tokens.
            ssl_mask: (B, max_obs) bool, True = visible, False = masked.
            token_info: dict with timestep_idx, feature_idx (B, max_obs).
            max_tokens: total number of token positions (max_obs).
            token_padding_mask: (B, max_obs) True = valid token, False = padding.

        Returns:
            (B, max_tokens, d_encoder) predicted representations per token.
        """
        B = encoded_visible.shape[0]
        d_pred = self.config.predictor_d_model

        vis_proj = self.encoder_proj(encoded_visible)  # (B, n_vis, d_pred)

        full_tokens = self.mask_token.expand(B, max_tokens, d_pred).clone()

        # Scatter visible tokens to original positions (same logic as MAEDecoder)
        vis_indices = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
        n_vis = vis_proj.shape[1]
        scatter_idx = vis_indices[:, :n_vis]
        scatter_idx_expanded = scatter_idx.unsqueeze(-1).expand(-1, -1, d_pred)
        full_tokens.scatter_(1, scatter_idx_expanded, vis_proj)

        # Add positional information
        timestep_idx = token_info["timestep_idx"]
        feature_idx = token_info["feature_idx"]

        full_tokens = full_tokens + self.feature_embed(feature_idx)
        full_tokens = full_tokens + self.time_pe[timestep_idx]
        full_tokens = self.embed_dropout(full_tokens)

        key_padding_mask = ~token_padding_mask

        decoded = self.predictor(full_tokens, src_key_padding_mask=key_padding_mask)

        # Project to encoder representation space
        predictions = self.output_proj(decoded)  # (B, max_tokens, d_encoder)

        return predictions


class JEPAObjective(BaseSSLObjective):
    """Observation-level JEPA for ICU time-series.

    Flow:
    1. encoder.tokenize(x, obs_mask) → observation tokens + padding + info
    2. Random mask: 75% tokens masked, 25% visible
    3. Online encoder.encode(visible_tokens) → context representations
    4. EMA target encoder tokenize+encode(ALL tokens) → target representations
    5. Predictor(context, mask, info) → predicted representations
    6. MSE/cosine loss on masked positions in latent space

    Requires ObservationTransformerEncoder with pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: JEPAConfig) -> None:
        super().__init__(encoder, config)
        self.config: JEPAConfig = config

        # Validate encoder type
        if not hasattr(encoder, "tokenize") or not hasattr(encoder, "encode"):
            raise ValueError(
                "JEPA requires an encoder with tokenize() and encode() methods "
                "(e.g., ObservationTransformerEncoder). Got: "
                f"{type(encoder).__name__}"
            )

        # Validate pooling
        encoder_pooling = getattr(encoder.config, "pooling", "none")
        if encoder_pooling != "none":
            raise ValueError(
                "JEPA requires encoder with pooling='none' to get per-token "
                f"representations, but got pooling='{encoder_pooling}'"
            )

        d_encoder = encoder.get_output_dim()
        n_features = encoder.config.d_input
        max_seq_length = encoder.config.max_seq_length

        self.missing_token = None

        # Create EMA target encoder
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

        # Create predictor
        self.predictor = JEPAPredictor(
            d_encoder=d_encoder,
            n_features=n_features,
            max_seq_length=max_seq_length,
            config=config,
        )

        self._current_momentum = config.momentum_base

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute JEPA loss.

        Args:
            x: Input tensor (B, T, D).
            obs_mask: Observation mask (B, T, D), True = observed.

        Returns:
            (loss, metrics_dict)
        """
        B, T, D = x.shape
        device = x.device

        # 1. Tokenize observed measurements (online encoder)
        tokens, padding_mask, token_info = self.encoder.tokenize(x, obs_mask)
        max_obs = tokens.shape[1]

        # 2. Create SSL mask on observation tokens
        ssl_mask = create_observation_mask(padding_mask, self.config.mask_ratio, device)

        # 3. Extract visible tokens
        visible_tokens, vis_padding = extract_visible(tokens, ssl_mask, padding_mask)

        # 4. Encode visible tokens only (online encoder)
        encoded_visible = self.encoder.encode(visible_tokens, vis_padding)

        # 5. Target encoder processes ALL tokens (no masking, no grad)
        # Keep target encoder in eval mode to disable dropout — Lightning's
        # model.train() call at epoch start would otherwise re-enable it,
        # corrupting target representations with stochastic noise.
        self.target_encoder.eval()
        with torch.no_grad():
            target_tokens, target_padding, _ = self.target_encoder.tokenize(x, obs_mask)
            target_repr = self.target_encoder.encode(
                target_tokens, target_padding
            )  # (B, max_obs, d_model)

        # 6. Predictor predicts target representations
        predicted_repr = self.predictor(
            encoded_visible=encoded_visible,
            ssl_mask=ssl_mask,
            token_info=token_info,
            max_tokens=max_obs,
            token_padding_mask=padding_mask,
        )  # (B, max_obs, d_encoder)

        # 7. Compute loss on masked positions
        loss, metrics = self._compute_loss(predicted_repr, target_repr, ssl_mask, padding_mask)

        return loss, metrics

    def _compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        ssl_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss on masked token representations.

        Args:
            predicted: (B, max_obs, d_encoder) predicted representations.
            target: (B, max_obs, d_encoder) target representations.
            ssl_mask: (B, max_obs) True = visible, False = masked.
            padding_mask: (B, max_obs) True = valid token.

        Returns:
            (loss, metrics_dict)
        """
        # Loss only on masked AND valid positions
        loss_mask = (~ssl_mask) & padding_mask  # (B, max_obs)

        if self.config.loss_type == "mse":
            element_loss = F.mse_loss(predicted, target, reduction="none")
            # Average over d_encoder dimension
            element_loss = element_loss.mean(dim=-1)  # (B, max_obs)
        elif self.config.loss_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # (B, max_obs)
            element_loss = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        loss = (element_loss * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

        with torch.no_grad():
            B = padding_mask.shape[0]
            n_total_tokens = padding_mask.sum().item()
            n_masked = loss_mask.sum().item()
            n_visible = (ssl_mask & padding_mask).sum().item()

            metrics = {
                "jepa_loss": loss.detach(),
                "reconstruction_loss": loss.detach(),
                "jepa_mask_ratio_actual": n_masked / max(n_total_tokens, 1),
                "jepa_n_tokens_per_sample": n_total_tokens / B,
                "jepa_n_visible_per_sample": n_visible / B,
                "jepa_n_masked_per_sample": n_masked / B,
                "jepa_momentum": self._current_momentum,
            }

        return loss, metrics

    @torch.no_grad()
    def momentum_update(self, progress: float) -> None:
        """Update target encoder with EMA.

        Called by SSLPretrainModule.on_train_batch_end via duck-typing hook.

        Args:
            progress: Training progress as fraction in [0, 1].
        """
        m = (
            self.config.momentum_base
            + (self.config.momentum_final - self.config.momentum_base) * progress
        )
        self._current_momentum = m

        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(m).add_(online_param.data, alpha=1 - m)
