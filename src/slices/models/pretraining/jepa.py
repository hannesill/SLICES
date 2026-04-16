"""Joint-Embedding Predictive Architecture (JEPA) for ICU time-series SSL.

Timestep-level tokenization variant: same masking as MAE, but predicts
latent representations of masked tokens instead of raw input values.

Architecture:
1. TransformerEncoder.tokenize() -> one token per timestep (B, T, d_model)
2. Random mask: 50% of timestep tokens are masked (configurable via mask_ratio)
3. Online encoder processes only visible tokens
4. EMA target encoder processes ALL tokens -> target representations
5. Predictor reassembles (visible encoded + mask tokens), predicts target repr
6. MSE/cosine loss on masked positions in latent space

Key difference from MAE: predicts in latent space (d_model vectors) not input space.
Key difference from Contrastive: local positional prediction, not global invariance.
"""

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from slices.models.common import build_sinusoidal_pe

from .base import BaseSSLObjective, SSLConfig
from .masking import create_block_timestep_mask, create_timestep_mask, extract_visible_timesteps


@dataclass
class JEPAConfig(SSLConfig):
    """Configuration for timestep-level JEPA objective."""

    name: str = "jepa"

    # Masking
    mask_ratio: float = 0.5
    mask_strategy: str = "block"  # "random" or "block"
    mask_n_blocks: int = 3  # Number of contiguous blocks (only for block strategy)

    # Predictor parameters (mirrors MAE decoder for fairness)
    predictor_d_model: int = 128
    predictor_n_layers: int = 2
    predictor_n_heads: int = 4
    predictor_d_ff: int = 512
    predictor_dropout: float = 0.1

    # Momentum encoder
    momentum_base: float = 0.999
    momentum_final: float = 1.0

    # Loss
    loss_type: str = "mse"  # "mse" or "cosine"


class JEPAPredictor(nn.Module):
    """Lightweight transformer predictor for JEPA.

    Same architecture as MAEDecoder but output_proj maps to d_encoder
    (representation vectors) instead of D feature values.
    """

    def __init__(
        self,
        d_encoder: int,
        max_seq_length: int,
        config: JEPAConfig,
    ) -> None:
        super().__init__()
        self.config = config
        d_pred = config.predictor_d_model

        self.encoder_proj = nn.Linear(d_encoder, d_pred)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_pred))
        nn.init.normal_(self.mask_token, std=0.02)

        self.register_buffer("time_pe", build_sinusoidal_pe(max_seq_length, d_pred))

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

        # Output projects to d_encoder (representation space)
        self.output_proj = nn.Linear(d_pred, d_encoder)

    def forward(
        self,
        encoded_visible: torch.Tensor,
        ssl_mask: torch.Tensor,
        token_info: Dict[str, torch.Tensor],
        n_timesteps: int,
    ) -> torch.Tensor:
        """Predict target representations at all timestep positions.

        Args:
            encoded_visible: (B, n_vis, d_encoder) encoded visible tokens.
            ssl_mask: (B, T) bool, True = visible, False = masked.
            token_info: dict with timestep_idx (B, T) and optional
                valid_timestep_mask (B, T) marking timesteps that had at
                least one observed variable.
            n_timesteps: total number of timesteps T.

        Returns:
            (B, T, d_encoder) predicted representations per timestep.
        """
        B = encoded_visible.shape[0]
        d_pred = self.config.predictor_d_model

        vis_proj = self.encoder_proj(encoded_visible)  # (B, n_vis, d_pred)

        full_tokens = self.mask_token.expand(B, n_timesteps, d_pred).clone()

        # Scatter visible tokens to original positions
        vis_indices = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
        n_vis = vis_proj.shape[1]
        scatter_idx = vis_indices[:, :n_vis]
        scatter_idx_expanded = scatter_idx.unsqueeze(-1).expand(-1, -1, d_pred)
        full_tokens.scatter_(1, scatter_idx_expanded, vis_proj.to(full_tokens.dtype))

        # Add time PE
        timestep_idx = token_info["timestep_idx"]
        full_tokens = full_tokens + self.time_pe[timestep_idx]
        full_tokens = self.embed_dropout(full_tokens)

        valid_timestep_mask = token_info.get("valid_timestep_mask")
        predictor_padding_mask = None
        if valid_timestep_mask is not None:
            predictor_padding_mask = ~valid_timestep_mask.to(
                device=full_tokens.device,
                dtype=torch.bool,
            )

        # Fully unobserved hours should not be available as predictor context.
        decoded = self.predictor(full_tokens, src_key_padding_mask=predictor_padding_mask)

        # Project to encoder representation space
        predictions = self.output_proj(decoded)  # (B, T, d_encoder)

        return predictions


class JEPAObjective(BaseSSLObjective):
    """Timestep-level JEPA for ICU time-series.

    Flow:
    1. encoder.tokenize(x, obs_mask) -> timestep tokens + padding + info
    2. Random mask: mask_ratio timesteps masked
    3. Online encoder.encode(visible_tokens) -> context representations
    4. EMA target encoder tokenize+encode(ALL tokens) -> target representations
    5. Predictor(context, mask, info) -> predicted representations
    6. MSE/cosine loss on masked positions in latent space

    Requires encoder with tokenize()/encode() and pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: JEPAConfig) -> None:
        super().__init__(encoder, config)
        self.config: JEPAConfig = config

        # Validate encoder has obs-aware tokenization
        if not getattr(getattr(encoder, "config", None), "obs_aware", False):
            raise ValueError(
                "JEPA requires an encoder with obs_aware=True "
                "(e.g., TransformerEncoder with obs_aware=True). Got: "
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
        self.d_encoder = d_encoder
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

        # 1. Tokenize timesteps (online encoder)
        tokens, padding_mask, token_info = self.encoder.tokenize(x, obs_mask)
        valid_timestep_mask = token_info["valid_timestep_mask"]

        # 2. Create SSL mask on timesteps
        if self.config.mask_strategy == "block":
            ssl_mask = create_block_timestep_mask(
                B,
                T,
                self.config.mask_ratio,
                device,
                n_blocks=self.config.mask_n_blocks,
                valid_timestep_mask=valid_timestep_mask,
            )
        else:
            ssl_mask = create_timestep_mask(
                B,
                T,
                self.config.mask_ratio,
                device,
                valid_timestep_mask=valid_timestep_mask,
            )
        effective_visible_mask = ssl_mask & valid_timestep_mask

        # 3. Extract visible tokens
        visible_tokens, vis_padding = extract_visible_timesteps(
            tokens,
            ssl_mask,
            valid_timestep_mask=valid_timestep_mask,
        )

        # 4. Encode visible tokens only (online encoder)
        encoded_visible = self.encoder.encode(visible_tokens, vis_padding)

        # 5. Target encoder processes ALL tokens (no masking, no grad)
        self.target_encoder.eval()
        with torch.no_grad():
            target_tokens, target_padding, _ = self.target_encoder.tokenize(x, obs_mask)
            target_repr = self.target_encoder.encode(
                target_tokens, target_padding
            )  # (B, T, d_model)

        # 6. Predictor predicts target representations
        predicted_repr = self.predictor(
            encoded_visible=encoded_visible,
            ssl_mask=effective_visible_mask,
            token_info=token_info,
            n_timesteps=T,
        )  # (B, T, d_encoder)

        # 7. Compute loss on masked positions
        loss, metrics = self._compute_loss(
            predicted_repr,
            target_repr,
            ssl_mask,
            valid_timestep_mask,
        )

        return loss, metrics

    def _compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        ssl_mask: torch.Tensor,
        valid_timestep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss on masked timestep representations.

        Args:
            predicted: (B, T, d_encoder) predicted representations.
            target: (B, T, d_encoder) target representations.
            ssl_mask: (B, T) True = visible, False = masked.

        Returns:
            (loss, metrics_dict)
        """
        # Loss only on masked timesteps that had at least one observation.
        loss_mask = (~ssl_mask) & valid_timestep_mask  # (B, T)

        # Collapse monitoring metrics (computed on raw target before normalization)
        with torch.no_grad():
            B, T = ssl_mask.shape
            n_masked = loss_mask.sum().item()
            n_visible = (ssl_mask & valid_timestep_mask).sum().item()

            # Flatten to (B*T, d_encoder) for batch-level statistics
            target_flat = target.reshape(-1, self.d_encoder)

            # Std of each feature dim across all positions — collapse → 0
            target_repr_std = target_flat.std(dim=0).mean().item()

            # Mean pairwise cosine similarity (sampled for efficiency)
            # Sample up to 256 vectors to keep cost O(1)
            n_vecs = target_flat.shape[0]
            if n_vecs > 256:
                idx = torch.randperm(n_vecs, device=target_flat.device)[:256]
                sampled = target_flat[idx]
            else:
                sampled = target_flat
            sampled_norm = F.normalize(sampled, dim=-1)
            cos_matrix = sampled_norm @ sampled_norm.T
            # Exclude diagonal (self-similarity = 1)
            n_sampled = sampled_norm.shape[0]
            mask_diag = ~torch.eye(n_sampled, dtype=torch.bool, device=cos_matrix.device)
            target_repr_mean_cos = cos_matrix[mask_diag].mean().item()

        # Normalize targets to prevent scale/shift collapse
        target = F.layer_norm(target, [self.d_encoder])

        if self.config.loss_type == "mse":
            element_loss = F.mse_loss(predicted, target, reduction="none")
            element_loss = element_loss.mean(dim=-1)  # (B, T)
        elif self.config.loss_type == "cosine":
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # (B, T)
            element_loss = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        loss = (element_loss * loss_mask.float()).sum() / loss_mask.float().sum().clamp(min=1)

        with torch.no_grad():
            metrics = {
                "jepa_loss": loss.detach(),
                "ssl_loss": loss.detach(),
                "jepa_mask_ratio_actual": n_masked / max(valid_timestep_mask.sum().item(), 1),
                "jepa_n_timesteps": T,
                "jepa_n_visible_per_sample": n_visible / B,
                "jepa_n_masked_per_sample": n_masked / B,
                "jepa_momentum": self._current_momentum,
                "jepa_target_repr_std": target_repr_std,
                "jepa_target_repr_mean_cos": target_repr_mean_cos,
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
