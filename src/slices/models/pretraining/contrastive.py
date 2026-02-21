"""Contrastive (SimCLR-style) SSL objective for ICU time-series.

Observation-level tokenization variant: uses two different random masks as
two augmented "views" of the same sample, then applies NT-Xent contrastive
loss on the pooled representations.

Architecture:
1. ObservationTransformerEncoder.tokenize() → one token per observed measurement
2. Two independent random masks → two different subsets of tokens (views)
3. Encoder processes each view separately → per-token representations
4. Mean-pool over visible tokens → sequence-level embeddings
5. Projection head → low-dimensional normalized embeddings
6. NT-Xent loss: positive pairs = same sample's two views

Key difference from MAE: discriminative (not reconstructive), global (not local).
Key difference from JEPA: global invariance (not local positional prediction).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLObjective, SSLConfig
from .masking import create_observation_mask, extract_visible


@dataclass
class ContrastiveConfig(SSLConfig):
    """Configuration for observation-level contrastive objective."""

    name: str = "contrastive"

    # Masking
    mask_ratio: float = 0.75

    # Projection head
    proj_hidden_dim: int = 512
    proj_output_dim: int = 128

    # Temperature for NT-Xent
    temperature: float = 0.1


class ProjectionHead(nn.Module):
    """MLP projection head with L2 normalization.

    Maps encoder output to a lower-dimensional space for contrastive loss.
    """

    def __init__(
        self,
        d_input: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize.

        Args:
            x: (B, d_input)

        Returns:
            (B, output_dim) L2-normalized projections.
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)


class ContrastiveObjective(BaseSSLObjective):
    """Observation-level contrastive (SimCLR-style) SSL for ICU time-series.

    Flow:
    1. encoder.tokenize(x, obs_mask) → observation tokens
    2. Two independent random masks → two views
    3. For each view: extract_visible → encode → mean_pool → (B, d_model)
    4. projection_head(pooled) → z1, z2 (B, proj_dim), L2-normalized
    5. NT-Xent loss: match positive pairs across views

    Requires ObservationTransformerEncoder with pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: ContrastiveConfig) -> None:
        super().__init__(encoder, config)
        self.config: ContrastiveConfig = config

        # Validate encoder type
        if not hasattr(encoder, "tokenize") or not hasattr(encoder, "encode"):
            raise ValueError(
                "Contrastive requires an encoder with tokenize() and encode() "
                "methods (e.g., ObservationTransformerEncoder). Got: "
                f"{type(encoder).__name__}"
            )

        # Validate pooling
        encoder_pooling = getattr(encoder.config, "pooling", "none")
        if encoder_pooling != "none":
            raise ValueError(
                "Contrastive requires encoder with pooling='none' to get "
                "per-token representations for mean-pooling, but got "
                f"pooling='{encoder_pooling}'"
            )

        d_encoder = encoder.get_output_dim()

        self.missing_token = None

        self.projection_head = ProjectionHead(
            d_input=d_encoder,
            hidden_dim=config.proj_hidden_dim,
            output_dim=config.proj_output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute contrastive NT-Xent loss.

        Args:
            x: Input tensor (B, T, D).
            obs_mask: Observation mask (B, T, D), True = observed.

        Returns:
            (loss, metrics_dict)
        """
        B, T, D = x.shape
        device = x.device

        # 1. Tokenize (shared between both views)
        tokens, padding_mask, token_info = self.encoder.tokenize(x, obs_mask)
        # 2. Two independent random masks
        ssl_mask_1 = create_observation_mask(padding_mask, self.config.mask_ratio, device)
        ssl_mask_2 = create_observation_mask(padding_mask, self.config.mask_ratio, device)

        # 3. View 1: extract → encode → mean pool
        vis_tokens_1, vis_padding_1 = extract_visible(tokens, ssl_mask_1, padding_mask)
        encoded_1 = self.encoder.encode(vis_tokens_1, vis_padding_1)
        pooled_1 = self._mean_pool(encoded_1, vis_padding_1)  # (B, d_model)

        # 4. View 2: extract → encode → mean pool
        vis_tokens_2, vis_padding_2 = extract_visible(tokens, ssl_mask_2, padding_mask)
        encoded_2 = self.encoder.encode(vis_tokens_2, vis_padding_2)
        pooled_2 = self._mean_pool(encoded_2, vis_padding_2)  # (B, d_model)

        # 5. Project to contrastive space
        z1 = self.projection_head(pooled_1)  # (B, proj_dim)
        z2 = self.projection_head(pooled_2)  # (B, proj_dim)

        # 6. NT-Xent loss
        loss, metrics = self._nt_xent_loss(z1, z2, ssl_mask_1, ssl_mask_2, padding_mask)

        return loss, metrics

    @staticmethod
    def _mean_pool(
        encoded: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool over valid tokens.

        Args:
            encoded: (B, n_vis, d_model)
            padding_mask: (B, n_vis) True = valid

        Returns:
            (B, d_model)
        """
        mask_expanded = padding_mask.unsqueeze(-1).float()  # (B, n_vis, 1)
        summed = (encoded * mask_expanded).sum(dim=1)  # (B, d_model)
        counts = padding_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)
        return summed / counts

    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        ssl_mask_1: torch.Tensor,
        ssl_mask_2: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            z1: (B, proj_dim) L2-normalized projections from view 1.
            z2: (B, proj_dim) L2-normalized projections from view 2.
            ssl_mask_1: (B, max_obs) mask for view 1 (for metrics).
            ssl_mask_2: (B, max_obs) mask for view 2 (for metrics).
            padding_mask: (B, max_obs) padding mask (for metrics).

        Returns:
            (loss, metrics_dict)
        """
        B = z1.shape[0]
        temperature = self.config.temperature

        # Concatenate both views: (2B, proj_dim)
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix: (2B, 2B)
        sim_matrix = torch.mm(z, z.t()) / temperature

        # Create labels: positive pair for i is i+B (and vice versa)
        labels = torch.cat(
            [torch.arange(B, 2 * B, device=z.device), torch.arange(B, device=z.device)]
        )

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        with torch.no_grad():
            # Top-1 retrieval accuracy
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

            # Positive pair similarities (before temperature scaling)
            pos_sim = F.cosine_similarity(z1, z2, dim=-1).mean()

            # Token statistics
            n_total = padding_mask.sum().item()
            n_vis_1 = (ssl_mask_1 & padding_mask).sum().item()
            n_vis_2 = (ssl_mask_2 & padding_mask).sum().item()
            n_masked_1 = ((~ssl_mask_1) & padding_mask).sum().item()
            n_masked_2 = ((~ssl_mask_2) & padding_mask).sum().item()

            metrics = {
                "contrastive_loss": loss.detach(),
                "reconstruction_loss": loss.detach(),
                "contrastive_accuracy": accuracy,
                "contrastive_pos_similarity": pos_sim,
                "contrastive_temperature": temperature,
                "contrastive_n_tokens_per_sample": n_total / B,
                "contrastive_n_visible_view1": n_vis_1 / B,
                "contrastive_n_visible_view2": n_vis_2 / B,
                "contrastive_n_masked_view1": n_masked_1 / B,
                "contrastive_n_masked_view2": n_masked_2 / B,
            }

        return loss, metrics
