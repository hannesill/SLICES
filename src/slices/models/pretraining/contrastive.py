"""Contrastive (SimCLR-style) SSL objective for ICU time-series.

Timestep-level tokenization variant: creates two "views" of the same sample
via timestep masks, then applies NT-Xent contrastive loss.

By default, views use **complementary masks** (view 2 = ~view 1), ensuring
zero overlap and forcing the encoder to learn abstract temporal semantics
from non-overlapping windows. Independent random masks are available as a
fallback (complementary_masks=False).

Supports two modes:
- **instance** (default): SimCLR-style — mean-pool each view to a single
  sequence-level embedding, then NT-Xent on a (2B x 2B) matrix.
  Compatible with both complementary and independent masks.
- **temporal**: Positive pairs are formed from timesteps visible in
  both views (~25% at mask_ratio=0.5). Each overlapping timestep has two
  independently contextualized representations (different self-attention
  contexts) that form a natural positive pair.  All other tokens across the
  batch are negatives.  NT-Xent operates on a (2N x 2N) matrix where
  N = number of overlap tokens.  Requires complementary_masks=False.

Architecture:
1. TransformerEncoder.tokenize() -> one token per timestep (B, T, d_model)
2. Complementary masks (default) or two independent random masks -> two views
3. Encoder processes each view separately -> per-token representations
4. instance: mean-pool -> project -> NT-Xent
   temporal: scatter to (B,T,d) -> gather overlap tokens -> project -> NT-Xent

Key difference from MAE: discriminative (not reconstructive).
Key difference from JEPA: invariance (not positional prediction).
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLObjective, SSLConfig
from .masking import create_timestep_mask, extract_visible_timesteps


@dataclass
class ContrastiveConfig(SSLConfig):
    """Configuration for timestep-level contrastive objective."""

    name: str = "contrastive"

    # Mode: "temporal" (per-timestep overlap pairs) or "instance" (mean-pool)
    mode: str = "instance"

    # Masking
    mask_ratio: float = 0.5

    # Projection head
    proj_hidden_dim: int = 512
    proj_output_dim: int = 128

    # Temperature for NT-Xent
    temperature: float = 0.07

    # Use complementary (non-overlapping) masks for views
    complementary_masks: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("temporal", "instance"):
            raise ValueError(
                f"ContrastiveConfig.mode must be 'temporal' or 'instance', " f"got '{self.mode}'"
            )
        if self.complementary_masks and self.mode == "temporal":
            raise ValueError(
                "complementary_masks=True is incompatible with mode='temporal'. "
                "Temporal mode requires overlapping masks to form positive pairs."
            )


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
            nn.LayerNorm(hidden_dim),
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
    """Timestep-level contrastive SSL for ICU time-series.

    Supports temporal mode (per-timestep overlap pairs) and instance mode
    (mean-pool, SimCLR-style).  See module docstring for details.

    Requires encoder with tokenize()/encode() and pooling='none'.
    """

    def __init__(self, encoder: nn.Module, config: ContrastiveConfig) -> None:
        super().__init__(encoder, config)
        self.config: ContrastiveConfig = config

        # Validate encoder has obs-aware tokenization
        if not getattr(getattr(encoder, "config", None), "obs_aware", False):
            raise ValueError(
                "Contrastive requires an encoder with obs_aware=True "
                "(e.g., TransformerEncoder with obs_aware=True). Got: "
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
        valid_timestep_mask = token_info["valid_timestep_mask"]

        # 2. Create two views via timestep masks
        ssl_mask_1 = create_timestep_mask(
            B,
            T,
            self.config.mask_ratio,
            device,
            valid_timestep_mask=valid_timestep_mask,
        )
        if self.config.complementary_masks:
            ssl_mask_2 = ~ssl_mask_1
        else:
            ssl_mask_2 = create_timestep_mask(
                B,
                T,
                self.config.mask_ratio,
                device,
                valid_timestep_mask=valid_timestep_mask,
            )

        effective_mask_1 = ssl_mask_1 & valid_timestep_mask
        effective_mask_2 = ssl_mask_2 & valid_timestep_mask

        # 3. Encode both views
        vis_tokens_1, vis_padding_1 = extract_visible_timesteps(
            tokens,
            ssl_mask_1,
            valid_timestep_mask=valid_timestep_mask,
        )
        encoded_1 = self.encoder.encode(vis_tokens_1, vis_padding_1)

        vis_tokens_2, vis_padding_2 = extract_visible_timesteps(
            tokens,
            ssl_mask_2,
            valid_timestep_mask=valid_timestep_mask,
        )
        encoded_2 = self.encoder.encode(vis_tokens_2, vis_padding_2)

        if self.config.mode == "temporal":
            # 4a. Scatter encoded tokens back to full (B, T, d) grid
            full_1 = self._scatter_to_full(encoded_1, effective_mask_1, T)
            full_2 = self._scatter_to_full(encoded_2, effective_mask_2, T)

            # 5a. Temporal NT-Xent on overlapping timesteps
            loss, metrics = self._temporal_nt_xent_loss(
                full_1,
                full_2,
                effective_mask_1,
                effective_mask_2,
            )
        else:
            # 4b. Instance mode: mean-pool -> project -> instance NT-Xent
            pooled_1 = self._mean_pool(encoded_1, vis_padding_1)
            pooled_2 = self._mean_pool(encoded_2, vis_padding_2)

            z1 = self.projection_head(pooled_1)  # (B, proj_dim)
            z2 = self.projection_head(pooled_2)  # (B, proj_dim)

            loss, metrics = self._nt_xent_loss(z1, z2, effective_mask_1, effective_mask_2)

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

    @staticmethod
    def _scatter_to_full(
        encoded: torch.Tensor,
        ssl_mask: torch.Tensor,
        n_timesteps: int,
    ) -> torch.Tensor:
        """Scatter visible encoded tokens back to full (B, T, d) tensor.

        Uses the same argsort-scatter pattern as the MAE decoder and JEPA
        predictor to place encoded visible tokens at their original temporal
        positions, with zeros at masked positions.

        Args:
            encoded: (B, n_vis, d_enc) encoded visible tokens.
            ssl_mask: (B, T) True = visible.
            n_timesteps: Total number of timesteps T.

        Returns:
            (B, T, d_enc) with encoded tokens at visible positions, zeros elsewhere.
        """
        B, n_vis, d_enc = encoded.shape
        device = encoded.device

        full = torch.zeros(B, n_timesteps, d_enc, device=device, dtype=encoded.dtype)

        vis_indices = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
        scatter_idx = vis_indices[:, :n_vis].unsqueeze(-1).expand(-1, -1, d_enc)
        full.scatter_(1, scatter_idx, encoded)

        return full

    def _temporal_nt_xent_loss(
        self,
        full_1: torch.Tensor,
        full_2: torch.Tensor,
        ssl_mask_1: torch.Tensor,
        ssl_mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute temporal NT-Xent loss on overlapping timestep tokens.

        Timesteps visible in both views have two independently contextualized
        representations (from different self-attention contexts). These form
        natural positive pairs. All other tokens in the batch are negatives.

        Args:
            full_1: (B, T, d_enc) scattered encoded tokens from view 1.
            full_2: (B, T, d_enc) scattered encoded tokens from view 2.
            ssl_mask_1: (B, T) True = visible in view 1.
            ssl_mask_2: (B, T) True = visible in view 2.

        Returns:
            (loss, metrics_dict)
        """
        B = ssl_mask_1.shape[0]
        T = ssl_mask_1.shape[1]
        temperature = self.config.temperature

        overlap = ssl_mask_1 & ssl_mask_2  # (B, T)
        N = int(overlap.sum().item())

        # Edge case: not enough overlap tokens for contrastive learning
        if N < 2:
            loss = full_1.sum() * 0.0  # zero with grad connectivity
            with torch.no_grad():
                metrics = {
                    "contrastive_loss": loss.detach(),
                    "ssl_loss": loss.detach(),
                    "contrastive_accuracy": torch.tensor(0.0),
                    "contrastive_pos_similarity": torch.tensor(0.0),
                    "contrastive_temperature": temperature,
                    "contrastive_n_timesteps": T,
                    "contrastive_n_visible_view1": ssl_mask_1.sum().item() / B,
                    "contrastive_n_visible_view2": ssl_mask_2.sum().item() / B,
                    "contrastive_n_masked_view1": (~ssl_mask_1).sum().item() / B,
                    "contrastive_n_masked_view2": (~ssl_mask_2).sum().item() / B,
                    "contrastive_n_overlap_tokens": 0,
                    "contrastive_n_overlap_per_sample": 0.0,
                }
            return loss, metrics

        # Gather overlap tokens — boolean indexing in row-major order ensures
        # full_1[overlap] and full_2[overlap] are aligned (same batch, time pairs)
        tokens_1 = full_1[overlap]  # (N, d_enc)
        tokens_2 = full_2[overlap]  # (N, d_enc)

        # Project per-token through shared projection head
        z1 = self.projection_head(tokens_1)  # (N, proj_dim)
        z2 = self.projection_head(tokens_2)  # (N, proj_dim)

        # NT-Xent on (2N, 2N) similarity matrix
        z = torch.cat([z1, z2], dim=0)  # (2N, proj_dim)
        sim_matrix = torch.mm(z, z.t()) / temperature  # (2N, 2N)

        labels = torch.cat(
            [
                torch.arange(N, 2 * N, device=z.device),
                torch.arange(N, device=z.device),
            ]
        )

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        loss = F.cross_entropy(sim_matrix, labels)

        with torch.no_grad():
            preds = sim_matrix.argmax(dim=1)
            accuracy = (preds == labels).float().mean()
            pos_sim = F.cosine_similarity(z1, z2, dim=-1).mean()

            metrics = {
                "contrastive_loss": loss.detach(),
                "ssl_loss": loss.detach(),
                "contrastive_accuracy": accuracy,
                "contrastive_pos_similarity": pos_sim,
                "contrastive_temperature": temperature,
                "contrastive_n_timesteps": T,
                "contrastive_n_visible_view1": ssl_mask_1.sum().item() / B,
                "contrastive_n_visible_view2": ssl_mask_2.sum().item() / B,
                "contrastive_n_masked_view1": (~ssl_mask_1).sum().item() / B,
                "contrastive_n_masked_view2": (~ssl_mask_2).sum().item() / B,
                "contrastive_n_overlap_tokens": N,
                "contrastive_n_overlap_per_sample": N / B,
            }

        return loss, metrics

    def _nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        ssl_mask_1: torch.Tensor,
        ssl_mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            z1: (B, proj_dim) L2-normalized projections from view 1.
            z2: (B, proj_dim) L2-normalized projections from view 2.
            ssl_mask_1: (B, T) mask for view 1 (for metrics).
            ssl_mask_2: (B, T) mask for view 2 (for metrics).

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

            # --- Collapse monitoring (Wang & Isola 2020) ---
            # Alignment: mean L2 distance between positive pairs (lower = better)
            alignment = (z1 - z2).norm(dim=-1).pow(2).mean()

            # Uniformity: log avg pairwise Gaussian potential on hypersphere
            # Lower = more uniform = better spread of representations
            # Use z1 only (one view per sample) to avoid inflating with positives
            sq_pdist = torch.cdist(z1, z1, p=2).pow(2)  # (B, B)
            uniformity = sq_pdist.mul(-2).exp().mean().log()

            # Effective rank via singular value entropy (Roy & Vetterli 2007)
            # High effective rank = diverse representation dimensions
            # Low effective rank → collapse to a low-dim subspace
            _, s, _ = torch.svd_lowrank(z1 - z1.mean(dim=0), q=min(B, z1.shape[1]))
            p = s.clamp(min=0).pow(2)  # eigenvalues (squared singular values)
            p = p / p.sum().clamp(min=1e-12)
            eff_rank = (-p * p.clamp(min=1e-7).log()).sum().exp()

            # Timestep statistics
            T = ssl_mask_1.shape[1]
            n_vis_1 = ssl_mask_1.sum().item()
            n_vis_2 = ssl_mask_2.sum().item()
            n_masked_1 = (~ssl_mask_1).sum().item()
            n_masked_2 = (~ssl_mask_2).sum().item()

            metrics = {
                "contrastive_loss": loss.detach(),
                "ssl_loss": loss.detach(),
                "contrastive_accuracy": accuracy,
                "contrastive_pos_similarity": pos_sim,
                "contrastive_alignment": alignment,
                "contrastive_uniformity": uniformity,
                "contrastive_effective_rank": eff_rank,
                "contrastive_temperature": temperature,
                "contrastive_n_timesteps": T,
                "contrastive_n_visible_view1": n_vis_1 / B,
                "contrastive_n_visible_view2": n_vis_2 / B,
                "contrastive_n_masked_view1": n_masked_1 / B,
                "contrastive_n_masked_view2": n_masked_2 / B,
            }

        return loss, metrics
