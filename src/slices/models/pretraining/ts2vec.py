"""TS2Vec-style temporal contrastive SSL objective for ICU time-series.

Addresses the augmentation limitation of the standard contrastive objective
(masking-only) by combining timestamp masking with input-level Gaussian noise
and optional random cropping — augmentations natural to contrastive learning.

Key differences from the standard contrastive objective:
- **Input-level augmentation**: Gaussian noise applied to raw values before
  tokenization, so the obs-aware MLP sees genuinely different inputs per view
  (not just different attention contexts from masking).
- **Independent masks**: Two independent random masks (not complementary),
  producing ~25% temporal overlap at mask_ratio=0.5.
- **Temporal contrastive loss**: Per-timestep positive pairs (same timestep
  across two views). Within-sample timesteps are hard negatives, cross-sample
  timesteps are additional negatives.
- **Hierarchical pooling**: Max-pool over increasing temporal scales, computing
  the temporal loss at each scale. Captures multi-scale temporal structure.

Architecture:
1. Add independent Gaussian noise to create two input views
2. Tokenize each view independently via encoder.tokenize()
3. Apply independent random timestep masks to each view
4. Encode visible tokens for each view via encoder.encode()
5. Scatter back to full (B, T, d) grids
6. Project per-timestep through shared projection head
7. Compute hierarchical temporal contrastive loss on overlapping timesteps

Reference: Yue et al., "TS2Vec: Towards Universal Representation of Time
Series", AAAI 2022.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSSLObjective, SSLConfig
from .masking import create_timestep_mask, extract_visible_timesteps


@dataclass
class TS2VecConfig(SSLConfig):
    """Configuration for TS2Vec-style temporal contrastive objective."""

    name: str = "ts2vec"

    # Masking (same default as other objectives for comparability)
    mask_ratio: float = 0.5

    # Input-level augmentation
    noise_scale: float = 0.01  # Gaussian noise std (0 = masking only)
    crop_ratio: float = 1.0  # Sub-sequence crop ratio (1.0 = no crop)

    # Projection head
    proj_hidden_dim: int = 256
    proj_output_dim: int = 64

    # Temperature for temporal NT-Xent
    temperature: float = 0.05

    # Hierarchical temporal pooling
    n_hierarchical_scales: int = 4  # Number of max-pool scales (1 = no hierarchy)

    def __post_init__(self) -> None:
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {self.noise_scale}")
        if not 0 < self.crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in (0, 1], got {self.crop_ratio}")
        if self.n_hierarchical_scales < 1:
            raise ValueError(
                f"n_hierarchical_scales must be >= 1, got {self.n_hierarchical_scales}"
            )


class TemporalProjectionHead(nn.Module):
    """Per-timestep MLP projection head with L2 normalization."""

    def __init__(self, d_input: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize per-timestep representations.

        Args:
            x: (B, T, d_input) or (N, d_input)

        Returns:
            L2-normalized projections, same leading dims.
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)


class TS2VecObjective(BaseSSLObjective):
    """TS2Vec-style temporal contrastive SSL for ICU time-series.

    Uses input-level augmentation (noise + masking) and hierarchical temporal
    contrastive loss. See module docstring for details.
    """

    def __init__(self, encoder: nn.Module, config: TS2VecConfig) -> None:
        super().__init__(encoder, config)
        self.config: TS2VecConfig = config

        # Validate encoder
        if not getattr(getattr(encoder, "config", None), "obs_aware", False):
            raise ValueError(
                "TS2Vec requires an encoder with obs_aware=True. " f"Got: {type(encoder).__name__}"
            )
        encoder_pooling = getattr(encoder.config, "pooling", "none")
        if encoder_pooling != "none":
            raise ValueError(
                "TS2Vec requires encoder with pooling='none' for per-token "
                f"representations, but got pooling='{encoder_pooling}'"
            )

        d_encoder = encoder.get_output_dim()
        self.missing_token = None

        self.projection_head = TemporalProjectionHead(
            d_input=d_encoder,
            hidden_dim=config.proj_hidden_dim,
            output_dim=config.proj_output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hierarchical temporal contrastive loss.

        Args:
            x: Input tensor (B, T, D).
            obs_mask: Observation mask (B, T, D), True = observed.

        Returns:
            (loss, metrics_dict)
        """
        B, T, D = x.shape
        device = x.device

        # 1. Create two augmented input views
        x1, x2 = self._create_augmented_views(x, obs_mask)

        # 2. Optional random cropping to shared sub-interval
        if self.config.crop_ratio < 1.0:
            crop_len = max(2, int(T * self.config.crop_ratio))
            start = torch.randint(0, T - crop_len + 1, (1,)).item()
            x1 = x1[:, start : start + crop_len]
            x2 = x2[:, start : start + crop_len]
            obs_mask = obs_mask[:, start : start + crop_len]
            T = crop_len

        # 3. Tokenize each view independently (noise means different MLP inputs)
        tokens_1, _, token_info_1 = self.encoder.tokenize(x1, obs_mask)
        tokens_2, _, _ = self.encoder.tokenize(x2, obs_mask)
        valid_timestep_mask = token_info_1["valid_timestep_mask"]

        # 4. Independent timestep masks
        ssl_mask_1 = create_timestep_mask(
            B,
            T,
            self.config.mask_ratio,
            device,
            valid_timestep_mask=valid_timestep_mask,
        )
        ssl_mask_2 = create_timestep_mask(
            B,
            T,
            self.config.mask_ratio,
            device,
            valid_timestep_mask=valid_timestep_mask,
        )
        effective_mask_1 = ssl_mask_1 & valid_timestep_mask
        effective_mask_2 = ssl_mask_2 & valid_timestep_mask

        # 5. Encode visible tokens for each view
        vis_1, vp_1 = extract_visible_timesteps(
            tokens_1,
            ssl_mask_1,
            valid_timestep_mask=valid_timestep_mask,
        )
        enc_1 = self.encoder.encode(vis_1, vp_1)

        vis_2, vp_2 = extract_visible_timesteps(
            tokens_2,
            ssl_mask_2,
            valid_timestep_mask=valid_timestep_mask,
        )
        enc_2 = self.encoder.encode(vis_2, vp_2)

        # 6. Scatter back to full (B, T, d) grids
        full_1 = self._scatter_to_full(enc_1, effective_mask_1, T)
        full_2 = self._scatter_to_full(enc_2, effective_mask_2, T)

        # 7. Project per-timestep
        z1 = self.projection_head(full_1)  # (B, T, proj_dim)
        z2 = self.projection_head(full_2)  # (B, T, proj_dim)

        # 8. Find overlap and valid masks
        overlap = effective_mask_1 & effective_mask_2  # (B, T)

        # 9. Hierarchical temporal contrastive loss
        #    Pass per-view masks so max-pool can ignore masked positions
        loss, metrics = self._hierarchical_temporal_loss(
            z1,
            z2,
            overlap,
            effective_mask_1,
            effective_mask_2,
        )

        # Add masking statistics
        with torch.no_grad():
            metrics.update(
                {
                    "ts2vec_n_timesteps": torch.tensor(T),
                    "ts2vec_n_visible_view1": torch.tensor(effective_mask_1.sum().item() / B),
                    "ts2vec_n_visible_view2": torch.tensor(effective_mask_2.sum().item() / B),
                    "ts2vec_n_overlap_per_sample": torch.tensor(overlap.sum().item() / B),
                    "ts2vec_overlap_ratio": overlap.sum().float()
                    / valid_timestep_mask.sum().clamp(min=1).float(),
                }
            )

        return loss, metrics

    def _create_augmented_views(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create two input views with independent Gaussian noise.

        Noise is applied only to observed values (via obs_mask) to avoid
        injecting signal into missing positions.

        Args:
            x: (B, T, D)
            obs_mask: (B, T, D) True = observed

        Returns:
            (x1, x2) each (B, T, D)
        """
        if self.config.noise_scale > 0 and self.training:
            noise_mask = obs_mask.float()
            x1 = x + self.config.noise_scale * torch.randn_like(x) * noise_mask
            x2 = x + self.config.noise_scale * torch.randn_like(x) * noise_mask
        else:
            x1 = x
            x2 = x
        return x1, x2

    @staticmethod
    def _scatter_to_full(
        encoded: torch.Tensor,
        ssl_mask: torch.Tensor,
        n_timesteps: int,
    ) -> torch.Tensor:
        """Scatter visible encoded tokens back to full (B, T, d) tensor.

        Args:
            encoded: (B, n_vis, d_enc)
            ssl_mask: (B, T) True = visible
            n_timesteps: Total T

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

    @staticmethod
    def _masked_max_pool1d(
        z: torch.Tensor,
        valid_mask: torch.Tensor,
        pool_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Max-pool along temporal dim, ignoring masked (invalid) positions.

        Fills invalid positions with -inf before pooling so they are never
        selected. A pooled position is valid if ANY timestep in its window
        was valid.

        Args:
            z: (B, T, d) representations
            valid_mask: (B, T) True = valid position
            pool_size: Pooling kernel size

        Returns:
            (z_pooled, valid_pooled) where z_pooled is (B, T', d) and
            valid_pooled is (B, T') bool.
        """
        B, T, d = z.shape

        # Truncate to multiple of pool_size
        T_trunc = (T // pool_size) * pool_size
        z_t = z[:, :T_trunc].permute(0, 2, 1)  # (B, d, T_trunc)
        mask_trunc = valid_mask[:, :T_trunc]  # (B, T_trunc)

        # Fill invalid positions with -inf so max-pool ignores them
        inv_mask = ~mask_trunc.unsqueeze(1).expand_as(z_t)  # (B, d, T_trunc)
        z_t = z_t.masked_fill(inv_mask, float("-inf"))

        z_pooled = F.max_pool1d(z_t, kernel_size=pool_size).permute(0, 2, 1)

        # A pooled position is valid if any input position was valid
        valid_pooled = (
            F.max_pool1d(mask_trunc.float().unsqueeze(1), kernel_size=pool_size).squeeze(1).bool()
        )

        # Replace any remaining -inf (all-masked windows) with 0
        z_pooled = z_pooled.masked_fill(z_pooled == float("-inf"), 0.0)

        return z_pooled, valid_pooled

    def _hierarchical_temporal_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        overlap: torch.Tensor,
        mask_1: torch.Tensor,
        mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute temporal contrastive loss at multiple temporal scales.

        At each scale, max-pool over non-overlapping windows (ignoring masked
        positions via -inf fill), then compute the temporal contrastive loss
        on the pooled representations.

        Args:
            z1: (B, T, proj_dim) projected representations from view 1
            z2: (B, T, proj_dim) projected representations from view 2
            overlap: (B, T) True = timestep visible in both views
            mask_1: (B, T) True = visible in view 1
            mask_2: (B, T) True = visible in view 2

        Returns:
            (loss, metrics_dict)
        """
        B, T, proj_dim = z1.shape
        device = z1.device

        # Compute pool sizes: [1, 2, 4, ...] up to n_hierarchical_scales
        pool_sizes = [2**i for i in range(self.config.n_hierarchical_scales)]
        # Filter out scales larger than T
        pool_sizes = [p for p in pool_sizes if p <= T]
        if not pool_sizes:
            pool_sizes = [1]

        total_loss = torch.tensor(0.0, device=device)
        n_scales = 0
        total_overlap_tokens = 0

        for pool_size in pool_sizes:
            if pool_size == 1:
                z1_pooled = z1
                z2_pooled = z2
                overlap_pooled = overlap
            else:
                z1_pooled, valid_1 = self._masked_max_pool1d(z1, mask_1, pool_size)
                z2_pooled, valid_2 = self._masked_max_pool1d(z2, mask_2, pool_size)
                overlap_pooled = valid_1 & valid_2

            # Compute temporal contrastive loss at this scale
            scale_loss, n_tokens = self._temporal_contrastive_loss(
                z1_pooled, z2_pooled, overlap_pooled
            )

            if n_tokens > 0:
                total_loss = total_loss + scale_loss
                n_scales += 1
                total_overlap_tokens += n_tokens

        if n_scales > 0:
            total_loss = total_loss / n_scales
        else:
            # No overlap at any scale — zero loss with grad connectivity
            total_loss = z1.sum() * 0.0

        with torch.no_grad():
            # Collapse monitoring (Wang & Isola 2020) — computed at scale 1
            # using overlap tokens only for consistency
            collapse_metrics = self._compute_collapse_metrics(z1, z2, overlap)

            metrics = {
                "ts2vec_loss": total_loss.detach(),
                "ssl_loss": total_loss.detach(),
                "ts2vec_n_scales_active": torch.tensor(n_scales),
                "ts2vec_total_overlap_tokens": torch.tensor(total_overlap_tokens),
            }
            metrics.update(collapse_metrics)

        return total_loss, metrics

    def _temporal_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        overlap: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Compute temporal contrastive loss at a single scale.

        For each overlapping timestep, the two views' representations form a
        positive pair. All other overlap tokens across the batch (both temporal
        neighbors within the same sample and tokens from other samples) serve
        as negatives in a standard NT-Xent formulation.

        Args:
            z1: (B, T', proj_dim) projected representations from view 1
            z2: (B, T', proj_dim) projected representations from view 2
            overlap: (B, T') True = valid overlap position

        Returns:
            (loss, n_overlap_tokens)
        """
        temperature = self.config.temperature

        N = int(overlap.sum().item())
        if N < 2:
            return torch.tensor(0.0, device=z1.device), 0

        # Gather overlap tokens from both views
        tokens_1 = z1[overlap]  # (N, proj_dim)
        tokens_2 = z2[overlap]  # (N, proj_dim)

        # Re-normalize after potential max-pool
        tokens_1 = F.normalize(tokens_1, dim=-1)
        tokens_2 = F.normalize(tokens_2, dim=-1)

        # Standard NT-Xent on (2N, 2N) — combines temporal and cross-sample
        z = torch.cat([tokens_1, tokens_2], dim=0)  # (2N, proj_dim)
        sim_matrix = torch.mm(z, z.t()) / temperature  # (2N, 2N)

        # Positive pair labels: token i pairs with i+N
        labels = torch.cat(
            [
                torch.arange(N, 2 * N, device=z.device),
                torch.arange(N, device=z.device),
            ]
        )

        # Mask self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        loss = F.cross_entropy(sim_matrix, labels)

        return loss, N

    @staticmethod
    def _compute_collapse_metrics(
        z1: torch.Tensor,
        z2: torch.Tensor,
        overlap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute collapse monitoring metrics on scale-1 overlap tokens.

        Mirrors the standard contrastive objective's collapse monitoring
        (Wang & Isola 2020, Roy & Vetterli 2007) for consistent analysis.

        Args:
            z1: (B, T, proj_dim) from view 1
            z2: (B, T, proj_dim) from view 2
            overlap: (B, T) True = overlap position

        Returns:
            Dict of collapse metrics.
        """
        N = int(overlap.sum().item())
        if N < 2:
            return {
                "ts2vec_alignment": torch.tensor(0.0),
                "ts2vec_uniformity": torch.tensor(0.0),
                "ts2vec_effective_rank": torch.tensor(0.0),
            }

        t1 = F.normalize(z1[overlap], dim=-1)  # (N, proj_dim)
        t2 = F.normalize(z2[overlap], dim=-1)

        # Alignment: mean squared L2 distance of positive pairs (lower = better)
        alignment = (t1 - t2).norm(dim=-1).pow(2).mean()

        # Uniformity: log avg Gaussian potential on hypersphere (lower = better)
        # Use view 1 tokens to avoid inflating with positives
        sq_pdist = torch.cdist(t1, t1, p=2).pow(2)
        uniformity = sq_pdist.mul(-2).exp().mean().log()

        # Effective rank via singular value entropy
        centered = t1 - t1.mean(dim=0)
        _, s, _ = torch.svd_lowrank(centered, q=min(N, t1.shape[1]))
        p = s.clamp(min=0).pow(2)
        p = p / p.sum().clamp(min=1e-12)
        eff_rank = (-p * p.clamp(min=1e-7).log()).sum().exp()

        return {
            "ts2vec_alignment": alignment,
            "ts2vec_uniformity": uniformity,
            "ts2vec_effective_rank": eff_rank,
        }
