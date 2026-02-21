"""Shared masking utilities for observation-level SSL objectives.

Provides random observation masking and visible token extraction used by
MAE, JEPA, and Contrastive objectives.
"""

from typing import Tuple

import torch


def create_observation_mask(
    padding_mask: torch.Tensor,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Create random mask on observation tokens.

    Args:
        padding_mask: (B, max_obs) True = valid token.
        mask_ratio: Fraction of valid tokens to mask.
        device: Device.

    Returns:
        ssl_mask: (B, max_obs) True = visible, False = masked.
                  Padding positions are always marked visible (excluded from loss).
    """
    B, N = padding_mask.shape

    rand_vals = torch.rand(B, N, device=device)

    # Only mask valid (non-padding) tokens
    ssl_mask = (rand_vals >= mask_ratio) | (~padding_mask)

    # Ensure at least 1 visible token per sample
    n_valid = padding_mask.sum(dim=1)  # (B,)
    n_visible = (ssl_mask & padding_mask).sum(dim=1)  # (B,)

    needs_fix = (n_visible == 0) & (n_valid > 0)
    if needs_fix.any():
        for b in range(B):
            if needs_fix[b]:
                first_valid = padding_mask[b].nonzero(as_tuple=True)[0][0]
                ssl_mask[b, first_valid] = True

    return ssl_mask


def extract_visible(
    tokens: torch.Tensor,
    ssl_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract visible tokens from full token sequence.

    Args:
        tokens: (B, max_obs, d_model)
        ssl_mask: (B, max_obs) True = visible
        padding_mask: (B, max_obs) True = valid

    Returns:
        visible_tokens: (B, max_vis, d_model)
        vis_padding: (B, max_vis) True = valid visible token
    """
    B, N, d_model = tokens.shape

    visible_mask = ssl_mask & padding_mask  # (B, N)

    n_visible = visible_mask.sum(dim=1)  # (B,)
    max_vis = max(int(n_visible.max().item()), 1)

    sort_idx = visible_mask.float().argsort(dim=1, descending=True, stable=True)
    sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, d_model)

    sorted_tokens = tokens.gather(1, sort_idx_expanded)
    visible_tokens = sorted_tokens[:, :max_vis, :]  # (B, max_vis, d_model)

    vis_positions = torch.arange(max_vis, device=tokens.device).unsqueeze(0)
    vis_padding = vis_positions < n_visible.unsqueeze(1)  # (B, max_vis)

    return visible_tokens, vis_padding
