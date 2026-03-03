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


def create_timestep_mask(
    batch_size: int,
    n_timesteps: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Create random mask at timestep level.

    Args:
        batch_size: Batch size B.
        n_timesteps: Number of timesteps T.
        mask_ratio: Fraction of timesteps to mask.
        device: Device.

    Returns:
        ssl_mask: (B, T) bool mask, True = visible, False = masked.
    """
    rand_vals = torch.rand(batch_size, n_timesteps, device=device)
    ssl_mask = rand_vals >= mask_ratio  # True = visible

    # Ensure at least 1 visible timestep per sample
    n_visible = ssl_mask.sum(dim=1)  # (B,)
    needs_fix = n_visible == 0
    if needs_fix.any():
        for b in range(batch_size):
            if needs_fix[b]:
                ssl_mask[b, 0] = True

    return ssl_mask


def create_block_timestep_mask(
    batch_size: int,
    n_timesteps: int,
    mask_ratio: float,
    device: torch.device,
    n_blocks: int = 3,
) -> torch.Tensor:
    """Create contiguous block mask at timestep level.

    Masks n_blocks contiguous segments that together cover approximately
    mask_ratio of the sequence. Forces the model to predict from distant
    context rather than interpolate from adjacent visible timesteps.

    Strategy: divides the sequence into n_blocks equal zones, then places one
    randomly-sized masked block within each zone. Block sizes are drawn from a
    Dirichlet-like split of the total masked budget. Fully vectorized — no
    Python loops over batch elements.

    Args:
        batch_size: Batch size B.
        n_timesteps: Number of timesteps T.
        mask_ratio: Fraction of timesteps to mask.
        device: Device.
        n_blocks: Number of contiguous blocks to mask (default 3).

    Returns:
        ssl_mask: (B, T) bool mask, True = visible, False = masked.
    """
    n_masked_total = max(int(n_timesteps * mask_ratio), 1)
    n_masked_total = min(n_masked_total, n_timesteps - 1)

    # Split total masked budget into n_blocks random lengths per sample.
    # Use Dirichlet-like splitting: draw n_blocks uniform values, normalize,
    # then scale to sum to n_masked_total. Add 1 to each to ensure min length 1.
    raw = torch.rand(batch_size, n_blocks)  # CPU for speed
    raw = raw / raw.sum(dim=1, keepdim=True)  # normalize to sum=1
    # Reserve 1 per block, distribute the rest proportionally
    extra = n_masked_total - n_blocks
    if extra > 0:
        block_lengths = 1 + (raw * extra).int()
        # Fix rounding: adjust last block to hit exact total
        block_lengths[:, -1] = n_masked_total - block_lengths[:, :-1].sum(dim=1)
    else:
        # Edge case: fewer masked timesteps than blocks — give 1 to first blocks
        block_lengths = torch.zeros(batch_size, n_blocks, dtype=torch.int)
        block_lengths[:, :n_masked_total] = 1

    # Clamp to valid range
    block_lengths = block_lengths.clamp(min=0, max=n_timesteps)

    # Divide sequence into n_blocks equal zones. Each block is placed randomly
    # within its zone, avoiding cross-zone overlap by construction.
    zone_size = n_timesteps // n_blocks
    zone_starts = torch.arange(n_blocks) * zone_size  # (n_blocks,)

    # Random offset within each zone (vectorized over batch)
    # Max offset = zone_size - block_length (so block fits within zone)
    max_offsets = zone_size - block_lengths  # (B, n_blocks)
    max_offsets = max_offsets.clamp(min=0)
    offsets = (torch.rand(batch_size, n_blocks) * (max_offsets.float() + 1)).int()
    offsets = offsets.clamp(max=max_offsets)

    # Compute absolute start positions: zone_start + offset
    starts = zone_starts.unsqueeze(0) + offsets  # (B, n_blocks)

    # Build mask: create time indices and compare with block ranges
    # (B, n_blocks, T) — True where timestep t falls in block k
    t = torch.arange(n_timesteps).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    starts_3d = starts.unsqueeze(2)  # (B, n_blocks, 1)
    ends_3d = (starts + block_lengths).unsqueeze(2)  # (B, n_blocks, 1)
    in_block = (t >= starts_3d) & (t < ends_3d)  # (B, n_blocks, T)

    # Union across blocks -> masked positions
    masked = in_block.any(dim=1)  # (B, T)
    ssl_mask = (~masked).to(device=device)

    return ssl_mask


def extract_visible_timesteps(
    tokens: torch.Tensor,
    ssl_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract visible timestep tokens from full sequence.

    Args:
        tokens: (B, T, d_model)
        ssl_mask: (B, T) True = visible, False = masked.

    Returns:
        visible_tokens: (B, max_vis, d_model)
        vis_padding: (B, max_vis) True = valid visible token
    """
    B, T, d_model = tokens.shape

    n_visible = ssl_mask.sum(dim=1)  # (B,)
    max_vis = max(int(n_visible.max().item()), 1)

    # Argsort: visible (True=1) first
    sort_idx = ssl_mask.float().argsort(dim=1, descending=True, stable=True)
    sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, d_model)

    sorted_tokens = tokens.gather(1, sort_idx_expanded)
    visible_tokens = sorted_tokens[:, :max_vis, :]  # (B, max_vis, d_model)

    vis_positions = torch.arange(max_vis, device=tokens.device).unsqueeze(0)
    vis_padding = vis_positions < n_visible.unsqueeze(1)  # (B, max_vis)

    return visible_tokens, vis_padding
