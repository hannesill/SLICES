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
    valid_timestep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create random mask at timestep level.

    Args:
        batch_size: Batch size B.
        n_timesteps: Number of timesteps T.
        mask_ratio: Fraction of timesteps to mask.
        device: Device.
        valid_timestep_mask: Optional (B, T) bool mask marking timesteps with
            at least one observed variable. When provided, fully unobserved
            timesteps are always treated as visible/excluded from SSL masking.

    Returns:
        ssl_mask: (B, T) bool mask, True = visible, False = masked.
    """
    if valid_timestep_mask is None:
        valid_timestep_mask = torch.ones(batch_size, n_timesteps, dtype=torch.bool, device=device)
    else:
        valid_timestep_mask = valid_timestep_mask.to(device=device, dtype=torch.bool)

    ssl_mask = torch.ones(batch_size, n_timesteps, dtype=torch.bool, device=device)
    for b in range(batch_size):
        valid_idx = valid_timestep_mask[b].nonzero(as_tuple=True)[0]
        n_valid = int(valid_idx.numel())
        n_masked = _masked_timestep_budget(n_valid, mask_ratio)
        if n_masked == 0:
            continue

        masked_ordinals = torch.randperm(n_valid, device=device)[:n_masked]
        ssl_mask[b, valid_idx[masked_ordinals]] = False

    return ssl_mask


def create_complementary_timestep_masks(
    primary_mask: torch.Tensor,
    valid_timestep_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create paired timestep masks without allowing an empty eligible view.

    The secondary mask is complementary over eligible timesteps whenever
    possible. Sparse samples need a fallback:

    - 0 eligible timesteps: both masks remain effectively empty
    - 1 eligible timestep: expose it in both views
    - >=2 eligible timesteps: keep views complementary, but if the primary view
      happened to keep every eligible timestep visible, move one timestep into
      the secondary view so both remain encodable
    """
    primary = primary_mask.to(dtype=torch.bool).clone()
    valid = valid_timestep_mask.to(device=primary.device, dtype=torch.bool)
    secondary = (~primary) | (~valid)

    batch_size = primary.shape[0]
    for b in range(batch_size):
        valid_idx = valid[b].nonzero(as_tuple=True)[0]
        n_valid = int(valid_idx.numel())

        if n_valid == 0:
            secondary[b] = True
            continue

        if n_valid == 1:
            idx = valid_idx[0]
            primary[b, idx] = True
            secondary[b, idx] = True
            continue

        secondary_visible = (secondary[b] & valid[b]).nonzero(as_tuple=True)[0]
        if secondary_visible.numel() == 0:
            primary_visible = (primary[b] & valid[b]).nonzero(as_tuple=True)[0]
            idx = primary_visible[0]
            primary[b, idx] = False
            secondary[b, idx] = True

    return primary, secondary


def _masked_timestep_budget(n_valid: int, mask_ratio: float) -> int:
    """Return the integer masked-token budget while keeping one valid token visible."""
    if n_valid <= 1 or mask_ratio <= 0:
        return 0

    requested = int(n_valid * mask_ratio + 0.5)
    return min(max(requested, 1), n_valid - 1)


def _random_positive_composition(total: int, parts: int) -> torch.Tensor:
    """Split ``total`` into ``parts`` positive integer pieces."""
    if parts <= 1:
        return torch.tensor([total], dtype=torch.long)
    if total == parts:
        return torch.ones(parts, dtype=torch.long)

    cuts = torch.randperm(total - 1)[: parts - 1] + 1
    cuts = cuts.sort().values
    boundaries = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            cuts.to(dtype=torch.long),
            torch.tensor([total], dtype=torch.long),
        ]
    )
    return boundaries[1:] - boundaries[:-1]


def _random_nonnegative_composition(total: int, parts: int) -> torch.Tensor:
    """Split ``total`` into ``parts`` non-negative integer pieces."""
    if parts <= 0:
        return torch.empty(0, dtype=torch.long)
    if total <= 0:
        return torch.zeros(parts, dtype=torch.long)

    assignments = torch.randint(parts, (total,))
    return torch.bincount(assignments, minlength=parts).to(dtype=torch.long)


def create_block_timestep_mask(
    batch_size: int,
    n_timesteps: int,
    mask_ratio: float,
    device: torch.device,
    n_blocks: int = 3,
    valid_timestep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create contiguous block mask at timestep level.

    Masks up to ``n_blocks`` non-overlapping contiguous spans that hit the
    requested masked-token budget after taking the union over valid timesteps.
    Fully unobserved timesteps are never counted in the SSL budget and are
    marked visible so losses ignore them.

    Args:
        batch_size: Batch size B.
        n_timesteps: Number of timesteps T.
        mask_ratio: Fraction of timesteps to mask.
        device: Device.
        n_blocks: Number of contiguous blocks to mask (default 3).
        valid_timestep_mask: Optional (B, T) bool mask marking timesteps with
            at least one observed variable.

    Returns:
        ssl_mask: (B, T) bool mask, True = visible, False = masked.
    """
    if valid_timestep_mask is None:
        valid_timestep_mask = torch.ones(batch_size, n_timesteps, dtype=torch.bool, device=device)
    else:
        valid_timestep_mask = valid_timestep_mask.to(device=device, dtype=torch.bool)

    ssl_mask = torch.ones(batch_size, n_timesteps, dtype=torch.bool, device=device)
    requested_blocks = max(int(n_blocks), 1)

    for b in range(batch_size):
        valid_idx = valid_timestep_mask[b].nonzero(as_tuple=True)[0]
        n_valid = int(valid_idx.numel())
        n_masked = _masked_timestep_budget(n_valid, mask_ratio)
        if n_masked == 0:
            continue

        # Distinct blocks need at least one visible eligible timestep between
        # them. Reduce the block count when the requested budget leaves too few
        # visible timesteps to separate every block.
        max_separated_blocks = n_valid - n_masked + 1
        block_count = min(requested_blocks, n_masked, max_separated_blocks)

        lengths = _random_positive_composition(n_masked, block_count)
        total_gap = n_valid - n_masked
        extra_gap = total_gap - (block_count - 1)
        gap_extras = _random_nonnegative_composition(extra_gap, block_count + 1)
        gaps = gap_extras.clone()
        if block_count > 1:
            gaps[1:block_count] += 1

        ordinal_mask = torch.zeros(n_valid, dtype=torch.bool, device=device)
        cursor = int(gaps[0].item())
        for block_idx in range(block_count):
            length = int(lengths[block_idx].item())
            ordinal_mask[cursor : cursor + length] = True
            cursor += length + int(gaps[block_idx + 1].item())

        ssl_mask[b, valid_idx[ordinal_mask]] = False

    return ssl_mask


def extract_visible_timesteps(
    tokens: torch.Tensor,
    ssl_mask: torch.Tensor,
    valid_timestep_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract visible timestep tokens from full sequence.

    Args:
        tokens: (B, T, d_model)
        ssl_mask: (B, T) True = visible, False = masked.
        valid_timestep_mask: Optional (B, T) bool mask for timesteps that
            should participate in SSL tokenization.

    Returns:
        visible_tokens: (B, max_vis, d_model)
        vis_padding: (B, max_vis) True = valid visible token
    """
    B, T, d_model = tokens.shape

    if valid_timestep_mask is None:
        visible_mask = ssl_mask
    else:
        visible_mask = ssl_mask & valid_timestep_mask.to(device=tokens.device, dtype=torch.bool)

    n_visible = visible_mask.sum(dim=1)  # (B,)
    max_vis = max(int(n_visible.max().item()), 1)

    # Argsort: visible (True=1) first
    sort_idx = visible_mask.float().argsort(dim=1, descending=True, stable=True)
    sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, d_model)

    sorted_tokens = tokens.gather(1, sort_idx_expanded)
    visible_tokens = sorted_tokens[:, :max_vis, :]  # (B, max_vis, d_model)

    vis_positions = torch.arange(max_vis, device=tokens.device).unsqueeze(0)
    vis_padding = vis_positions < n_visible.unsqueeze(1)  # (B, max_vis)

    return visible_tokens, vis_padding


def scatter_visible_timesteps(
    visible_tokens: torch.Tensor,
    visible_mask: torch.Tensor,
    n_timesteps: int,
    fill_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Place variable-count visible tokens back on a full timestep grid.

    `extract_visible_timesteps` pads each batch row to the maximum visible-token
    count. This helper scatters only the real visible tokens for each sample, so
    padded tokens cannot overwrite masked timestep positions.

    Args:
        visible_tokens: (B, max_vis, d_model) tokens returned by an encoder.
        visible_mask: (B, T) True at the original visible timestep positions.
        n_timesteps: Total number of timesteps T.
        fill_value: Optional (1, 1, d_model) or broadcastable fill tensor. When
            omitted, masked positions are filled with zeros.

    Returns:
        (B, T, d_model) tensor with visible tokens restored to their original
        timestep positions.
    """
    B, max_vis, d_model = visible_tokens.shape
    device = visible_tokens.device
    visible_mask = visible_mask.to(device=device, dtype=torch.bool)

    if visible_mask.shape != (B, n_timesteps):
        raise ValueError(
            f"visible_mask must have shape ({B}, {n_timesteps}), "
            f"got {tuple(visible_mask.shape)}"
        )

    if fill_value is None:
        full = visible_tokens.new_zeros((B, n_timesteps, d_model))
    else:
        fill = fill_value.to(device=device, dtype=visible_tokens.dtype)
        full = fill.expand(B, n_timesteps, d_model).clone()

    visible_indices = visible_mask.float().argsort(dim=1, descending=True, stable=True)
    visible_counts = visible_mask.sum(dim=1).clamp(max=max_vis)

    for b in range(B):
        n_visible = int(visible_counts[b].item())
        if n_visible == 0:
            continue
        full[b, visible_indices[b, :n_visible]] = visible_tokens[b, :n_visible].to(full.dtype)

    return full
