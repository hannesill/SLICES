"""Data augmentation and transformation utilities for SSL.

Implements various masking strategies for self-supervised learning on ICU time-series:
- Random masking (uniform random selection)
- Block masking (contiguous time blocks)
- Structured masking (entire timesteps or features)
"""

from typing import Dict, Literal, Optional, Tuple

import torch


MaskingStrategy = Literal["random", "block", "timestep", "feature"]


def create_random_mask(
    shape: Tuple[int, ...],
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create random binary mask.
    
    Args:
        shape: Shape of mask tensor.
        mask_ratio: Fraction of elements to mask (set to False).
        device: Device to create mask on.
        generator: Optional random generator for reproducibility.
        
    Returns:
        Boolean mask where False indicates masked positions.
    """
    mask = torch.rand(shape, device=device, generator=generator) > mask_ratio
    return mask


def create_block_mask(
    shape: Tuple[int, int, int],  # (B, T, D)
    mask_ratio: float,
    min_block_size: int = 3,
    max_block_size: int = 10,
    device: torch.device = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create block-wise time masking.
    
    Masks contiguous blocks of timesteps to encourage learning temporal structure.
    Block sizes are capped to not exceed the target mask ratio.
    
    Args:
        shape: Shape (B, T, D) of input.
        mask_ratio: Target fraction of timesteps to mask.
        min_block_size: Minimum block size.
        max_block_size: Maximum block size.
        device: Device to create mask on.
        generator: Optional random generator for reproducibility.
        
    Returns:
        Boolean mask where False indicates masked positions.
    """
    B, T, D = shape
    if device is None:
        device = torch.device("cpu")
    
    mask = torch.ones((B, T, D), dtype=torch.bool, device=device)
    
    for b in range(B):
        masked_count = 0
        target_masked = int(T * mask_ratio)
        
        while masked_count < target_masked:
            # Calculate remaining budget
            remaining = target_masked - masked_count
            
            # Cap block size to not exceed remaining budget
            actual_max = min(max_block_size, remaining)
            if actual_max < min_block_size:
                # Can't add another block without exceeding target
                break
            
            # Random block size within allowed range
            block_size = torch.randint(
                min_block_size, actual_max + 1, (1,), generator=generator
            ).item()
            # Random start position
            start = torch.randint(
                0, max(1, T - block_size + 1), (1,), generator=generator
            ).item()
            end = min(start + block_size, T)
            
            # Mask the block across all features
            mask[b, start:end, :] = False
            masked_count += (end - start)
    
    return mask


def create_timestep_mask(
    shape: Tuple[int, int, int],  # (B, T, D)
    mask_ratio: float,
    device: torch.device = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create timestep-wise masking (mask entire timesteps across all features).
    
    Args:
        shape: Shape (B, T, D) of input.
        mask_ratio: Fraction of timesteps to mask.
        device: Device to create mask on.
        generator: Optional random generator for reproducibility.
        
    Returns:
        Boolean mask where False indicates masked positions.
    """
    B, T, D = shape
    if device is None:
        device = torch.device("cpu")
    
    # Create mask for timesteps only
    timestep_mask = torch.rand(B, T, device=device, generator=generator) > mask_ratio
    # Broadcast to all features
    mask = timestep_mask.unsqueeze(-1).expand(B, T, D)  # (B, T, D)
    
    return mask


def create_feature_mask(
    shape: Tuple[int, int, int],  # (B, T, D)
    mask_ratio: float,
    device: torch.device = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create feature-wise masking (mask entire features across all timesteps).
    
    Args:
        shape: Shape (B, T, D) of input.
        mask_ratio: Fraction of features to mask.
        device: Device to create mask on.
        generator: Optional random generator for reproducibility.
        
    Returns:
        Boolean mask where False indicates masked positions.
    """
    B, T, D = shape
    if device is None:
        device = torch.device("cpu")
    
    # Create mask for features only
    feature_mask = torch.rand(B, D, device=device, generator=generator) > mask_ratio
    # Broadcast to all timesteps
    mask = feature_mask.unsqueeze(1).expand(B, T, D)  # (B, T, D)
    
    return mask


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """Apply binary mask to input tensor.
    
    Args:
        x: Input tensor of shape (B, T, D).
        mask: Boolean mask where False indicates positions to mask.
        mask_value: Value to use for masked positions.
        
    Returns:
        Masked tensor.
    """
    return torch.where(mask, x, torch.tensor(mask_value, device=x.device, dtype=x.dtype))


def create_ssl_mask(
    shape: Tuple[int, int, int],  # (B, T, D)
    mask_ratio: float,
    strategy: MaskingStrategy = "random",
    obs_mask: Optional[torch.Tensor] = None,
    min_block_size: int = 3,
    max_block_size: int = 10,
    device: torch.device = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Create SSL mask using specified strategy.
    
    Respects observation mask - only masks observed values.
    
    Args:
        shape: Shape (B, T, D) of input.
        mask_ratio: Fraction of elements to mask.
        strategy: Masking strategy to use.
        obs_mask: Optional observation mask (True = observed, False = missing).
                  If provided, only observed values can be masked.
        min_block_size: Min block size for block masking.
        max_block_size: Max block size for block masking.
        device: Device to create mask on.
        generator: Optional random generator for reproducibility.
        
    Returns:
        Boolean mask where False indicates positions to mask for SSL.
    """
    if device is None:
        device = torch.device("cpu")
    
    # Create base mask based on strategy
    if strategy == "random":
        ssl_mask = create_random_mask(shape, mask_ratio, device, generator)
    elif strategy == "block":
        ssl_mask = create_block_mask(
            shape, mask_ratio, min_block_size, max_block_size, device, generator
        )
    elif strategy == "timestep":
        ssl_mask = create_timestep_mask(shape, mask_ratio, device, generator)
    elif strategy == "feature":
        ssl_mask = create_feature_mask(shape, mask_ratio, device, generator)
    else:
        raise ValueError(
            f"Unknown masking strategy '{strategy}'. "
            f"Choose from: random, block, timestep, feature"
        )
    
    # Only mask observed values (respect observation mask)
    if obs_mask is not None:
        # obs_mask: True = observed, False = missing
        # ssl_mask: False = masked for SSL
        # Result: Can only mask where obs_mask is True
        ssl_mask = ssl_mask | (~obs_mask)  # Unmask missing values
    
    return ssl_mask


def apply_gaussian_noise(
    x: torch.Tensor,
    std: float = 0.1,
    obs_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add Gaussian noise for SSL augmentation.
    
    Only adds noise to observed values if obs_mask is provided.
    
    Args:
        x: Input tensor of shape (B, T, D).
        std: Standard deviation of noise.
        obs_mask: Optional observation mask (True = observed).
        
    Returns:
        Noisy tensor.
    """
    noise = torch.randn_like(x) * std
    
    if obs_mask is not None:
        # Only add noise to observed values
        noise = noise * obs_mask.float()
    
    return x + noise

