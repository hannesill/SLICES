"""Tensor preprocessing functions for ICU time-series data.

Pure functions for converting raw nested-list data to padded tensors,
computing normalization statistics, and applying normalization + imputation.
Extracted from ICUDataset._precompute_tensors for modularity.
"""

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    import polars as pl
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Constants
MIN_STD_THRESHOLD = 1e-6  # Minimum standard deviation to avoid division by zero


def extract_tensors_from_dataframe(
    timeseries_df: "pl.DataFrame",
    seq_length: int,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract tensors directly from a Polars DataFrame using zero-copy operations.

    Uses Polars explode() to flatten nested List(List(...)) columns into flat
    arrays, then reshapes to 3D tensors. This avoids to_list() which creates
    Python objects (~28 bytes per float) and can use 3-4x more memory than the
    actual data, causing OOM on large datasets.

    Requires uniform sequence lengths (all rows must have exactly seq_length
    timesteps with n_features features each). Falls back to convert_raw_to_tensors
    if sequences are non-uniform.

    Args:
        timeseries_df: Polars DataFrame with 'timeseries' and 'mask' columns,
            where each is List(List(Float64|Boolean)).
        seq_length: Expected sequence length per sample.
        n_features: Expected number of features per timestep.

    Returns:
        Tuple of (timeseries_tensor, masks_tensor), each shaped
        (n_samples, seq_length, n_features).
    """
    import polars as pl

    n_samples = len(timeseries_df)
    logger.debug("[1/3] Extracting tensors from DataFrame...")

    # Verify uniform sequence lengths for reshape safety
    ts_lens = timeseries_df["timeseries"].list.len()
    if ts_lens.min() != ts_lens.max():
        logger.info("Non-uniform sequence lengths detected, falling back to list conversion")
        raw_timeseries = timeseries_df["timeseries"].to_list()
        raw_masks = timeseries_df["mask"].to_list()
        return convert_raw_to_tensors(raw_timeseries, raw_masks, seq_length, n_features)

    # Check if stored sequence length differs from requested — fall back to
    # convert_raw_to_tensors which handles truncation/padding correctly
    stored_seq_length = int(ts_lens.min())  # type: ignore[arg-type]
    if stored_seq_length != seq_length:
        logger.info(
            "Stored sequence length (%d) differs from requested (%d),"
            " falling back to list conversion for truncation/padding",
            stored_seq_length,
            seq_length,
        )
        raw_timeseries = timeseries_df["timeseries"].to_list()
        raw_masks = timeseries_df["mask"].to_list()
        return convert_raw_to_tensors(raw_timeseries, raw_masks, seq_length, n_features)

    # Fast path: explode nested lists to flat array, then reshape.
    # This stays in Arrow/numpy memory without creating Python objects.
    # Extract columns as Series first, then process sequentially to limit
    # peak memory (avoids holding both float64 intermediate + final tensor).
    ts_series = timeseries_df["timeseries"]
    mask_series = timeseries_df["mask"]

    # Cast to Float32 in Polars to avoid a float64 intermediate array (~5 GB saved)
    timeseries_np = (
        ts_series.explode()
        .explode()
        .cast(pl.Float32)
        .to_numpy(writable=True)
        .reshape(n_samples, seq_length, n_features)
    )
    del ts_series
    timeseries_tensor = torch.from_numpy(timeseries_np)

    masks_np = mask_series.explode().explode().to_numpy().reshape(n_samples, seq_length, n_features)
    del mask_series
    masks_tensor = torch.from_numpy(masks_np.copy())  # copy needed: bool dtype
    del masks_np

    return timeseries_tensor, masks_tensor


def convert_raw_to_tensors(
    raw_timeseries: List[List[List[float]]],
    raw_masks: List[List[List[bool]]],
    seq_length: int,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert nested lists to padded tensors (Step 1 of preprocessing).

    Converts all samples to numpy arrays in batches for memory efficiency,
    then stacks into single tensors. Samples shorter than seq_length are
    padded with NaN/False; longer samples are truncated.

    Note: For large datasets, prefer extract_tensors_from_dataframe() which
    avoids the expensive to_list() conversion.

    Args:
        raw_timeseries: List of timeseries arrays (n_samples x seq_len x n_features).
        raw_masks: List of mask arrays (n_samples x seq_len x n_features).
        seq_length: Target sequence length (pad/truncate to this).
        n_features: Number of features per timestep.

    Returns:
        Tuple of (timeseries_tensor, masks_tensor), each shaped
        (n_samples, seq_length, n_features).
    """
    n_samples = len(raw_timeseries)
    logger.debug("[1/3] Converting to tensors...")

    # Pre-allocate output arrays to avoid 2x peak memory from list + np.stack
    timeseries_np = np.full((n_samples, seq_length, n_features), np.nan, dtype=np.float32)
    masks_np = np.zeros((n_samples, seq_length, n_features), dtype=bool)

    sample_iter = range(n_samples)
    if n_samples >= 50000:  # Only show progress for large datasets
        sample_iter = tqdm(sample_iter, desc="  Converting samples", unit="sample")

    for i in sample_iter:
        ts_data = raw_timeseries[i]
        mask_data = raw_masks[i]
        actual_len = min(len(ts_data), seq_length)

        timeseries_np[i, :actual_len] = np.array(ts_data[:actual_len], dtype=np.float32)
        masks_np[i, :actual_len] = np.array(mask_data[:actual_len], dtype=bool)

    # Convert to tensors (torch.from_numpy shares memory, no copy)
    timeseries_tensor = torch.from_numpy(timeseries_np)  # (n_samples, seq_len, n_features)
    masks_tensor = torch.from_numpy(masks_np)  # (n_samples, seq_len, n_features)

    return timeseries_tensor, masks_tensor


def compute_normalization_stats(
    timeseries_tensor: torch.Tensor,
    masks_tensor: torch.Tensor,
    train_indices: List[int],
    n_features: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std from training samples only (Step 2).

    CRITICAL: Uses only training samples to prevent data leakage from
    validation/test sets.

    Args:
        timeseries_tensor: Full timeseries tensor (n_samples, seq_len, n_features).
        masks_tensor: Full mask tensor (n_samples, seq_len, n_features).
        train_indices: Indices of training samples.
        n_features: Number of features.
        normalize: Whether normalization is enabled (affects std computation).

    Returns:
        Tuple of (feature_means, feature_stds), each shaped (n_features,).
    """
    logger.debug("[2/3] Computing normalization statistics...")

    feature_means = torch.zeros(n_features)
    feature_stds = torch.ones(n_features)

    # CRITICAL: Use only training samples to prevent data leakage
    train_ts = timeseries_tensor[train_indices]  # (n_train, seq_len, n_features)
    train_masks = masks_tensor[train_indices]

    # Vectorized computation of mean and std per feature
    # Reshape to (n_samples * seq_len, n_features) for easier computation
    flat_ts = train_ts.reshape(-1, n_features)  # (n_train * seq_len, n_features)
    flat_masks = train_masks.reshape(-1, n_features)

    # Create combined mask: observed AND not NaN
    valid_mask = flat_masks & ~torch.isnan(flat_ts)

    # Vectorized mean computation using masked tensor operations
    # Replace invalid values with 0 for sum, count valid entries
    masked_ts = torch.where(valid_mask, flat_ts, torch.zeros_like(flat_ts))
    valid_counts = valid_mask.sum(dim=0).float()  # (n_features,)
    feature_sums = masked_ts.sum(dim=0)  # (n_features,)

    # Compute means (avoid div by zero)
    feature_means = torch.where(
        valid_counts > 0,
        feature_sums / valid_counts,
        torch.zeros(n_features),
    )

    if normalize:
        # Vectorized std computation
        # Compute squared deviations from mean
        deviations = torch.where(
            valid_mask,
            (flat_ts - feature_means.unsqueeze(0)) ** 2,
            torch.zeros_like(flat_ts),
        )
        variance = torch.where(
            valid_counts > 1,
            deviations.sum(dim=0) / (valid_counts - 1),  # Bessel's correction
            torch.ones(n_features),
        )
        feature_stds = torch.sqrt(variance)

        # Clamp minimum std to avoid division by zero
        feature_stds = torch.clamp(feature_stds, min=MIN_STD_THRESHOLD)

    logger.debug(f"Computed normalization stats for {n_features} features")
    return feature_means, feature_stds


def apply_normalization_and_imputation(
    timeseries_tensor: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    normalize: bool,
    n_features: int,
) -> torch.Tensor:
    """Normalize then impute missing values (Step 3 of preprocessing).

    When normalize=True: z-score normalize, then zero-fill (0 = population mean).
    When normalize=False: impute with feature means (avoids physiologically
    impossible values like 0 heart rate).

    Args:
        timeseries_tensor: Tensor to normalize (n_samples, seq_len, n_features).
            Modified in-place and returned.
        feature_means: Per-feature means (n_features,).
        feature_stds: Per-feature stds (n_features,).
        normalize: Whether to apply z-score normalization.
        n_features: Number of features.

    Returns:
        The normalized/imputed timeseries tensor.
    """
    logger.debug("[3/3] Applying normalization and imputation...")

    if normalize:
        # Normalize first (NaN positions stay NaN through arithmetic)
        timeseries_tensor = (timeseries_tensor - feature_means) / feature_stds
        # After z-score normalization, 0 = feature mean in original space.
        # This is the most neutral default: "no information, assume population average."
        timeseries_tensor = torch.nan_to_num(timeseries_tensor, nan=0.0)
    else:
        # Without normalization, impute with feature means (not 0).
        # Zero-filling in original space creates physiologically impossible values
        # (e.g., 0 heart rate, 0 blood pressure). Feature means are a better default.
        for f in range(n_features):
            nan_mask = torch.isnan(timeseries_tensor[:, :, f])
            timeseries_tensor[:, :, f][nan_mask] = feature_means[f]

    return timeseries_tensor


def compute_single_sample_stages(
    raw_ts: List[List[float]],
    raw_mask: List[List[bool]],
    seq_length: int,
    n_features: int,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    normalize: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Re-run the preprocessing pipeline for a single sample, capturing each stage.

    Useful for debugging to see exactly what transformations are applied.

    The stages are:
        - grid: Raw 2D tensor (seq_length, n_features) with NaN for missing
        - normalized: After z-score normalization + zero-fill (what model sees)

    Args:
        raw_ts: Raw timeseries nested list for one sample.
        raw_mask: Raw mask nested list for one sample.
        seq_length: Target sequence length.
        n_features: Number of features.
        feature_means: Per-feature means from training set.
        feature_stds: Per-feature stds from training set.
        normalize: Whether normalization is enabled.

    Returns:
        Dict with keys 'grid', 'normalized', each containing:
            - 'timeseries': The tensor at that stage
            - 'mask': The observation mask (same for all stages)
    """
    # Stage 1: GRID - Convert to tensor with padding/truncation
    actual_len = len(raw_ts)
    if actual_len >= seq_length:
        grid_ts = np.array(raw_ts[:seq_length], dtype=np.float32)
        mask_arr = np.array(raw_mask[:seq_length], dtype=bool)
    else:
        grid_ts = np.full((seq_length, n_features), np.nan, dtype=np.float32)
        mask_arr = np.zeros((seq_length, n_features), dtype=bool)
        grid_ts[:actual_len] = np.array(raw_ts, dtype=np.float32)
        mask_arr[:actual_len] = np.array(raw_mask, dtype=bool)

    grid_tensor = torch.from_numpy(grid_ts)
    mask_tensor = torch.from_numpy(mask_arr)

    # Stage 2: NORMALIZED - z-score normalize then impute
    if normalize:
        normalized_tensor = (grid_tensor - feature_means) / feature_stds
        normalized_tensor = torch.nan_to_num(normalized_tensor, nan=0.0)
    else:
        normalized_tensor = grid_tensor.clone()
        for f in range(n_features):
            nan_mask = torch.isnan(normalized_tensor[:, f])
            normalized_tensor[:, f][nan_mask] = feature_means[f]

    return {
        "grid": {
            "timeseries": grid_tensor,
            "mask": mask_tensor,
        },
        "normalized": {
            "timeseries": normalized_tensor,
            "mask": mask_tensor,
        },
    }
