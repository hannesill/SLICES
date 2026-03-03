"""Tensor caching and normalization stats I/O for ICU datasets.

Handles loading/saving of preprocessed tensor caches and normalization
statistics to avoid recomputation on subsequent runs.
Extracted from ICUDataset for modularity.
"""

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import yaml

logger = logging.getLogger(__name__)


def load_normalization_stats(
    data_dir: Path,
    current_train_indices: Optional[List[int]],
    normalize: bool,
) -> Optional[Dict[str, Any]]:
    """Load cached normalization statistics from file if they exist and match current split.

    Validates that cached statistics were computed on the same training set to prevent
    data leakage from validation/test sets.

    Args:
        data_dir: Path to data directory containing normalization_stats.yaml.
        current_train_indices: List of training indices for current split,
                             or None for unsupervised.
        normalize: Whether normalization is enabled.

    Returns:
        Dictionary with 'feature_means' and 'feature_stds' if file exists and split matches,
        None otherwise.
    """
    if not normalize:
        return None

    stats_path = data_dir / "normalization_stats.yaml"
    if not stats_path.exists():
        return None

    try:
        with open(stats_path) as f:
            stats = yaml.safe_load(f)

        # Validate that cached stats were computed on the same training split
        cached_train_indices = stats.get("train_indices")
        current_train_set = set(current_train_indices) if current_train_indices else None
        cached_train_set = set(cached_train_indices) if cached_train_indices else None

        if current_train_set != cached_train_set:
            warnings.warn(
                f"Cached normalization stats were computed on a different training split. "
                f"Cached: {len(cached_train_set) if cached_train_set else 0} samples, "
                f"Current: {len(current_train_set) if current_train_set else 0} samples. "
                "Recomputing statistics to prevent data leakage.",
                UserWarning,
            )
            return None

        return stats
    except Exception as e:
        warnings.warn(
            f"Failed to load normalization stats from {stats_path}: {e}. "
            "Will recompute statistics.",
            UserWarning,
        )
        return None


def save_normalization_stats(
    data_dir: Path,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    feature_names: List[str],
    train_indices: Optional[List[int]],
    normalize: bool,
) -> None:
    """Save computed normalization statistics to file for reproducibility.

    Args:
        data_dir: Path to data directory.
        feature_means: Per-feature means tensor.
        feature_stds: Per-feature stds tensor.
        feature_names: List of feature names.
        train_indices: Optional list of training indices used to compute stats.
        normalize: Whether normalization is enabled.
    """
    stats_path = data_dir / "normalization_stats.yaml"

    stats = {
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "feature_names": feature_names,
        "train_indices": train_indices,
        "normalize": normalize,
    }

    try:
        with open(stats_path, "w") as f:
            yaml.dump(stats, f, default_flow_style=False)
    except Exception as e:
        warnings.warn(
            f"Failed to save normalization stats to {stats_path}: {e}",
            UserWarning,
        )


def get_tensor_cache_key(
    normalize: bool,
    seq_length: int,
    n_features: int,
    train_indices: Optional[List[int]],
    excluded_stay_ids: Optional[Set[int]],
) -> str:
    """Generate a hash key for tensor caching based on preprocessing parameters.

    The cache key includes:
    - normalize flag
    - seq_length
    - hash of train_indices (for normalization stats consistency)
    - hash of excluded_stay_ids (for filtering consistency)

    Args:
        normalize: Whether normalization is enabled.
        seq_length: Sequence length.
        n_features: Number of features.
        train_indices: Optional list of training indices.
        excluded_stay_ids: Optional set of excluded stay IDs.

    Returns:
        Hash string to use as cache identifier.
    """
    # Create a deterministic string representation of parameters
    params = {
        "normalize": normalize,
        "seq_length": seq_length,
        "n_features": n_features,
    }

    # Hash train_indices if provided (too large to store directly)
    if train_indices is not None:
        indices_str = ",".join(map(str, sorted(train_indices)))
        indices_hash = hashlib.md5(indices_str.encode()).hexdigest()[:8]
        params["train_indices_hash"] = indices_hash
    else:
        params["train_indices_hash"] = "none"

    # Hash excluded_stay_ids if provided (ensures cache invalidation when filtering changes)
    if excluded_stay_ids:
        excluded_str = ",".join(map(str, sorted(excluded_stay_ids)))
        excluded_hash = hashlib.md5(excluded_str.encode()).hexdigest()[:8]
        params["excluded_stays_hash"] = excluded_hash
    else:
        params["excluded_stays_hash"] = "none"

    # Create overall hash
    params_str = str(sorted(params.items()))
    return hashlib.md5(params_str.encode()).hexdigest()[:12]


def get_tensor_cache_path(
    data_dir: Path,
    normalize: bool,
    seq_length: int,
    n_features: int,
    train_indices: Optional[List[int]],
    excluded_stay_ids: Optional[Set[int]],
) -> Path:
    """Get the path to the tensor cache file.

    Args:
        data_dir: Path to data directory.
        normalize: Whether normalization is enabled.
        seq_length: Sequence length.
        n_features: Number of features.
        train_indices: Optional list of training indices.
        excluded_stay_ids: Optional set of excluded stay IDs.

    Returns:
        Path to the tensor cache file.
    """
    cache_key = get_tensor_cache_key(
        normalize, seq_length, n_features, train_indices, excluded_stay_ids
    )
    cache_dir = data_dir / ".tensor_cache"
    return cache_dir / f"tensors_{cache_key}.pt"


def _try_load_cache(
    cache_path: Path, n_features: int, seq_length: int, normalize: bool
) -> Optional[Dict[str, Any]]:
    """Try to load and validate a tensor cache file.

    Args:
        cache_path: Path to the cache file.
        n_features: Expected number of features.
        seq_length: Expected sequence length.
        normalize: Expected normalize flag.

    Returns:
        Dictionary with cached tensors if valid, None otherwise.
    """
    if not cache_path.exists():
        return None

    try:
        logger.debug(f"Loading cached tensors from {cache_path.name}")
        cached = torch.load(cache_path, weights_only=True, mmap=True)

        # Validate cache metadata
        if cached.get("n_features") != n_features:
            logger.debug("Cached tensors have different feature count, recomputing")
            return None
        if cached.get("seq_length") != seq_length:
            logger.debug("Cached tensors have different sequence length, recomputing")
            return None
        if cached.get("normalize") != normalize:
            logger.debug("Cached tensors have different normalize flag, recomputing")
            return None

        # Validate tensor shapes (support both old list and new stacked format)
        # NOTE: cannot use `x or y` here — `or` calls bool() on tensors, which
        # raises RuntimeError for multi-element tensors.
        timeseries = cached.get("timeseries_tensor")
        if timeseries is None:
            timeseries = cached.get("timeseries_tensors")
        masks = cached.get("mask_tensor")
        if masks is None:
            masks = cached.get("mask_tensors")
        if timeseries is None or masks is None:
            logger.debug("Cached tensors missing data, recomputing")
            return None

        # Convert old list format to stacked if needed
        if isinstance(timeseries, list):
            logger.debug("Converting cached tensors from list to stacked format")
            timeseries = torch.stack(timeseries)
            masks = torch.stack(masks)
            cached["timeseries_tensor"] = timeseries
            cached["mask_tensor"] = masks

        n_samples = timeseries.shape[0] if hasattr(timeseries, "shape") else len(timeseries)
        logger.info(f"Loaded {n_samples:,} cached samples")
        return cached

    except Exception as e:
        logger.debug(f"Failed to load cached tensors: {e}, recomputing")
        return None


def load_cached_tensors(
    data_dir: Path,
    normalize: bool,
    seq_length: int,
    n_features: int,
    train_indices: Optional[List[int]],
    excluded_stay_ids: Optional[Set[int]],
) -> Optional[Dict[str, Any]]:
    """Load cached preprocessed tensors if they exist and are valid.

    Args:
        data_dir: Path to data directory.
        normalize: Whether normalization is enabled.
        seq_length: Sequence length.
        n_features: Number of features.
        train_indices: Optional list of training indices.
        excluded_stay_ids: Optional set of excluded stay IDs.

    Returns:
        Dictionary with cached tensors and metadata if valid, None otherwise.
    """
    cache_path = get_tensor_cache_path(
        data_dir, normalize, seq_length, n_features, train_indices, excluded_stay_ids
    )
    return _try_load_cache(cache_path, n_features, seq_length, normalize)


def save_cached_tensors(
    data_dir: Path,
    timeseries_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    feature_means: torch.Tensor,
    feature_stds: torch.Tensor,
    normalize: bool,
    seq_length: int,
    n_features: int,
    train_indices: Optional[List[int]],
    excluded_stay_ids: Optional[Set[int]],
) -> None:
    """Save preprocessed tensors to cache file.

    Args:
        data_dir: Path to data directory.
        timeseries_tensor: Preprocessed timeseries tensor.
        mask_tensor: Observation mask tensor.
        feature_means: Per-feature means.
        feature_stds: Per-feature stds.
        normalize: Whether normalization is enabled.
        seq_length: Sequence length.
        n_features: Number of features.
        train_indices: Optional list of training indices.
        excluded_stay_ids: Optional set of excluded stay IDs.
    """
    cache_path = get_tensor_cache_path(
        data_dir, normalize, seq_length, n_features, train_indices, excluded_stay_ids
    )
    cache_dir = cache_path.parent
    cache_dir.mkdir(exist_ok=True)

    cache_data = {
        "timeseries_tensor": timeseries_tensor,
        "mask_tensor": mask_tensor,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "n_features": n_features,
        "seq_length": seq_length,
        "normalize": normalize,
    }

    try:
        logger.debug(f"Saving tensors to cache: {cache_path.name}")
        torch.save(cache_data, cache_path)
    except Exception as e:
        logger.warning(f"Failed to save tensor cache to {cache_path}: {e}")


def save_dataset_metadata(
    data_dir: Path,
    task_name: Optional[str],
    handle_missing_labels: str,
    removed_samples: List[tuple],
) -> None:
    """Save dataset metadata including removed samples for reproducibility.

    Creates or updates dataset_metadata.yaml with information about:
    - Number of samples used (original vs. final after filtering)
    - Removed samples and reasons for removal
    - Task name and label handling strategy

    Args:
        data_dir: Path to data directory.
        task_name: Name of the task.
        handle_missing_labels: Label handling strategy.
        removed_samples: List of (stay_id, reason) tuples.
    """
    if not removed_samples:
        return  # Nothing to save if no samples were removed

    metadata_path = data_dir / "dataset_metadata.yaml"

    metadata = {
        "task_name": task_name,
        "handle_missing_labels": handle_missing_labels,
        "removed_samples_count": len(removed_samples),
        "removed_samples": [
            {"stay_id": stay_id, "reason": reason} for stay_id, reason in removed_samples
        ],
    }

    try:
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
    except Exception as e:
        warnings.warn(
            f"Failed to save dataset metadata to {metadata_path}: {e}",
            UserWarning,
        )
