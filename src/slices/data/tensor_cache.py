"""Tensor caching and normalization stats I/O for ICU datasets.

Handles loading/saving of preprocessed tensor caches and normalization
statistics to avoid recomputation on subsequent runs.
Extracted from ICUDataset for modularity.
"""

import hashlib
import inspect
import json
import logging
import os
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)
_CACHE_FINGERPRINT_VERSION = "2026-04-07"


def _fingerprint_payload(payload: Dict[str, Any]) -> str:
    """Return a short stable fingerprint for a JSON-serializable payload."""
    content = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _path_signature(path: Path) -> Dict[str, Any]:
    """Return a lightweight fingerprint for a file or directory path."""
    if not path.exists():
        return {"exists": False}

    stat = path.stat()
    signature: Dict[str, Any] = {
        "exists": True,
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

    if path.is_dir():
        signature["children"] = sorted(child.name for child in path.iterdir())

    return signature


def get_data_fingerprint(data_dir: Path) -> str:
    """Fingerprint the processed dataset contents used to build caches."""
    payload = {
        "version": _CACHE_FINGERPRINT_VERSION,
        "metadata": _path_signature(data_dir / "metadata.yaml"),
        "static": _path_signature(data_dir / "static.parquet"),
        "timeseries": _path_signature(data_dir / "timeseries.parquet"),
        "labels": _path_signature(data_dir / "labels.parquet"),
    }
    return _fingerprint_payload(payload)


@lru_cache(maxsize=1)
def get_preprocessing_fingerprint() -> str:
    """Fingerprint the tensor-preprocessing code path that builds caches."""
    from slices.data import tensor_preprocessing as tp

    payload = {"version": _CACHE_FINGERPRINT_VERSION}
    for name in (
        "extract_tensors_from_dataframe",
        "convert_raw_to_tensors",
        "compute_normalization_stats",
        "apply_normalization_and_imputation",
    ):
        payload[name] = inspect.getsource(getattr(tp, name))
    return _fingerprint_payload(payload)


def _validate_cache_fingerprints(
    cached: Dict[str, Any],
    *,
    artifact_name: str,
    expected_data_fingerprint: str,
    expected_preprocessing_fingerprint: str,
) -> bool:
    """Return True when cached metadata matches current data and code fingerprints."""
    cached_data_fingerprint = cached.get("data_fingerprint")
    cached_preprocessing_fingerprint = cached.get("preprocessing_fingerprint")

    if not cached_data_fingerprint or not cached_preprocessing_fingerprint:
        warnings.warn(
            f"{artifact_name} cache is missing freshness fingerprints. "
            "Ignoring cache and recomputing.",
            UserWarning,
        )
        return False

    if cached_data_fingerprint != expected_data_fingerprint:
        logger.debug("%s data fingerprint mismatch, recomputing", artifact_name)
        return False

    if cached_preprocessing_fingerprint != expected_preprocessing_fingerprint:
        logger.debug("%s preprocessing fingerprint mismatch, recomputing", artifact_name)
        return False

    return True


def _compute_split_hash(train_indices: List[int], normalize: bool) -> str:
    """Compute a stable hash from sorted train indices and normalize flag.

    Used to key normalization stats files so concurrent runs with different
    splits write to different files instead of overwriting each other.
    """
    content = f"{sorted(train_indices)}|{normalize}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _normalization_stats_path(data_dir: Path, split_hash: str) -> Path:
    """Return the hash-keyed normalization stats file path."""
    return data_dir / f"normalization_stats_{split_hash}.yaml"


def load_normalization_stats(
    data_dir: Path,
    current_train_indices: Optional[List[int]],
    normalize: bool,
) -> Optional[Dict[str, Any]]:
    """Load cached normalization statistics from file if they exist and match current split.

    Looks for a hash-keyed file first (normalization_stats_<hash>.yaml), then
    falls back to the legacy normalization_stats.yaml with set-comparison validation.

    Args:
        data_dir: Path to data directory containing normalization stats.
        current_train_indices: List of training indices for current split,
                             or None for unsupervised.
        normalize: Whether normalization is enabled.

    Returns:
        Dictionary with 'feature_means' and 'feature_stds' if file exists and split matches,
        None otherwise.
    """
    if not normalize:
        return None

    expected_data_fingerprint = get_data_fingerprint(data_dir)
    expected_preprocessing_fingerprint = get_preprocessing_fingerprint()

    # Hash-keyed path (new format) — hash guarantees index match
    if current_train_indices is not None:
        split_hash = _compute_split_hash(current_train_indices, normalize)
        hashed_path = _normalization_stats_path(data_dir, split_hash)
        if hashed_path.exists():
            try:
                with open(hashed_path) as f:
                    stats = yaml.safe_load(f)
                if not _validate_cache_fingerprints(
                    stats,
                    artifact_name="Normalization stats",
                    expected_data_fingerprint=expected_data_fingerprint,
                    expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
                ):
                    return None
                logger.debug(f"Loaded normalization stats from {hashed_path}")
                return stats
            except Exception as e:
                warnings.warn(
                    f"Failed to load normalization stats from {hashed_path}: {e}. "
                    "Will recompute statistics.",
                    UserWarning,
                )
                return None

    # Legacy fallback — validate by comparing train_indices sets
    legacy_path = data_dir / "normalization_stats.yaml"
    if not legacy_path.exists():
        return None

    try:
        with open(legacy_path) as f:
            stats = yaml.safe_load(f)

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

        if not _validate_cache_fingerprints(
            stats,
            artifact_name="Legacy normalization stats",
            expected_data_fingerprint=expected_data_fingerprint,
            expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        ):
            return None

        return stats
    except Exception as e:
        warnings.warn(
            f"Failed to load normalization stats from {legacy_path}: {e}. "
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
    """Save computed normalization statistics to a hash-keyed file atomically.

    Uses tempfile + os.replace for atomic writes, preventing corruption from
    concurrent runs. The file is keyed by a hash of train_indices + normalize,
    so different splits write to different files.

    Args:
        data_dir: Path to data directory.
        feature_means: Per-feature means tensor.
        feature_stds: Per-feature stds tensor.
        feature_names: List of feature names.
        train_indices: Optional list of training indices used to compute stats.
        normalize: Whether normalization is enabled.
    """
    if train_indices is not None:
        split_hash = _compute_split_hash(train_indices, normalize)
        stats_path = _normalization_stats_path(data_dir, split_hash)
    else:
        split_hash = "unsupervised"
        stats_path = data_dir / "normalization_stats.yaml"

    stats = {
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "feature_names": feature_names,
        "split_hash": split_hash,
        "train_indices_count": len(train_indices) if train_indices else 0,
        "train_indices": train_indices,
        "normalize": normalize,
        "data_fingerprint": get_data_fingerprint(data_dir),
        "preprocessing_fingerprint": get_preprocessing_fingerprint(),
    }

    try:
        fd, tmp_path = tempfile.mkstemp(dir=data_dir, suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(stats, f, default_flow_style=False)
            os.replace(tmp_path, stats_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        warnings.warn(
            f"Failed to save normalization stats to {stats_path}: {e}",
            UserWarning,
        )


def get_tensor_cache_key(
    seq_length: int,
    n_features: int,
) -> str:
    """Generate a hash key for raw tensor caching based on shape parameters.

    The cache stores raw (unnormalized) tensors for the full dataset.
    Normalization is applied at runtime using run-specific train_indices,
    so the cache key only depends on tensor shape parameters.

    Args:
        seq_length: Sequence length.
        n_features: Number of features.

    Returns:
        Hash string to use as cache identifier.
    """
    params = {
        "seq_length": seq_length,
        "n_features": n_features,
    }
    params_str = str(sorted(params.items()))
    return hashlib.md5(params_str.encode()).hexdigest()[:12]


def get_tensor_cache_path(
    data_dir: Path,
    seq_length: int,
    n_features: int,
) -> Path:
    """Get the path to the raw tensor cache file.

    Args:
        data_dir: Path to data directory.
        seq_length: Sequence length.
        n_features: Number of features.

    Returns:
        Path to the tensor cache file.
    """
    cache_key = get_tensor_cache_key(seq_length, n_features)
    cache_dir = data_dir / ".tensor_cache"
    return cache_dir / f"tensors_{cache_key}.pt"


def _try_load_cache(
    cache_path: Path,
    n_features: int,
    seq_length: int,
    *,
    expected_data_fingerprint: str,
    expected_preprocessing_fingerprint: str,
) -> Optional[Dict[str, Any]]:
    """Try to load and validate a raw tensor cache file.

    Args:
        cache_path: Path to the cache file.
        n_features: Expected number of features.
        seq_length: Expected sequence length.

    Returns:
        Dictionary with cached raw tensors if valid, None otherwise.
    """
    if not cache_path.exists():
        return None

    try:
        logger.debug(f"Loading cached tensors from {cache_path.name}")
        cached = torch.load(cache_path, weights_only=True)

        # Validate cache metadata
        if cached.get("n_features") != n_features:
            logger.debug("Cached tensors have different feature count, recomputing")
            return None
        if cached.get("seq_length") != seq_length:
            logger.debug("Cached tensors have different sequence length, recomputing")
            return None
        if not _validate_cache_fingerprints(
            cached,
            artifact_name="Tensor",
            expected_data_fingerprint=expected_data_fingerprint,
            expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        ):
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
        logger.info(f"Loaded {n_samples:,} cached raw samples")
        return cached

    except Exception as e:
        logger.debug(f"Failed to load cached tensors: {e}, recomputing")
        return None


def load_cached_tensors(
    data_dir: Path,
    seq_length: int,
    n_features: int,
) -> Optional[Dict[str, Any]]:
    """Load cached raw tensors if they exist and are valid.

    The cache stores raw (unnormalized) tensors for the full dataset.
    Normalization and stay filtering are applied after loading.

    Args:
        data_dir: Path to data directory.
        seq_length: Sequence length.
        n_features: Number of features.

    Returns:
        Dictionary with cached raw tensors and metadata if valid, None otherwise.
    """
    cache_path = get_tensor_cache_path(data_dir, seq_length, n_features)
    return _try_load_cache(
        cache_path,
        n_features,
        seq_length,
        expected_data_fingerprint=get_data_fingerprint(data_dir),
        expected_preprocessing_fingerprint=get_preprocessing_fingerprint(),
    )


def save_cached_tensors(
    data_dir: Path,
    timeseries_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    seq_length: int,
    n_features: int,
) -> None:
    """Save raw tensors to cache file.

    Stores raw (unnormalized) tensors for the full dataset. Normalization
    and stay filtering are applied at runtime after loading.

    Args:
        data_dir: Path to data directory.
        timeseries_tensor: Raw timeseries tensor (unnormalized).
        mask_tensor: Observation mask tensor.
        seq_length: Sequence length.
        n_features: Number of features.
    """
    cache_path = get_tensor_cache_path(data_dir, seq_length, n_features)
    cache_dir = cache_path.parent
    cache_dir.mkdir(exist_ok=True)

    cache_data = {
        "timeseries_tensor": timeseries_tensor,
        "mask_tensor": mask_tensor,
        "n_features": n_features,
        "seq_length": seq_length,
        "data_fingerprint": get_data_fingerprint(data_dir),
        "preprocessing_fingerprint": get_preprocessing_fingerprint(),
    }

    try:
        logger.debug(f"Saving raw tensors to cache: {cache_path.name}")
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
