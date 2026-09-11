"""PyTorch Dataset for ICU time-series data.

Loads preprocessed Parquet files created by the extraction pipeline and
returns (timeseries, mask, labels, static_features) tuples for training.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import polars as pl
import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from slices.data.tensor_cache import (
    load_cached_tensors,
    load_normalization_stats,
    save_cached_tensors,
    save_dataset_metadata,
    save_normalization_stats,
)
from slices.data.tensor_preprocessing import (
    apply_normalization_and_imputation,
    compute_normalization_stats,
    compute_single_sample_stages,
    convert_raw_to_tensors,
    extract_tensors_from_dataframe,
)

# Module-level logger
logger = logging.getLogger(__name__)

# Constants
LARGE_DATASET_WARNING_THRESHOLD = 100_000  # Warn for datasets larger than this
TQDM_MIN_ITEMS = 1000  # Only show progress bar for collections larger than this


def _maybe_tqdm(iterable, desc: str = "", unit: str = "it", disable: bool = False):
    """Conditionally wrap iterable with tqdm progress bar.

    Only shows progress bar for large collections to reduce noise.

    Args:
        iterable: The iterable to wrap.
        desc: Description for the progress bar.
        unit: Unit name for the progress bar.
        disable: Force disable the progress bar.

    Returns:
        The iterable, possibly wrapped with tqdm.
    """
    try:
        length = len(iterable)
    except TypeError:
        length = TQDM_MIN_ITEMS + 1  # Assume large if length unknown

    if disable or length < TQDM_MIN_ITEMS:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


def _log_label_filtering(
    task_name: str,
    original_count: int,
    removed_count: int,
    kept_count: int,
    removal_pct: float,
) -> None:
    """Log label filtering results.

    Args:
        task_name: Name of the task with missing labels.
        original_count: Original number of samples.
        removed_count: Number of samples removed.
        kept_count: Number of samples kept.
        removal_pct: Percentage of samples removed.
    """
    logger.warning(
        f"Label Filtering for task '{task_name}': "
        f"Original={original_count:,}, Removed={removed_count:,} ({removal_pct:.1f}%), "
        f"Kept={kept_count:,}"
    )


class ICUDataset(Dataset):
    """PyTorch Dataset for ICU stays.

    Loads extracted data from Parquet files and returns tensors for training.

    Expected directory structure (created by extractor.run()):
        data_dir/
            static.parquet      # Stay-level metadata (demographics)
            timeseries.parquet  # Dense hourly features with masks
            labels.parquet      # Task labels
            metadata.yaml       # Feature names, task names, etc.

    Example:
        >>> dataset = ICUDataset("data/processed/miiv")
        >>> sample = dataset[0]
        >>> sample["timeseries"].shape  # (seq_length, n_features)
        torch.Size([48, 9])
        >>> sample["mask"].shape  # Same shape
        torch.Size([48, 9])
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        task_name: Optional[str] = None,  # TODO: Add support for multiple tasks
        seq_length: Optional[int] = None,  # TODO: Add support for different sequence lengths
        normalize: bool = True,  # TODO: Add support for different normalization strategies
        train_indices: Optional[List[int]] = None,
        normalization_train_indices: Optional[List[int]] = None,
        handle_missing_labels: str = "filter",
        _excluded_stay_ids: Optional[Set[int]] = None,
    ) -> None:
        """Initialize dataset from extracted Parquet files.

        Args:
            data_dir: Path to directory containing extracted parquet files.
            task_name: Name of the task for label extraction (e.g., 'mortality_24h').
                      If None, no labels are returned.
            seq_length: Override sequence length (uses metadata default if None).
            normalize: Whether to z-score features before imputation. When False,
                      data remains in original units and missing values are
                      imputed from the available feature means.
            train_indices: Optional list of indices for training set. If provided,
                          records the task-filtered training split associated with this
                          dataset. When normalization_train_indices is not provided, these
                          indices are also used for normalization for backwards compatibility.
            normalization_train_indices: Optional list of full raw-cohort training indices
                          used for normalization statistics. DataModule passes this
                          separately for task-filtered downstream runs so AKI uses the same
                          dataset/seed normalizer as SSL pretraining.
            handle_missing_labels: How to handle stays with missing labels when task_name
                                  is specified. Options:
                                  - 'filter': Remove samples with missing labels (default)
                                  - 'raise': Raise ValueError if any labels are missing
            _excluded_stay_ids: Internal parameter. Set of stay_ids already filtered
                               by DataModule. If provided, these stays will be excluded
                               from the dataset to maintain index consistency.
        """
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.normalize = normalize
        self.train_indices = train_indices
        self.normalization_train_indices = (
            normalization_train_indices
            if normalization_train_indices is not None
            else train_indices
        )
        self.handle_missing_labels = handle_missing_labels
        self._excluded_stay_ids = _excluded_stay_ids or set()

        if handle_missing_labels not in ("filter", "raise"):
            raise ValueError(
                f"Invalid handle_missing_labels='{handle_missing_labels}'. "
                "Must be 'filter' or 'raise'."
            )

        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load metadata
        metadata_path = self.data_dir / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                "Run the extraction pipeline first: "
                "uv run python scripts/preprocessing/extract_ricu.py"
            )

        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)

        self.feature_names: List[str] = self.metadata["feature_names"]
        self.n_features: int = len(self.feature_names)
        self.seq_length: int = seq_length or self.metadata["seq_length_hours"]
        self.task_names: List[str] = self.metadata.get("task_names", [])
        self.task_types: Dict[str, str] = self._resolve_task_types()
        self._timeseries_tensor: torch.Tensor
        self._mask_tensor: torch.Tensor

        # Validate task_name if provided
        if task_name is not None and task_name not in self.task_names:
            raise ValueError(
                f"Task '{task_name}' not found in extracted data. "
                f"Available tasks: {self.task_names}"
            )

        # Track removed samples due to missing labels
        self.removed_samples: List[Tuple[int, str]] = []  # List of (stay_id, reason) tuples

        # Load Parquet files and pre-compute all tensors
        self._load_data()

        # Save dataset metadata (including removed samples) for reproducibility
        save_dataset_metadata(
            self.data_dir, self.task_name, self.handle_missing_labels, self.removed_samples
        )

    @staticmethod
    def _task_config_dirs() -> List[Path]:
        """Return candidate task-config directories for task-type resolution."""
        repo_tasks = Path(__file__).resolve().parents[3] / "configs" / "tasks"
        package_tasks = Path(__file__).resolve().parent / "tasks"
        return [repo_tasks, package_tasks]

    def _resolve_task_type_from_config(self, task_name: str) -> Optional[str]:
        """Resolve task type from the checked-in task configuration."""
        for task_dir in self._task_config_dirs():
            config_path = task_dir / f"{task_name}.yaml"
            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            task_type = config.get("task_type")
            if isinstance(task_type, str):
                return task_type

        return None

    def _resolve_task_types(self) -> Dict[str, str]:
        """Resolve task types from metadata with a config-file fallback."""
        label_manifest = self.metadata.get("label_manifest") or {}
        task_types: Dict[str, str] = {}

        for task_name in self.task_names:
            task_type = None

            manifest_entry = label_manifest.get(task_name)
            if isinstance(manifest_entry, dict):
                manifest_task_type = manifest_entry.get("task_type")
                if isinstance(manifest_task_type, str):
                    task_type = manifest_task_type

            if task_type is None:
                task_type = self._resolve_task_type_from_config(task_name)

            if task_type is None:
                logger.warning(
                    "Could not resolve task_type for '%s'; defaulting statistics to binary.",
                    task_name,
                )
                task_type = "binary"

            task_types[task_name] = task_type

        return task_types

    def _load_data(self) -> None:
        """Load data from Parquet files into memory.

        Uses a two-phase approach for efficiency:
        1. Load raw tensors from cache (or extract from parquet on cache miss)
        2. Normalize using full-cohort train indices, then filter excluded stays

        The raw tensor cache is shared across all runs for the same dataset,
        regardless of task, seed, or label_fraction. This reduces cache size
        from ~140GB (one per run config) to ~0.8GB (one per dataset).
        """
        logger.info("Loading data from Parquet files...")

        timeseries_path = self.data_dir / "timeseries.parquet"

        # Load static features and labels (always needed)
        static_path = self.data_dir / "static.parquet"
        logger.debug(f"Loading static features from {static_path.name}")
        self.static_df = pl.read_parquet(static_path)

        labels_path = self.data_dir / "labels.parquet"
        logger.debug(f"Loading labels from {labels_path.name}")
        self.labels_df = pl.read_parquet(labels_path)

        # Phase 1: Get raw tensors (from cache or parquet)
        cached_tensors = load_cached_tensors(self.data_dir, self.seq_length, self.n_features)
        if cached_tensors is not None:
            timeseries_tensor = cached_tensors["timeseries_tensor"]
            masks_tensor = cached_tensors["mask_tensor"]
            # Load stay_ids from parquet (lightweight, just one column)
            all_stay_ids = pl.read_parquet(timeseries_path, columns=["stay_id"])[
                "stay_id"
            ].to_list()
            logger.info("Using cached raw tensors")
        else:
            # Extract from parquet and save raw cache
            logger.debug(f"Loading timeseries from {timeseries_path.name}")
            timeseries_df = pl.read_parquet(timeseries_path)

            n_stays = len(timeseries_df)
            if n_stays > LARGE_DATASET_WARNING_THRESHOLD:
                logger.warning(
                    f"Large dataset detected ({n_stays:,} stays). "
                    "Loading entire dataset into memory."
                )

            all_stay_ids = timeseries_df["stay_id"].to_list()

            timeseries_tensor, masks_tensor = extract_tensors_from_dataframe(
                timeseries_df, self.seq_length, self.n_features
            )
            del timeseries_df

            # Save raw tensors to cache (shared across all runs for this dataset)
            save_cached_tensors(
                self.data_dir,
                timeseries_tensor,
                masks_tensor,
                self.seq_length,
                self.n_features,
            )

        # Phase 2: Normalize before any task-label filtering. This keeps
        # downstream AKI scaling aligned with SSL pretraining because both use
        # the same full-cohort train indices for a dataset/seed.
        cached_stats = load_normalization_stats(
            self.data_dir,
            self.normalization_train_indices,
            self.normalize,
        )
        self._precompute_tensors(
            precomputed_tensors=(timeseries_tensor, masks_tensor),
            train_indices=self.normalization_train_indices,
            cached_stats=cached_stats,
        )

        # Phase 3: Apply stay exclusion filtering (task-specific, e.g. missing labels)
        self.stay_ids = all_stay_ids
        if self._excluded_stay_ids:
            logger.debug(f"Filtering {len(self._excluded_stay_ids):,} excluded stays")
            keep_indices = [
                i for i, sid in enumerate(all_stay_ids) if sid not in self._excluded_stay_ids
            ]
            idx_tensor = torch.tensor(keep_indices, dtype=torch.long)
            self._timeseries_tensor = self._timeseries_tensor[idx_tensor]
            self._mask_tensor = self._mask_tensor[idx_tensor]
            self.stay_ids = [all_stay_ids[i] for i in keep_indices]

            # Filter static and labels DataFrames to match
            excluded_list = list(self._excluded_stay_ids)
            self.static_df = self.static_df.filter(~pl.col("stay_id").is_in(excluded_list))
            self.labels_df = self.labels_df.filter(~pl.col("stay_id").is_in(excluded_list))
            logger.debug(f"Filtered down to {len(self.stay_ids):,} stays")

        # Build stay_id -> index mapping
        logger.debug("Building stay_id index mapping")
        self.stay_id_to_idx = {sid: idx for idx, sid in enumerate(self.stay_ids)}
        logger.info(f"Loaded {len(self.stay_ids):,} stays")

        # Pre-compute labels and static features
        self._precompute_labels_and_static()

    def _precompute_tensors(
        self,
        raw_timeseries: Optional[List[List[List[float]]]] = None,
        raw_masks: Optional[List[List[List[bool]]]] = None,
        train_indices: Optional[List[int]] = None,
        cached_stats: Optional[Dict[str, Any]] = None,
        precomputed_tensors: Optional[tuple] = None,
    ) -> None:
        """Pre-compute all tensors at initialization for fast __getitem__.

        This converts raw nested lists to tensors and applies imputation/normalization
        once, rather than on every access. Uses vectorized operations for speed.

        Args:
            raw_timeseries: List of timeseries arrays (n_samples x seq_len x n_features).
                Not needed if precomputed_tensors is provided.
            raw_masks: List of mask arrays (n_samples x seq_len x n_features).
                Not needed if precomputed_tensors is provided.
            train_indices: Optional list of indices for training set.
            cached_stats: Optional cached normalization statistics.
            precomputed_tensors: Optional tuple of (timeseries_tensor, masks_tensor)
                already extracted. Skips the list-to-tensor conversion step.
        """
        if precomputed_tensors is not None:
            timeseries_tensor, masks_tensor = precomputed_tensors
        else:
            timeseries_tensor, masks_tensor = convert_raw_to_tensors(
                raw_timeseries, raw_masks, self.seq_length, self.n_features
            )
            del raw_timeseries, raw_masks

        n_samples = timeseries_tensor.shape[0]
        logger.info(f"Preprocessing {n_samples:,} samples...")

        # Step 2: Compute preprocessing statistics used for normalization/imputation
        self.feature_means = torch.zeros(self.n_features)
        self.feature_stds = torch.ones(self.n_features)

        if cached_stats is not None:
            # Use cached statistics for reproducibility
            self.feature_means = torch.tensor(cached_stats["feature_means"], dtype=torch.float32)
            self.feature_stds = torch.tensor(cached_stats["feature_stds"], dtype=torch.float32)
            logger.debug("Using cached normalization statistics")
        elif train_indices is not None:
            self.feature_means, self.feature_stds = compute_normalization_stats(
                timeseries_tensor, masks_tensor, train_indices, self.n_features, self.normalize
            )
            # Save computed stats for reproducibility
            save_normalization_stats(
                self.data_dir,
                self.feature_means,
                self.feature_stds,
                self.feature_names,
                train_indices,
                self.normalize,
            )
        elif self.normalize:
            raise ValueError(
                "train_indices must be provided when normalize=True to prevent data leakage. "
                "Pass train_indices from your data splits, or set normalize=False."
            )
        else:
            all_indices = list(range(n_samples))
            self.feature_means, self.feature_stds = compute_normalization_stats(
                timeseries_tensor,
                masks_tensor,
                all_indices,
                self.n_features,
                normalize=False,
            )
            logger.debug(
                "Computed feature means on the loaded dataset for normalize=False imputation"
            )

        # Step 3: Normalize then impute missing values
        timeseries_tensor = apply_normalization_and_imputation(
            timeseries_tensor,
            self.feature_means,
            self.feature_stds,
            self.normalize,
            self.n_features,
        )

        # Keep tensors stacked for memory efficiency and faster access
        self._timeseries_tensor = timeseries_tensor  # (n_samples, seq_len, n_features)
        self._mask_tensor = masks_tensor  # (n_samples, seq_len, n_features)

        logger.info("Preprocessing complete")

    def _precompute_labels_and_static(self) -> None:
        """Pre-compute labels and static features for fast __getitem__ access.

        When task_name is specified, validates that all samples have labels.
        Missing labels are either filtered out or raise an error based on
        handle_missing_labels setting.
        """
        logger.info("Pre-computing labels and static features...")

        # Create lookup dicts using Polars operations (much faster than iter_rows)
        logger.debug("Building lookup dictionaries")
        labels_dict = self.labels_df.to_dicts()
        labels_by_stay = {row["stay_id"]: row for row in labels_dict}

        static_dict = self.static_df.to_dicts()
        static_by_stay = {row["stay_id"]: row for row in static_dict}

        # Pre-compute labels
        self._labels_tensor: Optional[torch.Tensor] = None
        indices_to_keep = []
        original_count = len(self.stay_ids)

        if self.task_name is not None:
            logger.debug(f"Extracting labels for task '{self.task_name}'")

            # Detect multi-label tasks: look for columns prefixed with "{task_name}_"
            multilabel_cols = [
                col for col in self.labels_df.columns if col.startswith(f"{self.task_name}_")
            ]
            is_multilabel = (
                len(multilabel_cols) > 0 and self.task_name not in self.labels_df.columns
            )

            # Validate labels exist for all samples
            missing_label_stays = []
            label_values = []

            for idx, stay_id in enumerate(
                _maybe_tqdm(self.stay_ids, desc="  Validating labels", unit="stay")
            ):
                label_row = labels_by_stay.get(stay_id, {})

                if is_multilabel:
                    # Multi-label: extract vector of values from prefixed columns
                    vals = [label_row.get(col) for col in multilabel_cols]
                    if any(v is None for v in vals):
                        missing_label_stays.append(stay_id)
                    else:
                        label_values.append(vals)
                        indices_to_keep.append(idx)
                else:
                    # Single-label: extract scalar value
                    label_val = label_row.get(self.task_name)
                    if label_val is not None:
                        label_values.append(label_val)
                        indices_to_keep.append(idx)
                    else:
                        missing_label_stays.append(stay_id)

            # Handle missing labels based on configuration
            if missing_label_stays:
                if self.handle_missing_labels == "raise":
                    raise ValueError(
                        f"Missing labels for {len(missing_label_stays)} stays out of "
                        f"{original_count} total. Task: '{self.task_name}'. "
                        f"First 5 missing: {missing_label_stays[:5]}. "
                        "Use task_name=None for unsupervised training or "
                        "handle_missing_labels='filter' to remove them."
                    )
                elif self.handle_missing_labels == "filter":
                    # Track removed samples
                    for stay_id in missing_label_stays:
                        self.removed_samples.append((stay_id, f"missing_{self.task_name}_label"))

                    # Filter out stays with missing labels
                    missing_set = set(missing_label_stays)
                    self.stay_ids = [sid for sid in self.stay_ids if sid not in missing_set]

                    # Filter tensors using index selection (efficient for stacked tensors)
                    indices_tensor = torch.tensor(indices_to_keep, dtype=torch.long)
                    self._timeseries_tensor = self._timeseries_tensor[indices_tensor]
                    self._mask_tensor = self._mask_tensor[indices_tensor]

                    # Filter labels_df and static_df to match (keeps get_label_statistics correct)
                    kept_stay_ids = pl.Series("stay_id", self.stay_ids)
                    self.labels_df = self.labels_df.filter(pl.col("stay_id").is_in(kept_stay_ids))
                    self.static_df = self.static_df.filter(pl.col("stay_id").is_in(kept_stay_ids))

                    # Log filtering
                    removal_pct = (len(missing_label_stays) / original_count) * 100
                    _log_label_filtering(
                        task_name=self.task_name,
                        original_count=original_count,
                        removed_count=len(missing_label_stays),
                        kept_count=len(self.stay_ids),
                        removal_pct=removal_pct,
                    )

            # Convert to stacked tensor for efficient access
            # Multi-label: shape (N, n_classes), single-label: shape (N,)
            self._labels_tensor = torch.tensor(label_values, dtype=torch.float32)
            logger.info(f"Extracted {len(label_values):,} labels")
        else:
            # No labels required for unsupervised training
            self._labels_tensor = None
            indices_to_keep = list(range(len(self.stay_ids)))

        # Pre-compute static features using vectorized Polars operations
        logger.debug("Extracting static features")
        self._static_data = []
        for stay_id in self.stay_ids:
            static_row = static_by_stay.get(stay_id, {})
            self._static_data.append(
                {
                    "age": static_row.get("age"),
                    "gender": static_row.get("gender"),
                    "los_days": static_row.get("los_days"),
                }
            )
        logger.info(f"Pre-computed {len(self._static_data):,} static feature records")

    def __len__(self) -> int:
        """Return number of ICU stays in dataset."""
        return len(self.stay_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single ICU stay sample.

        Args:
            idx: Index of the stay.

        Returns:
            Dictionary with:
                - 'timeseries': FloatTensor of shape (seq_length, n_features)
                - 'mask': BoolTensor of shape (seq_length, n_features)
                         True = observed, False = missing/imputed
                - 'stay_id': Stay identifier (int)
                - 'label': FloatTensor with task label (if task_name specified)
                - 'static': Dict with static features (age, gender, etc.)
        """
        stay_id = self.stay_ids[idx]

        # Get pre-computed tensors via direct indexing (efficient for stacked tensors)
        timeseries = self._timeseries_tensor[idx]
        mask = self._mask_tensor[idx]

        # Build result dictionary
        result = {
            "timeseries": timeseries,
            "mask": mask,
            "stay_id": stay_id,
        }

        # Add label if task specified (use pre-computed stacked tensor)
        if self.task_name is not None and self._labels_tensor is not None:
            result["label"] = self._labels_tensor[idx]

        # Add static features (use pre-computed lookup)
        result["static"] = self._static_data[idx]

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names

    def get_task_names(self) -> List[str]:
        """Return list of available task names."""
        return self.task_names

    def get_label_statistics(
        self, indices: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compute label statistics for each task or subset.

        Returns task-type-aware statistics:
        - binary: {total, positive, negative, prevalence}
        - regression: {total, mean, std, min, max}
        - multiclass: {total, n_classes, class_counts}
        - multilabel: per-subtask prevalence plus aggregate mean prevalence

        Args:
            indices: Optional dataset indices to restrict the computation to.
                When None, computes statistics over the full dataset.

        Returns:
            Dict mapping task_name -> task-specific summary statistics.
        """
        labels_df = self.labels_df
        if indices is not None:
            subset_stay_ids = [self.stay_ids[i] for i in indices]
            labels_df = labels_df.filter(pl.col("stay_id").is_in(subset_stay_ids))

        stats: Dict[str, Dict[str, Any]] = {}
        for task_name in self.task_names:
            task_type = self.task_types.get(task_name, "binary")
            multilabel_cols = [c for c in labels_df.columns if c.startswith(f"{task_name}_")]

            if task_type == "multilabel" and multilabel_cols:
                subtask_stats = {}
                prevalences = []
                for col in multilabel_cols:
                    col_labels = labels_df[col].drop_nulls()
                    pos = (col_labels == 1).sum()
                    tot = len(col_labels)
                    prev = pos / tot if tot > 0 else 0.0
                    subtask_stats[col] = {
                        "task_type": "binary",
                        "total": tot,
                        "positive": pos,
                        "negative": tot - pos,
                        "prevalence": prev,
                    }
                    prevalences.append(prev)
                stats[task_name] = {
                    "task_type": "multilabel",
                    "total": len(labels_df),
                    "n_labels": len(multilabel_cols),
                    "mean_prevalence": (
                        sum(prevalences) / len(prevalences) if prevalences else 0.0
                    ),
                    "subtasks": subtask_stats,
                }
            elif task_name in labels_df.columns:
                labels = labels_df[task_name].drop_nulls()
                total = len(labels)

                if task_type == "regression":
                    labels_float = labels.cast(pl.Float64)
                    std = labels_float.std()
                    stats[task_name] = {
                        "task_type": "regression",
                        "total": total,
                        "mean": float(labels_float.mean()) if total > 0 else 0.0,
                        "std": float(std) if std is not None else 0.0,
                        "min": float(labels_float.min()) if total > 0 else 0.0,
                        "max": float(labels_float.max()) if total > 0 else 0.0,
                    }
                elif task_type == "multiclass":
                    class_counts = {
                        str(label): count
                        for label, count in sorted(Counter(labels.to_list()).items())
                    }
                    stats[task_name] = {
                        "task_type": "multiclass",
                        "total": total,
                        "n_classes": len(class_counts),
                        "class_counts": class_counts,
                    }
                else:
                    positive = (labels == 1).sum()
                    stats[task_name] = {
                        "task_type": "binary",
                        "total": total,
                        "positive": positive,
                        "negative": total - positive,
                        "prevalence": positive / total if total > 0 else 0.0,
                    }
            elif multilabel_cols:
                subtask_stats = {}
                prevalences = []
                for col in multilabel_cols:
                    col_labels = labels_df[col].drop_nulls()
                    pos = (col_labels == 1).sum()
                    tot = len(col_labels)
                    prev = pos / tot if tot > 0 else 0.0
                    subtask_stats[col] = {
                        "task_type": "binary",
                        "total": tot,
                        "positive": pos,
                        "negative": tot - pos,
                        "prevalence": prev,
                    }
                    prevalences.append(prev)
                stats[task_name] = {
                    "task_type": "multilabel",
                    "total": len(labels_df),
                    "n_labels": len(multilabel_cols),
                    "mean_prevalence": sum(prevalences) / len(prevalences) if prevalences else 0.0,
                    "subtasks": subtask_stats,
                }
        return stats

    def get_preprocessing_stages(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get intermediate preprocessing stages for a single sample.

        Reloads raw data from Parquet on demand (the in-memory DataFrame
        is freed after tensor loading to save ~3-5 GB RAM). Debug only.

        The stages are:
            - grid: Raw 2D tensor (seq_length, n_features) with NaN for missing
            - normalized: After z-score normalization + zero-fill (what model sees)

        Args:
            idx: Sample index in the dataset.

        Returns:
            Dict with keys 'grid', 'normalized', each containing:
                - 'timeseries': The tensor at that stage
                - 'mask': The observation mask (same for all stages)

        Example:
            >>> dataset = ICUDataset("data/processed/miiv")
            >>> stages = dataset.get_preprocessing_stages(0)
            >>> stages['grid']['timeseries'].shape
            torch.Size([48, 9])
            >>> torch.isnan(stages['grid']['timeseries']).any()
            True  # Grid has NaN
            >>> torch.isnan(stages['normalized']['timeseries']).any()
            False  # Normalized has no NaN
        """
        stay_id = self.stay_ids[idx]

        # Reload from parquet (timeseries_df freed after init)
        timeseries_df = pl.read_parquet(self.data_dir / "timeseries.parquet")
        if self._excluded_stay_ids:
            timeseries_df = timeseries_df.filter(
                ~pl.col("stay_id").is_in(list(self._excluded_stay_ids))
            )

        row = timeseries_df.filter(pl.col("stay_id") == stay_id).row(0, named=True)
        raw_ts = row["timeseries"]
        raw_mask = row["mask"]

        return compute_single_sample_stages(
            raw_ts,
            raw_mask,
            self.seq_length,
            self.n_features,
            self.feature_means,
            self.feature_stds,
            self.normalize,
        )
