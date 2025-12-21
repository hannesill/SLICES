"""PyTorch Dataset for ICU time-series data.

Loads preprocessed Parquet files created by the extraction pipeline and
returns (timeseries, mask, labels, static_features) tuples for training.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import torch
import yaml
from torch.utils.data import Dataset


def _log_label_filtering(
    task_name: str,
    original_count: int,
    removed_count: int,
    kept_count: int,
    removal_pct: float,
) -> None:
    """Log label filtering results to console.

    Args:
        task_name: Name of the task with missing labels.
        original_count: Original number of samples.
        removed_count: Number of samples removed.
        kept_count: Number of samples kept.
        removal_pct: Percentage of samples removed.
    """
    try:
        from rich.console import Console

        console = Console()
        console.print(
            f"[yellow]⚠️  Label Filtering for task '{task_name}'[/yellow]\n"
            f"  Original samples: {original_count:,}\n"
            f"  Removed (missing labels): {removed_count:,} ({removal_pct:.1f}%)\n"
            f"  Kept: {kept_count:,}\n"
        )
    except ImportError:
        # Fallback if rich is not available
        print(
            f"Warning: Label Filtering for task '{task_name}'\n"
            f"  Original samples: {original_count}\n"
            f"  Removed (missing labels): {removed_count} ({removal_pct:.1f}%)\n"
            f"  Kept: {kept_count}"
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
        >>> dataset = ICUDataset("data/processed/mimic-iv-demo")
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
        impute_strategy: str = "forward_fill",
        train_indices: Optional[List[int]] = None,
        handle_missing_labels: str = "filter",
    ) -> None:
        """Initialize dataset from extracted Parquet files.

        Args:
            data_dir: Path to directory containing extracted parquet files.
            task_name: Name of the task for label extraction (e.g., 'mortality_24h').
                      If None, no labels are returned.
            seq_length: Override sequence length (uses metadata default if None).
            normalize: Whether to normalize features (z-score per feature).
            impute_strategy: Strategy for imputing missing values.
                           Options: 'forward_fill', 'zero', 'mean', 'none'
            train_indices: Optional list of indices for training set. If provided,
                          normalization statistics are computed only on these samples.
                          This prevents data leakage from val/test sets.
            handle_missing_labels: How to handle stays with missing labels when task_name
                                  is specified. Options:
                                  - 'filter': Remove samples with missing labels (default)
                                  - 'raise': Raise ValueError if any labels are missing
        """
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.normalize = normalize
        self.impute_strategy = impute_strategy
        self.train_indices = train_indices
        self.handle_missing_labels = handle_missing_labels

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
                "Run the extraction pipeline first: uv run python scripts/extract_mimic_iv.py"
            )

        with open(metadata_path) as f:
            self.metadata = yaml.safe_load(f)

        self.feature_names: List[str] = self.metadata["feature_names"]
        self.n_features: int = len(self.feature_names)
        self.seq_length: int = seq_length or self.metadata["seq_length_hours"]
        self.task_names: List[str] = self.metadata.get("task_names", [])

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
        self._save_dataset_metadata()

    def _load_data(self) -> None:
        """Load data from Parquet files into memory."""
        # Load timeseries (dense format with nested lists)
        timeseries_path = self.data_dir / "timeseries.parquet"
        self.timeseries_df = pl.read_parquet(timeseries_path)

        # Warn for large datasets that may cause memory issues
        n_stays = len(self.timeseries_df)
        if n_stays > 100000:
            import warnings

            warnings.warn(
                f"Large dataset detected ({n_stays:,} stays). "
                "Loading entire dataset into memory. "
                "Consider using chunked loading for datasets > 200K stays.",
                UserWarning,
            )

        # Load static features
        static_path = self.data_dir / "static.parquet"
        self.static_df = pl.read_parquet(static_path)

        # Load labels
        labels_path = self.data_dir / "labels.parquet"
        self.labels_df = pl.read_parquet(labels_path)

        # Create stay_id -> index mapping
        self.stay_ids = self.timeseries_df["stay_id"].to_list()
        self.stay_id_to_idx = {sid: idx for idx, sid in enumerate(self.stay_ids)}

        # Pre-extract raw arrays
        raw_timeseries = self.timeseries_df["timeseries"].to_list()
        raw_masks = self.timeseries_df["mask"].to_list()

        # Try to load existing normalization stats (for reproducibility)
        cached_stats = self._load_normalization_stats(self.train_indices)

        # Pre-compute tensors for all samples (much faster __getitem__)
        self._precompute_tensors(raw_timeseries, raw_masks, self.train_indices, cached_stats)

        # Pre-compute labels and static features
        self._precompute_labels_and_static()

    def _precompute_tensors(
        self,
        raw_timeseries: List[List[List[float]]],
        raw_masks: List[List[List[bool]]],
        train_indices: Optional[List[int]] = None,
        cached_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Pre-compute all tensors at initialization for fast __getitem__.

        This converts raw nested lists to tensors and applies imputation/normalization
        once, rather than on every access.

        IMPORTANT: If train_indices is provided, normalization statistics are computed
        ONLY on training samples to prevent data leakage from validation/test sets.

        Normalization statistics are cached to metadata for reproducibility across
        dataset reloads.

        Args:
            raw_timeseries: List of timeseries arrays (n_samples x seq_len x n_features).
            raw_masks: List of mask arrays (n_samples x seq_len x n_features).
            train_indices: Optional list of indices for training set. If provided,
                          normalization stats computed only on these samples.
            cached_stats: Optional cached normalization statistics. If provided, uses
                         these instead of recomputing (for reproducibility).
        """
        n_samples = len(raw_timeseries)

        # First pass: convert to tensors and compute normalization stats
        timeseries_list = []
        mask_list = []

        # For computing mean/std on observed values only
        observed_sums = torch.zeros(self.n_features)
        observed_sq_sums = torch.zeros(self.n_features)
        observed_counts = torch.zeros(self.n_features)

        # Determine which indices to use for statistics computation
        # If train_indices is provided, ONLY use those for normalization stats
        # Otherwise, use ALL samples (backwards compatible with existing code)
        indices_for_stats = (
            set(train_indices) if train_indices is not None else set(range(n_samples))
        )

        # If cached stats exist, skip computation
        should_compute_stats = cached_stats is None and self.normalize

        for i in range(n_samples):
            ts_data = raw_timeseries[i]
            mask_data = raw_masks[i]
            actual_len = min(len(ts_data), self.seq_length)

            # Initialize tensors
            ts_tensor = torch.full((self.seq_length, self.n_features), float("nan"))
            mask_tensor = torch.zeros((self.seq_length, self.n_features), dtype=torch.bool)

            # Fill from raw data
            for t in range(actual_len):
                for f in range(self.n_features):
                    val = ts_data[t][f]
                    mask_val = mask_data[t][f]
                    mask_tensor[t, f] = mask_val
                    if val is not None and not math.isnan(val):
                        ts_tensor[t, f] = val
                        # Only accumulate stats for training set samples if computing
                        if should_compute_stats and mask_val and i in indices_for_stats:
                            observed_sums[f] += val
                            observed_sq_sums[f] += val * val
                            observed_counts[f] += 1

            timeseries_list.append(ts_tensor)
            mask_list.append(mask_tensor)

        # Compute mean and std from observed values or load from cache
        self.feature_means = torch.zeros(self.n_features)
        self.feature_stds = torch.ones(self.n_features)

        if cached_stats is not None:
            # Use cached statistics for reproducibility
            self.feature_means = torch.tensor(cached_stats["feature_means"], dtype=torch.float32)
            self.feature_stds = torch.tensor(cached_stats["feature_stds"], dtype=torch.float32)
        elif should_compute_stats:
            # Compute stats from training data
            for f in range(self.n_features):
                if observed_counts[f] > 0:
                    mean = observed_sums[f] / observed_counts[f]
                    variance = (observed_sq_sums[f] / observed_counts[f]) - (mean * mean)
                    std = math.sqrt(max(variance, 0))
                    self.feature_means[f] = mean
                    self.feature_stds[f] = std if std > 1e-6 else 1.0
            # Save computed stats for reproducibility
            self._save_normalization_stats(train_indices)

        # Second pass: apply imputation and normalization
        self._timeseries_tensors = []
        self._mask_tensors = mask_list  # Masks don't need processing

        for i in range(n_samples):
            ts_tensor = timeseries_list[i]
            mask_tensor = mask_list[i]

            # Apply imputation
            ts_tensor = self._impute_tensor(ts_tensor, mask_tensor)

            # Apply normalization
            if self.normalize:
                ts_tensor = (ts_tensor - self.feature_means) / self.feature_stds

            self._timeseries_tensors.append(ts_tensor)

    def _impute_tensor(self, timeseries: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Impute missing values in a single timeseries tensor.

        Args:
            timeseries: FloatTensor of shape (seq_length, n_features) with NaN for missing.
            mask: BoolTensor of shape (seq_length, n_features).

        Returns:
            Imputed timeseries tensor.
        """
        if self.impute_strategy == "none":
            return timeseries

        if self.impute_strategy == "zero":
            return torch.nan_to_num(timeseries, nan=0.0)

        if self.impute_strategy == "mean":
            imputed = timeseries.clone()
            for f in range(self.n_features):
                nan_mask = torch.isnan(imputed[:, f])
                imputed[nan_mask, f] = self.feature_means[f]
            return imputed

        if self.impute_strategy == "forward_fill":
            imputed = timeseries.clone()
            for f in range(self.n_features):
                last_valid = self.feature_means[f].item()  # Default to mean if no prior
                for t in range(self.seq_length):
                    if mask[t, f] and not torch.isnan(imputed[t, f]):
                        last_valid = imputed[t, f].item()
                    elif torch.isnan(imputed[t, f]):
                        imputed[t, f] = last_valid
            return imputed

        raise ValueError(f"Unknown imputation strategy: {self.impute_strategy}")

    def _load_normalization_stats(
        self, current_train_indices: Optional[List[int]]
    ) -> Optional[Dict[str, Any]]:
        """Load cached normalization statistics from file if they exist and match current split.

        Validates that cached statistics were computed on the same training set to prevent
        data leakage from validation/test sets.

        Args:
            current_train_indices: List of training indices for current split,
                                 or None for unsupervised.

        Returns:
            Dictionary with 'feature_means' and 'feature_stds' if file exists and split matches,
            None otherwise.
        """
        if not self.normalize:
            return None

        stats_path = self.data_dir / "normalization_stats.yaml"
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
                import warnings

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
            import warnings

            warnings.warn(
                f"Failed to load normalization stats from {stats_path}: {e}. "
                "Will recompute statistics.",
                UserWarning,
            )
            return None

    def _save_normalization_stats(self, train_indices: Optional[List[int]]) -> None:
        """Save computed normalization statistics to file for reproducibility.

        Args:
            train_indices: Optional list of training indices used to compute stats.
        """
        stats_path = self.data_dir / "normalization_stats.yaml"

        stats = {
            "feature_means": self.feature_means.tolist(),
            "feature_stds": self.feature_stds.tolist(),
            "feature_names": self.feature_names,
            "train_indices": train_indices,
            "normalize": self.normalize,
            "impute_strategy": self.impute_strategy,
        }

        try:
            with open(stats_path, "w") as f:
                yaml.dump(stats, f, default_flow_style=False)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to save normalization stats to {stats_path}: {e}",
                UserWarning,
            )

    def _save_dataset_metadata(self) -> None:
        """Save dataset metadata including removed samples for reproducibility.

        Creates or updates dataset_metadata.yaml with information about:
        - Number of samples used (original vs. final after filtering)
        - Removed samples and reasons for removal
        - Task name and label handling strategy
        """
        if not self.removed_samples:
            return  # Nothing to save if no samples were removed

        metadata_path = self.data_dir / "dataset_metadata.yaml"

        metadata = {
            "task_name": self.task_name,
            "handle_missing_labels": self.handle_missing_labels,
            "removed_samples_count": len(self.removed_samples),
            "removed_samples": [
                {"stay_id": stay_id, "reason": reason} for stay_id, reason in self.removed_samples
            ],
        }

        try:
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to save dataset metadata to {metadata_path}: {e}",
                UserWarning,
            )

    def _precompute_labels_and_static(self) -> None:
        """Pre-compute labels and static features for fast __getitem__ access.

        When task_name is specified, validates that all samples have labels.
        Missing labels are either filtered out or raise an error based on
        handle_missing_labels setting.
        """
        # Create lookup dicts for fast access
        labels_by_stay = {row["stay_id"]: row for row in self.labels_df.iter_rows(named=True)}
        static_by_stay = {row["stay_id"]: row for row in self.static_df.iter_rows(named=True)}

        # Pre-compute labels
        self._labels_tensors: List[Optional[torch.Tensor]] = []
        indices_to_keep = []  # Track which samples to keep
        original_count = len(self.stay_ids)

        if self.task_name is not None:
            # Validate labels exist for all samples
            missing_label_stays = []

            for idx, stay_id in enumerate(self.stay_ids):
                label_row = labels_by_stay.get(stay_id, {})
                label_val = label_row.get(self.task_name)

                if label_val is not None:
                    self._labels_tensors.append(torch.tensor(label_val, dtype=torch.float32))
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
                    self.stay_ids = [sid for sid in self.stay_ids if sid not in missing_label_stays]

                    # Filter tensors (must be done after they're precomputed)
                    self._timeseries_tensors = [
                        self._timeseries_tensors[idx] for idx in indices_to_keep
                    ]
                    self._mask_tensors = [self._mask_tensors[idx] for idx in indices_to_keep]

                    # Log filtering to console
                    removal_pct = (len(missing_label_stays) / original_count) * 100
                    _log_label_filtering(
                        task_name=self.task_name,
                        original_count=original_count,
                        removed_count=len(missing_label_stays),
                        kept_count=len(self.stay_ids),
                        removal_pct=removal_pct,
                    )
        else:
            # No labels required for unsupervised training
            self._labels_tensors = [None] * len(self.stay_ids)
            indices_to_keep = list(range(len(self.stay_ids)))

        # Pre-compute static features (only for kept samples)
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

        # Get pre-computed tensors (fast lookup)
        timeseries = self._timeseries_tensors[idx]
        mask = self._mask_tensors[idx]

        # Build result dictionary
        result = {
            "timeseries": timeseries,
            "mask": mask,
            "stay_id": stay_id,
        }

        # Add label if task specified (use pre-computed lookup)
        if self.task_name is not None:
            result["label"] = self._labels_tensors[idx]

        # Add static features (use pre-computed lookup)
        result["static"] = self._static_data[idx]

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names

    def get_task_names(self) -> List[str]:
        """Return list of available task names."""
        return self.task_names

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute label statistics for each task.

        Returns:
            Dict mapping task_name -> {count, positive, negative, prevalence}
        """
        stats = {}
        for task_name in self.task_names:
            if task_name in self.labels_df.columns:
                labels = self.labels_df[task_name].drop_nulls()
                positive = (labels == 1).sum()
                total = len(labels)
                stats[task_name] = {
                    "total": total,
                    "positive": positive,
                    "negative": total - positive,
                    "prevalence": positive / total if total > 0 else 0.0,
                }
        return stats
