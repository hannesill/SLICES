"""PyTorch Dataset for ICU time-series data.

Loads preprocessed Parquet files created by the extraction pipeline and
returns (timeseries, mask, labels, static_features) tuples for training.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import Dataset


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
        task_name: Optional[str] = None,
        seq_length: Optional[int] = None,
        normalize: bool = True,
        impute_strategy: str = "forward_fill",
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
        """
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.normalize = normalize
        self.impute_strategy = impute_strategy
        
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
        
        # Load Parquet files and pre-compute all tensors
        self._load_data()

    def _load_data(self) -> None:
        """Load data from Parquet files into memory."""
        # Load timeseries (dense format with nested lists)
        timeseries_path = self.data_dir / "timeseries.parquet"
        self.timeseries_df = pl.read_parquet(timeseries_path)
        
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
        
        # Pre-compute tensors for all samples (much faster __getitem__)
        self._precompute_tensors(raw_timeseries, raw_masks)
        
        # Pre-compute labels and static features
        self._precompute_labels_and_static()

    def _precompute_tensors(
        self, 
        raw_timeseries: List[List[List[float]]], 
        raw_masks: List[List[List[bool]]]
    ) -> None:
        """Pre-compute all tensors at initialization for fast __getitem__.
        
        This converts raw nested lists to tensors and applies imputation/normalization
        once, rather than on every access.
        
        Args:
            raw_timeseries: List of timeseries arrays (n_samples x seq_len x n_features).
            raw_masks: List of mask arrays (n_samples x seq_len x n_features).
        """
        n_samples = len(raw_timeseries)
        
        # First pass: convert to tensors and compute normalization stats
        timeseries_list = []
        mask_list = []
        
        # For computing mean/std on observed values only
        observed_sums = torch.zeros(self.n_features)
        observed_sq_sums = torch.zeros(self.n_features)
        observed_counts = torch.zeros(self.n_features)
        
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
                        if mask_val:  # Only observed values for stats
                            observed_sums[f] += val
                            observed_sq_sums[f] += val * val
                            observed_counts[f] += 1
            
            timeseries_list.append(ts_tensor)
            mask_list.append(mask_tensor)
        
        # Compute mean and std from observed values
        self.feature_means = torch.zeros(self.n_features)
        self.feature_stds = torch.ones(self.n_features)
        
        for f in range(self.n_features):
            if observed_counts[f] > 0:
                mean = observed_sums[f] / observed_counts[f]
                variance = (observed_sq_sums[f] / observed_counts[f]) - (mean * mean)
                std = math.sqrt(max(variance, 0))
                self.feature_means[f] = mean
                self.feature_stds[f] = std if std > 1e-6 else 1.0
        
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

    def _precompute_labels_and_static(self) -> None:
        """Pre-compute labels and static features for fast __getitem__ access."""
        n_samples = len(self.stay_ids)
        
        # Create lookup dicts for fast access
        labels_by_stay = {
            row["stay_id"]: row 
            for row in self.labels_df.iter_rows(named=True)
        }
        static_by_stay = {
            row["stay_id"]: row 
            for row in self.static_df.iter_rows(named=True)
        }
        
        # Pre-compute labels
        self._labels_tensors = []
        for stay_id in self.stay_ids:
            if self.task_name is not None:
                label_row = labels_by_stay.get(stay_id, {})
                label_val = label_row.get(self.task_name)
                if label_val is not None:
                    self._labels_tensors.append(torch.tensor(label_val, dtype=torch.float32))
                else:
                    self._labels_tensors.append(torch.tensor(float("nan"), dtype=torch.float32))
            else:
                self._labels_tensors.append(None)
        
        # Pre-compute static features
        self._static_data = []
        for stay_id in self.stay_ids:
            static_row = static_by_stay.get(stay_id, {})
            self._static_data.append({
                "age": static_row.get("age"),
                "gender": static_row.get("gender"),
                "los_days": static_row.get("los_days"),
            })

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

