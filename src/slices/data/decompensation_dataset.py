"""Sliding-window dataset for decompensation prediction.

Reads RICU sparse timeseries directly (not the truncated dense version).
Generates multiple (window, label) pairs per stay, where each label indicates
whether the patient dies within a prediction window after the observation ends.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DecompensationDataset(Dataset):
    """Sliding-window dataset for decompensation prediction.

    Reads RICU sparse timeseries directly (not the 48h truncated dense version).
    Generates multiple (window, label) pairs per stay.

    Label definition for a window ending at hour t + obs_window:
    - Label = 1 if patient dies within [t + obs_window, t + obs_window + pred_window)
    - Label = 0 if patient survives past t + obs_window + pred_window
    - Sample excluded if patient dies during [t, t + obs_window)
    - Sample excluded if stay ends before t + obs_window
    """

    # RICU categorical encodings (same as RicuExtractor)
    CATEGORICAL_ENCODINGS: Dict[str, Dict[str, float]] = {
        "avpu": {"A": 0.0, "V": 1.0, "P": 2.0, "U": 3.0},
        "mech_vent": {"noninvasive": 1.0, "invasive": 2.0},
    }

    def __init__(
        self,
        ricu_timeseries_path: Path,
        stays_path: Path,
        mortality_path: Path,
        feature_names: List[str],
        obs_window_hours: int = 48,
        pred_window_hours: int = 24,
        stride_hours: int = 6,
        stay_ids: Optional[List[int]] = None,
        normalize: bool = True,
        feature_means: Optional[torch.Tensor] = None,
        feature_stds: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize decompensation dataset.

        Args:
            ricu_timeseries_path: Path to ricu_timeseries.parquet (sparse format).
            stays_path: Path to ricu_stays.parquet.
            mortality_path: Path to ricu_mortality.parquet.
            feature_names: Ordered list of feature column names.
            obs_window_hours: Observation window size in hours.
            pred_window_hours: Prediction window size in hours.
            stride_hours: Hours between consecutive window starts.
            stay_ids: Optional list of stay_ids to include (for splits).
            normalize: Whether to z-score normalize features.
            feature_means: Pre-computed feature means (from training set).
            feature_stds: Pre-computed feature stds (from training set).
        """
        self.obs_window_hours = obs_window_hours
        self.pred_window_hours = pred_window_hours
        self.stride_hours = stride_hours
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.normalize = normalize
        self.feature_means = feature_means
        self.feature_stds = feature_stds

        # Load data
        timeseries_df = pl.read_parquet(ricu_timeseries_path)
        stays_df = pl.read_parquet(stays_path)
        mortality_df = pl.read_parquet(mortality_path)

        # Encode categoricals
        timeseries_df = self._encode_categorical_columns(timeseries_df)

        # Filter to requested stays
        if stay_ids is not None:
            timeseries_df = timeseries_df.filter(pl.col("stay_id").is_in(stay_ids))
            stays_df = stays_df.filter(pl.col("stay_id").is_in(stay_ids))
            mortality_df = mortality_df.filter(pl.col("stay_id").is_in(stay_ids))

        # Compute death_hour for each stay
        self._death_hours = self._compute_death_hours(stays_df, mortality_df)

        # Compute stay lengths in hours
        self._stay_lengths = dict(
            zip(
                stays_df["stay_id"].to_list(),
                (stays_df["los_days"] * 24.0).cast(pl.Float64).to_list(),
            )
        )

        # Group timeseries by stay_id for efficient lookup
        self._stay_timeseries: Dict[int, pl.DataFrame] = {}
        for sid in stays_df["stay_id"].to_list():
            stay_ts = timeseries_df.filter(pl.col("stay_id") == sid)
            self._stay_timeseries[sid] = stay_ts

        # Generate all valid (stay_id, window_start, label) tuples
        self.samples: List[Tuple[int, int, int]] = []
        self._generate_samples()

        logger.info(
            f"DecompensationDataset: {len(self.samples)} samples from "
            f"{len(self._stay_timeseries)} stays "
            f"(obs={obs_window_hours}h, pred={pred_window_hours}h, stride={stride_hours}h)"
        )

    def _encode_categorical_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert categorical strings to floats, durations to hours."""
        for col_name, mapping in self.CATEGORICAL_ENCODINGS.items():
            if col_name not in df.columns:
                continue
            if df[col_name].dtype in (pl.Utf8, pl.String):
                df = df.with_columns(
                    pl.col(col_name).replace_strict(mapping, default=None, return_dtype=pl.Float64)
                )

        for col_name in df.columns:
            if df[col_name].dtype == pl.Duration or str(df[col_name].dtype).startswith("Duration"):
                df = df.with_columns(
                    (pl.col(col_name).dt.total_milliseconds() / 3_600_000.0).alias(col_name)
                )

        return df

    def _compute_death_hours(
        self, stays_df: pl.DataFrame, mortality_df: pl.DataFrame
    ) -> Dict[int, Optional[float]]:
        """Compute hours from admission to death for each stay.

        Returns:
            Dict mapping stay_id -> death_hour (None if survived).
        """
        death_hours: Dict[int, Optional[float]] = {}

        joined = stays_df.select("stay_id", "intime").join(
            mortality_df.select("stay_id", "date_of_death"),
            on="stay_id",
            how="left",
        )

        for row in joined.iter_rows(named=True):
            sid = row["stay_id"]
            if row["date_of_death"] is None:
                death_hours[sid] = None
            else:
                delta = row["date_of_death"] - row["intime"]
                # Handle both timedelta and duration
                if hasattr(delta, "total_seconds"):
                    death_hours[sid] = delta.total_seconds() / 3600.0
                else:
                    death_hours[sid] = delta.total_seconds() / 3600.0

        return death_hours

    def _generate_samples(self) -> None:
        """Generate all valid (stay_id, window_start, label) tuples."""
        for stay_id, stay_length_hours in self._stay_lengths.items():
            death_hour = self._death_hours.get(stay_id)

            # Generate windows: t_start = 0, stride, 2*stride, ...
            t_start = 0
            while True:
                obs_end = t_start + self.obs_window_hours
                pred_end = obs_end + self.pred_window_hours

                # Stay too short for this window
                if obs_end > stay_length_hours:
                    break

                # Check if patient dies during observation [t_start, obs_end)
                if death_hour is not None and t_start <= death_hour < obs_end:
                    t_start += self.stride_hours
                    continue

                # Determine label
                if death_hour is not None and obs_end <= death_hour < pred_end:
                    label = 1  # Death within prediction window
                else:
                    label = 0  # Survived past prediction window (or no death)

                self.samples.append((stay_id, t_start, label))
                t_start += self.stride_hours

    def _extract_dense_window(
        self, stay_id: int, t_start: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a dense (obs_window, n_features) tensor from sparse timeseries.

        Args:
            stay_id: Stay identifier.
            t_start: Window start hour.

        Returns:
            Tuple of (timeseries, mask) tensors, each (obs_window, n_features).
        """
        stay_ts = self._stay_timeseries[stay_id]
        obs_end = t_start + self.obs_window_hours

        # Filter to window range
        window_ts = stay_ts.filter((pl.col("hour") >= t_start) & (pl.col("hour") < obs_end))

        # Initialize dense arrays
        timeseries = np.full((self.obs_window_hours, self.n_features), np.nan, dtype=np.float32)
        mask = np.zeros((self.obs_window_hours, self.n_features), dtype=bool)

        if len(window_ts) == 0:
            tensor = torch.from_numpy(timeseries)
            mask_tensor = torch.from_numpy(mask)
            if self.normalize and self.feature_means is not None:
                tensor = (tensor - self.feature_means) / self.feature_stds
            tensor = torch.nan_to_num(tensor, nan=0.0)
            return tensor, mask_tensor

        # Fill in observed values
        for row in window_ts.iter_rows(named=True):
            hour_idx = int(row["hour"]) - t_start
            if 0 <= hour_idx < self.obs_window_hours:
                for feat_idx, feat_name in enumerate(self.feature_names):
                    val = row.get(feat_name)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        timeseries[hour_idx, feat_idx] = val
                        mask[hour_idx, feat_idx] = True

                        # Also check for mask columns
                        mask_col = f"{feat_name}_mask"
                        if mask_col in row:
                            mask_val = row[mask_col]
                            if mask_val is not None:
                                mask[hour_idx, feat_idx] = bool(mask_val)

        tensor = torch.from_numpy(timeseries)
        mask_tensor = torch.from_numpy(mask)

        # Apply normalization
        if self.normalize and self.feature_means is not None:
            tensor = (tensor - self.feature_means) / self.feature_stds

        # Zero-fill NaN
        tensor = torch.nan_to_num(tensor, nan=0.0)

        return tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stay_id, t_start, label = self.samples[idx]

        timeseries, mask = self._extract_dense_window(stay_id, t_start)

        return {
            "timeseries": timeseries,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.float32),
            "stay_id": torch.tensor(stay_id),
            "window_start": torch.tensor(t_start),
        }

    def compute_normalization_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature means and stds from all samples in this dataset.

        Returns:
            Tuple of (feature_means, feature_stds) tensors.
        """
        all_values = []
        all_masks = []

        for idx in range(len(self.samples)):
            stay_id, t_start, _ = self.samples[idx]
            stay_ts = self._stay_timeseries[stay_id]
            obs_end = t_start + self.obs_window_hours

            window_ts = stay_ts.filter((pl.col("hour") >= t_start) & (pl.col("hour") < obs_end))

            for row in window_ts.iter_rows(named=True):
                vals = []
                ms = []
                for feat_name in self.feature_names:
                    val = row.get(feat_name)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        vals.append(val)
                        ms.append(True)
                    else:
                        vals.append(0.0)
                        ms.append(False)
                all_values.append(vals)
                all_masks.append(ms)

        if not all_values:
            return torch.zeros(self.n_features), torch.ones(self.n_features)

        values_tensor = torch.tensor(all_values, dtype=torch.float32)
        masks_tensor = torch.tensor(all_masks, dtype=torch.bool)

        # Compute masked mean and std
        masked_values = torch.where(masks_tensor, values_tensor, torch.zeros_like(values_tensor))
        valid_counts = masks_tensor.sum(dim=0).float()

        means = torch.where(
            valid_counts > 0,
            masked_values.sum(dim=0) / valid_counts,
            torch.zeros(self.n_features),
        )

        deviations = torch.where(
            masks_tensor,
            (values_tensor - means.unsqueeze(0)) ** 2,
            torch.zeros_like(values_tensor),
        )
        variance = torch.where(
            valid_counts > 1,
            deviations.sum(dim=0) / (valid_counts - 1),
            torch.ones(self.n_features),
        )
        stds = torch.sqrt(variance)
        stds = torch.clamp(stds, min=1e-6)

        return means, stds

    def get_label_distribution(self) -> Dict[str, int | float]:
        """Return label distribution."""
        n_positive = sum(1 for _, _, label in self.samples if label == 1)
        n_negative = len(self.samples) - n_positive
        return {
            "total": len(self.samples),
            "positive": n_positive,
            "negative": n_negative,
            "prevalence": n_positive / len(self.samples) if self.samples else 0.0,
        }
