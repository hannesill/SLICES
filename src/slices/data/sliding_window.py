"""Sliding window dataset wrapper for SSL pretraining and decompensation.

Creates multiple training samples from longer ICU stays by extracting
overlapping windows. This increases the effective training set size
significantly (e.g., 46k stays -> 100k+ windows with 168h stays).

The windowing is a training-time augmentation that respects patient-level
splits - windows from the same stay never appear in different splits.

For decompensation prediction, computes per-window binary labels on-the-fly
from stay-level death_hours using ``decompensation_pred_hours``.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from slices.data.dataset import ICUDataset

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """Dataset wrapper that creates sliding windows from longer sequences.

    Wraps a base ICUDataset and extracts fixed-size windows from each stay.
    Windows can overlap based on the stride parameter.

    This is designed for SSL pretraining where more diverse windows help
    learn better representations. It also supports decompensation prediction
    via on-the-fly per-window label computation from stay-level death_hours.

    Example:
        >>> base_dataset = ICUDataset("data/processed/mimic-iv-168h")  # 168h sequences
        >>> windowed = SlidingWindowDataset(
        ...     base_dataset,
        ...     window_size=48,  # 48h windows
        ...     stride=24,       # 24h stride (50% overlap)
        ...     stay_indices=[0, 1, 2, 10, 20],  # Only these stays (e.g., train split)
        ... )
        >>> len(windowed)  # More samples than base dataset
        15
        >>> sample = windowed[0]
        >>> sample["timeseries"].shape
        torch.Size([48, 9])

    Attributes:
        base_dataset: The underlying ICUDataset.
        window_size: Size of each window in timesteps.
        stride: Step size between consecutive windows.
        stay_indices: Which stays from base dataset to include.
        decompensation_pred_hours: If set, compute per-window binary labels.
    """

    def __init__(
        self,
        base_dataset: ICUDataset,
        window_size: int,
        stride: Optional[int] = None,
        stay_indices: Optional[List[int]] = None,
        decompensation_pred_hours: Optional[int] = None,
    ) -> None:
        """Initialize sliding window dataset.

        Args:
            base_dataset: The underlying ICUDataset to extract windows from.
            window_size: Size of each window in timesteps (hours).
            stride: Step size between consecutive windows. Defaults to window_size // 2
                   (50% overlap). Use stride=window_size for non-overlapping windows.
            stay_indices: Optional list of stay indices to include. If provided,
                         only windows from these stays are created. This is used
                         to respect patient-level splits (e.g., only train stays).
                         If None, all stays are included.
            decompensation_pred_hours: If set, enables decompensation mode.
                         Per-window binary labels are computed on-the-fly:
                         1 if death occurs in [obs_end, obs_end + pred_hours),
                         0 otherwise. Windows where death occurs during
                         observation [window_start, window_start + window_size)
                         are excluded.
        """
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.stay_indices = (
            stay_indices if stay_indices is not None else list(range(len(base_dataset)))
        )
        self.decompensation_pred_hours = decompensation_pred_hours

        # Validate parameters
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.window_size > base_dataset.seq_length:
            raise ValueError(
                f"window_size ({self.window_size}) cannot exceed base dataset "
                f"seq_length ({base_dataset.seq_length})"
            )

        # Pre-build death_hours lookup for decompensation mode
        self._death_hours: Optional[Dict[int, float]] = None
        if self.decompensation_pred_hours is not None:
            self._death_hours = self._build_death_hours_lookup()

        # Pre-compute window index mapping for fast __getitem__
        self._window_index: List[Tuple[int, int]] = []
        self._build_window_index()

        decomp_info = (
            f", decompensation_pred_hours={decompensation_pred_hours}"
            if decompensation_pred_hours
            else ""
        )
        logger.info(
            f"SlidingWindowDataset created: {len(self._window_index)} "
            f"windows from {len(self.stay_indices)} stays "
            f"(window_size={window_size}, stride={self.stride}"
            f"{decomp_info})"
        )

    def _build_death_hours_lookup(self) -> Dict[int, float]:
        """Build stay_id -> death_hours lookup from base dataset labels.

        Returns:
            Dict mapping stay_idx to death_hours value (inf for survivors).
        """
        death_hours = {}
        for stay_idx in self.stay_indices:
            sample = self.base_dataset[stay_idx]
            if "label" in sample:
                death_hours[stay_idx] = float(sample["label"].item())
        return death_hours

    def _get_actual_length(self, stay_idx: int) -> int:
        """Get actual (non-padded) length of a stay using the observation mask.

        Returns the last timestep with at least one observed feature + 1.
        Falls back to seq_length if mask is all True or unavailable.
        """
        mask = self.base_dataset._mask_tensor[stay_idx]  # (seq_len, n_features)
        # Any feature observed at each timestep
        any_observed = mask.any(dim=1)  # (seq_len,)
        observed_indices = any_observed.nonzero(as_tuple=True)[0]
        if len(observed_indices) == 0:
            return 0
        return int(observed_indices[-1].item()) + 1

    def _build_window_index(self) -> None:
        """Build mapping from window_idx -> (stay_idx, window_start).

        Pre-computes all valid window positions for fast random access.
        A window is valid if:
        - It fits entirely within the sequence length
        - It contains at least one observed timestep (not entirely zero-padded)

        In decompensation mode, windows where death occurs during the
        observation period [window_start, window_start + window_size) are excluded.
        """
        self._window_index = []
        seq_length = self.base_dataset.seq_length

        for stay_idx in self.stay_indices:
            # Calculate number of valid windows for this stay
            max_start = seq_length - self.window_size
            if max_start < 0:
                continue

            # Get actual length to skip windows entirely in zero-padded region
            actual_length = self._get_actual_length(stay_idx)
            if actual_length == 0:
                continue

            # Get death_hours for decompensation exclusion
            death_hour = None
            if self._death_hours is not None:
                death_hour = self._death_hours.get(stay_idx)

            window_start = 0
            while window_start <= max_start:
                # Skip windows that start at or beyond actual data
                # (entire window would be in zero-padded region)
                if window_start >= actual_length:
                    break

                # In decompensation mode, exclude windows where death occurs
                # during observation [window_start, window_start + window_size)
                if death_hour is not None and not math.isinf(death_hour):
                    obs_end = window_start + self.window_size
                    if window_start <= death_hour < obs_end:
                        window_start += self.stride
                        continue

                self._window_index.append((stay_idx, window_start))
                window_start += self.stride

    def __len__(self) -> int:
        """Return total number of windows across all stays."""
        return len(self._window_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single window sample.

        Args:
            idx: Window index.

        Returns:
            Dictionary with same structure as ICUDataset, but with windowed data:
                - 'timeseries': FloatTensor of shape (window_size, n_features)
                - 'mask': BoolTensor of shape (window_size, n_features)
                - 'stay_id': Stay identifier (int)
                - 'window_start': Start position of window in original sequence (int)
                - 'window_idx': Index of this window within the stay (int)
                - 'label': Binary label (decompensation mode) or stay-level label
        """
        if idx < 0 or idx >= len(self._window_index):
            raise IndexError(f"Window index {idx} out of range [0, {len(self._window_index)})")

        stay_idx, window_start = self._window_index[idx]

        # Get full sample from base dataset
        sample = self.base_dataset[stay_idx]

        # Extract window from timeseries and mask
        window_end = window_start + self.window_size
        windowed_timeseries = sample["timeseries"][window_start:window_end]
        windowed_mask = sample["mask"][window_start:window_end]

        # Build result dictionary
        result = {
            "timeseries": windowed_timeseries,
            "mask": windowed_mask,
            "stay_id": sample["stay_id"],
            "window_start": window_start,
            "window_idx": self._get_window_idx_within_stay(stay_idx, window_start),
        }

        # Compute per-window label for decompensation
        if self.decompensation_pred_hours is not None and "label" in sample:
            death_hour = float(sample["label"].item())
            obs_end = window_start + self.window_size
            pred_end = obs_end + self.decompensation_pred_hours

            if not math.isinf(death_hour) and obs_end <= death_hour <= pred_end:
                result["label"] = torch.tensor(1.0, dtype=torch.float32)
            else:
                result["label"] = torch.tensor(0.0, dtype=torch.float32)
        elif "label" in sample:
            # Stay-level labels passed through unchanged
            result["label"] = sample["label"]

        return result

    def _get_window_idx_within_stay(self, stay_idx: int, window_start: int) -> int:
        """Get the index of a window within its stay.

        Args:
            stay_idx: Index of the stay in base dataset.
            window_start: Start position of the window.

        Returns:
            0-indexed position of this window among all windows from the same stay.
        """
        return window_start // self.stride

    def get_windows_per_stay(self) -> Dict[int, int]:
        """Get the number of windows for each stay.

        Returns:
            Dict mapping stay_idx -> number of windows.
        """
        windows_per_stay: Dict[int, int] = {}
        for stay_idx, _ in self._window_index:
            windows_per_stay[stay_idx] = windows_per_stay.get(stay_idx, 0) + 1
        return windows_per_stay

    def get_window_count_statistics(self) -> Dict[str, float]:
        """Get statistics about window counts per stay.

        Returns:
            Dict with min, max, mean, total windows per stay.
        """
        windows_per_stay = self.get_windows_per_stay()
        if not windows_per_stay:
            return {"min": 0, "max": 0, "mean": 0.0, "total": 0}

        counts = list(windows_per_stay.values())
        return {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts) / len(counts),
            "total": len(self._window_index),
        }

    @property
    def n_features(self) -> int:
        """Return number of features from base dataset."""
        return self.base_dataset.n_features

    @property
    def seq_length(self) -> int:
        """Return window size (effective sequence length for this dataset)."""
        return self.window_size

    def get_feature_names(self) -> List[str]:
        """Return feature names from base dataset."""
        return self.base_dataset.get_feature_names()
