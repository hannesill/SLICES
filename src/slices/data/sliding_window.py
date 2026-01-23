"""Sliding window dataset wrapper for SSL pretraining.

Creates multiple training samples from longer ICU stays by extracting
overlapping windows. This increases the effective training set size
significantly (e.g., 46k stays -> 100k+ windows with 168h stays).

The windowing is a training-time augmentation that respects patient-level
splits - windows from the same stay never appear in different splits.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from slices.data.dataset import ICUDataset

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """Dataset wrapper that creates sliding windows from longer sequences.

    Wraps a base ICUDataset and extracts fixed-size windows from each stay.
    Windows can overlap based on the stride parameter.

    This is designed for SSL pretraining where more diverse windows help
    learn better representations. For supervised tasks, windowing may not
    be appropriate as labels are stay-level.

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
    """

    def __init__(
        self,
        base_dataset: ICUDataset,
        window_size: int,
        stride: Optional[int] = None,
        stay_indices: Optional[List[int]] = None,
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
        """
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.stay_indices = (
            stay_indices if stay_indices is not None else list(range(len(base_dataset)))
        )

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

        # Pre-compute window index mapping for fast __getitem__
        self._window_index: List[Tuple[int, int]] = []
        self._build_window_index()

        logger.info(
            f"SlidingWindowDataset created: {len(self._window_index)} windows "
            f"from {len(self.stay_indices)} stays (window_size={window_size}, stride={self.stride})"
        )

    def _build_window_index(self) -> None:
        """Build mapping from window_idx -> (stay_idx, window_start).

        Pre-computes all valid window positions for fast random access.
        A window is valid if it fits entirely within the sequence length.
        """
        self._window_index = []
        seq_length = self.base_dataset.seq_length

        for stay_idx in self.stay_indices:
            # Calculate number of valid windows for this stay
            # A window starting at position `start` covers [start, start + window_size)
            # Valid if: start + window_size <= seq_length
            # So: start <= seq_length - window_size
            max_start = seq_length - self.window_size
            if max_start < 0:
                continue  # No valid windows (shouldn't happen if validation passed)

            # Generate window starts: 0, stride, 2*stride, ... while start <= max_start
            window_start = 0
            while window_start <= max_start:
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

        # Labels are stay-level, so include them unchanged if present
        # (though for SSL pretraining, labels are typically not used)
        if "label" in sample:
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
