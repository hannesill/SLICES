"""Lightning DataModule for ICU data.

Handles patient-level splits, data loading, and batching for training.
Ensures no patient appears in multiple splits (prevents data leakage).
Supports sliding windows for SSL pretraining with longer sequences.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from slices.constants import (
    NORMALIZE,
    PIN_MEMORY,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from slices.data.dataset import ICUDataset
from slices.data.sliding_window import SlidingWindowDataset
from slices.data.splits import (
    compute_patient_level_splits,
    save_global_split_info,
    subsample_train_indices,
)

# Module-level logger
logger = logging.getLogger(__name__)


def icu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching ICU samples.

    Handles both regular ICUDataset samples and SlidingWindowDataset samples
    which include additional window_start and window_idx fields.

    Args:
        batch: List of sample dictionaries from ICUDataset or SlidingWindowDataset.

    Returns:
        Batched dictionary with stacked tensors.
    """
    # Stack tensors
    timeseries = torch.stack([s["timeseries"] for s in batch])  # (B, T, D)
    mask = torch.stack([s["mask"] for s in batch])  # (B, T, D)
    stay_ids = torch.tensor([s["stay_id"] for s in batch])  # (B,)

    result = {
        "timeseries": timeseries,
        "mask": mask,
        "stay_id": stay_ids,
    }

    # Stack labels if present
    if "label" in batch[0]:
        labels = torch.stack([s["label"] for s in batch])  # (B,)
        result["label"] = labels

    # Collate static features if present
    if "static" in batch[0]:
        static_keys = batch[0]["static"].keys()
        result["static"] = {key: [s["static"].get(key) for s in batch] for key in static_keys}

    # Stack sliding window metadata if present
    if "window_start" in batch[0]:
        result["window_start"] = torch.tensor([s["window_start"] for s in batch])  # (B,)
    if "window_idx" in batch[0]:
        result["window_idx"] = torch.tensor([s["window_idx"] for s in batch])  # (B,)

    return result


class ICUDataModule(L.LightningDataModule):
    """Lightning DataModule for ICU data.

    Implements patient-level splits to prevent data leakage.
    Uses hashing of patient_id for deterministic, reproducible splits.

    Example:
        >>> dm = ICUDataModule(
        ...     processed_dir="data/processed/miiv",
        ...     task_name="mortality_24h",
        ...     batch_size=32,
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        task_name: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        seq_length: Optional[int] = None,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        seed: int = 42,
        normalize: bool = NORMALIZE,
        pin_memory: bool = PIN_MEMORY,
        # Label fraction for semi-supervised / label-efficiency ablations
        label_fraction: float = 1.0,
        # Sliding window parameters for SSL pretraining
        enable_sliding_windows: bool = False,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
    ) -> None:
        """Initialize DataModule.

        Args:
            processed_dir: Directory containing extracted parquet files.
            task_name: Task for label extraction (e.g., 'mortality_24h').
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            seq_length: Override sequence length (uses metadata default if None).
            train_ratio: Fraction of patients for training.
            val_ratio: Fraction of patients for validation.
            test_ratio: Fraction of patients for testing.
            seed: Random seed for reproducible splits.
            normalize: Whether to normalize features.
            pin_memory: Whether to pin memory for faster GPU transfer.
            label_fraction: Fraction of labeled training data to use (0, 1].
                Default 1.0 uses all training data. Lower values subsample
                the training set for label-efficiency ablations (e.g., 0.01
                for 1% of labels). Val/test sets always use all data.
                Subsampling is deterministic given the seed.
            enable_sliding_windows: Whether to use sliding windows for training.
                Useful for SSL pretraining with longer sequences (e.g., 168h).
                When enabled, train_dataloader uses overlapping windows and
                val_dataloader uses non-overlapping windows.
            window_size: Size of sliding windows in timesteps (hours).
                Defaults to seq_length if None (no windowing effect).
            window_stride: Step between consecutive windows.
                Defaults to window_size // 2 (50% overlap) for training.
                Validation always uses stride=window_size (non-overlapping).
        """
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.task_name = task_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.normalize = normalize
        self.pin_memory = pin_memory
        self.label_fraction = label_fraction

        if not 0.0 < label_fraction <= 1.0:
            raise ValueError(f"label_fraction must be in (0, 1], got {label_fraction}")

        # Sliding window parameters
        self.enable_sliding_windows = enable_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride

        # Validate ratios
        ratios = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
        for name, ratio in ratios.items():
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {ratio}")

        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Will be set in setup()
        self.dataset: Optional[ICUDataset] = None
        self.full_train_indices: List[int] = []
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test.

        Creates patient-level splits BEFORE initializing the dataset to ensure
        normalization statistics are computed only on training data (no leakage).

        Args:
            stage: Stage name ('fit', 'validate', 'test', or None).
        """
        # Skip if already set up (Lightning calls setup() again in trainer.fit())
        if self.dataset is not None:
            logger.debug("DataModule already set up, skipping")
            return

        logger.info("Setting up ICUDataModule")

        # CRITICAL: Get splits FIRST before creating dataset
        # This also filters stays with missing labels to ensure index consistency
        logger.debug("[Step 1/3] Computing patient-level splits")
        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            self._static_df,
            self._labels_df,
            self._all_stay_ids,
            self._filtered_stay_ids,
            self._excluded_stay_ids,
        ) = compute_patient_level_splits(
            self.processed_dir,
            self.task_name,
            self.seed,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

        # Save full train indices for normalization BEFORE subsampling.
        # Normalization must always use the full training split to avoid
        # noisy stats at low label fractions and pretrain/finetune mismatch.
        self.full_train_indices = list(self.train_indices)
        normalization_train_indices = list(self.full_train_indices)

        # Subsample training indices for label-efficiency ablations
        if self.label_fraction < 1.0:
            self.train_indices = subsample_train_indices(
                self.train_indices, self.label_fraction, self.seed
            )

        # Create dataset with FULL training indices for normalization.
        # self.train_indices (possibly subsampled) is used by train_dataloader
        # to create a Subset for optimization.
        logger.debug("[Step 2/3] Creating ICUDataset")
        self.dataset = ICUDataset(
            data_dir=self.processed_dir,
            task_name=self.task_name,
            seq_length=self.seq_length,
            normalize=self.normalize,
            train_indices=normalization_train_indices,
            # Use 'raise' since we pre-filtered - any missing labels now is a bug
            handle_missing_labels="raise" if self.task_name else "filter",
            # Pass excluded stays so Dataset can validate consistency
            _excluded_stay_ids=getattr(self, "_excluded_stay_ids", None),
        )

        # Validate dataset size matches expected filtered size
        expected_size = (
            len(self._filtered_stay_ids) if hasattr(self, "_filtered_stay_ids") else None
        )
        if expected_size is not None and len(self.dataset) != expected_size:
            raise RuntimeError(
                f"Dataset size mismatch! Expected {expected_size} stays "
                f"(after filtering), got {len(self.dataset)}. "
                "This indicates an index consistency bug."
            )

        # Save split information only when the canonical prep artifact is absent.
        # Prepared datasets already own a stable splits.yaml, and downstream runs
        # must not rewrite it after task filtering or label-efficiency subsampling.
        logger.debug("[Step 3/3] Preserving canonical split information")
        splits_path = self.processed_dir / "splits.yaml"
        if not splits_path.exists():
            save_global_split_info(
                processed_dir=self.processed_dir,
                static_df=self._static_df,
                stay_ids=self._all_stay_ids,
                seed=self.seed,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                dataset=self.dataset if self.label_fraction < 1.0 else None,
                train_subset_indices=self.train_indices if self.label_fraction < 1.0 else None,
                label_fraction=self.label_fraction,
            )
        else:
            logger.debug("Canonical splits.yaml already exists; leaving it unchanged")

        # Free temporary data used only during setup — Dataset holds its own copies
        del self._static_df, self._labels_df
        del self._all_stay_ids, self._filtered_stay_ids, self._excluded_stay_ids

        logger.info(
            f"DataModule setup complete: "
            f"Train={len(self.train_indices):,}, Val={len(self.val_indices):,}, "
            f"Test={len(self.test_indices):,} stays"
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader.

        If sliding windows are enabled, wraps the base dataset with
        SlidingWindowDataset using overlapping windows.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")

        if self.enable_sliding_windows:
            window_size = self.window_size or self.dataset.seq_length
            window_stride = self.window_stride or window_size // 2

            train_dataset = SlidingWindowDataset(
                self.dataset,
                window_size=window_size,
                stride=window_stride,
                stay_indices=self.train_indices,
            )
            logger.info(
                f"Train dataloader using sliding windows: {len(train_dataset)} windows "
                f"(window_size={window_size}, stride={window_stride})"
            )
        else:
            train_dataset = Subset(self.dataset, self.train_indices)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
            drop_last=True,  # Avoid small batches
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader.

        If sliding windows are enabled, uses NON-overlapping windows
        (stride=window_size) to avoid inflated validation metrics.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")

        if self.enable_sliding_windows:
            window_size = self.window_size or self.dataset.seq_length
            val_stride = window_size  # Non-overlapping

            val_dataset = SlidingWindowDataset(
                self.dataset,
                window_size=window_size,
                stride=val_stride,
                stay_indices=self.val_indices,
            )
            logger.info(
                f"Val dataloader using sliding windows: {len(val_dataset)} windows "
                f"(window_size={window_size}, stride={val_stride})"
            )
        else:
            val_dataset = Subset(self.dataset, self.val_indices)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before test_dataloader()")

        test_dataset = Subset(self.dataset, self.test_indices)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=icu_collate_fn,
        )

    def get_feature_dim(self) -> int:
        """Return number of input features."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_feature_dim()")
        return self.dataset.n_features

    def get_seq_length(self) -> int:
        """Return sequence length."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_seq_length()")
        return self.dataset.seq_length

    def get_split_info(self) -> Dict[str, Any]:
        """Return information about data splits.

        Returns:
            Dict with stay counts, patient counts, and actual ratios.
            Use this to verify splits are reasonable.
        """
        total_stays = len(self.dataset) if self.dataset else 0

        # Count unique patients per split
        if self.dataset is not None:
            static_df = self.dataset.static_df
            stay_to_patient = dict(
                zip(static_df["stay_id"].to_list(), static_df["patient_id"].to_list())
            )

            train_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.train_indices}
            val_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.val_indices}
            test_patients = {stay_to_patient[self.dataset.stay_ids[i]] for i in self.test_indices}

            n_train_patients = len(train_patients)
            n_val_patients = len(val_patients)
            n_test_patients = len(test_patients)
            total_patients = n_train_patients + n_val_patients + n_test_patients
        else:
            n_train_patients = n_val_patients = n_test_patients = total_patients = 0

        return {
            # Stay counts
            "train_stays": len(self.train_indices),
            "full_train_stays": len(self.full_train_indices),
            "val_stays": len(self.val_indices),
            "test_stays": len(self.test_indices),
            "total_stays": total_stays,
            # Patient counts
            "train_patients": n_train_patients,
            "val_patients": n_val_patients,
            "test_patients": n_test_patients,
            "total_patients": total_patients,
            # Actual ratios (for verification)
            "actual_train_ratio": len(self.train_indices) / total_stays if total_stays > 0 else 0,
            "actual_full_train_ratio": (
                len(self.full_train_indices) / total_stays if total_stays > 0 else 0
            ),
            "actual_val_ratio": len(self.val_indices) / total_stays if total_stays > 0 else 0,
            "actual_test_ratio": len(self.test_indices) / total_stays if total_stays > 0 else 0,
        }

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return label statistics for all tasks."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_label_statistics()")
        return self.dataset.get_label_statistics()

    def get_train_label_statistics(self, use_full_train: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return label statistics for the train split.

        Args:
            use_full_train: When True, compute statistics on the full patient-level
                train split before any label-efficiency subsampling. When False,
                compute statistics on the optimization subset actually used for
                training in this run.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_train_label_statistics()")

        train_indices = self.full_train_indices if use_full_train else self.train_indices
        return self.dataset.get_label_statistics(indices=train_indices)
