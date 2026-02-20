"""Lightning DataModule for decompensation prediction.

Implements patient-level splits with sliding windows over full stays.
Different strides for training (sparse) and evaluation (dense).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from slices.data.decompensation_dataset import DecompensationDataset

logger = logging.getLogger(__name__)


def decompensation_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function for decompensation samples."""
    return {
        "timeseries": torch.stack([s["timeseries"] for s in batch]),
        "mask": torch.stack([s["mask"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
        "stay_id": torch.stack([s["stay_id"] for s in batch]),
        "window_start": torch.stack([s["window_start"] for s in batch]),
    }


class DecompensationDataModule(L.LightningDataModule):
    """DataModule for decompensation with patient-level splits.

    Patient-level splits ensure ALL windows from the same patient
    are in the same split (no leakage). Computes normalization stats
    from training windows only.

    Uses stride_hours for training (default 6) and eval_stride_hours
    for val/test (default 1 for fine-grained evaluation).
    """

    def __init__(
        self,
        ricu_parquet_root: Path,
        processed_dir: Path,
        batch_size: int = 64,
        num_workers: int = 4,
        obs_window_hours: int = 48,
        pred_window_hours: int = 24,
        stride_hours: int = 6,
        eval_stride_hours: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        pin_memory: bool = True,
    ) -> None:
        """Initialize DataModule.

        Args:
            ricu_parquet_root: Path to RICU output directory containing
                ricu_timeseries.parquet, ricu_stays.parquet, ricu_mortality.parquet.
            processed_dir: Path to processed directory (for metadata/splits cache).
            batch_size: Batch size for all dataloaders.
            num_workers: Number of data loading workers.
            obs_window_hours: Observation window size in hours.
            pred_window_hours: Prediction window after observation.
            stride_hours: Stride for training windows.
            eval_stride_hours: Stride for val/test windows (finer grained).
            train_ratio: Fraction of patients for training.
            val_ratio: Fraction of patients for validation.
            test_ratio: Fraction of patients for testing.
            seed: Random seed for reproducible splits.
            pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()
        self.ricu_parquet_root = Path(ricu_parquet_root)
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.obs_window_hours = obs_window_hours
        self.pred_window_hours = pred_window_hours
        self.stride_hours = stride_hours
        self.eval_stride_hours = eval_stride_hours
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.pin_memory = pin_memory

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Set in setup()
        self.train_dataset: Optional[DecompensationDataset] = None
        self.val_dataset: Optional[DecompensationDataset] = None
        self.test_dataset: Optional[DecompensationDataset] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_means: Optional[torch.Tensor] = None
        self._feature_stds: Optional[torch.Tensor] = None
        self._train_ids: List[int] = []
        self._val_ids: List[int] = []
        self._test_ids: List[int] = []
        self._n_train_patients: int = 0
        self._n_val_patients: int = 0
        self._n_test_patients: int = 0

    def _get_feature_names(self) -> List[str]:
        """Load feature names from RICU metadata."""
        metadata_path = self.ricu_parquet_root / "ricu_metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        return metadata["feature_names"]

    def _get_patient_level_splits(
        self, stays_df: pl.DataFrame
    ) -> Tuple[List[int], List[int], List[int], int, int, int]:
        """Split stays into train/val/test by patient.

        Args:
            stays_df: DataFrame with stay_id and patient_id columns.

        Returns:
            Tuple of (train_stay_ids, val_stay_ids, test_stay_ids,
                       n_train_patients, n_val_patients, n_test_patients).
        """
        stay_to_patient = dict(zip(stays_df["stay_id"].to_list(), stays_df["patient_id"].to_list()))

        unique_patients = list(set(stay_to_patient.values()))
        n_patients = len(unique_patients)

        # Shuffle deterministically
        rng = np.random.RandomState(self.seed)
        indices = np.arange(n_patients)
        rng.shuffle(indices)
        shuffled = [unique_patients[i] for i in indices]

        n_train = int(n_patients * self.train_ratio)
        n_val = int(n_patients * self.val_ratio)

        train_patients = set(shuffled[:n_train])
        val_patients = set(shuffled[n_train : n_train + n_val])
        remaining_patients = set(shuffled[n_train + n_val :])

        # Map back to stay_ids
        train_ids, val_ids, test_ids = [], [], []
        for sid, pid in stay_to_patient.items():
            if pid in train_patients:
                train_ids.append(sid)
            elif pid in val_patients:
                val_ids.append(sid)
            elif pid in remaining_patients:
                test_ids.append(sid)

        return (
            train_ids,
            val_ids,
            test_ids,
            len(train_patients),
            len(val_patients),
            len(remaining_patients),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets with patient-level splits."""
        if self.train_dataset is not None:
            return  # Already set up

        logger.info("Setting up DecompensationDataModule")

        # Load feature names
        self._feature_names = self._get_feature_names()

        # Load stays for splitting
        stays_path = self.ricu_parquet_root / "ricu_stays.parquet"
        stays_df = pl.read_parquet(stays_path)

        # Get patient-level splits
        train_ids, val_ids, test_ids, n_train_p, n_val_p, n_test_p = self._get_patient_level_splits(
            stays_df
        )
        self._train_ids = train_ids
        self._val_ids = val_ids
        self._test_ids = test_ids
        self._n_train_patients = n_train_p
        self._n_val_patients = n_val_p
        self._n_test_patients = n_test_p
        logger.info(
            f"Patient splits: {len(train_ids)} train, "
            f"{len(val_ids)} val, {len(test_ids)} test stays"
        )

        ts_path = self.ricu_parquet_root / "ricu_timeseries.parquet"
        mortality_path = self.ricu_parquet_root / "ricu_mortality.parquet"

        # Create train dataset (without normalization first to compute stats)
        self.train_dataset = DecompensationDataset(
            ricu_timeseries_path=ts_path,
            stays_path=stays_path,
            mortality_path=mortality_path,
            feature_names=self._feature_names,
            obs_window_hours=self.obs_window_hours,
            pred_window_hours=self.pred_window_hours,
            stride_hours=self.stride_hours,
            stay_ids=train_ids,
            normalize=False,  # Compute stats first
        )

        # Compute normalization stats from training data
        self._feature_means, self._feature_stds = self.train_dataset.compute_normalization_stats()

        # Apply normalization to train dataset
        self.train_dataset.normalize = True
        self.train_dataset.feature_means = self._feature_means
        self.train_dataset.feature_stds = self._feature_stds

        # Create val/test datasets with train normalization stats
        self.val_dataset = DecompensationDataset(
            ricu_timeseries_path=ts_path,
            stays_path=stays_path,
            mortality_path=mortality_path,
            feature_names=self._feature_names,
            obs_window_hours=self.obs_window_hours,
            pred_window_hours=self.pred_window_hours,
            stride_hours=self.eval_stride_hours,
            stay_ids=val_ids,
            normalize=True,
            feature_means=self._feature_means,
            feature_stds=self._feature_stds,
        )

        self.test_dataset = DecompensationDataset(
            ricu_timeseries_path=ts_path,
            stays_path=stays_path,
            mortality_path=mortality_path,
            feature_names=self._feature_names,
            obs_window_hours=self.obs_window_hours,
            pred_window_hours=self.pred_window_hours,
            stride_hours=self.eval_stride_hours,
            stay_ids=test_ids,
            normalize=True,
            feature_means=self._feature_means,
            feature_stds=self._feature_stds,
        )

        logger.info(
            f"Decompensation samples: {len(self.train_dataset)} train, "
            f"{len(self.val_dataset)} val, {len(self.test_dataset)} test"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=decompensation_collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=decompensation_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before test_dataloader()")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=decompensation_collate_fn,
        )

    def get_feature_dim(self) -> int:
        """Return number of input features."""
        if self._feature_names is None:
            raise RuntimeError("Call setup() before get_feature_dim()")
        return len(self._feature_names)

    def get_seq_length(self) -> int:
        """Return observation window size (sequence length for encoder)."""
        return self.obs_window_hours

    def get_split_info(self) -> Dict[str, Any]:
        """Return split information."""
        return {
            "train_patients": self._n_train_patients,
            "val_patients": self._n_val_patients,
            "test_patients": self._n_test_patients,
            "total_patients": self._n_train_patients + self._n_val_patients + self._n_test_patients,
            "train_stays": len(self._train_ids),
            "val_stays": len(self._val_ids),
            "test_stays": len(self._test_ids),
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
        }

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return label distribution for all splits."""
        stats = {}
        if self.train_dataset:
            stats["decompensation"] = self.train_dataset.get_label_distribution()
        return stats
