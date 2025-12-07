"""Lightning DataModule for ICU data.

Handles patient-level splits, data loading, and batching for training.
Ensures no patient appears in multiple splits (prevents data leakage).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from slices.data.dataset import ICUDataset


def icu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching ICU samples.
    
    Args:
        batch: List of sample dictionaries from ICUDataset.
        
    Returns:
        Batched dictionary with stacked tensors.
    """
    # Stack tensors
    timeseries = torch.stack([s["timeseries"] for s in batch])  # (B, T, D)
    mask = torch.stack([s["mask"] for s in batch])  # (B, T, D)
    stay_ids = torch.tensor([s["stay_id"] for s in batch]) # (B,)
    
    result = {
        "timeseries": timeseries,
        "mask": mask,
        "stay_id": stay_ids,
    }
    
    # Stack labels if present
    if "label" in batch[0]:
        labels = torch.stack([s["label"] for s in batch]) # (B,)
        result["label"] = labels
    
    return result


class ICUDataModule(L.LightningDataModule):
    """Lightning DataModule for ICU data.
    
    Implements patient-level splits to prevent data leakage.
    Uses hashing of patient_id for deterministic, reproducible splits.
    
    Example:
        >>> dm = ICUDataModule(
        ...     processed_dir="data/processed/mimic-iv-demo",
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
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        normalize: bool = True,
        impute_strategy: str = "forward_fill",
        pin_memory: bool = True,
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
            impute_strategy: Imputation strategy ('forward_fill', 'zero', 'mean', 'none').
            pin_memory: Whether to pin memory for faster GPU transfer.
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
        self.impute_strategy = impute_strategy
        self.pin_memory = pin_memory
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Will be set in setup()
        self.dataset: Optional[ICUDataset] = None
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

    def _get_patient_level_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """Create patient-level train/val/test splits.
        
        Uses deterministic hashing of patient_id to assign patients to splits.
        All stays from a patient go to the same split.
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        # Load static data to get patient_id mapping
        static_df = self.dataset.static_df
        
        # Get stay_id -> patient_id mapping
        stay_to_patient = dict(zip(
            static_df["stay_id"].to_list(),
            static_df["patient_id"].to_list()
        ))
        
        # Get unique patients
        unique_patients = list(set(stay_to_patient.values()))
        n_patients = len(unique_patients)
        
        # Shuffle patients deterministically using seed
        rng = np.random.RandomState(self.seed)
        patient_indices = np.arange(n_patients)
        rng.shuffle(patient_indices)
        shuffled_patients = [unique_patients[i] for i in patient_indices]
        
        # Split patients
        n_train = int(n_patients * self.train_ratio)
        n_val = int(n_patients * self.val_ratio)
        
        train_patients = set(shuffled_patients[:n_train])
        val_patients = set(shuffled_patients[n_train:n_train + n_val])
        test_patients = set(shuffled_patients[n_train + n_val:])
        
        # Verify no patient overlap between splits (data leakage check)
        assert train_patients.isdisjoint(val_patients), \
            "Patient leakage detected: train/val splits have overlapping patients"
        assert train_patients.isdisjoint(test_patients), \
            "Patient leakage detected: train/test splits have overlapping patients"
        assert val_patients.isdisjoint(test_patients), \
            "Patient leakage detected: val/test splits have overlapping patients"
        
        # Map back to stay indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx, stay_id in enumerate(self.dataset.stay_ids):
            patient_id = stay_to_patient.get(stay_id)
            if patient_id in train_patients:
                train_indices.append(idx)
            elif patient_id in val_patients:
                val_indices.append(idx)
            else:
                test_indices.append(idx)
        
        return train_indices, val_indices, test_indices

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for train/val/test.
        
        Args:
            stage: Stage name ('fit', 'validate', 'test', or None).
        """
        # Create full dataset (loads data once)
        self.dataset = ICUDataset(
            data_dir=self.processed_dir,
            task_name=self.task_name,
            seq_length=self.seq_length,
            normalize=self.normalize,
            impute_strategy=self.impute_strategy,
        )
        
        # Create patient-level splits
        self.train_indices, self.val_indices, self.test_indices = (
            self._get_patient_level_splits()
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        
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
        """Return validation DataLoader."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        
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
            stay_to_patient = dict(zip(
                static_df["stay_id"].to_list(),
                static_df["patient_id"].to_list()
            ))
            
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
            "actual_val_ratio": len(self.val_indices) / total_stays if total_stays > 0 else 0,
            "actual_test_ratio": len(self.test_indices) / total_stays if total_stays > 0 else 0,
        }

    def get_label_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return label statistics for all tasks."""
        if self.dataset is None:
            raise RuntimeError("Call setup() before get_label_statistics()")
        return self.dataset.get_label_statistics()

