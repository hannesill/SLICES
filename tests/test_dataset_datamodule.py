"""Tests for ICUDataset and ICUDataModule."""

import pytest
import torch
import yaml
import numpy as np
import polars as pl
from pathlib import Path

from slices.data.dataset import ICUDataset
from slices.data.datamodule import ICUDataModule, icu_collate_fn


@pytest.fixture
def mock_extracted_data(tmp_path):
    """Create mock extracted data files that mimic the extractor output."""
    data_dir = tmp_path / "processed"
    data_dir.mkdir(parents=True)
    
    # Create metadata
    metadata = {
        "dataset": "mock",
        "feature_set": "core",
        "feature_names": ["heart_rate", "sbp", "resp_rate"],
        "n_features": 3,
        "seq_length_hours": 48,
        "min_stay_hours": 6,
        "task_names": ["mortality_24h", "mortality_hospital"],
        "n_stays": 10,
    }
    
    with open(data_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    
    # Create static features
    static_df = pl.DataFrame({
        "stay_id": list(range(1, 11)),
        "patient_id": [100, 100, 101, 101, 102, 103, 104, 105, 106, 107],  # Some patients have multiple stays
        "age": [65, 65, 45, 45, 70, 55, 60, 75, 50, 80],
        "gender": ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
        "los_days": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0],
    })
    static_df.write_parquet(data_dir / "static.parquet")
    
    # Create timeseries (dense format with nested lists)
    # Each stay has 48 hours x 3 features
    seq_length = 48
    n_features = 3
    
    timeseries_data = []
    mask_data = []
    
    np.random.seed(42)
    for stay_id in range(1, 11):
        # Create random timeseries with some missing values
        ts = np.random.randn(seq_length, n_features) * 10 + 70  # Around 70 for heart rate
        mask = np.random.rand(seq_length, n_features) > 0.3  # ~70% observed
        
        # Set unobserved values to NaN
        ts[~mask] = float("nan")
        
        timeseries_data.append(ts.tolist())
        mask_data.append(mask.tolist())
    
    timeseries_df = pl.DataFrame({
        "stay_id": list(range(1, 11)),
        "timeseries": timeseries_data,
        "mask": mask_data,
    })
    timeseries_df.write_parquet(data_dir / "timeseries.parquet")
    
    # Create labels
    labels_df = pl.DataFrame({
        "stay_id": list(range(1, 11)),
        "mortality_24h": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "mortality_hospital": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    })
    labels_df.write_parquet(data_dir / "labels.parquet")
    
    return data_dir


class TestICUDataset:
    """Tests for ICUDataset class."""
    
    def test_initialization(self, mock_extracted_data):
        """Test dataset initialization loads data correctly."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h")
        
        assert len(dataset) == 10
        assert dataset.n_features == 3
        assert dataset.seq_length == 48
        assert dataset.feature_names == ["heart_rate", "sbp", "resp_rate"]
        assert dataset.task_names == ["mortality_24h", "mortality_hospital"]
    
    def test_initialization_no_task(self, mock_extracted_data):
        """Test dataset initialization without task_name."""
        dataset = ICUDataset(mock_extracted_data, task_name=None)
        
        assert len(dataset) == 10
        sample = dataset[0]
        assert "label" not in sample
    
    def test_invalid_task_raises_error(self, mock_extracted_data):
        """Test that invalid task name raises ValueError."""
        with pytest.raises(ValueError, match="not found in extracted data"):
            ICUDataset(mock_extracted_data, task_name="invalid_task")
    
    def test_nonexistent_directory_raises_error(self, tmp_path):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            ICUDataset(tmp_path / "nonexistent")
    
    def test_missing_metadata_raises_error(self, tmp_path):
        """Test that missing metadata file raises FileNotFoundError."""
        data_dir = tmp_path / "incomplete"
        data_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            ICUDataset(data_dir)
    
    def test_getitem_returns_correct_tensors(self, mock_extracted_data):
        """Test __getitem__ returns correct tensor shapes and types."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h")
        
        sample = dataset[0]
        
        assert "timeseries" in sample
        assert "mask" in sample
        assert "stay_id" in sample
        assert "label" in sample
        assert "static" in sample
        
        # Check shapes
        assert sample["timeseries"].shape == (48, 3)
        assert sample["mask"].shape == (48, 3)
        
        # Check types
        assert sample["timeseries"].dtype == torch.float32
        assert sample["mask"].dtype == torch.bool
        assert isinstance(sample["stay_id"], int)
        assert sample["label"].dtype == torch.float32
    
    def test_imputation_forward_fill(self, mock_extracted_data):
        """Test forward fill imputation strategy."""
        dataset = ICUDataset(
            mock_extracted_data, 
            task_name="mortality_24h",
            impute_strategy="forward_fill"
        )
        
        sample = dataset[0]
        
        # After imputation, there should be no NaN values
        assert not torch.isnan(sample["timeseries"]).any()
    
    def test_imputation_zero(self, mock_extracted_data):
        """Test zero imputation strategy."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            impute_strategy="zero",
            normalize=False,  # Disable to check raw zeros
        )
        
        sample = dataset[0]
        
        # After imputation, no NaN values
        assert not torch.isnan(sample["timeseries"]).any()
    
    def test_imputation_none(self, mock_extracted_data):
        """Test no imputation keeps NaN values."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            impute_strategy="none",
            normalize=False,
        )
        
        # Some samples might have NaN values
        sample = dataset[0]
        # With ~30% missing, there should be some NaN
        mask = sample["mask"]
        missing_count = (~mask).sum()
        assert missing_count > 0  # Should have some missing values
    
    def test_normalization(self, mock_extracted_data):
        """Test that normalization is applied correctly."""
        dataset_norm = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=True,
        )
        
        dataset_no_norm = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )
        
        # Normalized data should have different values
        sample_norm = dataset_norm[0]
        sample_no_norm = dataset_no_norm[0]
        
        # Values should be different (after normalization)
        assert not torch.allclose(sample_norm["timeseries"], sample_no_norm["timeseries"])
    
    def test_get_label_statistics(self, mock_extracted_data):
        """Test label statistics computation."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h")
        
        stats = dataset.get_label_statistics()
        
        assert "mortality_24h" in stats
        assert "mortality_hospital" in stats
        
        # Check mortality_24h stats (3 positive out of 10)
        assert stats["mortality_24h"]["total"] == 10
        assert stats["mortality_24h"]["positive"] == 3
        assert stats["mortality_24h"]["negative"] == 7
        assert stats["mortality_24h"]["prevalence"] == pytest.approx(0.3)
    
    def test_override_seq_length(self, mock_extracted_data):
        """Test that seq_length can be overridden."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            seq_length=24,  # Override to 24 hours
        )
        
        assert dataset.seq_length == 24
        sample = dataset[0]
        assert sample["timeseries"].shape == (24, 3)


class TestICUDataModule:
    """Tests for ICUDataModule class."""
    
    def test_initialization(self, mock_extracted_data):
        """Test datamodule initialization."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            batch_size=4,
        )
        
        assert dm.batch_size == 4
        assert dm.task_name == "mortality_24h"
    
    def test_setup_creates_splits(self, mock_extracted_data):
        """Test that setup() creates train/val/test splits."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            batch_size=2,
        )
        
        dm.setup()
        
        # Check splits are created
        assert len(dm.train_indices) > 0
        assert len(dm.val_indices) > 0
        assert len(dm.test_indices) > 0
        
        # Check no overlap between splits
        train_set = set(dm.train_indices)
        val_set = set(dm.val_indices)
        test_set = set(dm.test_indices)
        
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        
        # Check all indices are covered
        assert len(train_set | val_set | test_set) == len(dm.dataset)
    
    def test_patient_level_splits(self, mock_extracted_data):
        """Test that splits are patient-level (no leakage)."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            batch_size=2,
        )
        
        dm.setup()
        
        # Get patient_ids for each split
        train_patients = set()
        val_patients = set()
        test_patients = set()
        
        for idx in dm.train_indices:
            stay_id = dm.dataset.stay_ids[idx]
            patient_row = dm.dataset.static_df.filter(pl.col("stay_id") == stay_id)
            if len(patient_row) > 0:
                train_patients.add(patient_row["patient_id"][0])
        
        for idx in dm.val_indices:
            stay_id = dm.dataset.stay_ids[idx]
            patient_row = dm.dataset.static_df.filter(pl.col("stay_id") == stay_id)
            if len(patient_row) > 0:
                val_patients.add(patient_row["patient_id"][0])
        
        for idx in dm.test_indices:
            stay_id = dm.dataset.stay_ids[idx]
            patient_row = dm.dataset.static_df.filter(pl.col("stay_id") == stay_id)
            if len(patient_row) > 0:
                test_patients.add(patient_row["patient_id"][0])
        
        # No patient should appear in multiple splits
        assert train_patients.isdisjoint(val_patients)
        assert train_patients.isdisjoint(test_patients)
        assert val_patients.isdisjoint(test_patients)
    
    def test_dataloaders_return_batches(self, mock_extracted_data):
        """Test that dataloaders return properly batched data."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            batch_size=2,
            num_workers=0,  # Use main process for testing
        )
        
        dm.setup()
        
        # Get one batch from train loader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        # Check batch structure
        assert "timeseries" in batch
        assert "mask" in batch
        assert "stay_id" in batch
        assert "label" in batch
        
        # Check batch size (may be smaller due to drop_last=True and small dataset)
        assert batch["timeseries"].shape[0] <= 2
        assert batch["timeseries"].shape[1] == 48
        assert batch["timeseries"].shape[2] == 3
    
    def test_reproducible_splits(self, mock_extracted_data):
        """Test that splits are reproducible with same seed."""
        dm1 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=42,
        )
        dm1.setup()
        
        dm2 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=42,
        )
        dm2.setup()
        
        assert dm1.train_indices == dm2.train_indices
        assert dm1.val_indices == dm2.val_indices
        assert dm1.test_indices == dm2.test_indices
    
    def test_different_seeds_different_splits(self, mock_extracted_data):
        """Test that different seeds produce different splits."""
        dm1 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=42,
        )
        dm1.setup()
        
        dm2 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=123,
        )
        dm2.setup()
        
        # At least some indices should be different
        assert dm1.train_indices != dm2.train_indices or \
               dm1.val_indices != dm2.val_indices
    
    def test_get_split_info(self, mock_extracted_data):
        """Test get_split_info returns correct information."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )
        dm.setup()
        
        info = dm.get_split_info()
        
        assert "train_stays" in info
        assert "val_stays" in info
        assert "test_stays" in info
        assert "total_stays" in info
        
        assert info["total_stays"] == 10
        assert info["train_stays"] + info["val_stays"] + info["test_stays"] == 10
    
    def test_get_feature_dim(self, mock_extracted_data):
        """Test get_feature_dim returns correct dimension."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )
        dm.setup()
        
        assert dm.get_feature_dim() == 3
    
    def test_invalid_split_ratios(self):
        """Test that invalid split ratios raise ValueError."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            ICUDataModule(
                processed_dir=".",  # Doesn't matter, will fail before setup
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
            )


class TestCollateFn:
    """Tests for the collate function."""
    
    def test_collate_basic(self):
        """Test basic collation of samples."""
        samples = [
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 1,
                "label": torch.tensor(0.0),
            },
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 2,
                "label": torch.tensor(1.0),
            },
        ]
        
        batch = icu_collate_fn(samples)
        
        assert batch["timeseries"].shape == (2, 48, 3)
        assert batch["mask"].shape == (2, 48, 3)
        assert batch["stay_id"].shape == (2,)
        assert batch["label"].shape == (2,)
    
    def test_collate_without_labels(self):
        """Test collation without labels."""
        samples = [
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 1,
            },
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 2,
            },
        ]
        
        batch = icu_collate_fn(samples)
        
        assert "label" not in batch
        assert batch["timeseries"].shape == (2, 48, 3)
