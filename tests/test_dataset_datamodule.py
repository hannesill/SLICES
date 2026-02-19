"""Tests for ICUDataset and ICUDataModule."""

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from slices.data.datamodule import ICUDataModule, icu_collate_fn
from slices.data.dataset import ICUDataset


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
    static_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "patient_id": [
                100,
                100,
                101,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
            ],  # Some patients have multiple stays
            "age": [65, 65, 45, 45, 70, 55, 60, 75, 50, 80],
            "gender": ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
            "los_days": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 2.5, 3.5, 4.5, 2.0],
        }
    )
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

    timeseries_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "timeseries": timeseries_data,
            "mask": mask_data,
        }
    )
    timeseries_df.write_parquet(data_dir / "timeseries.parquet")

    # Create labels
    labels_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "mortality_24h": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            "mortality_hospital": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        }
    )
    labels_df.write_parquet(data_dir / "labels.parquet")

    return data_dir


class TestICUDataset:
    """Tests for ICUDataset class."""

    def test_initialization(self, mock_extracted_data):
        """Test dataset initialization loads data correctly."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        assert len(dataset) == 10
        assert dataset.n_features == 3
        assert dataset.seq_length == 48
        assert dataset.feature_names == ["heart_rate", "sbp", "resp_rate"]
        assert dataset.task_names == ["mortality_24h", "mortality_hospital"]

    def test_initialization_no_task(self, mock_extracted_data):
        """Test dataset initialization without task_name."""
        dataset = ICUDataset(mock_extracted_data, task_name=None, normalize=False)

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
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

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

    def test_zero_fill_after_preprocessing(self, mock_extracted_data):
        """Test that zero-fill removes all NaN values."""
        dataset = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=False,
        )

        sample = dataset[0]

        # After zero-fill, there should be no NaN values
        assert not torch.isnan(sample["timeseries"]).any()

        # Missing positions (mask=False) should be zero
        mask = sample["mask"]
        missing_values = sample["timeseries"][~mask]
        if len(missing_values) > 0:
            assert (missing_values == 0.0).all()

    def test_normalization(self, mock_extracted_data):
        """Test that normalization is applied correctly."""
        # Need to provide train_indices when normalize=True to prevent data leakage
        train_indices = list(range(7))  # First 7 samples as training

        dataset_norm = ICUDataset(
            mock_extracted_data,
            task_name="mortality_24h",
            normalize=True,
            train_indices=train_indices,
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
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

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
            normalize=False,
        )

        assert dataset.seq_length == 24
        sample = dataset[0]
        assert sample["timeseries"].shape == (24, 3)

    def test_get_feature_names(self, mock_extracted_data):
        """Test get_feature_names returns correct list."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        feature_names = dataset.get_feature_names()

        assert feature_names == ["heart_rate", "sbp", "resp_rate"]
        assert len(feature_names) == 3

    def test_get_task_names(self, mock_extracted_data):
        """Test get_task_names returns available tasks."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        task_names = dataset.get_task_names()

        assert "mortality_24h" in task_names
        assert "mortality_hospital" in task_names

    def test_static_features_in_sample(self, mock_extracted_data):
        """Test that static features are included in sample."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        sample = dataset[0]

        assert "static" in sample
        assert "age" in sample["static"]
        assert "gender" in sample["static"]
        assert "los_days" in sample["static"]

    def test_normalize_without_train_indices_raises_error(self, mock_extracted_data):
        """Test that normalize=True without train_indices raises ValueError (Issue #4 fix).

        This prevents data leakage by ensuring normalization statistics are only
        computed from training data, not from all data including val/test sets.
        """
        with pytest.raises(ValueError, match="train_indices must be provided when normalize=True"):
            ICUDataset(
                mock_extracted_data,
                task_name="mortality_24h",
                normalize=True,
                train_indices=None,  # This should raise an error
            )

    def test_all_samples_accessible(self, mock_extracted_data):
        """Test that all samples can be accessed without error."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        for i in range(len(dataset)):
            sample = dataset[i]
            assert "timeseries" in sample
            assert "mask" in sample
            assert "stay_id" in sample

    def test_single_sample_dataset(self, tmp_path):
        """Test dataset with single sample works correctly."""
        data_dir = tmp_path / "single_sample"
        data_dir.mkdir(parents=True)

        # Create metadata
        metadata = {
            "dataset": "mock",
            "feature_set": "core",
            "feature_names": ["heart_rate"],
            "n_features": 1,
            "seq_length_hours": 10,
            "min_stay_hours": 1,
            "task_names": ["mortality_24h"],
            "n_stays": 1,
        }

        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create single-sample data
        import numpy as np

        static_df = pl.DataFrame(
            {
                "stay_id": [1],
                "patient_id": [100],
                "age": [65],
                "gender": ["M"],
                "los_days": [2.0],
            }
        )
        static_df.write_parquet(data_dir / "static.parquet")

        ts = np.random.randn(10, 1).tolist()
        mask = [[True] for _ in range(10)]
        timeseries_df = pl.DataFrame(
            {
                "stay_id": [1],
                "timeseries": [ts],
                "mask": [mask],
            }
        )
        timeseries_df.write_parquet(data_dir / "timeseries.parquet")

        labels_df = pl.DataFrame(
            {
                "stay_id": [1],
                "mortality_24h": [0],
            }
        )
        labels_df.write_parquet(data_dir / "labels.parquet")

        # Load and test (normalize=False since we don't have train_indices)
        dataset = ICUDataset(data_dir, task_name="mortality_24h", normalize=False)

        assert len(dataset) == 1
        sample = dataset[0]
        assert sample["timeseries"].shape == (10, 1)

    def test_missing_labels_filter_default_behavior(self, tmp_path):
        """Test that missing labels are filtered by default (handle_missing_labels='filter').

        This tests the fix for Issue #5: Missing labels should not silently become NaN.
        Instead, samples with missing labels should be removed with a warning.
        """
        data_dir = tmp_path / "missing_labels"
        data_dir.mkdir(parents=True)

        # Create metadata
        metadata = {
            "dataset": "mock",
            "feature_set": "core",
            "feature_names": ["heart_rate"],
            "n_features": 1,
            "seq_length_hours": 10,
            "min_stay_hours": 1,
            "task_names": ["mortality_24h"],
            "n_stays": 5,
        }

        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create static data
        static_df = pl.DataFrame(
            {
                "stay_id": list(range(1, 6)),
                "patient_id": [100, 101, 102, 103, 104],
                "age": [65, 45, 70, 55, 60],
                "gender": ["M", "F", "M", "F", "M"],
                "los_days": [2.0, 3.0, 2.5, 3.5, 2.0],
            }
        )
        static_df.write_parquet(data_dir / "static.parquet")

        # Create timeseries for all 5 stays
        ts_data = [np.random.randn(10, 1).tolist() for _ in range(5)]
        mask_data = [[[True] for _ in range(10)] for _ in range(5)]
        timeseries_df = pl.DataFrame(
            {
                "stay_id": list(range(1, 6)),
                "timeseries": ts_data,
                "mask": mask_data,
            }
        )
        timeseries_df.write_parquet(data_dir / "timeseries.parquet")

        # Create labels with missing values for stays 3 and 5
        labels_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "mortality_24h": [0.0, 1.0, None, 0.0, None],  # Stays 3 and 5 missing
            }
        )
        labels_df.write_parquet(data_dir / "labels.parquet")

        # Load dataset with default handle_missing_labels='filter'
        dataset = ICUDataset(data_dir, task_name="mortality_24h", normalize=False)

        # Should have only 3 samples (stay_ids 1, 2, 4)
        assert len(dataset) == 3
        assert set(dataset.stay_ids) == {1, 2, 4}

        # Should have tracked removed samples
        assert len(dataset.removed_samples) == 2
        assert (3, "missing_mortality_24h_label") in dataset.removed_samples
        assert (5, "missing_mortality_24h_label") in dataset.removed_samples

        # Should have saved metadata
        metadata_file = data_dir / "dataset_metadata.yaml"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            saved_metadata = yaml.safe_load(f)

        assert saved_metadata["removed_samples_count"] == 2
        assert saved_metadata["task_name"] == "mortality_24h"
        assert saved_metadata["handle_missing_labels"] == "filter"

    def test_missing_labels_raise_error(self, tmp_path):
        """Test that missing labels raise ValueError when handle_missing_labels='raise'.

        This allows strict validation if users want to ensure no data loss.
        """
        data_dir = tmp_path / "missing_labels_strict"
        data_dir.mkdir(parents=True)

        # Create metadata
        metadata = {
            "dataset": "mock",
            "feature_set": "core",
            "feature_names": ["heart_rate"],
            "n_features": 1,
            "seq_length_hours": 10,
            "min_stay_hours": 1,
            "task_names": ["mortality_24h"],
            "n_stays": 3,
        }

        with open(data_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create static data
        static_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 101, 102],
                "age": [65, 45, 70],
                "gender": ["M", "F", "M"],
                "los_days": [2.0, 3.0, 2.5],
            }
        )
        static_df.write_parquet(data_dir / "static.parquet")

        # Create timeseries
        ts_data = [np.random.randn(10, 1).tolist() for _ in range(3)]
        mask_data = [[[True] for _ in range(10)] for _ in range(3)]
        timeseries_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "timeseries": ts_data,
                "mask": mask_data,
            }
        )
        timeseries_df.write_parquet(data_dir / "timeseries.parquet")

        # Create labels with missing value
        labels_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "mortality_24h": [0.0, 1.0, None],  # Stay 3 missing
            }
        )
        labels_df.write_parquet(data_dir / "labels.parquet")

        # Should raise ValueError when handle_missing_labels='raise'
        with pytest.raises(ValueError, match="Missing labels for 1 stays"):
            ICUDataset(
                data_dir, task_name="mortality_24h", handle_missing_labels="raise", normalize=False
            )

    def test_no_nan_labels_in_output(self, mock_extracted_data):
        """Verify that labels are never NaN in the output (core fix for Issue #5).

        This is the critical assertion: with the fix, we should NEVER see NaN labels,
        which would corrupt gradient computation during training.
        """
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        # Check all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample["label"]

            # Label should never be NaN
            assert not torch.isnan(label).any(), (
                f"Sample {i} has NaN label - this would cause silent training failure. "
                "Labels should be filtered out or raise an error before reaching this point."
            )

            # Label should be valid 0 or 1
            assert label.item() in (0.0, 1.0), f"Label should be 0 or 1, got {label.item()}"

    def test_all_samples_have_labels_when_task_specified(self, mock_extracted_data):
        """Test that all loaded samples have labels when task_name is specified."""
        dataset = ICUDataset(mock_extracted_data, task_name="mortality_24h", normalize=False)

        # Should have same number of labels as samples (stacked tensor)
        assert dataset._labels_tensor is not None
        assert len(dataset._labels_tensor) == len(dataset.stay_ids)

        # All labels should be valid tensors with correct dtype and no NaN
        assert isinstance(dataset._labels_tensor, torch.Tensor)
        assert dataset._labels_tensor.dtype == torch.float32
        assert not torch.isnan(dataset._labels_tensor).any()


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
        assert dm1.train_indices != dm2.train_indices or dm1.val_indices != dm2.val_indices

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

    def test_train_dataloader_before_setup_raises_error(self, mock_extracted_data):
        """Test that calling train_dataloader before setup raises RuntimeError."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        with pytest.raises(RuntimeError, match="Call setup"):
            dm.train_dataloader()

    def test_val_dataloader_before_setup_raises_error(self, mock_extracted_data):
        """Test that calling val_dataloader before setup raises RuntimeError."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        with pytest.raises(RuntimeError, match="Call setup"):
            dm.val_dataloader()

    def test_test_dataloader_before_setup_raises_error(self, mock_extracted_data):
        """Test that calling test_dataloader before setup raises RuntimeError."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        with pytest.raises(RuntimeError, match="Call setup"):
            dm.test_dataloader()

    def test_get_feature_dim_before_setup_raises_error(self, mock_extracted_data):
        """Test that calling get_feature_dim before setup raises RuntimeError."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        with pytest.raises(RuntimeError, match="Call setup"):
            dm.get_feature_dim()

    def test_get_seq_length(self, mock_extracted_data):
        """Test get_seq_length returns correct value."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )
        dm.setup()

        assert dm.get_seq_length() == 48

    def test_get_label_statistics(self, mock_extracted_data):
        """Test get_label_statistics returns correct information."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )
        dm.setup()

        stats = dm.get_label_statistics()

        assert "mortality_24h" in stats
        assert "mortality_hospital" in stats
        assert "total" in stats["mortality_24h"]
        assert "positive" in stats["mortality_24h"]
        assert "prevalence" in stats["mortality_24h"]

    def test_custom_split_ratios(self, mock_extracted_data):
        """Test custom split ratios are applied."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        dm.setup()

        info = dm.get_split_info()

        # Check that splits are approximately correct (accounting for patient-level)
        total = info["total_stays"]
        assert info["train_stays"] + info["val_stays"] + info["test_stays"] == total

    def test_num_workers_configuration(self, mock_extracted_data):
        """Test that num_workers is properly configured."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            num_workers=2,
        )

        assert dm.num_workers == 2

    def test_pin_memory_configuration(self, mock_extracted_data):
        """Test that pin_memory is properly configured."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            pin_memory=False,
        )

        assert dm.pin_memory is False

    def test_setup_with_stage_fit(self, mock_extracted_data):
        """Test setup with stage='fit'."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        dm.setup(stage="fit")

        # Should still create all splits
        assert len(dm.train_indices) > 0
        assert len(dm.val_indices) > 0
        assert len(dm.test_indices) > 0

    def test_setup_with_stage_test(self, mock_extracted_data):
        """Test setup with stage='test'."""
        dm = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
        )

        dm.setup(stage="test")

        # Should still have test indices
        assert len(dm.test_indices) > 0

    def test_cached_splits_load_correctly(self, mock_extracted_data):
        """Test that cached splits with patient lists are saved and loaded correctly (Issue #2 fix).

        This verifies that:
        1. First setup creates and saves splits with full patient lists
        2. Second setup with same seed loads cached splits
        3. The loaded splits match the original
        """
        # First setup - creates splits
        dm1 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=12345,
        )
        dm1.setup()

        # Verify splits.yaml was created with patient lists
        splits_path = mock_extracted_data / "splits.yaml"
        assert splits_path.exists(), "splits.yaml should be created"

        with open(splits_path) as f:
            saved_splits = yaml.safe_load(f)

        # Check that patient lists are saved (not just counts)
        assert "train_patients" in saved_splits, "train_patients list should be saved"
        assert "val_patients" in saved_splits, "val_patients list should be saved"
        assert "test_patients" in saved_splits, "test_patients list should be saved"
        assert isinstance(saved_splits["train_patients"], list), "train_patients should be a list"

        # Second setup with same seed - should load cached splits
        dm2 = ICUDataModule(
            processed_dir=mock_extracted_data,
            task_name="mortality_24h",
            seed=12345,
        )
        dm2.setup()

        # Verify splits match
        assert dm1.train_indices == dm2.train_indices, "Train indices should match after reload"
        assert dm1.val_indices == dm2.val_indices, "Val indices should match after reload"
        assert dm1.test_indices == dm2.test_indices, "Test indices should match after reload"


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

    def test_normalization_uses_only_train_set(self, mock_extracted_data):
        """Verify normalization stats exclude val/test data (prevents data leakage).

        This is a CRITICAL test for the normalization leakage fix.
        The issue was that normalization statistics were computed over ALL data
        before splits were created, causing test set information to leak into training.
        """
        data_dir = mock_extracted_data

        # Verify that the DataModule properly passes train_indices to the dataset
        dm = ICUDataModule(
            processed_dir=data_dir,
            task_name="mortality_24h",
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            normalize=True,
        )
        dm.setup()

        # The dataset should have been created with train_indices
        assert dm.dataset is not None
        assert len(dm.train_indices) > 0
        assert dm.dataset.train_indices is not None, (
            "Dataset should have received train_indices from DataModule. "
            "This is the core of the normalization leakage fix."
        )
        assert set(dm.dataset.train_indices) == set(
            dm.train_indices
        ), "Dataset train_indices should match DataModule train_indices"

        # Verify no patient overlap (existing check)
        train_patients = {
            dm.dataset.static_df.filter(pl.col("stay_id") == dm.dataset.stay_ids[i])[
                "patient_id"
            ].item()
            for i in dm.train_indices
        }
        val_patients = {
            dm.dataset.static_df.filter(pl.col("stay_id") == dm.dataset.stay_ids[i])[
                "patient_id"
            ].item()
            for i in dm.val_indices
        }
        test_patients = {
            dm.dataset.static_df.filter(pl.col("stay_id") == dm.dataset.stay_ids[i])[
                "patient_id"
            ].item()
            for i in dm.test_indices
        }

        assert (
            train_patients.isdisjoint(val_patients)
            and val_patients.isdisjoint(test_patients)
            and train_patients.isdisjoint(test_patients)
        ), "Patient leakage between splits"

        # Verify that normalization is applied correctly to val/test data
        # using training statistics (not val/test statistics)
        sample_train = dm.dataset[dm.train_indices[0]]
        sample_val = dm.dataset[dm.val_indices[0]]

        # Both should have normalized values (within reasonable range)
        assert torch.all(
            ~torch.isnan(sample_train["timeseries"])
        ), "Training sample should not have NaN after normalization"
        assert torch.all(
            ~torch.isnan(sample_val["timeseries"])
        ), "Validation sample should not have NaN after normalization"

        # Both should have reasonable normalized values
        # After z-score normalization with training stats, test values
        # could be outside [-3, 3] if they're outliers, but shouldn't be extreme
        assert (
            torch.max(torch.abs(sample_val["timeseries"][sample_val["mask"]])) < 100
        ), "Normalized values seem too extreme; normalization might not be working"

    def test_normalization_stats_persistence(self, mock_extracted_data):
        """Verify normalization statistics are saved and reloaded for reproducibility.

        This test ensures that:
        1. Normalization stats are computed and saved to normalization_stats.yaml
        2. When dataset is reloaded, it uses cached stats instead of recomputing
        3. The cached stats produce identical results
        """
        data_dir = mock_extracted_data

        # Create first dataset (should compute and save stats)
        dataset1 = ICUDataset(
            data_dir=data_dir,
            task_name="mortality_24h",
            normalize=True,
            train_indices=[0, 1, 2, 3],  # Only first 4 samples are training
        )

        stats_file = data_dir / "normalization_stats.yaml"
        assert stats_file.exists(), "normalization_stats.yaml should be created"

        # Get stats from first dataset
        means1 = dataset1.feature_means.clone()
        stds1 = dataset1.feature_stds.clone()

        # Create second dataset (should load cached stats)
        dataset2 = ICUDataset(
            data_dir=data_dir,
            task_name="mortality_24h",
            normalize=True,
            train_indices=[0, 1, 2, 3],  # Same training indices
        )

        # Stats should be identical (loaded from cache)
        means2 = dataset2.feature_means
        stds2 = dataset2.feature_stds

        assert torch.allclose(
            means1, means2, rtol=1e-5
        ), "Feature means should match when using cached stats"
        assert torch.allclose(
            stds1, stds2, rtol=1e-5
        ), "Feature stds should match when using cached stats"

        # Verify that samples are identically normalized in both datasets
        sample1_data = dataset1[0]
        sample2_data = dataset2[0]

        assert torch.allclose(
            sample1_data["timeseries"], sample2_data["timeseries"], rtol=1e-5
        ), "Samples should be identically normalized"
        assert torch.allclose(
            sample1_data["mask"].float(), sample2_data["mask"].float()
        ), "Masks should be identical"
