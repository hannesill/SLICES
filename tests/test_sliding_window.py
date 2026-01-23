"""Tests for SlidingWindowDataset."""

import numpy as np
import polars as pl
import pytest
import torch
import yaml
from slices.data.datamodule import ICUDataModule, icu_collate_fn
from slices.data.dataset import ICUDataset
from slices.data.sliding_window import SlidingWindowDataset


@pytest.fixture
def mock_long_sequence_data(tmp_path):
    """Create mock extracted data with longer sequences (168h) for sliding window tests."""
    data_dir = tmp_path / "processed_168h"
    data_dir.mkdir(parents=True)

    # Create metadata for 168h sequences
    metadata = {
        "dataset": "mock",
        "feature_set": "core",
        "feature_names": ["heart_rate", "sbp", "resp_rate"],
        "n_features": 3,
        "seq_length_hours": 168,  # 7 days
        "min_stay_hours": 48,
        "task_names": ["mortality_24h"],
        "n_stays": 10,
    }

    with open(data_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    # Create static features (multiple patients with multiple stays)
    static_df = pl.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "patient_id": [
                100,
                100,  # Patient 100 has 2 stays
                101,
                101,  # Patient 101 has 2 stays
                102,  # Patient 102 has 1 stay
                103,  # Patient 103 has 1 stay
                104,  # Patient 104 has 1 stay
                105,  # Patient 105 has 1 stay
                106,  # Patient 106 has 1 stay
                107,  # Patient 107 has 1 stay
            ],
            "age": [65, 65, 45, 45, 70, 55, 60, 75, 50, 80],
            "gender": ["M", "M", "F", "F", "M", "F", "M", "F", "M", "F"],
            "los_days": [7.0, 8.0, 7.0, 9.0, 7.0, 8.0, 7.5, 8.5, 9.0, 7.0],
        }
    )
    static_df.write_parquet(data_dir / "static.parquet")

    # Create timeseries (dense format with nested lists)
    # Each stay has 168 hours x 3 features
    seq_length = 168
    n_features = 3

    timeseries_data = []
    mask_data = []

    np.random.seed(42)
    for stay_id in range(1, 11):
        # Create random timeseries with some missing values
        ts = np.random.randn(seq_length, n_features) * 10 + 70
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
        }
    )
    labels_df.write_parquet(data_dir / "labels.parquet")

    return data_dir


class TestSlidingWindowDataset:
    """Tests for SlidingWindowDataset class."""

    def test_initialization(self, mock_long_sequence_data):
        """Test sliding window dataset initialization."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        assert windowed.window_size == 48
        assert windowed.stride == 24
        assert windowed.n_features == 3
        assert windowed.seq_length == 48  # Window size is the effective seq_length

    def test_window_count_calculation(self, mock_long_sequence_data):
        """Test correct window count is calculated.

        With 168h sequences, 48h windows, and 24h stride:
        - Each stay can have: (168 - 48) / 24 + 1 = 6 windows
        - Total: 10 stays * 6 windows = 60 windows
        """
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        # With 168h sequences, 48h windows, 24h stride
        # Each stay gets: floor((168-48)/24) + 1 = floor(120/24) + 1 = 5 + 1 = 6 windows
        expected_windows_per_stay = 6
        expected_total = len(base_dataset) * expected_windows_per_stay

        assert len(windowed) == expected_total
        assert len(windowed) == 60

    def test_window_count_non_overlapping(self, mock_long_sequence_data):
        """Test window count with non-overlapping windows (stride=window_size)."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=48,  # Non-overlapping
        )

        # With 168h sequences, 48h windows, 48h stride
        # Each stay gets: floor((168-48)/48) + 1 = floor(120/48) + 1 = 2 + 1 = 3 windows
        expected_windows_per_stay = 3
        expected_total = len(base_dataset) * expected_windows_per_stay

        assert len(windowed) == expected_total
        assert len(windowed) == 30

    def test_window_tensor_shapes(self, mock_long_sequence_data):
        """Test that windowed samples have correct tensor shapes."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        sample = windowed[0]

        assert sample["timeseries"].shape == (48, 3)
        assert sample["mask"].shape == (48, 3)
        assert isinstance(sample["stay_id"], int)
        assert isinstance(sample["window_start"], int)
        assert isinstance(sample["window_idx"], int)

    def test_window_metadata_fields(self, mock_long_sequence_data):
        """Test that window metadata fields are correct."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        # First window of first stay
        sample0 = windowed[0]
        assert sample0["window_start"] == 0
        assert sample0["window_idx"] == 0

        # Second window of first stay
        sample1 = windowed[1]
        assert sample1["window_start"] == 24
        assert sample1["window_idx"] == 1
        assert sample1["stay_id"] == sample0["stay_id"]  # Same stay

    def test_patient_level_isolation(self, mock_long_sequence_data):
        """Test that stay_indices parameter respects patient-level splits.

        Windows from different stays should never be mixed when using stay_indices.
        """
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)

        # Only use stays 0, 1, 2 (indices)
        stay_indices = [0, 1, 2]
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
            stay_indices=stay_indices,
        )

        # Get all stay_ids from windowed dataset
        window_stay_ids = set()
        for i in range(len(windowed)):
            sample = windowed[i]
            window_stay_ids.add(sample["stay_id"])

        # Should only contain stay_ids from the specified indices
        expected_stay_ids = {base_dataset.stay_ids[i] for i in stay_indices}
        assert window_stay_ids == expected_stay_ids

    def test_default_stride(self, mock_long_sequence_data):
        """Test that default stride is window_size // 2."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            # stride not specified, should default to 24
        )

        assert windowed.stride == 24

    def test_invalid_window_size_raises_error(self, mock_long_sequence_data):
        """Test that window_size larger than seq_length raises error."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)

        with pytest.raises(ValueError, match="cannot exceed base dataset"):
            SlidingWindowDataset(
                base_dataset,
                window_size=200,  # Larger than 168
                stride=24,
            )

    def test_invalid_window_size_zero(self, mock_long_sequence_data):
        """Test that window_size of 0 raises error."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)

        with pytest.raises(ValueError, match="window_size must be positive"):
            SlidingWindowDataset(
                base_dataset,
                window_size=0,
                stride=24,
            )

    def test_invalid_stride_zero(self, mock_long_sequence_data):
        """Test that stride of 0 raises error."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)

        with pytest.raises(ValueError, match="stride must be positive"):
            SlidingWindowDataset(
                base_dataset,
                window_size=48,
                stride=0,
            )

    def test_window_content_correctness(self, mock_long_sequence_data):
        """Test that windows extract correct data from base dataset."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None, normalize=False)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        # Get base sample
        base_sample = base_dataset[0]

        # Get first window of first stay
        window0 = windowed[0]
        assert torch.allclose(
            window0["timeseries"],
            base_sample["timeseries"][:48],
            equal_nan=True,
        )

        # Get second window of first stay (starts at position 24)
        window1 = windowed[1]
        assert torch.allclose(
            window1["timeseries"],
            base_sample["timeseries"][24:72],
            equal_nan=True,
        )

    def test_get_windows_per_stay(self, mock_long_sequence_data):
        """Test get_windows_per_stay returns correct counts."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        windows_per_stay = windowed.get_windows_per_stay()

        # Each stay should have 6 windows
        for count in windows_per_stay.values():
            assert count == 6

    def test_get_window_count_statistics(self, mock_long_sequence_data):
        """Test get_window_count_statistics returns correct stats."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        stats = windowed.get_window_count_statistics()

        assert stats["min"] == 6
        assert stats["max"] == 6
        assert stats["mean"] == 6.0
        assert stats["total"] == 60

    def test_feature_names_passthrough(self, mock_long_sequence_data):
        """Test that feature names are accessible through windowed dataset."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        assert windowed.get_feature_names() == base_dataset.get_feature_names()

    def test_index_out_of_range(self, mock_long_sequence_data):
        """Test that out of range index raises error."""
        base_dataset = ICUDataset(mock_long_sequence_data, task_name=None)
        windowed = SlidingWindowDataset(
            base_dataset,
            window_size=48,
            stride=24,
        )

        with pytest.raises(IndexError):
            _ = windowed[len(windowed)]

        with pytest.raises(IndexError):
            _ = windowed[-len(windowed) - 1]


class TestDataModuleSlidingWindows:
    """Tests for ICUDataModule with sliding windows enabled."""

    def test_datamodule_sliding_windows_disabled_by_default(self, mock_long_sequence_data):
        """Test that sliding windows are disabled by default."""
        dm = ICUDataModule(
            processed_dir=mock_long_sequence_data,
            task_name=None,
            batch_size=4,
        )
        dm.setup()

        # Without sliding windows, train dataset should be a Subset
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Should be full sequence length (168h)
        assert batch["timeseries"].shape[1] == 168
        assert "window_start" not in batch

    def test_datamodule_sliding_windows_enabled(self, mock_long_sequence_data):
        """Test DataModule with sliding windows enabled."""
        dm = ICUDataModule(
            processed_dir=mock_long_sequence_data,
            task_name=None,
            batch_size=4,
            enable_sliding_windows=True,
            window_size=48,
            window_stride=24,
        )
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Should be windowed (48h)
        assert batch["timeseries"].shape[1] == 48
        assert "window_start" in batch
        assert "window_idx" in batch

    def test_datamodule_train_has_overlapping_windows(self, mock_long_sequence_data):
        """Test that training uses overlapping windows."""
        dm = ICUDataModule(
            processed_dir=mock_long_sequence_data,
            task_name=None,
            batch_size=4,
            enable_sliding_windows=True,
            window_size=48,
            window_stride=24,  # 50% overlap
        )
        dm.setup()

        train_loader = dm.train_dataloader()

        # Count training samples
        train_sample_count = len(train_loader.dataset)

        # With overlapping windows, should have more samples than stays
        train_stay_count = len(dm.train_indices)
        assert train_sample_count > train_stay_count

    def test_datamodule_val_has_non_overlapping_windows(self, mock_long_sequence_data):
        """Test that validation uses non-overlapping windows."""
        dm = ICUDataModule(
            processed_dir=mock_long_sequence_data,
            task_name=None,
            batch_size=4,
            enable_sliding_windows=True,
            window_size=48,
            window_stride=24,  # Ignored for validation
        )
        dm.setup()

        val_loader = dm.val_dataloader()

        # For validation, stride should equal window_size (non-overlapping)
        # With 168h sequences and 48h non-overlapping windows:
        # Each stay gets 3 windows (168/48 = 3.5, floor = 3 complete windows)
        val_stay_count = len(dm.val_indices)
        val_sample_count = len(val_loader.dataset)

        # Non-overlapping: (168 - 48) / 48 + 1 = 3 windows per stay
        expected_windows_per_stay = 3
        assert val_sample_count == val_stay_count * expected_windows_per_stay

    def test_datamodule_patient_splits_preserved(self, mock_long_sequence_data):
        """Test that patient-level splits are preserved with sliding windows."""
        dm = ICUDataModule(
            processed_dir=mock_long_sequence_data,
            task_name=None,
            batch_size=4,
            enable_sliding_windows=True,
            window_size=48,
            window_stride=24,
        )
        dm.setup()

        # Get patient IDs from each split
        train_patient_ids = set()
        val_patient_ids = set()

        # Collect from train loader
        train_loader = dm.train_dataloader()
        for batch in train_loader:
            for stay_id in batch["stay_id"].tolist():
                patient_row = dm.dataset.static_df.filter(pl.col("stay_id") == stay_id)
                if len(patient_row) > 0:
                    train_patient_ids.add(patient_row["patient_id"][0])

        # Collect from val loader
        val_loader = dm.val_dataloader()
        for batch in val_loader:
            for stay_id in batch["stay_id"].tolist():
                patient_row = dm.dataset.static_df.filter(pl.col("stay_id") == stay_id)
                if len(patient_row) > 0:
                    val_patient_ids.add(patient_row["patient_id"][0])

        # No patient should appear in both splits
        assert train_patient_ids.isdisjoint(val_patient_ids)


class TestCollateFnWithSlidingWindows:
    """Tests for collate function with sliding window metadata."""

    def test_collate_with_window_metadata(self):
        """Test collation handles window metadata fields."""
        samples = [
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 1,
                "window_start": 0,
                "window_idx": 0,
            },
            {
                "timeseries": torch.randn(48, 3),
                "mask": torch.ones(48, 3, dtype=torch.bool),
                "stay_id": 1,
                "window_start": 24,
                "window_idx": 1,
            },
        ]

        batch = icu_collate_fn(samples)

        assert "window_start" in batch
        assert "window_idx" in batch
        assert batch["window_start"].shape == (2,)
        assert batch["window_idx"].shape == (2,)
        assert batch["window_start"].tolist() == [0, 24]
        assert batch["window_idx"].tolist() == [0, 1]

    def test_collate_without_window_metadata(self):
        """Test collation still works without window metadata (backward compatibility)."""
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

        assert "window_start" not in batch
        assert "window_idx" not in batch
        assert batch["timeseries"].shape == (2, 48, 3)
