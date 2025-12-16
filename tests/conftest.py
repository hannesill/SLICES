"""Pytest configuration and fixtures.

This file contains shared fixtures and configuration for all tests in the project.
Fixtures defined here are automatically available to all test files without imports.
"""

import gzip

import polars as pl
import pytest
import torch


@pytest.fixture
def sample_batch() -> dict:
    """Create a sample batch for testing.

    Returns:
        Dictionary with sample batch data including timeseries and labels.
    """
    batch_size = 4
    seq_length = 48
    n_features = 35

    return {
        "timeseries": torch.randn(batch_size, seq_length, n_features),
        "mask": torch.ones(batch_size, seq_length, n_features, dtype=torch.bool),
        "padding_mask": torch.ones(batch_size, seq_length, dtype=torch.bool),
        "labels": torch.randint(0, 2, (batch_size,)),
    }


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration for testing.

    Returns:
        Dictionary with sample configuration values.
    """
    return {
        "parquet_root": "/tmp/test_data",
        "output_dir": "/tmp/test_output",
        "seq_length_hours": 48,
        "feature_set": "core",
    }


@pytest.fixture
def sample_csv_data() -> dict:
    """Sample CSV data for testing conversions.

    Returns:
        Dictionary mapping table names to CSV content (as bytes).
    """
    return {
        "patients": b"subject_id,gender,anchor_age\n1,M,65\n2,F,42\n3,M,58\n",
        "admissions": b"hadm_id,subject_id,admittime\n1000,1,2180-01-01\n1001,2,2180-01-02\n",
        "icustays": b"stay_id,subject_id,intime,outtime\n"
        b"100,1,2180-01-01 10:00:00,2180-01-02 10:00:00\n",
    }


@pytest.fixture
def create_test_csv_structure(tmp_path, sample_csv_data):
    """Factory fixture to create a complete test CSV directory structure.

    Args:
        tmp_path: Pytest built-in fixture providing temporary directory.
        sample_csv_data: Fixture providing sample CSV data.

    Returns:
        Function that creates CSV structure with optional custom data.
    """

    def _create(data_dict=None):
        """Create CSV structure with given data.

        Args:
            data_dict: Optional dictionary of table names to CSV bytes.
                      If None, uses sample_csv_data.

        Returns:
            Path to created CSV root directory.
        """
        if data_dict is None:
            data_dict = sample_csv_data

        csv_root = tmp_path / "test_csv"
        (csv_root / "hosp").mkdir(parents=True)
        (csv_root / "icu").mkdir(parents=True)

        # Create CSV.gz files
        for filename, content in data_dict.items():
            if filename in ["patients", "admissions"]:
                schema = "hosp"
            else:
                schema = "icu"

            filepath = csv_root / schema / f"{filename}.csv.gz"
            with gzip.open(filepath, "wb") as f:
                f.write(content)

        return csv_root

    return _create


@pytest.fixture
def sample_parquet_df() -> pl.DataFrame:
    """Sample Polars DataFrame for testing.

    Returns:
        Polars DataFrame with sample ICU time-series data.
    """
    return pl.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "patient_id": [100, 101, 102, 103, 104],
            "hour": [0, 0, 1, 1, 2],
            "heart_rate": [75.0, 82.0, 78.0, 90.0, 85.0],
            "respiratory_rate": [16.0, 18.0, 17.0, 20.0, 19.0],
        }
    )
