"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def sample_batch() -> dict:
    """Create a sample batch for testing.
    
    Returns:
        Dictionary with sample batch data.
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
        Dictionary with sample configuration.
    """
    return {
        "data_dir": "/tmp/test_data",
        "output_dir": "/tmp/test_output",
        "seq_length_hours": 48,
        "feature_set": "core",
    }

