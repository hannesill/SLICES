"""Basic sanity tests for project setup.

Tests that the project is installed correctly and modules can be imported.
"""

import pytest


def test_placeholder() -> None:
    """Placeholder test to verify pytest setup."""
    assert True


def test_imports() -> None:
    """Test that main modules can be imported."""
    from slices.data.extractors.base import BaseExtractor, ExtractorConfig

    # Verify classes exist
    assert BaseExtractor is not None
    assert ExtractorConfig is not None


def test_data_io_imports() -> None:
    """Test that data_io module imports successfully."""
    from slices.data.data_io import convert_csv_to_parquet

    assert convert_csv_to_parquet is not None


def test_version() -> None:
    """Test that package version is accessible."""
    import slices

    assert hasattr(slices, "__version__")
    assert slices.__version__ == "0.1.0"

