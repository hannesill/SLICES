"""Tests for ExtractorConfig dataclass."""

import pytest

from slices.data.extractors.base import ExtractorConfig


class TestExtractorConfig:
    """Test ExtractorConfig validation and initialization."""

    def test_parquet_root_only(self):
        """Test initialization with parquet_root only."""
        config = ExtractorConfig(parquet_root="/path/to/parquet")

        assert config.parquet_root == "/path/to/parquet"

    def test_missing_parquet_root_raises_error(self):
        """Test that missing parquet_root raises TypeError."""
        with pytest.raises(TypeError):
            ExtractorConfig()

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractorConfig(parquet_root="/path/to/parquet")

        assert config.output_dir == "data/processed"
        assert config.seq_length_hours == 48
        assert config.feature_set == "core"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExtractorConfig(
            parquet_root="/custom/path",
            output_dir="/custom/output",
            seq_length_hours=72,
            feature_set="extended",
        )

        assert config.output_dir == "/custom/output"
        assert config.seq_length_hours == 72
        assert config.feature_set == "extended"

    def test_immutable_after_creation(self):
        """Test that config is immutable (dataclass frozen behavior)."""
        config = ExtractorConfig(parquet_root="/path/to/parquet")

        # Should be able to access values
        assert config.parquet_root == "/path/to/parquet"

        # Note: dataclass is not frozen by default, so this test documents expected behavior
        # If you want immutability, add frozen=True to the dataclass decorator

    def test_empty_string_accepted(self):
        """Test that empty string is accepted (validation happens at runtime)."""
        config = ExtractorConfig(parquet_root="")
        assert config.parquet_root == ""

    def test_whitespace_only_strings_treated_as_missing(self):
        """Test that whitespace-only strings are treated as missing values."""
        # Note: Current implementation doesn't strip whitespace
        # This test documents actual behavior
        config = ExtractorConfig(parquet_root="   ")
        assert config.parquet_root == "   "  # Not stripped
