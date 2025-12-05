"""Tests for CSV-to-Parquet conversion functionality."""

import gzip
import os
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
import pytest

from slices.data.data_io import _csv_to_parquet_all, convert_csv_to_parquet


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create temporary CSV directory structure with test data.
    
    Args:
        tmp_path: Pytest fixture providing temporary directory.
        
    Returns:
        Path to CSV root directory.
    """
    csv_root = tmp_path / "csv"

    # Create directory structure
    (csv_root / "hosp").mkdir(parents=True)
    (csv_root / "icu").mkdir(parents=True)

    # Create dummy CSV.gz files
    # Small patients CSV
    patients_data = b"subject_id,gender,anchor_age\n1,M,65\n2,F,42\n"
    with gzip.open(csv_root / "hosp" / "patients.csv.gz", "wb") as f:
        f.write(patients_data)

    # Small icustays CSV
    icu_data = b"stay_id,subject_id,hadm_id,intime\n100,1,1000,2180-01-01 10:00:00\n"
    with gzip.open(csv_root / "icu" / "icustays.csv.gz", "wb") as f:
        f.write(icu_data)

    return csv_root


@pytest.fixture
def temp_parquet_dir(tmp_path):
    """Create temporary Parquet output directory.
    
    Args:
        tmp_path: Pytest fixture providing temporary directory.
        
    Returns:
        Path to Parquet output directory.
    """
    return tmp_path / "parquet"


class TestCsvToParquetAll:
    """Test _csv_to_parquet_all function."""

    def test_successful_conversion(self, temp_csv_dir, temp_parquet_dir):
        """Test successful conversion of all CSV files."""
        success = _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)

        assert success is True
        assert (temp_parquet_dir / "hosp" / "patients.parquet").exists()
        assert (temp_parquet_dir / "icu" / "icustays.parquet").exists()

    def test_no_csv_files_found(self, tmp_path):
        """Test error when no CSV files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_dir = tmp_path / "output"

        success = _csv_to_parquet_all(empty_dir, output_dir)

        assert success is False

    def test_preserves_directory_structure(self, temp_csv_dir, temp_parquet_dir):
        """Test that directory structure is preserved."""
        _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)

        # Check structure mirrors input
        assert (temp_parquet_dir / "hosp").is_dir()
        assert (temp_parquet_dir / "icu").is_dir()

    def test_parquet_readable_by_duckdb(self, temp_csv_dir, temp_parquet_dir):
        """Test that output Parquet files are readable by DuckDB."""
        _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)

        # Read and verify
        con = duckdb.connect()
        patients_path = temp_parquet_dir / "hosp" / "patients.parquet"
        result = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{patients_path}')"
        ).fetchone()

        assert result[0] == 2  # 2 patients in test data
        con.close()

    def test_parquet_data_integrity(self, temp_csv_dir, temp_parquet_dir):
        """Test that converted data matches source CSV."""
        _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)

        # Verify data content
        con = duckdb.connect()
        patients_path = temp_parquet_dir / "hosp" / "patients.parquet"
        result = con.execute(
            f"SELECT subject_id, gender, anchor_age FROM read_parquet('{patients_path}') ORDER BY subject_id"
        ).fetchall()

        assert result[0] == (1, "M", 65)
        assert result[1] == (2, "F", 42)
        con.close()

    def test_respects_max_workers_env_var(self, temp_csv_dir, temp_parquet_dir, monkeypatch):
        """Test that MAX_WORKERS environment variable is respected."""
        monkeypatch.setenv("SLICES_CONVERT_MAX_WORKERS", "2")

        # Run conversion (should use 2 workers)
        success = _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)

        assert success is True


class TestConvertCsvToParquet:
    """Test convert_csv_to_parquet public API."""

    def test_converts_successfully(self, temp_csv_dir, temp_parquet_dir):
        """Test successful conversion via public API."""
        success = convert_csv_to_parquet(
            csv_root=temp_csv_dir,
            parquet_root=temp_parquet_dir,
            dataset_name="test_dataset",
        )

        assert success is True
        assert temp_parquet_dir.exists()

    def test_csv_root_not_found(self, tmp_path):
        """Test error when CSV root doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        success = convert_csv_to_parquet(csv_root=nonexistent, parquet_root=output)

        assert success is False

    def test_creates_parquet_root(self, temp_csv_dir, tmp_path):
        """Test that parquet_root is created if it doesn't exist."""
        output = tmp_path / "new" / "nested" / "output"

        convert_csv_to_parquet(csv_root=temp_csv_dir, parquet_root=output)

        assert output.exists()
        assert output.is_dir()

    def test_dataset_name_optional(self, temp_csv_dir, temp_parquet_dir):
        """Test that dataset_name is optional."""
        success = convert_csv_to_parquet(
            csv_root=temp_csv_dir, parquet_root=temp_parquet_dir
        )

        assert success is True


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_default_values_used(self, temp_csv_dir, temp_parquet_dir):
        """Test that defaults are used when env vars not set."""
        # Clear any existing env vars
        for var in [
            "SLICES_CONVERT_MAX_WORKERS",
            "SLICES_DUCKDB_MEM",
            "SLICES_DUCKDB_THREADS",
        ]:
            os.environ.pop(var, None)

        success = _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)
        assert success is True

    def test_custom_env_vars_respected(self, temp_csv_dir, temp_parquet_dir, monkeypatch):
        """Test that custom environment variables work."""
        monkeypatch.setenv("SLICES_CONVERT_MAX_WORKERS", "8")
        monkeypatch.setenv("SLICES_DUCKDB_MEM", "4GB")
        monkeypatch.setenv("SLICES_DUCKDB_THREADS", "4")

        success = _csv_to_parquet_all(temp_csv_dir, temp_parquet_dir)
        assert success is True
