"""Tests for BaseExtractor functionality."""

from pathlib import Path

import polars as pl
import pytest

from slices.data.extractors.base import BaseExtractor, ExtractorConfig


class MockExtractor(BaseExtractor):
    """Mock implementation of BaseExtractor for testing."""

    def extract_stays(self) -> pl.DataFrame:
        """Mock stay extraction.
        
        Returns:
            DataFrame with mock ICU stay data.
        """
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 101, 102],
                "intime": ["2020-01-01", "2020-01-02", "2020-01-03"],
            }
        )

    def extract_timeseries(self, stay_ids) -> pl.DataFrame:
        """Mock timeseries extraction.
        
        Args:
            stay_ids: List of stay IDs to extract.
            
        Returns:
            DataFrame with mock timeseries data.
        """
        return pl.DataFrame(
            {
                "stay_id": stay_ids,
                "hour": [0] * len(stay_ids),
                "value": [1.0] * len(stay_ids),
            }
        )

    def extract_labels(self, stay_ids) -> pl.DataFrame:
        """Mock label extraction.
        
        Args:
            stay_ids: List of stay IDs to extract.
            
        Returns:
            DataFrame with mock labels.
        """
        return pl.DataFrame(
            {
                "stay_id": stay_ids,
                "mortality_48h": [0] * len(stay_ids),
            }
        )


@pytest.fixture
def temp_parquet_structure(tmp_path):
    """Create temporary Parquet directory structure with test files.
    
    Args:
        tmp_path: Pytest fixture providing temporary directory.
        
    Returns:
        Path to Parquet root directory.
    """
    parquet_root = tmp_path / "parquet"
    (parquet_root / "hosp").mkdir(parents=True)
    (parquet_root / "icu").mkdir(parents=True)

    # Create dummy Parquet files
    patients_df = pl.DataFrame({"subject_id": [1, 2], "gender": ["M", "F"]})
    patients_df.write_parquet(parquet_root / "hosp" / "patients.parquet")

    icustays_df = pl.DataFrame({"stay_id": [1, 2], "subject_id": [1, 2]})
    icustays_df.write_parquet(parquet_root / "icu" / "icustays.parquet")

    return parquet_root


class TestBaseExtractor:
    """Test BaseExtractor abstract class."""

    def test_initialization_success(self, temp_parquet_structure):
        """Test successful initialization with valid directory."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        assert extractor.parquet_root == temp_parquet_structure
        assert extractor.conn is not None  # DuckDB connection

    def test_initialization_nonexistent_directory(self, tmp_path):
        """Test that nonexistent directory raises ValueError."""
        nonexistent = tmp_path / "nonexistent"
        config = ExtractorConfig(parquet_root=str(nonexistent))

        with pytest.raises(ValueError, match="Parquet directory not found"):
            MockExtractor(config)

    def test_parquet_path_generation(self, temp_parquet_structure):
        """Test _parquet_path method generates correct paths."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        path = extractor._parquet_path("hosp", "patients")

        assert path == temp_parquet_structure / "hosp" / "patients.parquet"
        assert isinstance(path, Path)

    def test_query_execution(self, temp_parquet_structure):
        """Test _query method executes SQL and returns Polars DataFrame."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        patients_path = extractor._parquet_path("hosp", "patients")
        sql = f"SELECT * FROM read_parquet('{patients_path}')"
        result = extractor._query(sql)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "subject_id" in result.columns

    def test_query_with_filter(self, temp_parquet_structure):
        """Test _query method with WHERE clause."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        patients_path = extractor._parquet_path("hosp", "patients")
        sql = f"SELECT * FROM read_parquet('{patients_path}') WHERE subject_id = 1"
        result = extractor._query(sql)

        assert len(result) == 1
        assert result["subject_id"][0] == 1

    def test_abstract_methods_implemented(self, temp_parquet_structure):
        """Test that mock implementation provides all abstract methods."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        # Should not raise NotImplementedError
        stays = extractor.extract_stays()
        assert isinstance(stays, pl.DataFrame)
        assert len(stays) == 3

        timeseries = extractor.extract_timeseries([1, 2])
        assert isinstance(timeseries, pl.DataFrame)
        assert len(timeseries) == 2

        labels = extractor.extract_labels([1, 2])
        assert isinstance(labels, pl.DataFrame)
        assert len(labels) == 2

    def test_bin_to_hourly_not_implemented(self, temp_parquet_structure):
        """Test that bin_to_hourly raises NotImplementedError."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        with pytest.raises(NotImplementedError):
            extractor.bin_to_hourly(pl.DataFrame(), pl.DataFrame())

    def test_run_method_not_implemented(self, temp_parquet_structure):
        """Test that run method raises NotImplementedError."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        with pytest.raises(NotImplementedError):
            extractor.run()

    def test_output_dir_created(self, temp_parquet_structure, tmp_path):
        """Test that output directory is set correctly."""
        output_dir = tmp_path / "output"
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure), output_dir=str(output_dir)
        )
        extractor = MockExtractor(config)

        assert extractor.output_dir == output_dir

    def test_multiple_extractors_independent(self, temp_parquet_structure):
        """Test that multiple extractor instances are independent."""
        config1 = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        config2 = ExtractorConfig(parquet_root=str(temp_parquet_structure))

        extractor1 = MockExtractor(config1)
        extractor2 = MockExtractor(config2)

        # Should have independent DuckDB connections
        assert extractor1.conn is not extractor2.conn
