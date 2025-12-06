"""Integration tests for extractor with task system."""

import pytest
from unittest.mock import Mock, patch
import polars as pl
from datetime import datetime

from slices.data.extractors.mimic_iv import MIMICIVExtractor, ExtractorConfig
from slices.data.labels import LabelConfig


class TestExtractorIntegration:
    """Integration tests for MIMICIVExtractor with task system."""

    @pytest.fixture
    def mock_extractor(self, tmp_path):
        """Create a mock extractor with temporary paths."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        
        # Create mock parquet directory structure
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        return MIMICIVExtractor(config)

    def test_extract_data_source_unknown_source(self, mock_extractor):
        """Test that extract_data_source raises error for unknown source."""
        with pytest.raises(ValueError, match="Unknown data source"):
            mock_extractor.extract_data_source("unknown_source", [1, 2, 3])

    def test_extract_data_source_mortality_info_signature(self, mock_extractor):
        """Test that mortality_info extraction has correct signature."""
        # Mock the _query method to return a DataFrame with expected columns
        mock_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "date_of_death": [None, datetime(2020, 1, 1), None],
            "hospital_expire_flag": [0, 1, 0],
            "dischtime": [
                datetime(2020, 1, 5),
                datetime(2020, 1, 2),
                datetime(2020, 1, 10),
            ],
            "discharge_location": ["HOME", "DIED", "HOME"],
        })
        
        with patch.object(mock_extractor, '_query', return_value=mock_df):
            result = mock_extractor.extract_data_source("mortality_info", [1, 2, 3])
            
            # Verify columns
            assert "stay_id" in result.columns
            assert "date_of_death" in result.columns
            assert "hospital_expire_flag" in result.columns
            assert "dischtime" in result.columns
            assert "discharge_location" in result.columns
            
            # Verify shape
            assert result.shape[0] == 3

    def test_extract_labels_calls_extract_data_source(self, mock_extractor):
        """Test that extract_labels properly calls extract_data_source."""
        # Mock stays data
        stays_df = pl.DataFrame({
            "stay_id": [1, 2],
            "patient_id": [100, 200],
            "intime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 2, 12, 0)],
            "outtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
            "length_of_stay_days": [2.0, 3.0],
        })
        
        # Mock mortality data
        mortality_df = pl.DataFrame({
            "stay_id": [1, 2],
            "date_of_death": [datetime(2020, 1, 1, 20, 0), None],
            "hospital_expire_flag": [1, 0],
            "dischtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
            "discharge_location": ["DIED", "HOME"],
        })
        
        # Create task config
        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )
        
        with patch.object(mock_extractor, 'extract_stays', return_value=stays_df):
            with patch.object(mock_extractor, 'extract_data_source', return_value=mortality_df) as mock_extract:
                labels = mock_extractor.extract_labels([1, 2], [task_config])
                
                # Verify extract_data_source was called with correct arguments
                mock_extract.assert_called_once_with("mortality_info", [1, 2])
                
                # Verify labels DataFrame structure
                assert "stay_id" in labels.columns
                assert "mortality_24h" in labels.columns
                assert labels.shape[0] == 2

    def test_extract_multiple_tasks_no_column_conflicts(self, mock_extractor):
        """Test extracting multiple tasks at once - critical for real-world usage."""
        # Mock stays data
        stays_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "patient_id": [100, 200, 300],
            "intime": [
                datetime(2020, 1, 1, 10, 0),
                datetime(2020, 1, 2, 12, 0),
                datetime(2020, 1, 3, 8, 0),
            ],
            "outtime": [
                datetime(2020, 1, 3, 10, 0),
                datetime(2020, 1, 5, 12, 0),
                datetime(2020, 1, 6, 8, 0),
            ],
            "length_of_stay_days": [2.0, 3.0, 3.0],
        })
        
        # Mock mortality data
        mortality_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "date_of_death": [
                datetime(2020, 1, 1, 20, 0),  # 10h after admission (within 24h)
                datetime(2020, 1, 4, 13, 0),  # 49h after admission (outside 24h, within 48h)
                None,  # Survived
            ],
            "hospital_expire_flag": [1, 1, 0],
            "dischtime": [
                datetime(2020, 1, 3, 10, 0),
                datetime(2020, 1, 5, 12, 0),
                datetime(2020, 1, 6, 8, 0),
            ],
            "discharge_location": ["DIED", "DIED", "HOME"],
        })
        
        # Create MULTIPLE task configs (this is the critical test case)
        task_configs = [
            LabelConfig(
                task_name="mortality_24h",
                task_type="binary_classification",
                prediction_window_hours=24,
                label_sources=["stays", "mortality_info"],
            ),
            LabelConfig(
                task_name="mortality_48h",
                task_type="binary_classification",
                prediction_window_hours=48,
                label_sources=["stays", "mortality_info"],
            ),
            LabelConfig(
                task_name="mortality_hospital",
                task_type="binary_classification",
                prediction_window_hours=None,
                label_sources=["stays", "mortality_info"],
            ),
        ]
        
        with patch.object(mock_extractor, 'extract_stays', return_value=stays_df):
            with patch.object(mock_extractor, 'extract_data_source', return_value=mortality_df):
                labels = mock_extractor.extract_labels([1, 2, 3], task_configs)
                
                # Verify all columns exist and are unique
                assert "stay_id" in labels.columns
                assert "mortality_24h" in labels.columns
                assert "mortality_48h" in labels.columns
                assert "mortality_hospital" in labels.columns
                
                # Verify no duplicate columns
                assert len(labels.columns) == len(set(labels.columns)), \
                    f"Duplicate columns found: {labels.columns}"
                
                # Verify shape
                assert labels.shape == (3, 4), f"Expected (3, 4), got {labels.shape}"
                
                # Verify label values for stay 1 (died at 10h)
                row1 = labels.filter(pl.col("stay_id") == 1)
                assert row1["mortality_24h"][0] == 1  # Within 24h
                assert row1["mortality_48h"][0] == 1  # Within 48h
                assert row1["mortality_hospital"][0] == 1  # Died in hospital
                
                # Verify label values for stay 2 (died at 49h)
                row2 = labels.filter(pl.col("stay_id") == 2)
                assert row2["mortality_24h"][0] == 0  # Not within 24h (died at 49h)
                assert row2["mortality_48h"][0] == 0  # Outside 48h window (died at 49h)
                assert row2["mortality_hospital"][0] == 1  # Died in hospital
                
                # Verify label values for stay 3 (survived)
                row3 = labels.filter(pl.col("stay_id") == 3)
                assert row3["mortality_24h"][0] == 0
                assert row3["mortality_48h"][0] == 0
                assert row3["mortality_hospital"][0] == 0


class TestExtractDataSourceAvailability:
    """Test which data sources are available."""

    def test_available_data_sources(self, tmp_path):
        """Test listing available data sources."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        
        # Create mock parquet directory
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        extractor = MIMICIVExtractor(config)
        
        # Test that mortality_info is available
        try:
            # This will fail on SQL execution, but shouldn't fail on dispatch
            with patch.object(extractor, '_query', side_effect=Exception("SQL error")):
                with pytest.raises(Exception, match="SQL error"):
                    extractor.extract_data_source("mortality_info", [1])
        except ValueError:
            pytest.fail("mortality_info should be a recognized data source")
        
        # Test that unknown source raises ValueError
        with pytest.raises(ValueError, match="Unknown data source"):
            extractor.extract_data_source("nonexistent_source", [1])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_extract_labels_with_empty_stay_list(self, tmp_path):
        """Test extracting labels for empty stay list."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        extractor = MIMICIVExtractor(config)
        
        # Mock empty stays data with proper schema
        empty_stays_df = pl.DataFrame({
            "stay_id": [],
            "patient_id": [],
            "intime": [],
            "outtime": [],
            "length_of_stay_days": [],
        }).cast({"stay_id": pl.Int64, "patient_id": pl.Int64, "length_of_stay_days": pl.Float64})
        
        # Mock empty mortality data with proper schema
        empty_mortality_df = pl.DataFrame({
            "stay_id": [],
            "date_of_death": [],
            "hospital_expire_flag": [],
            "dischtime": [],
            "discharge_location": [],
        }).cast({"stay_id": pl.Int64, "hospital_expire_flag": pl.Int32})
        
        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )
        
        with patch.object(extractor, 'extract_stays', return_value=empty_stays_df):
            with patch.object(extractor, 'extract_data_source', return_value=empty_mortality_df):
                labels = extractor.extract_labels([], [task_config])
                
                # Should return empty DataFrame with correct columns
                assert "stay_id" in labels.columns
                assert "mortality_24h" in labels.columns
                assert labels.shape[0] == 0

    def test_extract_labels_no_tasks(self, tmp_path):
        """Test extracting labels with no task configs provided."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        extractor = MIMICIVExtractor(config)
        
        stays_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "patient_id": [100, 200, 300],
            "intime": [datetime(2020, 1, 1)] * 3,
            "outtime": [datetime(2020, 1, 2)] * 3,
            "length_of_stay_days": [1.0] * 3,
        })
        
        with patch.object(extractor, 'extract_stays', return_value=stays_df):
            labels = extractor.extract_labels([1, 2, 3], [])
            
            # Should return just stay_ids
            assert list(labels.columns) == ["stay_id"]
            assert labels.shape == (3, 1)

    def test_extract_labels_with_nulls_in_mortality_data(self, tmp_path):
        """Test handling of null values in mortality data."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        extractor = MIMICIVExtractor(config)
        
        # Mock stays data
        stays_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "patient_id": [100, 200, 300],
            "intime": [datetime(2020, 1, 1, 10, 0)] * 3,
            "outtime": [datetime(2020, 1, 3, 10, 0)] * 3,
            "length_of_stay_days": [2.0] * 3,
        })
        
        # Mock mortality data with nulls
        mortality_df = pl.DataFrame({
            "stay_id": [1, 2, 3],
            "date_of_death": [None, None, None],  # All survived
            "hospital_expire_flag": [0, 0, 0],
            "dischtime": [datetime(2020, 1, 3, 10, 0)] * 3,
            "discharge_location": ["HOME"] * 3,
        })
        
        task_config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )
        
        with patch.object(extractor, 'extract_stays', return_value=stays_df):
            with patch.object(extractor, 'extract_data_source', return_value=mortality_df):
                labels = extractor.extract_labels([1, 2, 3], [task_config])
                
                # All should be 0 (survived)
                assert labels["mortality_hospital"].to_list() == [0, 0, 0]

    def test_many_tasks_extraction(self, tmp_path):
        """Stress test: extract many tasks at once to verify join performance."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        
        extractor = MIMICIVExtractor(config)
        
        # Mock stays data with 100 stays
        num_stays = 100
        stays_df = pl.DataFrame({
            "stay_id": list(range(1, num_stays + 1)),
            "patient_id": list(range(100, 100 + num_stays)),
            "intime": [datetime(2020, 1, 1, 10, 0)] * num_stays,
            "outtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
            "length_of_stay_days": [2.0] * num_stays,
        })
        
        mortality_df = pl.DataFrame({
            "stay_id": list(range(1, num_stays + 1)),
            "date_of_death": [None] * num_stays,
            "hospital_expire_flag": [0] * num_stays,
            "dischtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
            "discharge_location": ["HOME"] * num_stays,
        })
        
        # Create 10 different task configs
        task_configs = [
            LabelConfig(
                task_name=f"mortality_task_{i}",
                task_type="binary_classification",
                prediction_window_hours=24 * i if i > 0 else None,
                label_sources=["stays", "mortality_info"],
            )
            for i in range(10)
        ]
        
        with patch.object(extractor, 'extract_stays', return_value=stays_df):
            with patch.object(extractor, 'extract_data_source', return_value=mortality_df):
                labels = extractor.extract_labels(list(range(1, num_stays + 1)), task_configs)
                
                # Verify shape: 100 stays x (1 stay_id + 10 task columns)
                assert labels.shape == (num_stays, 11)
                
                # Verify all task columns exist
                for i in range(10):
                    assert f"mortality_task_{i}" in labels.columns
                
                # Verify no duplicate columns
                assert len(labels.columns) == len(set(labels.columns))
