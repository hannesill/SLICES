"""Integration tests for extractor with task system."""

from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor
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
        mock_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [None, datetime(2020, 1, 1), None],
                "hospital_expire_flag": [0, 1, 0],
                "dischtime": [
                    datetime(2020, 1, 5),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 10),
                ],
                "discharge_location": ["HOME", "DIED", "HOME"],
            }
        )

        with patch.object(mock_extractor, "_query", return_value=mock_df):
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
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "patient_id": [100, 200],
                "intime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 2, 12, 0)],
                "outtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
                "length_of_stay_days": [2.0, 3.0],
            }
        )

        # Mock mortality data
        mortality_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "date_of_death": [datetime(2020, 1, 1, 20, 0), None],
                "hospital_expire_flag": [1, 0],
                "dischtime": [datetime(2020, 1, 3, 10, 0), datetime(2020, 1, 5, 12, 0)],
                "discharge_location": ["DIED", "HOME"],
            }
        )

        # Create task config
        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        with patch.object(mock_extractor, "extract_stays", return_value=stays_df):
            with patch.object(
                mock_extractor, "extract_data_source", return_value=mortality_df
            ) as mock_extract:
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
        stays_df = pl.DataFrame(
            {
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
            }
        )

        # Mock mortality data
        mortality_df = pl.DataFrame(
            {
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
            }
        )

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

        with patch.object(mock_extractor, "extract_stays", return_value=stays_df):
            with patch.object(mock_extractor, "extract_data_source", return_value=mortality_df):
                labels = mock_extractor.extract_labels([1, 2, 3], task_configs)

                # Verify all columns exist and are unique
                assert "stay_id" in labels.columns
                assert "mortality_24h" in labels.columns
                assert "mortality_48h" in labels.columns
                assert "mortality_hospital" in labels.columns

                # Verify no duplicate columns
                assert len(labels.columns) == len(
                    set(labels.columns)
                ), f"Duplicate columns found: {labels.columns}"

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
            with patch.object(extractor, "_query", side_effect=Exception("SQL error")):
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
        empty_stays_df = pl.DataFrame(
            {
                "stay_id": [],
                "patient_id": [],
                "intime": [],
                "outtime": [],
                "length_of_stay_days": [],
            }
        ).cast({"stay_id": pl.Int64, "patient_id": pl.Int64, "length_of_stay_days": pl.Float64})

        # Mock empty mortality data with proper schema
        empty_mortality_df = pl.DataFrame(
            {
                "stay_id": [],
                "date_of_death": [],
                "hospital_expire_flag": [],
                "dischtime": [],
                "discharge_location": [],
            }
        ).cast({"stay_id": pl.Int64, "hospital_expire_flag": pl.Int32})

        task_config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        with patch.object(extractor, "extract_stays", return_value=empty_stays_df):
            with patch.object(extractor, "extract_data_source", return_value=empty_mortality_df):
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

        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 200, 300],
                "intime": [datetime(2020, 1, 1)] * 3,
                "outtime": [datetime(2020, 1, 2)] * 3,
                "length_of_stay_days": [1.0] * 3,
            }
        )

        with patch.object(extractor, "extract_stays", return_value=stays_df):
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
        stays_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 200, 300],
                "intime": [datetime(2020, 1, 1, 10, 0)] * 3,
                "outtime": [datetime(2020, 1, 3, 10, 0)] * 3,
                "length_of_stay_days": [2.0] * 3,
            }
        )

        # Mock mortality data with nulls
        mortality_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [None, None, None],  # All survived
                "hospital_expire_flag": [0, 0, 0],
                "dischtime": [datetime(2020, 1, 3, 10, 0)] * 3,
                "discharge_location": ["HOME"] * 3,
            }
        )

        task_config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )

        with patch.object(extractor, "extract_stays", return_value=stays_df):
            with patch.object(extractor, "extract_data_source", return_value=mortality_df):
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
        stays_df = pl.DataFrame(
            {
                "stay_id": list(range(1, num_stays + 1)),
                "patient_id": list(range(100, 100 + num_stays)),
                "intime": [datetime(2020, 1, 1, 10, 0)] * num_stays,
                "outtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
                "length_of_stay_days": [2.0] * num_stays,
            }
        )

        mortality_df = pl.DataFrame(
            {
                "stay_id": list(range(1, num_stays + 1)),
                "date_of_death": [None] * num_stays,
                "hospital_expire_flag": [0] * num_stays,
                "dischtime": [datetime(2020, 1, 3, 10, 0)] * num_stays,
                "discharge_location": ["HOME"] * num_stays,
            }
        )

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

        with patch.object(extractor, "extract_stays", return_value=stays_df):
            with patch.object(extractor, "extract_data_source", return_value=mortality_df):
                labels = extractor.extract_labels(list(range(1, num_stays + 1)), task_configs)

                # Verify shape: 100 stays x (1 stay_id + 10 task columns)
                assert labels.shape == (num_stays, 11)

                # Verify all task columns exist
                for i in range(10):
                    assert f"mortality_task_{i}" in labels.columns

                # Verify no duplicate columns
                assert len(labels.columns) == len(set(labels.columns))


class TestSchemaValidation:
    """Test schema validation for extracted data."""

    @pytest.fixture
    def mock_extractor(self, tmp_path):
        """Create a mock extractor with temporary paths."""
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
        )
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)
        return MIMICIVExtractor(config)

    def test_validate_raw_events_schema_valid_data(self, mock_extractor):
        """Test validation passes for correctly formatted raw events."""
        valid_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "charttime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 1, 11, 0),
                    datetime(2020, 1, 1, 12, 0),
                ],
                "feature_name": ["heart_rate", "sbp", "heart_rate"],
                "valuenum": [75.0, 120.0, 78.0],
            }
        )
        # Should not raise any exception
        mock_extractor._validate_raw_events_schema(valid_df)

    def test_validate_raw_events_schema_empty_dataframe(self, mock_extractor):
        """Test validation passes for empty DataFrames."""
        empty_df = pl.DataFrame(
            {
                "stay_id": [],
                "charttime": [],
                "feature_name": [],
                "valuenum": [],
            }
        )
        # Should not raise any exception
        mock_extractor._validate_raw_events_schema(empty_df)

    def test_validate_raw_events_schema_missing_column(self, mock_extractor):
        """Test validation fails when required columns are missing."""
        invalid_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 1, 11, 0)],
                # Missing feature_name
                "valuenum": [75.0, 120.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required column 'feature_name'"):
            mock_extractor._validate_raw_events_schema(invalid_df)

    def test_validate_raw_events_schema_wrong_column_type(self, mock_extractor):
        """Test validation fails when column types don't match."""
        invalid_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 1, 11, 0)],
                "feature_name": ["heart_rate", "sbp"],
                "valuenum": ["75", "120"],  # Should be float, not string
            }
        )
        with pytest.raises(ValueError, match="has type.*expected"):
            mock_extractor._validate_raw_events_schema(invalid_df)

    def test_extract_raw_events_validates_schema(self, mock_extractor):
        """Test that _extract_raw_events validates schema before returning."""
        from slices.data.config_schemas import ItemIDSource, TimeSeriesConceptConfig

        # Mock _extract_by_itemid to return valid data
        valid_df = pl.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 1, 11, 0)],
                "feature_name": ["heart_rate", "heart_rate"],
                "valuenum": [75.0, 80.0],
            }
        )

        feature_mapping = {
            "heart_rate": TimeSeriesConceptConfig(
                description="Heart rate",
                units="bpm",
                mimic_iv=[
                    ItemIDSource(
                        table="chartevents",
                        itemid=[220045],
                        value_col="valuenum",
                        time_col="charttime",
                    )
                ],
            )
        }

        with patch.object(mock_extractor, "_extract_by_itemid", return_value=valid_df):
            # Should not raise any exception
            result = mock_extractor._extract_raw_events(
                stay_ids=[1, 2], feature_mapping=feature_mapping
            )
            assert result.shape[0] == 2

    def test_extract_raw_events_schema_mismatch_raises_error(self, mock_extractor):
        """Test that schema validation catches Polars version changes."""
        from slices.data.config_schemas import ItemIDSource, TimeSeriesConceptConfig

        # Mock _extract_by_itemid to return data with wrong types
        invalid_df = pl.DataFrame(
            {
                "stay_id": ["1", "2"],  # Wrong type: should be Int64
                "charttime": [datetime(2020, 1, 1, 10, 0), datetime(2020, 1, 1, 11, 0)],
                "feature_name": ["heart_rate", "heart_rate"],
                "valuenum": [75.0, 80.0],
            }
        )

        feature_mapping = {
            "heart_rate": TimeSeriesConceptConfig(
                description="Heart rate",
                units="bpm",
                mimic_iv=[
                    ItemIDSource(
                        table="chartevents",
                        itemid=[220045],
                        value_col="valuenum",
                        time_col="charttime",
                    )
                ],
            )
        }

        with patch.object(mock_extractor, "_extract_by_itemid", return_value=invalid_df):
            with pytest.raises(ValueError, match="has type.*expected"):
                mock_extractor._extract_raw_events(
                    stay_ids=[1, 2],
                    feature_mapping=feature_mapping,
                )


class TestStaticFeatureExtraction:
    """Tests for static feature extraction."""

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

    def test_extract_static_features_method_exists(self, mock_extractor):
        """Test that extract_static_features method exists."""
        assert hasattr(mock_extractor, "extract_static_features")
        assert callable(mock_extractor.extract_static_features)

    def test_extract_static_columns_success(self, mock_extractor):
        """Test that _extract_static_columns extracts column-based features."""
        from slices.data.config_schemas import StaticConceptConfig, StaticExtractionSource

        # Create mock static concepts
        concepts = {
            "age": StaticConceptConfig(
                description="Patient age",
                dtype="numeric",
                mimic_iv=StaticExtractionSource(table="patients", column="anchor_age"),
            ),
            "gender": StaticConceptConfig(
                description="Patient gender",
                dtype="categorical",
                mimic_iv=StaticExtractionSource(table="patients", column="gender"),
            ),
        }

        # Mock the _query method
        mock_df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "age": [65, 45, 70],
                "gender": ["M", "F", "M"],
            }
        )

        with patch.object(mock_extractor, "_query", return_value=mock_df):
            result = mock_extractor._extract_static_columns([1, 2, 3], concepts)

        assert "stay_id" in result.columns
        assert "age" in result.columns
        assert "gender" in result.columns
        assert len(result) == 3

    def test_extract_static_by_itemid_success(self, mock_extractor):
        """Test that _extract_static_by_itemid extracts itemid-based features."""
        from slices.data.config_schemas import StaticConceptConfig, StaticExtractionSource

        # Create mock static concepts for height/weight
        concepts = {
            "height": StaticConceptConfig(
                description="Patient height",
                dtype="numeric",
                mimic_iv=StaticExtractionSource(
                    table="chartevents", column="valuenum", itemid=226730
                ),
            ),
            "weight": StaticConceptConfig(
                description="Patient weight",
                dtype="numeric",
                mimic_iv=StaticExtractionSource(
                    table="chartevents", column="valuenum", itemid=226512
                ),
            ),
        }

        # Mock the _query method to return height and weight dataframes
        def mock_query(sql):
            if "226730" in sql:  # height itemid
                return pl.DataFrame({"stay_id": [1, 2], "height": [170.0, 165.0]})
            elif "226512" in sql:  # weight itemid
                return pl.DataFrame({"stay_id": [1, 2, 3], "weight": [80.0, 60.0, 75.0]})
            return pl.DataFrame()

        with patch.object(mock_extractor, "_query", side_effect=mock_query):
            result = mock_extractor._extract_static_by_itemid([1, 2, 3], concepts)

        assert "stay_id" in result.columns
        assert "height" in result.columns
        assert "weight" in result.columns
        assert len(result) == 3

    def test_extract_static_features_integration(self, mock_extractor):
        """Test full extract_static_features integration."""
        from slices.data.config_schemas import StaticConceptConfig, StaticExtractionSource

        # Mock static concepts to include both column-based and itemid-based
        mock_concepts = {
            "age": StaticConceptConfig(
                description="Patient age",
                dtype="numeric",
                mimic_iv=StaticExtractionSource(table="patients", column="anchor_age"),
            ),
            "height": StaticConceptConfig(
                description="Patient height",
                dtype="numeric",
                mimic_iv=StaticExtractionSource(
                    table="chartevents", column="valuenum", itemid=226730
                ),
            ),
        }

        # Mock the config loader and query methods
        with (
            patch(
                "slices.data.extractors.mimic_iv.load_static_concepts",
                return_value=mock_concepts,
            ),
            patch.object(
                mock_extractor,
                "_extract_static_columns",
                return_value=pl.DataFrame({"stay_id": [1, 2], "age": [65, 45]}),
            ),
            patch.object(
                mock_extractor,
                "_extract_static_by_itemid",
                return_value=pl.DataFrame({"stay_id": [1, 2], "height": [170.0, 165.0]}),
            ),
        ):
            result = mock_extractor.extract_static_features([1, 2])

        assert "stay_id" in result.columns
        assert "age" in result.columns
        assert "height" in result.columns
        assert len(result) == 2

    def test_extract_static_features_empty_config(self, mock_extractor):
        """Test extract_static_features with empty config returns base DataFrame."""
        with patch("slices.data.extractors.mimic_iv.load_static_concepts", return_value={}):
            result = mock_extractor.extract_static_features([1, 2, 3])

        assert "stay_id" in result.columns
        assert len(result) == 3
        assert result["stay_id"].to_list() == [1, 2, 3]
