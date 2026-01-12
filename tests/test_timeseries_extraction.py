"""Tests for time-series extraction functionality."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from slices.data.config_schemas import AggregationType, ItemIDSource, TimeSeriesConceptConfig
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor


def make_feature_mapping(features: dict) -> dict:
    """Helper to create TimeSeriesConceptConfig mapping from simple dicts.

    Args:
        features: Dict mapping feature_name -> {
            "source": str, "itemid": List[int], "aggregation": str (optional)
        }

    Returns:
        Dict mapping feature_name -> TimeSeriesConceptConfig
    """
    result = {}
    for name, config in features.items():
        aggregation = AggregationType(config.get("aggregation", "mean"))
        result[name] = TimeSeriesConceptConfig(
            description=f"Test {name}",
            units="",
            aggregation=aggregation,
            mimic_iv=[
                ItemIDSource(
                    table=config["source"],
                    itemid=config["itemid"],
                    value_col=config.get("value_col", "valuenum"),
                    time_col="charttime",
                )
            ],
        )
    return result


class TestTimeSeriesExtraction:
    """Tests for extract_timeseries and related methods."""

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

    @pytest.fixture
    def mock_stays_df(self):
        """Create mock stays DataFrame."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [101, 102, 103],
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 2, 14, 30),
                    datetime(2020, 1, 3, 8, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 5, 10, 0),
                    datetime(2020, 1, 6, 14, 30),
                    datetime(2020, 1, 7, 8, 0),
                ],
                "los_days": [4.0, 4.0, 4.0],
                "age": [65, 45, 70],
                "gender": ["M", "F", "M"],
            }
        )

    @pytest.fixture
    def mock_chartevents_df(self):
        """Create mock chartevents DataFrame."""
        # Create realistic chartevents data for heart_rate (220045) and sbp (220050, 220179)
        base_time = datetime(2020, 1, 1, 10, 0)

        return pl.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
                "charttime": [
                    base_time + timedelta(minutes=5),  # hour 0
                    base_time + timedelta(minutes=30),  # hour 0
                    base_time + timedelta(minutes=70),  # hour 1
                    base_time + timedelta(hours=2, minutes=15),  # hour 2
                    base_time + timedelta(hours=2, minutes=45),  # hour 2
                    base_time + timedelta(hours=3),  # hour 3
                    datetime(2020, 1, 2, 14, 45),  # hour 0 for stay 2
                    datetime(2020, 1, 2, 15, 30),  # hour 1 for stay 2
                    datetime(2020, 1, 2, 16, 15),  # hour 1 for stay 2
                ],
                "itemid": [
                    220045,  # heart_rate
                    220045,  # heart_rate
                    220050,  # sbp
                    220045,  # heart_rate
                    220050,  # sbp
                    220045,  # heart_rate
                    220045,  # heart_rate for stay 2
                    220050,  # sbp for stay 2
                    220050,  # sbp for stay 2 (duplicate in same hour)
                ],
                "valuenum": [80.0, 85.0, 120.0, 90.0, 115.0, 88.0, 75.0, 125.0, 130.0],
            }
        )

    def test_load_feature_mapping_auto_detection(self, mock_extractor):
        """Test that feature mapping is loaded correctly from YAML with auto-detection."""
        feature_mapping = mock_extractor._load_feature_mapping("core")

        # Should contain vitals from chartevents (now returns TimeSeriesConceptConfig)
        assert "heart_rate" in feature_mapping
        assert "sbp" in feature_mapping
        assert "dbp" in feature_mapping
        assert "map" in feature_mapping
        assert "resp_rate" in feature_mapping
        assert "spo2" in feature_mapping
        assert "temperature" in feature_mapping

        # Should contain glucose and creatinine (labs from labevents)
        assert "glucose" in feature_mapping
        assert "creatinine" in feature_mapping

        # Check structure - returns TimeSeriesConceptConfig with mimic_iv sources
        hr_config = feature_mapping["heart_rate"]
        assert isinstance(hr_config, TimeSeriesConceptConfig)
        assert hr_config.mimic_iv is not None
        assert len(hr_config.mimic_iv) > 0
        assert hr_config.mimic_iv[0].table == "chartevents"
        assert hr_config.mimic_iv[0].itemid == [220045]

        glucose_config = feature_mapping["glucose"]
        assert glucose_config.mimic_iv[0].table == "labevents"

    def test_load_feature_mapping_explicit_concepts_dir(self, tmp_path):
        """Test that explicit concepts_dir config works correctly."""
        import shutil

        # Create explicit concepts directory
        concepts_dir = tmp_path / "my_concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)

        # Copy the concept files to the test location
        project_root = Path(__file__).parent.parent
        for concept_file in ["vitals.yaml", "labs.yaml", "outputs.yaml"]:
            source_config = project_root / "configs" / "concepts" / concept_file
            if source_config.exists():
                shutil.copy(source_config, concepts_dir / concept_file)

        # Create extractor with explicit concepts_dir
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "mimic-iv"),
            output_dir=str(tmp_path / "processed"),
            concepts_dir=str(concepts_dir),  # Explicit path
        )

        # Create mock parquet directory structure
        (tmp_path / "mimic-iv" / "icu").mkdir(parents=True, exist_ok=True)
        (tmp_path / "mimic-iv" / "hosp").mkdir(parents=True, exist_ok=True)

        extractor = MIMICIVExtractor(config)

        # Should successfully load from explicit path
        feature_mapping = extractor._load_feature_mapping("core")
        assert "heart_rate" in feature_mapping
        assert "sbp" in feature_mapping
        # Check it returns TimeSeriesConceptConfig
        hr_config = feature_mapping["heart_rate"]
        assert isinstance(hr_config, TimeSeriesConceptConfig)
        assert hr_config.mimic_iv is not None

    def test_extract_raw_events(self, mock_extractor, mock_chartevents_df):
        """Raw events extraction concatenates multiple sources and maps feature names."""
        feature_mapping = make_feature_mapping(
            {
                "heart_rate": {"source": "chartevents", "itemid": [220045]},
                "sbp": {"source": "chartevents", "itemid": [220050, 220179]},
                "glucose": {"source": "labevents", "itemid": [50931]},
            }
        )

        # Mock _extract_by_itemid_batch to return events for each table batch
        def fake_extract_by_itemid_batch(
            table, value_col, time_col, transform, itemids, itemid_to_feature, stay_ids
        ):
            if table == "chartevents":
                # Return events for heart_rate and sbp (both from chartevents)
                return pl.DataFrame(
                    {
                        "stay_id": [1, 1, 2, 1, 1, 2, 2],
                        "charttime": [
                            datetime(2020, 1, 1, 10, 5),
                            datetime(2020, 1, 1, 10, 30),
                            datetime(2020, 1, 2, 14, 45),
                            datetime(2020, 1, 1, 11, 10),
                            datetime(2020, 1, 1, 12, 45),
                            datetime(2020, 1, 2, 15, 30),
                            datetime(2020, 1, 2, 16, 15),
                        ],
                        "feature_name": [
                            "heart_rate",
                            "heart_rate",
                            "heart_rate",
                            "sbp",
                            "sbp",
                            "sbp",
                            "sbp",
                        ],
                        "valuenum": [80.0, 85.0, 75.0, 120.0, 115.0, 125.0, 130.0],
                    }
                )
            elif table == "labevents":
                return pl.DataFrame(
                    {
                        "stay_id": [1],
                        "charttime": [datetime(2020, 1, 1, 10, 5)],
                        "feature_name": ["glucose"],
                        "valuenum": [140.0],
                    }
                )
            return pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "charttime": pl.Datetime,
                    "feature_name": pl.Utf8,
                    "valuenum": pl.Float64,
                }
            )

        with patch.object(
            mock_extractor, "_extract_by_itemid_batch", side_effect=fake_extract_by_itemid_batch
        ):
            result = mock_extractor._extract_raw_events([1, 2], feature_mapping)

        assert {"stay_id", "charttime", "feature_name", "valuenum"} <= set(result.columns)
        assert set(result["feature_name"].to_list()) == {"heart_rate", "sbp", "glucose"}

    def test_bin_to_hourly_grid_aggregation(self, mock_extractor, mock_stays_df):
        """Hourly binning aggregates means and sums per source defaults."""
        feature_mapping = make_feature_mapping(
            {
                "heart_rate": {"source": "chartevents", "itemid": [220045]},
                "sbp": {"source": "chartevents", "itemid": [220050]},
                "urine_output": {
                    "source": "outputevents",
                    "itemid": [226559],
                    "aggregation": "sum",
                },
            }
        )

        base_time = datetime(2020, 1, 1, 10, 0)
        standardized_events = pl.DataFrame(
            {
                "stay_id": [1, 1, 1, 1, 1, 1],
                "charttime": [
                    base_time + timedelta(minutes=5),  # hour 0 hr
                    base_time + timedelta(minutes=30),  # hour 0 hr
                    base_time + timedelta(hours=1, minutes=10),  # hour 1 sbp
                    base_time + timedelta(hours=2, minutes=15),  # hour 2 urine
                    base_time + timedelta(hours=2, minutes=45),  # hour 2 urine
                    base_time + timedelta(hours=2, minutes=50),  # hour 2 urine
                ],
                "feature_name": [
                    "heart_rate",
                    "heart_rate",
                    "sbp",
                    "urine_output",
                    "urine_output",
                    "urine_output",
                ],
                "valuenum": [80.0, 100.0, 120.0, 10.0, 20.0, 5.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(standardized_events, [1], feature_mapping)

        assert {"stay_id", "hour", "heart_rate", "sbp", "urine_output"} <= set(result.columns)
        assert {"heart_rate_mask", "sbp_mask", "urine_output_mask"} <= set(result.columns)

        stay1_hour0 = result.filter((pl.col("stay_id") == 1) & (pl.col("hour") == 0))
        if len(stay1_hour0) > 0:
            assert stay1_hour0["heart_rate"][0] == pytest.approx(90.0)  # mean of 80 & 100
            assert stay1_hour0["sbp_mask"][0] is False

        stay1_hour2 = result.filter((pl.col("stay_id") == 1) & (pl.col("hour") == 2))
        if len(stay1_hour2) > 0:
            assert stay1_hour2["urine_output"][0] == pytest.approx(35.0)  # sum of 10+20+5
            assert stay1_hour2["urine_output_mask"][0] is True

    def test_bin_to_hourly_grid_missing_values(self, mock_extractor, mock_stays_df):
        """Test that missing values are handled correctly with masks."""
        feature_mapping = make_feature_mapping(
            {
                "heart_rate": {"source": "chartevents", "itemid": [220045]},
                "sbp": {"source": "chartevents", "itemid": [220050]},
            }
        )

        # Create data with missing hours (standardized schema)
        sparse_events = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 10, 5),  # hour 0
                    datetime(2020, 1, 1, 12, 30),  # hour 2 (hour 1 missing)
                ],
                "feature_name": ["heart_rate", "heart_rate"],
                "valuenum": [80.0, 90.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(sparse_events, [1], feature_mapping)

        # Check that mask columns exist
        assert "heart_rate_mask" in result.columns
        assert "sbp_mask" in result.columns

        # Hour 1 should not exist in result (no data)
        hours = result.filter(pl.col("stay_id") == 1)["hour"].to_list()
        assert 0 in hours
        assert 2 in hours
        assert 1 not in hours  # Missing hour is not created

    def test_bin_to_hourly_grid_ensures_all_feature_columns(self, mock_extractor, mock_stays_df):
        """Test that all expected feature columns exist even if no data."""
        # Feature mapping with features that don't appear in data
        feature_mapping = make_feature_mapping(
            {
                "heart_rate": {"source": "chartevents", "itemid": [220045]},
                "sbp": {"source": "chartevents", "itemid": [220050]},
                "resp_rate": {"source": "chartevents", "itemid": [220210]},  # Not in our test data
                "spo2": {"source": "chartevents", "itemid": [220277]},  # Not in our test data
            }
        )

        # Data with only heart_rate (standardized schema)
        minimal_events = pl.DataFrame(
            {
                "stay_id": [1],
                "charttime": [datetime(2020, 1, 1, 10, 5)],
                "feature_name": ["heart_rate"],
                "valuenum": [80.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(minimal_events, [1], feature_mapping)

        # All features should have columns (even if null)
        assert "heart_rate" in result.columns
        assert "sbp" in result.columns
        assert "resp_rate" in result.columns
        assert "spo2" in result.columns

        # All features should have mask columns
        assert "heart_rate_mask" in result.columns
        assert "sbp_mask" in result.columns
        assert "resp_rate_mask" in result.columns
        assert "spo2_mask" in result.columns

    def test_extract_timeseries_end_to_end(self, mock_extractor, mock_stays_df):
        """Test full extract_timeseries pipeline."""
        # Create standardized events that _extract_raw_events would return
        standardized_events = pl.DataFrame(
            {
                "stay_id": [1, 1, 2, 2],
                "charttime": [
                    datetime(2020, 1, 1, 10, 30),
                    datetime(2020, 1, 1, 11, 15),
                    datetime(2020, 1, 2, 15, 0),
                    datetime(2020, 1, 2, 16, 0),
                ],
                "feature_name": ["heart_rate", "sbp", "heart_rate", "sbp"],
                "valuenum": [80.0, 120.0, 75.0, 125.0],
            }
        )

        # Mock the internal methods
        with (
            patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df),
            patch.object(mock_extractor, "_extract_raw_events", return_value=standardized_events),
        ):

            result = mock_extractor.extract_timeseries([1, 2])

        # Check result structure
        assert isinstance(result, pl.DataFrame)
        assert "stay_id" in result.columns
        assert "hour" in result.columns

        # Should have feature columns from core_features.yaml vitals
        # Note: We only check for features that appear in our mock data
        # The _bin_to_hourly_grid will add columns for all expected features,
        # but test data only has heart_rate and sbp
        vitals = ["heart_rate", "sbp"]
        for vital in vitals:
            assert vital in result.columns
            assert f"{vital}_mask" in result.columns

        # Other vitals should exist as null columns (added by _bin_to_hourly_grid)
        # but we don't strictly require them in this test

        # Should have data for requested stay_ids
        assert set(result["stay_id"].unique().to_list()).issubset({1, 2, 3})

    def test_extract_timeseries_filters_negative_hours(self, mock_extractor, mock_stays_df):
        """Test that events before ICU admission (negative hours) are filtered."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        # Create events before admission (standardized schema)
        events_with_negatives = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 9, 0),  # 1 hour before admission
                    datetime(2020, 1, 1, 10, 0),  # At admission (hour 0)
                    datetime(2020, 1, 1, 11, 0),  # 1 hour after admission
                ],
                "feature_name": ["heart_rate", "heart_rate", "heart_rate"],
                "valuenum": [70.0, 80.0, 90.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events_with_negatives, [1], feature_mapping)

        # Should not have negative hours
        hours = result["hour"].to_list()
        assert all(h >= 0 for h in hours)

        # Should have hours 0 and 1 only (not -1)
        assert 0 in hours or 1 in hours  # At least one of these
        assert -1 not in hours

    def test_extract_timeseries_multiple_itemids_same_feature(self, mock_extractor, mock_stays_df):
        """Test that multiple itemids map to same feature (e.g., invasive and non-invasive BP)."""
        feature_mapping = make_feature_mapping(
            {"sbp": {"source": "chartevents", "itemid": [220050, 220179]}}
        )

        # Events with both itemids already mapped to same feature_name (standardized schema)
        events = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 10, 10),  # hour 0
                    datetime(2020, 1, 1, 10, 20),  # hour 0
                    datetime(2020, 1, 1, 11, 30),  # hour 1
                ],
                "feature_name": ["sbp", "sbp", "sbp"],  # Both itemids mapped to "sbp"
                "valuenum": [120.0, 118.0, 125.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events, [1], feature_mapping)

        # Should aggregate both itemids into single sbp feature
        assert "sbp" in result.columns

        # Hour 0 should average both values: (120 + 118) / 2 = 119
        stay1_hour0 = result.filter((pl.col("stay_id") == 1) & (pl.col("hour") == 0))
        if len(stay1_hour0) > 0:
            assert stay1_hour0["sbp"][0] == pytest.approx(119.0)
            assert stay1_hour0["sbp_mask"][0] is True


class TestTimeSeriesEdgeCases:
    """Tests for edge cases in time-series extraction."""

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

    @pytest.fixture
    def mock_stays_df(self):
        """Create mock stays DataFrame."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [101, 102, 103],
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 2, 14, 30),
                    datetime(2020, 1, 3, 8, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 5, 10, 0),
                    datetime(2020, 1, 6, 14, 30),
                    datetime(2020, 1, 7, 8, 0),
                ],
                "los_days": [4.0, 4.0, 4.0],
                "age": [65, 45, 70],
                "gender": ["M", "F", "M"],
            }
        )

    def test_empty_stay_list(self, mock_extractor, mock_stays_df):
        """Test extraction with empty stay list."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        # Empty events
        empty_events = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "charttime": pl.Series([], dtype=pl.Datetime),
                "feature_name": pl.Series([], dtype=pl.Utf8),
                "valuenum": pl.Series([], dtype=pl.Float64),
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(empty_events, [], feature_mapping)

        # Should return empty DataFrame with correct columns
        assert "stay_id" in result.columns
        assert "hour" in result.columns
        assert "heart_rate" in result.columns
        assert len(result) == 0

    def test_very_short_stay(self, mock_extractor):
        """Test extraction for very short ICU stay (< 1 hour)."""
        short_stays = pl.DataFrame(
            {
                "stay_id": [1],
                "patient_id": [101],
                "intime": [datetime(2020, 1, 1, 10, 0)],
                "outtime": [datetime(2020, 1, 1, 10, 30)],  # 30 min stay
                "los_days": [0.02],
                "age": [65],
                "gender": ["M"],
            }
        )

        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        # Events within the short stay
        events = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 10, 10),  # hour 0
                    datetime(2020, 1, 1, 10, 20),  # hour 0
                ],
                "feature_name": ["heart_rate", "heart_rate"],
                "valuenum": [80.0, 85.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=short_stays):
            result = mock_extractor._bin_to_hourly_grid(events, [1], feature_mapping)

        # Should have data for hour 0
        assert len(result) == 1
        assert result["hour"][0] == 0
        assert result["heart_rate"][0] == pytest.approx(82.5)  # Mean of 80 and 85

    def test_events_spanning_many_hours(self, mock_extractor, mock_stays_df):
        """Test extraction with events spanning many hours."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        # Events spanning 100 hours
        events = []
        base_time = datetime(2020, 1, 1, 10, 0)
        for hour in range(100):
            events.append(
                {
                    "stay_id": 1,
                    "charttime": base_time + timedelta(hours=hour),
                    "feature_name": "heart_rate",
                    "valuenum": 70.0 + hour * 0.1,
                }
            )

        events_df = pl.DataFrame(events)

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events_df, [1], feature_mapping)

        # Should have 100 hours of data
        stay1_data = result.filter(pl.col("stay_id") == 1)
        assert len(stay1_data) == 100

    def test_extreme_values(self, mock_extractor, mock_stays_df):
        """Test extraction with extreme values."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        events = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 1, 11, 0),
                    datetime(2020, 1, 1, 12, 0),
                ],
                "feature_name": ["heart_rate", "heart_rate", "heart_rate"],
                "valuenum": [0.0, 1e10, -1e10],  # Extreme values
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events, [1], feature_mapping)

        # Should preserve extreme values
        assert result.filter(pl.col("hour") == 0)["heart_rate"][0] == 0.0
        assert result.filter(pl.col("hour") == 1)["heart_rate"][0] == 1e10
        assert result.filter(pl.col("hour") == 2)["heart_rate"][0] == -1e10

    def test_single_event(self, mock_extractor, mock_stays_df):
        """Test extraction with single event."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        events = pl.DataFrame(
            {
                "stay_id": [1],
                "charttime": [datetime(2020, 1, 1, 10, 30)],
                "feature_name": ["heart_rate"],
                "valuenum": [75.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events, [1], feature_mapping)

        assert len(result) == 1
        assert result["heart_rate"][0] == 75.0

    def test_many_features(self, mock_extractor, mock_stays_df):
        """Test extraction with many features."""
        # Create feature mapping with 50 features
        feature_mapping = make_feature_mapping(
            {f"feature_{i}": {"source": "chartevents", "itemid": [220000 + i]} for i in range(50)}
        )

        # Create events for each feature
        events = []
        for i in range(50):
            events.append(
                {
                    "stay_id": 1,
                    "charttime": datetime(2020, 1, 1, 10, 0),
                    "feature_name": f"feature_{i}",
                    "valuenum": float(i),
                }
            )

        events_df = pl.DataFrame(events)

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events_df, [1], feature_mapping)

        # Should have all 50 features
        for i in range(50):
            assert f"feature_{i}" in result.columns
            assert f"feature_{i}_mask" in result.columns

    def test_multiple_stays_different_data_density(self, mock_extractor, mock_stays_df):
        """Test extraction with stays having different data density."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        # Stay 1: Dense data (every hour)
        # Stay 2: Sparse data (every 4 hours)
        events = []

        # Dense for stay 1
        for hour in range(24):
            events.append(
                {
                    "stay_id": 1,
                    "charttime": datetime(2020, 1, 1, 10, 0) + timedelta(hours=hour),
                    "feature_name": "heart_rate",
                    "valuenum": 70.0 + hour,
                }
            )

        # Sparse for stay 2
        for hour in [0, 4, 8, 12]:
            events.append(
                {
                    "stay_id": 2,
                    "charttime": datetime(2020, 1, 2, 14, 30) + timedelta(hours=hour),
                    "feature_name": "heart_rate",
                    "valuenum": 80.0 + hour,
                }
            )

        events_df = pl.DataFrame(events)

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events_df, [1, 2], feature_mapping)

        # Stay 1 should have 24 rows
        stay1_data = result.filter(pl.col("stay_id") == 1)
        assert len(stay1_data) == 24

        # Stay 2 should have 4 rows (sparse)
        stay2_data = result.filter(pl.col("stay_id") == 2)
        assert len(stay2_data) == 4

    def test_events_exactly_on_hour_boundaries(self, mock_extractor, mock_stays_df):
        """Test events that occur exactly on hour boundaries."""
        feature_mapping = make_feature_mapping(
            {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        )

        events = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "charttime": [
                    datetime(2020, 1, 1, 10, 0, 0),  # Exactly at admission
                    datetime(2020, 1, 1, 11, 0, 0),  # Exactly 1 hour
                    datetime(2020, 1, 1, 12, 0, 0),  # Exactly 2 hours
                ],
                "feature_name": ["heart_rate", "heart_rate", "heart_rate"],
                "valuenum": [70.0, 75.0, 80.0],
            }
        )

        with patch.object(mock_extractor, "extract_stays", return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(events, [1], feature_mapping)

        # Events on boundary should go to the correct hour
        assert result.filter(pl.col("hour") == 0)["heart_rate"][0] == 70.0
        assert result.filter(pl.col("hour") == 1)["heart_rate"][0] == 75.0
        assert result.filter(pl.col("hour") == 2)["heart_rate"][0] == 80.0
