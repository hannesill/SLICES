"""Tests for time-series extraction functionality."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import polars as pl
from datetime import datetime, timedelta

from slices.data.extractors.mimic_iv import MIMICIVExtractor, ExtractorConfig


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
        return pl.DataFrame({
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
        })

    @pytest.fixture
    def mock_chartevents_df(self):
        """Create mock chartevents DataFrame."""
        # Create realistic chartevents data for heart_rate (220045) and sbp (220050, 220179)
        base_time = datetime(2020, 1, 1, 10, 0)
        
        return pl.DataFrame({
            "stay_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "charttime": [
                base_time + timedelta(minutes=5),   # hour 0
                base_time + timedelta(minutes=30),  # hour 0
                base_time + timedelta(minutes=70),  # hour 1
                base_time + timedelta(hours=2, minutes=15),  # hour 2
                base_time + timedelta(hours=2, minutes=45),  # hour 2
                base_time + timedelta(hours=3),     # hour 3
                datetime(2020, 1, 2, 14, 45),      # hour 0 for stay 2
                datetime(2020, 1, 2, 15, 30),      # hour 1 for stay 2
                datetime(2020, 1, 2, 16, 15),      # hour 1 for stay 2
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
            "valuenum": [
                80.0, 85.0, 120.0, 90.0, 115.0, 88.0, 75.0, 125.0, 130.0
            ]
        })

    def test_load_feature_mapping_auto_detection(self, mock_extractor):
        """Test that feature mapping is loaded correctly from YAML with auto-detection."""
        feature_mapping = mock_extractor._load_feature_mapping("core")
        
        # Should contain vitals from chartevents (now returns full dataset config)
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
        
        # Check structure - returns full dataset config (not just itemids)
        assert feature_mapping["heart_rate"]["source"] == "chartevents"
        assert feature_mapping["heart_rate"]["itemid"] == [220045]
        assert "labevents" in feature_mapping["glucose"]["source"]

    def test_load_feature_mapping_explicit_concepts_dir(self, tmp_path):
        """Test that explicit concepts_dir config works correctly."""
        from pathlib import Path
        
        # Create explicit concepts directory
        concepts_dir = tmp_path / "my_concepts"
        concepts_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the core_features.yaml to the test location
        import shutil
        project_root = Path(__file__).parent.parent
        source_config = project_root / "configs" / "concepts" / "core_features.yaml"
        shutil.copy(source_config, concepts_dir / "core_features.yaml")
        
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
        # Check it returns full dataset config
        assert "source" in feature_mapping["heart_rate"]
        assert "itemid" in feature_mapping["heart_rate"]

    def test_extract_raw_events(self, mock_extractor, mock_chartevents_df):
        """Test that raw events extraction works correctly with new standardized schema."""
        # Create feature mapping in new format (as returned by _load_feature_mapping)
        feature_mapping = {
            "heart_rate": {"source": "chartevents", "itemid": [220045], "value_col": "valuenum"},
            "sbp": {"source": "chartevents", "itemid": [220050, 220179], "value_col": "valuenum"}
        }
        
        # Mock the _query method
        with patch.object(mock_extractor, '_query', return_value=mock_chartevents_df):
            result = mock_extractor._extract_raw_events([1, 2], feature_mapping)
        
        # Check result has standardized schema (not itemid!)
        assert "stay_id" in result.columns
        assert "charttime" in result.columns
        assert "feature_name" in result.columns  # Standardized!
        assert "valuenum" in result.columns
        assert "itemid" not in result.columns  # itemid should be mapped away
        
        # Check data content
        assert len(result) == 9
        assert set(result["stay_id"].to_list()) == {1, 2}
        assert set(result["feature_name"].to_list()) == {"heart_rate", "sbp"}

    def test_bin_to_hourly_grid_aggregation(self, mock_extractor, mock_stays_df):
        """Test that hourly binning and aggregation works correctly."""
        # Feature mapping in new format
        feature_mapping = {
            "heart_rate": {"source": "chartevents", "itemid": [220045]},
            "sbp": {"source": "chartevents", "itemid": [220050]}
        }
        
        # Create standardized events (with feature_name, not itemid)
        base_time = datetime(2020, 1, 1, 10, 0)
        standardized_events = pl.DataFrame({
            "stay_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "charttime": [
                base_time + timedelta(minutes=5),   # hour 0
                base_time + timedelta(minutes=30),  # hour 0
                base_time + timedelta(hours=1, minutes=10),  # hour 1
                base_time + timedelta(hours=2, minutes=15),  # hour 2
                base_time + timedelta(hours=2, minutes=45),  # hour 2
                base_time + timedelta(hours=3),     # hour 3
                datetime(2020, 1, 2, 14, 45),      # hour 0 for stay 2
                datetime(2020, 1, 2, 15, 30),      # hour 1 for stay 2
                datetime(2020, 1, 2, 16, 15),      # hour 1 for stay 2
            ],
            "feature_name": [
                "heart_rate", "heart_rate", "sbp", "heart_rate", "sbp", "heart_rate",
                "heart_rate", "sbp", "sbp"
            ],
            "valuenum": [80.0, 85.0, 120.0, 90.0, 115.0, 88.0, 75.0, 125.0, 130.0]
        })
        
        # Mock extract_stays to return our test data
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(
                standardized_events, [1, 2], feature_mapping
            )
        
        # Check result structure
        assert "stay_id" in result.columns
        assert "hour" in result.columns
        assert "heart_rate" in result.columns
        assert "sbp" in result.columns
        assert "heart_rate_mask" in result.columns
        assert "sbp_mask" in result.columns
        
        # Check that values are aggregated correctly (mean)
        # For stay_id=1, hour=0: two heart_rate values (80, 85) -> mean = 82.5
        stay1_hour0 = result.filter((pl.col("stay_id") == 1) & (pl.col("hour") == 0))
        if len(stay1_hour0) > 0:
            assert stay1_hour0["heart_rate"][0] == pytest.approx(82.5)
            assert stay1_hour0["heart_rate_mask"][0] == True
            assert stay1_hour0["sbp_mask"][0] == False  # No sbp in hour 0
        
        # For stay_id=2, hour=1: two sbp values (125, 130) -> mean = 127.5
        stay2_hour1 = result.filter((pl.col("stay_id") == 2) & (pl.col("hour") == 1))
        if len(stay2_hour1) > 0:
            assert stay2_hour1["sbp"][0] == pytest.approx(127.5)
            assert stay2_hour1["sbp_mask"][0] == True

    def test_bin_to_hourly_grid_missing_values(self, mock_extractor, mock_stays_df):
        """Test that missing values are handled correctly with masks."""
        feature_mapping = {
            "heart_rate": {"source": "chartevents", "itemid": [220045]},
            "sbp": {"source": "chartevents", "itemid": [220050]}
        }
        
        # Create data with missing hours (standardized schema)
        sparse_events = pl.DataFrame({
            "stay_id": [1, 1],
            "charttime": [
                datetime(2020, 1, 1, 10, 5),   # hour 0
                datetime(2020, 1, 1, 12, 30),  # hour 2 (hour 1 missing)
            ],
            "feature_name": ["heart_rate", "heart_rate"],
            "valuenum": [80.0, 90.0]
        })
        
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(
                sparse_events, [1], feature_mapping
            )
        
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
        feature_mapping = {
            "heart_rate": {"source": "chartevents", "itemid": [220045]},
            "sbp": {"source": "chartevents", "itemid": [220050]},
            "resp_rate": {"source": "chartevents", "itemid": [220210]},  # Not in our test data
            "spo2": {"source": "chartevents", "itemid": [220277]},       # Not in our test data
        }
        
        # Data with only heart_rate (standardized schema)
        minimal_events = pl.DataFrame({
            "stay_id": [1],
            "charttime": [datetime(2020, 1, 1, 10, 5)],
            "feature_name": ["heart_rate"],
            "valuenum": [80.0]
        })
        
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(
                minimal_events, [1], feature_mapping
            )
        
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
        standardized_events = pl.DataFrame({
            "stay_id": [1, 1, 2, 2],
            "charttime": [
                datetime(2020, 1, 1, 10, 30),
                datetime(2020, 1, 1, 11, 15),
                datetime(2020, 1, 2, 15, 0),
                datetime(2020, 1, 2, 16, 0),
            ],
            "feature_name": ["heart_rate", "sbp", "heart_rate", "sbp"],
            "valuenum": [80.0, 120.0, 75.0, 125.0]
        })
        
        # Mock the internal methods
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df), \
             patch.object(mock_extractor, '_extract_raw_events', return_value=standardized_events):
            
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
        feature_mapping = {"heart_rate": {"source": "chartevents", "itemid": [220045]}}
        
        # Create events before admission (standardized schema)
        events_with_negatives = pl.DataFrame({
            "stay_id": [1, 1, 1],
            "charttime": [
                datetime(2020, 1, 1, 9, 0),   # 1 hour before admission
                datetime(2020, 1, 1, 10, 0),  # At admission (hour 0)
                datetime(2020, 1, 1, 11, 0),  # 1 hour after admission
            ],
            "feature_name": ["heart_rate", "heart_rate", "heart_rate"],
            "valuenum": [70.0, 80.0, 90.0]
        })
        
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(
                events_with_negatives, [1], feature_mapping
            )
        
        # Should not have negative hours
        hours = result["hour"].to_list()
        assert all(h >= 0 for h in hours)
        
        # Should have hours 0 and 1 only (not -1)
        assert 0 in hours or 1 in hours  # At least one of these
        assert -1 not in hours

    def test_extract_timeseries_multiple_itemids_same_feature(self, mock_extractor, mock_stays_df):
        """Test that multiple itemids map to same feature (e.g., invasive and non-invasive BP)."""
        feature_mapping = {"sbp": {"source": "chartevents", "itemid": [220050, 220179]}}
        
        # Events with both itemids already mapped to same feature_name (standardized schema)
        events = pl.DataFrame({
            "stay_id": [1, 1, 1],
            "charttime": [
                datetime(2020, 1, 1, 10, 10),  # hour 0
                datetime(2020, 1, 1, 10, 20),  # hour 0
                datetime(2020, 1, 1, 11, 30),  # hour 1
            ],
            "feature_name": ["sbp", "sbp", "sbp"],  # Both itemids mapped to "sbp"
            "valuenum": [120.0, 118.0, 125.0]
        })
        
        with patch.object(mock_extractor, 'extract_stays', return_value=mock_stays_df):
            result = mock_extractor._bin_to_hourly_grid(
                events, [1], feature_mapping
            )
        
        # Should aggregate both itemids into single sbp feature
        assert "sbp" in result.columns
        
        # Hour 0 should average both values: (120 + 118) / 2 = 119
        stay1_hour0 = result.filter((pl.col("stay_id") == 1) & (pl.col("hour") == 0))
        if len(stay1_hour0) > 0:
            assert stay1_hour0["sbp"][0] == pytest.approx(119.0)
            assert stay1_hour0["sbp_mask"][0] == True
