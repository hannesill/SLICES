"""Tests for BaseExtractor functionality."""

import math
from datetime import datetime
from typing import List

import numpy as np
import polars as pl
import pytest
import yaml
from slices.data.extractors.base import BaseExtractor, ExtractorConfig


class MockExtractor(BaseExtractor):
    """Mock implementation of BaseExtractor for testing."""

    def _get_dataset_name(self) -> str:
        return "mock"

    def extract_stays(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "patient_id": [100, 101, 102],
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

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        stays = self.extract_stays().filter(pl.col("stay_id").is_in(stay_ids))
        events = []
        for stay_row in stays.iter_rows(named=True):
            stay_id = stay_row["stay_id"]
            for hour in [0, 1, 2]:
                events.append(
                    {
                        "stay_id": stay_id,
                        "hour": hour,
                        "heart_rate": 70.0 + hour,
                        "sbp": 120.0 + hour if hour != 1 else None,
                        "heart_rate_mask": True,
                        "sbp_mask": hour != 1,
                    }
                )
        if not events:
            return pl.DataFrame(
                {
                    "stay_id": pl.Series([], dtype=pl.Int64),
                    "hour": pl.Series([], dtype=pl.Int64),
                    "heart_rate": pl.Series([], dtype=pl.Float64),
                    "sbp": pl.Series([], dtype=pl.Float64),
                    "heart_rate_mask": pl.Series([], dtype=pl.Boolean),
                    "sbp_mask": pl.Series([], dtype=pl.Boolean),
                }
            )
        return pl.DataFrame(events)

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        if source_name == "mortality_info":
            return pl.DataFrame(
                {
                    "stay_id": stay_ids,
                    "date_of_death": [None] * len(stay_ids),
                    "hospital_expire_flag": [0] * len(stay_ids),
                    "dischtime": [datetime(2020, 1, 5, 10, 0)] * len(stay_ids),
                    "discharge_location": ["HOME"] * len(stay_ids),
                }
            )
        elif source_name == "stays":
            return self.extract_stays().filter(pl.col("stay_id").is_in(stay_ids))
        else:
            raise ValueError(f"Unknown data source: {source_name}")

    def run(self) -> None:
        """Run mock extraction pipeline using base class utilities."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stays = self.extract_stays()

        # Filter short stays
        min_hours = self.config.min_stay_hours
        if min_hours > 0:
            stays = stays.filter(pl.col("los_days") * 24 >= min_hours)

        if len(stays) == 0:
            return

        stay_ids = stays["stay_id"].to_list()

        # Extract timeseries
        timeseries = self.extract_timeseries(stay_ids)
        feature_names = [
            c
            for c in timeseries.columns
            if c not in ["stay_id", "hour"] and not c.endswith("_mask")
        ]

        # Create dense timeseries
        dense = self._create_dense_timeseries(timeseries, stays, feature_names)

        # Extract labels
        task_configs = self._load_task_configs(self.config.tasks or [])
        labels = self.extract_labels(stay_ids, task_configs)

        # Save files atomically
        self._atomic_write(
            self.output_dir / "static.parquet",
            lambda tmp: stays.write_parquet(tmp),
        )
        self._atomic_write(
            self.output_dir / "timeseries.parquet",
            lambda tmp: dense.write_parquet(tmp),
        )
        self._atomic_write(
            self.output_dir / "labels.parquet",
            lambda tmp: labels.write_parquet(tmp),
        )

        metadata = {
            "dataset": self._get_dataset_name(),
            "feature_set": self.config.feature_set,
            "seq_length_hours": self.config.seq_length_hours,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "n_stays": len(stays),
            "task_names": self.config.tasks or [],
        }
        self._atomic_write(
            self.output_dir / "metadata.yaml",
            lambda tmp: yaml.dump(metadata, open(tmp, "w"), default_flow_style=False),
            suffix=".yaml",
        )


@pytest.fixture
def temp_parquet_structure(tmp_path):
    """Create temporary Parquet directory structure with test files."""
    parquet_root = tmp_path / "parquet"
    parquet_root.mkdir(parents=True)

    # Create dummy Parquet files
    patients_df = pl.DataFrame({"subject_id": [1, 2], "gender": ["M", "F"]})
    patients_df.write_parquet(parquet_root / "patients.parquet")

    return parquet_root


class TestBaseExtractor:
    """Test BaseExtractor abstract class."""

    def test_initialization_success(self, temp_parquet_structure):
        """Test successful initialization with valid directory."""
        config = ExtractorConfig(parquet_root=str(temp_parquet_structure))
        extractor = MockExtractor(config)

        assert extractor.parquet_root == temp_parquet_structure

    def test_initialization_nonexistent_directory(self, tmp_path):
        """Test that nonexistent directory raises ValueError."""
        nonexistent = tmp_path / "nonexistent"
        config = ExtractorConfig(parquet_root=str(nonexistent))

        with pytest.raises(ValueError, match="Parquet directory not found"):
            MockExtractor(config)

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
        assert len(timeseries) >= 2
        assert "stay_id" in timeseries.columns
        assert "hour" in timeseries.columns

        # extract_labels with empty list returns just stay_ids
        labels = extractor.extract_labels([1, 2], [])
        assert isinstance(labels, pl.DataFrame)
        assert len(labels) == 2
        assert "stay_id" in labels.columns

    def test_run_method_exists_and_callable(self, temp_parquet_structure, tmp_path):
        """Test that run method exists and is callable."""
        output_dir = tmp_path / "output"
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure),
            output_dir=str(output_dir),
            tasks=[],
        )
        extractor = MockExtractor(config)

        assert hasattr(extractor, "run")
        assert callable(extractor.run)

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

        assert extractor1 is not extractor2
        assert extractor1.config is not extractor2.config


class TestCreateDenseTimeseries:
    """Tests for _create_dense_timeseries method."""

    def test_basic_dense_conversion(self, temp_parquet_structure):
        """Test basic conversion from sparse to dense format."""
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure),
            seq_length_hours=6,
        )
        extractor = MockExtractor(config)

        # Create sparse timeseries with gaps
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 1, 2, 2],
                "hour": [0, 2, 4, 0, 1],
                "heart_rate": [70.0, 72.0, 74.0, 80.0, 82.0],
                "sbp": [120.0, None, 125.0, 115.0, 118.0],
                "heart_rate_mask": [True, True, True, True, True],
                "sbp_mask": [True, False, True, True, True],
            }
        )

        stays = extractor.extract_stays()
        feature_names = ["heart_rate", "sbp"]

        result = extractor._create_dense_timeseries(sparse, stays, feature_names)

        # Check structure
        assert "stay_id" in result.columns
        assert "timeseries" in result.columns
        assert "mask" in result.columns

        # Check shapes (should have 3 stays - all stays in stays DataFrame are included,
        # even those without data, to prevent dimension mismatch with labels)
        assert len(result) == 3

        # Check timeseries shape for stay 1
        stay1 = result.filter(pl.col("stay_id") == 1)
        ts1 = stay1["timeseries"][0]
        mask1 = stay1["mask"][0]

        assert len(ts1) == 6  # seq_length_hours
        assert len(ts1[0]) == 2  # 2 features

        # Hour 0 should have values
        assert ts1[0][0] == 70.0  # heart_rate
        assert ts1[0][1] == 120.0  # sbp
        assert mask1[0][0] is True
        assert mask1[0][1] is True

        # Hour 1 should be NaN (no data in sparse)
        assert np.isnan(ts1[1][0])
        assert mask1[1][0] is False

    def test_dense_conversion_truncates_to_seq_length(self, temp_parquet_structure):
        """Test that data beyond seq_length is truncated."""
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure),
            seq_length_hours=3,  # Short sequence
        )
        extractor = MockExtractor(config)

        # Create sparse with data at hour 5 (beyond seq_length)
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "hour": [0, 2, 5],  # Hour 5 is beyond seq_length
                "heart_rate": [70.0, 72.0, 99.0],
                "sbp": [120.0, 122.0, 199.0],
                "heart_rate_mask": [True, True, True],
                "sbp_mask": [True, True, True],
            }
        )

        stays = extractor.extract_stays()
        feature_names = ["heart_rate", "sbp"]

        result = extractor._create_dense_timeseries(sparse, stays, feature_names)

        stay1 = result.filter(pl.col("stay_id") == 1)
        ts1 = stay1["timeseries"][0]

        # Should have exactly seq_length_hours
        assert len(ts1) == 3

        # Hour 5 data should not appear
        assert ts1[0][0] == 70.0
        assert ts1[2][0] == 72.0
        # All hours in sequence should be accounted for
        assert np.isnan(ts1[1][0])  # Hour 1 missing

    def test_dense_conversion_empty_stays(self, temp_parquet_structure):
        """Test dense conversion with no data for some stays.

        All stays should be included in output, even those with no observations,
        to prevent dimension mismatch between timeseries.parquet and labels.parquet.
        """
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure),
            seq_length_hours=4,
        )
        extractor = MockExtractor(config)

        # Sparse with data for only stay 1
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "heart_rate": [70.0],
                "sbp": [120.0],
                "heart_rate_mask": [True],
                "sbp_mask": [True],
            }
        )

        stays = extractor.extract_stays()  # Returns 3 stays
        feature_names = ["heart_rate", "sbp"]

        result = extractor._create_dense_timeseries(sparse, stays, feature_names)

        # ALL stays should be included (even those without observations)
        assert len(result) == 3
        assert set(result["stay_id"].to_list()) == {1, 2, 3}

        # Stay 1 should have data
        stay1 = result.filter(pl.col("stay_id") == 1)
        assert stay1["mask"][0][0][0] is True  # First hour, first feature observed

        # Stays 2 and 3 should have all-NaN values and all-False masks
        stay2 = result.filter(pl.col("stay_id") == 2)
        stay3 = result.filter(pl.col("stay_id") == 3)
        for stay in [stay2, stay3]:
            mask = stay["mask"][0]
            # All mask values should be False for stays with no data
            assert all(not m for hour in mask for m in hour)

    def test_dense_conversion_preserves_mask(self, temp_parquet_structure):
        """Test that observation masks are correctly preserved."""
        config = ExtractorConfig(
            parquet_root=str(temp_parquet_structure),
            seq_length_hours=4,
        )
        extractor = MockExtractor(config)

        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "hour": [0, 1],
                "heart_rate": [70.0, 75.0],
                "sbp": [120.0, None],  # sbp missing at hour 1
                "heart_rate_mask": [True, True],
                "sbp_mask": [True, False],  # Explicitly marked as not observed
            }
        )

        stays = extractor.extract_stays()
        feature_names = ["heart_rate", "sbp"]

        result = extractor._create_dense_timeseries(sparse, stays, feature_names)

        mask = result.filter(pl.col("stay_id") == 1)["mask"][0]

        # Hour 0: both observed
        assert mask[0][0] is True  # heart_rate
        assert mask[0][1] is True  # sbp

        # Hour 1: only heart_rate observed
        assert mask[1][0] is True  # heart_rate
        assert mask[1][1] is False  # sbp not observed


class TestExtractorRun:
    """Tests for the run() method - the main pipeline entry point."""

    @pytest.fixture
    def temp_extraction_setup(self, tmp_path):
        """Create temporary setup for run() tests."""
        parquet_root = tmp_path / "parquet"
        output_dir = tmp_path / "output"
        tasks_dir = tmp_path / "tasks"

        # Create parquet structure
        parquet_root.mkdir(parents=True)

        # Create minimal parquet files
        patients_df = pl.DataFrame({"subject_id": [1, 2], "gender": ["M", "F"]})
        patients_df.write_parquet(parquet_root / "patients.parquet")

        # Create task config directory and files
        tasks_dir.mkdir(parents=True)

        # Create a minimal mortality task config
        mortality_config = {
            "task_name": "mortality_hospital",
            "task_type": "binary_classification",
            "prediction_window_hours": None,
            "label_sources": ["stays", "mortality_info"],
            "primary_metric": "auroc",
        }
        with open(tasks_dir / "mortality_hospital.yaml", "w") as f:
            yaml.dump(mortality_config, f)

        return {
            "parquet_root": parquet_root,
            "output_dir": output_dir,
            "tasks_dir": tasks_dir,
        }

    def test_run_creates_output_directory(self, temp_extraction_setup):
        """Test that run() creates the output directory."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=["mortality_hospital"],
            seq_length_hours=4,
            min_stay_hours=0,
        )
        extractor = MockExtractor(config)

        assert not setup["output_dir"].exists()

        extractor.run()

        assert setup["output_dir"].exists()

    def test_run_creates_all_output_files(self, temp_extraction_setup):
        """Test that run() creates all expected output files."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=["mortality_hospital"],
            seq_length_hours=4,
            min_stay_hours=0,
        )
        extractor = MockExtractor(config)
        extractor.run()

        output_dir = setup["output_dir"]

        assert (output_dir / "static.parquet").exists()
        assert (output_dir / "timeseries.parquet").exists()
        assert (output_dir / "labels.parquet").exists()
        assert (output_dir / "metadata.yaml").exists()

    def test_run_metadata_content(self, temp_extraction_setup):
        """Test that metadata.yaml contains correct information."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=["mortality_hospital"],
            seq_length_hours=6,
            min_stay_hours=0,
            feature_set="core",
        )
        extractor = MockExtractor(config)
        extractor.run()

        with open(setup["output_dir"] / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        assert metadata["dataset"] == "mock"
        assert metadata["feature_set"] == "core"
        assert metadata["seq_length_hours"] == 6
        assert "feature_names" in metadata
        assert "n_features" in metadata
        assert "task_names" in metadata
        assert "mortality_hospital" in metadata["task_names"]
        assert metadata["n_stays"] > 0

    def test_run_filters_short_stays(self, temp_extraction_setup):
        """Test that run() filters stays shorter than min_stay_hours."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=[],
            seq_length_hours=4,
            min_stay_hours=200,  # Filter all stays (mock stays are 4 days = 96 hours)
        )
        extractor = MockExtractor(config)

        extractor.run()

        assert setup["output_dir"].exists()

        # The run method returns early when no stays remain, so no metadata file
        metadata_path = setup["output_dir"] / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = yaml.safe_load(f)
            assert metadata["n_stays"] == 0

    def test_run_with_empty_tasks(self, temp_extraction_setup):
        """Test that run() works with empty task list."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=[],
            seq_length_hours=4,
            min_stay_hours=0,
        )
        extractor = MockExtractor(config)
        extractor.run()

        labels_df = pl.read_parquet(setup["output_dir"] / "labels.parquet")
        assert "stay_id" in labels_df.columns
        assert len(labels_df.columns) == 1  # Only stay_id

    def test_run_timeseries_format(self, temp_extraction_setup):
        """Test that output timeseries has correct dense format."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=[],
            seq_length_hours=4,
            min_stay_hours=0,
        )
        extractor = MockExtractor(config)
        extractor.run()

        ts_df = pl.read_parquet(setup["output_dir"] / "timeseries.parquet")

        assert "stay_id" in ts_df.columns
        assert "timeseries" in ts_df.columns
        assert "mask" in ts_df.columns

        first_ts = ts_df["timeseries"].to_list()[0]
        assert isinstance(first_ts, list)
        assert len(first_ts) == 4  # seq_length_hours

    def test_run_static_contains_demographics(self, temp_extraction_setup):
        """Test that static output contains demographic information."""
        setup = temp_extraction_setup

        config = ExtractorConfig(
            parquet_root=str(setup["parquet_root"]),
            output_dir=str(setup["output_dir"]),
            tasks_dir=str(setup["tasks_dir"]),
            tasks=[],
            seq_length_hours=4,
            min_stay_hours=0,
        )
        extractor = MockExtractor(config)
        extractor.run()

        static_df = pl.read_parquet(setup["output_dir"] / "static.parquet")

        assert "stay_id" in static_df.columns
        assert "patient_id" in static_df.columns
        assert "age" in static_df.columns
        assert "gender" in static_df.columns
        assert "los_days" in static_df.columns


class TestLoadTaskConfigs:
    """Tests for _load_task_configs method."""

    def test_load_valid_task_config(self, tmp_path):
        """Test loading a valid task config from YAML."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        task_config = {
            "task_name": "test_task",
            "task_type": "binary_classification",
            "prediction_window_hours": 24,
            "label_sources": ["stays", "mortality_info"],
            "primary_metric": "auroc",
        }
        with open(tasks_dir / "test_task.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
        )
        extractor = MockExtractor(config)

        loaded = extractor._load_task_configs(["test_task"])

        assert len(loaded) == 1
        assert loaded[0].task_name == "test_task"
        assert loaded[0].task_type == "binary_classification"
        assert loaded[0].prediction_window_hours == 24

    def test_load_missing_task_config_skipped(self, tmp_path):
        """Test that missing task configs are skipped with warning."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
        )
        extractor = MockExtractor(config)

        loaded = extractor._load_task_configs(["nonexistent_task"])

        assert len(loaded) == 0

    def test_load_multiple_task_configs(self, tmp_path):
        """Test loading multiple task configs at once."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        for task_name, window in [("mortality_24h", 24), ("mortality_hospital", None)]:
            config_dict = {
                "task_name": task_name,
                "task_type": "binary_classification",
                "prediction_window_hours": window,
                "label_sources": ["stays", "mortality_info"],
            }
            with open(tasks_dir / f"{task_name}.yaml", "w") as f:
                yaml.dump(config_dict, f)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
        )
        extractor = MockExtractor(config)

        loaded = extractor._load_task_configs(["mortality_24h", "mortality_hospital"])

        assert len(loaded) == 2
        task_names = {tc.task_name for tc in loaded}
        assert "mortality_24h" in task_names
        assert "mortality_hospital" in task_names

    def test_observation_window_validation_raises_on_mismatch(self, tmp_path):
        """Test that loading task with mismatched observation_window_hours raises error."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        task_config = {
            "task_name": "mortality_24h",
            "task_type": "binary_classification",
            "prediction_window_hours": 24,
            "observation_window_hours": 24,
            "gap_hours": 0,
            "label_sources": ["stays", "mortality_info"],
        }
        with open(tasks_dir / "mortality_24h.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
            seq_length_hours=48,
        )
        extractor = MockExtractor(config)

        with pytest.raises(ValueError, match="Configuration mismatch"):
            extractor._load_task_configs(["mortality_24h"])

    def test_observation_window_validation_passes_when_matching(self, tmp_path):
        """Test that loading task with matching observation_window_hours succeeds."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        task_config = {
            "task_name": "mortality_24h",
            "task_type": "binary_classification",
            "prediction_window_hours": 24,
            "observation_window_hours": 48,
            "gap_hours": 0,
            "label_sources": ["stays", "mortality_info"],
        }
        with open(tasks_dir / "mortality_24h.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
            seq_length_hours=48,
        )
        extractor = MockExtractor(config)

        loaded = extractor._load_task_configs(["mortality_24h"])
        assert len(loaded) == 1
        assert loaded[0].observation_window_hours == 48

    def test_observation_window_validation_skipped_when_none(self, tmp_path):
        """Test that validation is skipped for legacy tasks without observation_window_hours."""
        parquet_root = tmp_path / "parquet"
        tasks_dir = tmp_path / "tasks"
        parquet_root.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)

        task_config = {
            "task_name": "mortality_hospital",
            "task_type": "binary_classification",
            "prediction_window_hours": None,
            "label_sources": ["stays", "mortality_info"],
        }
        with open(tasks_dir / "mortality_hospital.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
            seq_length_hours=48,
        )
        extractor = MockExtractor(config)

        loaded = extractor._load_task_configs(["mortality_hospital"])
        assert len(loaded) == 1
        assert loaded[0].observation_window_hours is None


class TestPathResolution:
    """Tests for path resolution methods."""

    def test_get_tasks_path_uses_explicit_config(self, tmp_path):
        """Test _get_tasks_path uses explicit tasks_dir when provided."""
        tasks_dir = tmp_path / "my_tasks"
        tasks_dir.mkdir(parents=True)

        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            tasks_dir=str(tasks_dir),
        )
        extractor = MockExtractor(config)

        result = extractor._get_tasks_path()
        assert result == tasks_dir


class TestAtomicWriteAndFileLocking:
    """Tests for atomic writes and file locking to prevent race conditions."""

    def test_atomic_write_parquet(self, tmp_path):
        """Test atomic write creates parquet file without corruption."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(parquet_root=str(parquet_root))
        extractor = MockExtractor(config)

        test_df = pl.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )

        output_path = tmp_path / "test.parquet"
        extractor._atomic_write(output_path, lambda tmp: test_df.write_parquet(tmp))

        assert output_path.exists()
        read_df = pl.read_parquet(output_path)
        assert read_df.equals(test_df)

    def test_atomic_write_yaml(self, tmp_path):
        """Test atomic write works with YAML files."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(parquet_root=str(parquet_root))
        extractor = MockExtractor(config)

        metadata = {
            "dataset": "test",
            "features": ["heart_rate", "sbp"],
            "n_stays": 100,
        }

        output_path = tmp_path / "metadata.yaml"
        extractor._atomic_write(
            output_path,
            lambda tmp: yaml.dump(metadata, open(tmp, "w"), default_flow_style=False),
            suffix=".yaml",
        )

        assert output_path.exists()
        with open(output_path) as f:
            read_metadata = yaml.safe_load(f)
        assert read_metadata == metadata

    def test_atomic_write_creates_temp_file_and_renames(self, tmp_path):
        """Test atomic write uses temp file + rename pattern."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(parquet_root=str(parquet_root))
        extractor = MockExtractor(config)

        test_df = pl.DataFrame({"col1": [1, 2, 3]})
        output_path = tmp_path / "test.parquet"

        assert not output_path.exists()

        extractor._atomic_write(output_path, lambda tmp: test_df.write_parquet(tmp))

        assert output_path.exists()
        temp_files = list(tmp_path.glob("*.parquet.tmp"))
        assert len(temp_files) == 0

    def test_atomic_write_cleanup_on_failure(self, tmp_path):
        """Test atomic write cleans up temp file if write fails."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(parquet_root=str(parquet_root))
        extractor = MockExtractor(config)

        output_path = tmp_path / "test.parquet"

        def failing_write(tmp):
            raise ValueError("Intentional failure")

        with pytest.raises(ValueError, match="Intentional failure"):
            extractor._atomic_write(output_path, failing_write)

        assert not output_path.exists()

    def test_file_locking_context_manager(self, tmp_path):
        """Test file locking context manager creates and releases lock."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        config = ExtractorConfig(parquet_root=str(parquet_root))
        extractor = MockExtractor(config)

        output_path = tmp_path / "test.parquet"
        lock_path = output_path.with_suffix(output_path.suffix + ".lock")

        assert not lock_path.exists()

        with extractor._with_file_lock(output_path):
            pass

        # Verify no exception was raised

    def test_resume_extraction_with_atomic_writes(self, tmp_path):
        """Test that resume extraction uses atomic writes and file locking."""
        parquet_root = tmp_path / "parquet"
        parquet_root.mkdir(parents=True)

        output_dir = tmp_path / "output"

        config = ExtractorConfig(
            parquet_root=str(parquet_root),
            output_dir=str(output_dir),
        )
        extractor = MockExtractor(config)

        # Run first extraction
        extractor.run()

        assert (output_dir / "static.parquet").exists()
        assert (output_dir / "timeseries.parquet").exists()
        assert (output_dir / "labels.parquet").exists()
        assert (output_dir / "metadata.yaml").exists()

        # Run extraction again (resume)
        extractor.run()

        resumed_static = pl.read_parquet(output_dir / "static.parquet")
        assert len(resumed_static) > 0

        assert (output_dir / "timeseries.parquet").exists()
        timeseries = pl.read_parquet(output_dir / "timeseries.parquet")
        assert len(timeseries) > 0


# ============================================================================
# Vectorized dense timeseries tests
# ============================================================================


class MinimalExtractor(BaseExtractor):
    """Minimal extractor for testing _create_dense_timeseries without path validation."""

    def __init__(self, config: ExtractorConfig):
        self.config = config

    def _get_dataset_name(self) -> str:
        return "test"

    def extract_stays(self) -> pl.DataFrame:
        return pl.DataFrame()

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        return pl.DataFrame()

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        return pl.DataFrame()

    def run(self) -> None:
        pass


def _make_minimal_extractor(seq_length: int = 48) -> MinimalExtractor:
    cfg = ExtractorConfig(
        parquet_root="/tmp/fake",
        output_dir="/tmp/fake_out",
        seq_length_hours=seq_length,
        feature_set="core",
        tasks=[],
        min_stay_hours=0,
        batch_size=100,
    )
    return MinimalExtractor(cfg)


class TestVectorizedDenseTimeseries:
    """Tests for the vectorized dense timeseries conversion."""

    def test_basic_conversion(self):
        """Test basic sparse-to-dense conversion with known values."""
        ext = _make_minimal_extractor(seq_length=4)
        features = ["hr", "sbp"]

        stays = pl.DataFrame({"stay_id": [1, 2]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "hour": [0, 2, 1],
                "hr": [70.0, 72.0, 80.0],
                "hr_mask": [True, True, True],
                "sbp": [120.0, None, 130.0],
                "sbp_mask": [True, False, True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)

        assert len(result) == 2
        assert set(result.columns) == {"stay_id", "timeseries", "mask"}

        row1 = result.filter(pl.col("stay_id") == 1)
        ts1 = row1["timeseries"][0]
        mask1 = row1["mask"][0]

        assert ts1[0][0] == 70.0
        assert mask1[0][0] is True
        assert ts1[0][1] == 120.0
        assert mask1[0][1] is True
        assert math.isnan(ts1[1][0])
        assert mask1[1][0] is False
        assert ts1[2][0] == 72.0
        assert mask1[2][0] is True

    def test_stays_without_data_produce_nan_arrays(self):
        """Stays with no observations get all-NaN timeseries."""
        ext = _make_minimal_extractor(seq_length=3)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1, 2]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "hr": [70.0],
                "hr_mask": [True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 2

        row2 = result.filter(pl.col("stay_id") == 2)
        ts2 = row2["timeseries"][0]
        mask2 = row2["mask"][0]

        for h in range(3):
            assert math.isnan(ts2[h][0])
            assert mask2[h][0] is False

    def test_overflow_hours_discarded(self):
        """Hours beyond seq_length are discarded."""
        ext = _make_minimal_extractor(seq_length=2)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "hour": [0, 1, 5],
                "hr": [70.0, 71.0, 99.0],
                "hr_mask": [True, True, True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        ts = result["timeseries"][0]

        assert len(ts) == 2
        assert ts[0][0] == 70.0
        assert ts[1][0] == 71.0

    def test_mask_false_produces_nan(self):
        """When mask is False, value should be NaN even if a value is present."""
        ext = _make_minimal_extractor(seq_length=2)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "hr": [70.0],
                "hr_mask": [False],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        ts = result["timeseries"][0]
        mask = result["mask"][0]

        assert math.isnan(ts[0][0])
        assert mask[0][0] is False

    def test_output_shape_consistency(self):
        """Verify output dimensions: seq_length x n_features."""
        ext = _make_minimal_extractor(seq_length=10)
        features = ["a", "b", "c"]
        stays = pl.DataFrame({"stay_id": [1, 2, 3]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "a": [1.0],
                "a_mask": [True],
                "b": [2.0],
                "b_mask": [True],
                "c": [3.0],
                "c_mask": [True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 3

        for row in result.iter_rows(named=True):
            ts = row["timeseries"]
            mask = row["mask"]
            assert len(ts) == 10
            assert len(mask) == 10
            for h in range(10):
                assert len(ts[h]) == 3
                assert len(mask[h]) == 3

    def test_empty_sparse_timeseries(self):
        """Handle completely empty sparse timeseries."""
        ext = _make_minimal_extractor(seq_length=3)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "hour": pl.Series([], dtype=pl.Int64),
                "hr": pl.Series([], dtype=pl.Float64),
                "hr_mask": pl.Series([], dtype=pl.Boolean),
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 1

        ts = result["timeseries"][0]
        for h in range(3):
            assert math.isnan(ts[h][0])

    def test_vectorized_matches_reference_loop(self):
        """Regression test: vectorized output must match a reference loop-based implementation."""
        seq_length = 6
        ext = _make_minimal_extractor(seq_length=seq_length)
        feature_names = ["hr", "sbp", "temp"]
        mask_names = [f"{f}_mask" for f in feature_names]
        n_features = len(feature_names)

        # Synthetic data with varied patterns: full, partial, and missing hours
        stays = pl.DataFrame({"stay_id": [10, 20, 30]})
        sparse = pl.DataFrame(
            {
                "stay_id": [10, 10, 10, 20, 20, 30],
                "hour": [0, 2, 5, 1, 3, 0],
                "hr": [72.0, 80.0, 65.0, 90.0, 88.0, 70.0],
                "hr_mask": [True, True, True, True, False, True],
                "sbp": [120.0, None, 130.0, 140.0, 135.0, None],
                "sbp_mask": [True, False, True, True, True, False],
                "temp": [36.5, 37.0, 38.2, 36.8, None, 37.1],
                "temp_mask": [True, True, True, True, False, True],
            }
        )

        # --- Reference loop-based implementation ---
        ref_rows = []
        for sid in stays["stay_id"].to_list():
            stay_data = sparse.filter(pl.col("stay_id") == sid)
            ts = []
            mask = []
            for h in range(seq_length):
                hour_data = stay_data.filter(pl.col("hour") == h)
                feat_vals = []
                mask_vals = []
                for fi, f in enumerate(feature_names):
                    m = mask_names[fi]
                    if len(hour_data) == 0:
                        feat_vals.append(float("nan"))
                        mask_vals.append(False)
                    else:
                        raw_val = hour_data[f][0]
                        raw_mask = hour_data[m][0]
                        if raw_mask and raw_val is not None:
                            feat_vals.append(float(raw_val))
                        else:
                            feat_vals.append(float("nan"))
                        mask_vals.append(bool(raw_mask))
                ts.append(feat_vals)
                mask.append(mask_vals)
            ref_rows.append({"stay_id": sid, "timeseries": ts, "mask": mask})

        # --- Vectorized implementation ---
        result = ext._create_dense_timeseries(sparse, stays, feature_names)

        # Compare row by row
        for ref in ref_rows:
            sid = ref["stay_id"]
            row = result.filter(pl.col("stay_id") == sid)
            assert len(row) == 1, f"Missing stay_id={sid} in vectorized output"

            vec_ts = row["timeseries"][0]
            vec_mask = row["mask"][0]
            ref_ts = ref["timeseries"]
            ref_mask = ref["mask"]

            assert len(vec_ts) == seq_length
            assert len(vec_mask) == seq_length

            for h in range(seq_length):
                assert len(vec_ts[h]) == n_features
                assert len(vec_mask[h]) == n_features
                for fi in range(n_features):
                    # Compare masks
                    assert vec_mask[h][fi] == ref_mask[h][fi], (
                        f"Mask mismatch at stay={sid}, hour={h}, feat={fi}: "
                        f"vec={vec_mask[h][fi]}, ref={ref_mask[h][fi]}"
                    )
                    # Compare values (NaN == NaN for this comparison)
                    v_vec = vec_ts[h][fi]
                    v_ref = ref_ts[h][fi]
                    if math.isnan(v_ref):
                        assert math.isnan(v_vec), (
                            f"Value mismatch at stay={sid}, hour={h}, feat={fi}: "
                            f"expected NaN, got {v_vec}"
                        )
                    else:
                        assert v_vec == pytest.approx(v_ref), (
                            f"Value mismatch at stay={sid}, hour={h}, feat={fi}: "
                            f"vec={v_vec}, ref={v_ref}"
                        )
