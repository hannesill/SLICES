"""Tests for RicuExtractor — reads pre-extracted RICU parquet files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.ricu import RicuExtractor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ricu_output_dir(tmp_path: Path) -> Path:
    """Create a complete mock RICU output directory with all parquet files."""
    ricu_dir = tmp_path / "ricu_output"
    ricu_dir.mkdir()

    # --- ricu_metadata.yaml ---
    metadata = {
        "dataset": "miiv",
        "feature_names": ["hr", "sbp", "crea"],
        "n_features": 3,
        "seq_length_hours": 6,
        "raw_export_horizon_hours": 6,
        "n_stays": 3,
        "ricu_version": "0.7.0",
    }
    with open(ricu_dir / "ricu_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    # --- ricu_stays.parquet ---
    stays = pl.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "patient_id": [10, 20, 30],
            "hadm_id": [1000, 2000, 3000],
            "intime": [
                datetime(2020, 1, 1, 8, 0),
                datetime(2020, 1, 2, 10, 0),
                datetime(2020, 1, 3, 12, 0),
            ],
            "outtime": [
                datetime(2020, 1, 5, 8, 0),
                datetime(2020, 1, 6, 10, 0),
                datetime(2020, 1, 7, 12, 0),
            ],
            "los_days": [4.0, 4.0, 4.0],
            "age": [65.0, 50.0, 72.0],
            "gender": ["M", "F", "M"],
            "race": ["WHITE", "BLACK", None],
            "admission_type": ["EMERGENCY", "ELECTIVE", "EMERGENCY"],
            "insurance": ["Medicare", "Private", "Medicaid"],
            "first_careunit": ["MICU", "SICU", "MICU"],
            "height": [175.0, 160.0, None],
            "weight": [80.0, 65.0, 90.0],
        }
    )
    stays.write_parquet(ricu_dir / "ricu_stays.parquet")

    # --- ricu_timeseries.parquet ---
    rows = []
    for stay_id in [100, 200, 300]:
        for hour in range(6):
            rows.append(
                {
                    "stay_id": stay_id,
                    "hour": hour,
                    "hr": 70.0 + hour if hour % 2 == 0 else None,
                    "sbp": 120.0 + hour if hour < 3 else None,
                    "crea": 1.0 if hour == 0 else None,
                    "hr_mask": hour % 2 == 0,
                    "sbp_mask": hour < 3,
                    "crea_mask": hour == 0,
                }
            )
    timeseries = pl.DataFrame(rows)
    timeseries.write_parquet(ricu_dir / "ricu_timeseries.parquet")

    # --- ricu_mortality.parquet ---
    mortality = pl.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "date_of_death": [None, None, datetime(2020, 1, 8, 0, 0)],
            "hospital_expire_flag": [0, 0, 1],
            "dischtime": [
                datetime(2020, 1, 5, 8, 0),
                datetime(2020, 1, 6, 10, 0),
                datetime(2020, 1, 7, 12, 0),
            ],
            "discharge_location": ["HOME", "HOME", "DIED"],
        }
    )
    mortality.write_parquet(ricu_dir / "ricu_mortality.parquet")

    # --- ricu_diagnoses.parquet ---
    diagnoses = pl.DataFrame(
        {
            "stay_id": [100, 100, 200, 300],
            "icd_code": ["I10", "E119", "J189", "N179"],
            "icd_version": [10, 10, 10, 10],
        }
    )
    diagnoses.write_parquet(ricu_dir / "ricu_diagnoses.parquet")

    return ricu_dir


def _create_aki_benchmark_ricu_output(tmp_path: Path) -> Path:
    """Create mock RICU output for the 24h-input / 48h-raw AKI benchmark."""
    ricu_dir = tmp_path / "ricu_output_aki"
    ricu_dir.mkdir()

    metadata = {
        "dataset": "miiv",
        "feature_names": ["crea"],
        "n_features": 1,
        "seq_length_hours": 48,
        "raw_export_horizon_hours": 48,
        "n_stays": 3,
        "ricu_version": "0.7.0",
    }
    with open(ricu_dir / "ricu_metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    stays = pl.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "patient_id": [10, 20, 30],
            "hadm_id": [1000, 2000, 3000],
            "intime": [
                datetime(2020, 1, 1, 8, 0),
                datetime(2020, 1, 2, 8, 0),
                datetime(2020, 1, 3, 8, 0),
            ],
            "outtime": [
                datetime(2020, 1, 4, 12, 0),
                datetime(2020, 1, 5, 12, 0),
                datetime(2020, 1, 6, 12, 0),
            ],
            "los_days": [3.2, 3.2, 3.2],
            "age": [65.0, 70.0, 58.0],
            "gender": ["M", "F", "M"],
            "race": ["WHITE", "BLACK", "ASIAN"],
            "admission_type": ["EMERGENCY", "EMERGENCY", "EMERGENCY"],
            "insurance": ["Medicare", "Private", "Medicaid"],
            "first_careunit": ["MICU", "MICU", "MICU"],
            "height": [175.0, 165.0, 180.0],
            "weight": [80.0, 68.0, 92.0],
        }
    )
    stays.write_parquet(ricu_dir / "ricu_stays.parquet")

    timeseries = pl.DataFrame(
        {
            "stay_id": [100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300],
            "hour": [0, 12, 23, 30, 0, 12, 23, 30, 0, 12, 23],
            "crea": [1.0, 0.9, 1.0, 1.6, 1.0, 1.1, 1.0, 1.1, 0.8, 0.9, 0.85],
            "crea_mask": [True] * 11,
        }
    )
    timeseries.write_parquet(ricu_dir / "ricu_timeseries.parquet")

    mortality = pl.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "date_of_death": [None, None, None],
            "hospital_expire_flag": [0, 0, 0],
            "dischtime": [
                datetime(2020, 1, 7, 8, 0),
                datetime(2020, 1, 8, 8, 0),
                datetime(2020, 1, 9, 8, 0),
            ],
            "discharge_location": ["HOME", "HOME", "HOME"],
        }
    )
    mortality.write_parquet(ricu_dir / "ricu_mortality.parquet")

    diagnoses = pl.DataFrame(
        {
            "stay_id": [100, 200, 300],
            "icd_code": ["N179", "I10", "E119"],
            "icd_version": [10, 10, 10],
        }
    )
    diagnoses.write_parquet(ricu_dir / "ricu_diagnoses.parquet")

    return ricu_dir


@pytest.fixture
def ricu_extractor(ricu_output_dir: Path, tmp_path: Path) -> RicuExtractor:
    """Create a RicuExtractor instance with mock data."""
    config = ExtractorConfig(
        parquet_root=str(ricu_output_dir),
        output_dir=str(tmp_path / "processed"),
        seq_length_hours=6,
        min_stay_hours=0,
        tasks=[],
    )
    return RicuExtractor(config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestRicuExtractorInit:
    """Tests for RicuExtractor initialization."""

    def test_init_success(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(tmp_path / "processed"),
            seq_length_hours=6,
            tasks=[],
        )
        extractor = RicuExtractor(config)

        assert extractor.parquet_root == ricu_output_dir
        assert extractor._metadata is not None
        assert extractor._metadata["dataset"] == "miiv"

    def test_init_missing_directory(self, tmp_path: Path) -> None:
        config = ExtractorConfig(
            parquet_root=str(tmp_path / "nonexistent"),
            output_dir=str(tmp_path / "processed"),
        )
        with pytest.raises(ValueError, match="Parquet directory not found"):
            RicuExtractor(config)

    def test_init_missing_metadata(self, tmp_path: Path) -> None:
        ricu_dir = tmp_path / "ricu_no_meta"
        ricu_dir.mkdir()
        config = ExtractorConfig(
            parquet_root=str(ricu_dir),
            output_dir=str(tmp_path / "processed"),
        )
        with pytest.raises(ValueError, match="ricu_metadata.yaml not found"):
            RicuExtractor(config)

    def test_no_duckdb_connection(self, ricu_extractor: RicuExtractor) -> None:
        """Verify that RicuExtractor does not have a DuckDB connection."""
        assert not hasattr(ricu_extractor, "conn")

    def test_init_raises_when_python_horizon_exceeds_ricu_export(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(tmp_path / "processed"),
            seq_length_hours=8,
        )

        with pytest.raises(ValueError, match="upstream RICU export only contains 6 hours"):
            RicuExtractor(config)

    def test_init_raises_when_task_requires_longer_raw_export_horizon(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_config = {
            "task_name": "aki_kdigo",
            "task_type": "binary",
            "observation_window_hours": 6,
            "prediction_window_hours": 4,
            "gap_hours": 0,
            "label_sources": ["stays", "timeseries"],
            "label_params": {"creatinine_col": "crea"},
            "primary_metric": "auprc",
        }
        with open(tasks_dir / "aki_kdigo.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(tmp_path / "processed"),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["aki_kdigo"],
            tasks_dir=str(tasks_dir),
        )

        with pytest.raises(
            ValueError,
            match="Active task labels require a longer upstream raw timeseries horizon",
        ):
            RicuExtractor(config)

    def test_init_accepts_explicit_raw_export_horizon_longer_than_input_window(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        with open(ricu_output_dir / "ricu_metadata.yaml") as f:
            metadata = yaml.safe_load(f)
        metadata["raw_export_horizon_hours"] = 10
        with open(ricu_output_dir / "ricu_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_config = {
            "task_name": "aki_kdigo",
            "task_type": "binary",
            "observation_window_hours": 6,
            "prediction_window_hours": 4,
            "gap_hours": 0,
            "label_sources": ["stays", "timeseries"],
            "label_params": {"creatinine_col": "crea"},
            "primary_metric": "auprc",
        }
        with open(tasks_dir / "aki_kdigo.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(tmp_path / "processed"),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["aki_kdigo"],
            tasks_dir=str(tasks_dir),
        )

        extractor = RicuExtractor(config)
        assert extractor._get_raw_export_horizon_hours() == 10

    def test_existing_extraction_is_invalidated_when_upstream_inputs_change(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )

        RicuExtractor(config).run()

        resumed = RicuExtractor(config)
        assert resumed._check_existing_extraction() is not None

        timeseries_path = ricu_output_dir / "ricu_timeseries.parquet"
        timeseries = pl.read_parquet(timeseries_path)
        updated = pl.concat(
            [
                timeseries,
                pl.DataFrame(
                    {
                        "stay_id": [999],
                        "hour": [0],
                        "hr": [88.0],
                        "sbp": [120.0],
                        "crea": [1.2],
                        "hr_mask": [True],
                        "sbp_mask": [True],
                        "crea_mask": [True],
                    }
                ),
            ],
            how="vertical_relaxed",
        )
        updated.write_parquet(timeseries_path)

        after_change = RicuExtractor(config)
        assert after_change._check_existing_extraction() is None


# ---------------------------------------------------------------------------
# Core extraction methods
# ---------------------------------------------------------------------------


class TestExtractStays:
    """Tests for extract_stays()."""

    def test_returns_dataframe(self, ricu_extractor: RicuExtractor) -> None:
        stays = ricu_extractor.extract_stays()
        assert isinstance(stays, pl.DataFrame)

    def test_expected_columns(self, ricu_extractor: RicuExtractor) -> None:
        stays = ricu_extractor.extract_stays()
        for col in ["stay_id", "patient_id", "los_days", "age", "gender"]:
            assert col in stays.columns, f"Missing column: {col}"

    def test_correct_row_count(self, ricu_extractor: RicuExtractor) -> None:
        stays = ricu_extractor.extract_stays()
        assert len(stays) == 3

    def test_stay_ids_correct(self, ricu_extractor: RicuExtractor) -> None:
        stays = ricu_extractor.extract_stays()
        assert set(stays["stay_id"].to_list()) == {100, 200, 300}


class TestExtractTimeseries:
    """Tests for extract_timeseries()."""

    def test_returns_dataframe(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100, 200])
        assert isinstance(ts, pl.DataFrame)

    def test_filters_by_stay_ids(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100])
        assert set(ts["stay_id"].unique().to_list()) == {100}

    def test_all_stay_ids(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100, 200, 300])
        assert set(ts["stay_id"].unique().to_list()) == {100, 200, 300}

    def test_has_hour_column(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100])
        assert "hour" in ts.columns

    def test_has_feature_columns(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100])
        for feat in ["hr", "sbp", "crea"]:
            assert feat in ts.columns, f"Missing feature column: {feat}"

    def test_has_mask_columns(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100])
        for feat in ["hr_mask", "sbp_mask", "crea_mask"]:
            assert feat in ts.columns, f"Missing mask column: {feat}"

    def test_empty_filter_returns_empty(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([999])
        assert len(ts) == 0

    def test_hours_per_stay(self, ricu_extractor: RicuExtractor) -> None:
        ts = ricu_extractor.extract_timeseries([100])
        assert len(ts) == 6  # 6 hours of data

    def test_reads_from_partitioned_directory(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        """Timeseries stored as a directory of part files (chunked output)."""
        ts_file = ricu_output_dir / "ricu_timeseries.parquet"
        original = pl.read_parquet(ts_file)

        # Split into two part files in a directory with the same name
        ts_file.unlink()
        ts_file.mkdir()
        part1 = original.filter(pl.col("stay_id") == 100)
        part2 = original.filter(pl.col("stay_id").is_in([200, 300]))
        part1.write_parquet(ts_file / "part_0001.parquet")
        part2.write_parquet(ts_file / "part_0002.parquet")

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(tmp_path / "processed_partitioned"),
            seq_length_hours=6,
            tasks=[],
        )
        extractor = RicuExtractor(config)
        ts = extractor.extract_timeseries([100, 200, 300])

        assert set(ts["stay_id"].unique().to_list()) == {100, 200, 300}
        assert len(ts) == len(original)
        assert set(ts.columns) == set(original.columns)


class TestExtractDataSource:
    """Tests for extract_data_source()."""

    def test_mortality_info(self, ricu_extractor: RicuExtractor) -> None:
        mort = ricu_extractor.extract_data_source("mortality_info", [100, 300])
        assert isinstance(mort, pl.DataFrame)
        assert set(mort["stay_id"].to_list()) == {100, 300}
        assert "hospital_expire_flag" in mort.columns

    def test_mortality_values(self, ricu_extractor: RicuExtractor) -> None:
        mort = ricu_extractor.extract_data_source("mortality_info", [100, 300])
        row100 = mort.filter(pl.col("stay_id") == 100)
        row300 = mort.filter(pl.col("stay_id") == 300)
        assert row100["hospital_expire_flag"][0] == 0
        assert row300["hospital_expire_flag"][0] == 1

    def test_diagnoses(self, ricu_extractor: RicuExtractor) -> None:
        diag = ricu_extractor.extract_data_source("diagnoses", [100])
        assert isinstance(diag, pl.DataFrame)
        assert set(diag["stay_id"].unique().to_list()) == {100}
        assert len(diag) == 2  # stay 100 has 2 diagnoses

    def test_diagnoses_columns(self, ricu_extractor: RicuExtractor) -> None:
        diag = ricu_extractor.extract_data_source("diagnoses", [100])
        assert "icd_code" in diag.columns
        assert "icd_version" in diag.columns

    def test_unknown_source_raises(self, ricu_extractor: RicuExtractor) -> None:
        with pytest.raises(ValueError, match="Unknown data source"):
            ricu_extractor.extract_data_source("nonexistent", [100])

    def test_filter_by_stay_ids(self, ricu_extractor: RicuExtractor) -> None:
        mort = ricu_extractor.extract_data_source("mortality_info", [200])
        assert len(mort) == 1
        assert mort["stay_id"][0] == 200

    def test_legacy_mortality_datetime_is_migrated_conservatively(
        self, ricu_extractor: RicuExtractor
    ) -> None:
        mort = ricu_extractor.extract_data_source("mortality_info", [300])
        row = mort.filter(pl.col("stay_id") == 300)

        assert row["death_time_precision"][0] == "date"
        assert row["death_time"][0] is None
        assert row["death_date"][0].isoformat() == "2020-01-08"


class TestExtractRawEvents:
    """Tests for _extract_raw_events() — should raise NotImplementedError."""

    def test_raises_not_implemented(self, ricu_extractor: RicuExtractor) -> None:
        with pytest.raises(NotImplementedError, match="pre-binned data"):
            ricu_extractor._extract_raw_events([], {})


class TestGetDatasetName:
    """Tests for _get_dataset_name()."""

    def test_returns_metadata_dataset(self, ricu_extractor: RicuExtractor) -> None:
        assert ricu_extractor._get_dataset_name() == "miiv"


# ---------------------------------------------------------------------------
# End-to-end run()
# ---------------------------------------------------------------------------


class TestRicuExtractorRun:
    """Tests for the full run() pipeline."""

    def test_run_creates_output_directory(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        extractor = RicuExtractor(config)
        assert not output_dir.exists()

        extractor.run()

        assert output_dir.exists()

    def test_run_creates_all_output_files(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        extractor = RicuExtractor(config)
        extractor.run()

        assert (output_dir / "static.parquet").exists()
        assert (output_dir / "timeseries.parquet").exists()
        assert (output_dir / "labels.parquet").exists()
        assert (output_dir / "metadata.yaml").exists()

    def test_run_static_output(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        RicuExtractor(config).run()

        static = pl.read_parquet(output_dir / "static.parquet")
        assert len(static) == 3
        assert "stay_id" in static.columns
        assert "patient_id" in static.columns
        assert "age" in static.columns

    def test_run_timeseries_format(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        RicuExtractor(config).run()

        ts = pl.read_parquet(output_dir / "timeseries.parquet")
        assert len(ts) == 3
        assert "stay_id" in ts.columns
        assert "timeseries" in ts.columns
        assert "mask" in ts.columns

        # Check dense array dimensions
        first_ts = ts["timeseries"].to_list()[0]
        first_mask = ts["mask"].to_list()[0]
        assert len(first_ts) == 6  # seq_length_hours
        assert len(first_ts[0]) == 3  # n_features (hr, sbp, crea)
        assert len(first_mask) == 6
        assert len(first_mask[0]) == 3

    def test_run_labels_empty_tasks(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        RicuExtractor(config).run()

        labels = pl.read_parquet(output_dir / "labels.parquet")
        assert "stay_id" in labels.columns
        assert len(labels.columns) == 1  # Only stay_id

    def test_run_metadata_content(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        RicuExtractor(config).run()

        with open(output_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)

        assert meta["dataset"] == "miiv"
        assert meta["n_features"] == 3
        assert meta["seq_length_hours"] == 6
        assert meta["input_seq_length_hours"] == 6
        assert meta["raw_export_horizon_hours"] == 6
        assert meta["required_raw_export_horizon_hours"] == 6
        assert meta["n_stays"] == 3
        assert meta["feature_names"] == ["hr", "sbp", "crea"]
        assert meta["label_quality_stats"] == {}
        assert "ricu_metadata" in meta
        assert "upstream_source_signature" in meta
        assert len(meta["upstream_source_signature"]["files"]) >= 4

    def test_run_filters_short_stays(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=200,  # All mock stays are 4 days = 96 hours
            tasks=[],
        )
        RicuExtractor(config).run()

        # Should complete without error but not write output files
        # (early return when no stays remain)
        assert output_dir.exists()

    def test_run_timeseries_values_correct(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        """Verify that dense timeseries preserves values from RICU output."""
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )
        RicuExtractor(config).run()

        ts = pl.read_parquet(output_dir / "timeseries.parquet")
        # Find stay 100
        stay100 = ts.filter(pl.col("stay_id") == 100)
        values = stay100["timeseries"].to_list()[0]
        mask = stay100["mask"].to_list()[0]

        # Hour 0: hr=70.0 (observed), sbp=120.0 (observed), crea=1.0 (observed)
        assert values[0][0] == 70.0
        assert values[0][1] == 120.0
        assert values[0][2] == 1.0
        assert mask[0] == [True, True, True]

        # Hour 1: hr=None (not observed), sbp=121.0 (observed), crea=None
        assert np.isnan(values[1][0])
        assert values[1][1] == 121.0
        assert np.isnan(values[1][2])
        assert mask[1] == [False, True, False]

    def test_run_with_task_labels(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        """Test run() with a real task config (mortality_hospital)."""
        output_dir = tmp_path / "processed"
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        # Create a mortality task config
        task_config = {
            "task_name": "mortality_hospital",
            "task_type": "binary_classification",
            "prediction_window_hours": None,
            "label_sources": ["stays", "mortality_info"],
            "primary_metric": "auroc",
        }
        with open(tasks_dir / "mortality_hospital.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["mortality_hospital"],
            tasks_dir=str(tasks_dir),
        )
        RicuExtractor(config).run()

        labels = pl.read_parquet(output_dir / "labels.parquet")
        assert "stay_id" in labels.columns
        assert "mortality_hospital" in labels.columns
        assert len(labels) == 3

    def test_run_aki_24h_input_with_48h_raw_export(self, tmp_path: Path) -> None:
        """AKI extraction should use 48h raw creatinine while saving 24h model inputs."""
        ricu_output_dir = _create_aki_benchmark_ricu_output(tmp_path)
        output_dir = tmp_path / "processed_aki"
        tasks_dir = tmp_path / "tasks_aki"
        tasks_dir.mkdir()

        task_config = {
            "task_name": "aki_kdigo",
            "task_type": "binary",
            "observation_window_hours": 24,
            "prediction_window_hours": 24,
            "gap_hours": 0,
            "label_sources": ["stays", "timeseries"],
            "label_params": {
                "creatinine_col": "crea",
                "baseline_window_hours": 24,
                "absolute_rise_threshold": 0.3,
                "relative_rise_threshold": 1.5,
                "relative_window_hours": 168,
            },
            "quality_checks": {"max_missing_percentage": 28.0},
            "primary_metric": "auprc",
        }
        with open(tasks_dir / "aki_kdigo.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=24,
            min_stay_hours=24,
            tasks=["aki_kdigo"],
            tasks_dir=str(tasks_dir),
        )
        RicuExtractor(config).run()

        labels = pl.read_parquet(output_dir / "labels.parquet")
        labels_dict = dict(zip(labels["stay_id"], labels["aki_kdigo"]))
        assert labels_dict[100] == 1
        assert labels_dict[200] == 0
        assert labels_dict[300] is None

        dense_timeseries = pl.read_parquet(output_dir / "timeseries.parquet")
        stay100_ts = dense_timeseries.filter(pl.col("stay_id") == 100)["timeseries"].to_list()[0]
        assert len(stay100_ts) == 24  # model input remains 24h

        with open(output_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)

        assert meta["input_seq_length_hours"] == 24
        assert meta["raw_export_horizon_hours"] == 48
        assert meta["required_raw_export_horizon_hours"] == 48

        quality_stats = meta["label_quality_stats"]["aki_kdigo"]
        assert quality_stats["total_stays"] == 3
        assert quality_stats["positive_labels"] == 1
        assert quality_stats["negative_labels"] == 1
        assert quality_stats["null_labels"] == 1
        assert quality_stats["null_reason_counts"] == {
            "no_creatinine_or_baseline": 0,
            "no_post_obs_creatinine": 1,
        }

    def test_run_resume_skips_existing(self, ricu_output_dir: Path, tmp_path: Path) -> None:
        """Test that a second run() resumes without duplicating stays."""
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )

        # First run
        RicuExtractor(config).run()
        static_first = pl.read_parquet(output_dir / "static.parquet")
        n_first = len(static_first)

        # Second run (should resume and find all stays already extracted)
        RicuExtractor(config).run()
        static_second = pl.read_parquet(output_dir / "static.parquet")

        # Should have same number of stays (no duplicates)
        assert len(static_second) == n_first

    def test_run_rebuilds_when_task_config_changes(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        """Task-config drift should invalidate resume instead of silently skipping work."""
        output_dir = tmp_path / "processed"
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_config = {
            "task_name": "mortality_24h",
            "task_type": "binary",
            "observation_window_hours": 6,
            "prediction_window_hours": 24,
            "gap_hours": 0,
            "label_sources": ["stays", "mortality_info"],
            "label_params": {},
        }
        task_path = tasks_dir / "mortality_24h.yaml"
        with open(task_path, "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["mortality_24h"],
            tasks_dir=str(tasks_dir),
        )

        RicuExtractor(config).run()
        with open(output_dir / "metadata.yaml") as f:
            initial_metadata = yaml.safe_load(f)

        task_config["gap_hours"] = 6
        with open(task_path, "w") as f:
            yaml.dump(task_config, f)

        RicuExtractor(config).run()
        with open(output_dir / "metadata.yaml") as f:
            updated_metadata = yaml.safe_load(f)

        assert (
            updated_metadata["label_manifest"]["mortality_24h"]["config_hash"]
            != initial_metadata["label_manifest"]["mortality_24h"]["config_hash"]
        )

    def test_run_rebuilds_when_builder_version_changes(
        self, ricu_output_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Builder-version drift should invalidate resume instead of merging stale labels."""
        from slices.data.labels.mortality import MortalityLabelBuilder

        output_dir = tmp_path / "processed"
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_config = {
            "task_name": "mortality_hospital",
            "task_type": "binary",
            "observation_window_hours": 6,
            "prediction_window_hours": None,
            "gap_hours": 0,
            "label_sources": ["stays", "mortality_info"],
            "label_params": {},
        }
        with open(tasks_dir / "mortality_hospital.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["mortality_hospital"],
            tasks_dir=str(tasks_dir),
        )

        RicuExtractor(config).run()
        with open(output_dir / "metadata.yaml") as f:
            initial_metadata = yaml.safe_load(f)

        initial_version = MortalityLabelBuilder.SEMANTIC_VERSION
        monkeypatch.setattr(MortalityLabelBuilder, "SEMANTIC_VERSION", "9.9.9")

        RicuExtractor(config).run()
        with open(output_dir / "metadata.yaml") as f:
            updated_metadata = yaml.safe_load(f)

        assert (
            initial_metadata["label_manifest"]["mortality_hospital"]["builder_version"]
            == initial_version
        )
        assert (
            updated_metadata["label_manifest"]["mortality_hospital"]["builder_version"] == "9.9.9"
        )

    def test_run_excludes_invalid_los_before_validation(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        """Exclude invalid LOS rows before validating the benchmark cohort."""
        output_dir = tmp_path / "processed"

        stays_path = ricu_output_dir / "ricu_stays.parquet"
        stays = pl.read_parquet(stays_path).with_columns(
            pl.when(pl.col("stay_id") == 200)
            .then(None)
            .otherwise(pl.col("los_days"))
            .alias("los_days")
        )
        stays.write_parquet(stays_path)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )

        RicuExtractor(config).run()

        static = pl.read_parquet(output_dir / "static.parquet")
        assert static["stay_id"].to_list() == [100, 300]

    def test_run_fails_closed_on_missing_timeseries_coverage(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        """Missing timeseries rows for an extracted stay should abort the run."""
        output_dir = tmp_path / "processed"

        timeseries_path = ricu_output_dir / "ricu_timeseries.parquet"
        timeseries = pl.read_parquet(timeseries_path).filter(pl.col("stay_id") != 300)
        timeseries.write_parquet(timeseries_path)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )

        with pytest.raises(ValueError, match="no timeseries data"):
            RicuExtractor(config).run()

    def test_run_fails_closed_on_missing_labels(
        self, ricu_output_dir: Path, tmp_path: Path, monkeypatch
    ) -> None:
        """Missing label rows should abort extraction instead of producing a degraded dataset."""
        output_dir = tmp_path / "processed"
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_config = {
            "task_name": "mortality_hospital",
            "task_type": "binary",
            "observation_window_hours": 6,
            "prediction_window_hours": None,
            "gap_hours": 0,
            "label_sources": ["stays", "mortality_info"],
            "label_params": {},
        }
        with open(tasks_dir / "mortality_hospital.yaml", "w") as f:
            yaml.dump(task_config, f)

        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=["mortality_hospital"],
            tasks_dir=str(tasks_dir),
        )

        extractor = RicuExtractor(config)
        original_extract_labels = extractor.extract_labels

        def drop_one_label(stay_ids, task_configs):
            labels = original_extract_labels(stay_ids, task_configs)
            return labels.filter(pl.col("stay_id") != 300)

        monkeypatch.setattr(extractor, "extract_labels", drop_one_label)

        with pytest.raises(ValueError, match="no labels"):
            extractor.run()

    def test_run_rebuilds_when_upstream_timeseries_changes(
        self, ricu_output_dir: Path, tmp_path: Path
    ) -> None:
        """Changing upstream parquet inputs should invalidate resume and rebuild outputs."""
        output_dir = tmp_path / "processed"
        config = ExtractorConfig(
            parquet_root=str(ricu_output_dir),
            output_dir=str(output_dir),
            seq_length_hours=6,
            min_stay_hours=0,
            tasks=[],
        )

        RicuExtractor(config).run()

        ts_path = ricu_output_dir / "ricu_timeseries.parquet"
        updated = pl.read_parquet(ts_path).with_columns(
            pl.when((pl.col("stay_id") == 100) & (pl.col("hour") == 0))
            .then(pl.lit(999.0))
            .otherwise(pl.col("hr"))
            .alias("hr")
        )
        updated.write_parquet(ts_path)

        RicuExtractor(config).run()

        timeseries = pl.read_parquet(output_dir / "timeseries.parquet")
        stay100 = timeseries.filter(pl.col("stay_id") == 100)
        assert stay100["timeseries"].to_list()[0][0][0] == 999.0
