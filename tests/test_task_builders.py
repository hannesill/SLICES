"""Tests for task label extraction and builders.

Comprehensive tests for mortality tasks and task builder infrastructure.
"""

from datetime import datetime

import polars as pl
import pytest
from slices.data.labels import LabelBuilderFactory, LabelConfig
from slices.data.labels.mortality import MortalityLabelBuilder


class TestLabelConfig:
    """Tests for LabelConfig dataclass."""

    def test_label_config_creation(self):
        """Test basic LabelConfig instantiation."""
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
            primary_metric="auroc",
        )

        assert config.task_name == "mortality_24h"
        assert config.task_type == "binary_classification"
        assert config.prediction_window_hours == 24
        assert "stays" in config.label_sources

    def test_label_config_defaults(self):
        """Test LabelConfig with default values."""
        config = LabelConfig(
            task_name="test_task",
            task_type="binary_classification",
        )

        assert config.prediction_window_hours is None
        assert config.gap_hours == 0
        assert config.label_sources == []
        assert config.primary_metric == "auroc"

    def test_label_config_all_fields(self):
        """Test LabelConfig with all fields populated."""
        config = LabelConfig(
            task_name="aki_kdigo",
            task_type="multiclass_classification",
            prediction_window_hours=48,
            observation_window_hours=24,
            gap_hours=6,
            label_sources=["labs", "creatinine"],
            label_params={"threshold": 1.5},
            primary_metric="auprc",
            additional_metrics=["accuracy", "f1"],
            n_classes=4,
            class_names=["No AKI", "Stage 1", "Stage 2", "Stage 3"],
            positive_class="Stage 3",
        )

        assert config.task_name == "aki_kdigo"
        assert config.observation_window_hours == 24
        assert config.gap_hours == 6
        assert config.n_classes == 4
        assert len(config.class_names) == 4


class TestMortalityLabelBuilder:
    """Tests for MortalityLabelBuilder."""

    @pytest.fixture
    def sample_stays(self):
        """Sample ICU stay data."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "intime": [
                    datetime(2020, 1, 1, 10, 0),
                    datetime(2020, 1, 2, 12, 0),
                    datetime(2020, 1, 3, 8, 0),
                    datetime(2020, 1, 4, 14, 0),
                ],
                "outtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 12, 0),
                    datetime(2020, 1, 6, 8, 0),
                    datetime(2020, 1, 10, 14, 0),
                ],
            }
        )

    @pytest.fixture
    def sample_mortality_info(self):
        """Sample mortality data."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "date_of_death": [
                    datetime(2020, 1, 1, 20, 0),  # Died 10h after admission
                    None,  # Survived
                    datetime(2020, 1, 5, 10, 0),  # Died 50h after admission
                    datetime(2020, 1, 6, 14, 0),  # Died 48h after admission
                ],
                "hospital_expire_flag": [1, 0, 1, 1],
                "dischtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 12, 0),
                    datetime(2020, 1, 6, 8, 0),
                    datetime(2020, 1, 10, 14, 0),
                ],
                "discharge_location": ["DIED", "HOME", "DIED", "DIED"],
            }
        )

    def test_hospital_mortality(self, sample_stays, sample_mortality_info):
        """Test hospital mortality prediction (no time window)."""
        config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        raw_data = {
            "stays": sample_stays,
            "mortality_info": sample_mortality_info,
        }

        labels = builder.build_labels(raw_data)

        assert labels.shape[0] == 4
        assert "stay_id" in labels.columns
        assert "label" in labels.columns

        # Check specific cases
        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Died
        assert labels_dict[2] == 0  # Survived
        assert labels_dict[3] == 1  # Died
        assert labels_dict[4] == 1  # Died

    def test_24h_mortality(self, sample_stays, sample_mortality_info):
        """Test 24-hour mortality prediction."""
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        raw_data = {
            "stays": sample_stays,
            "mortality_info": sample_mortality_info,
        }

        labels = builder.build_labels(raw_data)

        # Check specific cases
        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Died within 24h (10h)
        assert labels_dict[2] == 0  # Survived
        assert labels_dict[3] == 0  # Died after 24h (50h)
        assert labels_dict[4] == 0  # Died after 24h (48h)

    def test_hospital_mortality_with_obs_window(self, sample_stays, sample_mortality_info):
        """Test hospital mortality with observation window excludes obs-window deaths."""
        config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            observation_window_hours=48,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        raw_data = {
            "stays": sample_stays,
            "mortality_info": sample_mortality_info,
        }

        labels = builder.build_labels(raw_data)

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        # Stay 1: died at 10h (during 48h obs window), outtime=48h (still in ICU at obs end)
        # outtime (Jan 3 10:00) >= obs_end (Jan 3 10:00), so NOT left_icu_during_obs -> label=1
        assert labels_dict[1] == 1  # hospital_expire_flag=1
        assert labels_dict[2] == 0  # Survived
        assert labels_dict[3] == 1  # Died in hospital (hospital_expire_flag=1)
        assert labels_dict[4] == 1  # Died in hospital (hospital_expire_flag=1)

    def test_icu_mortality(self, sample_stays):
        """Test ICU mortality prediction (death during ICU stay, window_hours=-1)."""
        # Create mortality info where some patients died during ICU, some after
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "date_of_death": [
                    datetime(2020, 1, 2, 10, 0),  # Died during ICU stay
                    None,  # Survived
                    datetime(2020, 1, 10, 10, 0),  # Died after ICU discharge
                    datetime(2020, 1, 6, 8, 0),  # Died at exact ICU discharge time
                ],
                "hospital_expire_flag": [1, 0, 1, 1],
                "dischtime": [
                    datetime(2020, 1, 3, 10, 0),
                    datetime(2020, 1, 5, 12, 0),
                    datetime(2020, 1, 10, 10, 0),
                    datetime(2020, 1, 10, 14, 0),
                ],
                "discharge_location": ["DIED", "HOME", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_icu",
            task_type="binary_classification",
            prediction_window_hours=-1,  # ICU mortality
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        raw_data = {
            "stays": sample_stays,
            "mortality_info": mortality_info,
        }

        labels = builder.build_labels(raw_data)

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Died during ICU stay
        assert labels_dict[2] == 0  # Survived
        assert labels_dict[3] == 0  # Died after ICU discharge (not ICU mortality)
        assert labels_dict[4] == 1  # Died at exact ICU discharge (counts as ICU)

    def test_empty_stays_returns_empty_dataframe(self):
        """Test that empty stays input returns empty DataFrame with correct schema."""
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)

        # Empty DataFrames
        empty_stays = pl.DataFrame(
            {
                "stay_id": [],
                "intime": [],
                "outtime": [],
            }
        ).cast({"stay_id": pl.Int64})

        empty_mortality = pl.DataFrame(
            {
                "stay_id": [],
                "date_of_death": [],
                "hospital_expire_flag": [],
                "dischtime": [],
                "discharge_location": [],
            }
        ).cast({"stay_id": pl.Int64, "hospital_expire_flag": pl.Int32})

        raw_data = {
            "stays": empty_stays,
            "mortality_info": empty_mortality,
        }

        labels = builder.build_labels(raw_data)

        # Should return empty DataFrame with correct columns
        assert len(labels) == 0
        assert "stay_id" in labels.columns
        assert "label" in labels.columns


class TestMortalityBoundaryConditions:
    """Tests for boundary conditions in mortality prediction."""

    @pytest.fixture
    def boundary_stays(self):
        """Stays for boundary condition testing."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 5,  # All same admission time
                "outtime": [datetime(2020, 1, 3, 0, 0)] * 5,  # All same discharge time
            }
        )

    def test_death_exactly_at_24h_boundary(self, boundary_stays):
        """Test death exactly at 24-hour boundary (should be included)."""
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "date_of_death": [
                    datetime(2020, 1, 2, 0, 0),  # Exactly 24h (should be included)
                    datetime(2020, 1, 1, 23, 59, 59),  # Just before 24h (included)
                    datetime(2020, 1, 2, 0, 0, 1),  # Just after 24h (excluded)
                    None,  # Survived
                    datetime(2020, 1, 1, 12, 0),  # 12h (included)
                ],
                "hospital_expire_flag": [1, 1, 1, 0, 1],
                "dischtime": [datetime(2020, 1, 3, 0, 0)] * 5,
                "discharge_location": ["DIED", "DIED", "DIED", "HOME", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": boundary_stays,
                "mortality_info": mortality_info,
            }
        )

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Exactly at 24h - included
        assert labels_dict[2] == 1  # Just before 24h - included
        assert labels_dict[3] == 0  # Just after 24h - excluded
        assert labels_dict[4] == 0  # Survived
        assert labels_dict[5] == 1  # Well within 24h - included

    def test_legacy_time_bounded_boundary(self, boundary_stays):
        """Test death exactly at time boundary for legacy time-bounded mortality."""
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [
                    datetime(2020, 1, 3, 0, 0),  # Exactly 48h
                    datetime(2020, 1, 3, 0, 0, 1),  # Just after 48h
                    datetime(2020, 1, 2, 23, 59, 59),  # Just before 48h
                ],
                "hospital_expire_flag": [1, 1, 1],
                "dischtime": [datetime(2020, 1, 3, 0, 0)] * 3,
                "discharge_location": ["DIED", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_custom",
            task_type="binary_classification",
            prediction_window_hours=48,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": boundary_stays.filter(pl.col("stay_id").is_in([1, 2, 3])),
                "mortality_info": mortality_info,
            }
        )

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Exactly at 48h - included
        assert labels_dict[2] == 0  # Just after 48h - excluded
        assert labels_dict[3] == 1  # Just before 48h - included

    def test_all_survivors(self, boundary_stays):
        """Test dataset where all patients survive."""
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "date_of_death": [None] * 5,
                "hospital_expire_flag": [0] * 5,
                "dischtime": [datetime(2020, 1, 3, 0, 0)] * 5,
                "discharge_location": ["HOME"] * 5,
            }
        )

        config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": boundary_stays,
                "mortality_info": mortality_info,
            }
        )

        # All should be 0
        assert labels["label"].sum() == 0
        assert len(labels) == 5

    def test_all_deaths(self, boundary_stays):
        """Test dataset where all patients die."""
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "date_of_death": [datetime(2020, 1, 1, 12, 0)] * 5,
                "hospital_expire_flag": [1] * 5,
                "dischtime": [datetime(2020, 1, 1, 12, 0)] * 5,
                "discharge_location": ["DIED"] * 5,
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": boundary_stays,
                "mortality_info": mortality_info,
            }
        )

        # All should be 1
        assert labels["label"].sum() == 5
        assert len(labels) == 5

    def test_single_stay(self):
        """Test with single stay in dataset."""
        stays = pl.DataFrame(
            {
                "stay_id": [1],
                "intime": [datetime(2020, 1, 1, 0, 0)],
                "outtime": [datetime(2020, 1, 3, 0, 0)],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1],
                "date_of_death": [datetime(2020, 1, 1, 12, 0)],
                "hospital_expire_flag": [1],
                "dischtime": [datetime(2020, 1, 1, 12, 0)],
                "discharge_location": ["DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": stays,
                "mortality_info": mortality_info,
            }
        )

        assert len(labels) == 1
        assert labels["label"][0] == 1

    def test_missing_mortality_info_for_stay(self):
        """Test when mortality info is missing for some stays (left join behavior)."""
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 3,
                "outtime": [datetime(2020, 1, 3, 0, 0)] * 3,
            }
        )

        # Only mortality info for stay 1
        mortality_info = pl.DataFrame(
            {
                "stay_id": [1],
                "date_of_death": [datetime(2020, 1, 1, 12, 0)],
                "hospital_expire_flag": [1],
                "dischtime": [datetime(2020, 1, 1, 12, 0)],
                "discharge_location": ["DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_hospital",
            task_type="binary_classification",
            prediction_window_hours=None,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels(
            {
                "stays": stays,
                "mortality_info": mortality_info,
            }
        )

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Has mortality info
        assert labels_dict[2] == 0  # Missing info defaults to 0
        assert labels_dict[3] == 0  # Missing info defaults to 0


class TestWindowedMortalityLabels:
    """Tests for windowed mortality prediction (observation + prediction windows).

    This tests the recommended approach where:
    - observation_window_hours defines how much data the model sees
    - prediction_window_hours defines the prediction target window AFTER observation
    - gap_hours optionally adds buffer between observation and prediction
    """

    def test_windowed_mortality_basic(self):
        """Test basic windowed mortality with obs=48h, gap=0h, pred=24h.

        Timeline: |-- obs (48h) --|-- pred (24h) --|
                  0h            48h              72h

        Key: outtime determines if we have complete observation data.
        - If outtime < obs_end: incomplete observation → excluded (null)
        - If outtime >= obs_end: complete observation → usable (0 or 1)

        Deaths at:
        - 24h with outtime=24h: left ICU during obs → null (excluded)
        - 50h with outtime=50h: died during prediction → 1
        - 72h with outtime=72h: at prediction boundary → 1
        - 80h with outtime=80h: after prediction → 0
        - None with outtime=216h: survived → 0
        - 47h with outtime=47h: during obs, left before obs end → null (excluded)
        - 48h+1s with outtime=48h+1s: just after obs → 1
        """
        # Stays with realistic outtime matching when patient left ICU
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6, 7],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 7,
                "outtime": [
                    datetime(2020, 1, 2, 0, 0),  # 24h - left during obs
                    datetime(2020, 1, 3, 2, 0),  # 50h - left during pred
                    datetime(2020, 1, 4, 0, 0),  # 72h - left at pred boundary
                    datetime(2020, 1, 4, 8, 0),  # 80h - left after pred
                    datetime(2020, 1, 10, 0, 0),  # 216h - long stay, survived
                    datetime(2020, 1, 2, 23, 0),  # 47h - left during obs
                    datetime(2020, 1, 3, 0, 0, 1),  # 48h + 1s - left just after obs
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6, 7],
                "date_of_death": [
                    datetime(2020, 1, 2, 0, 0),  # 24h - during obs
                    datetime(2020, 1, 3, 2, 0),  # 50h - during pred
                    datetime(2020, 1, 4, 0, 0),  # 72h - at pred boundary
                    datetime(2020, 1, 4, 8, 0),  # 80h - after pred
                    None,  # Survived
                    datetime(2020, 1, 2, 23, 0),  # 47h - during obs
                    datetime(2020, 1, 3, 0, 0, 1),  # 48h + 1s - just after obs
                ],
                "hospital_expire_flag": [1, 1, 1, 1, 0, 1, 1],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 7,
                "discharge_location": ["DIED", "DIED", "DIED", "DIED", "HOME", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        # Left ICU during observation → excluded (null) - incomplete observation data
        assert labels_dict[1] is None, "Death at 24h with outtime=24h should be excluded"
        assert labels_dict[6] is None, "Death at 47h with outtime=47h should be excluded"

        # Death during prediction window → positive
        assert labels_dict[2] == 1, "Death at 50h should be positive (during pred)"
        assert labels_dict[3] == 1, "Death at 72h should be positive (pred boundary)"
        assert labels_dict[7] == 1, "Death at 48h+1s should be positive (just after obs)"

        # Death after prediction or survived → negative
        assert labels_dict[4] == 0, "Death at 80h should be negative (after pred)"
        assert labels_dict[5] == 0, "Survivor should be negative"

    def test_windowed_mortality_with_gap(self):
        """Test windowed mortality with gap between observation and prediction.

        Timeline: |-- obs (24h) --|-- gap (6h) --|-- pred (24h) --|
                  0h            24h            30h              54h

        Deaths during gap are labeled as 0 (survived the prediction window).
        outtime must be < obs_end for a patient to be excluded as "died during obs".
        """
        # Stays with realistic outtime values
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 5,
                "outtime": [
                    datetime(2020, 1, 1, 20, 0),  # 20h - left during obs
                    datetime(2020, 1, 2, 3, 0),  # 27h - left during gap
                    datetime(2020, 1, 2, 8, 0),  # 32h - left during pred
                    datetime(2020, 1, 3, 10, 0),  # 58h - left after pred
                    datetime(2020, 1, 10, 0, 0),  # 216h - long stay, survived
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "date_of_death": [
                    datetime(2020, 1, 1, 20, 0),  # 20h - during obs
                    datetime(2020, 1, 2, 3, 0),  # 27h - during gap
                    datetime(2020, 1, 2, 8, 0),  # 32h - during pred
                    datetime(2020, 1, 3, 10, 0),  # 58h - after pred
                    None,  # Survived
                ],
                "hospital_expire_flag": [1, 1, 1, 1, 0],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 5,
                "discharge_location": ["DIED", "DIED", "DIED", "DIED", "HOME"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=24,
            gap_hours=6,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        assert labels_dict[1] is None, "Death during obs should be excluded"
        assert labels_dict[2] == 0, "Death during gap should be negative (not in pred window)"
        assert labels_dict[3] == 1, "Death during pred should be positive"
        assert labels_dict[4] == 0, "Death after pred should be negative"
        assert labels_dict[5] == 0, "Survivor should be negative"

    def test_windowed_mortality_boundary_at_obs_end(self):
        """Test boundary condition exactly at observation window end.

        With outtime-based logic:
        - If outtime < obs_end: left ICU during observation → excluded
        - If outtime >= obs_end: complete observation data → usable

        Death at exactly 48h with outtime=48h means patient was in ICU for full
        observation period, so they're usable. The death is at pred_start (48h)
        which is >= pred_start, so it's a positive case.
        """
        # Stays with outtime matching death time
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 3,
                "outtime": [
                    datetime(2020, 1, 3, 0, 0),  # 48h - exactly at obs boundary
                    datetime(2020, 1, 2, 23, 59, 59),  # 1 second before 48h
                    datetime(2020, 1, 3, 0, 0, 1),  # 1 second after 48h
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [
                    datetime(2020, 1, 3, 0, 0),  # Exactly 48h
                    datetime(2020, 1, 2, 23, 59, 59),  # 1 second before 48h
                    datetime(2020, 1, 3, 0, 0, 1),  # 1 second after 48h
                ],
                "hospital_expire_flag": [1, 1, 1],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 3,
                "discharge_location": ["DIED", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        # At exactly obs boundary with outtime=48h → NOT excluded (outtime >= obs_end)
        # Death at pred_start (48h) is in prediction window → positive
        assert labels_dict[1] == 1, "Death at exactly 48h with outtime=48h should be positive"
        # Before obs boundary with outtime<48h → excluded (incomplete observation)
        assert labels_dict[2] is None, "Death before 48h with outtime<48h should be excluded"
        # After obs boundary → in prediction window
        assert labels_dict[3] == 1, "Death just after 48h should be positive"

    def test_windowed_mortality_boundary_at_pred_end(self):
        """Test boundary condition exactly at prediction window end."""
        # Stays with outtime matching death time (all after obs_end, so all usable)
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 3,
                "outtime": [
                    datetime(2020, 1, 4, 0, 0),  # 72h
                    datetime(2020, 1, 3, 23, 59, 59),  # ~72h
                    datetime(2020, 1, 4, 0, 0, 1),  # 72h + 1s
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [
                    datetime(2020, 1, 4, 0, 0),  # Exactly 72h (48+24)
                    datetime(2020, 1, 3, 23, 59, 59),  # 1 second before 72h
                    datetime(2020, 1, 4, 0, 0, 1),  # 1 second after 72h
                ],
                "hospital_expire_flag": [1, 1, 1],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 3,
                "discharge_location": ["DIED", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        # At exactly pred boundary → included (died_during_pred uses <=)
        assert labels_dict[1] == 1, "Death at exactly 72h should be positive"
        # Before pred boundary → positive
        assert labels_dict[2] == 1, "Death before 72h should be positive"
        # After pred boundary → negative
        assert labels_dict[3] == 0, "Death after 72h should be negative"

    def test_death_exactly_at_prediction_start_is_positive(self):
        """Test that death exactly at prediction window start is positive.

        The prediction window uses >= pred_start, so a death exactly at the moment
        the prediction window begins should be counted as a positive case.

        With outtime-based logic, death at 48h with outtime=48h means patient was
        in ICU for full observation, so they're usable (not excluded).

        Timeline: |-- obs (48h) --|-- pred (24h) --|
                  0h            48h              72h
        """
        # Stays with outtime matching death time
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 4,
                "outtime": [
                    datetime(2020, 1, 3, 0, 0, 1),  # 48h + 1s
                    datetime(2020, 1, 3, 0, 0),  # Exactly 48h
                    datetime(2020, 1, 3, 12, 0),  # 60h
                    datetime(2020, 1, 4, 0, 0),  # 72h
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "date_of_death": [
                    datetime(2020, 1, 3, 0, 0, 1),  # 48h + 1s - just after obs ends
                    datetime(2020, 1, 3, 0, 0),  # Exactly 48h - at prediction start
                    datetime(2020, 1, 3, 12, 0),  # 60h - middle of pred window
                    datetime(2020, 1, 4, 0, 0),  # 72h - at pred end
                ],
                "hospital_expire_flag": [1, 1, 1, 1],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 4,
                "discharge_location": ["DIED", "DIED", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        # Death just after obs ends → positive (in prediction window)
        assert labels_dict[1] == 1, "Death 1s after obs end should be positive"
        # Death exactly at obs boundary with outtime=48h → positive (at pred_start)
        # Patient was in ICU for full 48h observation, death at pred_start is in pred window
        assert labels_dict[2] == 1, "Death at exactly 48h with outtime=48h should be positive"
        # Death in middle of pred window → positive
        assert labels_dict[3] == 1, "Death at 60h should be positive"
        # Death at pred end → positive
        assert labels_dict[4] == 1, "Death at 72h should be positive"

    def test_windowed_mortality_all_excluded(self):
        """Test when all patients leave ICU during observation (all excluded).

        For a patient to be excluded, they must have left ICU during observation
        (outtime < obs_end). All patients here have outtime < 48h.
        """
        # Stays with outtime during observation (all < 48h)
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 3,
                "outtime": [
                    datetime(2020, 1, 1, 12, 0),  # 12h - during obs
                    datetime(2020, 1, 2, 0, 0),  # 24h - during obs
                    datetime(2020, 1, 2, 23, 59, 59),  # 47h 59m 59s - just before obs end
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "date_of_death": [
                    datetime(2020, 1, 1, 12, 0),  # 12h
                    datetime(2020, 1, 2, 0, 0),  # 24h
                    datetime(2020, 1, 2, 23, 59, 59),  # 47h 59m 59s
                ],
                "hospital_expire_flag": [1, 1, 1],
                "dischtime": [datetime(2020, 1, 3, 0, 0)] * 3,
                "discharge_location": ["DIED", "DIED", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            prediction_window_hours=24,
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        # All should be null (excluded) - all left ICU during observation
        assert labels["label"].null_count() == 3, "All patients should be excluded"

    def test_windowed_mortality_until_icu_discharge(self):
        """Test windowed mortality until ICU discharge (prediction_window_hours=-1).

        Timeline: |-- obs (48h) --|-- pred (until ICU discharge) --|
                  0h            48h                            outtime

        This tests the "mortality" task which predicts death during the
        remaining ICU stay after observation ends.

        Note: outtime must be consistent with date_of_death (outtime ~ death time for deaths).
        """
        # Different outtime for each stay to test the dynamic prediction window
        # outtime is consistent with death times
        stays = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6],
                "intime": [datetime(2020, 1, 1, 0, 0)] * 6,
                "outtime": [
                    datetime(2020, 1, 2, 0, 0),  # 24h - died during obs, left ICU at death
                    datetime(2020, 1, 4, 0, 0),  # 72h - died during pred
                    datetime(2020, 1, 5, 0, 0),  # 96h - died at outtime
                    datetime(2020, 1, 4, 0, 0),  # 72h - died at outtime (short stay)
                    datetime(2020, 1, 10, 0, 0),  # 216h stay (long stay, survived)
                    datetime(2020, 1, 5, 0, 0),  # 96h - discharged, died later
                ],
            }
        )

        mortality_info = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5, 6],
                "date_of_death": [
                    datetime(2020, 1, 2, 0, 0),  # 24h - during obs
                    datetime(2020, 1, 4, 0, 0),  # 72h - during pred, at outtime
                    datetime(2020, 1, 5, 0, 0),  # 96h - at outtime
                    datetime(2020, 1, 4, 0, 0),  # 72h - at outtime (short stay)
                    None,  # Survived (long stay)
                    datetime(2020, 1, 6, 0, 0),  # 120h - after outtime (died post-discharge)
                ],
                "hospital_expire_flag": [1, 1, 1, 1, 0, 1],
                "dischtime": [datetime(2020, 1, 10, 0, 0)] * 6,
                "discharge_location": ["DIED", "DIED", "DIED", "DIED", "HOME", "DIED"],
            }
        )

        config = LabelConfig(
            task_name="mortality",
            task_type="binary_classification",
            prediction_window_hours=-1,  # Until ICU discharge
            observation_window_hours=48,
            gap_hours=0,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "mortality_info": mortality_info})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))

        # Death during observation with outtime during obs → excluded (null)
        assert labels_dict[1] is None, "Death at 24h should be excluded (during obs)"

        # Death during prediction window (between obs end and outtime) → positive
        assert labels_dict[2] == 1, "Death at 72h should be positive (during pred)"
        assert labels_dict[3] == 1, "Death at 96h should be positive (at outtime)"
        assert labels_dict[4] == 1, "Death at 72h should be positive (at outtime for short stay)"

        # Survivor or death after discharge → negative
        assert labels_dict[5] == 0, "Survivor should be negative"
        assert labels_dict[6] == 0, "Death at 120h should be negative (after outtime)"


class TestLabelBuilderFactory:
    """Tests for LabelBuilderFactory (minimal - mortality only)."""

    def test_factory_creates_mortality_builder(self):
        """Test factory creates MortalityLabelBuilder."""
        config = LabelConfig(
            task_name="mortality_24h",
            task_type="binary_classification",
            label_sources=["stays", "mortality_info"],
        )

        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, MortalityLabelBuilder)

    def test_factory_handles_underscores(self):
        """Test factory extracts category from task names with underscores."""
        # All of these should create MortalityLabelBuilder
        for task_name in ["mortality_24h", "mortality_hospital"]:
            config = LabelConfig(
                task_name=task_name,
                task_type="binary_classification",
                label_sources=["stays", "mortality_info"],
            )
            builder = LabelBuilderFactory.create(config)
            assert isinstance(builder, MortalityLabelBuilder)

    def test_factory_unknown_task_raises_error(self):
        """Test factory raises error for unknown task category."""
        config = LabelConfig(
            task_name="unknown_task",
            task_type="binary_classification",
            label_sources=["stays"],
        )

        with pytest.raises(ValueError, match="No LabelBuilder registered"):
            LabelBuilderFactory.create(config)

    def test_factory_list_available(self):
        """Test listing available label builders."""
        available = LabelBuilderFactory.list_available()

        assert "mortality" in available
        assert isinstance(available["mortality"], type)
