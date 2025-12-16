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

    def test_48h_mortality(self, sample_stays, sample_mortality_info):
        """Test 48-hour mortality prediction."""
        config = LabelConfig(
            task_name="mortality_48h",
            task_type="binary_classification",
            prediction_window_hours=48,
            label_sources=["stays", "mortality_info"],
        )

        builder = MortalityLabelBuilder(config)
        raw_data = {
            "stays": sample_stays,
            "mortality_info": sample_mortality_info,
        }

        labels = builder.build_labels(raw_data)

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Died within 48h (10h)
        assert labels_dict[2] == 0  # Survived
        assert labels_dict[3] == 0  # Died after 48h (50h)
        assert labels_dict[4] == 1  # Died within 48h (exactly 48h)

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

    def test_death_exactly_at_48h_boundary(self, boundary_stays):
        """Test death exactly at 48-hour boundary."""
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
            task_name="mortality_48h",
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
        for task_name in ["mortality_24h", "mortality_48h", "mortality_hospital"]:
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
