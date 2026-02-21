"""Tests for DeathHoursLabelBuilder."""

from datetime import datetime, timedelta
from typing import Dict, List

import polars as pl
import pytest
from slices.data.labels import DeathHoursLabelBuilder, LabelBuilderFactory, LabelConfig


def _make_config() -> LabelConfig:
    return LabelConfig(
        task_name="decompensation",
        task_type="binary",
        observation_window_hours=48,
        prediction_window_hours=24,
        label_sources=["stays", "mortality_info"],
        label_params={"window_label_mode": "death_hours", "stride_hours": 6},
    )


def _make_raw_data(
    stays_data: List[Dict],
    mortality_data: List[Dict],
) -> Dict[str, pl.DataFrame]:
    return {
        "stays": pl.DataFrame(stays_data),
        "mortality_info": pl.DataFrame(mortality_data),
    }


INTIME = datetime(2020, 1, 1, 0, 0)


class TestDeathHoursOutput:
    def test_one_row_per_stay(self):
        """Each stay should produce exactly one label row."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                },
                {
                    "stay_id": 2,
                    "patient_id": 2,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=80),
                    "los_days": 80 / 24.0,
                },
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
                {"stay_id": 2, "date_of_death": INTIME + timedelta(hours=60)},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert len(labels) == 2
        assert set(labels["stay_id"].to_list()) == {1, 2}

    def test_survivor_gets_inf(self):
        """Survivors should have label = inf."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert len(labels) == 1
        label_val = labels["label"][0]
        assert label_val == float("inf")

    def test_deceased_gets_correct_hours(self):
        """Death at 60h from admission should yield label = 60.0."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=60)},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert len(labels) == 1
        assert abs(labels["label"][0] - 60.0) < 0.01

    def test_deceased_fractional_hours(self):
        """Death at 30 minutes past hour 10 -> 10.5h."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=10, minutes=30)},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert abs(labels["label"][0] - 10.5) < 0.01

    def test_mixed_survivors_and_deceased(self):
        """Multiple stays with mix of survivors and deceased."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                },
                {
                    "stay_id": 2,
                    "patient_id": 2,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                },
                {
                    "stay_id": 3,
                    "patient_id": 3,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                },
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
                {"stay_id": 2, "date_of_death": INTIME + timedelta(hours=72)},
                {"stay_id": 3, "date_of_death": None},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        labels_dict = {row["stay_id"]: row["label"] for row in labels.iter_rows(named=True)}
        assert labels_dict[1] == float("inf")
        assert abs(labels_dict[2] - 72.0) < 0.01
        assert labels_dict[3] == float("inf")


class TestOutputSchema:
    def test_schema_types(self):
        """Output should have stay_id (Int64) and label (Float64)."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert "stay_id" in labels.columns
        assert "label" in labels.columns
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Float64

    def test_no_window_start_column(self):
        """Output should NOT have window_start (that's the old sliding-window format)."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=100),
                    "los_days": 100 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
            ],
        )
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert "window_start" not in labels.columns

    def test_empty_stays(self):
        """Empty stays -> empty DataFrame with correct schema."""
        raw = _make_raw_data(stays_data=[], mortality_data=[])
        builder = DeathHoursLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert len(labels) == 0
        assert "stay_id" in labels.columns
        assert "label" in labels.columns
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Float64


class TestValidation:
    def test_validate_inputs_missing_source(self):
        """Missing mortality_info raises ValueError."""
        raw = {"stays": pl.DataFrame({"stay_id": [1]})}
        builder = DeathHoursLabelBuilder(_make_config())
        with pytest.raises(ValueError, match="missing"):
            builder.build_labels(raw)


class TestFactory:
    def test_factory_creates_builder(self):
        """LabelBuilderFactory.create() returns DeathHoursLabelBuilder."""
        config = _make_config()
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, DeathHoursLabelBuilder)
