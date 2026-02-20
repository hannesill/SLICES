"""Tests for DecompensationLabelBuilder."""

from datetime import datetime, timedelta
from typing import Dict, List

import polars as pl
import pytest
import yaml
from slices.data.labels import DecompensationLabelBuilder, LabelBuilderFactory, LabelConfig


def _make_config(
    obs: int = 48,
    pred: int = 24,
    stride: int = 6,
) -> LabelConfig:
    return LabelConfig(
        task_name="decompensation",
        task_type="binary",
        observation_window_hours=obs,
        prediction_window_hours=pred,
        label_sources=["stays", "mortality_info"],
        label_params={"stride_hours": stride},
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


class TestWindowGeneration:
    def test_window_count_with_stride(self):
        """100h stay, obs=48, stride=6 -> windows at t=0,6,...,52 -> 9 windows."""
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
        builder = DecompensationLabelBuilder(_make_config(stride=6))
        labels = builder.build_labels(raw)

        # t + 48 <= 100 -> t <= 52 -> t in {0,6,12,18,24,30,36,42,48,52}
        expected_starts = list(range(0, 53, 6))
        assert len(labels) == len(expected_starts)
        assert sorted(labels["window_start"].to_list()) == expected_starts

    def test_stay_too_short(self):
        """30h stay with obs=48 -> 0 samples."""
        raw = _make_raw_data(
            stays_data=[
                {
                    "stay_id": 1,
                    "patient_id": 1,
                    "intime": INTIME,
                    "outtime": INTIME + timedelta(hours=30),
                    "los_days": 30 / 24.0,
                }
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config())
        labels = builder.build_labels(raw)
        assert len(labels) == 0

    def test_empty_stays(self):
        """Empty stays -> empty DataFrame with correct schema."""
        raw = _make_raw_data(stays_data=[], mortality_data=[])
        builder = DecompensationLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert len(labels) == 0
        assert "stay_id" in labels.columns
        assert "window_start" in labels.columns
        assert "label" in labels.columns
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["window_start"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Int32


class TestLabelCorrectness:
    def test_death_in_prediction_window(self):
        """Death at hour 80, window t=30: obs_end=78, pred=[78,102) -> label=1."""
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
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=80)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=1))
        labels = builder.build_labels(raw)

        row = labels.filter(pl.col("window_start") == 30)
        assert len(row) == 1
        assert row["label"][0] == 1

    def test_death_after_prediction_window(self):
        """Death at hour 80, window t=0: obs_end=48, pred=[48,72) -> label=0."""
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
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=80)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=1))
        labels = builder.build_labels(raw)

        row = labels.filter(pl.col("window_start") == 0)
        assert len(row) == 1
        assert row["label"][0] == 0

    def test_death_during_observation_excluded(self):
        """Death at hour 30: windows where 30 is in [t, t+48) are absent."""
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
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=30)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=6))
        labels = builder.build_labels(raw)

        for row in labels.iter_rows(named=True):
            t_start = row["window_start"]
            obs_end = t_start + 48
            assert not (
                t_start <= 30 < obs_end
            ), f"Window at t={t_start} should be excluded (death at 30 in obs)"

    def test_death_exactly_at_obs_end(self):
        """Death at hour 48, t=0: obs=[0,48), death NOT in obs. pred=[48,72), label=1."""
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
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=48)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=48))
        labels = builder.build_labels(raw)

        row = labels.filter(pl.col("window_start") == 0)
        assert len(row) == 1
        assert row["label"][0] == 1

    def test_death_exactly_at_pred_end(self):
        """Death at hour 72, t=0: pred=[48,72), death at 72 -> label=0 (half-open)."""
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
                {"stay_id": 1, "date_of_death": INTIME + timedelta(hours=72)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=48))
        labels = builder.build_labels(raw)

        row = labels.filter(pl.col("window_start") == 0)
        assert len(row) == 1
        assert row["label"][0] == 0

    def test_survivor_all_labels_zero(self):
        """Survivor stay: every label should be 0."""
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
        builder = DecompensationLabelBuilder(_make_config(stride=6))
        labels = builder.build_labels(raw)

        assert len(labels) > 0
        assert (labels["label"] == 0).all()

    def test_multiple_stays_mixed(self):
        """Survivor + deceased patients produce correct labels."""
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
            ],
            mortality_data=[
                {"stay_id": 1, "date_of_death": None},
                {"stay_id": 2, "date_of_death": INTIME + timedelta(hours=60)},
            ],
        )
        builder = DecompensationLabelBuilder(_make_config(stride=6))
        labels = builder.build_labels(raw)

        # Stay 1 (survivor): all labels 0
        stay1 = labels.filter(pl.col("stay_id") == 1)
        assert (stay1["label"] == 0).all()

        # Stay 2 (death at 60): some windows should have label=1
        stay2 = labels.filter(pl.col("stay_id") == 2)
        assert stay2["label"].sum() > 0  # At least one positive label


class TestOutputSchema:
    def test_output_has_window_start_column(self):
        """Verify schema includes window_start."""
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
        builder = DecompensationLabelBuilder(_make_config())
        labels = builder.build_labels(raw)

        assert "window_start" in labels.columns
        assert "stay_id" in labels.columns
        assert "label" in labels.columns
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["window_start"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Int32


class TestValidation:
    def test_validate_inputs_missing_source(self):
        """Missing mortality_info raises ValueError."""
        raw = {"stays": pl.DataFrame({"stay_id": [1]})}
        builder = DecompensationLabelBuilder(_make_config())
        with pytest.raises(ValueError, match="missing"):
            builder.build_labels(raw)


class TestFactory:
    def test_factory_creates_builder(self):
        """LabelBuilderFactory.create() returns DecompensationLabelBuilder."""
        config = _make_config()
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, DecompensationLabelBuilder)


class TestConsistencyWithDataset:
    """Verify that the builder produces identical labels to DecompensationDataset."""

    def test_consistency_with_dataset(self, tmp_path):
        """Same input -> identical (stay_id, window_start, label) tuples."""
        from slices.data.decompensation_dataset import DecompensationDataset

        intime = INTIME
        feature_names = ["hr", "sbp"]
        obs, pred, stride = 48, 24, 6

        stays_data = [
            {
                "stay_id": 1,
                "patient_id": 1,
                "intime": intime,
                "outtime": intime + timedelta(hours=100),
                "los_days": 100 / 24.0,
            },
            {
                "stay_id": 2,
                "patient_id": 2,
                "intime": intime,
                "outtime": intime + timedelta(hours=80),
                "los_days": 80 / 24.0,
            },
        ]
        mortality_data = [
            {
                "stay_id": 1,
                "date_of_death": intime + timedelta(hours=70),
                "hospital_expire_flag": 1,
                "dischtime": intime + timedelta(hours=100),
                "discharge_location": "DIED",
            },
            {
                "stay_id": 2,
                "date_of_death": None,
                "hospital_expire_flag": 0,
                "dischtime": intime + timedelta(hours=80),
                "discharge_location": "HOME",
            },
        ]
        timeseries_data = [
            {"stay_id": sid, "hour": h, "hr": 80.0, "sbp": 120.0}
            for sid in [1, 2]
            for h in range(100)
        ]

        # Write parquet files for DecompensationDataset
        stays_df = pl.DataFrame(stays_data)
        mortality_df = pl.DataFrame(mortality_data)
        ts_df = pl.DataFrame(timeseries_data)

        stays_path = tmp_path / "ricu_stays.parquet"
        mortality_path = tmp_path / "ricu_mortality.parquet"
        ts_path = tmp_path / "ricu_timeseries.parquet"

        stays_df.write_parquet(stays_path)
        mortality_df.write_parquet(mortality_path)
        ts_df.write_parquet(ts_path)

        metadata = {"dataset": "test", "feature_names": feature_names}
        with open(tmp_path / "ricu_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Get labels from DecompensationDataset
        ds = DecompensationDataset(
            ricu_timeseries_path=ts_path,
            stays_path=stays_path,
            mortality_path=mortality_path,
            feature_names=feature_names,
            obs_window_hours=obs,
            pred_window_hours=pred,
            stride_hours=stride,
            normalize=False,
        )
        dataset_tuples = sorted(ds.samples)

        # Get labels from DecompensationLabelBuilder
        raw = {
            "stays": stays_df,
            "mortality_info": mortality_df,
        }
        builder = DecompensationLabelBuilder(_make_config(obs=obs, pred=pred, stride=stride))
        builder_labels = builder.build_labels(raw)
        builder_tuples = sorted(
            zip(
                builder_labels["stay_id"].to_list(),
                builder_labels["window_start"].to_list(),
                builder_labels["label"].to_list(),
            )
        )

        assert dataset_tuples == builder_tuples, (
            f"Builder and Dataset produced different labels.\n"
            f"Dataset:  {dataset_tuples}\n"
            f"Builder:  {builder_tuples}"
        )
