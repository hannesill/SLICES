"""Tests for AKI (KDIGO) label builder."""

import polars as pl
import pytest
from slices.data.labels import AKILabelBuilder, LabelBuilderFactory, LabelConfig


class TestAKILabelBuilder:
    """Tests for AKILabelBuilder."""

    def _make_config(self, obs_hours: int = 48, prediction_hours: int = 24) -> LabelConfig:
        return LabelConfig(
            task_name="aki_kdigo",
            task_type="binary",
            observation_window_hours=obs_hours,
            prediction_window_hours=prediction_hours,
            label_sources=["stays", "timeseries"],
            label_params={
                "creatinine_col": "crea",
                "baseline_window_hours": 48,
                "absolute_rise_threshold": 0.3,
                "relative_rise_threshold": 1.5,
            },
        )

    def _make_stays(self, stay_ids: list[int]) -> pl.DataFrame:
        return pl.DataFrame({"stay_id": stay_ids})

    def _make_timeseries(
        self, stay_id: int, hours: list[int], crea_values: list[float]
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "stay_id": [stay_id] * len(hours),
                "hour": hours,
                "crea": crea_values,
            }
        )

    def test_stable_creatinine_no_aki(self):
        """Stable creatinine in prediction window (48-72) -> label=0."""
        stays = self._make_stays([1])
        # Baseline creatinine ~1.0 in hours 0-48, stable in prediction window 48-72
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66],
            crea_values=[1.0, 1.0, 1.1, 1.0, 1.05, 1.0, 1.0, 1.05, 1.1, 1.05, 1.1, 1.0],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 0

    def test_absolute_criterion_triggers_aki(self):
        """Creatinine rise > 0.3 mg/dL within 48h in prediction window -> label=1."""
        stays = self._make_stays([1])
        # Baseline in 0-48, rise in prediction window 48-72
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 36, 48, 54, 60],
            crea_values=[1.0, 1.0, 1.0, 1.0, 1.1, 1.4],  # 1.4 - 1.0 = 0.4 >= 0.3 within 12h
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_relative_criterion_triggers_aki(self):
        """Creatinine >= 1.5x baseline in prediction window -> label=1."""
        stays = self._make_stays([1])
        # Baseline min in first 48h: 1.0
        # Prediction window value: 1.6 >= 1.5 * 1.0 = 1.5 -> AKI
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 36, 48, 60],
            crea_values=[1.0, 1.1, 0.9, 0.9, 1.6],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_aki_during_obs_only_no_aki_after(self):
        """AKI criteria met only during observation window (0-48) -> label=0.

        Creatinine spike is in the baseline/observation period, not the prediction window.
        """
        stays = self._make_stays([1])
        # Spike during obs (hour 24-30), but stable in prediction window (hour 48+)
        # Baseline min in first 48h: 1.0
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 24, 30, 36, 48, 54, 60, 66],
            crea_values=[1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.1, 1.0, 1.05],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 0

    def test_no_creatinine_measurements_null_label(self):
        """No creatinine measurements -> label=null."""
        stays = self._make_stays([1])
        # Timeseries without creatinine column
        ts = pl.DataFrame(
            {
                "stay_id": [1, 1],
                "hour": [0, 24],
                "heart_rate": [80.0, 85.0],
            }
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_all_creatinine_null_null_label(self):
        """Creatinine column exists but all values null -> label=null."""
        stays = self._make_stays([1])
        ts = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "hour": [0, 24, 48],
                "crea": [None, None, None],
            }
        ).cast({"crea": pl.Float64})

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_no_baseline_no_measurements_in_first_48h(self):
        """No creatinine in first 48h (no baseline) -> label=null."""
        stays = self._make_stays([1])
        # First creatinine at hour 50 (after baseline window)
        ts = self._make_timeseries(
            stay_id=1,
            hours=[50, 60, 66],
            crea_values=[1.0, 1.5, 2.0],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_both_criteria_independently_trigger(self):
        """Both absolute and relative criteria should independently trigger AKI."""
        stays = self._make_stays([1, 2])

        # Stay 1: only absolute criterion (baseline=1.0, rise of 0.35 in prediction window)
        ts1 = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 36, 48, 60],
            crea_values=[1.0, 1.0, 1.0, 1.0, 1.35],
        )

        # Stay 2: only relative criterion (baseline=0.6, prediction window=0.91 >= 0.9)
        # 0.91 >= 1.5 * 0.6 = 0.9 -> AKI (relative)
        # But 0.91 - 0.7 = 0.21 < 0.3 (absolute not met)
        ts2 = self._make_timeseries(
            stay_id=2,
            hours=[0, 12, 36, 48, 60],
            crea_values=[0.6, 0.7, 0.7, 0.7, 0.91],
        )

        ts = pl.concat([ts1, ts2])

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Absolute criterion
        assert labels_dict[2] == 1  # Relative criterion

    def test_no_prediction_window_creatinine_null_label(self):
        """Creatinine only in observation window (0-48), none in prediction window -> label=null."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 24, 36],
            crea_values=[1.0, 1.5, 2.0, 1.8],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_empty_stays(self):
        """Empty stays returns empty DataFrame with correct schema."""
        stays = pl.DataFrame({"stay_id": pl.Series([], dtype=pl.Int64)})
        ts = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "hour": pl.Series([], dtype=pl.Int64),
                "crea": pl.Series([], dtype=pl.Float64),
            }
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert len(labels) == 0
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Int32

    def test_multiple_stays_mixed_outcomes(self):
        """Multiple stays with different outcomes."""
        stays = self._make_stays([1, 2, 3])

        # Stay 1: AKI (relative criterion in prediction window)
        ts1 = self._make_timeseries(1, [0, 12, 36, 48, 60], [0.8, 0.9, 0.9, 0.9, 1.5])
        # Stay 2: No AKI (stable in prediction window)
        ts2 = self._make_timeseries(2, [0, 12, 36, 48, 60], [1.0, 1.0, 1.0, 1.0, 1.1])
        # Stay 3: No creatinine data -> null
        ts3 = pl.DataFrame(
            {
                "stay_id": [3, 3],
                "hour": [0, 48],
                "heart_rate": [80.0, 85.0],
            }
        )

        ts = pl.concat([ts1, ts2, ts3], how="diagonal")
        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # AKI
        assert labels_dict[2] == 0  # No AKI
        assert labels_dict[3] is None  # No creatinine

    def test_validate_inputs_missing_source(self):
        """Should raise ValueError if required source is missing."""
        config = self._make_config()
        builder = AKILabelBuilder(config)
        with pytest.raises(ValueError, match="missing"):
            builder.build_labels({"stays": self._make_stays([1])})

    def test_missing_prediction_window_raises(self):
        """AKI builder must raise if prediction_window_hours is not set."""
        config = LabelConfig(
            task_name="aki_kdigo",
            task_type="binary",
            observation_window_hours=48,
            prediction_window_hours=None,
            label_sources=["stays", "timeseries"],
            label_params={"creatinine_col": "crea", "baseline_window_hours": 48},
        )
        stays = self._make_stays([1])
        ts = self._make_timeseries(1, [0, 12, 48, 60], [1.0, 1.0, 1.0, 1.5])
        builder = AKILabelBuilder(config)
        with pytest.raises(ValueError, match="prediction_window_hours"):
            builder.build_labels({"stays": stays, "timeseries": ts})

    def test_absolute_rise_exactly_at_threshold(self):
        """Rise of exactly 0.3 should trigger AKI (>= threshold)."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 36, 48, 54],
            crea_values=[1.0, 1.0, 1.0, 1.0, 1.3],  # Rise = exactly 0.3
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_relative_rise_exactly_at_threshold(self):
        """Creatinine exactly at 1.5x baseline should trigger AKI."""
        stays = self._make_stays([1])
        # Baseline = 1.0, threshold = 1.5 * 1.0 = 1.5
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 36, 48, 60],
            crea_values=[1.0, 1.0, 1.0, 1.0, 1.5],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_short_stay_only_baseline_data_null_label(self):
        """Creatinine only in hours 0-48 (baseline), none in 48-72 -> label=null."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 6, 12, 24, 36, 42],
            crea_values=[1.0, 1.1, 1.0, 1.2, 1.0, 1.1],
        )

        config = self._make_config()
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None


class TestAKIFactory:
    """Tests for factory integration with AKILabelBuilder."""

    def test_factory_creates_aki_builder(self):
        """Factory creates AKILabelBuilder for 'aki_kdigo' task name."""
        config = LabelConfig(
            task_name="aki_kdigo",
            task_type="binary",
            label_sources=["stays", "timeseries"],
        )
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, AKILabelBuilder)

    def test_factory_creates_aki_builder_for_any_aki_task(self):
        """Factory creates AKILabelBuilder for any 'aki_*' task name."""
        config = LabelConfig(
            task_name="aki_stage3",
            task_type="binary",
            label_sources=["stays", "timeseries"],
        )
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, AKILabelBuilder)
