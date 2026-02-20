"""Tests for AKI (KDIGO) label builder."""

import polars as pl
import pytest
from slices.data.labels import AKILabelBuilder, LabelBuilderFactory, LabelConfig


class TestAKILabelBuilder:
    """Tests for AKILabelBuilder."""

    def _make_config(self, obs_hours: int = 48) -> LabelConfig:
        return LabelConfig(
            task_name="aki_kdigo",
            task_type="binary",
            observation_window_hours=obs_hours,
            label_sources=["stays", "timeseries"],
            label_params={
                "creatinine_col": "crea",
                "baseline_window_hours": 24,
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
        """Stable creatinine trajectory -> label=0."""
        stays = self._make_stays([1])
        # Baseline creatinine ~1.0, stable after obs window
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 6, 12, 18, 24, 30, 48, 54, 60, 72],
            crea_values=[1.0, 1.0, 1.1, 1.0, 1.05, 1.0, 1.1, 1.05, 1.1, 1.0],
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 0

    def test_absolute_criterion_triggers_aki(self):
        """Creatinine rise > 0.3 mg/dL within 48h after obs -> label=1."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 48, 54, 60],
            crea_values=[1.0, 1.0, 1.0, 1.1, 1.4],  # 1.4 - 1.0 = 0.4 >= 0.3 within 12h
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_relative_criterion_triggers_aki(self):
        """Creatinine >= 1.5x baseline after obs -> label=1."""
        stays = self._make_stays([1])
        # Baseline min in first 24h: 1.0
        # Post-obs value: 1.6 >= 1.5 * 1.0 = 1.5 -> AKI
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 48, 60],
            crea_values=[1.0, 1.1, 0.9, 1.6],
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_aki_during_obs_only_no_aki_after(self):
        """AKI criteria met only during observation period -> label=0."""
        stays = self._make_stays([1])
        # Spike during obs (hour 24-30), but stable after obs (hour 48+)
        # Baseline min in first 24h: 1.0
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 24, 30, 48, 60, 72],
            crea_values=[1.0, 1.0, 1.5, 1.0, 1.0, 1.1, 1.0],
        )

        config = self._make_config(obs_hours=48)
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

        config = self._make_config(obs_hours=48)
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

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_no_baseline_no_measurements_in_first_24h(self):
        """No creatinine in first 24h (no baseline) -> label=null."""
        stays = self._make_stays([1])
        # First creatinine at hour 30 (after baseline window)
        ts = self._make_timeseries(
            stay_id=1,
            hours=[30, 48, 60],
            crea_values=[1.0, 1.5, 2.0],
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] is None

    def test_both_criteria_independently_trigger(self):
        """Both absolute and relative criteria should independently trigger AKI."""
        stays = self._make_stays([1, 2])

        # Stay 1: only absolute criterion (baseline=1.0, rise of 0.35 in 48h)
        ts1 = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 48, 60],
            crea_values=[1.0, 1.0, 1.0, 1.35],
        )

        # Stay 2: only relative criterion (baseline=0.6, post-obs=0.91 >= 0.9)
        # 0.91 >= 1.5 * 0.6 = 0.9 -> AKI (relative)
        # But 0.91 - 0.7 = 0.21 < 0.3 (absolute not met)
        ts2 = self._make_timeseries(
            stay_id=2,
            hours=[0, 12, 48, 60],
            crea_values=[0.6, 0.7, 0.7, 0.91],
        )

        ts = pl.concat([ts1, ts2])

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == 1  # Absolute criterion
        assert labels_dict[2] == 1  # Relative criterion

    def test_no_post_obs_creatinine_label_zero(self):
        """Creatinine only during obs window, none after -> label=0."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 24],
            crea_values=[1.0, 1.5, 2.0],
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 0

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

        # Stay 1: AKI (relative criterion)
        ts1 = self._make_timeseries(1, [0, 12, 48, 60], [0.8, 0.9, 0.9, 1.5])
        # Stay 2: No AKI (stable)
        ts2 = self._make_timeseries(2, [0, 12, 48, 60], [1.0, 1.0, 1.0, 1.1])
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

    def test_absolute_rise_exactly_at_threshold(self):
        """Rise of exactly 0.3 should trigger AKI (>= threshold)."""
        stays = self._make_stays([1])
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 48, 54],
            crea_values=[1.0, 1.0, 1.0, 1.3],  # Rise = exactly 0.3
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1

    def test_relative_rise_exactly_at_threshold(self):
        """Creatinine exactly at 1.5x baseline should trigger AKI."""
        stays = self._make_stays([1])
        # Baseline = 1.0, threshold = 1.5 * 1.0 = 1.5
        ts = self._make_timeseries(
            stay_id=1,
            hours=[0, 12, 48, 60],
            crea_values=[1.0, 1.0, 1.0, 1.5],
        )

        config = self._make_config(obs_hours=48)
        builder = AKILabelBuilder(config)
        labels = builder.build_labels({"stays": stays, "timeseries": ts})

        assert labels.filter(pl.col("stay_id") == 1)["label"][0] == 1


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
