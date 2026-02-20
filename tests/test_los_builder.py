"""Tests for remaining length of stay label builder."""

import polars as pl
import pytest
from slices.data.labels import LabelBuilderFactory, LabelConfig
from slices.data.labels.los import LOSLabelBuilder


class TestLOSLabelBuilder:
    """Tests for LOSLabelBuilder."""

    def _make_stays(self, los_days_list: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "stay_id": list(range(1, len(los_days_list) + 1)),
                "los_days": los_days_list,
            }
        )

    def _make_config(self, obs_hours: int | None = 48) -> LabelConfig:
        return LabelConfig(
            task_name="los_remaining",
            task_type="regression",
            observation_window_hours=obs_hours,
            label_sources=["stays"],
        )

    def test_basic_computation(self):
        """stay with los_days=5.0, obs=48h -> label=3.0."""
        stays = self._make_stays([5.0])
        config = self._make_config(obs_hours=48)
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert labels["label"][0] == pytest.approx(3.0)

    def test_clipping_negative(self):
        """stay with los_days=1.0, obs=48h -> label=0.0 (not negative)."""
        stays = self._make_stays([1.0])
        config = self._make_config(obs_hours=48)
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert labels["label"][0] == pytest.approx(0.0)

    def test_exactly_at_boundary(self):
        """los_days=2.0, obs=48h -> label=0.0."""
        stays = self._make_stays([2.0])
        config = self._make_config(obs_hours=48)
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert labels["label"][0] == pytest.approx(0.0)

    def test_obs_hours_none_defaults_to_zero(self):
        """obs_hours=None -> label=los_days."""
        stays = self._make_stays([7.5])
        config = self._make_config(obs_hours=None)
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert labels["label"][0] == pytest.approx(7.5)

    def test_empty_stays(self):
        """Empty stays returns empty DataFrame with correct schema."""
        stays = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "los_days": pl.Series([], dtype=pl.Float64),
            }
        )
        config = self._make_config()
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert len(labels) == 0
        assert "stay_id" in labels.columns
        assert "label" in labels.columns
        assert labels["stay_id"].dtype == pl.Int64
        assert labels["label"].dtype == pl.Float64

    def test_dtype_is_float64(self):
        """Label dtype should be Float64."""
        stays = self._make_stays([5.0, 1.0, 10.0])
        config = self._make_config()
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        assert labels["label"].dtype == pl.Float64

    def test_multiple_stays(self):
        """Multiple stays computed correctly."""
        stays = self._make_stays([5.0, 1.0, 2.0, 10.0])
        config = self._make_config(obs_hours=48)
        builder = LOSLabelBuilder(config)
        labels = builder.build_labels({"stays": stays})

        labels_dict = dict(zip(labels["stay_id"], labels["label"]))
        assert labels_dict[1] == pytest.approx(3.0)  # 5.0 - 2.0
        assert labels_dict[2] == pytest.approx(0.0)  # clipped
        assert labels_dict[3] == pytest.approx(0.0)  # exactly at boundary
        assert labels_dict[4] == pytest.approx(8.0)  # 10.0 - 2.0

    def test_validate_inputs_missing_source(self):
        """Should raise ValueError if 'stays' is missing from raw_data."""
        config = self._make_config()
        builder = LOSLabelBuilder(config)
        with pytest.raises(ValueError, match="missing"):
            builder.build_labels({})


class TestLOSFactory:
    """Tests for factory integration with LOSLabelBuilder."""

    def test_factory_creates_los_builder_for_los_remaining(self):
        """Factory creates LOSLabelBuilder for 'los_remaining' task name."""
        config = LabelConfig(
            task_name="los_remaining",
            task_type="regression",
            label_sources=["stays"],
        )
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, LOSLabelBuilder)

    def test_factory_creates_los_builder_for_los_category(self):
        """Factory creates LOSLabelBuilder for any 'los_*' task name."""
        config = LabelConfig(
            task_name="los_icu",
            task_type="regression",
            label_sources=["stays"],
        )
        builder = LabelBuilderFactory.create(config)
        assert isinstance(builder, LOSLabelBuilder)
