"""Tests for slices/debug/sampling.py.

Tests sentinel patient selection and stratification for debugging.
"""

import numpy as np
import polars as pl
import pytest
from slices.debug.sampling import (
    SelectionStrategy,
    SentinelConfig,
    SentinelSlot,
    StratificationCriterion,
    _apply_criterion,
    _apply_slot_filter,
    _select_from_slot,
    compute_missingness,
    default_age_criterion,
    default_los_criterion,
    default_mortality_criterion,
    default_unit_criterion,
    get_default_criteria,
    get_default_sentinel_slots,
    select_by_missingness,
    select_extreme_stays,
    select_sentinel_patients,
)


class TestStratificationCriterion:
    """Tests for StratificationCriterion dataclass."""

    def test_criterion_with_values(self):
        """Criterion with values should be valid."""
        criterion = StratificationCriterion(
            column="gender",
            values=["M", "F"],
        )
        assert criterion.column == "gender"
        assert criterion.values == ["M", "F"]

    def test_criterion_with_bins(self):
        """Criterion with bins should be valid."""
        criterion = StratificationCriterion(
            column="age",
            bins=[0, 40, 65, float("inf")],
            bin_labels=["young", "middle", "elderly"],
        )
        assert criterion.column == "age"
        assert criterion.bins == [0, 40, 65, float("inf")]

    def test_criterion_requires_values_or_bins(self):
        """Criterion without values or bins should raise."""
        with pytest.raises(ValueError, match="must have either 'values' or 'bins'"):
            StratificationCriterion(column="test")

    def test_criterion_cannot_have_both(self):
        """Criterion with both values and bins should raise."""
        with pytest.raises(ValueError, match="cannot have both"):
            StratificationCriterion(
                column="test",
                values=[1, 2],
                bins=[0, 1, 2],
            )


class TestSentinelSlot:
    """Tests for SentinelSlot dataclass."""

    def test_slot_basic(self):
        """Basic slot creation should work."""
        slot = SentinelSlot(
            name="test_slot",
            filters={"age": (">", 65)},
        )
        assert slot.name == "test_slot"
        assert slot.filters == {"age": (">", 65)}

    def test_slot_with_sorting(self):
        """Slot with sorting options should work."""
        slot = SentinelSlot(
            name="sorted_slot",
            filters={"los_days": ("<", 2)},
            sort_by="los_days",
            sort_descending=False,
        )
        assert slot.sort_by == "los_days"
        assert slot.sort_descending is False

    def test_slot_empty_filters(self):
        """Slot with no filters should work (random selection)."""
        slot = SentinelSlot(name="random_slot", filters={})
        assert slot.filters == {}


class TestDefaultCriteria:
    """Tests for default criterion presets."""

    def test_default_los_criterion(self):
        """Default LOS criterion should have correct bins."""
        criterion = default_los_criterion()

        assert criterion.column == "los_days"
        assert len(criterion.bins) == 5
        assert criterion.bin_labels is not None

    def test_default_age_criterion(self):
        """Default age criterion should have correct bins."""
        criterion = default_age_criterion()

        assert criterion.column == "age"
        assert criterion.bins == [0, 40, 65, float("inf")]

    def test_default_mortality_criterion(self):
        """Default mortality criterion should have values 0 and 1."""
        criterion = default_mortality_criterion()

        assert criterion.column == "mortality_24h"
        assert criterion.values == [0, 1]

    def test_default_unit_criterion(self):
        """Default unit criterion should have ICU types."""
        criterion = default_unit_criterion()

        assert criterion.column == "first_careunit"
        assert "MICU" in criterion.values[0]

    def test_get_default_criteria(self):
        """get_default_criteria should return LOS and age."""
        criteria = get_default_criteria()

        assert len(criteria) == 2
        columns = [c.column for c in criteria]
        assert "los_days" in columns
        assert "age" in columns


class TestDefaultSentinelSlots:
    """Tests for default sentinel slot presets."""

    def test_get_default_sentinel_slots_count(self):
        """Default slots should have 8 entries."""
        slots = get_default_sentinel_slots()
        assert len(slots) == 8

    def test_default_slots_have_names(self):
        """All default slots should have unique names."""
        slots = get_default_sentinel_slots()
        names = [s.name for s in slots]

        assert len(names) == len(set(names))  # All unique

    def test_default_slots_cover_edge_cases(self):
        """Default slots should cover key edge cases."""
        slots = get_default_sentinel_slots()
        names = [s.name for s in slots]

        assert "short_stay_survived" in names
        assert "long_stay" in names
        assert "random_sample" in names


class TestApplySlotFilter:
    """Tests for _apply_slot_filter function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for filter testing."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "age": [25, 45, 55, 70, 85],
                "los_days": [0.5, 1.5, 2.5, 5.0, 10.0],
                "gender": ["M", "F", "M", "F", "M"],
            }
        )

    def test_filter_equals(self, sample_df):
        """Filter with '==' should match exactly."""
        result = _apply_slot_filter(sample_df, "gender", "==", "M")
        assert len(result) == 3
        assert result["gender"].to_list() == ["M", "M", "M"]

    def test_filter_not_equals(self, sample_df):
        """Filter with '!=' should exclude matches."""
        result = _apply_slot_filter(sample_df, "gender", "!=", "M")
        assert len(result) == 2
        assert result["gender"].to_list() == ["F", "F"]

    def test_filter_less_than(self, sample_df):
        """Filter with '<' should match less than value."""
        result = _apply_slot_filter(sample_df, "age", "<", 50)
        assert len(result) == 2
        assert max(result["age"]) < 50

    def test_filter_greater_than(self, sample_df):
        """Filter with '>' should match greater than value."""
        result = _apply_slot_filter(sample_df, "age", ">", 60)
        assert len(result) == 2
        assert min(result["age"]) > 60

    def test_filter_less_equal(self, sample_df):
        """Filter with '<=' should match less than or equal."""
        result = _apply_slot_filter(sample_df, "age", "<=", 45)
        assert len(result) == 2
        assert max(result["age"]) <= 45

    def test_filter_greater_equal(self, sample_df):
        """Filter with '>=' should match greater than or equal."""
        result = _apply_slot_filter(sample_df, "age", ">=", 70)
        assert len(result) == 2
        assert min(result["age"]) >= 70

    def test_filter_in_list(self, sample_df):
        """Filter with 'in' should match values in list."""
        result = _apply_slot_filter(sample_df, "age", "in", [25, 85])
        assert len(result) == 2
        assert set(result["age"].to_list()) == {25, 85}

    def test_filter_between(self, sample_df):
        """Filter with 'between' should match range [min, max)."""
        result = _apply_slot_filter(sample_df, "los_days", "between", (1.0, 3.0))
        assert len(result) == 2
        assert all(1.0 <= v < 3.0 for v in result["los_days"])

    def test_filter_between_invalid_value_raises(self, sample_df):
        """Filter 'between' with non-tuple should raise."""
        with pytest.raises(ValueError, match="requires a tuple"):
            _apply_slot_filter(sample_df, "age", "between", 50)

    def test_filter_unknown_operator_raises(self, sample_df):
        """Filter with unknown operator should raise."""
        with pytest.raises(ValueError, match="Unknown operator"):
            _apply_slot_filter(sample_df, "age", "~=", 50)

    def test_filter_missing_column_raises(self, sample_df):
        """Filter on missing column should raise."""
        with pytest.raises(ValueError, match="not found"):
            _apply_slot_filter(sample_df, "nonexistent", "==", 5)


class TestApplyCriterion:
    """Tests for _apply_criterion function."""

    def test_apply_categorical_criterion(self):
        """Categorical criterion should create stratum column."""
        df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "gender": ["M", "F", "M"],
            }
        )

        criterion = StratificationCriterion(column="gender", values=["M", "F"])
        result = _apply_criterion(df, criterion)

        assert "gender_stratum" in result.columns
        assert result["gender_stratum"].to_list() == ["M", "F", "M"]

    def test_apply_numeric_criterion(self):
        """Numeric criterion should bin values."""
        df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4],
                "age": [30, 50, 70, 90],
            }
        )

        criterion = StratificationCriterion(
            column="age",
            bins=[0, 40, 65, float("inf")],
            bin_labels=["young", "middle", "elderly"],
        )
        result = _apply_criterion(df, criterion)

        assert "age_stratum" in result.columns

    def test_apply_criterion_missing_column_raises(self):
        """Criterion on missing column should raise."""
        df = pl.DataFrame({"stay_id": [1, 2, 3]})

        criterion = StratificationCriterion(column="missing", values=[1, 2])

        with pytest.raises(ValueError, match="not found"):
            _apply_criterion(df, criterion)


class TestSelectFromSlot:
    """Tests for _select_from_slot function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for selection testing."""
        return pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "age": [25, 45, 55, 70, 85],
                "los_days": [0.5, 1.5, 2.5, 5.0, 10.0],
            }
        )

    def test_select_matching_patient(self, sample_df):
        """Should select patient matching slot criteria."""
        slot = SentinelSlot(
            name="elderly",
            filters={"age": (">", 80)},
        )
        rng = np.random.default_rng(42)

        stay_id = _select_from_slot(sample_df, slot, set(), rng)

        assert stay_id == 5  # Only patient with age > 80

    def test_select_with_sorting(self, sample_df):
        """Should select based on sort order."""
        slot = SentinelSlot(
            name="longest_stay",
            filters={"los_days": (">", 1.0)},
            sort_by="los_days",
            sort_descending=True,
        )
        rng = np.random.default_rng(42)

        stay_id = _select_from_slot(sample_df, slot, set(), rng)

        assert stay_id == 5  # Patient with longest stay

    def test_select_excludes_already_selected(self, sample_df):
        """Should not select already selected patients."""
        slot = SentinelSlot(name="any", filters={})
        rng = np.random.default_rng(42)

        # All but one already selected
        already_selected = {1, 2, 3, 4}
        stay_id = _select_from_slot(sample_df, slot, already_selected, rng)

        assert stay_id == 5

    def test_select_returns_none_no_match(self, sample_df):
        """Should return None when no patients match."""
        slot = SentinelSlot(
            name="impossible",
            filters={"age": (">", 100)},
        )
        rng = np.random.default_rng(42)

        stay_id = _select_from_slot(sample_df, slot, set(), rng)

        assert stay_id is None


class TestSelectSentinelPatients:
    """Tests for select_sentinel_patients function."""

    @pytest.fixture
    def sample_static_df(self):
        """Create sample static DataFrame."""
        return pl.DataFrame(
            {
                "stay_id": list(range(1, 101)),
                "age": [20 + (i % 70) for i in range(100)],
                "los_days": [0.5 + (i % 10) for i in range(100)],
                "gender": ["M" if i % 2 == 0 else "F" for i in range(100)],
            }
        )

    def test_select_with_default_config(self, sample_static_df):
        """Should select patients with default slots."""
        result = select_sentinel_patients(sample_static_df)

        assert "stay_id" in result.columns
        assert "slot_name" in result.columns
        assert len(result) > 0

    def test_select_with_custom_slots(self, sample_static_df):
        """Should select patients based on custom slots."""
        config = SentinelConfig(
            slots=[
                SentinelSlot(name="young", filters={"age": ("<", 30)}),
                SentinelSlot(name="old", filters={"age": (">", 70)}),
            ]
        )

        result = select_sentinel_patients(sample_static_df, config)

        assert len(result) == 2
        assert "young" in result["slot_name"].to_list()
        assert "old" in result["slot_name"].to_list()

    def test_select_includes_original_columns(self, sample_static_df):
        """Result should include original static columns."""
        result = select_sentinel_patients(sample_static_df)

        assert "age" in result.columns
        assert "los_days" in result.columns

    def test_select_reproducible_with_seed(self, sample_static_df):
        """Selection should be reproducible with same seed."""
        config1 = SentinelConfig(
            slots=[SentinelSlot(name="random", filters={})],
            seed=42,
        )
        config2 = SentinelConfig(
            slots=[SentinelSlot(name="random", filters={})],
            seed=42,
        )

        result1 = select_sentinel_patients(sample_static_df, config1)
        result2 = select_sentinel_patients(sample_static_df, config2)

        assert result1["stay_id"].to_list() == result2["stay_id"].to_list()


class TestComputeMissingness:
    """Tests for compute_missingness function."""

    def test_compute_missingness_all_observed(self):
        """All observed should have 0% missingness."""
        df = pl.DataFrame(
            {
                "stay_id": [1],
                "mask": [[[True, True], [True, True]]],  # 2x2, all observed
            }
        )

        result = compute_missingness(df)

        assert result["missingness_pct"][0] == pytest.approx(0.0)
        assert result["n_observed"][0] == 4
        assert result["n_total"][0] == 4

    def test_compute_missingness_half_missing(self):
        """Half missing should have 50% missingness."""
        df = pl.DataFrame(
            {
                "stay_id": [1],
                "mask": [[[True, False], [True, False]]],  # 2x2, half observed
            }
        )

        result = compute_missingness(df)

        assert result["missingness_pct"][0] == pytest.approx(50.0)
        assert result["n_observed"][0] == 2

    def test_compute_missingness_all_missing(self):
        """All missing should have 100% missingness."""
        df = pl.DataFrame(
            {
                "stay_id": [1],
                "mask": [[[False, False], [False, False]]],
            }
        )

        result = compute_missingness(df)

        assert result["missingness_pct"][0] == pytest.approx(100.0)

    def test_compute_missingness_missing_column_raises(self):
        """Missing mask column should raise."""
        df = pl.DataFrame({"stay_id": [1], "values": [[1, 2, 3]]})

        with pytest.raises(ValueError, match="not found"):
            compute_missingness(df)


class TestSelectByMissingness:
    """Tests for select_by_missingness function."""

    def test_select_by_missingness_basic(self):
        """Should select patients stratified by missingness."""
        # Create timeseries with varying missingness
        df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "mask": [
                    [[True] * 10],  # 0% missing
                    [[True] * 8 + [False] * 2],  # 20% missing
                    [[True] * 5 + [False] * 5],  # 50% missing
                    [[True] * 3 + [False] * 7],  # 70% missing
                    [[False] * 10],  # 100% missing
                ],
            }
        )

        result = select_by_missingness(df, n_per_bin=1)

        assert "missingness_bin" in result.columns
        assert "missingness_pct" in result.columns


class TestSelectExtremeStays:
    """Tests for select_extreme_stays function."""

    def test_select_extreme_stays_basic(self):
        """Should select min and max values."""
        df = pl.DataFrame(
            {
                "stay_id": list(range(1, 11)),
                "los_days": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )

        result = select_extreme_stays(df, column="los_days", n_each=2)

        assert len(result) == 4  # 2 min + 2 max
        assert set(result["extreme_type"].to_list()) == {"min", "max"}

        min_stays = result.filter(pl.col("extreme_type") == "min")
        assert set(min_stays["los_days"].to_list()) == {1.0, 2.0}

        max_stays = result.filter(pl.col("extreme_type") == "max")
        assert set(max_stays["los_days"].to_list()) == {9.0, 10.0}

    def test_select_extreme_stays_missing_column_raises(self):
        """Missing column should raise."""
        df = pl.DataFrame({"stay_id": [1, 2, 3]})

        with pytest.raises(ValueError, match="not found"):
            select_extreme_stays(df, column="nonexistent")

    def test_select_extreme_stays_handles_nulls(self):
        """Should handle null values in column."""
        df = pl.DataFrame(
            {
                "stay_id": [1, 2, 3, 4, 5],
                "los_days": [1.0, None, 3.0, None, 5.0],
            }
        )

        result = select_extreme_stays(df, column="los_days", n_each=1)

        # Should only include non-null values
        assert len(result) == 2
        assert None not in result["los_days"].to_list()


class TestSelectionStrategy:
    """Tests for SelectionStrategy enum."""

    def test_selection_strategy_values(self):
        """SelectionStrategy should have expected values."""
        assert SelectionStrategy.RANDOM.value == "random"
        assert SelectionStrategy.EXTREME.value == "extreme"
        assert SelectionStrategy.PERCENTILE.value == "percentile"
