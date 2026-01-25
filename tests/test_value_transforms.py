"""Tests for slices/data/value_transforms.py.

Tests the value transformation registry and unit conversion functions used
during feature extraction.
"""

import numpy as np
import polars as pl
import pytest
from slices.data.value_transforms import (
    _TRANSFORM_REGISTRY,
    apply_transform,
    eicu_age_parse,
    fahrenheit_to_celsius,
    fio2_to_fraction_robust,
    gcs_eye_text_to_numeric,
    get_transform,
    list_transforms,
    mg_dl_to_mmol_l_glucose,
    minutes_to_days,
    minutes_to_hours,
    percent_to_fraction,
    register_transform,
    to_celsius,
)


class TestTransformRegistry:
    """Tests for the transform registry system."""

    def test_list_transforms_returns_registered(self):
        """list_transforms should return all registered transform names."""
        transforms = list_transforms()
        assert isinstance(transforms, list)
        assert "fahrenheit_to_celsius" in transforms
        assert "mg_dl_to_mmol_l_glucose" in transforms
        assert "to_celsius" in transforms

    def test_get_transform_returns_function(self):
        """get_transform should return callable for registered transforms."""
        transform = get_transform("fahrenheit_to_celsius")
        assert callable(transform)
        assert transform is fahrenheit_to_celsius

    def test_get_transform_raises_for_unknown(self):
        """get_transform should raise ValueError for unknown transforms."""
        with pytest.raises(ValueError, match="Unknown transform 'not_a_transform'"):
            get_transform("not_a_transform")

    def test_register_transform_decorator(self):
        """register_transform decorator should add function to registry."""

        # Register a new transform
        @register_transform("test_double")
        def test_double(values: pl.Series) -> pl.Series:
            return values * 2

        assert "test_double" in _TRANSFORM_REGISTRY
        assert get_transform("test_double") is test_double

        # Clean up
        del _TRANSFORM_REGISTRY["test_double"]


class TestUnitConversions:
    """Tests for unit conversion transforms."""

    def test_fahrenheit_to_celsius_known_values(self):
        """Fahrenheit to Celsius should convert correctly."""
        # Known conversions: 32F = 0C, 212F = 100C, 98.6F = 37C
        values = pl.Series([32.0, 212.0, 98.6])
        result = fahrenheit_to_celsius(values)

        expected = [0.0, 100.0, 37.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fahrenheit_to_celsius_handles_nulls(self):
        """Fahrenheit conversion should propagate null values."""
        values = pl.Series([98.6, None, 100.4])
        result = fahrenheit_to_celsius(values)

        assert result[0] == pytest.approx(37.0)
        assert result[1] is None
        assert result[2] == pytest.approx(38.0)

    def test_mg_dl_to_mmol_l_glucose(self):
        """Glucose mg/dL to mmol/L conversion should be correct."""
        # 180 mg/dL = 10 mmol/L, 90 mg/dL = 5 mmol/L
        values = pl.Series([180.0, 90.0, 126.0])
        result = mg_dl_to_mmol_l_glucose(values)

        expected = [10.0, 5.0, 7.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_minutes_to_days(self):
        """Minutes to days conversion should be correct."""
        # 1440 minutes = 1 day, 2880 = 2 days
        values = pl.Series([1440.0, 2880.0, 720.0])
        result = minutes_to_days(values)

        expected = [1.0, 2.0, 0.5]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_minutes_to_hours(self):
        """Minutes to hours conversion should be correct."""
        values = pl.Series([60.0, 120.0, 30.0])
        result = minutes_to_hours(values)

        expected = [1.0, 2.0, 0.5]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_percent_to_fraction(self):
        """Percent to fraction conversion should divide by 100."""
        values = pl.Series([21.0, 50.0, 100.0])
        result = percent_to_fraction(values)

        expected = [0.21, 0.50, 1.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fio2_robust_handles_percentages(self):
        """FiO2 robust transform should convert percentage values (>1.5) to fractions."""
        # Values clearly in percentage format (21-100)
        values = pl.Series([21.0, 50.0, 100.0, 40.0])
        result = fio2_to_fraction_robust(values)

        expected = [0.21, 0.50, 1.0, 0.40]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fio2_robust_handles_fractions(self):
        """FiO2 robust transform should preserve fraction values (<=1.5)."""
        # Values already in fraction format (0.21-1.0)
        values = pl.Series([0.21, 0.50, 1.0, 0.40])
        result = fio2_to_fraction_robust(values)

        expected = [0.21, 0.50, 1.0, 0.40]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fio2_robust_handles_mixed_inputs(self):
        """FiO2 robust transform should handle mixed percentage and fraction inputs."""
        # Mixed values: some percentage, some fraction
        values = pl.Series([21.0, 0.5, 100.0, 0.21, 50.0, 1.0])
        result = fio2_to_fraction_robust(values)

        # 21.0 > 1.5 → 0.21, 0.5 <= 1.5 → 0.5, 100 > 1.5 → 1.0, etc.
        expected = [0.21, 0.5, 1.0, 0.21, 0.50, 1.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fio2_robust_handles_edge_case_values(self):
        """FiO2 robust transform should handle edge case values around threshold."""
        # Test values around the 1.5 threshold
        values = pl.Series([1.0, 1.5, 1.6, 2.0])
        result = fio2_to_fraction_robust(values)

        # 1.0 <= 1.5 → 1.0 (fraction)
        # 1.5 <= 1.5 → 1.5 (fraction, edge case)
        # 1.6 > 1.5 → 0.016 (percentage)
        # 2.0 > 1.5 → 0.02 (percentage)
        expected = [1.0, 1.5, 0.016, 0.02]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_fio2_robust_handles_nulls(self):
        """FiO2 robust transform should propagate null values."""
        values = pl.Series([21.0, None, 0.5])
        result = fio2_to_fraction_robust(values)

        assert result[0] == pytest.approx(0.21)
        assert result[1] is None
        assert result[2] == pytest.approx(0.5)


class TestDataFrameTransforms:
    """Tests for DataFrame-based transforms with context."""

    def test_to_celsius_converts_fahrenheit_items(self):
        """to_celsius should convert only Fahrenheit itemids."""
        # itemid 223761 = Fahrenheit, 223762 = Celsius
        df = pl.DataFrame(
            {
                "itemid": [223761, 223762, 223761],
                "valuenum": [98.6, 37.0, 100.4],
            }
        )

        result = to_celsius(df, {})

        # First and third rows should be converted, second unchanged
        assert result["valuenum"][0] == pytest.approx(37.0)
        assert result["valuenum"][1] == pytest.approx(37.0)
        assert result["valuenum"][2] == pytest.approx(38.0)

    def test_to_celsius_preserves_celsius_values(self):
        """to_celsius should not modify already-Celsius values."""
        df = pl.DataFrame(
            {
                "itemid": [223762, 223762],  # All Celsius
                "valuenum": [36.5, 37.5],
            }
        )

        result = to_celsius(df, {})

        np.testing.assert_array_almost_equal(
            result["valuenum"].to_numpy(),
            [36.5, 37.5],
        )


class TestTextToNumericTransforms:
    """Tests for text-to-numeric value transforms."""

    def test_gcs_eye_text_to_numeric_valid(self):
        """GCS eye text should map to correct numeric scores."""
        values = pl.Series(["Spontaneous", "To Voice", "To Pain", "None"])
        result = gcs_eye_text_to_numeric(values)

        expected = [4.0, 3.0, 2.0, 1.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_gcs_eye_text_to_numeric_unknown(self):
        """GCS eye should return None for unknown values."""
        values = pl.Series(["Unknown", "Invalid"])
        result = gcs_eye_text_to_numeric(values)

        assert result[0] is None
        assert result[1] is None

    def test_eicu_age_parse_numeric(self):
        """eICU age parser should handle normal numeric ages."""
        values = pl.Series(["45", "72", "30"])
        result = eicu_age_parse(values)

        expected = [45.0, 72.0, 30.0]
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_eicu_age_parse_greater_than_89(self):
        """eICU age parser should convert '>89' to 90."""
        values = pl.Series([">89", "> 89", "45"])
        result = eicu_age_parse(values)

        assert result[0] == 90.0
        assert result[1] == 90.0  # Any value starting with '>' becomes 90
        assert result[2] == 45.0

    def test_eicu_age_parse_nulls(self):
        """eICU age parser should handle null values."""
        values = pl.Series([None, "55", None])
        result = eicu_age_parse(values)

        assert result[0] is None
        assert result[1] == 55.0
        assert result[2] is None


class TestApplyTransform:
    """Tests for the apply_transform convenience function."""

    def test_apply_transform_with_series(self):
        """apply_transform should work with Series input."""
        values = pl.Series([98.6, 100.4])
        result = apply_transform("fahrenheit_to_celsius", values)

        assert isinstance(result, pl.Series)
        assert result[0] == pytest.approx(37.0)
        assert result[1] == pytest.approx(38.0)

    def test_apply_transform_with_dataframe(self):
        """apply_transform should work with DataFrame input for context transforms."""
        df = pl.DataFrame(
            {
                "itemid": [223761],
                "valuenum": [98.6],
            }
        )
        result = apply_transform("to_celsius", df)

        assert isinstance(result, pl.DataFrame)
        assert result["valuenum"][0] == pytest.approx(37.0)

    def test_apply_transform_unknown_raises(self):
        """apply_transform should raise for unknown transforms."""
        values = pl.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform("not_registered", values)
