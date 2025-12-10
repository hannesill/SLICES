"""Tests for data transformation callbacks."""

import polars as pl
import pytest

from slices.data.callbacks import get_callback, list_callbacks, register_callback


def test_list_callbacks():
    """Test that built-in callbacks are registered."""
    callbacks = list_callbacks()
    assert "to_celsius" in callbacks


def test_get_callback():
    """Test getting a callback by name."""
    callback = get_callback("to_celsius")
    assert callable(callback)


def test_get_unknown_callback_raises_error():
    """Test that getting an unknown callback raises ValueError."""
    with pytest.raises(ValueError, match="Unknown callback 'nonexistent'"):
        get_callback("nonexistent")


def test_to_celsius_conversion():
    """Test to_celsius callback converts Fahrenheit correctly."""
    # Test data with both Fahrenheit (223761) and Celsius (223762)
    df = pl.DataFrame({
        "itemid": [223761, 223762, 223761, 223762],
        "valuenum": [98.6, 37.0, 100.4, 38.0],  # Mix of F and C
        "stay_id": [1, 1, 2, 2],
        "charttime": ["2020-01-01 10:00", "2020-01-01 10:05", 
                      "2020-01-01 11:00", "2020-01-01 11:05"],
    })
    
    callback = get_callback("to_celsius")
    result = callback(df, {})
    
    # Check that Fahrenheit values were converted
    fahrenheit_rows = result.filter(pl.col("itemid") == 223761)
    # 98.6°F = 37.0°C, 100.4°F = 38.0°C
    assert abs(fahrenheit_rows["valuenum"][0] - 37.0) < 0.1
    assert abs(fahrenheit_rows["valuenum"][1] - 38.0) < 0.1
    
    # Check that Celsius values were unchanged
    celsius_rows = result.filter(pl.col("itemid") == 223762)
    assert celsius_rows["valuenum"][0] == 37.0
    assert celsius_rows["valuenum"][1] == 38.0


def test_to_celsius_edge_cases():
    """Test to_celsius with edge cases."""
    df = pl.DataFrame({
        "itemid": [223761, 223761, 223761],
        "valuenum": [32.0, 212.0, 0.0],  # Freezing, boiling, below freezing
        "stay_id": [1, 1, 1],
    })
    
    callback = get_callback("to_celsius")
    result = callback(df, {})
    
    # 32°F = 0°C, 212°F = 100°C, 0°F = -17.78°C
    assert abs(result["valuenum"][0] - 0.0) < 0.01
    assert abs(result["valuenum"][1] - 100.0) < 0.01
    assert abs(result["valuenum"][2] - (-17.78)) < 0.01


def test_register_custom_callback():
    """Test registering a custom callback."""
    @register_callback("test_double")
    def test_double(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
        return df.with_columns([
            (pl.col("valuenum") * 2).alias("valuenum")
        ])
    
    # Verify it was registered
    assert "test_double" in list_callbacks()
    
    # Test using it
    df = pl.DataFrame({
        "itemid": [1, 2],
        "valuenum": [10.0, 20.0],
    })
    
    callback = get_callback("test_double")
    result = callback(df, {})
    
    assert result["valuenum"][0] == 20.0
    assert result["valuenum"][1] == 40.0


def test_callback_preserves_other_columns():
    """Test that callbacks preserve non-transformed columns."""
    df = pl.DataFrame({
        "itemid": [223761, 223762],
        "valuenum": [98.6, 37.0],
        "stay_id": [100, 200],
        "charttime": ["2020-01-01 10:00", "2020-01-01 11:00"],
        "other_col": ["a", "b"],
    })
    
    callback = get_callback("to_celsius")
    result = callback(df, {})
    
    # Check that other columns are preserved
    assert "stay_id" in result.columns
    assert "charttime" in result.columns
    assert "other_col" in result.columns
    assert result["stay_id"][0] == 100
    assert result["stay_id"][1] == 200
    assert result["other_col"][0] == "a"
    assert result["other_col"][1] == "b"


def test_callback_with_metadata():
    """Test that callbacks receive metadata correctly."""
    received_metadata = {}
    
    @register_callback("test_metadata")
    def test_with_metadata(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
        # Store metadata for verification
        received_metadata.update(metadata)
        return df
    
    df = pl.DataFrame({
        "itemid": [1],
        "valuenum": [10.0],
    })
    
    test_metadata = {"units": "mmHg", "source": "chartevents"}
    callback = get_callback("test_metadata")
    callback(df, test_metadata)
    
    assert received_metadata["units"] == "mmHg"
    assert received_metadata["source"] == "chartevents"

