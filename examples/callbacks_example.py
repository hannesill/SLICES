"""Example demonstrating the callback system for feature transformations.

This example shows:
1. How callbacks are used in concept YAML files
2. How to use the built-in to_celsius callback
3. How to register custom callbacks for your own transformations
"""

import polars as pl
from slices.data.callbacks import get_callback, list_callbacks, register_callback


def example_list_callbacks():
    """List all available callbacks."""
    print("Available callbacks:")
    for callback_name in list_callbacks():
        print(f"  - {callback_name}")


def example_use_to_celsius():
    """Example using the built-in to_celsius callback."""
    print("\n=== Example: to_celsius callback ===")

    # Sample data with mixed Fahrenheit and Celsius temperatures
    df = pl.DataFrame(
        {
            "itemid": [223761, 223762, 223761],  # F, C, F
            "valuenum": [98.6, 37.0, 100.4],  # Values in original units
            "stay_id": [1, 1, 2],
            "charttime": ["2020-01-01 10:00", "2020-01-01 10:05", "2020-01-01 11:00"],
        }
    )

    print("Input data:")
    print(df)

    # Get and apply the callback
    to_celsius = get_callback("to_celsius")
    result = to_celsius(df, {})

    print("\nAfter to_celsius callback:")
    print(result)
    print("\nNote: itemid 223761 (Fahrenheit) values were converted to Celsius")
    print("      itemid 223762 (Celsius) values were unchanged")


def example_register_custom_callback():
    """Example of registering a custom callback."""
    print("\n=== Example: Custom callback ===")

    # Define a custom callback to convert mg/dL to mmol/L (glucose)
    @register_callback("glucose_to_mmol")
    def glucose_to_mmol(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
        """Convert glucose from mg/dL to mmol/L.

        Conversion: mmol/L = mg/dL / 18.0182
        """
        return df.with_columns([(pl.col("valuenum") / 18.0182).alias("valuenum")])

    # Test data
    df = pl.DataFrame(
        {
            "itemid": [50931, 50931],
            "valuenum": [90.0, 180.0],  # mg/dL
            "stay_id": [1, 2],
        }
    )

    print("Input (glucose in mg/dL):")
    print(df)

    # Apply the callback
    callback = get_callback("glucose_to_mmol")
    result = callback(df, {})

    print("\nAfter glucose_to_mmol callback:")
    print(result)
    print("\nNote: 90 mg/dL → 5.0 mmol/L, 180 mg/dL → 10.0 mmol/L")


def example_conditional_callback():
    """Example of a callback that uses metadata for conditional transforms."""
    print("\n=== Example: Conditional callback using metadata ===")

    @register_callback("normalize_by_itemid")
    def normalize_by_itemid(df: pl.DataFrame, metadata: dict) -> pl.DataFrame:
        """Normalize values based on itemid-specific ranges from metadata.

        This shows how callbacks can use the metadata dict to access
        feature configuration (e.g., min/max values, units, etc.).
        """
        # Get normalization ranges from metadata
        itemid_ranges = metadata.get("ranges", {})

        # Apply normalization conditionally
        result = df.clone()
        for itemid, (min_val, max_val) in itemid_ranges.items():
            mask = pl.col("itemid") == itemid
            result = result.with_columns(
                [
                    pl.when(mask)
                    .then((pl.col("valuenum") - min_val) / (max_val - min_val))
                    .otherwise(pl.col("valuenum"))
                    .alias("valuenum")
                ]
            )

        return result

    # Test data
    df = pl.DataFrame(
        {
            "itemid": [220045, 220210],  # Heart rate, Respiratory rate
            "valuenum": [75.0, 16.0],
            "stay_id": [1, 1],
        }
    )

    print("Input:")
    print(df)

    # Metadata with normalization ranges
    metadata = {
        "ranges": {
            220045: (40, 180),  # HR: 40-180 bpm
            220210: (5, 40),  # RR: 5-40 breaths/min
        }
    }

    callback = get_callback("normalize_by_itemid")
    result = callback(df, metadata)

    print("\nAfter normalization:")
    print(result)
    print("\nNote: HR 75 → 0.25 in [40,180], RR 16 → 0.314 in [5,40]")


def example_yaml_usage():
    """Show how callbacks are used in concept YAML files."""
    print("\n=== Example: Using callbacks in YAML configs ===")

    yaml_example = """
# In configs/concepts/core_features.yaml:

vitals:
  temperature:
    mimic_iv:
      source: chartevents
      itemid: [223761, 223762]  # Fahrenheit, Celsius
      value_col: valuenum
      transform: to_celsius  # <-- Apply callback here
    units: celsius
    min: 32
    max: 42

# Future example with custom callback:
labs:
  glucose:
    mimic_iv:
      source: labevents
      itemid: [50931, 50809]
      value_col: valuenum
      transform: glucose_to_mmol  # Custom callback
    units: mmol/L
    min: 2
    max: 30
"""

    print(yaml_example)
    print("\nThe extractor automatically:")
    print("1. Loads the concept YAML")
    print("2. Extracts raw data from the database")
    print("3. Applies the specified callback transformation")
    print("4. Returns transformed values in standardized units")


def main():
    """Run all examples."""
    print("=" * 70)
    print("SLICES Callback System Examples")
    print("=" * 70)

    example_list_callbacks()
    example_use_to_celsius()
    example_register_custom_callback()
    example_conditional_callback()
    example_yaml_usage()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("1. Use @register_callback('name') to add new transformations")
    print("2. Specify 'transform: callback_name' in concept YAML files")
    print("3. Callbacks receive DataFrame and metadata dict")
    print("4. Callbacks must return transformed DataFrame")
    print("5. Access itemid, valuenum, and other columns in your callback")
    print("\nFor more details, see:")
    print("  - src/slices/data/callbacks.py (implementation)")
    print("  - tests/test_callbacks.py (comprehensive tests)")
    print("  - configs/concepts/core_features.yaml (usage in configs)")


if __name__ == "__main__":
    main()
