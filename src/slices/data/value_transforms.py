"""Value transformation functions for concept extraction.

This module provides a registry of transformation functions that can be applied
during feature extraction to convert units, parse text values, etc.

Transforms are specified in concept YAML files using the 'transform' field:
    temperature:
      mimic_iv:
        - table: chartevents
          type: itemid
          itemid: [223761]
          value_col: valuenum
          time_col: charttime
          transform: fahrenheit_to_celsius

Two transform signatures are supported:
1. Simple: (pl.Series) -> pl.Series - for element-wise value transformations
2. DataFrame: (pl.DataFrame, Dict) -> pl.DataFrame - for complex transforms needing context
"""

from typing import Callable, Dict, Union

import polars as pl

# Type aliases
SimpleTransform = Callable[[pl.Series], pl.Series]
DataFrameTransform = Callable[[pl.DataFrame, Dict], pl.DataFrame]
TransformFunction = Union[SimpleTransform, DataFrameTransform]


# Registry
_TRANSFORM_REGISTRY: Dict[str, TransformFunction] = {}


def register_transform(name: str) -> Callable[[TransformFunction], TransformFunction]:
    """Decorator to register a transform function.

    Usage:
        @register_transform("my_transform")
        def my_transform(values: pl.Series) -> pl.Series:
            return values * 2

    Args:
        name: Name of the transform (used in YAML configs).

    Returns:
        Decorator function.
    """

    def decorator(func: TransformFunction) -> TransformFunction:
        _TRANSFORM_REGISTRY[name] = func
        return func

    return decorator


def get_transform(name: str) -> TransformFunction:
    """Get a transform function by name.

    Args:
        name: Name of the transform.

    Returns:
        Transform function.

    Raises:
        ValueError: If transform name is not registered.
    """
    if name not in _TRANSFORM_REGISTRY:
        available = list(_TRANSFORM_REGISTRY.keys())
        raise ValueError(
            f"Unknown transform '{name}'. Available transforms: {available}\n"
            f"Hint: Register new transforms using @register_transform decorator"
        )
    return _TRANSFORM_REGISTRY[name]


def list_transforms() -> list[str]:
    """List all registered transform names.

    Returns:
        List of transform names.
    """
    return list(_TRANSFORM_REGISTRY.keys())


def apply_transform(
    name: str,
    data: Union[pl.Series, pl.DataFrame],
    metadata: Dict = None,
) -> Union[pl.Series, pl.DataFrame]:
    """Apply a named transform to data.

    Automatically detects whether to use simple or DataFrame signature.

    Args:
        name: Transform name.
        data: Series or DataFrame to transform.
        metadata: Optional metadata dict (for DataFrame transforms).

    Returns:
        Transformed data (same type as input).
    """
    transform = get_transform(name)
    metadata = metadata or {}

    if isinstance(data, pl.Series):
        # Try simple signature first
        try:
            return transform(data)
        except TypeError:
            # Fall back to DataFrame signature
            df = pl.DataFrame({"value": data})
            result = transform(df, metadata)
            return result["value"]
    else:
        # DataFrame - try DataFrame signature first
        try:
            return transform(data, metadata)
        except TypeError:
            # Simple transform on valuenum column
            return data.with_columns(transform(pl.col("valuenum")).alias("valuenum"))


# =============================================================================
# Unit Conversion Transforms
# =============================================================================


@register_transform("fahrenheit_to_celsius")
def fahrenheit_to_celsius(values: pl.Series) -> pl.Series:
    """Convert Fahrenheit to Celsius: C = (F - 32) * 5/9"""
    return (values - 32) * 5 / 9


@register_transform("mg_dl_to_mmol_l_glucose")
def mg_dl_to_mmol_l_glucose(values: pl.Series) -> pl.Series:
    """Convert glucose from mg/dL to mmol/L."""
    return values / 18.0


@register_transform("minutes_to_days")
def minutes_to_days(values: pl.Series) -> pl.Series:
    """Convert minutes to days."""
    return values / 1440.0


@register_transform("minutes_to_hours")
def minutes_to_hours(values: pl.Series) -> pl.Series:
    """Convert minutes to hours."""
    return values / 60.0


@register_transform("percent_to_fraction")
def percent_to_fraction(values: pl.Series) -> pl.Series:
    """Convert percentage (0-100) to fraction (0-1).

    Used for FiO2 which is stored as percentage in MIMIC-IV (e.g., 21-100)
    but clinically expressed as fraction (e.g., 0.21-1.0).
    """
    return values / 100.0


# =============================================================================
# DataFrame-based Transforms (for context-dependent transformations)
# =============================================================================


@register_transform("to_celsius")
def to_celsius(df: pl.DataFrame, metadata: Dict) -> pl.DataFrame:  # noqa: ARG001
    """Convert temperature from Fahrenheit to Celsius based on itemid.

    This is a backwards-compatible transform that checks itemid to determine
    which rows need conversion.

    MIMIC-IV temperature itemids:
        - 223761: Temperature Fahrenheit (needs conversion)
        - 223762: Temperature Celsius (no conversion)

    Args:
        df: DataFrame with columns: itemid, valuenum, ...
        metadata: Dict with feature config (unused here but available).

    Returns:
        DataFrame with valuenum converted to Celsius where applicable.
    """
    FAHRENHEIT_ITEMID = 223761

    return df.with_columns(
        pl.when(pl.col("itemid") == FAHRENHEIT_ITEMID)
        .then((pl.col("valuenum") - 32) * 5 / 9)
        .otherwise(pl.col("valuenum"))
        .alias("valuenum")
    )


# =============================================================================
# Text to Numeric Transforms
# =============================================================================


@register_transform("gcs_eye_text_to_numeric")
def gcs_eye_text_to_numeric(values: pl.Series) -> pl.Series:
    """Convert GCS eye opening text to numeric score."""
    mapping = {"Spontaneous": 4, "To Voice": 3, "To Pain": 2, "None": 1}
    return values.map_elements(lambda x: mapping.get(str(x).strip(), None), return_dtype=pl.Float64)


@register_transform("eicu_age_parse")
def eicu_age_parse(values: pl.Series) -> pl.Series:
    """Parse eICU age (handles '>89' as 90)."""

    def parse_age(x):
        if x is None:
            return None
        s = str(x)
        if s.startswith(">"):
            return 90.0
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    return values.map_elements(parse_age, return_dtype=pl.Float64)
