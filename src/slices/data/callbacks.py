"""Callback functions for transforming extracted feature values.

This module provides a registry of transformation callbacks that can be applied
to features during extraction. Inspired by ricu's callback system.

Callbacks are specified in concept YAML files using the 'transform' field:
    temperature:
      mimic_iv:
        source: chartevents
        itemid: [223761, 223762]  # Fahrenheit, Celsius
        value_col: valuenum
        transform: to_celsius  # Apply Fahrenheit -> Celsius conversion
      units: celsius

Each callback takes a Polars DataFrame with a 'valuenum' column and returns
the transformed DataFrame. Callbacks can also access metadata like itemid
for conditional transformations.
"""

from typing import Callable, Dict

import polars as pl


# Type alias for callback functions
CallbackFunction = Callable[[pl.DataFrame, Dict], pl.DataFrame]


# Registry of available callbacks
_CALLBACK_REGISTRY: Dict[str, CallbackFunction] = {}


def register_callback(name: str) -> Callable[[CallbackFunction], CallbackFunction]:
    """Decorator to register a callback function.
    
    Usage:
        @register_callback("my_transform")
        def my_transform(df: pl.DataFrame, metadata: Dict) -> pl.DataFrame:
            # Transform valuenum column
            return df.with_columns([
                (pl.col("valuenum") * 2).alias("valuenum")
            ])
    
    Args:
        name: Name of the callback (used in YAML configs).
        
    Returns:
        Decorator function.
    """
    def decorator(func: CallbackFunction) -> CallbackFunction:
        _CALLBACK_REGISTRY[name] = func
        return func
    return decorator


def get_callback(name: str) -> CallbackFunction:
    """Get a callback function by name.
    
    Args:
        name: Name of the callback.
        
    Returns:
        Callback function.
        
    Raises:
        ValueError: If callback name is not registered.
    """
    if name not in _CALLBACK_REGISTRY:
        available = list(_CALLBACK_REGISTRY.keys())
        raise ValueError(
            f"Unknown callback '{name}'. Available callbacks: {available}\n"
            f"Hint: Register new callbacks using @register_callback decorator"
        )
    return _CALLBACK_REGISTRY[name]


def list_callbacks() -> list[str]:
    """List all registered callback names.
    
    Returns:
        List of callback names.
    """
    return list(_CALLBACK_REGISTRY.keys())


# =============================================================================
# Built-in Callbacks
# =============================================================================

@register_callback("to_celsius")
def to_celsius(df: pl.DataFrame, metadata: Dict) -> pl.DataFrame:
    """Convert temperature from Fahrenheit to Celsius based on itemid.
    
    MIMIC-IV temperature itemids:
        - 223761: Temperature Fahrenheit (needs conversion)
        - 223762: Temperature Celsius (no conversion)
    
    This callback checks the itemid for each row and converts only
    Fahrenheit values: C = (F - 32) * 5/9
    
    Args:
        df: DataFrame with columns: itemid, valuenum, ...
        metadata: Dict with feature config (unused here but available).
        
    Returns:
        DataFrame with valuenum converted to Celsius where applicable.
        
    Example:
        Input:
            itemid    valuenum
            223761    98.6      (Fahrenheit)
            223762    37.0      (Celsius)
            223761    100.4     (Fahrenheit)
            
        Output:
            itemid    valuenum
            223761    37.0      (converted from 98.6°F)
            223762    37.0      (unchanged)
            223761    38.0      (converted from 100.4°F)
    """
    # Fahrenheit itemid that needs conversion
    FAHRENHEIT_ITEMID = 223761
    
    return df.with_columns([
        pl.when(pl.col("itemid") == FAHRENHEIT_ITEMID)
          .then((pl.col("valuenum") - 32) * 5 / 9)
          .otherwise(pl.col("valuenum"))
          .alias("valuenum")
    ])


# Future callbacks can be added here using @register_callback decorator
# Examples:
#
# @register_callback("convert_units")
# def convert_units(df: pl.DataFrame, metadata: Dict) -> pl.DataFrame:
#     """Convert between different unit systems."""
#     # Implementation here
#     pass
#
# @register_callback("normalize_range")
# def normalize_range(df: pl.DataFrame, metadata: Dict) -> pl.DataFrame:
#     """Normalize values to [0, 1] range."""
#     # Implementation here
#     pass
