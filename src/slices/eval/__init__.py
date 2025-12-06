"""Evaluation module for metrics and analysis.

This module provides:
- Configurable metric collections for different task types
- Factory functions for building metrics from config
- (Future) Fairness and subgroup analysis utilities
"""

from slices.eval.metrics import (
    MetricConfig,
    build_metrics,
    get_default_metrics,
)

__all__ = [
    "MetricConfig",
    "build_metrics",
    "get_default_metrics",
]
