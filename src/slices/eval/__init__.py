"""Evaluation module for metrics, fairness, and imputation analysis.

This module provides:
- Configurable metric collections for different task types
- Factory functions for building metrics from config
- Fairness evaluation across protected demographic attributes
- Imputation evaluation for SSL encoder quality assessment
"""

from slices.eval.fairness_evaluator import FairnessEvaluator
from slices.eval.imputation import ImputationEvaluator
from slices.eval.metrics import (
    MetricConfig,
    build_metrics,
    get_default_metrics,
)

__all__ = [
    "MetricConfig",
    "build_metrics",
    "get_default_metrics",
    "FairnessEvaluator",
    "ImputationEvaluator",
]
