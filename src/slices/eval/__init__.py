"""Evaluation module for metrics, fairness, and imputation analysis.

This module provides:
- Configurable metric collections for different task types
- Factory functions for building metrics from config
- Fairness evaluation across protected demographic attributes
- Imputation evaluation for SSL encoder quality assessment
"""

from slices.eval.fairness_evaluator import FairnessEvaluator
from slices.eval.imputation import ImputationEvaluator
from slices.eval.inference import run_inference
from slices.eval.metrics import (
    MetricConfig,
    build_metrics,
    get_default_metrics,
)
from slices.eval.statistical import (
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    paired_bootstrap_test,
    paired_wilcoxon_signed_rank,
)

__all__ = [
    "MetricConfig",
    "build_metrics",
    "get_default_metrics",
    "FairnessEvaluator",
    "ImputationEvaluator",
    "bonferroni_correction",
    "bootstrap_ci",
    "cohens_d",
    "paired_bootstrap_test",
    "paired_wilcoxon_signed_rank",
    "run_inference",
]
