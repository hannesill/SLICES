"""Base classes for label extraction."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import polars as pl


@dataclass
class LabelConfig:
    """Configuration for a downstream prediction task label.

    This defines WHAT to predict, not HOW to predict it.
    Separates label definition from extraction logic and model architecture.
    """

    task_name: str  # Unique identifier (e.g., 'mortality_24h', 'aki_kdigo')
    task_type: str  # 'binary', 'multiclass', 'multilabel', 'regression'

    # Prediction parameters
    prediction_window_hours: Optional[int] = None  # How far ahead to predict (None = in-hospital)
    observation_window_hours: Optional[int] = None  # How much history to use
    gap_hours: int = 0  # Gap between observation and prediction (prevent leakage)

    # Label definition
    label_sources: List[str] = field(default_factory=list)  # Required data sources
    label_params: Dict = field(default_factory=dict)  # Task-specific parameters
    quality_checks: Dict = field(default_factory=dict)  # Optional alert thresholds only

    # Evaluation metrics
    primary_metric: str = "auroc"
    additional_metrics: List[str] = field(default_factory=list)

    # Class information (for classification tasks)
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    positive_class: Optional[str] = None  # For binary tasks

    # Dataset restrictions
    supported_datasets: Optional[List[str]] = None  # None = all datasets allowed


class LabelBuilder(ABC):
    """Abstract base class for building task labels from raw extracted data.

    LabelBuilders implement the logic to convert raw clinical data
    (e.g., mortality flags, creatinine values) into prediction labels.
    """

    SEMANTIC_VERSION: str = "1.0.0"

    @staticmethod
    def config_hash(config: LabelConfig) -> str:
        """Compute a deterministic hash of the label-affecting config fields.

        Returns:
            16-char hex digest of the config's label-relevant fields.
        """
        # NOTE: quality_checks intentionally excluded because they only control
        # warnings/analysis thresholds, not the label semantics themselves.
        supported_datasets = (
            sorted(config.supported_datasets) if config.supported_datasets is not None else None
        )
        hashable = {
            "task_name": config.task_name,
            "task_type": config.task_type,
            "prediction_window_hours": config.prediction_window_hours,
            "observation_window_hours": config.observation_window_hours,
            "gap_hours": config.gap_hours,
            "label_sources": sorted(config.label_sources),
            "label_params": config.label_params,
            "supported_datasets": supported_datasets,
        }
        content = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def __init__(self, config: LabelConfig) -> None:
        """Initialize label builder with configuration.

        Args:
            config: Label configuration specifying label definition.
        """
        self.config = config
        self._last_quality_stats: Dict[str, Any] = {}

    def required_raw_timeseries_horizon_hours(self) -> int:
        """Return the raw timeseries horizon needed to build this task's labels.

        This is independent of the model input sequence length. The extractor uses
        it to validate that the upstream export retains enough post-observation
        data for forward-looking labels.

        Returns:
            Maximum hour offset needed from the raw ``timeseries`` source.
            Returns 0 for tasks that do not depend on raw timeseries labels.
        """
        if "timeseries" not in self.config.label_sources:
            return 0

        return int(self.config.observation_window_hours or 0)

    def build_quality_stats(self, labels: pl.DataFrame) -> Dict[str, Any]:
        """Build serializable task-level quality stats from extracted labels.

        Args:
            labels: Builder output with ``stay_id`` and ``label`` columns.

        Returns:
            Dictionary of quality stats suitable for persistence in metadata.yaml.
        """
        total = len(labels)
        if "label" not in labels.columns:
            stats: Dict[str, Any] = {"total_stays": total}
            self._last_quality_stats = stats
            return stats

        null_count = labels["label"].null_count()
        non_null = total - null_count

        stats = {
            "total_stays": total,
            "non_null_labels": non_null,
            "null_labels": null_count,
            "null_percentage": ((null_count / total) * 100.0) if total > 0 else 0.0,
        }

        if self.config.task_type in {"binary", "binary_classification"}:
            positives = labels.filter(pl.col("label") == 1).height
            negatives = labels.filter(pl.col("label") == 0).height
            stats.update(
                {
                    "positive_labels": positives,
                    "negative_labels": negatives,
                    "positive_prevalence_non_null": (
                        (positives / non_null) if non_null > 0 else None
                    ),
                }
            )

        self._last_quality_stats = stats
        return stats

    @abstractmethod
    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build task labels from raw extracted data.

        Args:
            raw_data: Dictionary mapping source names to DataFrames.
                     Keys correspond to config.label_sources.
                     Each DataFrame has 'stay_id' as identifier.

        Returns:
            DataFrame with columns:
                - stay_id: int64
                - label: target label (int for classification, float for regression)
                - [optional] Additional metadata columns

        Raises:
            ValueError: If required data sources are missing or invalid.
        """
        pass

    def validate_inputs(self, raw_data: Dict[str, pl.DataFrame]) -> None:
        """Validate that all required data sources are present.

        Args:
            raw_data: Dictionary of raw DataFrames.

        Raises:
            ValueError: If required sources are missing.
        """
        missing = set(self.config.label_sources) - set(raw_data.keys())
        if missing:
            raise ValueError(
                f"Task '{self.config.task_name}' requires sources {self.config.label_sources}, "
                f"but missing: {missing}"
            )

    def _merge_with_stays(
        self,
        stays: pl.DataFrame,
        labels: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge labels with stay metadata, ensuring all stays are included.

        Args:
            stays: DataFrame with stay_id and temporal information.
            labels: DataFrame with stay_id and computed labels.

        Returns:
            DataFrame with all stays, missing labels set to null.
        """
        return stays.select("stay_id").join(labels, on="stay_id", how="left")
