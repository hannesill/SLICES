"""Base classes for label extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import polars as pl


@dataclass
class LabelConfig:
    """Configuration for a downstream prediction task label.

    This defines WHAT to predict, not HOW to predict it.
    Separates label definition from extraction logic and model architecture.
    """

    task_name: str  # Unique identifier (e.g., 'mortality_24h', 'aki_kdigo')
    task_type: str  # 'binary_classification', 'multiclass_classification', 'regression'

    # Prediction parameters
    prediction_window_hours: Optional[int] = None  # How far ahead to predict (None = in-hospital)
    observation_window_hours: Optional[int] = None  # How much history to use
    gap_hours: int = 0  # Gap between observation and prediction (prevent leakage)

    # Label definition
    label_sources: List[str] = field(default_factory=list)  # Required data sources
    label_params: Dict = field(default_factory=dict)  # Task-specific parameters

    # Evaluation metrics
    primary_metric: str = "auroc"
    additional_metrics: List[str] = field(default_factory=list)

    # Class information (for classification tasks)
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    positive_class: Optional[str] = None  # For binary tasks


class LabelBuilder(ABC):
    """Abstract base class for building task labels from raw extracted data.

    LabelBuilders implement the logic to convert raw clinical data
    (e.g., mortality flags, creatinine values) into prediction labels.
    """

    def __init__(self, config: LabelConfig) -> None:
        """Initialize label builder with configuration.

        Args:
            config: Label configuration specifying label definition.
        """
        self.config = config

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
