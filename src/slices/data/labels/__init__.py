"""Label extraction and builders.

This module handles label extraction for downstream tasks (mortality, phenotyping, etc.).
Separate from src/slices/models/heads/ which contains model prediction heads for finetuning.
"""

from .base import LabelBuilder, LabelConfig
from .factory import LabelBuilderFactory
from .phenotyping import PhenotypingLabelBuilder

__all__ = ["LabelConfig", "LabelBuilder", "LabelBuilderFactory", "PhenotypingLabelBuilder"]
