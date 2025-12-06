"""Label extraction and builders.

This module handles label extraction for downstream tasks (mortality, AKI, etc.).
Separate from src/slices/tasks/ which contains model prediction heads.
"""

from .base import LabelConfig, LabelBuilder
from .factory import LabelBuilderFactory

__all__ = ["LabelConfig", "LabelBuilder", "LabelBuilderFactory"]
