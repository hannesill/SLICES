"""Label extraction and builders.

This module handles label extraction for downstream tasks (mortality, AKI, etc.).
Separate from src/slices/models/heads/ which contains model prediction heads for finetuning.
"""

from .base import LabelBuilder, LabelConfig
from .factory import LabelBuilderFactory

__all__ = ["LabelConfig", "LabelBuilder", "LabelBuilderFactory"]
