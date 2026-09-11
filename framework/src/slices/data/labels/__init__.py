"""Label extraction and builders.

This module handles label extraction for downstream tasks (mortality, AKI, LOS, etc.).
Separate from src/slices/models/heads/ which contains model prediction heads for finetuning.
"""

from .aki import AKILabelBuilder
from .base import LabelBuilder, LabelConfig
from .factory import LabelBuilderFactory
from .los import LOSLabelBuilder
from .mortality import MortalityLabelBuilder

__all__ = [
    "AKILabelBuilder",
    "LabelConfig",
    "LabelBuilder",
    "LabelBuilderFactory",
    "LOSLabelBuilder",
    "MortalityLabelBuilder",
]
