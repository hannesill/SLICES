"""Label extraction and builders.

This module handles label extraction for downstream tasks (mortality, phenotyping, etc.).
Separate from src/slices/models/heads/ which contains model prediction heads for finetuning.
"""

from .aki import AKILabelBuilder
from .base import LabelBuilder, LabelConfig
from .decompensation import DeathHoursLabelBuilder
from .factory import LabelBuilderFactory
from .los import LOSLabelBuilder
from .mortality import MortalityLabelBuilder
from .phenotyping import PhenotypingLabelBuilder

__all__ = [
    "AKILabelBuilder",
    "DeathHoursLabelBuilder",
    "LabelConfig",
    "LabelBuilder",
    "LabelBuilderFactory",
    "LOSLabelBuilder",
    "MortalityLabelBuilder",
    "PhenotypingLabelBuilder",
]
