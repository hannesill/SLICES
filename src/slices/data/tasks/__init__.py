"""Task label extraction and builders.

This module handles label extraction for downstream tasks (mortality, AKI, etc.).
Separate from src/slices/tasks/ which contains model prediction heads.
"""

from .base import TaskConfig, TaskBuilder
from .factory import TaskBuilderFactory

__all__ = ["TaskConfig", "TaskBuilder", "TaskBuilderFactory"]
