"""Factory for creating label builders from configurations."""

from typing import Dict, Type

from .aki import AKILabelBuilder
from .base import LabelBuilder, LabelConfig
from .los import LOSLabelBuilder
from .mortality import MortalityLabelBuilder


class LabelBuilderFactory:
    """Factory for creating LabelBuilder instances from configurations.

    Usage:
        config = LabelConfig(task_name='mortality_24h', ...)
        builder = LabelBuilderFactory.create(config)
        labels = builder.build_labels(raw_data)
    """

    # Registry mapping task categories to builder classes
    _REGISTRY: Dict[str, Type[LabelBuilder]] = {
        "aki": AKILabelBuilder,
        "los": LOSLabelBuilder,
        "mortality": MortalityLabelBuilder,
    }

    @classmethod
    def create(cls, config: LabelConfig) -> LabelBuilder:
        """Create a LabelBuilder instance from configuration.

        Args:
            config: Label configuration.

        Returns:
            Instantiated LabelBuilder for the specified task.

        Raises:
            ValueError: If task name doesn't match any registered builder.
        """
        # Extract task category from task_name (e.g., 'mortality_24h' -> 'mortality')
        task_category = cls._extract_category(config.task_name)

        if task_category not in cls._REGISTRY:
            raise ValueError(
                f"No LabelBuilder registered for task category '{task_category}'. "
                f"Available categories: {list(cls._REGISTRY.keys())}"
            )

        builder_class = cls._REGISTRY[task_category]
        return builder_class(config)

    @classmethod
    def register(cls, category: str, builder_class: Type[LabelBuilder]) -> None:
        """Register a new LabelBuilder class for a task category.

        Args:
            category: Task category name (e.g., 'aki', 'sepsis').
            builder_class: LabelBuilder subclass to register.
        """
        cls._REGISTRY[category] = builder_class

    @staticmethod
    def _extract_category(task_name: str) -> str:
        """Extract task category from full task name.

        Examples:
            'mortality_24h' -> 'mortality'
            'mortality_hospital' -> 'mortality'

        Args:
            task_name: Full task name.

        Returns:
            Task category (first part before underscore).
        """
        return task_name.split("_")[0]

    @classmethod
    def list_available(cls) -> Dict[str, Type[LabelBuilder]]:
        """List all registered label builders.

        Returns:
            Dictionary mapping categories to builder classes.
        """
        return cls._REGISTRY.copy()
