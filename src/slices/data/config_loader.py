"""Configuration loading utilities for concept extraction.

This module provides functions to load and validate dataset and concept
configurations from YAML files using Pydantic schemas.

Functions:
    load_dataset_config: Load dataset-level config (time handling, IDs, tables)
    load_timeseries_concepts: Load time-series concept configs from category files
    load_static_concepts: Load static/demographic concept configs
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config_schemas import (
    ColumnSource,
    DatasetConfig,
    ExtractionSource,
    ItemIDSource,
    RegexMatchSource,
    StaticConceptConfig,
    StaticExtractionSource,
    StringMatchSource,
    TimeSeriesConceptConfig,
)

# Time-series concept files to load (order matters for consistent feature ordering)
TIMESERIES_FILES = [
    "vitals.yaml",
    "assessments.yaml",
    "labs.yaml",
    "outputs.yaml",
    "medications.yaml",
]


def load_dataset_config(datasets_dir: Path, dataset_name: str) -> DatasetConfig:
    """Load dataset-level configuration from YAML.

    Args:
        datasets_dir: Path to configs/datasets directory.
        dataset_name: Dataset name (e.g., "mimic_iv").

    Returns:
        Validated DatasetConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config fails Pydantic validation.
    """
    config_path = datasets_dir / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return DatasetConfig(**data)


def _parse_extraction_source(source_dict: Dict) -> ExtractionSource:
    """Parse a source dictionary into the appropriate ExtractionSource type.

    Args:
        source_dict: Dictionary with source configuration.

    Returns:
        Typed ExtractionSource (ItemIDSource, ColumnSource, etc.)

    Raises:
        ValueError: If source type is unknown.
    """
    source_type = source_dict.get("type", "itemid")  # Default to itemid for backwards compat

    if source_type == "itemid":
        return ItemIDSource(**source_dict)
    elif source_type == "column":
        return ColumnSource(**source_dict)
    elif source_type == "string":
        return StringMatchSource(**source_dict)
    elif source_type == "regex":
        return RegexMatchSource(**source_dict)
    else:
        raise ValueError(f"Unknown extraction source type: {source_type}")


def _parse_timeseries_concept(
    config_dict: Dict,
    dataset_name: str,
) -> Optional[TimeSeriesConceptConfig]:
    """Parse a time-series concept configuration.

    Args:
        config_dict: Raw configuration dictionary.
        dataset_name: Dataset to check for (e.g., "mimic_iv").

    Returns:
        Validated TimeSeriesConceptConfig or None if dataset not supported.
    """
    # Check if dataset is defined for this concept
    dataset_sources = config_dict.get(dataset_name)
    if dataset_sources is None:
        return None

    # Parse dataset-specific sources
    if isinstance(dataset_sources, list):
        parsed_sources = [_parse_extraction_source(s) for s in dataset_sources]
    else:
        # Single source (shouldn't happen in new format, but handle gracefully)
        parsed_sources = [_parse_extraction_source(dataset_sources)]

    # Build config dict with parsed sources
    parsed_config = {
        k: v
        for k, v in config_dict.items()
        if k not in ("mimic_iv", "eicu")  # Remove raw dataset configs
    }
    parsed_config[dataset_name] = parsed_sources

    return TimeSeriesConceptConfig(**parsed_config)


def load_timeseries_concepts(
    concepts_dir: Path,
    dataset_name: str,
    feature_set: str = "core",
    categories: Optional[List[str]] = None,
) -> Dict[str, TimeSeriesConceptConfig]:
    """Load time-series concept configs filtered by feature set.

    Args:
        concepts_dir: Path to configs/concepts directory.
        dataset_name: Dataset to filter for (e.g., "mimic_iv").
        feature_set: Feature set to filter ("core" or "extended").
        categories: Optional list of categories to load (default: all).

    Returns:
        Dictionary mapping concept name to validated config.

    Raises:
        ValueError: If duplicate concept names are found.
    """
    concepts: Dict[str, TimeSeriesConceptConfig] = {}

    files_to_load = TIMESERIES_FILES
    if categories:
        files_to_load = [f"{cat}.yaml" for cat in categories]

    for filename in files_to_load:
        yaml_file = concepts_dir / filename
        if not yaml_file.exists():
            continue

        with open(yaml_file) as f:
            file_concepts = yaml.safe_load(f) or {}

        for name, config_dict in file_concepts.items():
            if name in concepts:
                raise ValueError(f"Duplicate concept '{name}' in {yaml_file}")

            # Parse and validate
            config = _parse_timeseries_concept(config_dict, dataset_name)

            if config is None:
                # Dataset not supported for this concept
                continue

            # Filter by feature set
            if feature_set not in config.feature_set:
                continue

            concepts[name] = config

    return concepts


def _parse_static_source(source_dict: Dict) -> StaticExtractionSource:
    """Parse a static extraction source dictionary.

    Args:
        source_dict: Dictionary with source configuration.

    Returns:
        Validated StaticExtractionSource.
    """
    return StaticExtractionSource(**source_dict)


def load_static_concepts(
    concepts_dir: Path,
    dataset_name: str,
) -> Dict[str, StaticConceptConfig]:
    """Load static/demographic concept configs.

    Args:
        concepts_dir: Path to configs/concepts directory.
        dataset_name: Dataset to filter for.

    Returns:
        Dictionary mapping concept name to validated config.
    """
    static_file = concepts_dir / "static.yaml"
    if not static_file.exists():
        return {}

    with open(static_file) as f:
        file_concepts = yaml.safe_load(f) or {}

    concepts: Dict[str, StaticConceptConfig] = {}

    for name, config_dict in file_concepts.items():
        # Check if dataset is defined
        dataset_source = config_dict.get(dataset_name)
        if dataset_source is None:
            continue

        # Parse dataset-specific source (single source, not list)
        parsed_config = {k: v for k, v in config_dict.items() if k not in ("mimic_iv", "eicu")}

        if dataset_source is not None:
            parsed_config[dataset_name] = _parse_static_source(dataset_source)

        config = StaticConceptConfig(**parsed_config)
        concepts[name] = config

    return concepts


def get_feature_names(
    concepts_dir: Path,
    dataset_name: str,
    feature_set: str = "core",
) -> List[str]:
    """Get ordered list of feature names for a dataset.

    Args:
        concepts_dir: Path to configs/concepts directory.
        dataset_name: Dataset to filter for.
        feature_set: Feature set to filter.

    Returns:
        Ordered list of feature names.
    """
    concepts = load_timeseries_concepts(concepts_dir, dataset_name, feature_set)
    return list(concepts.keys())


def get_static_feature_names(
    concepts_dir: Path,
    dataset_name: str,
) -> List[str]:
    """Get ordered list of static feature names for a dataset.

    Args:
        concepts_dir: Path to configs/concepts directory.
        dataset_name: Dataset to filter for.

    Returns:
        Ordered list of static feature names.
    """
    concepts = load_static_concepts(concepts_dir, dataset_name)
    return list(concepts.keys())
