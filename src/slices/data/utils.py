"""Shared data utilities."""

from pathlib import Path


def get_package_data_dir() -> Path:
    """Get the package data directory containing benchmark YAML configs.

    Returns:
        Path to ``src/slices/data/`` (the directory this module lives in).
    """
    return Path(__file__).parent
