"""Shared metadata contract for post-run fairness summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FAIRNESS_SUMMARY_SCHEMA_VERSION = "2026-04-23.v2"
FAIRNESS_SCRIPT_VERSION = "evaluate_fairness.py:2026-04-23.v2"
FAIRNESS_DEFAULT_PROTECTED_ATTRIBUTES = ["gender", "age_group", "race"]
FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE = 50

EVAL_ARTIFACT_PATH_KEY = "_eval_artifact_path"
EVAL_ARTIFACT_SHA256_KEY = "_eval_artifact_sha256"
FAIRNESS_SCHEMA_VERSION_KEY = "_fairness_summary_schema_version"
FAIRNESS_SCRIPT_VERSION_KEY = "_fairness_script_version"
FAIRNESS_ARTIFACT_PATH_KEY = "_fairness_artifact_path"
FAIRNESS_ARTIFACT_SHA256_KEY = "_fairness_artifact_sha256"
FAIRNESS_ARTIFACT_SOURCE_KEY = "_fairness_artifact_source"
FAIRNESS_CHECKPOINT_SOURCE_KEY = "_fairness_checkpoint_source"
FAIRNESS_PROTECTED_ATTRIBUTES_KEY = "_fairness_protected_attributes"
FAIRNESS_MIN_SUBGROUP_SIZE_KEY = "_fairness_min_subgroup_size"

FAIRNESS_METADATA_COLUMNS = [
    FAIRNESS_SCHEMA_VERSION_KEY,
    FAIRNESS_SCRIPT_VERSION_KEY,
    FAIRNESS_ARTIFACT_PATH_KEY,
    FAIRNESS_ARTIFACT_SHA256_KEY,
    FAIRNESS_ARTIFACT_SOURCE_KEY,
    FAIRNESS_CHECKPOINT_SOURCE_KEY,
    FAIRNESS_PROTECTED_ATTRIBUTES_KEY,
    FAIRNESS_MIN_SUBGROUP_SIZE_KEY,
]

FAIRNESS_CLEAR_PREFIXES = ("fairness/", "_fairness_")


def normalize_protected_attributes(protected_attributes: list[str]) -> list[str]:
    """Return de-duplicated protected attributes while preserving order."""
    return list(dict.fromkeys(str(attr) for attr in protected_attributes))


def encode_protected_attributes(protected_attributes: list[str]) -> str:
    """Encode protected attributes into a stable W&B summary scalar."""
    return json.dumps(normalize_protected_attributes(protected_attributes))


def decode_protected_attributes(value: Any) -> list[str] | None:
    """Decode protected-attribute metadata from W&B/export representations."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        parsed = list(value)
    else:
        return None
    return normalize_protected_attributes([str(attr) for attr in parsed])


def canonical_artifact_id(path_value: Any) -> str:
    """Return a path identifier stable across output-root rebases."""
    if path_value is None:
        return ""

    text = str(path_value).strip().replace("\\", "/")
    if not text:
        return ""

    marker = "/outputs/"
    if marker in text:
        return "outputs/" + text.split(marker, 1)[1].strip("/")
    if text.startswith("outputs/"):
        return text.strip("/")
    index = text.find("outputs/")
    if index >= 0:
        return text[index:].strip("/")
    return text.strip("/")


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 digest for a local artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
