"""Pydantic models for data configuration.

This module defines the schema for dataset and concept configurations used
by the extraction pipeline, as well as data loading configurations used by
training scripts.

Schemas:
    DataConfig: Training data loading configuration (paths, preprocessing, splits)
    DatasetConfig: Dataset-level metadata (time handling, IDs, tables)
    TimeSeriesConceptConfig: Per-concept extraction rules for time-series features
    StaticConceptConfig: Per-concept extraction rules for static/demographic features
"""

from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from slices.constants import (
    EXTRACTION_BATCH_SIZE,
    IMPUTE_STRATEGY,
    MIN_STAY_HOURS,
    NORMALIZE,
    PIN_MEMORY,
    SEQ_LENGTH_HOURS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


class ExtractionType(str, Enum):
    """Extraction method type for time-series concepts."""

    ITEMID = "itemid"  # Numeric ID matching (MIMIC chartevents/labevents)
    COLUMN = "column"  # Direct column read (eICU vitalperiodic)
    STRING = "string"  # Exact string match (eICU lab table)
    REGEX = "regex"  # Pattern matching (medications)


class AggregationType(str, Enum):
    """Aggregation function for hourly binning."""

    MEAN = "mean"  # Average (default for most vitals/labs)
    SUM = "sum"  # Sum (urine output, fluid intake)
    MAX = "max"  # Maximum value (peak values)
    MIN = "min"  # Minimum value (trough values)
    LAST = "last"  # Last value in bin (GCS, RASS assessments)
    FIRST = "first"  # First value in bin (admission labs)
    ANY = "any"  # True if any present (boolean medication flags)


class ValueType(str, Enum):
    """Value type for features."""

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


# =============================================================================
# Dataset Configuration
# =============================================================================


class TimeConfig(BaseModel):
    """Time handling configuration for a dataset."""

    format: Literal["timestamp", "offset_minutes", "offset_hours"]
    reference_table: Optional[str] = None  # e.g., "icu/icustays"
    reference_col: Optional[str] = None  # e.g., "intime"

    @model_validator(mode="after")
    def validate_timestamp_requires_reference(self) -> "TimeConfig":
        """Timestamp format requires reference table and column."""
        if self.format == "timestamp":
            if not self.reference_table or not self.reference_col:
                raise ValueError("Timestamp format requires reference_table and reference_col")
        return self


class IDConfig(BaseModel):
    """ID column names for a dataset."""

    stay: str  # e.g., "stay_id" for MIMIC, "patientunitstayid" for eICU
    patient: str  # e.g., "subject_id" for MIMIC
    admission: str  # e.g., "hadm_id" for MIMIC


class DatasetConfig(BaseModel):
    """Dataset-level configuration (e.g., mimic_iv.yaml, eicu.yaml).

    Defines dataset-wide settings for ID columns, time handling, and table paths.
    """

    name: str
    description: Optional[str] = None
    time: TimeConfig
    ids: IDConfig
    tables: Dict[str, str]  # logical name -> path (e.g., "chartevents" -> "icu/chartevents")

    @field_validator("tables")
    @classmethod
    def validate_tables_not_empty(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Tables mapping cannot be empty."""
        if not v:
            raise ValueError("tables mapping cannot be empty")
        return v


# =============================================================================
# Time-Series Extraction Sources
# =============================================================================


class ItemIDSource(BaseModel):
    """Extraction source using itemid matching (MIMIC-style).

    Used for MIMIC-IV chartevents, labevents, outputevents where features
    are identified by numeric item IDs.
    """

    table: str
    type: Literal["itemid"] = "itemid"
    itemid: List[int]
    value_col: str = "valuenum"
    time_col: str
    transform: Optional[str] = None

    @field_validator("itemid")
    @classmethod
    def validate_itemid_not_empty(cls, v: List[int]) -> List[int]:
        """Item ID list cannot be empty."""
        if not v:
            raise ValueError("itemid list cannot be empty")
        return v


class ColumnSource(BaseModel):
    """Extraction source using direct column read (eICU-style).

    Used for eICU tables like vitalperiodic where each vital has its own column.
    """

    table: str
    type: Literal["column"] = "column"
    value_col: str
    time_col: str
    transform: Optional[str] = None


class StringMatchSource(BaseModel):
    """Extraction source using exact string matching.

    Used for eICU lab table where labs are identified by labname string.
    """

    table: str
    type: Literal["string"] = "string"
    match_col: str
    match_value: str
    value_col: str
    time_col: str
    transform: Optional[str] = None


class RegexMatchSource(BaseModel):
    """Extraction source using regex pattern matching.

    Used for medication extraction where drug names need pattern matching.
    """

    table: str
    type: Literal["regex"] = "regex"
    match_col: str
    pattern: str
    value_col: Optional[str] = None  # May not have value for boolean flags
    time_col: str
    transform: Optional[str] = None


# Discriminated union for any extraction source
ExtractionSource = Annotated[
    Union[ItemIDSource, ColumnSource, StringMatchSource, RegexMatchSource],
    Field(discriminator="type"),
]


# =============================================================================
# Time-Series Concept Configuration
# =============================================================================


class TimeSeriesConceptConfig(BaseModel):
    """Configuration for a time-series concept (vital, lab, output, etc.).

    Each concept can have extraction rules for multiple datasets. The dataset
    sources are lists to support combining multiple extraction patterns
    (e.g., temperature from both Fahrenheit and Celsius item IDs).
    """

    # Metadata
    description: Optional[str] = None
    omopid: Optional[int] = None
    feature_set: List[str] = Field(default_factory=lambda: ["core", "extended"])

    # Validation & units
    units: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    value_type: ValueType = ValueType.NUMERIC
    aggregation: AggregationType = AggregationType.MEAN

    # Dataset-specific extraction rules (list of sources per dataset)
    mimic_iv: Optional[List[ExtractionSource]] = None
    eicu: Optional[List[ExtractionSource]] = None

    model_config = {"extra": "allow"}  # Allow future datasets


# =============================================================================
# Static Feature Configuration
# =============================================================================


class StaticExtractionSource(BaseModel):
    """Extraction source for static features (simpler than time-series).

    Static features are extracted once per stay, not over time.
    """

    table: str
    column: str
    itemid: Optional[int] = None  # For cases like height/weight from chartevents
    transform: Optional[str] = None


class StaticConceptConfig(BaseModel):
    """Configuration for a static/demographic concept.

    Static features don't vary over time (age, gender, race, etc.).
    Each dataset has a single extraction source (not a list).
    """

    # Metadata
    description: Optional[str] = None
    omopid: Optional[int] = None

    # Type and validation
    dtype: Literal["numeric", "categorical"] = "numeric"
    units: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    categories: Optional[List[str]] = None

    # Dataset-specific extraction (single source per dataset, not list)
    mimic_iv: Optional[StaticExtractionSource] = None
    eicu: Optional[StaticExtractionSource] = None

    model_config = {"extra": "allow"}  # Allow future datasets

    @model_validator(mode="after")
    def validate_categorical_has_dtype(self) -> "StaticConceptConfig":
        """Categorical features should have dtype='categorical'."""
        if self.categories and self.dtype != "categorical":
            raise ValueError("Features with categories should have dtype='categorical'")
        return self


# =============================================================================
# Unified Data Configuration
# =============================================================================


class DataConfig(BaseModel):
    """Unified configuration for the complete data pipeline.

    This schema validates the 'data' section of configs/data/mimic_iv.yaml,
    which is the single source of truth for all data settings including:
    - Data paths (csv_root, parquet_root, processed_dir)
    - Extraction settings (seq_length_hours, feature_set, tasks)
    - Training data loading (num_workers, split ratios, normalization)

    Used by both extraction scripts and training scripts.
    """

    # Dataset identifier
    name: Optional[str] = None

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    csv_root: Optional[str] = None  # Raw CSV path (only for convert_csv_to_parquet)
    parquet_root: Optional[str] = None  # Parquet files (input for extraction)
    processed_dir: str  # Extracted features (output of extraction, input for training)

    # ==========================================================================
    # Extraction Settings
    # ==========================================================================
    seq_length_hours: int = SEQ_LENGTH_HOURS
    min_stay_hours: int = MIN_STAY_HOURS
    feature_set: Literal["core", "extended"] = "core"  # Feature set to extract
    categories: Optional[List[str]] = None  # Feature categories (null = all)
    extraction_batch_size: int = EXTRACTION_BATCH_SIZE
    tasks: List[str] = Field(default_factory=lambda: ["mortality_24h"])

    # Config directory paths (auto-detected if null)
    concepts_dir: Optional[str] = None
    datasets_dir: Optional[str] = None
    tasks_dir: Optional[str] = None

    # ==========================================================================
    # Training Data Loading
    # ==========================================================================
    num_workers: int = 4
    pin_memory: bool = PIN_MEMORY

    # Patient-level split ratios
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO

    # Preprocessing applied during training
    normalize: bool = NORMALIZE
    impute_strategy: Literal["forward_fill", "zero", "mean", "none"] = IMPUTE_STRATEGY

    model_config = {"extra": "allow"}  # Allow additional fields from Hydra

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DataConfig":
        """Validate that split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self

    @field_validator("processed_dir")
    @classmethod
    def validate_processed_dir_not_empty(cls, v: str) -> str:
        """Validate that processed_dir is not empty."""
        if not v or not v.strip():
            raise ValueError("processed_dir cannot be empty")
        return v

    @field_validator("seq_length_hours", "min_stay_hours", "extraction_batch_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that value is positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
