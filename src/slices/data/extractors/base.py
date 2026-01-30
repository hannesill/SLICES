"""Abstract base class for ICU data extractors.

Reads from local Parquet files using DuckDB for efficient SQL queries.
Users specify parquet_root pointing to their local Parquet files.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, Generator, List, Optional

import duckdb
import polars as pl
import portalocker
import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from slices.data.config_loader import load_timeseries_concepts
from slices.data.config_schemas import AggregationType, TimeSeriesConceptConfig
from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig

console = Console()


class ExtractorConfig(BaseModel):
    """Configuration for data extraction using Pydantic validation.

    Users must specify parquet_root - the path to their local Parquet files.

    Attributes:
        parquet_root: Path to directory containing Parquet files (required).
        output_dir: Directory to write extracted data to.
        seq_length_hours: Length of time-series sequences in hours (must be > 0).
        feature_set: Feature set to extract ('core' or 'extended').
        concepts_dir: Path to concepts directory (auto-detected if None).
        datasets_dir: Path to dataset configs directory (auto-detected if None).
        tasks_dir: Path to task config directory (auto-detected if None).
        tasks: List of task names to extract labels for.
        min_stay_hours: Minimum ICU stay length to include (must be >= 0).
        batch_size: Number of stays to process in each batch (for memory efficiency).
    """

    parquet_root: str
    output_dir: str = "data/processed"
    seq_length_hours: int = Field(default=48, gt=0)
    feature_set: str = "core"  # core | extended
    concepts_dir: Optional[str] = None
    datasets_dir: Optional[str] = None  # Path to dataset configs (auto-detected if None)
    tasks_dir: Optional[str] = None
    tasks: List[str] = Field(
        default_factory=lambda: ["mortality_24h", "mortality_48h", "mortality_hospital"]
    )
    min_stay_hours: int = Field(default=6, ge=0)
    batch_size: int = Field(default=5000, gt=0)
    categories: Optional[List[str]] = None  # e.g., ["vitals_dense"] for subset extraction

    model_config = {"extra": "forbid"}

    @field_validator("parquet_root", "output_dir")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that path is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty or whitespace-only")
        return v

    @field_validator("feature_set")
    @classmethod
    def validate_feature_set(cls, v: str) -> str:
        """Validate that feature_set is a known value."""
        valid_feature_sets = {"core", "extended"}
        if v not in valid_feature_sets:
            raise ValueError(f"feature_set must be one of {valid_feature_sets}, got '{v}'")
        return v


class BaseExtractor(ABC):
    """Abstract base class for ICU data extractors.

    Reads from local Parquet files using DuckDB for efficient SQL queries.
    Users specify parquet_root pointing to their local Parquet files.
    """

    def __init__(self, config: ExtractorConfig) -> None:
        """Initialize extractor with configuration.

        Args:
            config: Extractor configuration containing parquet_root and other settings.

        Raises:
            ValueError: If parquet directory does not exist.
        """
        self.config = config
        self.parquet_root = Path(config.parquet_root)
        self.output_dir = Path(config.output_dir)
        self.conn = duckdb.connect()  # In-memory DuckDB for queries

        # Caches to avoid repeated expensive queries during batch processing
        self._stays_cache: Optional[pl.DataFrame] = None
        self._feature_mapping_cache: Optional[Dict[str, TimeSeriesConceptConfig]] = None

        # Validate parquet directory exists
        if not self.parquet_root.exists():
            raise ValueError(f"Parquet directory not found: {self.parquet_root}")

    def __del__(self) -> None:
        """Cleanup DuckDB connection on deletion."""
        if hasattr(self, "conn"):
            try:
                self.conn.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def _get_stays_cached(self) -> pl.DataFrame:
        """Get stays DataFrame with caching to avoid repeated queries.

        This is used internally during batch processing to avoid re-querying
        the database for each batch.

        Returns:
            Cached stays DataFrame.
        """
        if self._stays_cache is None:
            self._stays_cache = self.extract_stays()
        return self._stays_cache

    def _get_feature_mapping_cached(self, feature_set: str) -> Dict[str, TimeSeriesConceptConfig]:
        """Get feature mapping with caching to avoid repeated YAML loading.

        Args:
            feature_set: Name of feature set to load (e.g., 'core').

        Returns:
            Cached feature mapping dictionary.
        """
        if self._feature_mapping_cache is None:
            self._feature_mapping_cache = self._load_feature_mapping(feature_set)
        return self._feature_mapping_cache

    def _query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query on local Parquet files and return Polars DataFrame.

        Args:
            sql: SQL query string.

        Returns:
            Polars DataFrame with query results.
        """
        return self.conn.execute(sql).pl()

    def _parquet_path(self, schema: str, table: str) -> Path:
        """Get path to a Parquet file: {parquet_root}/{schema}/{table}.parquet.

        Args:
            schema: Schema/directory name (e.g., 'hosp', 'icu').
            table: Table name (e.g., 'patients', 'chartevents').

        Returns:
            Path to the Parquet file.
        """
        return self.parquet_root / schema / f"{table}.parquet"

    def _get_project_root(self) -> Optional[Path]:
        """Find project root by looking for pyproject.toml.

        Returns:
            Path to project root or None if not found.
        """
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / "pyproject.toml").exists():
                return parent
        return None

    def _get_concepts_path(self) -> Path:
        """Get path to concepts directory with robust fallback strategy.

        Returns:
            Path to concepts directory containing feature YAML files.

        Strategy:
            1. Use concepts_dir from config if explicitly provided
            2. Try to find project root and look for configs/concepts
            3. Try relative to this source file (development mode)

        Raises:
            FileNotFoundError: If concepts directory cannot be found.
        """
        # Strategy 1: Explicit config path (most robust for deployment)
        if self.config.concepts_dir is not None:
            concepts_path = Path(self.config.concepts_dir)
            if concepts_path.exists():
                return concepts_path
            raise FileNotFoundError(
                f"Concepts directory specified in config not found: {concepts_path}"
            )

        # Strategy 2: Try to find project root (works for editable installs)
        project_root = self._get_project_root()
        if project_root:
            concepts_dir = project_root / "configs" / "concepts"
            if concepts_dir.exists():
                return concepts_dir

        # Strategy 3: Relative to source file (development mode fallback)
        # From src/slices/data/extractors/base.py -> configs/concepts
        relative_path = Path(__file__).parents[4] / "configs" / "concepts"
        if relative_path.exists():
            return relative_path

        # If nothing works, give helpful error
        raise FileNotFoundError(
            "Could not locate concepts directory. Options:\n"
            "1. Set 'concepts_dir' in ExtractorConfig (recommended for deployment)\n"
            "2. Run from project root containing pyproject.toml\n"
            "3. Ensure configs/concepts exists relative to source tree"
        )

    def _get_tasks_path(self) -> Path:
        """Get path to tasks directory with robust fallback strategy.

        Returns:
            Path to tasks directory containing task YAML files.

        Raises:
            FileNotFoundError: If tasks directory cannot be found.
        """
        # Strategy 1: Explicit config path
        if self.config.tasks_dir is not None:
            tasks_path = Path(self.config.tasks_dir)
            if tasks_path.exists():
                return tasks_path
            raise FileNotFoundError(f"Tasks directory specified in config not found: {tasks_path}")

        # Strategy 2: Try to find project root
        project_root = self._get_project_root()
        if project_root:
            tasks_dir = project_root / "configs" / "tasks"
            if tasks_dir.exists():
                return tasks_dir

        # Strategy 3: Relative to source file
        relative_path = Path(__file__).parents[4] / "configs" / "tasks"
        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            "Could not locate tasks directory. Options:\n"
            "1. Set 'tasks_dir' in ExtractorConfig\n"
            "2. Run from project root containing pyproject.toml\n"
            "3. Ensure configs/tasks exists relative to source tree"
        )

    def _get_datasets_path(self) -> Path:
        """Get path to datasets directory with robust fallback strategy.

        Returns:
            Path to datasets directory containing dataset config YAML files.

        Raises:
            FileNotFoundError: If datasets directory cannot be found.
        """
        # Strategy 1: Explicit config path
        if self.config.datasets_dir is not None:
            datasets_path = Path(self.config.datasets_dir)
            if datasets_path.exists():
                return datasets_path
            raise FileNotFoundError(
                f"Datasets directory specified in config not found: {datasets_path}"
            )

        # Strategy 2: Try to find project root
        project_root = self._get_project_root()
        if project_root:
            datasets_dir = project_root / "configs" / "datasets"
            if datasets_dir.exists():
                return datasets_dir

        # Strategy 3: Relative to source file
        relative_path = Path(__file__).parents[4] / "configs" / "datasets"
        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            "Could not locate datasets directory. Options:\n"
            "1. Set 'datasets_dir' in ExtractorConfig\n"
            "2. Run from project root containing pyproject.toml\n"
            "3. Ensure configs/datasets exists relative to source tree"
        )

    def _load_task_configs(self, task_names: List[str]) -> List[LabelConfig]:
        """Load task configurations from YAML files.

        Args:
            task_names: List of task names to load (e.g., ['mortality_24h']).

        Returns:
            List of LabelConfig instances.

        Raises:
            ValueError: If task's observation_window_hours doesn't match seq_length_hours.
        """
        tasks_path = self._get_tasks_path()
        task_configs = []

        for task_name in task_names:
            config_file = tasks_path / f"{task_name}.yaml"
            if not config_file.exists():
                console.print(f"[yellow]Warning: Task config not found: {config_file}[/yellow]")
                continue

            with open(config_file) as f:
                config_dict = yaml.safe_load(f)

            task_config = LabelConfig(**config_dict)

            # Validate observation_window_hours matches seq_length_hours
            self._validate_observation_window(task_config)

            task_configs.append(task_config)

        return task_configs

    def _validate_observation_window(self, task_config: LabelConfig) -> None:
        """Validate that task's observation_window_hours matches extraction seq_length_hours.

        This is critical for preventing data leakage. If the model sees more data
        (seq_length_hours) than the task expects (observation_window_hours), patients
        who died during the observation window might not be properly excluded.

        Args:
            task_config: Task configuration to validate.

        Raises:
            ValueError: If observation_window_hours doesn't match seq_length_hours.
        """
        obs_window = task_config.observation_window_hours
        seq_length = self.config.seq_length_hours

        if obs_window is None:
            # Legacy mode (no observation window) - no validation needed
            return

        if obs_window != seq_length:
            raise ValueError(
                f"Configuration mismatch for task '{task_config.task_name}': "
                f"observation_window_hours ({obs_window}) != seq_length_hours ({seq_length}). "
                f"This can cause data leakage - patients who died during the observation "
                f"window may not be properly excluded. "
                f"Either update the task config's observation_window_hours to {seq_length}, "
                f"or change seq_length_hours in your extraction config to {obs_window}."
            )

    def _load_feature_mapping(self, feature_set: str) -> Dict[str, TimeSeriesConceptConfig]:
        """Load feature mapping from YAML config files.

        Uses the new config_loader to load validated Pydantic models from
        split concept files (vitals.yaml, labs.yaml, outputs.yaml, etc.).

        Args:
            feature_set: Name of feature set to load (e.g., 'core').

        Returns:
            Dictionary mapping feature names to validated TimeSeriesConceptConfig.

        Raises:
            FileNotFoundError: If concepts directory cannot be found.
            ValueError: If no features found for this dataset.
        """
        concepts_dir = self._get_concepts_path()
        dataset_name = self._get_dataset_name()

        # Use new config loader to load validated Pydantic models
        feature_mapping = load_timeseries_concepts(
            concepts_dir=concepts_dir,
            dataset_name=dataset_name,
            feature_set=feature_set,
            categories=self.config.categories,
        )

        return feature_mapping

    def _bin_to_hourly_grid(
        self,
        raw_events: pl.DataFrame,
        stay_ids: List[int],
        feature_mapping: Dict[str, TimeSeriesConceptConfig],
    ) -> pl.DataFrame:
        """Bin raw events to hourly grid and create observation masks (generic).

        This method is completely dataset-agnostic. It expects raw_events to have:
        - stay_id, charttime, feature_name, valuenum columns

        Args:
            raw_events: Raw events DataFrame with standardized schema.
            stay_ids: List of ICU stay IDs.
            feature_mapping: Feature mapping with TimeSeriesConceptConfig values.

        Returns:
            Wide-format DataFrame with hourly bins and observation masks.
        """
        # Get stay info for intime calculation (use cache to avoid repeated queries)
        stays = self._get_stays_cached().filter(pl.col("stay_id").is_in(stay_ids))

        # Join with stays to get intime and calculate hour offset
        events_with_hours = (
            raw_events.join(
                stays.select(["stay_id", "intime"]), on="stay_id", how="left"
            ).with_columns(
                [
                    # Calculate hour offset from ICU admission (floor to get hour bins)
                    ((pl.col("charttime") - pl.col("intime")).dt.total_seconds() / 3600)
                    .floor()
                    .cast(pl.Int32)
                    .alias("hour")
                ]
            )
            # Filter out events before admission (hour < 0)
            .filter(pl.col("hour") >= 0)
        )

        # Group features by aggregation type
        agg_groups: Dict[AggregationType, List[str]] = {}
        for feature_name, config in feature_mapping.items():
            agg_type = config.aggregation
            agg_groups.setdefault(agg_type, []).append(feature_name)

        aggregated_parts: List[pl.DataFrame] = []

        # Process each aggregation type
        for agg_type, features in agg_groups.items():
            filtered = events_with_hours.filter(pl.col("feature_name").is_in(features))
            if filtered.is_empty():
                continue

            grouped = filtered.group_by(["stay_id", "hour", "feature_name"])

            if agg_type == AggregationType.MEAN:
                agg_df = grouped.agg(
                    [pl.col("valuenum").mean().alias("value"), pl.lit(True).alias("observed")]
                )
            elif agg_type == AggregationType.SUM:
                agg_df = grouped.agg(
                    [pl.col("valuenum").sum().alias("value"), pl.lit(True).alias("observed")]
                )
            elif agg_type == AggregationType.MAX:
                agg_df = grouped.agg(
                    [pl.col("valuenum").max().alias("value"), pl.lit(True).alias("observed")]
                )
            elif agg_type == AggregationType.MIN:
                agg_df = grouped.agg(
                    [pl.col("valuenum").min().alias("value"), pl.lit(True).alias("observed")]
                )
            elif agg_type == AggregationType.LAST:
                # Sort by charttime to ensure chronological ordering before taking last
                agg_df = grouped.agg(
                    [
                        pl.col("valuenum").sort_by("charttime").last().alias("value"),
                        pl.lit(True).alias("observed"),
                    ]
                )
            elif agg_type == AggregationType.FIRST:
                # Sort by charttime to ensure chronological ordering before taking first
                agg_df = grouped.agg(
                    [
                        pl.col("valuenum").sort_by("charttime").first().alias("value"),
                        pl.lit(True).alias("observed"),
                    ]
                )
            elif agg_type == AggregationType.ANY:
                agg_df = grouped.agg([pl.lit(True).alias("value"), pl.lit(True).alias("observed")])
            else:
                # Fallback to mean
                agg_df = grouped.agg(
                    [pl.col("valuenum").mean().alias("value"), pl.lit(True).alias("observed")]
                )

            aggregated_parts.append(agg_df)

        aggregated = (
            pl.concat(aggregated_parts)
            if aggregated_parts
            else pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "hour": pl.Int64,
                    "feature_name": pl.Utf8,
                    "value": pl.Float64,
                    "observed": pl.Boolean,
                }
            )
        )

        # Pivot to wide format: one column per feature
        # First pivot the values
        pivoted_values = aggregated.pivot(
            values="value", index=["stay_id", "hour"], columns="feature_name"
        )

        # Pivot the observation masks
        pivoted_masks = aggregated.pivot(
            values="observed", index=["stay_id", "hour"], columns="feature_name"
        )

        # Rename mask columns to have _mask suffix (exclude index columns)
        mask_columns = {
            col: f"{col}_mask" for col in pivoted_masks.columns if col not in ["stay_id", "hour"]
        }
        pivoted_masks = pivoted_masks.rename(mask_columns)

        # Join values and masks
        result = pivoted_values.join(pivoted_masks, on=["stay_id", "hour"], how="left")

        # Ensure all expected feature columns exist (even if no data)
        expected_features = list(feature_mapping.keys())
        for feature in expected_features:
            # Add value column if missing
            if feature not in result.columns:
                result = result.with_columns([pl.lit(None, dtype=pl.Float64).alias(feature)])
            # Add mask column if missing
            mask_col = f"{feature}_mask"
            if mask_col not in result.columns:
                result = result.with_columns([pl.lit(False).alias(mask_col)])

        # Fill missing mask values with False (not observed)
        for col in result.columns:
            if col.endswith("_mask"):
                result = result.with_columns([pl.col(col).fill_null(False)])

        # Sort by stay_id and hour for cleaner output
        result = result.sort(["stay_id", "hour"])

        return result

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for this extractor.

        Returns:
            Dataset name (e.g., 'mimic_iv', 'eicu', 'hirid').
            Used to parse dataset-specific configs from concept YAML files.
        """
        pass

    @abstractmethod
    def extract_stays(self) -> pl.DataFrame:
        """Extract ICU stay metadata (stay_id, patient_id, times).

        Returns:
            DataFrame with columns: stay_id, patient_id, intime, outtime,
            length_of_stay_days, age, gender, admission_type, first_careunit,
            last_careunit.
        """
        pass

    @abstractmethod
    def _extract_raw_events(
        self, stay_ids: List[int], feature_mapping: Dict[str, TimeSeriesConceptConfig]
    ) -> pl.DataFrame:
        """Extract raw time-series events for specified features (dataset-specific).

        This is the dataset-specific method that queries the appropriate tables
        and returns raw events. Each dataset implements this differently.

        Args:
            stay_ids: List of ICU stay IDs to extract.
            feature_mapping: Dictionary mapping feature names to TimeSeriesConceptConfig.

        Returns:
            DataFrame with columns:
                - stay_id: ICU stay identifier
                - charttime: Timestamp of observation (or similar time column)
                - feature_name: Name of feature (canonical name from config)
                - valuenum: Numeric value
        """
        pass

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract time-series features for given stays (generic orchestration).

        This method provides generic orchestration for time-series extraction:
        1. Load feature mapping from concept configs
        2. Extract raw events (dataset-specific via _extract_raw_events)
        3. Bin to hourly grid and create observation masks

        Args:
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with columns:
                - stay_id: ICU stay identifier
                - hour: Hour offset from ICU admission (0, 1, 2, ...)
                - {feature_name}: Value for each feature (mean aggregated)
                - {feature_name}_mask: Boolean indicating if value was observed
        """
        # Use feature_set from config (or default to 'core')
        feature_set = self.config.feature_set

        # Load feature mapping for this dataset (use cache for batch processing)
        feature_mapping = self._get_feature_mapping_cached(feature_set)

        # Validate feature mapping is non-empty
        if not feature_mapping:
            raise ValueError(
                f"No features loaded from concept set '{feature_set}' for dataset "
                f"'{self._get_dataset_name()}'. Check that your YAML config has "
                f"entries for this dataset under each feature."
            )

        # Extract raw events (dataset-specific implementation)
        raw_events = self._extract_raw_events(stay_ids, feature_mapping)

        # Bin to hourly grid (generic)
        hourly_binned = self._bin_to_hourly_grid(raw_events, stay_ids, feature_mapping)

        return hourly_binned

    @abstractmethod
    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        """Extract raw data for a specific source (e.g., 'mortality_info', 'creatinine').

        This is a low-level method that extracts raw clinical data without computing labels.
        LabelBuilders call this to get the data they need.

        Args:
            source_name: Name of data source to extract (e.g., 'mortality_info',
                        'vasopressor_usage', 'creatinine').
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with raw data for the specified source.

        Raises:
            ValueError: If source_name is not recognized.
        """
        pass

    def _validate_stays(self, stays: pl.DataFrame) -> None:
        """Validate stay metadata for required fields and data quality.

        Args:
            stays: DataFrame with stay metadata.

        Raises:
            ValueError: If validation fails.
        """
        # Check for null patient_ids (required for patient-level splits)
        null_patients = stays.filter(pl.col("patient_id").is_null())
        if len(null_patients) > 0:
            raise ValueError(
                f"Found {len(null_patients)} stays with null patient_id. "
                "Patient IDs are required for patient-level splits."
            )

        # Check for duplicate stay_ids
        duplicates = stays.filter(pl.col("stay_id").is_duplicated())
        if len(duplicates) > 0:
            raise ValueError(
                f"Found {len(duplicates)} duplicate stay_ids. Each stay must have a unique ID."
            )

        # Check for negative or invalid LOS
        invalid_los = stays.filter((pl.col("los_days").is_null()) | (pl.col("los_days") < 0))
        if len(invalid_los) > 0:
            console.print(
                f"[yellow]Warning: Found {len(invalid_los)} stays with invalid LOS[/yellow]"
            )

    def _validate_timeseries(
        self, timeseries: pl.DataFrame, stay_ids: List[int], feature_names: List[str]
    ) -> None:
        """Validate timeseries data consistency.

        Args:
            timeseries: Sparse timeseries DataFrame.
            stay_ids: List of expected stay IDs.
            feature_names: List of expected feature names.

        Raises:
            ValueError: If validation fails.
        """
        # Check that all stays have timeseries data
        timeseries_stay_ids = set(timeseries["stay_id"].unique().to_list())
        expected_stay_ids = set(stay_ids)

        missing = expected_stay_ids - timeseries_stay_ids
        if missing:
            console.print(f"[yellow]Warning: {len(missing)} stays have no timeseries data[/yellow]")

        # Check that expected features are present
        timeseries_features = {
            col
            for col in timeseries.columns
            if col not in ["stay_id", "hour"] and not col.endswith("_mask")
        }
        missing_features = set(feature_names) - timeseries_features
        if missing_features:
            console.print(
                f"[yellow]Warning: Missing features in timeseries: {missing_features}[/yellow]"
            )

    def _validate_labels(self, labels: pl.DataFrame, stay_ids: List[int]) -> None:
        """Validate label data consistency.

        Args:
            labels: Labels DataFrame.
            stay_ids: List of expected stay IDs.

        Raises:
            ValueError: If validation fails.
        """
        if len(labels) == 0:
            return

        label_stay_ids = set(labels["stay_id"].to_list())
        expected_stay_ids = set(stay_ids)

        # Check for missing labels
        missing = expected_stay_ids - label_stay_ids
        if missing:
            console.print(f"[yellow]Warning: {len(missing)} stays have no labels[/yellow]")

        # Check for extra labels (shouldn't happen, but worth checking)
        extra = label_stay_ids - expected_stay_ids
        if extra:
            console.print(f"[yellow]Warning: {len(extra)} labels for stays not in dataset[/yellow]")

    def extract_labels(self, stay_ids: List[int], task_configs: List[LabelConfig]) -> pl.DataFrame:
        """Extract labels for multiple downstream tasks.

        This is a framework method that orchestrates label extraction:
        1. Identifies all required data sources from task configs
        2. Extracts raw data for each source (via extract_data_source)
        3. Uses LabelBuilders to compute labels from raw data
        4. Combines labels into a single DataFrame

        Args:
            stay_ids: List of ICU stay IDs to extract labels for.
            task_configs: List of task configurations defining which labels to compute.

        Returns:
            DataFrame with stay_id and one column per task (named by task_name).
        """
        # Step 1: Identify all required data sources
        required_sources = set()
        for task_config in task_configs:
            required_sources.update(task_config.label_sources)

        # Always include 'stays' as it's needed for temporal alignment
        required_sources.add("stays")

        # Step 2: Extract all raw data sources
        raw_data: Dict[str, pl.DataFrame] = {}
        for source in required_sources:
            if source == "stays":
                # Use full stays DataFrame (not filtered by stay_ids)
                raw_data["stays"] = self.extract_stays().filter(pl.col("stay_id").is_in(stay_ids))
            else:
                # Extract other data sources via abstract method
                raw_data[source] = self.extract_data_source(source, stay_ids)

        # Step 3: Build labels for each task
        all_labels = []
        for task_config in task_configs:
            # Create appropriate LabelBuilder
            builder: LabelBuilder = LabelBuilderFactory.create(task_config)

            # Compute labels
            task_labels = builder.build_labels(raw_data)

            # For single-label tasks, the builder returns a 'label' column
            # that we rename to the task name.
            # For multi-label tasks (e.g., phenotyping), the builder returns
            # pre-named columns (no 'label' column) — use as-is.
            if "label" in task_labels.columns:
                task_labels = task_labels.rename({"label": task_config.task_name})

            all_labels.append(task_labels)

        # Step 4: Combine all task labels into single DataFrame
        if not all_labels:
            # No tasks specified - return just stay_ids
            return raw_data["stays"].select("stay_id")

        # Start with first task
        combined = all_labels[0]

        # Join remaining tasks (use left join since all tasks process same stay_ids)
        for task_labels in all_labels[1:]:
            combined = combined.join(task_labels, on="stay_id", how="left")

        return combined

    def _create_dense_timeseries(
        self,
        sparse_timeseries: pl.DataFrame,
        stays: pl.DataFrame,
        feature_names: List[str],
    ) -> pl.DataFrame:
        """Convert sparse hourly timeseries to dense fixed-length arrays.

        Creates a dense representation where each stay has exactly seq_length_hours
        timesteps, with missing hours filled with NaN and masks set to False.

        Args:
            sparse_timeseries: Sparse timeseries with stay_id, hour, features, masks.
            stays: Stay metadata with stay_id, los_days for length calculation.
            feature_names: List of feature column names (without _mask suffix).

        Returns:
            DataFrame with one row per stay containing:
                - stay_id: ICU stay identifier
                - timeseries: List of lists (seq_length x n_features) with values
                - mask: List of lists (seq_length x n_features) with observation flags
        """
        seq_length = self.config.seq_length_hours
        n_features = len(feature_names)

        # Use stay_ids from stays DataFrame (not sparse_timeseries) to ensure
        # ALL requested stays are included, even those with no observations.
        # This prevents dimension mismatch between timeseries.parquet and labels.parquet.
        stay_ids = stays["stay_id"].unique().to_list()

        # Track which stays have data for logging
        stays_with_data = set(sparse_timeseries["stay_id"].unique().to_list())
        stays_without_data = set(stay_ids) - stays_with_data
        if stays_without_data:
            console.print(
                f"[yellow]Warning: {len(stays_without_data)} stays have no timeseries "
                f"observations. They will have all-NaN/all-False arrays.[/yellow]"
            )

        results = []
        global_overflow_count = 0
        global_overflow_max_hour = 0

        for stay_id in stay_ids:
            # Get data for this stay (may be empty)
            stay_data = sparse_timeseries.filter(pl.col("stay_id") == stay_id)

            # Initialize dense arrays with NaN/False
            dense_values = [[float("nan")] * n_features for _ in range(seq_length)]
            dense_mask = [[False] * n_features for _ in range(seq_length)]

            # Track overflow for this stay
            stay_overflow_count = 0
            stay_overflow_max_hour = 0

            # Fill in observed values
            for row in stay_data.iter_rows(named=True):
                hour = row["hour"]
                if 0 <= hour < seq_length:
                    for feat_idx, feat_name in enumerate(feature_names):
                        value = row.get(feat_name)
                        mask_val = row.get(f"{feat_name}_mask", False)

                        if value is not None and mask_val:
                            dense_values[hour][feat_idx] = float(value)
                            dense_mask[hour][feat_idx] = True
                elif hour >= seq_length:
                    stay_overflow_count += 1
                    stay_overflow_max_hour = max(stay_overflow_max_hour, hour)

            # Accumulate global overflow stats
            global_overflow_count += stay_overflow_count
            global_overflow_max_hour = max(global_overflow_max_hour, stay_overflow_max_hour)

            results.append(
                {
                    "stay_id": stay_id,
                    "timeseries": dense_values,
                    "mask": dense_mask,
                }
            )

        # Warn if data was truncated
        if global_overflow_count > 0:
            console.print(
                f"[yellow]Warning: Discarded {global_overflow_count} data points beyond "
                f"seq_length={seq_length} (max hour observed: {global_overflow_max_hour}). "
                f"Consider increasing seq_length_hours if late observations are important.[/yellow]"
            )

        return pl.DataFrame(results)

    def _check_existing_extraction(self) -> Optional[Dict[str, pl.DataFrame]]:
        """Check if extraction output files already exist.

        Returns:
            Dictionary with existing DataFrames if all files exist, None otherwise.
            Keys: 'stays', 'timeseries', 'labels'
        """
        static_path = self.output_dir / "static.parquet"
        timeseries_path = self.output_dir / "timeseries.parquet"
        labels_path = self.output_dir / "labels.parquet"
        metadata_path = self.output_dir / "metadata.yaml"

        # Check if all required files exist
        if not all(p.exists() for p in [static_path, timeseries_path, labels_path, metadata_path]):
            return None

        # Check if metadata matches current config
        try:
            with open(metadata_path) as f:
                existing_metadata = yaml.safe_load(f)

            # Verify config matches (all critical fields for reproducibility)
            existing_task_names = set(existing_metadata.get("task_names", []))
            current_task_names = set(self.config.tasks or [])

            if (
                existing_metadata.get("feature_set") != self.config.feature_set
                or existing_metadata.get("seq_length_hours") != self.config.seq_length_hours
                or existing_metadata.get("categories") != self.config.categories
                or existing_metadata.get("min_stay_hours") != self.config.min_stay_hours
                or existing_task_names != current_task_names
            ):
                console.print(
                    "[yellow]Warning: Existing extraction has different config. "
                    "Will overwrite.[/yellow]"
                )
                return None
        except Exception:
            # If metadata can't be read, assume we should overwrite
            return None

        # Load existing data
        try:
            existing = {
                "stays": pl.read_parquet(static_path),
                "timeseries": pl.read_parquet(timeseries_path),
                "labels": pl.read_parquet(labels_path),
            }
            console.print(
                f"[green]Found existing extraction with {len(existing['stays'])} stays. "
                "Will resume and merge with new data.[/green]"
            )
            return existing
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load existing extraction: {e}[/yellow]")
            return None

    @contextmanager
    def _with_file_lock(self, path: Path) -> Generator[None, None, None]:
        """Context manager for file locking to prevent race conditions.

        Uses cross-platform file locking (portalocker) to ensure only one process
        can write to a file at a time. This prevents data duplication and corruption
        during concurrent extraction operations.

        Works on Windows, macOS, and Linux.

        Args:
            path: Path to the file to lock

        Yields:
            None

        Raises:
            portalocker.exceptions.LockException: If locking fails
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_file = None
        try:
            lock_file = open(lock_path, "w")
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            try:
                yield
            finally:
                portalocker.unlock(lock_file)
        finally:
            if lock_file is not None:
                lock_file.close()
                # Clean up lock file
                try:
                    lock_path.unlink()
                except OSError:
                    pass  # Lock file might be in use by another process

    def _atomic_write(self, path: Path, write_fn: Callable[[str], None], suffix: str = "") -> None:
        """Write file atomically using temp file + rename.

        Works with any file format by accepting a callback function that
        performs the actual write. Ensures atomic writes on POSIX and Windows by:
        1. Writing to a temporary file
        2. Atomically renaming to the target path

        This prevents partial/corrupted files if the process crashes
        or is interrupted during write.

        Args:
            path: Target path for the file
            write_fn: Callable that takes temp file path and writes to it
            suffix: File suffix for temp file (e.g., ".parquet", ".yaml")

        Example:
            # Write parquet
            self._atomic_write(
                path,
                lambda tmp: df.write_parquet(tmp),
                suffix=".parquet"
            )

            # Write YAML
            self._atomic_write(
                path,
                lambda tmp: yaml.dump(metadata, open(tmp, "w")),
                suffix=".yaml"
            )
        """
        if not suffix:
            suffix = path.suffix

        with NamedTemporaryFile(dir=path.parent, suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            write_fn(tmp_path)
            os.replace(tmp_path, path)  # Atomic on POSIX and Windows
        except Exception:
            # Clean up temp file if write failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def run(self) -> None:
        """Execute full extraction pipeline.

        Extracts and saves:
        - static.parquet: Stay-level metadata (demographics, admission info)
        - timeseries.parquet: Dense hourly features with observation masks
        - labels.parquet: Task labels for downstream prediction
        - metadata.yaml: Extraction metadata (feature names, task names, etc.)

        If output files already exist with matching configuration, will resume
        extraction by skipping already-extracted stays and merging results.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing extraction
        existing_data = self._check_existing_extraction()
        existing_stay_ids = set()
        if existing_data is not None:
            existing_stay_ids = set(existing_data["stays"]["stay_id"].to_list())

        console.print("\n[bold blue]SLICES Data Extraction[/bold blue]")
        console.print(f"Output directory: {self.output_dir}")
        console.print(f"Feature set: {self.config.feature_set}")
        console.print(f"Sequence length: {self.config.seq_length_hours} hours")
        console.print(f"Min stay length: {self.config.min_stay_hours} hours")
        if existing_stay_ids:
            console.print(f"Resuming: {len(existing_stay_ids)} stays already extracted\n")
        else:
            console.print("")

        # -------------------------------------------------------------------------
        # Step 1: Extract stays (static features)
        # -------------------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting ICU stays...", total=None)
            stays = self.extract_stays()
            progress.update(task, completed=True)

        # Filter by minimum stay length
        min_los_days = self.config.min_stay_hours / 24.0
        stays_filtered = stays.filter(pl.col("los_days") >= min_los_days)

        console.print(f"Found {len(stays)} ICU stays")
        console.print(
            f"After filtering (>={self.config.min_stay_hours}h): {len(stays_filtered)} stays"
        )

        # Filter out already-extracted stays if resuming
        if existing_stay_ids:
            stays_filtered = stays_filtered.filter(
                ~pl.col("stay_id").is_in(list(existing_stay_ids))
            )
            console.print(
                f"After excluding already-extracted: {len(stays_filtered)} stays to process"
            )

        stay_ids = stays_filtered["stay_id"].to_list()

        if len(stay_ids) == 0:
            if existing_stay_ids:
                console.print("[green]All stays already extracted. Nothing to do![/green]")
                return
            else:
                console.print("[red]Error: No stays remaining after filtering![/red]")
                return

        # Validate stays have required fields
        self._validate_stays(stays_filtered)

        # -------------------------------------------------------------------------
        # Step 2: Extract time-series features (batched for memory efficiency)
        # -------------------------------------------------------------------------
        batch_size = self.config.batch_size
        n_batches = (len(stay_ids) + batch_size - 1) // batch_size
        console.print(
            f"\n[bold]Extracting timeseries for {len(stay_ids)} stays "
            f"in {n_batches} batches (batch_size={batch_size})...[/bold]"
        )

        # Load feature mapping once (cached for reuse)
        feature_set = self.config.feature_set
        feature_mapping = self._get_feature_mapping_cached(feature_set)

        if not feature_mapping:
            raise ValueError(
                f"No features loaded from concept set '{feature_set}' for dataset "
                f"'{self._get_dataset_name()}'. Check that your YAML config has "
                f"entries for this dataset under each feature."
            )

        # Process stays in batches to avoid OOM
        sparse_batches: List[pl.DataFrame] = []
        feature_names: List[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing batches...", total=n_batches)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(stay_ids))
                batch_stay_ids = stay_ids[start_idx:end_idx]

                # Step 2a: Extract raw events for this batch
                raw_events = self._extract_raw_events(batch_stay_ids, feature_mapping)

                # Step 2b: Bin to hourly grid for this batch
                sparse_batch = self._bin_to_hourly_grid(raw_events, batch_stay_ids, feature_mapping)
                sparse_batches.append(sparse_batch)

                # Extract feature names from first non-empty batch
                if not feature_names and not sparse_batch.is_empty():
                    feature_names = [
                        col
                        for col in sparse_batch.columns
                        if col not in ["stay_id", "hour"] and not col.endswith("_mask")
                    ]

                progress.update(task, advance=1)

        # Concatenate all batches (use how="diagonal" to handle schema differences)
        # Different batches may have different columns present depending on data
        if sparse_batches:
            sparse_timeseries = pl.concat(sparse_batches, how="diagonal")
        else:
            # Empty result with expected schema
            sparse_timeseries = pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "hour": pl.Int32,
                }
            )

        # If no feature names were found, get them from feature mapping
        if not feature_names:
            feature_names = list(feature_mapping.keys())

        console.print(f"Extracted {len(feature_names)} features: {feature_names}")

        # Validate timeseries data
        self._validate_timeseries(sparse_timeseries, stay_ids, feature_names)

        # Convert to dense representation (also in batches)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Converting to dense timeseries...", total=None)
            dense_timeseries = self._create_dense_timeseries(
                sparse_timeseries, stays_filtered, feature_names
            )
            progress.update(task, completed=True)

        # -------------------------------------------------------------------------
        # Step 3: Extract labels
        # -------------------------------------------------------------------------
        task_configs = self._load_task_configs(self.config.tasks)
        task_names = [tc.task_name for tc in task_configs]

        if task_configs:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Extracting labels for {len(task_configs)} tasks...", total=None
                )
                labels = self.extract_labels(stay_ids, task_configs)
                progress.update(task, completed=True)

            console.print(f"Extracted labels for tasks: {task_names}")
        else:
            # No tasks configured - create DataFrame with just stay_ids
            labels = pl.DataFrame({"stay_id": stay_ids})
            console.print("[yellow]No task configs found - labels will be empty[/yellow]")

        # -------------------------------------------------------------------------
        # Step 4-5: Merge with existing data and save (with file locking)
        # -------------------------------------------------------------------------
        # Use file locking to prevent race conditions when merging and saving.
        # This protects against TOCTOU (time-of-check-to-time-of-use) race conditions
        # where two concurrent processes might both try to merge and write.
        static_path = self.output_dir / "static.parquet"
        timeseries_path = self.output_dir / "timeseries.parquet"
        labels_path = self.output_dir / "labels.parquet"

        with self._with_file_lock(static_path):
            # Re-check for existing data inside the lock to handle race conditions
            # Another process might have written new data while we were extracting
            existing_data_locked = self._check_existing_extraction()

            if existing_data_locked is not None:
                console.print("\n[bold]Merging with existing data...[/bold]")

                # Merge stays
                stays_filtered = pl.concat([existing_data_locked["stays"], stays_filtered])
                console.print(f"  ✓ Merged stays: {len(stays_filtered)} total")

                # Merge timeseries
                dense_timeseries = pl.concat([existing_data_locked["timeseries"], dense_timeseries])
                console.print(f"  ✓ Merged timeseries: {len(dense_timeseries)} total")

                # Merge labels
                labels = pl.concat([existing_data_locked["labels"], labels])
                console.print(f"  ✓ Merged labels: {len(labels)} total")

            # -------------------------------------------------------------------------
            # Step 5: Save to Parquet (atomic writes inside lock)
            # -------------------------------------------------------------------------
            console.print("\n[bold]Saving to Parquet...[/bold]")

            # Static features (demographics, admission info)
            self._atomic_write(static_path, lambda tmp: stays_filtered.write_parquet(tmp))
            console.print(f"  ✓ Static features: {static_path} ({len(stays_filtered)} stays)")

            # Time-series (dense format with masks)
            self._atomic_write(timeseries_path, lambda tmp: dense_timeseries.write_parquet(tmp))
            console.print(f"  ✓ Timeseries: {timeseries_path} ({len(dense_timeseries)} stays)")

            # Labels
            self._atomic_write(labels_path, lambda tmp: labels.write_parquet(tmp))
            console.print(f"  ✓ Labels: {labels_path} ({len(labels)} stays)")

            # Validate labels consistency
            self._validate_labels(labels, stay_ids)

            # Metadata (for ICUDataset to know feature names, etc.)
            # Written inside lock to ensure consistency with data files
            metadata = {
                "dataset": self._get_dataset_name(),
                "feature_set": self.config.feature_set,
                "categories": self.config.categories,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "seq_length_hours": self.config.seq_length_hours,
                "min_stay_hours": self.config.min_stay_hours,
                "task_names": task_names,
                "n_stays": len(stays_filtered),
                # Extraction configuration for reproducibility
                "extraction_config": {
                    "parquet_root": str(self.parquet_root),
                    "output_dir": str(self.output_dir),
                    "feature_set": self.config.feature_set,
                    "categories": self.config.categories,
                    "seq_length_hours": self.config.seq_length_hours,
                    "min_stay_hours": self.config.min_stay_hours,
                    "tasks": self.config.tasks,
                    "extraction_timestamp": datetime.now().isoformat(),
                },
            }

            metadata_path = self.output_dir / "metadata.yaml"

            def write_metadata(tmp: str) -> None:
                with open(tmp, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            self._atomic_write(metadata_path, write_metadata, suffix=".yaml")
            console.print(f"  ✓ Metadata: {metadata_path}")

        console.print("\n[bold green]Extraction complete![/bold green]")
