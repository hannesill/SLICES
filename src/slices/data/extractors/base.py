"""Abstract base class for ICU data extractors.

Provides shared infrastructure for extraction pipelines: configuration,
validation, label extraction, dense timeseries conversion, atomic file I/O,
and file locking.  Concrete subclasses (e.g. ``RicuExtractor``) implement
dataset-specific loading and the ``run()`` entry point.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, Generator, List, Optional

import polars as pl
import portalocker
import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

from slices.constants import EXTRACTION_BATCH_SIZE, MIN_STAY_HOURS, SEQ_LENGTH_HOURS
from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig
from slices.data.utils import get_package_data_dir

console = Console()


class ExtractorConfig(BaseModel):
    """Configuration for data extraction using Pydantic validation.

    Users must specify parquet_root - the path to their local Parquet files.

    Attributes:
        parquet_root: Path to directory containing Parquet files (required).
        output_dir: Directory to write extracted data to.
        seq_length_hours: Length of time-series sequences in hours (must be > 0).
        feature_set: Feature set to extract ('core' or 'extended').
        tasks_dir: Path to task config directory (auto-detected if None).
        tasks: List of task names to extract labels for.
        min_stay_hours: Minimum ICU stay length to include (must be >= 0).
        batch_size: Number of stays to process in each batch (for memory efficiency).
    """

    parquet_root: str
    output_dir: str = "data/processed"
    seq_length_hours: int = Field(default=SEQ_LENGTH_HOURS, gt=0)
    feature_set: str = "core"  # core | extended
    tasks_dir: Optional[str] = None
    tasks: List[str] = Field(default_factory=lambda: ["mortality_24h", "mortality_hospital"])
    min_stay_hours: int = Field(default=MIN_STAY_HOURS, ge=0)
    batch_size: int = Field(default=EXTRACTION_BATCH_SIZE, gt=0)
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

    Provides shared utilities for validation, label extraction, dense
    timeseries conversion, resume logic, atomic file I/O, and file locking.
    Subclasses implement dataset-specific data loading and ``run()``.
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

        # Validate parquet directory exists
        if not self.parquet_root.exists():
            raise ValueError(f"Parquet directory not found: {self.parquet_root}")

    def _get_tasks_path(self) -> Path:
        """Get path to tasks directory.

        Returns:
            Path to tasks directory containing task YAML files.

        Raises:
            FileNotFoundError: If tasks directory cannot be found.
        """
        if self.config.tasks_dir is not None:
            tasks_path = Path(self.config.tasks_dir)
            if tasks_path.exists():
                return tasks_path
            raise FileNotFoundError(f"Tasks directory specified in config not found: {tasks_path}")

        return get_package_data_dir() / "tasks"

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

    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for this extractor.

        Returns:
            Dataset name (e.g., 'mimic_iv', 'eicu', 'hirid').
        """
        pass

    @abstractmethod
    def extract_stays(self) -> pl.DataFrame:
        """Extract ICU stay metadata (stay_id, patient_id, times).

        Returns:
            DataFrame with columns: stay_id, patient_id, intime, outtime,
            los_days, age, gender, etc.
        """
        pass

    @abstractmethod
    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract time-series features for given stays.

        Args:
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with columns:
                - stay_id: ICU stay identifier
                - hour: Hour offset from ICU admission (0, 1, 2, ...)
                - {feature_name}: Value for each feature
                - {feature_name}_mask: Boolean indicating if value was observed
        """
        pass

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

        Uses vectorized Polars operations instead of Python loops for performance.

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
        mask_names = [f"{f}_mask" for f in feature_names]

        # Use stay_ids from stays DataFrame (not sparse_timeseries) to ensure
        # ALL requested stays are included, even those with no observations.
        stay_ids = stays["stay_id"].unique().to_list()

        # Track which stays have data for logging
        stays_with_data = set(sparse_timeseries["stay_id"].unique().to_list())
        stays_without_data = set(stay_ids) - stays_with_data
        if stays_without_data:
            console.print(
                f"[yellow]Warning: {len(stays_without_data)} stays have no timeseries "
                f"observations. They will have all-NaN/all-False arrays.[/yellow]"
            )

        # Log overflow stats
        overflow = sparse_timeseries.filter(pl.col("hour") >= seq_length)
        if len(overflow) > 0:
            max_hour = int(overflow["hour"].max())  # type: ignore[arg-type]
            console.print(
                f"[yellow]Warning: Discarded {len(overflow)} data points beyond "
                f"seq_length={seq_length} (max hour observed: {max_hour}). "
                f"Consider increasing seq_length_hours if late observations are important.[/yellow]"
            )

        # Filter to valid hours only
        valid = sparse_timeseries.filter((pl.col("hour") >= 0) & (pl.col("hour") < seq_length))

        # Build the full grid: every (stay_id, hour) combination
        all_stays = pl.DataFrame({"stay_id": stay_ids})
        all_hours = pl.DataFrame({"hour": list(range(seq_length))})
        grid = all_stays.join(all_hours, how="cross")

        # Ensure mask columns exist in valid (default False), cast feature cols to Float64
        for f in feature_names:
            if f not in valid.columns:
                valid = valid.with_columns(pl.lit(None).cast(pl.Float64).alias(f))
            else:
                valid = valid.with_columns(pl.col(f).cast(pl.Float64))
            m = f"{f}_mask"
            if m not in valid.columns:
                valid = valid.with_columns(pl.lit(False).alias(m))

        # Select only the columns we need for the join
        valid = valid.select(["stay_id", "hour"] + feature_names + mask_names)

        # Left join grid with valid data
        dense = grid.join(valid, on=["stay_id", "hour"], how="left")

        # Fill nulls: NaN for value columns, False for mask columns
        dense = dense.with_columns(
            [pl.col(f).fill_null(float("nan")).cast(pl.Float64) for f in feature_names]
            + [pl.col(m).fill_null(False).cast(pl.Boolean) for m in mask_names]
        )

        # Apply mask: where mask is False, set value to NaN
        dense = dense.with_columns(
            [
                pl.when(pl.col(mask_names[i]))
                .then(pl.col(feature_names[i]))
                .otherwise(float("nan"))
                .alias(feature_names[i])
                for i in range(n_features)
            ]
        )

        # Sort to guarantee consistent ordering
        dense = dense.sort(["stay_id", "hour"])

        # Build per-hour feature/mask vectors as lists, then group by stay_id.
        # This avoids map_elements/Python lambdas for a fully vectorized path.
        # For each (stay_id, hour) row, create a list [feat0, feat1, ...] and
        # [mask0, mask1, ...], then group by stay_id to get list-of-lists.
        dense = dense.with_columns(
            pl.concat_list([pl.col(f) for f in feature_names]).alias("_ts_row"),
            pl.concat_list([pl.col(m).cast(pl.Boolean) for m in mask_names]).alias("_mask_row"),
        )

        result = dense.group_by("stay_id", maintain_order=True).agg(
            pl.col("_ts_row").alias("timeseries"),
            pl.col("_mask_row").alias("mask"),
        )

        return result

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

    @abstractmethod
    def run(self) -> None:
        """Execute the full extraction pipeline.

        Subclasses implement this to extract and save:
        - static.parquet: Stay-level metadata (demographics, admission info)
        - timeseries.parquet: Dense hourly features with observation masks
        - labels.parquet: Task labels for downstream prediction
        - metadata.yaml: Extraction metadata (feature names, task names, etc.)
        """
        pass
