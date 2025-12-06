"""Abstract base class for ICU data extractors.

Reads from local Parquet files using DuckDB for efficient SQL queries.
Users specify parquet_root pointing to their local Parquet files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import polars as pl
from omegaconf import OmegaConf

from slices.data.tasks import TaskBuilder, TaskBuilderFactory, TaskConfig


@dataclass
class ExtractorConfig:
    """Configuration for data extraction.
    
    Users must specify parquet_root - the path to their local Parquet files.
    """

    parquet_root: str
    output_dir: str = "data/processed"
    seq_length_hours: int = 48
    feature_set: str = "core"  # core | extended
    concepts_dir: str | None = None  # Path to concepts directory (auto-detected if None)


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

        # Validate parquet directory exists
        if not self.parquet_root.exists():
            raise ValueError(f"Parquet directory not found: {self.parquet_root}")

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
        # Look for pyproject.toml as marker of project root
        current = Path.cwd()
        for parent in [current, *current.parents]:
            pyproject = parent / "pyproject.toml"
            concepts_dir = parent / "configs" / "concepts"
            if pyproject.exists() and concepts_dir.exists():
                return concepts_dir
        
        # Strategy 3: Relative to source file (development mode fallback)
        # From src/slices/data/extractors/mimic_iv.py -> configs/concepts
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

    def _load_feature_mapping(self, feature_set: str) -> Dict[str, Any]:
        """Load feature mapping from YAML config (generic).
        
        This method loads the concept YAML and extracts dataset-specific mappings.
        Subclasses don't need to override this unless they have special requirements.
        
        Args:
            feature_set: Name of feature set to load (e.g., 'core').
            
        Returns:
            Dictionary with dataset-specific feature configuration.
            Structure depends on dataset implementation.
            
        Raises:
            FileNotFoundError: If feature config file cannot be found.
        """
        config_path = self._get_concepts_path() / f"{feature_set}_features.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Feature config not found: {config_path}\n"
                f"Tried concepts_dir: {self._get_concepts_path()}\n"
                f"Hint: Set 'concepts_dir' in ExtractorConfig to point to your configs/concepts directory"
            )
        
        config = OmegaConf.load(config_path)
        dataset_name = self._get_dataset_name()
        
        # Extract dataset-specific feature configuration
        feature_mapping = {}
        
        # Process all feature categories (vitals, labs, etc.)
        for category in config:
            category_config = config[category]
            # OmegaConf objects may not be recognized by isinstance(dict)
            if category_config is not None and hasattr(category_config, 'items'):
                for feature_name, feature_config in category_config.items():
                    # Check if this feature has config for our dataset
                    if dataset_name in feature_config:
                        # Convert OmegaConf to dict for easier manipulation
                        feature_mapping[feature_name] = OmegaConf.to_container(
                            feature_config[dataset_name], resolve=True
                        )
        
        return feature_mapping

    def _bin_to_hourly_grid(
        self, 
        raw_events: pl.DataFrame, 
        stay_ids: List[int],
        feature_mapping: Dict[str, Any]
    ) -> pl.DataFrame:
        """Bin raw events to hourly grid and create observation masks (generic).
        
        This method is completely dataset-agnostic. It expects raw_events to have:
        - stay_id, charttime, feature_name, valuenum columns
        
        Args:
            raw_events: Raw events DataFrame with standardized schema.
            stay_ids: List of ICU stay IDs.
            feature_mapping: Feature mapping (only used to get expected feature names).
            
        Returns:
            Wide-format DataFrame with hourly bins and observation masks.
        """
        # Get stay info for intime calculation
        stays = self.extract_stays().filter(pl.col("stay_id").is_in(stay_ids))
        
        # Join with stays to get intime and calculate hour offset
        events_with_hours = (
            raw_events
            .join(stays.select(["stay_id", "intime"]), on="stay_id", how="left")
            .with_columns([
                # Calculate hour offset from ICU admission (floor to get hour bins)
                ((pl.col("charttime") - pl.col("intime")).dt.total_seconds() / 3600)
                .floor()
                .cast(pl.Int32)
                .alias("hour")
            ])
            # Filter out events before admission (hour < 0)
            .filter(pl.col("hour") >= 0)
        )
        
        # Aggregate by stay_id, hour, feature_name (mean for multiple values in same hour)
        # Also track that we observed at least one value
        aggregated = (
            events_with_hours
            .group_by(["stay_id", "hour", "feature_name"])
            .agg([
                pl.col("valuenum").mean().alias("value"),
                pl.lit(True).alias("observed")  # Mark as observed
            ])
        )
        
        # Pivot to wide format: one column per feature
        # First pivot the values
        pivoted_values = (
            aggregated
            .pivot(
                values="value",
                index=["stay_id", "hour"],
                columns="feature_name"
            )
        )
        
        # Pivot the observation masks
        pivoted_masks = (
            aggregated
            .pivot(
                values="observed",
                index=["stay_id", "hour"],
                columns="feature_name"
            )
        )
        
        # Rename mask columns to have _mask suffix (exclude index columns)
        mask_columns = {
            col: f"{col}_mask" 
            for col in pivoted_masks.columns 
            if col not in ["stay_id", "hour"]
        }
        pivoted_masks = pivoted_masks.rename(mask_columns)
        
        # Join values and masks
        result = pivoted_values.join(pivoted_masks, on=["stay_id", "hour"], how="left")
        
        # Ensure all expected feature columns exist (even if no data)
        expected_features = list(feature_mapping.keys())
        for feature in expected_features:
            # Add value column if missing
            if feature not in result.columns:
                result = result.with_columns([
                    pl.lit(None, dtype=pl.Float64).alias(feature)
                ])
            # Add mask column if missing
            mask_col = f"{feature}_mask"
            if mask_col not in result.columns:
                result = result.with_columns([
                    pl.lit(False).alias(mask_col)
                ])
        
        # Fill missing mask values with False (not observed)
        for col in result.columns:
            if col.endswith("_mask"):
                result = result.with_columns([
                    pl.col(col).fill_null(False)
                ])
        
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
        self,
        stay_ids: List[int],
        feature_mapping: Dict[str, Any]
    ) -> pl.DataFrame:
        """Extract raw time-series events for specified features (dataset-specific).
        
        This is the dataset-specific method that queries the appropriate tables
        and returns raw events. Each dataset implements this differently.
        
        Args:
            stay_ids: List of ICU stay IDs to extract.
            feature_mapping: Dataset-specific feature mapping from _load_feature_mapping.
            
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
        
        # Load feature mapping for this dataset
        feature_mapping = self._load_feature_mapping(feature_set)
        
        # Extract raw events (dataset-specific implementation)
        raw_events = self._extract_raw_events(stay_ids, feature_mapping)
        
        # Bin to hourly grid (generic)
        hourly_binned = self._bin_to_hourly_grid(raw_events, stay_ids, feature_mapping)
        
        return hourly_binned

    @abstractmethod
    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        """Extract raw data for a specific source (e.g., 'mortality_info', 'creatinine').
        
        This is a low-level method that extracts raw clinical data without computing labels.
        TaskBuilders call this to get the data they need.
        
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

    def extract_labels(
        self, 
        stay_ids: List[int], 
        task_configs: List[TaskConfig]
    ) -> pl.DataFrame:
        """Extract labels for multiple downstream tasks.
        
        This is a framework method that orchestrates label extraction:
        1. Identifies all required data sources from task configs
        2. Extracts raw data for each source (via extract_data_source)
        3. Uses TaskBuilders to compute labels from raw data
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
                raw_data["stays"] = self.extract_stays().filter(
                    pl.col("stay_id").is_in(stay_ids)
                )
            else:
                # Extract other data sources via abstract method
                raw_data[source] = self.extract_data_source(source, stay_ids)
        
        # Step 3: Build labels for each task
        all_labels = []
        for task_config in task_configs:
            # Create appropriate TaskBuilder
            builder: TaskBuilder = TaskBuilderFactory.create(task_config)
            
            # Compute labels
            task_labels = builder.build_labels(raw_data)
            
            # Rename 'label' column to task name
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


    def run(self) -> None:
        """Execute full extraction pipeline.
        
        Raises:
            NotImplementedError: Method not yet implemented.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stays = self.extract_stays()
        stay_ids = stays["stay_id"].to_list()

        timeseries = self.extract_timeseries(stay_ids)
        labels = self.extract_labels(stay_ids, [])

        # TODO: Combine and save to Parquet
        raise NotImplementedError

