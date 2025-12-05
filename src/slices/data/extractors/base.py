"""Abstract base class for ICU data extractors.

Reads from local Parquet files using DuckDB for efficient SQL queries.
Users specify parquet_root pointing to their local Parquet files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb
import polars as pl

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
    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract time-series features for given stays.
        
        Args:
            stay_ids: List of ICU stay IDs to extract.
            
        Returns:
            DataFrame with time-series data.
        """
        pass

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

    def bin_to_hourly(self, df: pl.DataFrame, stay_info: pl.DataFrame) -> pl.DataFrame:
        """Bin raw events to hourly resolution.
        
        Args:
            df: Raw event DataFrame.
            stay_info: DataFrame with stay metadata (stay_id, intime, etc.).
            
        Returns:
            DataFrame with hourly-binned data.
            
        Raises:
            NotImplementedError: Method not yet implemented.
        """
        # TODO: Implement hourly binning logic
        raise NotImplementedError

    def run(self) -> None:
        """Execute full extraction pipeline.
        
        Raises:
            NotImplementedError: Method not yet implemented.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stays = self.extract_stays()
        stay_ids = stays["stay_id"].to_list()

        timeseries = self.extract_timeseries(stay_ids)
        labels = self.extract_labels(stay_ids)

        # TODO: Combine and save to Parquet
        raise NotImplementedError

