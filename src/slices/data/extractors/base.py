"""Abstract base class for ICU data extractors.

Reads from local Parquet files using DuckDB for efficient SQL queries.
Users only need to specify data_dir pointing to their local data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import duckdb
import polars as pl


@dataclass
class ExtractorConfig:
    """Configuration for data extraction.
    
    Users only need to specify data_dir - the path to their local Parquet files.
    """

    data_dir: str  # Path to raw MIMIC-IV (or other) Parquet files
    output_dir: str = "data/processed"
    seq_length_hours: int = 48
    feature_set: str = "core"  # core | extended


class BaseExtractor(ABC):
    """Abstract base class for ICU data extractors.
    
    Reads from local Parquet files using DuckDB for efficient SQL queries.
    Users only need to specify data_dir pointing to their local data.
    """

    def __init__(self, config: ExtractorConfig) -> None:
        """Initialize extractor with configuration.
        
        Args:
            config: Extractor configuration containing data_dir and other settings.
            
        Raises:
            ValueError: If data directory does not exist.
        """
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.conn = duckdb.connect()  # In-memory DuckDB for queries

        # Validate data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

    def _query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query on local Parquet files and return Polars DataFrame.
        
        Args:
            sql: SQL query string.
            
        Returns:
            Polars DataFrame with query results.
        """
        return self.conn.execute(sql).pl()

    def _parquet_path(self, schema: str, table: str) -> Path:
        """Get path to a Parquet file: {data_dir}/{schema}/{table}.parquet.
        
        Args:
            schema: Schema/directory name (e.g., 'hosp', 'icu').
            table: Table name (e.g., 'patients', 'chartevents').
            
        Returns:
            Path to the Parquet file.
        """
        return self.data_dir / schema / f"{table}.parquet"

    @abstractmethod
    def extract_stays(self) -> pl.DataFrame:
        """Extract ICU stay metadata (stay_id, patient_id, times).
        
        Returns:
            DataFrame with columns: stay_id, patient_id, intime, outtime, etc.
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
    def extract_labels(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract task labels for given stays.
        
        Args:
            stay_ids: List of ICU stay IDs to extract.
            
        Returns:
            DataFrame with task labels.
        """
        pass

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

