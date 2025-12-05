"""MIMIC-IV data extractor using DuckDB on local Parquet files."""

from typing import List

import polars as pl

from .base import BaseExtractor, ExtractorConfig


class MIMICIVExtractor(BaseExtractor):
    """Extracts ICU data from MIMIC-IV Parquet files.
    
    This extractor reads from local Parquet files and provides both:
    1. Low-level data source extraction (mortality_info, creatinine, etc.)
    2. Time-series feature extraction for SSL pretraining
    """

    def extract_stays(self) -> pl.DataFrame:
        """Extract ICU stay metadata from MIMIC-IV.
        
        Returns:
            DataFrame with columns: stay_id, patient_id, hadm_id, intime, outtime,
            length_of_stay_days, age, gender, admission_type, first_careunit, 
            last_careunit.
        """
        icu_path = self._parquet_path("icu", "icustays")
        patients_path = self._parquet_path("hosp", "patients")
        admissions_path = self._parquet_path("hosp", "admissions")

        sql = f"""
        SELECT 
            -- Core identifiers & timing
            i.stay_id,
            i.subject_id AS patient_id,
            i.hadm_id,
            i.intime,
            i.outtime,
            i.los AS length_of_stay_days,
            -- Static features for modeling
            p.anchor_age AS age,
            p.gender,
            a.admission_type,
            i.first_careunit,
            i.last_careunit
        FROM 
            read_parquet('{icu_path}') AS i
        LEFT JOIN 
            read_parquet('{patients_path}') AS p
            ON i.subject_id = p.subject_id
        LEFT JOIN 
            read_parquet('{admissions_path}') AS a
            ON i.hadm_id = a.hadm_id
        ORDER BY 
            i.stay_id
        """

        return self._query(sql)

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract time-series features for given stays.
        
        TODO: Implement time-series extraction with hourly binning.
        
        Args:
            stay_ids: List of ICU stay IDs to extract.
            
        Returns:
            DataFrame with time-series data.
            
        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("Time-series extraction coming in Phase 2")

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        """Extract raw data for a specific source.
        
        Args:
            source_name: Name of data source (e.g., 'mortality_info').
            stay_ids: List of ICU stay IDs to extract.
            
        Returns:
            DataFrame with raw data for the specified source.
            
        Raises:
            ValueError: If source_name is not recognized.
        """
        # Dispatch to appropriate extraction method
        extraction_methods = {
            "mortality_info": self._extract_mortality_info,
            # TODO: Add more data sources as needed (creatinine, vasopressors, etc.)
        }

        if source_name not in extraction_methods:
            available = list(extraction_methods.keys())
            raise ValueError(
                f"Unknown data source '{source_name}'. Available sources: {available}"
            )

        return extraction_methods[source_name](stay_ids)

    # -------------------------------------------------------------------------
    # Private data source extraction methods
    # -------------------------------------------------------------------------

    def _extract_mortality_info(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract mortality-related information.
        
        Args:
            stay_ids: List of ICU stay IDs to extract.
        
        Returns:
            DataFrame with columns: stay_id, date_of_death, hospital_expire_flag,
            dischtime, discharge_location.
        """
        icu_path = self._parquet_path("icu", "icustays")
        patients_path = self._parquet_path("hosp", "patients")
        admissions_path = self._parquet_path("hosp", "admissions")

        stay_ids_str = ",".join(map(str, stay_ids))

        sql = f"""
        SELECT 
            i.stay_id,
            p.dod AS date_of_death,
            a.hospital_expire_flag,
            a.dischtime,
            a.discharge_location
        FROM 
            read_parquet('{icu_path}') AS i
        LEFT JOIN 
            read_parquet('{patients_path}') AS p
            ON i.subject_id = p.subject_id
        LEFT JOIN 
            read_parquet('{admissions_path}') AS a
            ON i.hadm_id = a.hadm_id
        WHERE 
            i.stay_id IN ({stay_ids_str})
        """

        return self._query(sql)