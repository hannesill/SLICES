"""MIMIC-IV data extractor using DuckDB on local Parquet files."""

from typing import Any, Dict, List

import polars as pl

from slices.data.callbacks import get_callback
from .base import BaseExtractor, ExtractorConfig


class MIMICIVExtractor(BaseExtractor):
    """Extracts ICU data from MIMIC-IV Parquet files.
    
    This extractor reads from local Parquet files and provides both:
    1. Low-level data source extraction (mortality_info, creatinine, etc.)
    2. Time-series feature extraction for SSL pretraining
    """

    def _get_dataset_name(self) -> str:
        """Get dataset name for config parsing."""
        return "mimic_iv"

    def extract_stays(self) -> pl.DataFrame:
        """Extract ICU stay metadata from MIMIC-IV.
    
        Joins icustays, patients, and admissions tables to create a comprehensive
        stay-level DataFrame with demographics and admission context. This data is
        used for patient-level splits, cohort selection, evaluation stratification,
        and optionally as static features in downstream task models.
            
        Returns:
            DataFrame with columns:
                - stay_id (int): Unique ICU stay identifier
                - patient_id (int): Patient identifier (for patient-level splits)
                - hadm_id (int): Hospital admission identifier
                - intime (datetime): ICU admission timestamp
                - outtime (datetime): ICU discharge timestamp  
                - los_days (float): ICU length of stay in days
                - age (int): Patient age at admission (anchor_age from patients)
                - gender (str): Patient gender ('M' or 'F')
                - race (str): Patient race/ethnicity
                - admission_type (str): Type of hospital admission (e.g., 'EMERGENCY', 'ELECTIVE')
                - admission_location (str): Location patient was admitted from
                - insurance (str): Insurance type (e.g., 'Medicare', 'Private')
                - first_careunit (str): First ICU care unit
                - last_careunit (str): Last ICU care unit
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
            i.los AS los_days,
            
            -- Demographics (for fairness analysis & modeling)
            p.anchor_age AS age,
            p.gender,
            
            -- Admission context (clinically relevant)
            a.race,
            a.admission_type,
            a.admission_location,
            a.insurance,
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

    def _extract_raw_events(
        self,
        stay_ids: List[int],
        feature_mapping: Dict[str, Any]
    ) -> pl.DataFrame:
        """Extract raw events for all configured sources (MIMIC-IV specific).

        Args:
            stay_ids: List of ICU stay IDs to extract.
            feature_mapping: Dict mapping feature_name -> mimic_iv config
                            (with 'source', 'itemid', 'value_col' keys).
            
        Returns:
            DataFrame with standardized schema:
                - stay_id: ICU stay identifier
                - charttime: Timestamp of observation
                - feature_name: Canonical feature name
                - valuenum: Numeric value
        """
        # Group features by source (chartevents, labevents, outputevents, ...)
        features_by_source: Dict[str, list[tuple[str, Dict[str, Any]]]] = {}
        for feature_name, config in feature_mapping.items():
            source = config.get("source")
            if source is None:
                continue
            features_by_source.setdefault(source, []).append((feature_name, config))

        raw_event_batches: List[pl.DataFrame] = []
        for source, features in features_by_source.items():
            raw_event_batches.append(
                self._extract_events_for_source(source=source, stay_ids=stay_ids, features=features)
            )

        if not raw_event_batches:
            return pl.DataFrame({
                "stay_id": [],
                "charttime": [],
                "feature_name": [],
                "valuenum": []
            })

        return pl.concat(raw_event_batches)
        

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

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_events_for_source(
        self,
        source: str,
        stay_ids: List[int],
        features: list[tuple[str, Dict[str, Any]]],
    ) -> pl.DataFrame:
        """Extract events for a single source (chartevents, labevents, outputevents)."""
        source_to_table = {
            "chartevents": ("icu", "chartevents", "charttime"),
            "labevents": ("hosp", "labevents", "charttime"),
            "outputevents": ("icu", "outputevents", "charttime"),
        }

        if source not in source_to_table:
            raise ValueError(f"Unsupported source '{source}' for MIMIC-IV extractor")

        schema, table, time_col = source_to_table[source]

        itemid_to_feature: Dict[int, str] = {}
        all_itemids: List[int] = []
        value_cols = set()

        for feature_name, config in features:
            itemids = config.get("itemid", [])
            if isinstance(itemids, int):
                itemids = [itemids]

            value_cols.add(config.get("value_col", "valuenum"))

            for itemid in itemids:
                itemid_to_feature[itemid] = feature_name
                all_itemids.append(itemid)

        if not all_itemids:
            return pl.DataFrame({
                "stay_id": [],
                "charttime": [],
                "feature_name": [],
                "valuenum": []
            })

        if len(value_cols) > 1:
            raise ValueError(
                f"Mixed value columns for source '{source}': {value_cols}. "
                "Configure a consistent value_col per source."
            )

        value_col = next(iter(value_cols))
        parquet_path = self._parquet_path(schema, table)
        stay_ids_str = ",".join(map(str, stay_ids))
        itemids_str = ",".join(map(str, all_itemids))

        # labevents doesn't have stay_id directly - needs join with icustays
        if source == "labevents":
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT 
                i.stay_id,
                l.{time_col} AS charttime,
                l.itemid,
                l.{value_col} AS valuenum
            FROM 
                read_parquet('{parquet_path}') AS l
            INNER JOIN 
                read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE 
                i.stay_id IN ({stay_ids_str})
                AND l.itemid IN ({itemids_str})
                AND l.{value_col} IS NOT NULL
            ORDER BY 
                i.stay_id, l.{time_col}
            """
        else:
            # chartevents and outputevents have stay_id directly
            sql = f"""
            SELECT 
                stay_id,
                {time_col} AS charttime,
                itemid,
                {value_col} AS valuenum
            FROM 
                read_parquet('{parquet_path}')
            WHERE 
                stay_id IN ({stay_ids_str})
                AND itemid IN ({itemids_str})
                AND {value_col} IS NOT NULL
            ORDER BY 
                stay_id, {time_col}
            """

        raw_events = self._query(sql)

        # Apply callbacks before mapping itemids to feature names
        for feature_name, config in features:
            if "transform" not in config:
                continue

            itemids = config.get("itemid", [])
            if isinstance(itemids, int):
                itemids = [itemids]

            callback_name = config["transform"]
            callback_func = get_callback(callback_name)

            feature_mask = pl.col("itemid").is_in(itemids)
            feature_rows = raw_events.filter(feature_mask)

            if len(feature_rows) > 0:
                transformed = callback_func(feature_rows, config)
                raw_events = pl.concat([
                    raw_events.filter(~feature_mask),
                    transformed,
                ]).sort(["stay_id", "charttime"])

        # Map itemid to feature_name using join (robust across Polars versions)
        itemid_mapping_df = pl.DataFrame({
            "itemid": list(itemid_to_feature.keys()),
            "feature_name": list(itemid_to_feature.values())
        })

        return (
            raw_events
            .join(itemid_mapping_df, on="itemid", how="inner")
            .select(["stay_id", "charttime", "feature_name", "valuenum"])
        )