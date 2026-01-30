"""MIMIC-IV data extractor using DuckDB on local Parquet files."""

from typing import Dict, List, Optional

import polars as pl
from rich.console import Console

from slices.data.config_loader import load_static_concepts
from slices.data.config_schemas import ItemIDSource, StaticConceptConfig, TimeSeriesConceptConfig
from slices.data.value_transforms import get_transform

from .base import BaseExtractor

console = Console()


class MIMICIVExtractor(BaseExtractor):
    """Extracts ICU data from MIMIC-IV Parquet files.

    This extractor reads from local Parquet files and provides both:
    1. Low-level data source extraction (mortality_info, creatinine, etc.)
    2. Time-series feature extraction for SSL pretraining
    """

    # Expected schema for raw events extracted from MIMIC-IV sources
    EXPECTED_RAW_EVENTS_SCHEMA = {
        "stay_id": pl.Int64,
        "charttime": pl.Datetime,
        "feature_name": pl.Utf8,
        "valuenum": pl.Float64,
    }

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

    def _validate_raw_events_schema(self, df: pl.DataFrame) -> None:
        """Validate raw events DataFrame schema matches expected structure.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If schema doesn't match expectations.
        """
        if df.is_empty():
            # Empty DataFrames are valid
            return

        # Check that all expected columns exist
        for col in self.EXPECTED_RAW_EVENTS_SCHEMA:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in raw events. "
                    f"Expected columns: {list(self.EXPECTED_RAW_EVENTS_SCHEMA.keys())}, "
                    f"Got: {df.columns}"
                )

        # Check column types match expectations
        for col, expected_dtype in self.EXPECTED_RAW_EVENTS_SCHEMA.items():
            actual_dtype = df[col].dtype
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"Column '{col}' has type {actual_dtype}, expected {expected_dtype}. "
                    f"This may indicate a Polars version change or data source schema change."
                )

    def _extract_raw_events(
        self, stay_ids: List[int], feature_mapping: Dict[str, TimeSeriesConceptConfig]
    ) -> pl.DataFrame:
        """Extract raw events for all configured sources (MIMIC-IV specific).

        Optimized to batch all itemids for the same table into a single query,
        avoiding repeated scans of large Parquet files like chartevents.

        Args:
            stay_ids: List of ICU stay IDs to extract.
            feature_mapping: Dict mapping feature_name -> TimeSeriesConceptConfig.

        Returns:
            DataFrame with standardized schema:
                - stay_id: ICU stay identifier
                - charttime: Timestamp of observation
                - feature_name: Canonical feature name
                - valuenum: Numeric value
        """
        # Group sources by table for batched queries (avoid repeated Parquet scans)
        # Key: (table, value_col, time_col, transform) -> List of (itemids, feature_name)
        table_batches: Dict[tuple, List[tuple]] = {}

        for feature_name, config in feature_mapping.items():
            sources = config.mimic_iv
            if sources is None:
                continue

            for source in sources:
                if isinstance(source, ItemIDSource):
                    # Group by table and columns to batch itemids
                    key = (source.table, source.value_col, source.time_col, source.transform)
                    if key not in table_batches:
                        table_batches[key] = []
                    table_batches[key].append((source.itemid, feature_name))
                else:
                    console.print(
                        f"[yellow]Warning: Extraction type '{source.type}' not yet "
                        f"implemented for MIMIC-IV. Skipping source for {feature_name}[/yellow]"
                    )

        raw_event_batches: List[pl.DataFrame] = []

        # Execute one query per table (batched itemids)
        for (table, value_col, time_col, transform), items in table_batches.items():
            # Collect all itemids and build itemid->feature_name mapping
            all_itemids: List[int] = []
            itemid_to_feature: Dict[int, str] = {}
            for itemids, feature_name in items:
                for itemid in itemids:
                    all_itemids.append(itemid)
                    itemid_to_feature[itemid] = feature_name

            batch = self._extract_by_itemid_batch(
                table=table,
                value_col=value_col,
                time_col=time_col,
                transform=transform,
                itemids=all_itemids,
                itemid_to_feature=itemid_to_feature,
                stay_ids=stay_ids,
            )
            if not batch.is_empty():
                raw_event_batches.append(batch)

        if not raw_event_batches:
            result = pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "charttime": pl.Datetime,
                    "feature_name": pl.Utf8,
                    "valuenum": pl.Float64,
                }
            )
        else:
            result = pl.concat(raw_event_batches)

        # Validate schema before returning
        self._validate_raw_events_schema(result)
        return result

    def _extract_by_itemid(
        self,
        source: ItemIDSource,
        stay_ids: List[int],
        concept_name: str,
    ) -> pl.DataFrame:
        """Extract events using itemid matching.

        Args:
            source: ItemIDSource config with table, itemid, value_col, time_col, transform.
            stay_ids: List of ICU stay IDs to extract.
            concept_name: Name of the concept for the feature_name column.

        Returns:
            DataFrame with columns: stay_id, charttime, feature_name, valuenum.
        """
        # Map table names to schema paths
        table_to_path = {
            "chartevents": ("icu", "chartevents"),
            "labevents": ("hosp", "labevents"),
            "outputevents": ("icu", "outputevents"),
            "inputevents": ("icu", "inputevents"),
        }

        if source.table not in table_to_path:
            raise ValueError(
                f"Unsupported table '{source.table}' for MIMIC-IV extractor. "
                f"Supported: {list(table_to_path.keys())}"
            )

        schema, table = table_to_path[source.table]
        parquet_path = self._parquet_path(schema, table)
        stay_ids_str = ",".join(map(str, stay_ids))
        itemids_str = ",".join(map(str, source.itemid))

        # labevents doesn't have stay_id directly - needs join with icustays
        # CRITICAL: Filter labs to only those within the ICU stay time window
        # to prevent data leakage from labs drawn before admission or after discharge
        if source.table == "labevents":
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT
                i.stay_id,
                l.{source.time_col} AS charttime,
                l.itemid,
                CAST(l.{source.value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}') AS l
            INNER JOIN
                read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE
                i.stay_id IN ({stay_ids_str})
                AND l.itemid IN ({itemids_str})
                AND l.{source.value_col} IS NOT NULL
                AND l.{source.time_col} >= i.intime
                AND l.{source.time_col} <= i.outtime
            ORDER BY
                i.stay_id, l.{source.time_col}
            """
        else:
            # chartevents, outputevents, inputevents have stay_id directly
            # CRITICAL: Filter to only events within the ICU stay time window
            # to prevent data leakage from events recorded before/after the stay
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT
                c.stay_id,
                c.{source.time_col} AS charttime,
                c.itemid,
                CAST(c.{source.value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}') AS c
            INNER JOIN
                read_parquet('{icustays_path}') AS i
                ON c.stay_id = i.stay_id
            WHERE
                c.stay_id IN ({stay_ids_str})
                AND c.itemid IN ({itemids_str})
                AND c.{source.value_col} IS NOT NULL
                AND c.{source.time_col} >= i.intime
                AND c.{source.time_col} <= i.outtime
            ORDER BY
                c.stay_id, c.{source.time_col}
            """

        raw_events = self._query(sql)

        # Apply transform if specified
        if source.transform and not raw_events.is_empty():
            transform_func = get_transform(source.transform)
            # Try simple series transform first
            try:
                raw_events = raw_events.with_columns(
                    transform_func(pl.col("valuenum")).alias("valuenum")
                )
            except TypeError:
                # Fall back to DataFrame transform (e.g., to_celsius that checks itemid)
                raw_events = transform_func(raw_events, {"itemid": source.itemid})

        # Add feature name column
        if not raw_events.is_empty():
            raw_events = raw_events.with_columns(pl.lit(concept_name).alias("feature_name")).select(
                ["stay_id", "charttime", "feature_name", "valuenum"]
            )

        return raw_events

    def _extract_by_itemid_batch(
        self,
        table: str,
        value_col: str,
        time_col: str,
        transform: Optional[str],
        itemids: List[int],
        itemid_to_feature: Dict[int, str],
        stay_ids: List[int],
    ) -> pl.DataFrame:
        """Extract events for multiple itemids in a single query (optimized).

        This batches all itemids for a table into one query to avoid repeated
        Parquet file scans.

        Args:
            table: Source table name (chartevents, labevents, etc.).
            value_col: Column containing the value.
            time_col: Column containing the timestamp.
            transform: Optional transform to apply to values.
            itemids: List of all itemids to extract.
            itemid_to_feature: Mapping from itemid to feature name.
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with columns: stay_id, charttime, feature_name, valuenum.
        """
        # Map table names to schema paths
        table_to_path = {
            "chartevents": ("icu", "chartevents"),
            "labevents": ("hosp", "labevents"),
            "outputevents": ("icu", "outputevents"),
            "inputevents": ("icu", "inputevents"),
        }

        if table not in table_to_path:
            raise ValueError(
                f"Unsupported table '{table}' for MIMIC-IV extractor. "
                f"Supported: {list(table_to_path.keys())}"
            )

        schema, table_name = table_to_path[table]
        parquet_path = self._parquet_path(schema, table_name)
        stay_ids_str = ",".join(map(str, stay_ids))
        itemids_str = ",".join(map(str, itemids))

        # labevents needs join with icustays (doesn't have stay_id directly)
        if table == "labevents":
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT
                i.stay_id,
                l.{time_col} AS charttime,
                l.itemid,
                CAST(l.{value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}') AS l
            INNER JOIN
                read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE
                i.stay_id IN ({stay_ids_str})
                AND l.itemid IN ({itemids_str})
                AND l.{value_col} IS NOT NULL
                AND l.{time_col} >= i.intime
                AND l.{time_col} <= i.outtime
            ORDER BY
                i.stay_id, l.{time_col}
            """
        else:
            # chartevents, outputevents, inputevents have stay_id directly
            # CRITICAL: Filter to only events within the ICU stay time window
            # to prevent data leakage from events recorded before/after the stay
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT
                c.stay_id,
                c.{time_col} AS charttime,
                c.itemid,
                CAST(c.{value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}') AS c
            INNER JOIN
                read_parquet('{icustays_path}') AS i
                ON c.stay_id = i.stay_id
            WHERE
                c.stay_id IN ({stay_ids_str})
                AND c.itemid IN ({itemids_str})
                AND c.{value_col} IS NOT NULL
                AND c.{time_col} >= i.intime
                AND c.{time_col} <= i.outtime
            ORDER BY
                c.stay_id, c.{time_col}
            """

        raw_events = self._query(sql)

        if raw_events.is_empty():
            return pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "charttime": pl.Datetime,
                    "feature_name": pl.Utf8,
                    "valuenum": pl.Float64,
                }
            )

        # Apply transform if specified
        if transform:
            transform_func = get_transform(transform)
            try:
                raw_events = raw_events.with_columns(
                    transform_func(pl.col("valuenum")).alias("valuenum")
                )
            except TypeError:
                # Fall back to DataFrame transform
                raw_events = transform_func(raw_events, {"itemid": itemids})

        # Map itemid to feature_name using Polars replace
        raw_events = raw_events.with_columns(
            pl.col("itemid").replace_strict(itemid_to_feature, default=None).alias("feature_name")
        ).select(["stay_id", "charttime", "feature_name", "valuenum"])

        return raw_events

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
            "diagnoses": self._extract_diagnoses,
        }

        if source_name not in extraction_methods:
            available = list(extraction_methods.keys())
            raise ValueError(f"Unknown data source '{source_name}'. Available sources: {available}")

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

    def _extract_diagnoses(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract ICD diagnosis codes per ICU stay.

        Joins diagnoses_icd (per hadm_id) with icustays to get per-stay mapping.
        Returns a dataset-agnostic format that the PhenotypingLabelBuilder expects.

        Args:
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with columns: stay_id, icd_code, icd_version.
        """
        icu_path = self._parquet_path("icu", "icustays")
        diagnoses_path = self._parquet_path("hosp", "diagnoses_icd")

        stay_ids_str = ",".join(map(str, stay_ids))

        sql = f"""
        SELECT
            i.stay_id,
            d.icd_code,
            d.icd_version
        FROM
            read_parquet('{icu_path}') AS i
        INNER JOIN
            read_parquet('{diagnoses_path}') AS d
            ON i.hadm_id = d.hadm_id
        WHERE
            i.stay_id IN ({stay_ids_str})
        ORDER BY
            i.stay_id, d.seq_num
        """

        return self._query(sql)

    # -------------------------------------------------------------------------
    # Static feature extraction
    # -------------------------------------------------------------------------

    def extract_static_features(self, stay_ids: List[int]) -> pl.DataFrame:
        """Extract static/demographic features from MIMIC-IV.

        Uses the static.yaml config to extract features in a config-driven way.
        Supports two extraction patterns:
        1. Column extraction: Direct column from table (e.g., age from patients.anchor_age)
        2. Itemid extraction: First value from chartevents by itemid (e.g., height, weight)

        Args:
            stay_ids: List of ICU stay IDs to extract.

        Returns:
            DataFrame with one row per stay_id and one column per static feature.
        """
        # Load static concept configs
        concepts_dir = self._get_concepts_path()
        static_configs = load_static_concepts(concepts_dir, self._get_dataset_name())

        if not static_configs:
            console.print("[yellow]Warning: No static concepts found for MIMIC-IV[/yellow]")
            return pl.DataFrame({"stay_id": stay_ids})

        # Start with base stays data
        result = pl.DataFrame({"stay_id": stay_ids})

        # Group concepts by extraction type
        column_concepts: Dict[str, StaticConceptConfig] = {}
        itemid_concepts: Dict[str, StaticConceptConfig] = {}

        for name, config in static_configs.items():
            source = config.mimic_iv
            if source is None:
                continue

            if source.itemid is not None:
                itemid_concepts[name] = config
            else:
                column_concepts[name] = config

        # Extract column-based features
        if column_concepts:
            column_df = self._extract_static_columns(stay_ids, column_concepts)
            result = result.join(column_df, on="stay_id", how="left")

        # Extract itemid-based features (height, weight)
        if itemid_concepts:
            itemid_df = self._extract_static_by_itemid(stay_ids, itemid_concepts)
            result = result.join(itemid_df, on="stay_id", how="left")

        return result

    def _extract_static_columns(
        self,
        stay_ids: List[int],
        concepts: Dict[str, StaticConceptConfig],
    ) -> pl.DataFrame:
        """Extract static features from direct table columns.

        Args:
            stay_ids: List of ICU stay IDs.
            concepts: Dict mapping feature_name -> StaticConceptConfig with column-based sources.

        Returns:
            DataFrame with stay_id and extracted feature columns.
        """
        # Map table names to (schema, table_name) paths
        table_to_path = {
            "patients": ("hosp", "patients"),
            "admissions": ("hosp", "admissions"),
            "icustays": ("icu", "icustays"),
        }

        # Group concepts by source table
        table_concepts: Dict[str, Dict[str, str]] = {}  # table -> {feature_name: column}
        for name, config in concepts.items():
            source = config.mimic_iv
            if source is None:
                continue
            table_concepts.setdefault(source.table, {})[name] = source.column

        icu_path = self._parquet_path("icu", "icustays")
        patients_path = self._parquet_path("hosp", "patients")
        admissions_path = self._parquet_path("hosp", "admissions")

        stay_ids_str = ",".join(map(str, stay_ids))

        # Build SELECT clause for each table's columns
        select_parts = ["i.stay_id"]

        for table, columns in table_concepts.items():
            if table not in table_to_path:
                console.print(
                    f"[yellow]Warning: Unknown table '{table}' for static extraction[/yellow]"
                )
                continue

            alias = table[0]  # Use first letter as alias (p, a, i)
            if table == "patients":
                alias = "p"
            elif table == "admissions":
                alias = "a"
            elif table == "icustays":
                alias = "i"

            for feature_name, column in columns.items():
                select_parts.append(f"{alias}.{column} AS {feature_name}")

        sql = f"""
        SELECT
            {', '.join(select_parts)}
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

    def _extract_static_by_itemid(
        self,
        stay_ids: List[int],
        concepts: Dict[str, StaticConceptConfig],
    ) -> pl.DataFrame:
        """Extract static features from chartevents using itemid (e.g., height, weight).

        Takes the first recorded value per stay for each feature.

        Args:
            stay_ids: List of ICU stay IDs.
            concepts: Dict mapping feature_name -> StaticConceptConfig with itemid sources.

        Returns:
            DataFrame with stay_id and extracted feature columns.
        """
        chartevents_path = self._parquet_path("icu", "chartevents")
        stay_ids_str = ",".join(map(str, stay_ids))

        feature_dfs = []

        icustays_path = self._parquet_path("icu", "icustays")

        for feature_name, config in concepts.items():
            source = config.mimic_iv
            if source is None or source.itemid is None:
                continue

            # Get first value per stay for this itemid
            # CRITICAL: Filter to only values recorded during the ICU stay
            # to prevent using pre-admission or post-discharge values
            sql = f"""
            WITH ranked AS (
                SELECT
                    c.stay_id,
                    c.{source.column} AS value,
                    ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY c.charttime) AS rn
                FROM
                    read_parquet('{chartevents_path}') AS c
                INNER JOIN
                    read_parquet('{icustays_path}') AS i
                    ON c.stay_id = i.stay_id
                WHERE
                    c.stay_id IN ({stay_ids_str})
                    AND c.itemid = {source.itemid}
                    AND c.{source.column} IS NOT NULL
                    AND c.charttime >= i.intime
                    AND c.charttime <= i.outtime
            )
            SELECT
                stay_id,
                value AS {feature_name}
            FROM ranked
            WHERE rn = 1
            """

            df = self._query(sql)
            if not df.is_empty():
                feature_dfs.append(df)

        if not feature_dfs:
            return pl.DataFrame({"stay_id": stay_ids})

        # Join all feature dataframes
        result = pl.DataFrame({"stay_id": stay_ids})
        for df in feature_dfs:
            result = result.join(df, on="stay_id", how="left")

        return result
