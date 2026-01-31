"""Multi-label phenotyping label builder.

Maps ICD diagnosis codes to 10 acute care phenotypes using HCUP CCS groups.
Follows the MIMIC-III benchmark (Harutyunyan et al. 2019) design.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import polars as pl
import yaml

from .base import LabelBuilder

logger = logging.getLogger(__name__)


def _load_phenotype_definitions(config_path: Path) -> Dict[str, Dict]:
    """Load phenotype definitions from YAML config.

    Args:
        config_path: Path to ccs_phenotypes.yaml.

    Returns:
        Dictionary mapping phenotype name to definition dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Phenotype config not found: {config_path}. "
            "Ensure phenotypes/ccs_phenotypes.yaml exists in the package data directory."
        )

    with open(config_path) as f:
        definitions = yaml.safe_load(f)

    return definitions


def _build_icd_to_phenotype_map(
    definitions: Dict[str, Dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build ICD code to phenotype name lookup maps.

    Creates separate maps for ICD-9 and ICD-10 codes.

    Args:
        definitions: Phenotype definitions from YAML.

    Returns:
        Tuple of (icd9_map, icd10_map) where each maps
        icd_code -> list of phenotype names.
    """
    icd9_map: Dict[str, List[str]] = {}
    icd10_map: Dict[str, List[str]] = {}

    for phenotype_name, defn in definitions.items():
        for code in defn.get("icd9_codes", []):
            code_str = str(code)
            icd9_map.setdefault(code_str, []).append(phenotype_name)

        for code in defn.get("icd10_codes", []):
            code_str = str(code)
            icd10_map.setdefault(code_str, []).append(phenotype_name)

    return icd9_map, icd10_map


def _find_phenotype_config(label_params: Dict) -> Path:
    """Locate the phenotype config file.

    Resolves the path relative to the package data directory
    (``src/slices/data/``).

    Args:
        label_params: Task label_params dict containing 'phenotype_config'.

    Returns:
        Resolved path to the phenotype config YAML.

    Raises:
        FileNotFoundError: If config cannot be found.
    """
    relative_path = label_params.get("phenotype_config", "phenotypes/ccs_phenotypes.yaml")

    # Resolve relative to package data directory (src/slices/data/)
    package_data_dir = Path(__file__).parent.parent
    resolved = package_data_dir / relative_path
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Could not locate phenotype config '{relative_path}' "
        f"in package data directory ({package_data_dir})."
    )


class PhenotypingLabelBuilder(LabelBuilder):
    """Build multi-label phenotyping labels from ICD diagnosis codes.

    Maps discharge ICD-9 and ICD-10 codes to 10 acute care phenotypes
    using HCUP CCS (Clinical Classifications Software) groups.

    Expected raw_data:
    - 'stays': stay_id, hadm_id, intime, outtime
    - 'diagnoses': stay_id, icd_code, icd_version

    Returns DataFrame with stay_id + one column per phenotype (0/1).
    Columns are named phenotyping_{phenotype_name}.
    """

    def build_labels(self, raw_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Build multi-label phenotyping labels.

        Args:
            raw_data: Dictionary with 'stays' and 'diagnoses' DataFrames.

        Returns:
            DataFrame with stay_id and one int column per phenotype (0 or 1).
            Null for excluded stays (multi-stay admissions if configured).
        """
        self.validate_inputs(raw_data)

        stays = raw_data["stays"]
        diagnoses = raw_data["diagnoses"]

        if len(stays) == 0:
            return self._empty_result()

        # Load phenotype definitions
        config_path = _find_phenotype_config(self.config.label_params)
        definitions = _load_phenotype_definitions(config_path)
        phenotype_names = list(definitions.keys())

        # Build ICD-to-phenotype lookup
        icd9_map, icd10_map = _build_icd_to_phenotype_map(definitions)

        # Identify multi-stay admissions to exclude
        excluded_stays = self._get_excluded_stays(stays)

        # Map diagnoses to phenotypes
        labels = self._map_diagnoses_to_phenotypes(
            stays, diagnoses, phenotype_names, icd9_map, icd10_map, excluded_stays
        )

        self._log_prevalence(labels, phenotype_names)

        return labels

    def _empty_result(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        schema = {"stay_id": pl.Int64}
        for name in self.config.class_names or []:
            schema[f"phenotyping_{name}"] = pl.Int32
        return pl.DataFrame(schema=schema)

    def _get_excluded_stays(self, stays: pl.DataFrame) -> Set[int]:
        """Identify stays to exclude (multi-stay admissions).

        If exclude_multi_stay_admissions is enabled, finds hadm_ids
        with multiple ICU stays and excludes all of them.

        Args:
            stays: Stays DataFrame with stay_id and hadm_id.

        Returns:
            Set of stay_ids to exclude.
        """
        exclude_multi = self.config.label_params.get("exclude_multi_stay_admissions", True)

        if not exclude_multi or "hadm_id" not in stays.columns:
            return set()

        # Find hadm_ids with multiple ICU stays
        multi_stay_hadms = (
            stays.group_by("hadm_id")
            .agg(pl.col("stay_id").count().alias("n_stays"))
            .filter(pl.col("n_stays") > 1)
            .select("hadm_id")
        )

        if len(multi_stay_hadms) == 0:
            return set()

        # Get all stay_ids for those hadm_ids
        excluded = stays.join(multi_stay_hadms, on="hadm_id", how="inner")
        excluded_ids = set(excluded["stay_id"].to_list())

        logger.info(
            f"Excluding {len(excluded_ids)} stays from {len(multi_stay_hadms)} "
            "multi-stay admissions"
        )

        return excluded_ids

    def _map_diagnoses_to_phenotypes(
        self,
        stays: pl.DataFrame,
        diagnoses: pl.DataFrame,
        phenotype_names: List[str],
        icd9_map: Dict[str, List[str]],
        icd10_map: Dict[str, List[str]],
        excluded_stays: Set[int],
    ) -> pl.DataFrame:
        """Map ICD diagnosis codes to phenotype binary labels.

        Uses vectorized Polars joins instead of row-by-row iteration.

        Args:
            stays: Stays DataFrame.
            diagnoses: Diagnoses DataFrame with stay_id, icd_code, icd_version.
            phenotype_names: Ordered list of phenotype names.
            icd9_map: ICD-9 code to phenotype name mapping.
            icd10_map: ICD-10 code to phenotype name mapping.
            excluded_stays: Set of stay_ids to set to null.

        Returns:
            DataFrame with stay_id and one column per phenotype.
        """
        all_stay_ids = stays["stay_id"].unique()

        # Build a lookup DataFrame from both ICD maps: (icd_code, icd_version, phenotype)
        lookup_rows: List[Dict[str, object]] = []
        for code, phenos in icd9_map.items():
            for p in phenos:
                lookup_rows.append({"icd_code": code, "icd_version": 9, "phenotype": p})
        for code, phenos in icd10_map.items():
            for p in phenos:
                lookup_rows.append({"icd_code": code, "icd_version": 10, "phenotype": p})

        if len(diagnoses) > 0 and "icd_code" in diagnoses.columns and lookup_rows:
            lookup_df = pl.DataFrame(lookup_rows)

            # Normalize icd_code in diagnoses to stripped strings for join
            diag_norm = diagnoses.with_columns(
                pl.col("icd_code").cast(pl.Utf8).str.strip_chars().alias("icd_code")
            )

            # Join diagnoses with lookup on (icd_code, icd_version) — hash join
            matched = diag_norm.join(lookup_df, on=["icd_code", "icd_version"], how="inner")

            # Deduplicate: one row per (stay_id, phenotype)
            matched = matched.select("stay_id", "phenotype").unique()

            # Add indicator column and pivot to wide format
            matched = matched.with_columns(pl.lit(1).alias("value"))
            pivoted = matched.pivot(on="phenotype", index="stay_id", values="value")
        else:
            # No diagnoses to process — start with empty pivoted frame
            pivoted = pl.DataFrame({"stay_id": pl.Series([], dtype=all_stay_ids.dtype)})

        # Left-join with all stay_ids to ensure every stay is present
        base = pl.DataFrame({"stay_id": all_stay_ids})
        result = base.join(pivoted, on="stay_id", how="left")

        # Ensure all phenotype columns exist (fill missing phenotypes with 0)
        for name in phenotype_names:
            if name not in result.columns:
                result = result.with_columns(pl.lit(None, dtype=pl.Int64).alias(name))

        # Fill nulls with 0 for phenotype columns, then rename to prefixed form
        rename_map: Dict[str, str] = {}
        fill_exprs = [pl.col("stay_id")]
        for name in phenotype_names:
            col_name = f"phenotyping_{name}"
            fill_exprs.append(pl.col(name).fill_null(0).alias(col_name))
            rename_map[name] = col_name

        result = result.select(fill_exprs)

        # Set excluded stays to null
        if excluded_stays:
            excluded_list = list(excluded_stays)
            is_excluded = pl.col("stay_id").is_in(excluded_list)
            null_exprs = [pl.col("stay_id")]
            for name in phenotype_names:
                col_name = f"phenotyping_{name}"
                null_exprs.append(
                    pl.when(is_excluded).then(None).otherwise(pl.col(col_name)).alias(col_name)
                )
            result = result.select(null_exprs)

        # Cast columns to correct types
        cast_exprs = [pl.col("stay_id").cast(pl.Int64)]
        for name in phenotype_names:
            col_name = f"phenotyping_{name}"
            cast_exprs.append(pl.col(col_name).cast(pl.Int32))
        result = result.select(cast_exprs)

        return result

    def _log_prevalence(self, labels: pl.DataFrame, phenotype_names: List[str]) -> None:
        """Log phenotype prevalence for sanity checking.

        Args:
            labels: Labels DataFrame with phenotype columns.
            phenotype_names: List of phenotype names.
        """
        total = len(labels)
        if total == 0:
            return

        prevalences = []
        for name in phenotype_names:
            col = f"phenotyping_{name}"
            if col in labels.columns:
                non_null = labels[col].drop_nulls()
                positive = (non_null == 1).sum()
                prev = positive / len(non_null) if len(non_null) > 0 else 0.0
                prevalences.append(f"{name}: {prev:.1%}")

        logger.info(f"Phenotype prevalence: {', '.join(prevalences)}")
