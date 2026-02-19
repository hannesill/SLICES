#!/usr/bin/env python
"""Diagnose phenotype prevalence inflation by decomposing ICD-9 vs ICD-10 contributions.

Queries MIMIC-IV source parquet files directly to understand why observed phenotype
prevalence is higher than published MIMIC-III benchmark rates.

Checks:
1. ICD version distribution: What fraction of stays have ICD-9 vs ICD-10 diagnoses?
2. Per-version prevalence: Are ICD-10-coded stays driving the inflation?
3. Per-phenotype ICD code hit analysis: Which specific codes contribute most?

Example usage:
    uv run python scripts/validation/diagnose_phenotype_prevalence.py \
        --parquet-root /path/to/mimic-iv-parquet \
        --data-dir data/processed/mimic-iv
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import duckdb
import polars as pl
import yaml

# Harutyunyan et al. 2019 (MIMIC-III) rates — kept here for cross-dataset comparison.
HARUTYUNYAN_RATES: Dict[str, float] = {
    "sepsis": 0.15,
    "respiratory_failure": 0.18,
    "acute_renal_failure": 0.12,
    "chf": 0.15,
    "shock": 0.07,
    "chronic_kidney_disease": 0.10,
    "diabetes": 0.22,
    "copd": 0.10,
    "pneumonia": 0.14,
    "coronary_atherosclerosis": 0.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose phenotype prevalence by ICD version.",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        required=True,
        help="Path to MIMIC-IV parquet files (contains hosp/, icu/ subdirs).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to extracted data directory containing labels.parquet.",
    )
    parser.add_argument(
        "--phenotype-config",
        type=Path,
        default=None,
        help="Path to ccs_phenotypes.yaml (auto-detected if omitted).",
    )
    return parser.parse_args()


def find_phenotype_config(explicit: Path | None) -> Path:
    if explicit and explicit.exists():
        return explicit
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            candidate = parent / "src" / "slices" / "data" / "phenotypes" / "ccs_phenotypes.yaml"
            if candidate.exists():
                return candidate
            break
    raise FileNotFoundError("Could not locate ccs_phenotypes.yaml")


def load_definitions(config_path: Path) -> Dict[str, Dict]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_icd_maps(
    definitions: Dict[str, Dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    icd9_map: Dict[str, List[str]] = {}
    icd10_map: Dict[str, List[str]] = {}
    for name, defn in definitions.items():
        for code in defn.get("icd9_codes", []):
            icd9_map.setdefault(str(code), []).append(name)
        for code in defn.get("icd10_codes", []):
            icd10_map.setdefault(str(code), []).append(name)
    return icd9_map, icd10_map


def query_diagnoses(parquet_root: Path, stay_ids: Set[int]) -> pl.DataFrame:
    """Query raw diagnosis codes from MIMIC-IV parquet files."""
    icu_path = parquet_root / "icu" / "icustays.parquet"
    diag_path = parquet_root / "hosp" / "diagnoses_icd.parquet"

    if not icu_path.exists():
        raise FileNotFoundError(f"icustays.parquet not found: {icu_path}")
    if not diag_path.exists():
        raise FileNotFoundError(f"diagnoses_icd.parquet not found: {diag_path}")

    stay_ids_str = ",".join(map(str, stay_ids))
    con = duckdb.connect()
    result = con.sql(
        f"""
        SELECT i.stay_id, d.icd_code, d.icd_version
        FROM read_parquet('{icu_path}') AS i
        INNER JOIN read_parquet('{diag_path}') AS d ON i.hadm_id = d.hadm_id
        WHERE i.stay_id IN ({stay_ids_str})
    """
    ).pl()
    con.close()
    return result


def main() -> None:
    args = parse_args()

    # Load labels
    labels_path = args.data_dir / "labels.parquet"
    labels_df = pl.read_parquet(labels_path)
    phenotype_cols = [c for c in labels_df.columns if c.startswith("phenotyping_")]
    phenotype_names = [c.removeprefix("phenotyping_") for c in phenotype_cols]

    # Get valid stay IDs (non-null labels)
    valid_mask = labels_df[phenotype_cols[0]].is_not_null()
    valid_stays = labels_df.filter(valid_mask)
    valid_stay_ids = set(valid_stays["stay_id"].to_list())
    all_stay_ids = set(labels_df["stay_id"].to_list())

    print("=" * 90)
    print("Phenotype Prevalence Diagnostic")
    print("=" * 90)
    print(f"\n  Total stays:      {len(labels_df):,}")
    print(f"  Valid stays:      {len(valid_stays):,}")
    print(f"  Excluded (null):  {len(labels_df) - len(valid_stays):,}")

    # Load phenotype definitions
    config_path = find_phenotype_config(args.phenotype_config)
    definitions = load_definitions(config_path)
    icd9_map, icd10_map = build_icd_maps(definitions)

    print(f"\n  ICD-9 codes in config:  {len(icd9_map)}")
    print(f"  ICD-10 codes in config: {len(icd10_map)}")

    # Query raw diagnoses
    print("\n  Querying MIMIC-IV diagnosis tables...")
    diagnoses = query_diagnoses(args.parquet_root, all_stay_ids)
    print(f"  Total diagnosis rows: {len(diagnoses):,}")

    # -------------------------------------------------------------------------
    # Analysis 1: ICD version distribution per stay
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Analysis 1: ICD Version Distribution")
    print(f"{'=' * 90}\n")

    stay_versions: Dict[int, Set[int]] = defaultdict(set)
    for row in diagnoses.iter_rows(named=True):
        stay_versions[row["stay_id"]].add(row["icd_version"])

    icd9_only = sum(1 for v in stay_versions.values() if v == {9})
    icd10_only = sum(1 for v in stay_versions.values() if v == {10})
    both = sum(1 for v in stay_versions.values() if 9 in v and 10 in v)
    no_diag = len(all_stay_ids) - len(stay_versions)

    total = len(all_stay_ids)
    print(f"  ICD-9 only:        {icd9_only:>8,}  ({icd9_only/total:>6.1%})")
    print(f"  ICD-10 only:       {icd10_only:>8,}  ({icd10_only/total:>6.1%})")
    print(f"  Both ICD-9 & 10:   {both:>8,}  ({both/total:>6.1%})")
    print(f"  No diagnoses:      {no_diag:>8,}  ({no_diag/total:>6.1%})")

    # -------------------------------------------------------------------------
    # Analysis 2: Per-version prevalence for each phenotype
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Analysis 2: Prevalence by ICD Version (valid stays only)")
    print(f"{'=' * 90}\n")

    # Classify valid stays by dominant ICD version
    icd9_stays = {sid for sid in valid_stay_ids if stay_versions.get(sid, set()) == {9}}
    icd10_stays = {sid for sid in valid_stay_ids if stay_versions.get(sid, set()) == {10}}
    mixed_stays = {
        sid
        for sid in valid_stay_ids
        if 9 in stay_versions.get(sid, set()) and 10 in stay_versions.get(sid, set())
    }

    print(f"  Valid ICD-9-only stays:  {len(icd9_stays):,}")
    print(f"  Valid ICD-10-only stays: {len(icd10_stays):,}")
    print(f"  Valid mixed stays:       {len(mixed_stays):,}")
    print()

    # For each stay, compute phenotypes from ICD-9 codes only and ICD-10 codes only
    # This tells us what each code version contributes
    stay_pheno_icd9: Dict[int, Set[str]] = defaultdict(set)
    stay_pheno_icd10: Dict[int, Set[str]] = defaultdict(set)

    for row in diagnoses.iter_rows(named=True):
        sid = row["stay_id"]
        if sid not in valid_stay_ids:
            continue
        code = str(row["icd_code"]).strip()
        ver = row["icd_version"]
        if ver == 9:
            for name in icd9_map.get(code, []):
                stay_pheno_icd9[sid].add(name)
        elif ver == 10:
            for name in icd10_map.get(code, []):
                stay_pheno_icd10[sid].add(name)

    header = (
        f"  {'Phenotype':<28} {'MIMIC-III':>9} {'Overall':>9} "
        f"{'ICD9-only':>10} {'ICD10-only':>10} {'ICD9 contrib':>12} {'ICD10 contrib':>13}"
    )
    print(header)
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*12} {'-'*13}")

    n_valid = len(valid_stay_ids)
    n_icd9 = len(icd9_stays)
    n_icd10 = len(icd10_stays)

    for name in phenotype_names:
        col = f"phenotyping_{name}"
        non_null = valid_stays[col].drop_nulls()
        overall = (non_null == 1).sum() / len(non_null) if len(non_null) > 0 else 0.0

        # Prevalence among ICD-9-only stays
        pos_icd9 = sum(1 for sid in icd9_stays if name in stay_pheno_icd9.get(sid, set()))
        prev_icd9 = pos_icd9 / n_icd9 if n_icd9 > 0 else 0.0

        # Prevalence among ICD-10-only stays
        pos_icd10 = sum(1 for sid in icd10_stays if name in stay_pheno_icd10.get(sid, set()))
        prev_icd10 = pos_icd10 / n_icd10 if n_icd10 > 0 else 0.0

        # How many of ALL valid positives come from each version?
        all_pos_from_icd9 = sum(
            1 for sid in valid_stay_ids if name in stay_pheno_icd9.get(sid, set())
        )
        all_pos_from_icd10 = sum(
            1 for sid in valid_stay_ids if name in stay_pheno_icd10.get(sid, set())
        )

        pub = HARUTYUNYAN_RATES.get(name, 0.0)

        print(
            f"  {name:<28} {pub:>8.1%} {overall:>8.1%} "
            f"{prev_icd9:>9.1%} {prev_icd10:>9.1%}  "
            f"{all_pos_from_icd9:>7,}/{n_valid:<5} {all_pos_from_icd10:>7,}/{n_valid:<5}"
        )

    # -------------------------------------------------------------------------
    # Analysis 3: Top contributing ICD codes per phenotype
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Analysis 3: Top 5 Contributing ICD Codes Per Phenotype")
    print(f"{'=' * 90}")

    # Count how many valid stays each ICD code matches
    code_stay_counts: Dict[str, Dict[str, int]] = {
        name: defaultdict(int) for name in phenotype_names
    }

    for row in diagnoses.iter_rows(named=True):
        sid = row["stay_id"]
        if sid not in valid_stay_ids:
            continue
        code = str(row["icd_code"]).strip()
        ver = row["icd_version"]
        version_map = icd9_map if ver == 9 else icd10_map if ver == 10 else {}
        for name in version_map.get(code, []):
            code_stay_counts[name][f"v{ver}:{code}"] += 1

    for name in phenotype_names:
        counts = code_stay_counts[name]
        top_codes = sorted(counts.items(), key=lambda x: -x[1])[:5]
        print(f"\n  {name}:")
        for code_key, count in top_codes:
            print(f"    {code_key:<20} {count:>6,} diagnosis rows")

    print()


if __name__ == "__main__":
    main()
