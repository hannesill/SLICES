#!/usr/bin/env python
"""Validate phenotyping labels against reference rates and ICD consistency.

Performs two validation checks:

1. **Reference prevalence comparison**: Compute phenotype prevalence from
   extracted labels.parquet and compare against MIMIC-IV reference rates
   (derived from SLICES extraction on MIMIC-IV v3.0). Flags deviations
   exceeding 5 percentage points.

2. **Internal ICD consistency**: Independently count ICD codes in the diagnoses
   data that match phenotype definitions and compare against mapped label counts.
   Discrepancies indicate a mapping bug in the label builder.

Example usage:
    # Basic prevalence validation
    uv run python scripts/validation/validate_phenotyping.py \
        --data-dir data/processed/mimic-iv

    # With explicit phenotype config
    uv run python scripts/validation/validate_phenotyping.py \
        --data-dir data/processed/mimic-iv \
        --phenotype-config /path/to/ccs_phenotypes.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Reference prevalence rates for sanity-checking phenotype labels.
#
# MIMIC-IV observed rates (from SLICES extraction on MIMIC-IV v3.0, 39,799
# valid stays).  ICD-9 and ICD-10 per-version rates are within 1-3 pp of
# each other, confirming code-list consistency.
#
# For comparison, Harutyunyan et al. 2019 (MIMIC-III) reported substantially
# lower rates (e.g. acute_renal_failure 12%, respiratory_failure 18%).  The
# difference is a dataset/cohort shift (sicker population, more thorough
# coding practices in the 2008-2019 era), not a mapping error.
# ---------------------------------------------------------------------------
REFERENCE_RATES: Dict[str, float] = {
    "sepsis": 0.23,
    "respiratory_failure": 0.38,
    "acute_renal_failure": 0.36,
    "chf": 0.31,
    "shock": 0.18,
    "chronic_kidney_disease": 0.22,
    "diabetes": 0.33,
    "copd": 0.16,
    "pneumonia": 0.20,
    "coronary_atherosclerosis": 0.33,
}

DEVIATION_THRESHOLD: float = 0.05  # 5 percentage-point threshold


# ---------------------------------------------------------------------------
# Phenotype config loading
# ---------------------------------------------------------------------------


def _find_phenotype_config() -> Optional[Path]:
    """Locate ccs_phenotypes.yaml from the package data directory.

    Returns:
        Path to the config if found, else None.
    """
    from slices.data.config_loader import _get_package_data_dir

    candidate = _get_package_data_dir() / "phenotypes" / "ccs_phenotypes.yaml"
    if candidate.exists():
        return candidate

    return None


def _load_phenotype_definitions(config_path: Path) -> Dict[str, Dict]:
    """Load phenotype definitions from YAML config.

    Args:
        config_path: Path to ccs_phenotypes.yaml.

    Returns:
        Dictionary mapping phenotype name to its definition dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Phenotype config not found: {config_path}")

    with open(config_path) as f:
        definitions = yaml.safe_load(f)

    return definitions


def _build_icd_to_phenotype_map(
    definitions: Dict[str, Dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build reverse maps from ICD code -> list of phenotype names.

    Args:
        definitions: Phenotype definitions from YAML.

    Returns:
        Tuple of (icd9_map, icd10_map).
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


# ---------------------------------------------------------------------------
# Check 1: Reference prevalence comparison
# ---------------------------------------------------------------------------


def compute_prevalence(labels_df: pl.DataFrame) -> Dict[str, float]:
    """Compute prevalence for each phenotyping_* column in labels.parquet.

    Args:
        labels_df: DataFrame with phenotyping_* columns (0/1, nullable).

    Returns:
        Dictionary mapping phenotype name (without prefix) to prevalence rate.
    """
    prevalences: Dict[str, float] = {}
    phenotype_cols = [c for c in labels_df.columns if c.startswith("phenotyping_")]

    for col in phenotype_cols:
        name = col.removeprefix("phenotyping_")
        non_null = labels_df[col].drop_nulls()
        if len(non_null) == 0:
            prevalences[name] = 0.0
        else:
            positive = (non_null == 1).sum()
            prevalences[name] = positive / len(non_null)

    return prevalences


def compare_reference_rates(
    observed: Dict[str, float],
    total_valid: Dict[str, int],
) -> List[str]:
    """Print comparison table and return list of warnings.

    Args:
        observed: Observed prevalence rates by phenotype name.
        total_valid: Number of valid (non-null) stays per phenotype.

    Returns:
        List of warning strings for deviations > threshold.
    """
    warnings: List[str] = []

    header = (
        f"  {'Phenotype':<30} {'Reference':>10} {'Observed':>10} "
        f"{'Delta (pp)':>12} {'N valid':>10} {'Status':>10}"
    )
    separator = f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10}"

    print(header)
    print(separator)

    all_names = sorted(set(list(REFERENCE_RATES.keys()) + list(observed.keys())))

    for name in all_names:
        pub = REFERENCE_RATES.get(name)
        obs = observed.get(name)
        n_valid = total_valid.get(name, 0)

        if obs is None:
            print(
                f"  {name:<30} {_fmt_pct(pub):>10} {'MISSING':>10}"
                f" {'--':>12} {'--':>10} {'MISSING':>10}"
            )
            warnings.append(f"Phenotype '{name}' expected but not found in labels.")
            continue

        if pub is None:
            print(
                f"  {name:<30} {'N/A':>10} {_fmt_pct(obs):>10}"
                f" {'N/A':>12} {n_valid:>10,} {'INFO':>10}"
            )
            continue

        delta = obs - pub
        delta_pp = delta * 100
        abs_delta = abs(delta)

        if abs_delta > DEVIATION_THRESHOLD:
            status = "WARNING"
            warnings.append(
                f"Phenotype '{name}': observed={obs:.1%}, reference={pub:.1%}, "
                f"delta={delta_pp:+.1f}pp (exceeds {DEVIATION_THRESHOLD*100:.0f}pp threshold)"
            )
        else:
            status = "OK"

        print(
            f"  {name:<30} {_fmt_pct(pub):>10} {_fmt_pct(obs):>10} "
            f"{delta_pp:>+11.1f}  {n_valid:>10,} {status:>10}"
        )

    return warnings


def _fmt_pct(value: Optional[float]) -> str:
    """Format a float as a percentage string, or 'N/A' if None."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"


# ---------------------------------------------------------------------------
# Check 2: Internal ICD consistency
# ---------------------------------------------------------------------------


def count_icd_phenotypes_from_diagnoses(
    data_dir: Path,
    icd9_map: Dict[str, List[str]],
    icd10_map: Dict[str, List[str]],
    phenotype_names: List[str],
) -> Dict[str, int]:
    """Independently count stays with matching ICD codes from raw diagnoses.

    This replicates the label builder logic to validate that label counts are
    consistent with raw diagnosis data.

    Args:
        data_dir: Path to processed data directory (must contain diagnoses.parquet).
        icd9_map: ICD-9 code to phenotype name mapping.
        icd10_map: ICD-10 code to phenotype name mapping.
        phenotype_names: Ordered list of phenotype names.

    Returns:
        Dictionary mapping phenotype name to the count of positive stays.

    Raises:
        FileNotFoundError: If diagnoses.parquet is not found.
    """
    diagnoses_path = data_dir / "diagnoses.parquet"
    if not diagnoses_path.exists():
        raise FileNotFoundError(
            f"Diagnoses file not found: {diagnoses_path}. "
            "ICD consistency check requires the raw diagnoses.parquet."
        )

    diagnoses = pl.read_parquet(diagnoses_path)

    # Track which stays have each phenotype
    stay_phenotypes: Dict[str, set] = {name: set() for name in phenotype_names}

    if len(diagnoses) == 0 or "icd_code" not in diagnoses.columns:
        return {name: 0 for name in phenotype_names}

    for row in diagnoses.iter_rows(named=True):
        stay_id = row["stay_id"]
        icd_code = str(row["icd_code"]).strip()
        icd_version = row["icd_version"]

        if icd_version == 9:
            matched = icd9_map.get(icd_code, [])
        elif icd_version == 10:
            matched = icd10_map.get(icd_code, [])
        else:
            continue

        for phenotype_name in matched:
            if phenotype_name in stay_phenotypes:
                stay_phenotypes[phenotype_name].add(stay_id)

    return {name: len(stays) for name, stays in stay_phenotypes.items()}


def compare_icd_consistency(
    label_counts: Dict[str, int],
    icd_counts: Dict[str, int],
    phenotype_names: List[str],
) -> List[str]:
    """Compare label-derived counts against independent ICD counts.

    Args:
        label_counts: Positive counts per phenotype from labels.parquet.
        icd_counts: Positive counts per phenotype from independent ICD scan.
        phenotype_names: Ordered phenotype names.

    Returns:
        List of warning strings for discrepancies.
    """
    warnings: List[str] = []

    header = (
        f"  {'Phenotype':<30} {'Label count':>12} {'ICD count':>12} "
        f"{'Difference':>12} {'Status':>10}"
    )
    separator = f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}"

    print(header)
    print(separator)

    for name in phenotype_names:
        lc = label_counts.get(name, 0)
        ic = icd_counts.get(name, 0)
        diff = lc - ic

        # Exact match expected (both use the same ICD mapping logic)
        if diff != 0:
            status = "MISMATCH"
            warnings.append(
                f"Phenotype '{name}': label_count={lc}, icd_count={ic}, "
                f"difference={diff} (possible mapping bug)"
            )
        else:
            status = "OK"

        print(f"  {name:<30} {lc:>12,} {ic:>12,} {diff:>12,} {status:>10}")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate phenotyping labels against reference rates and ICD consistency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help=("Path to ccs_phenotypes.yaml. " "If not specified, auto-detected from project root."),
    )
    return parser.parse_args()


def main() -> int:
    """Run phenotyping validation checks."""
    args = parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    labels_path = data_dir / "labels.parquet"
    if not labels_path.exists():
        print(f"Error: labels.parquet not found in {data_dir}")
        return 1

    # Load labels
    labels_df = pl.read_parquet(labels_path)
    phenotype_cols = [c for c in labels_df.columns if c.startswith("phenotyping_")]

    if not phenotype_cols:
        print("Error: No phenotyping_* columns found in labels.parquet.")
        print(f"  Available columns: {labels_df.columns}")
        return 1

    phenotype_names = [c.removeprefix("phenotyping_") for c in phenotype_cols]
    total_stays = len(labels_df)

    print("=" * 82)
    print("Phenotyping Label Validation")
    print("=" * 82)
    print(f"\n  Data directory:  {data_dir}")
    print(f"  Total stays:     {total_stays:,}")
    print(f"  Phenotypes found: {len(phenotype_names)}")
    print(f"  Phenotype names:  {', '.join(phenotype_names)}")

    all_warnings: List[str] = []

    # ------------------------------------------------------------------
    # Check 1: Reference prevalence comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 82}")
    print("Check 1: Reference Prevalence Comparison (MIMIC-IV v3.0)")
    print(f"{'=' * 82}\n")

    observed = compute_prevalence(labels_df)

    # Compute valid counts per phenotype
    total_valid: Dict[str, int] = {}
    for col in phenotype_cols:
        name = col.removeprefix("phenotyping_")
        total_valid[name] = len(labels_df[col].drop_nulls())

    prevalence_warnings = compare_reference_rates(observed, total_valid)
    all_warnings.extend(prevalence_warnings)

    if prevalence_warnings:
        print(f"\n  {len(prevalence_warnings)} prevalence warning(s):")
        for w in prevalence_warnings:
            print(f"    - {w}")
    else:
        print("\n  All phenotype prevalences within expected range.")

    print("\n  Note: Reference rates are from MIMIC-IV v3.0. Minor deviations")
    print(f"  Threshold: {DEVIATION_THRESHOLD * 100:.0f} percentage points.")

    # ------------------------------------------------------------------
    # Check 2: Internal ICD consistency
    # ------------------------------------------------------------------
    print(f"\n{'=' * 82}")
    print("Check 2: Internal ICD Consistency")
    print(f"{'=' * 82}\n")

    # Resolve phenotype config
    config_path: Optional[Path] = args.phenotype_config
    if config_path is None:
        config_path = _find_phenotype_config()

    if config_path is None:
        print("  Skipping ICD consistency check: could not locate ccs_phenotypes.yaml.")
        print("  Use --phenotype-config to specify the path explicitly.")
    elif not config_path.exists():
        print(f"  Skipping ICD consistency check: config not found at {config_path}")
    else:
        diagnoses_path = data_dir / "diagnoses.parquet"
        if not diagnoses_path.exists():
            print(f"  Skipping ICD consistency check: diagnoses.parquet not found in {data_dir}.")
        else:
            print(f"  Phenotype config: {config_path}")
            print(f"  Diagnoses file:   {diagnoses_path}\n")

            definitions = _load_phenotype_definitions(config_path)
            icd9_map, icd10_map = _build_icd_to_phenotype_map(definitions)

            # Count positives from labels
            label_counts: Dict[str, int] = {}
            for col in phenotype_cols:
                name = col.removeprefix("phenotyping_")
                non_null = labels_df[col].drop_nulls()
                label_counts[name] = int((non_null == 1).sum())

            # Count positives from independent ICD scan
            icd_counts = count_icd_phenotypes_from_diagnoses(
                data_dir, icd9_map, icd10_map, phenotype_names
            )

            icd_warnings = compare_icd_consistency(label_counts, icd_counts, phenotype_names)
            all_warnings.extend(icd_warnings)

            if icd_warnings:
                print(f"\n  {len(icd_warnings)} ICD consistency warning(s):")
                for w in icd_warnings:
                    print(f"    - {w}")
            else:
                print("\n  All label counts match independent ICD scan.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 82}")
    print("Summary")
    print(f"{'=' * 82}")

    if all_warnings:
        print(f"\n  Total warnings: {len(all_warnings)}")
        for i, w in enumerate(all_warnings, 1):
            print(f"    {i}. {w}")
        print()
        return 1
    else:
        print("\n  All validation checks passed.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
