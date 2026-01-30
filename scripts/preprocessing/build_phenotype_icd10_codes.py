"""Build ICD-10 code lists for phenotype definitions using GEMs crosswalk.

Reads CCS phenotype definitions (which contain ICD-9 codes) and uses the
CMS General Equivalence Mappings (GEMs) to derive corresponding ICD-10 codes.
Updates the phenotype YAML in-place, merging derived codes with any existing
manually-added ICD-10 codes.

Example usage:
    # Use default paths
    uv run python scripts/preprocessing/build_phenotype_icd10_codes.py

    # Specify custom paths
    uv run python scripts/preprocessing/build_phenotype_icd10_codes.py \
        --gems-path configs/phenotypes/gems_icd10_to_icd9.csv \
        --phenotype-config configs/phenotypes/ccs_phenotypes.yaml
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Derive ICD-10 codes for phenotypes using GEMs crosswalk.",
    )
    parser.add_argument(
        "--gems-path",
        type=Path,
        default=Path("configs/phenotypes/gems_icd10_to_icd9.csv"),
        help="Path to the GEMs ICD-10-CM to ICD-9-CM crosswalk CSV.",
    )
    parser.add_argument(
        "--phenotype-config",
        type=Path,
        default=Path("configs/phenotypes/ccs_phenotypes.yaml"),
        help="Path to the CCS phenotype definitions YAML.",
    )
    return parser.parse_args()


def load_gems_crosswalk(gems_path: Path) -> dict[str, list[str]]:
    """Load GEMs CSV and build a reverse mapping from ICD-9 to ICD-10 codes.

    The GEMs file maps ICD-10 -> ICD-9. We invert this so that for each
    ICD-9 code we know all ICD-10 codes that map to it.

    Args:
        gems_path: Path to the GEMs CSV file with columns icd10_code, icd9_code.

    Returns:
        Dictionary mapping ICD-9 codes to lists of ICD-10 codes.
    """
    icd9_to_icd10: dict[str, list[str]] = defaultdict(list)

    with open(gems_path, newline="") as f:
        reader = csv.DictReader(_skip_comments(f))
        for row in reader:
            icd10 = row["icd10_code"].strip()
            icd9 = row["icd9_code"].strip()
            if icd10 and icd9:
                icd9_to_icd10[icd9].append(icd10)

    return dict(icd9_to_icd10)


def _skip_comments(iterable):
    """Yield lines from an iterable, skipping comment lines starting with #."""
    for line in iterable:
        if not line.strip().startswith("#"):
            yield line


def load_phenotype_config(config_path: Path) -> dict:
    """Load the phenotype definitions YAML.

    Args:
        config_path: Path to the CCS phenotype definitions YAML.

    Returns:
        Parsed YAML content as a dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_phenotype_config(config: dict, config_path: Path) -> None:
    """Write the phenotype definitions YAML back to disk.

    Args:
        config: The phenotype configuration dictionary.
        config_path: Path to write the YAML file.
    """
    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )


def derive_icd10_codes(
    icd9_codes: list[str],
    icd9_to_icd10: dict[str, list[str]],
) -> list[str]:
    """Map a list of ICD-9 codes to ICD-10 codes using the GEMs crosswalk.

    Args:
        icd9_codes: List of ICD-9 codes (without dots).
        icd9_to_icd10: Reverse GEMs mapping from ICD-9 to ICD-10 codes.

    Returns:
        Sorted, deduplicated list of ICD-10 codes.
    """
    icd10_set: set[str] = set()
    for icd9 in icd9_codes:
        mapped = icd9_to_icd10.get(icd9, [])
        icd10_set.update(mapped)
    return sorted(icd10_set)


def merge_icd10_codes(
    existing: list[str] | None,
    derived: list[str],
) -> list[str]:
    """Merge existing (manually-added) ICD-10 codes with derived ones.

    Preserves any manually-added codes that are not in the derived set.

    Args:
        existing: Existing ICD-10 codes from the YAML (may be None).
        derived: Newly derived ICD-10 codes from the GEMs crosswalk.

    Returns:
        Sorted, deduplicated merged list of ICD-10 codes.
    """
    merged: set[str] = set(derived)
    if existing:
        merged.update(existing)
    return sorted(merged)


def update_phenotypes(
    config: dict,
    icd9_to_icd10: dict[str, list[str]],
) -> dict[str, int]:
    """Update all phenotypes in the config with derived ICD-10 codes.

    For each phenotype that has icd9_codes, derives the corresponding
    ICD-10 codes and merges them into the icd10_codes field.

    Args:
        config: The phenotype configuration dictionary.
        icd9_to_icd10: Reverse GEMs mapping from ICD-9 to ICD-10 codes.

    Returns:
        Dictionary mapping phenotype names to the number of ICD-10 codes added.
    """
    stats: dict[str, int] = {}
    phenotypes = config.get("phenotypes", config)

    for name, definition in phenotypes.items():
        if not isinstance(definition, dict):
            continue

        icd9_codes = definition.get("icd9_codes", [])
        if not icd9_codes:
            stats[name] = 0
            continue

        derived = derive_icd10_codes(icd9_codes, icd9_to_icd10)
        existing = definition.get("icd10_codes")
        merged = merge_icd10_codes(existing, derived)

        definition["icd10_codes"] = merged
        stats[name] = len(merged)

    return stats


def print_summary(stats: dict[str, int]) -> None:
    """Print summary statistics about the ICD-10 derivation.

    Args:
        stats: Dictionary mapping phenotype names to ICD-10 code counts.
    """
    total_phenotypes = len(stats)
    with_codes = sum(1 for count in stats.values() if count > 0)
    without_codes = total_phenotypes - with_codes
    total_codes = sum(stats.values())

    print(f"\n{'='*60}")
    print("Phenotype ICD-10 Code Derivation Summary")
    print(f"{'='*60}")
    print(f"Total phenotypes:            {total_phenotypes}")
    print(f"Phenotypes with ICD-10 codes: {with_codes}")
    print(f"Phenotypes without matches:   {without_codes}")
    print(f"Total ICD-10 codes assigned:  {total_codes}")
    print(f"{'='*60}")

    if stats:
        print(f"\n{'Phenotype':<40} {'ICD-10 codes':>12}")
        print(f"{'-'*40} {'-'*12}")
        for name, count in sorted(stats.items()):
            print(f"{name:<40} {count:>12}")
    print()


def main() -> None:
    """Derive ICD-10 codes for phenotypes and update the YAML config."""
    args = parse_args()

    # Validate input files exist
    if not args.gems_path.exists():
        raise FileNotFoundError(
            f"GEMs crosswalk not found: {args.gems_path}\n"
            "Download from: https://www.cms.gov/medicare/coding-billing/"
            "icd-10-codes/general-equivalence-mappings-gems"
        )
    if not args.phenotype_config.exists():
        raise FileNotFoundError(f"Phenotype config not found: {args.phenotype_config}")

    # Load data
    print(f"Loading GEMs crosswalk from: {args.gems_path}")
    icd9_to_icd10 = load_gems_crosswalk(args.gems_path)
    print(f"  Loaded {len(icd9_to_icd10)} unique ICD-9 -> ICD-10 mappings")

    print(f"Loading phenotype config from: {args.phenotype_config}")
    config = load_phenotype_config(args.phenotype_config)

    # Derive and update ICD-10 codes
    stats = update_phenotypes(config, icd9_to_icd10)

    # Save updated config
    save_phenotype_config(config, args.phenotype_config)
    print(f"Updated config written to: {args.phenotype_config}")

    # Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()
