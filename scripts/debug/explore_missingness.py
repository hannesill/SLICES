#!/usr/bin/env python
"""Explore missingness patterns in binned ICU time-series data.

This script analyzes:
1. Overall missingness percentage across all patients
2. Per-feature missingness rates
3. Per-patient missingness distribution
4. Missingness by patient groups (demographics)
5. True vs structural missingness (separating "not measured" from "patient left ICU")

Example usage:
    uv run python scripts/debug/explore_missingness.py \\
        --processed-dir data/processed/mimic-iv-demo \\
        --top-n 10
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def load_data(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Load timeseries, static data, and metadata."""
    timeseries_df = pl.read_parquet(data_dir / "timeseries.parquet")
    static_df = pl.read_parquet(data_dir / "static.parquet")

    with open(data_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    return timeseries_df, static_df, metadata


def compute_overall_missingness(timeseries_df: pl.DataFrame) -> dict:
    """Compute overall missingness statistics."""
    # Extract masks and compute stats
    total_observed = 0
    total_cells = 0

    for row in timeseries_df.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        total_observed += mask.sum()
        total_cells += mask.size

    missing_pct = 100 * (1 - total_observed / total_cells)

    return {
        "total_cells": total_cells,
        "total_observed": total_observed,
        "total_missing": total_cells - total_observed,
        "missing_pct": missing_pct,
        "observed_pct": 100 - missing_pct,
    }


def compute_per_feature_missingness(
    timeseries_df: pl.DataFrame, feature_names: list[str]
) -> pl.DataFrame:
    """Compute missingness rate per feature."""
    n_features = len(feature_names)
    feature_observed = np.zeros(n_features)
    feature_total = np.zeros(n_features)

    for row in timeseries_df.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        feature_observed += mask.sum(axis=0)
        feature_total += mask.shape[0]  # seq_len per feature

    missing_pct = 100 * (1 - feature_observed / feature_total)

    return pl.DataFrame(
        {
            "feature": feature_names,
            "observed_count": feature_observed.astype(int),
            "total_count": feature_total.astype(int),
            "missing_pct": missing_pct,
            "observed_pct": 100 - missing_pct,
        }
    ).sort("missing_pct", descending=True)


def compute_per_patient_missingness(timeseries_df: pl.DataFrame) -> pl.DataFrame:
    """Compute missingness rate per patient/stay."""
    results = []

    for row in timeseries_df.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)
        total = mask.size
        observed = mask.sum()
        missing_pct = 100 * (1 - observed / total)

        results.append(
            {
                "stay_id": row["stay_id"],
                "total_cells": total,
                "observed_cells": observed,
                "missing_cells": total - observed,
                "missing_pct": missing_pct,
            }
        )

    return pl.DataFrame(results).sort("missing_pct", descending=True)


def compute_group_missingness(
    patient_missingness: pl.DataFrame,
    static_df: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Compute missingness statistics by patient group."""
    # Join patient missingness with static data
    joined = patient_missingness.join(
        static_df.select(["stay_id", group_col]),
        on="stay_id",
    )

    # Group and aggregate
    return (
        joined.group_by(group_col)
        .agg(
            [
                pl.len().alias("n_patients"),
                pl.col("missing_pct").mean().alias("mean_missing_pct"),
                pl.col("missing_pct").std().alias("std_missing_pct"),
                pl.col("missing_pct").median().alias("median_missing_pct"),
                pl.col("missing_pct").min().alias("min_missing_pct"),
                pl.col("missing_pct").max().alias("max_missing_pct"),
            ]
        )
        .sort("mean_missing_pct", descending=True)
    )


def compute_temporal_missingness(timeseries_df: pl.DataFrame, seq_length: int) -> pl.DataFrame:
    """Compute missingness rate per hour (temporal pattern)."""
    hourly_observed = np.zeros(seq_length)
    hourly_total = np.zeros(seq_length)

    for row in timeseries_df.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        actual_len = min(mask.shape[0], seq_length)
        hourly_observed[:actual_len] += mask[:actual_len].sum(axis=1)
        hourly_total[:actual_len] += mask.shape[1]  # n_features per hour

    missing_pct = 100 * (1 - hourly_observed / hourly_total)

    return pl.DataFrame(
        {
            "hour": list(range(seq_length)),
            "observed_count": hourly_observed.astype(int),
            "total_count": hourly_total.astype(int),
            "missing_pct": missing_pct,
        }
    )


def print_section(title: str, width: int = 60):
    """Print a section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def compute_true_vs_structural_missingness(
    timeseries_df: pl.DataFrame,
    static_df: pl.DataFrame,
    seq_length: int,
    n_features: int,
) -> dict:
    """Separate true missingness from structural missingness.

    True missingness: Patient is in ICU but measurement not taken.
    Structural missingness: Patient has left ICU (hours beyond LOS).
    """
    # Join to get LOS for each stay
    joined = timeseries_df.join(
        static_df.select(["stay_id", "los_days"]),
        on="stay_id",
    )

    total_true_missing = 0
    total_structural_missing = 0
    total_observed = 0
    total_cells = 0

    per_patient_results = []

    for row in joined.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        los_days = row["los_days"]

        # Convert LOS to hours, capped at seq_length
        los_hours = min(int(np.ceil(los_days * 24)), seq_length)

        # Cells while patient was in ICU (true window)
        if los_hours > 0:
            true_window_mask = mask[:los_hours, :]
            true_window_observed = true_window_mask.sum()
            true_window_cells = los_hours * n_features
            true_window_missing = true_window_cells - true_window_observed
        else:
            true_window_observed = 0
            true_window_cells = 0
            true_window_missing = 0

        # Cells after patient left (structural)
        structural_cells = (seq_length - los_hours) * n_features
        structural_missing = structural_cells  # All are "missing" by definition

        total_observed += true_window_observed
        total_true_missing += true_window_missing
        total_structural_missing += structural_missing
        total_cells += mask.size

        # Per-patient stats
        if true_window_cells > 0:
            true_missing_pct = 100 * true_window_missing / true_window_cells
        else:
            true_missing_pct = 100.0

        per_patient_results.append(
            {
                "stay_id": row["stay_id"],
                "los_hours": los_hours,
                "true_window_cells": true_window_cells,
                "true_window_observed": true_window_observed,
                "true_missing_pct": true_missing_pct,
                "structural_cells": structural_cells,
            }
        )

    per_patient_df = pl.DataFrame(per_patient_results)

    return {
        "total_cells": total_cells,
        "total_observed": total_observed,
        "total_true_missing": total_true_missing,
        "total_structural_missing": total_structural_missing,
        "true_missing_pct": (
            100 * total_true_missing / (total_observed + total_true_missing)
            if (total_observed + total_true_missing) > 0
            else 0
        ),
        "structural_pct": 100 * total_structural_missing / total_cells,
        "per_patient_df": per_patient_df,
    }


def compute_temporal_missingness_true_only(
    timeseries_df: pl.DataFrame,
    static_df: pl.DataFrame,
    seq_length: int,
) -> pl.DataFrame:
    """Compute missingness per hour, only counting patients still in ICU."""
    # Join to get LOS for each stay
    joined = timeseries_df.join(
        static_df.select(["stay_id", "los_days"]),
        on="stay_id",
    )

    hourly_observed = np.zeros(seq_length)
    hourly_total = np.zeros(seq_length)  # Only count patients present at that hour

    for row in joined.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        los_days = row["los_days"]
        los_hours = min(int(np.ceil(los_days * 24)), seq_length)

        # Only count hours where patient was present
        for hour in range(los_hours):
            hourly_observed[hour] += mask[hour, :].sum()
            hourly_total[hour] += mask.shape[1]  # n_features

    # Avoid division by zero for hours with no patients
    with np.errstate(divide="ignore", invalid="ignore"):
        missing_pct = np.where(hourly_total > 0, 100 * (1 - hourly_observed / hourly_total), np.nan)

    return pl.DataFrame(
        {
            "hour": list(range(seq_length)),
            "observed_count": hourly_observed.astype(int),
            "patients_present": (hourly_total / mask.shape[1]).astype(int),
            "total_count": hourly_total.astype(int),
            "missing_pct": missing_pct,
        }
    )


def compute_per_feature_true_missingness(
    timeseries_df: pl.DataFrame,
    static_df: pl.DataFrame,
    feature_names: list[str],
    seq_length: int,
) -> pl.DataFrame:
    """Compute per-feature missingness only within actual ICU hours."""
    # Join to get LOS for each stay
    joined = timeseries_df.join(
        static_df.select(["stay_id", "los_days"]),
        on="stay_id",
    )

    n_features = len(feature_names)
    feature_observed = np.zeros(n_features)
    feature_total = np.zeros(n_features)

    for row in joined.iter_rows(named=True):
        mask = np.array(row["mask"], dtype=bool)  # (seq_len, n_features)
        los_days = row["los_days"]
        los_hours = min(int(np.ceil(los_days * 24)), seq_length)

        if los_hours > 0:
            # Only count within true window
            feature_observed += mask[:los_hours, :].sum(axis=0)
            feature_total += los_hours

    missing_pct = 100 * (1 - feature_observed / feature_total)

    return pl.DataFrame(
        {
            "feature": feature_names,
            "observed_count": feature_observed.astype(int),
            "total_count": feature_total.astype(int),
            "missing_pct": missing_pct,
            "observed_pct": 100 - missing_pct,
        }
    ).sort("missing_pct", descending=True)


def main():
    parser = argparse.ArgumentParser(description="Explore missingness in ICU data")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/mimic-iv-demo"),
        help="Path to processed data directory (with static.parquet, "
        "timeseries.parquet, metadata.yaml)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top/bottom items to show in rankings",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.processed_dir}")
    timeseries_df, static_df, metadata = load_data(args.processed_dir)

    feature_names = metadata["feature_names"]
    seq_length = metadata.get("seq_length_hours", 48)
    n_stays = len(timeseries_df)
    n_features = len(feature_names)

    print(f"Dataset: {metadata.get('dataset', 'unknown')}")
    print(f"Number of stays: {n_stays}")
    print(f"Number of features: {n_features}")
    print(f"Sequence length: {seq_length} hours")

    # 1. Overall missingness
    print_section("1. OVERALL MISSINGNESS")
    overall = compute_overall_missingness(timeseries_df)
    print(f"Total cells: {overall['total_cells']:,}")
    print(f"Observed cells: {overall['total_observed']:,}")
    print(f"Missing cells: {overall['total_missing']:,}")
    print(f"\nMissing percentage: {overall['missing_pct']:.2f}%")
    print(f"Observed percentage: {overall['observed_pct']:.2f}%")

    # 2. Per-feature missingness
    print_section("2. PER-FEATURE MISSINGNESS")
    feature_miss = compute_per_feature_missingness(timeseries_df, feature_names)

    print(f"\nTop {args.top_n} features with HIGHEST missingness:")
    print("-" * 50)
    for row in feature_miss.head(args.top_n).iter_rows(named=True):
        print(f"  {row['feature']:25s} {row['missing_pct']:6.2f}% missing")

    print(f"\nTop {args.top_n} features with LOWEST missingness:")
    print("-" * 50)
    for row in feature_miss.tail(args.top_n).reverse().iter_rows(named=True):
        print(f"  {row['feature']:25s} {row['missing_pct']:6.2f}% missing")

    # 3. Per-patient missingness distribution
    print_section("3. PER-PATIENT MISSINGNESS DISTRIBUTION")
    patient_miss = compute_per_patient_missingness(timeseries_df)

    stats = patient_miss.select(
        [
            pl.col("missing_pct").mean().alias("mean"),
            pl.col("missing_pct").std().alias("std"),
            pl.col("missing_pct").median().alias("median"),
            pl.col("missing_pct").min().alias("min"),
            pl.col("missing_pct").max().alias("max"),
            pl.col("missing_pct").quantile(0.25).alias("q25"),
            pl.col("missing_pct").quantile(0.75).alias("q75"),
        ]
    ).row(0, named=True)

    print(f"Mean:   {stats['mean']:.2f}%")
    print(f"Std:    {stats['std']:.2f}%")
    print(f"Median: {stats['median']:.2f}%")
    print(f"Min:    {stats['min']:.2f}%")
    print(f"Max:    {stats['max']:.2f}%")
    print(f"IQR:    [{stats['q25']:.2f}%, {stats['q75']:.2f}%]")

    # Histogram buckets
    print("\nDistribution of patient missingness:")
    print("-" * 50)
    buckets = [0, 50, 60, 70, 80, 90, 95, 100]
    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i + 1]
        count = patient_miss.filter(
            (pl.col("missing_pct") >= low) & (pl.col("missing_pct") < high)
        ).height
        pct = 100 * count / n_stays
        bar = "#" * int(pct / 2)
        print(f"  {low:3d}-{high:3d}%: {count:4d} patients ({pct:5.1f}%) {bar}")

    # 4. Temporal missingness pattern
    print_section("4. TEMPORAL MISSINGNESS PATTERN (by hour)")
    temporal_miss = compute_temporal_missingness(timeseries_df, seq_length)

    # Show every 6 hours
    print("Missingness by hour (every 6 hours):")
    print("-" * 50)
    for hour in range(0, seq_length, 6):
        row = temporal_miss.filter(pl.col("hour") == hour).row(0, named=True)
        bar = "#" * int(row["missing_pct"] / 2)
        print(f"  Hour {hour:2d}: {row['missing_pct']:6.2f}% {bar}")

    # 5. True vs Structural Missingness
    print_section("5. TRUE VS STRUCTURAL MISSINGNESS")
    print("Separating 'not measured' from 'patient left ICU'")
    print("-" * 50)

    true_vs_struct = compute_true_vs_structural_missingness(
        timeseries_df, static_df, seq_length, n_features
    )

    print(f"\nTotal cells in dense representation: {true_vs_struct['total_cells']:,}")
    print(
        f"  - Observed:             {true_vs_struct['total_observed']:,} "
        f"({100 * true_vs_struct['total_observed'] / true_vs_struct['total_cells']:.1f}%)"
    )
    print(
        f"  - True missing:         {true_vs_struct['total_true_missing']:,} "
        f"({100 * true_vs_struct['total_true_missing'] / true_vs_struct['total_cells']:.1f}%)"
    )
    print(
        f"  - Structural missing:   {true_vs_struct['total_structural_missing']:,} "
        f"({true_vs_struct['structural_pct']:.1f}%)"
    )

    print("\nWithin actual ICU hours only:")
    true_window_total = true_vs_struct["total_observed"] + true_vs_struct["total_true_missing"]
    print(f"  Total cells:    {true_window_total:,}")
    print(
        f"  Observed:       {true_vs_struct['total_observed']:,} "
        f"({100 * true_vs_struct['total_observed'] / true_window_total:.1f}%)"
    )
    print(
        f"  True missing:   {true_vs_struct['total_true_missing']:,} "
        f"({true_vs_struct['true_missing_pct']:.1f}%)"
    )

    # Per-patient true missingness distribution
    per_patient_true = true_vs_struct["per_patient_df"]
    true_stats = per_patient_true.select(
        [
            pl.col("true_missing_pct").mean().alias("mean"),
            pl.col("true_missing_pct").std().alias("std"),
            pl.col("true_missing_pct").median().alias("median"),
            pl.col("true_missing_pct").min().alias("min"),
            pl.col("true_missing_pct").max().alias("max"),
        ]
    ).row(0, named=True)

    print("\nPer-patient TRUE missingness (within their ICU stay):")
    print(f"  Mean:   {true_stats['mean']:.2f}%")
    print(f"  Std:    {true_stats['std']:.2f}%")
    print(f"  Median: {true_stats['median']:.2f}%")
    print(f"  Range:  [{true_stats['min']:.2f}%, {true_stats['max']:.2f}%]")

    # 5b. Temporal pattern with true missingness only
    print_section("5b. TEMPORAL PATTERN (patients present only)")
    temporal_true = compute_temporal_missingness_true_only(timeseries_df, static_df, seq_length)

    print("Missingness by hour (only counting patients still in ICU):")
    print("-" * 60)
    print(f"  {'Hour':>6}  {'Missing%':>10}  {'Patients':>10}  Pattern")
    print("-" * 60)
    for hour in range(0, seq_length, 6):
        row = temporal_true.filter(pl.col("hour") == hour).row(0, named=True)
        if row["patients_present"] > 0:
            bar = "#" * int(row["missing_pct"] / 2)
            print(f"  {hour:>6}  {row['missing_pct']:>9.2f}%  {row['patients_present']:>10}  {bar}")
        else:
            print(f"  {hour:>6}  {'N/A':>10}  {row['patients_present']:>10}  (no patients)")

    # Compare temporal patterns
    print("\nComparison: Dense vs True-only missingness by hour")
    print("-" * 70)
    print(f"  {'Hour':>6}  {'Dense%':>10}  {'True%':>10}  {'Diff':>10}  {'Patients':>10}")
    print("-" * 70)
    for hour in range(0, seq_length, 6):
        dense_row = temporal_miss.filter(pl.col("hour") == hour).row(0, named=True)
        true_row = temporal_true.filter(pl.col("hour") == hour).row(0, named=True)
        if true_row["patients_present"] > 0:
            diff = dense_row["missing_pct"] - true_row["missing_pct"]
            print(
                f"  {hour:>6}  {dense_row['missing_pct']:>9.2f}%  "
                f"{true_row['missing_pct']:>9.2f}%  {diff:>+9.2f}%  "
                f"{true_row['patients_present']:>10}"
            )

    # 5c. Per-feature true missingness
    print_section("5c. PER-FEATURE TRUE MISSINGNESS")
    feature_true_miss = compute_per_feature_true_missingness(
        timeseries_df, static_df, feature_names, seq_length
    )

    print(f"\nTop {args.top_n} features with HIGHEST true missingness:")
    print("-" * 50)
    for row in feature_true_miss.head(args.top_n).iter_rows(named=True):
        print(f"  {row['feature']:25s} {row['missing_pct']:6.2f}% missing")

    print(f"\nTop {args.top_n} features with LOWEST true missingness:")
    print("-" * 50)
    for row in feature_true_miss.tail(args.top_n).reverse().iter_rows(named=True):
        print(f"  {row['feature']:25s} {row['missing_pct']:6.2f}% missing")

    # 6. Missingness by patient groups
    print_section("6. MISSINGNESS BY PATIENT GROUPS")

    # Available grouping columns
    group_cols = ["gender", "admission_type", "first_careunit", "insurance"]

    # Check which columns exist in static_df
    available_cols = [c for c in group_cols if c in static_df.columns]

    for group_col in available_cols:
        print(f"\n--- By {group_col.upper()} ---")
        group_miss = compute_group_missingness(patient_miss, static_df, group_col)

        for row in group_miss.iter_rows(named=True):
            group_name = row[group_col] or "Unknown"
            print(
                f"  {group_name:30s} n={row['n_patients']:4d}  "
                f"mean={row['mean_missing_pct']:5.1f}%  "
                f"median={row['median_missing_pct']:5.1f}%  "
                f"range=[{row['min_missing_pct']:5.1f}%, {row['max_missing_pct']:5.1f}%]"
            )

    # 6b. Age groups (binned)
    if "age" in static_df.columns:
        print("\n--- By AGE GROUP ---")
        # Create age bins
        static_with_age_group = static_df.with_columns(
            pl.when(pl.col("age") < 40)
            .then(pl.lit("<40"))
            .when(pl.col("age") < 60)
            .then(pl.lit("40-59"))
            .when(pl.col("age") < 80)
            .then(pl.lit("60-79"))
            .otherwise(pl.lit("80+"))
            .alias("age_group")
        )
        group_miss = compute_group_missingness(patient_miss, static_with_age_group, "age_group")

        for row in group_miss.iter_rows(named=True):
            print(
                f"  {row['age_group']:30s} n={row['n_patients']:4d}  "
                f"mean={row['mean_missing_pct']:5.1f}%  "
                f"median={row['median_missing_pct']:5.1f}%  "
                f"range=[{row['min_missing_pct']:5.1f}%, {row['max_missing_pct']:5.1f}%]"
            )

    # 6c. Length of stay groups
    if "los_days" in static_df.columns:
        print("\n--- By LENGTH OF STAY ---")
        static_with_los_group = static_df.with_columns(
            pl.when(pl.col("los_days") < 1)
            .then(pl.lit("<1 day"))
            .when(pl.col("los_days") < 3)
            .then(pl.lit("1-3 days"))
            .when(pl.col("los_days") < 7)
            .then(pl.lit("3-7 days"))
            .otherwise(pl.lit("7+ days"))
            .alias("los_group")
        )
        group_miss = compute_group_missingness(patient_miss, static_with_los_group, "los_group")

        for row in group_miss.iter_rows(named=True):
            print(
                f"  {row['los_group']:30s} n={row['n_patients']:4d}  "
                f"mean={row['mean_missing_pct']:5.1f}%  "
                f"median={row['median_missing_pct']:5.1f}%  "
                f"range=[{row['min_missing_pct']:5.1f}%, {row['max_missing_pct']:5.1f}%]"
            )

    # 6d. Race (if available)
    if "race" in static_df.columns:
        print("\n--- By RACE ---")
        group_miss = compute_group_missingness(patient_miss, static_df, "race")

        for row in group_miss.iter_rows(named=True):
            race = row["race"] or "Unknown"
            # Truncate long race names
            race_display = race[:30] if len(race) > 30 else race
            print(
                f"  {race_display:30s} n={row['n_patients']:4d}  "
                f"mean={row['mean_missing_pct']:5.1f}%  "
                f"median={row['median_missing_pct']:5.1f}%"
            )

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"Overall missingness: {overall['missing_pct']:.1f}%")
    print(f"Patient missingness range: {stats['min']:.1f}% - {stats['max']:.1f}%")
    print(
        f"Most complete feature: {feature_miss.tail(1).row(0, named=True)['feature']} "
        f"({feature_miss.tail(1).row(0, named=True)['missing_pct']:.1f}% missing)"
    )
    print(
        f"Most missing feature: {feature_miss.head(1).row(0, named=True)['feature']} "
        f"({feature_miss.head(1).row(0, named=True)['missing_pct']:.1f}% missing)"
    )


if __name__ == "__main__":
    main()
