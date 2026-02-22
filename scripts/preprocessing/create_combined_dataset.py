"""Create a combined dataset by merging two processed datasets.

Merges MIMIC-IV and eICU (or any two datasets) into a single processed
directory for combined pretraining. Handles stay_id collision by adding
a dataset-specific offset to ensure globally unique IDs.

Usage:
    uv run python scripts/preprocessing/create_combined_dataset.py \
        --source data/processed/miiv data/processed/eicu \
        --output data/processed/combined

    # With custom dataset names (for metadata)
    uv run python scripts/preprocessing/create_combined_dataset.py \
        --source data/processed/miiv data/processed/eicu \
        --names miiv eicu \
        --output data/processed/combined
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

# Offset applied to stay_id and patient_id for the second dataset
# to avoid collisions. Large enough to exceed any real ID range.
DATASET_ID_OFFSET = 100_000_000


def load_dataset(processed_dir: Path) -> dict:
    """Load all parquet files and metadata from a processed directory."""
    static = pl.read_parquet(processed_dir / "static.parquet")
    timeseries = pl.read_parquet(processed_dir / "timeseries.parquet")
    labels = pl.read_parquet(processed_dir / "labels.parquet")

    metadata_path = processed_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = {}

    return {
        "static": static,
        "timeseries": timeseries,
        "labels": labels,
        "metadata": metadata,
    }


def offset_ids(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    """Add offset to stay_id and patient_id columns."""
    exprs = []
    if "stay_id" in df.columns:
        exprs.append((pl.col("stay_id") + offset).alias("stay_id"))
    if "patient_id" in df.columns:
        exprs.append((pl.col("patient_id") + offset).alias("patient_id"))
    if exprs:
        return df.with_columns(exprs)
    return df


def validate_no_id_collision(static_a: pl.DataFrame, static_b: pl.DataFrame) -> None:
    """Verify no stay_id overlap between two datasets."""
    ids_a = set(static_a["stay_id"].to_list())
    ids_b = set(static_b["stay_id"].to_list())
    overlap = ids_a & ids_b
    if overlap:
        raise ValueError(
            f"stay_id collision detected: {len(overlap)} overlapping IDs "
            f"(e.g., {list(overlap)[:5]}). This should not happen after "
            "applying the dataset offset."
        )


def validate_feature_compatibility(meta_a: dict, meta_b: dict) -> None:
    """Verify both datasets have the same feature set."""
    features_a = set(meta_a.get("feature_names", []))
    features_b = set(meta_b.get("feature_names", []))

    if features_a != features_b:
        only_a = features_a - features_b
        only_b = features_b - features_a
        raise ValueError(
            f"Feature mismatch between datasets.\n"
            f"  Only in dataset A: {only_a}\n"
            f"  Only in dataset B: {only_b}\n"
            "Both datasets must have the same features for combined training."
        )


def merge_labels(labels_a: pl.DataFrame, labels_b: pl.DataFrame) -> pl.DataFrame:
    """Merge label DataFrames, handling column mismatches."""
    cols_a = set(labels_a.columns)
    cols_b = set(labels_b.columns)

    # Add missing columns as null
    for col in cols_b - cols_a:
        dtype = labels_b[col].dtype
        labels_a = labels_a.with_columns(pl.lit(None).cast(dtype).alias(col))
    for col in cols_a - cols_b:
        dtype = labels_a[col].dtype
        labels_b = labels_b.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Ensure same column order
    col_order = sorted(labels_a.columns)
    labels_a = labels_a.select(col_order)
    labels_b = labels_b.select(col_order)

    return pl.concat([labels_a, labels_b])


def main():
    parser = argparse.ArgumentParser(
        description="Merge two processed datasets into a combined dataset."
    )
    parser.add_argument(
        "--source",
        nargs=2,
        required=True,
        help="Paths to two processed dataset directories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for combined dataset",
    )
    parser.add_argument(
        "--names",
        nargs=2,
        default=None,
        help="Dataset names (for metadata). Defaults to directory names.",
    )
    args = parser.parse_args()

    source_a = Path(args.source[0])
    source_b = Path(args.source[1])
    output_dir = Path(args.output)

    if not source_a.exists():
        print(f"Error: source directory not found: {source_a}")
        sys.exit(1)
    if not source_b.exists():
        print(f"Error: source directory not found: {source_b}")
        sys.exit(1)

    names = args.names or [source_a.name, source_b.name]

    print("Merging datasets:")
    print(f"  A: {source_a} ({names[0]})")
    print(f"  B: {source_b} ({names[1]})")
    print(f"  Output: {output_dir}")

    # Load datasets
    print("\nLoading dataset A...")
    data_a = load_dataset(source_a)
    print(f"  {len(data_a['static'])} stays, {len(data_a['timeseries'])} timeseries rows")

    print("Loading dataset B...")
    data_b = load_dataset(source_b)
    print(f"  {len(data_b['static'])} stays, {len(data_b['timeseries'])} timeseries rows")

    # Validate feature compatibility
    print("\nValidating feature compatibility...")
    validate_feature_compatibility(data_a["metadata"], data_b["metadata"])
    print("  Features match.")

    # Check for natural ID collisions
    ids_a = set(data_a["static"]["stay_id"].to_list())
    ids_b = set(data_b["static"]["stay_id"].to_list())
    natural_overlap = ids_a & ids_b

    if natural_overlap:
        print(
            f"\n  {len(natural_overlap)} stay_id collisions detected. "
            f"Applying offset of {DATASET_ID_OFFSET:,} to dataset B."
        )
        data_b["static"] = offset_ids(data_b["static"], DATASET_ID_OFFSET)
        data_b["timeseries"] = offset_ids(data_b["timeseries"], DATASET_ID_OFFSET)
        data_b["labels"] = offset_ids(data_b["labels"], DATASET_ID_OFFSET)
    else:
        print("  No stay_id collisions. No offset needed.")

    # Validate no collisions after offset
    validate_no_id_collision(data_a["static"], data_b["static"])

    # Add source_dataset column to static for tracking
    data_a["static"] = data_a["static"].with_columns(pl.lit(names[0]).alias("source_dataset"))
    data_b["static"] = data_b["static"].with_columns(pl.lit(names[1]).alias("source_dataset"))

    # Merge
    print("\nMerging...")
    static_combined = pl.concat([data_a["static"], data_b["static"]])
    timeseries_combined = pl.concat([data_a["timeseries"], data_b["timeseries"]])
    labels_combined = merge_labels(data_a["labels"], data_b["labels"])

    print(
        f"  Combined: {len(static_combined)} stays, " f"{len(timeseries_combined)} timeseries rows"
    )

    # Create combined metadata
    meta_a = data_a["metadata"]
    meta_b = data_b["metadata"]

    # Use common task names
    tasks_a = set(meta_a.get("task_names", []))
    tasks_b = set(meta_b.get("task_names", []))
    common_tasks = sorted(tasks_a & tasks_b)

    combined_metadata = {
        "dataset": "combined",
        "source_datasets": names,
        "source_dirs": [str(source_a), str(source_b)],
        "feature_set": meta_a.get("feature_set", "core"),
        "feature_names": meta_a.get("feature_names", []),
        "n_features": meta_a.get("n_features", 0),
        "seq_length_hours": meta_a.get("seq_length_hours", 48),
        "min_stay_hours": meta_a.get("min_stay_hours", 48),
        "task_names": common_tasks,
        "n_stays": len(static_combined),
        "stays_per_dataset": {
            names[0]: len(data_a["static"]),
            names[1]: len(data_b["static"]),
        },
        "id_offset_applied": DATASET_ID_OFFSET if natural_overlap else 0,
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving...")
    static_combined.write_parquet(output_dir / "static.parquet")
    print(f"  static.parquet ({len(static_combined)} rows)")

    timeseries_combined.write_parquet(output_dir / "timeseries.parquet")
    print(f"  timeseries.parquet ({len(timeseries_combined)} rows)")

    labels_combined.write_parquet(output_dir / "labels.parquet")
    print(f"  labels.parquet ({len(labels_combined)} rows)")

    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(combined_metadata, f, default_flow_style=False)
    print("  metadata.yaml")

    print(f"\nCombined dataset created at {output_dir}")
    print(f"  Total stays: {len(static_combined):,}")
    print(f"  Common tasks: {common_tasks}")

    if tasks_a != tasks_b:
        only_a = tasks_a - tasks_b
        only_b = tasks_b - tasks_a
        if only_a:
            print(
                f"  Tasks only in {names[0]}: {sorted(only_a)} "
                f"(labels will be null for {names[1]} stays)"
            )
        if only_b:
            print(
                f"  Tasks only in {names[1]}: {sorted(only_b)} "
                f"(labels will be null for {names[0]} stays)"
            )


if __name__ == "__main__":
    main()
