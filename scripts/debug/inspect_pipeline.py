"""CLI for inspecting pipeline data and exporting snapshots.

Exports CSV files for sentinel patients at different pipeline stages
to enable manual inspection and validation.

By default, selects 8 sentinel patients covering key edge cases:
- Short stays (boundary conditions)
- Patients who died (label alignment)
- Clean vs sparse data (missingness extremes)
- Long stays (truncation testing)
- Young and elderly patients (demographic edges)

Example usage:
    # Export snapshots for auto-selected sentinel patients (8 by default)
    uv run python scripts/debug/inspect_pipeline.py \
        processed_dir=data/processed/mimic-iv-demo

    # Export for specific stay IDs
    uv run python scripts/debug/inspect_pipeline.py \
        processed_dir=data/processed/mimic-iv-demo \
        'stay_ids=[30118103,30145082]'
"""

from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig
from slices.debug import (
    create_snapshots_from_processed,
    select_sentinel_patients,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig) -> None:
    """Export pipeline snapshots for debugging."""
    print("=" * 70)
    print("Pipeline Snapshot Export")
    print("=" * 70)

    processed_dir = Path(cfg.processed_dir)
    output_dir = Path(cfg.get("output_dir") or processed_dir / "debug_snapshots")

    print(f"\nProcessed directory: {processed_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading data...")
    static_df = pl.read_parquet(processed_dir / "static.parquet")
    timeseries_df = pl.read_parquet(processed_dir / "timeseries.parquet")

    labels_path = processed_dir / "labels.parquet"
    labels_df = pl.read_parquet(labels_path) if labels_path.exists() else None

    print(f"  - Static: {len(static_df)} stays")
    print(f"  - Timeseries: {len(timeseries_df)} stays")
    if labels_df is not None:
        print(f"  - Labels: {len(labels_df)} stays")

    # Determine stay IDs to export
    if cfg.get("stay_ids"):
        # Use specified stay IDs
        stay_ids = list(cfg.stay_ids)
        print(f"\nUsing {len(stay_ids)} specified stay IDs")
        sentinels_df = None
    else:
        # Select sentinel patients using default 8 slots
        print("\nSelecting sentinel patients (8 edge cases)...")

        sentinels_df = select_sentinel_patients(
            static_df,
            config=None,  # Use default 8 slots
            labels_df=labels_df,
            timeseries_df=timeseries_df,
        )

        stay_ids = sentinels_df["stay_id"].to_list()
        print(f"  Selected {len(stay_ids)} sentinel patients")

        # Show slot breakdown
        if "slot_name" in sentinels_df.columns:
            print("\n  Slots filled:")
            for row in sentinels_df.iter_rows(named=True):
                print(f"    - {row['slot_name']}: stay_id={row['stay_id']}")

    # Export snapshots
    print("\nExporting snapshots...")
    exported = create_snapshots_from_processed(
        processed_dir=processed_dir,
        stay_ids=stay_ids,
        output_dir=output_dir,
        include_labels=labels_df is not None,
        flatten_timeseries=cfg.get("flatten_timeseries", True),
    )

    print(f"\nExported {len(exported)} files to {output_dir}:")
    for name, path in exported.items():
        print(f"  - {name}: {path}")

    # Print summary of sentinel patients
    print("\n" + "=" * 70)
    print("Sentinel Patient Summary")
    print("=" * 70)

    sentinel_static = static_df.filter(pl.col("stay_id").is_in(stay_ids))
    print(f"\nSelected {len(sentinel_static)} patients:")

    # Show key stats
    for row in sentinel_static.iter_rows(named=True):
        stay_id = row["stay_id"]
        los = row.get("los_days", "N/A")
        age = row.get("age", "N/A")
        unit = row.get("first_careunit", "N/A")

        los_str = f"{los:.1f}d" if isinstance(los, (int, float)) else los
        print(f"  Stay {stay_id}: LOS={los_str}, Age={age}, Unit={unit}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print("\nTo inspect the exported data:")
    print(f"  - Static features: {output_dir}/sentinel_stays.csv")
    print(f"  - Timeseries (long format): {output_dir}/sentinel_dense.csv")
    if labels_df is not None:
        print(f"  - Labels: {output_dir}/sentinel_labels.csv")


if __name__ == "__main__":
    main()
