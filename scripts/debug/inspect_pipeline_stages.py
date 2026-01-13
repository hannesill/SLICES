"""CLI for inspecting pipeline stages with side-by-side data comparison.

This tool re-runs the extraction pipeline for a small set of sentinel patients
and captures data at each transformation stage:

    Stage 0: RAW         - Direct from DuckDB query (source parquet files)
    Stage 1: TRANSFORMED - After value transforms (e.g., Fahrenheit → Celsius)
    Stage 2: BINNED      - After hourly aggregation
    Stage 3: DENSE       - Final tensor format (fixed-length arrays)

The output includes:
    - Per-patient CSVs for each stage (easy to diff)
    - Combined stage CSVs (all patients per stage)
    - HTML report with side-by-side comparison

Example usage:
    # Using sentinel selection from existing processed data
    uv run python scripts/debug/inspect_pipeline_stages.py \
        parquet_root=/path/to/mimic-iv-parquet \
        processed_dir=data/processed/mimic-iv

    # Using specific stay IDs
    uv run python scripts/debug/inspect_pipeline_stages.py \
        parquet_root=/path/to/mimic-iv-parquet \
        'stay_ids=[30118103,30145082,30167837]'

    # With custom output directory and seed
    uv run python scripts/debug/inspect_pipeline_stages.py \
        parquet_root=/path/to/mimic-iv-parquet \
        processed_dir=data/processed/mimic-iv \
        output_dir=my_debug_output \
        sentinel.seed=123
"""

from pathlib import Path
from typing import List, Optional

import hydra
import polars as pl
from omegaconf import DictConfig
from rich.console import Console

console = Console()


def select_sentinels_from_processed(
    processed_dir: Path,
    seed: int = 42,
) -> List[int]:
    """Select sentinel patients from existing processed data.

    Uses the default 8 sentinel slots covering edge cases.

    Args:
        processed_dir: Path to processed data directory.
        seed: Random seed for reproducible selection.

    Returns:
        List of selected stay IDs.
    """
    from slices.debug import select_sentinel_patients

    static_df = pl.read_parquet(processed_dir / "static.parquet")

    labels_path = processed_dir / "labels.parquet"
    labels_df = pl.read_parquet(labels_path) if labels_path.exists() else None

    timeseries_path = processed_dir / "timeseries.parquet"
    timeseries_df = pl.read_parquet(timeseries_path) if timeseries_path.exists() else None

    sentinels_df = select_sentinel_patients(
        static_df,
        config=None,  # Use default 8 slots
        labels_df=labels_df,
        timeseries_df=timeseries_df,
    )

    return sentinels_df["stay_id"].to_list()


def get_stay_ids(cfg: DictConfig, processed_dir: Optional[Path]) -> List[int]:
    """Determine which stay IDs to use for stage capture.

    Priority:
    1. Explicit stay_ids in config
    2. Sentinel selection from processed_dir
    3. Error if neither available

    Args:
        cfg: Hydra config.
        processed_dir: Optional path to processed data.

    Returns:
        List of stay IDs.
    """
    # Check for explicit stay_ids
    if cfg.get("stay_ids"):
        stay_ids = list(cfg.stay_ids)
        console.print(f"Using {len(stay_ids)} specified stay IDs")
        return stay_ids

    # Try sentinel selection from processed data
    if processed_dir and processed_dir.exists():
        seed = cfg.get("sentinel", {}).get("seed", 42)
        console.print(f"Selecting sentinel patients from {processed_dir} (seed={seed})")
        stay_ids = select_sentinels_from_processed(processed_dir, seed=seed)
        console.print(f"Selected {len(stay_ids)} sentinel patients")
        return stay_ids

    raise ValueError(
        "Must provide either:\n"
        "  1. stay_ids=[...] - explicit list of stay IDs\n"
        "  2. processed_dir=path/to/processed - for sentinel selection\n"
        "Example: uv run python scripts/debug/inspect_pipeline_stages.py "
        "parquet_root=/data/mimic 'stay_ids=[30118103,30145082]'"
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig) -> None:
    """Run staged extraction pipeline with data capture at each stage."""
    console.print("=" * 70)
    console.print("[bold blue]Pipeline Stage Inspection Tool[/bold blue]")
    console.print("=" * 70)

    # Validate required config
    if not cfg.get("parquet_root"):
        raise ValueError(
            "parquet_root is required. Example:\n"
            "  uv run python scripts/debug/inspect_pipeline_stages.py "
            "parquet_root=/path/to/mimic-iv-parquet"
        )

    parquet_root = Path(cfg.parquet_root)
    if not parquet_root.exists():
        raise ValueError(f"parquet_root does not exist: {parquet_root}")

    # Determine processed_dir (for sentinel selection)
    processed_dir = Path(cfg.processed_dir) if cfg.get("processed_dir") else None

    # Determine output directory
    if cfg.get("output_dir"):
        output_dir = Path(cfg.output_dir)
    elif processed_dir:
        output_dir = processed_dir / "staged_snapshots"
    else:
        output_dir = Path("staged_snapshots")

    console.print(f"\nParquet root: {parquet_root}")
    console.print(f"Output directory: {output_dir}")
    if processed_dir:
        console.print(f"Processed directory: {processed_dir}")

    # Get stay IDs
    console.print("\n[bold]Selecting patients...[/bold]")
    stay_ids = get_stay_ids(cfg, processed_dir)
    console.print(f"Stay IDs: {stay_ids}")

    # Import here to avoid slow startup
    from slices.data.extractors.base import ExtractorConfig
    from slices.debug.debug_extractor import DebugMIMICIVExtractor

    # Configure extraction
    feature_set = cfg.get("feature_set", "core")
    seq_length_hours = cfg.get("seq_length_hours", 48)

    # Note: output_dir in config is not used since we don't save processed files
    # during debug extraction, but ExtractorConfig requires it
    config = ExtractorConfig(
        parquet_root=str(parquet_root),
        output_dir=str(output_dir),
        feature_set=feature_set,
        seq_length_hours=seq_length_hours,
        tasks=[],  # No label extraction needed for debug
    )

    # Create debug extractor
    console.print("\n[bold]Creating debug extractor...[/bold]")
    console.print(f"Feature set: {feature_set}")
    console.print(f"Sequence length: {seq_length_hours} hours")
    console.print(f"Extracting ONLY {len(stay_ids)} sentinel patients (fast mode)")

    extractor = DebugMIMICIVExtractor(
        config=config,
        capture_stay_ids=stay_ids,
    )

    # Run extraction with stage capture - ONLY for sentinel patients
    captures = extractor.run_with_stage_capture()

    # Export snapshots
    console.print("\n[bold]Exporting staged snapshots...[/bold]")
    exported = extractor.export_staged_snapshots(output_dir, generate_report=True)

    # Print summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]Staged extraction complete![/bold green]")
    console.print("=" * 70)

    summary = captures.get_summary()
    console.print(f"\nCaptured {summary['n_patients']} patients:")

    for stay_id, patient_summary in summary.get("per_patient", {}).items():
        console.print(f"\n  [bold]Stay {stay_id}:[/bold]")
        for stage, stats in patient_summary.items():
            rows = stats.get("rows", 0)
            console.print(f"    {stage.upper():12s}: {rows:6d} rows")

    console.print("\n[bold]Output files:[/bold]")
    console.print(f"  Per-patient CSVs: {output_dir / 'per_patient'}")
    console.print(f"  Combined stages:  {output_dir / 'stages'}")
    console.print(f"  HTML report:      {exported.get('html_report', 'N/A')}")
    console.print(f"  Summary YAML:     {exported.get('summary', 'N/A')}")

    console.print("\n[bold]To view the HTML report:[/bold]")
    console.print(
        f"  open {exported.get('html_report', output_dir / 'stage_comparison_report.html')}"
    )


if __name__ == "__main__":
    main()
