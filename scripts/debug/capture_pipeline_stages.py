"""Capture data at each pipeline stage (re-runs extraction for sentinel patients).

This tool re-runs the extraction pipeline for a small set of sentinel patients
and captures data at each transformation stage, enabling side-by-side comparison
to debug data transformations.

For quick inspection of already-processed data WITHOUT re-running extraction,
use export_snapshots.py instead.

Pipeline stages captured:

    Extraction stages (Extractor):
        Stage 0: RAW         - Direct from DuckDB query (source parquet files)
        Stage 1: TRANSFORMED - After value transforms (e.g., Fahrenheit → Celsius)
        Stage 2: BINNED      - After hourly aggregation (sparse grid format)
        Stage 3: DENSE       - Dense grid format (seq_length × n_features) with NaN

    Dataset stages (ICUDataset) - requires processed_dir:
        Stage 4: IMPUTED     - After imputation (no NaN, original scale)
        Stage 5: NORMALIZED  - After z-score normalization (model input)

The output includes:
    - Per-patient CSVs for each stage (easy to diff)
    - Combined stage CSVs (all patients per stage)
    - HTML report with side-by-side comparison

Example usage:
    # Full pipeline inspection (extraction + dataset stages)
    uv run python scripts/debug/capture_pipeline_stages.py \\
        parquet_root=/path/to/mimic-iv-parquet \\
        processed_dir=data/processed/mimic-iv

    # Using specific stay IDs
    uv run python scripts/debug/capture_pipeline_stages.py \\
        parquet_root=/path/to/mimic-iv-parquet \\
        processed_dir=data/processed/mimic-iv \\
        'stay_ids=[30118103,30145082,30167837]'

    # Extraction stages only (no processed_dir)
    uv run python scripts/debug/capture_pipeline_stages.py \\
        parquet_root=/path/to/mimic-iv-parquet \\
        'stay_ids=[30118103,30145082,30167837]'
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import hydra
import polars as pl
import torch
import yaml
from omegaconf import DictConfig
from rich.console import Console

if TYPE_CHECKING:
    from slices.debug.staged_snapshots import MultiStageCapture

console = Console()


def load_processed_metadata(processed_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metadata from processed data directory.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        Metadata dict if found, None otherwise.
    """
    metadata_path = processed_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return yaml.safe_load(f)


def tensor_to_grid_df(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    feature_names: List[str],
    stay_id: int,
) -> pl.DataFrame:
    """Convert a 2D tensor to grid-format DataFrame (hours × features).

    This shows the data exactly as the model sees it: each row is an hour,
    each column is a feature value.

    Args:
        tensor: Shape (seq_length, n_features).
        mask: Shape (seq_length, n_features), True = observed.
        feature_names: List of feature names.
        stay_id: The stay ID.

    Returns:
        DataFrame with columns: stay_id, hour, feat1, feat1_mask, feat2, feat2_mask, ...
    """
    seq_length, n_features = tensor.shape
    rows = []
    for hour in range(seq_length):
        row: Dict[str, any] = {
            "stay_id": stay_id,
            "hour": hour,
        }
        for feat_idx, feat_name in enumerate(feature_names):
            row[feat_name] = tensor[hour, feat_idx].item()
            row[f"{feat_name}_mask"] = mask[hour, feat_idx].item()
        rows.append(row)
    return pl.DataFrame(rows)


def capture_dataset_stages(
    processed_dir: Path,
    stay_ids: List[int],
    captures: "MultiStageCapture",
    impute_strategy: str = "forward_fill",
    normalize: bool = True,
    extraction_feature_names: Optional[List[str]] = None,
) -> None:
    """Capture dataset preprocessing stages (IMPUTED, NORMALIZED).

    Uses ICUDataset.get_preprocessing_stages() to capture the exact same
    transformations that are applied during training.

    Note: The GRID stage was removed as it's redundant with DENSE - both represent
    the same data structure (seq_length × n_features grid with NaN for missing).

    Args:
        processed_dir: Path to processed data directory.
        stay_ids: List of stay IDs to capture.
        captures: MultiStageCapture to add stage data to.
        impute_strategy: Imputation strategy (must match training config).
        normalize: Whether normalization is applied.
        extraction_feature_names: Feature names from extraction stages (0-3).
            If provided, validates that dataset features match.
    """
    from slices.data.dataset import ICUDataset
    from slices.debug.staged_snapshots import PipelineStage

    console.print("\n[bold]Stage 4-5: Capturing dataset preprocessing stages...[/bold]")
    console.print(f"  Loading ICUDataset from {processed_dir}")

    # Load dataset with same preprocessing settings as training
    dataset = ICUDataset(
        data_dir=processed_dir,
        task_name=None,  # No labels needed
        normalize=normalize,
        impute_strategy=impute_strategy,
    )

    feature_names = dataset.get_feature_names()
    console.print(f"  Features: {feature_names}")
    console.print(f"  Imputation: {impute_strategy}, Normalize: {normalize}")

    # Validate feature consistency between extraction and dataset stages
    if extraction_feature_names is not None:
        if set(extraction_feature_names) != set(feature_names):
            extraction_set = set(extraction_feature_names)
            dataset_set = set(feature_names)
            only_in_extraction = extraction_set - dataset_set
            only_in_dataset = dataset_set - extraction_set

            console.print("\n[bold red]WARNING: Feature mismatch between stages![/bold red]")
            if only_in_extraction:
                console.print(
                    f"  Features in extraction (0-3) but not in dataset (4-5): {only_in_extraction}"
                )
            if only_in_dataset:
                console.print(
                    f"  Features in dataset (4-5) but not in extraction (0-3): {only_in_dataset}"
                )
            console.print(
                "  [yellow]Stage comparisons may be misleading due to different features.[/yellow]"
            )
            console.print(
                "  [yellow]Ensure processed_dir was created with the same feature_set.[/yellow]\n"
            )
        elif extraction_feature_names != feature_names:
            # Same features but different order
            console.print(
                "  [yellow]Note: Feature order differs between extraction "
                "and dataset stages.[/yellow]"
            )
        else:
            console.print(
                "  [green]Feature consistency validated: extraction and dataset "
                "features match.[/green]"
            )

    # Find indices for our sentinel stay_ids
    for stay_id in stay_ids:
        if stay_id not in dataset.stay_id_to_idx:
            console.print(f"  [yellow]Warning: stay_id {stay_id} not found in dataset[/yellow]")
            continue

        idx = dataset.stay_id_to_idx[stay_id]

        # Get preprocessing stages using the dataset's method
        # This reuses the exact same preprocessing code as training
        stages = dataset.get_preprocessing_stages(idx)

        # Convert each stage to grid-format DataFrame (hours × features)
        # Skip 'grid' stage as it's redundant with DENSE (same data, just loaded into tensor)
        for stage_name, stage_data in stages.items():
            if stage_name == "grid":
                continue  # Skip redundant grid stage
            stage_enum = PipelineStage(stage_name)
            df = tensor_to_grid_df(
                tensor=stage_data["timeseries"],
                mask=stage_data["mask"],
                feature_names=feature_names,
                stay_id=stay_id,
            )
            captures.add_stage_data(
                stay_id=stay_id,
                stage=stage_enum,
                data=df,
                metadata={
                    "feature_names": feature_names,
                    "impute_strategy": impute_strategy,
                    "normalize": normalize,
                    "format": "grid",
                },
            )

        console.print(f"  Captured stages 4-5 for stay {stay_id}")

    console.print(f"  [green]Captured dataset stages for {len(stay_ids)} patients[/green]")


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

    # Load metadata from processed data to ensure feature consistency
    processed_metadata: Optional[Dict[str, Any]] = None
    if processed_dir and processed_dir.exists():
        processed_metadata = load_processed_metadata(processed_dir)
        if processed_metadata:
            console.print(
                f"  [green]Loaded metadata from processed data[/green]: "
                f"{processed_metadata.get('n_features', '?')} features, "
                f"{processed_metadata.get('seq_length_hours', '?')}h sequence"
            )
        else:
            console.print("  [yellow]Warning: No metadata.yaml found in processed_dir[/yellow]")

    # Get stay IDs
    console.print("\n[bold]Selecting patients...[/bold]")
    stay_ids = get_stay_ids(cfg, processed_dir)
    console.print(f"Stay IDs: {stay_ids}")

    # Import here to avoid slow startup
    from slices.data.extractors.base import ExtractorConfig
    from slices.debug.debug_extractor import DebugMIMICIVExtractor

    # Configure extraction - use processed data settings if available for consistency
    categories: Optional[List[str]] = None
    if processed_metadata:
        # Use settings from processed data to ensure stages 0-3 match stages 4-5
        feature_set = processed_metadata.get("feature_set", cfg.get("feature_set", "core"))
        seq_length_hours = processed_metadata.get(
            "seq_length_hours", cfg.get("seq_length_hours", 48)
        )
        categories = processed_metadata.get("categories")
        console.print(
            "\n[bold]Using extraction settings from processed data for consistency:[/bold]"
        )
    else:
        # Fall back to config settings (extraction-only mode)
        feature_set = cfg.get("feature_set", "core")
        seq_length_hours = cfg.get("seq_length_hours", 48)
        categories = cfg.get("categories")
        console.print("\n[bold]Using extraction settings from config:[/bold]")

    # Note: output_dir in config is not used since we don't save processed files
    # during debug extraction, but ExtractorConfig requires it
    config = ExtractorConfig(
        parquet_root=str(parquet_root),
        output_dir=str(output_dir),
        feature_set=feature_set,
        seq_length_hours=seq_length_hours,
        categories=categories,
        tasks=[],  # No label extraction needed for debug
    )

    # Create debug extractor
    console.print("\n[bold]Creating debug extractor...[/bold]")
    console.print(f"Feature set: {feature_set}")
    console.print(f"Sequence length: {seq_length_hours} hours")
    if categories:
        console.print(f"Categories: {categories}")
    console.print(f"Extracting ONLY {len(stay_ids)} sentinel patients (fast mode)")

    extractor = DebugMIMICIVExtractor(
        config=config,
        capture_stay_ids=stay_ids,
    )

    # Run extraction with stage capture - ONLY for sentinel patients
    captures = extractor.run_with_stage_capture()

    # Capture dataset preprocessing stages (4-5) if processed_dir is available
    if processed_dir and processed_dir.exists():
        impute_strategy = cfg.get("impute_strategy", "forward_fill")
        normalize = cfg.get("normalize", True)
        # Get extraction feature names for validation
        extraction_feature_names = captures.metadata.get("feature_names")
        capture_dataset_stages(
            processed_dir=processed_dir,
            stay_ids=stay_ids,
            captures=captures,
            impute_strategy=impute_strategy,
            normalize=normalize,
            extraction_feature_names=extraction_feature_names,
        )
    else:
        console.print("\n[yellow]Skipping dataset stages (4-5): no processed_dir provided[/yellow]")
        console.print("  To capture dataset stages, provide processed_dir=path/to/processed")

    # Export snapshots
    console.print("\n[bold]Exporting staged snapshots...[/bold]")
    exported = extractor.export_staged_snapshots(output_dir, generate_report=True)

    # Print summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]Pipeline stage capture complete![/bold green]")
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
