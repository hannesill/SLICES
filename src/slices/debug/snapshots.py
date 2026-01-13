"""Pipeline snapshot export for debugging.

Exports intermediate data at different extraction stages to CSV for
manual inspection and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import yaml


class PipelineStage(str, Enum):
    """Data pipeline stages for snapshot capture."""

    STAYS = "stays"
    RAW_EVENTS = "raw_events"
    HOURLY_BINNED = "hourly_binned"
    DENSE_TIMESERIES = "dense"
    LABELS = "labels"
    FINAL = "final"


@dataclass
class SnapshotConfig:
    """Configuration for pipeline snapshots.

    Attributes:
        output_dir: Directory to write CSV snapshots.
        stages: Which stages to capture (default: all).
        stay_ids: Specific stay IDs to capture (None = all).
        max_hours: Max hours to include in timeseries exports.
        include_masks: Whether to export observation masks.
        flatten_arrays: Whether to flatten nested arrays for CSV.
    """

    output_dir: Union[str, Path] = "debug_snapshots"
    stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))
    stay_ids: Optional[List[int]] = None
    max_hours: int = 48
    include_masks: bool = True
    flatten_arrays: bool = True


@dataclass
class PipelineSnapshot:
    """Container for captured pipeline data at a stage.

    Attributes:
        stage: Pipeline stage this snapshot represents.
        data: The DataFrame at this stage.
        metadata: Additional context (feature names, query info, etc.).
        timestamp: When snapshot was captured.
    """

    stage: PipelineStage
    data: pl.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Snapshot Capture Functions
# ---------------------------------------------------------------------------


def capture_stays_snapshot(
    stays_df: pl.DataFrame,
    stay_ids: Optional[List[int]] = None,
) -> PipelineSnapshot:
    """Capture snapshot after extract_stays().

    Args:
        stays_df: Full stays DataFrame from extractor.
        stay_ids: Optional filter to specific stays.

    Returns:
        PipelineSnapshot with filtered stays data.
    """
    df = stays_df.clone()
    if stay_ids is not None:
        df = df.filter(pl.col("stay_id").is_in(stay_ids))

    return PipelineSnapshot(
        stage=PipelineStage.STAYS,
        data=df,
        metadata={
            "n_stays": len(df),
            "columns": df.columns,
        },
    )


def capture_labels_snapshot(
    labels_df: pl.DataFrame,
    stay_ids: Optional[List[int]] = None,
) -> PipelineSnapshot:
    """Capture snapshot of labels.

    Args:
        labels_df: Labels DataFrame.
        stay_ids: Optional filter to specific stays.

    Returns:
        PipelineSnapshot with labels data.
    """
    df = labels_df.clone()
    if stay_ids is not None:
        df = df.filter(pl.col("stay_id").is_in(stay_ids))

    # Compute label statistics
    label_cols = [c for c in df.columns if c != "stay_id"]
    label_stats = {}
    for col in label_cols:
        if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
            label_stats[col] = {
                "mean": float(df[col].mean()) if df[col].mean() is not None else None,
                "sum": int(df[col].sum()) if df[col].sum() is not None else None,
                "null_count": int(df[col].null_count()),
            }

    return PipelineSnapshot(
        stage=PipelineStage.LABELS,
        data=df,
        metadata={
            "n_stays": len(df),
            "label_columns": label_cols,
            "label_stats": label_stats,
        },
    )


def capture_dense_snapshot(
    dense_df: pl.DataFrame,
    feature_names: List[str],
    stay_ids: Optional[List[int]] = None,
    flatten: bool = True,
) -> PipelineSnapshot:
    """Capture snapshot after _create_dense_timeseries().

    Args:
        dense_df: Dense timeseries with nested arrays.
        feature_names: List of feature names for column headers.
        stay_ids: Optional filter.
        flatten: Whether to flatten arrays to long format for CSV.

    Returns:
        PipelineSnapshot with dense data (optionally flattened).
    """
    df = dense_df.clone()
    if stay_ids is not None:
        df = df.filter(pl.col("stay_id").is_in(stay_ids))

    if flatten and len(df) > 0:
        df = flatten_dense_timeseries(df, feature_names)

    return PipelineSnapshot(
        stage=PipelineStage.DENSE_TIMESERIES,
        data=df,
        metadata={
            "n_stays": (
                dense_df.filter(pl.col("stay_id").is_in(stay_ids)).height
                if stay_ids
                else dense_df.height
            ),
            "feature_names": feature_names,
            "flattened": flatten,
        },
    )


# ---------------------------------------------------------------------------
# Flatten Functions
# ---------------------------------------------------------------------------


def flatten_dense_timeseries(
    dense_df: pl.DataFrame,
    feature_names: List[str],
    timeseries_col: str = "timeseries",
    mask_col: str = "mask",
) -> pl.DataFrame:
    """Flatten nested timeseries arrays to long format for CSV export.

    Converts from:
        stay_id | timeseries (list[list]) | mask (list[list])
    To:
        stay_id | hour | feature_name | value | observed

    Args:
        dense_df: Dense timeseries DataFrame.
        feature_names: Feature names for unpacking.
        timeseries_col: Column name for timeseries data.
        mask_col: Column name for observation mask.

    Returns:
        Long-format DataFrame suitable for CSV export.
    """
    rows = []

    for row in dense_df.iter_rows(named=True):
        stay_id = row["stay_id"]
        timeseries = np.array(row[timeseries_col])
        mask = np.array(row[mask_col]) if mask_col in row else None

        n_hours, n_features = timeseries.shape

        for hour in range(n_hours):
            for feat_idx, feat_name in enumerate(feature_names):
                value = timeseries[hour, feat_idx]
                observed = bool(mask[hour, feat_idx]) if mask is not None else None

                rows.append(
                    {
                        "stay_id": stay_id,
                        "hour": hour,
                        "feature_name": feat_name,
                        "value": float(value) if not np.isnan(value) else None,
                        "observed": observed,
                    }
                )

    return pl.DataFrame(rows)


def unflatten_timeseries(
    flat_df: pl.DataFrame,
    feature_names: List[str],
    seq_length: int = 48,
) -> pl.DataFrame:
    """Convert flattened timeseries back to nested format.

    Args:
        flat_df: Long-format DataFrame from flatten_dense_timeseries.
        feature_names: Feature names in order.
        seq_length: Sequence length (number of hours).

    Returns:
        Dense format DataFrame with nested arrays.
    """
    stay_ids = flat_df["stay_id"].unique().to_list()
    n_features = len(feature_names)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    results = []
    for stay_id in stay_ids:
        stay_data = flat_df.filter(pl.col("stay_id") == stay_id)

        timeseries = np.full((seq_length, n_features), np.nan)
        mask = np.zeros((seq_length, n_features), dtype=bool)

        for row in stay_data.iter_rows(named=True):
            hour = row["hour"]
            feat_idx = feature_to_idx.get(row["feature_name"])
            if feat_idx is not None and hour < seq_length:
                if row["value"] is not None:
                    timeseries[hour, feat_idx] = row["value"]
                if row["observed"] is not None:
                    mask[hour, feat_idx] = row["observed"]

        results.append(
            {
                "stay_id": stay_id,
                "timeseries": timeseries.tolist(),
                "mask": mask.tolist(),
            }
        )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------


def export_snapshot(
    snapshot: PipelineSnapshot,
    output_dir: Path,
    prefix: str = "",
) -> Path:
    """Export a single snapshot to CSV.

    Args:
        snapshot: PipelineSnapshot to export.
        output_dir: Output directory.
        prefix: Optional filename prefix (e.g., "sentinel_").

    Returns:
        Path to the exported CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}{snapshot.stage.value}.csv"
    filepath = output_dir / filename

    snapshot.data.write_csv(filepath)

    # Also save metadata
    meta_path = output_dir / f"{prefix}{snapshot.stage.value}_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(
            {
                "stage": snapshot.stage.value,
                "timestamp": snapshot.timestamp,
                **snapshot.metadata,
            },
            f,
            default_flow_style=False,
        )

    return filepath


def export_all_snapshots(
    snapshots: Dict[PipelineStage, PipelineSnapshot],
    config: SnapshotConfig,
) -> Dict[PipelineStage, Path]:
    """Export all captured snapshots to CSV.

    Args:
        snapshots: Dict mapping stages to snapshots.
        config: Snapshot configuration.

    Returns:
        Dict mapping stages to exported file paths.
    """
    output_dir = Path(config.output_dir)
    exported = {}

    for stage, snapshot in snapshots.items():
        if stage in config.stages:
            path = export_snapshot(snapshot, output_dir)
            exported[stage] = path

    return exported


# ---------------------------------------------------------------------------
# Standalone Snapshot from Processed Data
# ---------------------------------------------------------------------------


def create_snapshots_from_processed(
    processed_dir: Path,
    stay_ids: List[int],
    output_dir: Path,
    include_labels: bool = True,
    flatten_timeseries: bool = True,
) -> Dict[str, Path]:
    """Create CSV snapshots from already-processed data.

    This is useful when you want to inspect the final processed data
    without re-running extraction.

    Args:
        processed_dir: Directory with static.parquet, timeseries.parquet, labels.parquet.
        stay_ids: Stay IDs to export.
        output_dir: Where to write CSV files.
        include_labels: Whether to include labels.
        flatten_timeseries: Whether to flatten timeseries to long format.

    Returns:
        Dict mapping snapshot names to file paths.
    """
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    # Load metadata for feature names
    metadata_path = processed_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        feature_names = metadata.get("feature_names", [])
    else:
        feature_names = []

    # Static data
    static_path = processed_dir / "static.parquet"
    if static_path.exists():
        static_df = pl.read_parquet(static_path)
        snapshot = capture_stays_snapshot(static_df, stay_ids)
        exported["static"] = export_snapshot(snapshot, output_dir, prefix="sentinel_")

    # Timeseries data
    timeseries_path = processed_dir / "timeseries.parquet"
    if timeseries_path.exists():
        timeseries_df = pl.read_parquet(timeseries_path)
        snapshot = capture_dense_snapshot(
            timeseries_df,
            feature_names=feature_names,
            stay_ids=stay_ids,
            flatten=flatten_timeseries,
        )
        exported["timeseries"] = export_snapshot(snapshot, output_dir, prefix="sentinel_")

    # Labels data
    if include_labels:
        labels_path = processed_dir / "labels.parquet"
        if labels_path.exists():
            labels_df = pl.read_parquet(labels_path)
            snapshot = capture_labels_snapshot(labels_df, stay_ids)
            exported["labels"] = export_snapshot(snapshot, output_dir, prefix="sentinel_")

    # Write summary
    summary = {
        "stay_ids": stay_ids,
        "n_stays": len(stay_ids),
        "processed_dir": str(processed_dir),
        "timestamp": datetime.now().isoformat(),
        "files": {k: str(v) for k, v in exported.items()},
    }
    summary_path = output_dir / "snapshot_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, default_flow_style=False)

    exported["summary"] = summary_path

    return exported


# ---------------------------------------------------------------------------
# Mixin for Extractor Integration (Optional)
# ---------------------------------------------------------------------------


class SnapshotMixin:
    """Mixin to add snapshot capabilities to BaseExtractor.

    This is designed to be used as a mixin class that adds snapshot
    capture hooks to the extraction pipeline without modifying base.py.

    Usage:
        class DebuggableExtractor(SnapshotMixin, MIMICIVExtractor):
            pass

        extractor = DebuggableExtractor(config)
        extractor.enable_snapshots(snapshot_config)
        extractor.run()
        snapshots = extractor.get_snapshots()
    """

    _snapshot_config: Optional[SnapshotConfig] = None
    _snapshots: Dict[PipelineStage, PipelineSnapshot] = {}

    def enable_snapshots(self, config: SnapshotConfig) -> None:
        """Enable snapshot capture for this extractor run.

        Args:
            config: Snapshot configuration.
        """
        self._snapshot_config = config
        self._snapshots = {}

    def capture_snapshot(
        self,
        stage: PipelineStage,
        data: pl.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture data at a pipeline stage (called internally).

        Args:
            stage: Pipeline stage identifier.
            data: DataFrame to capture.
            metadata: Optional additional metadata.
        """
        if self._snapshot_config is None:
            return

        if stage not in self._snapshot_config.stages:
            return

        # Filter to specific stay_ids if configured
        df = data.clone()
        if self._snapshot_config.stay_ids is not None:
            if "stay_id" in df.columns:
                df = df.filter(pl.col("stay_id").is_in(self._snapshot_config.stay_ids))

        self._snapshots[stage] = PipelineSnapshot(
            stage=stage,
            data=df,
            metadata=metadata or {},
        )

    def get_snapshots(self) -> Dict[PipelineStage, PipelineSnapshot]:
        """Get all captured snapshots.

        Returns:
            Dict mapping stages to captured snapshots.
        """
        return self._snapshots.copy()

    def export_snapshots(self) -> Dict[PipelineStage, Path]:
        """Export all captured snapshots to disk.

        Returns:
            Dict mapping stages to exported file paths.
        """
        if self._snapshot_config is None:
            raise ValueError("Snapshots not enabled. Call enable_snapshots() first.")

        return export_all_snapshots(self._snapshots, self._snapshot_config)
