"""Debug extractor that captures data at each pipeline stage.

This module provides a specialized extractor that subclasses MIMICIVExtractor
and captures intermediate data at each transformation stage for debugging.

Example:
    >>> from slices.debug import DebugMIMICIVExtractor
    >>> from slices.data.extractors.base import ExtractorConfig
    >>>
    >>> config = ExtractorConfig(parquet_root="/path/to/mimic-iv")
    >>> extractor = DebugMIMICIVExtractor(
    ...     config=config,
    ...     capture_stay_ids=[30118103, 30145082],
    ... )
    >>>
    >>> # Run extraction with stage capture
    >>> captures = extractor.run_with_stage_capture()
    >>>
    >>> # Export to CSVs
    >>> extractor.export_staged_snapshots(Path("debug_output"))
"""

from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from rich.console import Console

from slices.data.config_schemas import TimeSeriesConceptConfig
from slices.data.extractors.base import ExtractorConfig
from slices.data.extractors.mimic_iv import MIMICIVExtractor
from slices.data.value_transforms import get_transform

from .staged_snapshots import (
    ExtractionStage,
    MultiStageCapture,
    filter_to_stay_ids,
    flatten_binned_to_long,
    generate_html_report,
)

console = Console()


class DebugMIMICIVExtractor(MIMICIVExtractor):
    """MIMIC-IV extractor that captures data at each pipeline stage.

    This subclass hooks into key extraction methods to capture intermediate
    data for debugging and validation. Data is captured only for specified
    sentinel stay IDs to keep output manageable.

    The captured stages are:
        - RAW: After DuckDB query, before transforms
        - TRANSFORMED: After value transforms (e.g., F→C)
        - BINNED: After hourly aggregation (sparse format)
        - DENSE: Final tensor format

    Attributes:
        capture_stay_ids: Set of stay IDs to capture data for.
        stage_captures: MultiStageCapture container with all captured data.
    """

    def __init__(
        self,
        config: ExtractorConfig,
        capture_stay_ids: Optional[List[int]] = None,
    ) -> None:
        """Initialize debug extractor.

        Args:
            config: Extractor configuration.
            capture_stay_ids: List of stay IDs to capture data for.
                If None, no data is captured (acts like regular extractor).
        """
        super().__init__(config)
        self.capture_stay_ids: set[int] = set(capture_stay_ids or [])
        self.stage_captures = MultiStageCapture()
        self._pre_transform_cache: Dict[int, pl.DataFrame] = {}

    def _extract_by_itemid_batch(
        self,
        table: str,
        value_col: str,
        time_col: str,
        transform: Optional[str],
        itemids: List[int],
        itemid_to_feature: Dict[int, str],
        stay_ids: List[int],
    ) -> pl.DataFrame:
        """Extract events with stage capture for raw and transformed data.

        Overrides parent to capture data before and after transforms.
        """
        # Map table names to schema paths (duplicated from parent for access)
        table_to_path = {
            "chartevents": ("icu", "chartevents"),
            "labevents": ("hosp", "labevents"),
            "outputevents": ("icu", "outputevents"),
            "inputevents": ("icu", "inputevents"),
        }

        if table not in table_to_path:
            raise ValueError(
                f"Unsupported table '{table}' for MIMIC-IV extractor. "
                f"Supported: {list(table_to_path.keys())}"
            )

        schema, table_name = table_to_path[table]
        parquet_path = self._parquet_path(schema, table_name)
        stay_ids_str = ",".join(map(str, stay_ids))
        itemids_str = ",".join(map(str, itemids))

        # labevents needs join with icustays
        if table == "labevents":
            icustays_path = self._parquet_path("icu", "icustays")
            sql = f"""
            SELECT
                i.stay_id,
                l.{time_col} AS charttime,
                l.itemid,
                CAST(l.{value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}') AS l
            INNER JOIN
                read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE
                i.stay_id IN ({stay_ids_str})
                AND l.itemid IN ({itemids_str})
                AND l.{value_col} IS NOT NULL
                AND l.{time_col} >= i.intime
                AND l.{time_col} <= i.outtime
            ORDER BY
                i.stay_id, l.{time_col}
            """
        else:
            sql = f"""
            SELECT
                stay_id,
                {time_col} AS charttime,
                itemid,
                CAST({value_col} AS DOUBLE) AS valuenum
            FROM
                read_parquet('{parquet_path}')
            WHERE
                stay_id IN ({stay_ids_str})
                AND itemid IN ({itemids_str})
                AND {value_col} IS NOT NULL
            ORDER BY
                stay_id, {time_col}
            """

        raw_events = self._query(sql)

        if raw_events.is_empty():
            return pl.DataFrame(
                schema={
                    "stay_id": pl.Int64,
                    "charttime": pl.Datetime,
                    "feature_name": pl.Utf8,
                    "valuenum": pl.Float64,
                }
            )

        # Map itemid to feature_name BEFORE transform capture
        raw_events = raw_events.with_columns(
            pl.col("itemid").replace_strict(itemid_to_feature, default=None).alias("feature_name")
        )

        # CAPTURE STAGE 0: RAW (before transforms)
        if self.capture_stay_ids:
            self._capture_raw_stage(raw_events, table)

        # Apply transform if specified
        if transform:
            transform_func = get_transform(transform)
            try:
                raw_events = raw_events.with_columns(
                    transform_func(pl.col("valuenum")).alias("valuenum")
                )
            except TypeError:
                raw_events = transform_func(raw_events, {"itemid": itemids})

            # CAPTURE STAGE 1: TRANSFORMED (after transforms)
            if self.capture_stay_ids:
                self._capture_transformed_stage(raw_events, table, transform)

        # Select final columns
        raw_events = raw_events.select(["stay_id", "charttime", "feature_name", "valuenum"])

        return raw_events

    def _capture_raw_stage(self, df: pl.DataFrame, table: str) -> None:
        """Capture raw stage data for sentinel patients."""
        filtered = filter_to_stay_ids(df, list(self.capture_stay_ids))
        if filtered.is_empty():
            return

        # Group by stay_id and add to captures
        for stay_id in self.capture_stay_ids:
            stay_data = filtered.filter(pl.col("stay_id") == stay_id)
            if not stay_data.is_empty():
                # Get existing data or create new
                existing = self.stage_captures.captures.get(stay_id)
                if existing and existing.raw:
                    # Append to existing raw data
                    combined = pl.concat([existing.raw.data, stay_data])
                    self.stage_captures.add_stage_data(
                        stay_id,
                        ExtractionStage.RAW,
                        combined,
                        {"tables": [table]},
                    )
                else:
                    self.stage_captures.add_stage_data(
                        stay_id,
                        ExtractionStage.RAW,
                        stay_data,
                        {"tables": [table]},
                    )

    def _capture_transformed_stage(self, df: pl.DataFrame, table: str, transform: str) -> None:
        """Capture transformed stage data for sentinel patients."""
        filtered = filter_to_stay_ids(df, list(self.capture_stay_ids))
        if filtered.is_empty():
            return

        for stay_id in self.capture_stay_ids:
            stay_data = filtered.filter(pl.col("stay_id") == stay_id)
            if not stay_data.is_empty():
                existing = self.stage_captures.captures.get(stay_id)
                if existing and existing.transformed:
                    combined = pl.concat([existing.transformed.data, stay_data])
                    self.stage_captures.add_stage_data(
                        stay_id,
                        ExtractionStage.TRANSFORMED,
                        combined,
                        {"tables": [table], "transform": transform},
                    )
                else:
                    self.stage_captures.add_stage_data(
                        stay_id,
                        ExtractionStage.TRANSFORMED,
                        stay_data,
                        {"tables": [table], "transform": transform},
                    )

    def _bin_to_hourly_grid(
        self,
        raw_events: pl.DataFrame,
        stay_ids: List[int],
        feature_mapping: Dict[str, TimeSeriesConceptConfig],
    ) -> pl.DataFrame:
        """Bin to hourly grid with stage capture.

        Overrides parent to capture binned data.
        """
        # Call parent implementation
        result = super()._bin_to_hourly_grid(raw_events, stay_ids, feature_mapping)

        # CAPTURE STAGE 2: BINNED
        if self.capture_stay_ids:
            feature_names = list(feature_mapping.keys())
            self._capture_binned_stage(result, feature_names)

        return result

    def _capture_binned_stage(self, df: pl.DataFrame, feature_names: List[str]) -> None:
        """Capture binned stage data for sentinel patients."""
        filtered = filter_to_stay_ids(df, list(self.capture_stay_ids))
        if filtered.is_empty():
            return

        for stay_id in self.capture_stay_ids:
            stay_data = filtered.filter(pl.col("stay_id") == stay_id)
            if not stay_data.is_empty():
                # Convert to long format for easier inspection
                long_format = flatten_binned_to_long(stay_data, feature_names)
                self.stage_captures.add_stage_data(
                    stay_id,
                    ExtractionStage.BINNED,
                    long_format,
                    {"feature_names": feature_names, "format": "long"},
                )

    def _create_dense_timeseries(
        self,
        sparse_timeseries: pl.DataFrame,
        stays: pl.DataFrame,
        feature_names: List[str],
    ) -> pl.DataFrame:
        """Create dense timeseries with stage capture.

        Overrides parent to capture dense data.
        """
        # Call parent implementation
        result = super()._create_dense_timeseries(sparse_timeseries, stays, feature_names)

        # CAPTURE STAGE 3: DENSE
        if self.capture_stay_ids:
            self._capture_dense_stage(result, feature_names)

        return result

    def _capture_dense_stage(self, df: pl.DataFrame, feature_names: List[str]) -> None:
        """Capture dense stage data for sentinel patients."""
        filtered = filter_to_stay_ids(df, list(self.capture_stay_ids))
        if filtered.is_empty():
            return

        # Import here to avoid circular dependency
        from .snapshots import flatten_dense_timeseries

        for stay_id in self.capture_stay_ids:
            stay_data = filtered.filter(pl.col("stay_id") == stay_id)
            if not stay_data.is_empty():
                # Flatten for CSV export
                flat = flatten_dense_timeseries(stay_data, feature_names)
                self.stage_captures.add_stage_data(
                    stay_id,
                    ExtractionStage.DENSE,
                    flat,
                    {"feature_names": feature_names, "format": "long"},
                )

    def run_with_stage_capture(self) -> MultiStageCapture:
        """Run extraction ONLY for sentinel patients and capture stage data.

        Unlike run(), this only extracts data for the configured capture_stay_ids,
        making it much faster for debugging purposes.

        Returns:
            MultiStageCapture with all captured stage data.
        """
        if not self.capture_stay_ids:
            raise ValueError(
                "capture_stay_ids must be set to use run_with_stage_capture(). "
                "Pass capture_stay_ids to the constructor."
            )

        console.print("\n[bold blue]Debug Extraction (Sentinel Only)[/bold blue]")
        console.print(f"Extracting ONLY {len(self.capture_stay_ids)} sentinel patients")
        console.print(f"Stay IDs: {list(self.capture_stay_ids)}")

        # Get stay metadata (needed for intime calculation in binning)
        stays = self.extract_stays()

        # Filter to ONLY our sentinel patients
        sentinel_stays = stays.filter(pl.col("stay_id").is_in(list(self.capture_stay_ids)))

        if len(sentinel_stays) == 0:
            raise ValueError(
                f"None of the specified stay_ids {list(self.capture_stay_ids)} "
                "were found in the dataset."
            )

        console.print(f"Found {len(sentinel_stays)} of {len(self.capture_stay_ids)} sentinel stays")

        # Cache the filtered stays for binning
        self._stays_cache = sentinel_stays

        stay_ids = list(self.capture_stay_ids)

        # Load feature mapping
        feature_set = self.config.feature_set
        feature_mapping = self._get_feature_mapping_cached(feature_set)
        feature_names = list(feature_mapping.keys())

        console.print(f"Extracting {len(feature_names)} features: {feature_names[:5]}...")

        # Extract raw events ONLY for sentinel patients
        console.print("\n[bold]Stage 0-1: Extracting raw events...[/bold]")
        raw_events = self._extract_raw_events(stay_ids, feature_mapping)
        console.print(f"  Raw events: {len(raw_events)} rows")

        # Bin to hourly grid
        console.print("\n[bold]Stage 2: Binning to hourly grid...[/bold]")
        sparse_timeseries = self._bin_to_hourly_grid(raw_events, stay_ids, feature_mapping)
        console.print(f"  Binned: {len(sparse_timeseries)} rows")

        # Create dense timeseries
        console.print("\n[bold]Stage 3: Creating dense timeseries...[/bold]")
        dense_timeseries = self._create_dense_timeseries(
            sparse_timeseries, sentinel_stays, feature_names
        )
        console.print(f"  Dense: {len(dense_timeseries)} stays")

        # Add metadata
        self.stage_captures.metadata = {
            "parquet_root": str(self.parquet_root),
            "feature_set": self.config.feature_set,
            "seq_length_hours": self.config.seq_length_hours,
            "capture_stay_ids": list(self.capture_stay_ids),
            "feature_names": feature_names,
        }

        return self.stage_captures

    def export_staged_snapshots(
        self,
        output_dir: Path,
        generate_report: bool = True,
    ) -> Dict[str, Path]:
        """Export all captured stage data to CSVs and optional HTML report.

        Args:
            output_dir: Directory to write output files.
            generate_report: Whether to generate HTML comparison report.

        Returns:
            Dict mapping output types to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: Dict[str, Path] = {}

        # Export per-patient and combined stage CSVs
        self.stage_captures.export_all(output_dir)
        exported["per_patient_dir"] = output_dir / "per_patient"
        exported["stages_dir"] = output_dir / "stages"

        # Generate HTML report
        if generate_report:
            report_path = output_dir / "stage_comparison_report.html"
            generate_html_report(self.stage_captures, report_path)
            exported["html_report"] = report_path
            console.print(f"[green]Generated HTML report: {report_path}[/green]")

        # Write summary YAML
        import yaml

        summary = self.stage_captures.get_summary()
        summary_path = output_dir / "capture_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.safe_dump(summary, f, default_flow_style=False)
        exported["summary"] = summary_path

        return exported


def run_staged_extraction(
    parquet_root: Path,
    capture_stay_ids: List[int],
    output_dir: Path,
    feature_set: str = "core",
    seq_length_hours: int = 48,
) -> MultiStageCapture:
    """Convenience function to run staged extraction for specific patients.

    This is the main entry point for running a debug extraction that
    captures data at each pipeline stage. It ONLY extracts data for
    the specified patients, making it fast even on large datasets.

    Args:
        parquet_root: Path to MIMIC-IV parquet files.
        capture_stay_ids: List of stay IDs to capture data for.
        output_dir: Directory to write output files.
        feature_set: Feature set to extract (default: "core").
        seq_length_hours: Sequence length in hours (default: 48).

    Returns:
        MultiStageCapture with all captured data.

    Example:
        >>> captures = run_staged_extraction(
        ...     parquet_root=Path("/data/mimic-iv"),
        ...     capture_stay_ids=[30118103, 30145082],
        ...     output_dir=Path("debug_output"),
        ... )
        >>> print(captures.get_summary())
    """
    # Note: output_dir in config is not used since we don't save processed files
    # during debug extraction, but ExtractorConfig requires it
    config = ExtractorConfig(
        parquet_root=str(parquet_root),
        output_dir=str(output_dir),
        feature_set=feature_set,
        seq_length_hours=seq_length_hours,
        tasks=[],  # No label extraction needed for debug
    )

    extractor = DebugMIMICIVExtractor(
        config=config,
        capture_stay_ids=capture_stay_ids,
    )

    captures = extractor.run_with_stage_capture()
    extractor.export_staged_snapshots(output_dir)

    console.print("\n[bold green]Staged extraction complete![/bold green]")
    console.print(f"View report: {output_dir / 'stage_comparison_report.html'}")

    return captures
