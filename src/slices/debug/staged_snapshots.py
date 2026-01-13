"""Multi-stage pipeline snapshots for debugging data transformations.

This module provides tools to capture and compare data at each stage of the
extraction pipeline, enabling visual inspection of how data is transformed:

    Stage 0: RAW       - DuckDB query results from source parquet files
    Stage 1: TRANSFORMED - After value transforms (e.g., Fahrenheit → Celsius)
    Stage 2: BINNED    - After hourly aggregation (sparse format)
    Stage 3: DENSE     - Final tensor format (fixed-length arrays)

Example:
    >>> from slices.debug.staged_snapshots import StagedExtractor, generate_html_report
    >>>
    >>> # Run extraction with stage capture
    >>> extractor = StagedExtractor(config, capture_stay_ids=[30118103, 30145082])
    >>> captures = extractor.run_with_captures()
    >>>
    >>> # Export CSVs and HTML report
    >>> extractor.export_staged_snapshots(output_dir)
    >>> generate_html_report(captures, output_dir / "report.html")
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


class ExtractionStage(str, Enum):
    """Stages in the extraction pipeline for multi-stage capture."""

    RAW = "raw"  # Stage 0: Direct from DuckDB query
    TRANSFORMED = "transformed"  # Stage 1: After value transforms
    BINNED = "binned"  # Stage 2: After hourly aggregation
    DENSE = "dense"  # Stage 3: Final tensor format

    @property
    def order(self) -> int:
        """Get the numeric order of this stage in the pipeline."""
        return {
            ExtractionStage.RAW: 0,
            ExtractionStage.TRANSFORMED: 1,
            ExtractionStage.BINNED: 2,
            ExtractionStage.DENSE: 3,
        }[self]

    @property
    def prefixed_name(self) -> str:
        """Get the stage name with numeric prefix (e.g., '0_raw')."""
        return f"{self.order}_{self.value}"


@dataclass
class StageData:
    """Container for data captured at a single pipeline stage.

    Attributes:
        stage: Which extraction stage this data is from.
        data: The DataFrame at this stage.
        metadata: Additional context (row counts, feature info, etc.).
        timestamp: When this was captured.
    """

    stage: ExtractionStage
    data: pl.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PatientStageCapture:
    """All stage data for a single patient.

    Attributes:
        stay_id: The ICU stay identifier.
        raw: Stage 0 data (from DuckDB query).
        transformed: Stage 1 data (after value transforms).
        binned: Stage 2 data (hourly aggregated).
        dense: Stage 3 data (fixed-length tensor).
    """

    stay_id: int
    raw: Optional[StageData] = None
    transformed: Optional[StageData] = None
    binned: Optional[StageData] = None
    dense: Optional[StageData] = None

    def get_stage(self, stage: ExtractionStage) -> Optional[StageData]:
        """Get data for a specific stage."""
        return getattr(self, stage.value, None)

    def set_stage(self, stage: ExtractionStage, data: StageData) -> None:
        """Set data for a specific stage."""
        setattr(self, stage.value, data)

    def to_csvs(self, output_dir: Path) -> Dict[str, Path]:
        """Export all stages to CSV files in a patient-specific directory.

        Files are prefixed with stage numbers to show pipeline order:
        0_raw.csv, 1_transformed.csv, 2_binned.csv, 3_dense.csv

        Args:
            output_dir: Base output directory.

        Returns:
            Dict mapping stage names to file paths.
        """
        patient_dir = output_dir / f"stay_{self.stay_id}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        exported = {}
        for stage in ExtractionStage:
            stage_data = self.get_stage(stage)
            if stage_data is not None and not stage_data.data.is_empty():
                path = patient_dir / f"{stage.prefixed_name}.csv"
                stage_data.data.write_csv(path)
                exported[stage.prefixed_name] = path

        return exported

    def get_stage_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for each captured stage.

        Returns:
            Dict mapping stage names to summary stats.
        """
        summary = {}
        for stage in ExtractionStage:
            stage_data = self.get_stage(stage)
            if stage_data is not None:
                df = stage_data.data
                stats: Dict[str, Any] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                }

                # Stage-specific stats
                if stage == ExtractionStage.RAW:
                    if "feature_name" in df.columns:
                        stats["n_features"] = df["feature_name"].n_unique()
                    if "charttime" in df.columns:
                        stats["time_range_hours"] = _compute_time_range(df)

                elif stage == ExtractionStage.TRANSFORMED:
                    if "feature_name" in df.columns:
                        stats["n_features"] = df["feature_name"].n_unique()
                    if "valuenum" in df.columns:
                        stats["value_range"] = (
                            float(df["valuenum"].min() or 0),
                            float(df["valuenum"].max() or 0),
                        )

                elif stage == ExtractionStage.BINNED:
                    if "hour" in df.columns:
                        stats["hour_range"] = (
                            int(df["hour"].min() or 0),
                            int(df["hour"].max() or 0),
                        )
                    # Count observed values
                    mask_cols = [c for c in df.columns if c.endswith("_mask")]
                    if mask_cols:
                        total_obs = sum(df[c].sum() or 0 for c in mask_cols)
                        stats["total_observations"] = int(total_obs)

                elif stage == ExtractionStage.DENSE:
                    if "hour" in df.columns:
                        stats["n_hours"] = df["hour"].n_unique()
                    if "observed" in df.columns:
                        obs_count = df.filter(pl.col("observed") == True).height  # noqa: E712
                        stats["observed_count"] = obs_count
                        stats["missing_count"] = len(df) - obs_count

                summary[stage.value] = stats

        return summary


def _compute_time_range(df: pl.DataFrame) -> Optional[float]:
    """Compute time range in hours from charttime column."""
    if "charttime" not in df.columns or df.is_empty():
        return None
    try:
        min_time = df["charttime"].min()
        max_time = df["charttime"].max()
        if min_time is None or max_time is None:
            return None
        delta = (max_time - min_time).total_seconds() / 3600
        return round(delta, 2)
    except Exception:
        return None


@dataclass
class MultiStageCapture:
    """Container for all patient stage captures.

    Attributes:
        captures: Dict mapping stay_id to PatientStageCapture.
        metadata: Global metadata (config, timestamps, etc.).
    """

    captures: Dict[int, PatientStageCapture] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage_data(
        self,
        stay_id: int,
        stage: ExtractionStage,
        data: pl.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add captured data for a patient at a specific stage.

        Args:
            stay_id: ICU stay identifier.
            stage: Extraction stage.
            data: DataFrame for this stage.
            metadata: Optional additional context.
        """
        if stay_id not in self.captures:
            self.captures[stay_id] = PatientStageCapture(stay_id=stay_id)

        stage_data = StageData(
            stage=stage,
            data=data,
            metadata=metadata or {},
        )
        self.captures[stay_id].set_stage(stage, stage_data)

    def export_all(self, output_dir: Path) -> Dict[int, Dict[str, Path]]:
        """Export all patient stage data to CSVs.

        Files are prefixed with stage numbers to show pipeline order:
        0_raw.csv, 1_transformed.csv, 2_binned.csv, 3_dense.csv

        Args:
            output_dir: Base output directory.

        Returns:
            Dict mapping stay_id to dict of stage -> path.
        """
        output_dir = Path(output_dir)
        per_patient_dir = output_dir / "per_patient"
        per_patient_dir.mkdir(parents=True, exist_ok=True)

        exported = {}
        for stay_id, capture in self.captures.items():
            exported[stay_id] = capture.to_csvs(per_patient_dir)

        # Also export combined stage files (all patients per stage)
        stages_dir = output_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)

        for stage in ExtractionStage:
            combined = self._combine_stage(stage)
            if combined is not None and not combined.is_empty():
                path = stages_dir / f"{stage.prefixed_name}.csv"
                combined.write_csv(path)

        return exported

    def _combine_stage(self, stage: ExtractionStage) -> Optional[pl.DataFrame]:
        """Combine data from all patients for a single stage."""
        dfs = []
        for capture in self.captures.values():
            stage_data = capture.get_stage(stage)
            if stage_data is not None and not stage_data.data.is_empty():
                dfs.append(stage_data.data)

        if not dfs:
            return None

        return pl.concat(dfs)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all captured data."""
        return {
            "n_patients": len(self.captures),
            "stay_ids": list(self.captures.keys()),
            "per_patient": {
                stay_id: capture.get_stage_summary() for stay_id, capture in self.captures.items()
            },
            "metadata": self.metadata,
        }


def filter_to_stay_ids(df: pl.DataFrame, stay_ids: List[int]) -> pl.DataFrame:
    """Filter DataFrame to specific stay IDs.

    Args:
        df: DataFrame with stay_id column.
        stay_ids: List of stay IDs to keep.

    Returns:
        Filtered DataFrame.
    """
    if "stay_id" not in df.columns:
        return df
    return df.filter(pl.col("stay_id").is_in(stay_ids))


def compute_stage_diff(
    before: pl.DataFrame,
    after: pl.DataFrame,
    key_cols: List[str],
    value_col: str = "valuenum",
) -> pl.DataFrame:
    """Compute differences between two stages for the same data.

    Useful for seeing what changed during a transformation (e.g., Fahrenheit → Celsius).

    Args:
        before: DataFrame before transformation.
        after: DataFrame after transformation.
        key_cols: Columns to join on (e.g., ["stay_id", "charttime", "feature_name"]).
        value_col: Column containing the value to compare.

    Returns:
        DataFrame with before, after, and diff columns.
    """
    # Rename value columns
    before_renamed = before.select(key_cols + [pl.col(value_col).alias("value_before")])
    after_renamed = after.select(key_cols + [pl.col(value_col).alias("value_after")])

    # Join and compute diff
    joined = before_renamed.join(after_renamed, on=key_cols, how="outer")

    return joined.with_columns(
        [
            (pl.col("value_after") - pl.col("value_before")).alias("value_diff"),
            (
                (pl.col("value_after") - pl.col("value_before"))
                / pl.col("value_before").abs()
                * 100
            ).alias("pct_change"),
        ]
    )


def flatten_binned_to_long(
    binned_df: pl.DataFrame,
    feature_names: List[str],
) -> pl.DataFrame:
    """Convert wide binned format to long format for easier comparison.

    Args:
        binned_df: Wide format with columns: stay_id, hour, feat1, feat1_mask, feat2, ...
        feature_names: List of feature column names (without _mask suffix).

    Returns:
        Long format: stay_id, hour, feature_name, value, observed
    """
    rows = []
    for row in binned_df.iter_rows(named=True):
        stay_id = row["stay_id"]
        hour = row["hour"]
        for feat_name in feature_names:
            value = row.get(feat_name)
            mask = row.get(f"{feat_name}_mask", False)
            rows.append(
                {
                    "stay_id": stay_id,
                    "hour": hour,
                    "feature_name": feat_name,
                    "value": value,
                    "observed": mask,
                }
            )

    return pl.DataFrame(rows)


def generate_html_report(
    captures: MultiStageCapture,
    output_path: Path,
    feature_subset: Optional[List[str]] = None,
    max_rows_per_stage: int = 100,
) -> Path:
    """Generate an HTML report comparing data across pipeline stages.

    Creates a single HTML file with:
    - Summary statistics per patient per stage
    - Side-by-side data tables (sampled)
    - Visual indicators for changes between stages

    Args:
        captures: MultiStageCapture with all patient data.
        output_path: Path to write HTML file.
        feature_subset: Optional list of features to focus on.
        max_rows_per_stage: Max rows to show per stage table.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Pipeline Stage Comparison Report</title>",
        "<style>",
        _get_css_styles(),
        "</style>",
        "</head>",
        "<body>",
        "<h1>Pipeline Stage Comparison Report</h1>",
        f"<p class='timestamp'>Generated: {datetime.now().isoformat()}</p>",
        f"<p>Patients: {len(captures.captures)}</p>",
    ]

    # Table of contents
    html_parts.append("<h2>Contents</h2>")
    html_parts.append("<ul>")
    for stay_id in captures.captures.keys():
        html_parts.append(f"<li><a href='#stay-{stay_id}'>Stay {stay_id}</a></li>")
    html_parts.append("</ul>")

    # Per-patient sections
    for stay_id, patient_capture in captures.captures.items():
        html_parts.append(f"<div class='patient-section' id='stay-{stay_id}'>")
        html_parts.append(f"<h2>Stay {stay_id}</h2>")

        # Summary table
        summary = patient_capture.get_stage_summary()
        html_parts.append(_render_summary_table(summary))

        # Stage comparison tables
        html_parts.append("<h3>Stage Data (Sampled)</h3>")
        html_parts.append("<div class='stage-tables'>")

        for stage in ExtractionStage:
            stage_data = patient_capture.get_stage(stage)
            if stage_data is not None and not stage_data.data.is_empty():
                html_parts.append("<div class='stage-table'>")
                html_parts.append(f"<h4>Stage: {stage.value.upper()}</h4>")
                html_parts.append(
                    _render_dataframe_table(
                        stage_data.data,
                        max_rows=max_rows_per_stage,
                        feature_subset=feature_subset,
                    )
                )
                html_parts.append("</div>")

        html_parts.append("</div>")  # stage-tables
        html_parts.append("</div>")  # patient-section

    html_parts.extend(["</body>", "</html>"])

    html_content = "\n".join(html_parts)
    output_path.write_text(html_content)

    return output_path


def _get_css_styles() -> str:
    """Return CSS styles for the HTML report."""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        h2 { color: #444; margin-top: 30px; }
        h3 { color: #555; }
        h4 { color: #666; margin: 10px 0 5px 0; }
        .timestamp { color: #888; font-size: 0.9em; }
        .patient-section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-table {
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }
        .summary-table th {
            background: #f0f0f0;
            font-weight: 600;
        }
        .summary-table tr:nth-child(even) { background: #fafafa; }
        .stage-tables {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .stage-table {
            flex: 1;
            min-width: 300px;
            max-width: 100%;
            overflow-x: auto;
        }
        .data-table {
            border-collapse: collapse;
            font-size: 0.8em;
            width: 100%;
        }
        .data-table th, .data-table td {
            border: 1px solid #ddd;
            padding: 4px 8px;
            text-align: right;
            white-space: nowrap;
        }
        .data-table th {
            background: #e8e8e8;
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        .data-table tr:nth-child(even) { background: #f9f9f9; }
        .data-table td.text { text-align: left; }
        .truncated-note {
            color: #888;
            font-style: italic;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .null-value { color: #ccc; }
    """


def _render_summary_table(summary: Dict[str, Dict[str, Any]]) -> str:
    """Render a summary table comparing stages."""
    stages = list(ExtractionStage)

    html = ["<table class='summary-table'>"]
    html.append("<tr><th>Metric</th>")
    for stage in stages:
        html.append(f"<th>{stage.value.upper()}</th>")
    html.append("</tr>")

    # Collect all metrics
    all_metrics: set[str] = set()
    for stage_summary in summary.values():
        all_metrics.update(stage_summary.keys())

    for metric in sorted(all_metrics):
        html.append(f"<tr><td>{metric}</td>")
        for stage in stages:
            value = summary.get(stage.value, {}).get(metric, "-")
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, tuple):
                value = f"{value[0]:.1f} - {value[1]:.1f}"
            html.append(f"<td>{value}</td>")
        html.append("</tr>")

    html.append("</table>")
    return "\n".join(html)


def _render_dataframe_table(
    df: pl.DataFrame,
    max_rows: int = 100,
    feature_subset: Optional[List[str]] = None,
) -> str:
    """Render a DataFrame as an HTML table."""
    if df.is_empty():
        return "<p>No data</p>"

    # Filter columns if feature_subset specified
    cols = df.columns
    if feature_subset:
        cols = [
            c
            for c in cols
            if c in feature_subset
            or c
            in ["stay_id", "hour", "charttime", "feature_name", "valuenum", "value", "observed"]
        ]
        if cols:
            df = df.select(cols)

    # Sample if too many rows
    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True

    html = ["<table class='data-table'>"]

    # Header
    html.append("<tr>")
    for col in df.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr>")

    # Rows
    for row in df.iter_rows(named=True):
        html.append("<tr>")
        for col in df.columns:
            value = row[col]
            if value is None:
                html.append("<td class='null-value'>null</td>")
            elif isinstance(value, float):
                if value != value:  # NaN check
                    html.append("<td class='null-value'>NaN</td>")
                else:
                    html.append(f"<td>{value:.4f}</td>")
            elif isinstance(value, str):
                html.append(f"<td class='text'>{value}</td>")
            else:
                html.append(f"<td>{value}</td>")
        html.append("</tr>")

    html.append("</table>")

    if truncated:
        html.append(f"<p class='truncated-note'>Showing first {max_rows} rows</p>")

    return "\n".join(html)
