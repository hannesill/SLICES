#!/usr/bin/env python3
"""Analyze concept files to understand data coverage and distributions.

This script analyzes concept YAML files and provides comprehensive statistics
about:
- Concept definitions vs actual data availability
- Measurement coverage per modality
- Value distributions with min/max and percentiles
- Data quality metrics (missing rates, outliers)
- Cross-modality coverage

Usage:
    uv run python scripts/analyze_concepts.py \
        --concepts-dir configs/concepts \
        --parquet-root /path/to/mimic-iv-parquet \
        [--output-dir reports/concept_analysis]

    # Legacy single-file support (deprecated):
    uv run python scripts/analyze_concepts.py \
        --concept-file configs/concepts/vitals.yaml \
        --parquet-root /path/to/mimic-iv-parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style for plots
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    console.print(
        "[yellow]Warning: matplotlib/seaborn not available. "
        "Install with: uv add matplotlib seaborn[/yellow]"
    )


# Concept category files to load
CONCEPT_FILES = ["vitals.yaml", "labs.yaml", "outputs.yaml"]

# Table to schema mapping (used to find parquet files)
TABLE_SCHEMA_MAP = {
    "chartevents": "icu",
    "labevents": "hosp",
    "outputevents": "icu",
    "inputevents": "icu",
    "icustays": "icu",
    "patients": "hosp",
    "admissions": "hosp",
}


class ConceptAnalyzer:
    """Analyzes concept files against actual MIMIC-IV data."""

    def __init__(
        self,
        parquet_root: Path,
        concepts_dir: Optional[Path] = None,
        concept_file: Optional[Path] = None,
    ):
        """Initialize analyzer.

        Args:
            parquet_root: Path to MIMIC-IV Parquet files
            concepts_dir: Path to concepts directory (new format)
            concept_file: Path to single concept YAML file (legacy format)
        """
        self.parquet_root = Path(parquet_root)
        self.concepts_dir = Path(concepts_dir) if concepts_dir else None
        self.concept_file = Path(concept_file) if concept_file else None
        self.conn = duckdb.connect()

        if not self.parquet_root.exists():
            raise FileNotFoundError(f"Parquet root not found: {parquet_root}")

        if self.concepts_dir and not self.concepts_dir.exists():
            raise FileNotFoundError(f"Concepts directory not found: {concepts_dir}")

        if self.concept_file and not self.concept_file.exists():
            raise FileNotFoundError(f"Concept file not found: {concept_file}")

        # Load concepts
        self.concepts = self._load_concepts()

    def _load_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Load concepts from directory or single file.

        Returns:
            Dictionary mapping modality -> {concept_name -> config}
        """
        concepts = {}

        if self.concepts_dir:
            # New format: load from concepts directory
            for filename in CONCEPT_FILES:
                filepath = self.concepts_dir / filename
                if not filepath.exists():
                    continue

                modality = filepath.stem  # e.g., "vitals" from "vitals.yaml"

                with open(filepath) as f:
                    file_concepts = yaml.safe_load(f) or {}

                if file_concepts:
                    concepts[modality] = file_concepts
        elif self.concept_file:
            # Legacy format: single file with modality sections
            with open(self.concept_file) as f:
                file_data = yaml.safe_load(f) or {}

            # Check if it's a modality-grouped file or a single-modality file
            if any(k in file_data for k in ["vitals", "labs", "outputs"]):
                # Old format with modality sections
                for modality in ["vitals", "labs", "outputs"]:
                    if modality in file_data:
                        concepts[modality] = file_data[modality]
            else:
                # Single modality file - infer from filename
                modality = self.concept_file.stem
                concepts[modality] = file_data
        else:
            raise ValueError("Must provide either concepts_dir or concept_file")

        return concepts

    def _query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query and return Polars DataFrame."""
        return self.conn.execute(sql).pl()

    def _parquet_path(self, table: str) -> Path:
        """Get path to Parquet file for a table.

        Args:
            table: Table name (e.g., "chartevents", "labevents")

        Returns:
            Full path to parquet file
        """
        schema = TABLE_SCHEMA_MAP.get(table, "icu")
        return self.parquet_root / schema / f"{table}.parquet"

    def _get_total_measurements(self, table: str, value_col: str = "valuenum") -> int:
        """Get total number of measurements in a source table.

        Args:
            table: Table name (chartevents, labevents, outputevents)
            value_col: Column name for values

        Returns:
            Total count of measurements with non-null values
        """
        table_path = self._parquet_path(table)

        if not table_path.exists():
            console.print(f"[yellow]Warning: Table not found: {table_path}[/yellow]")
            return 0

        sql = f"""
        SELECT COUNT(*) as total
        FROM read_parquet('{table_path}')
        WHERE {value_col} IS NOT NULL
        """
        result = self._query(sql)
        if len(result) > 0:
            row_dict = result.row(0, named=True)
            val = row_dict.get("total")
            if val is None:
                return 0
            if isinstance(val, pl.Series):
                return int(val[0] if len(val) > 0 else 0)
            if isinstance(val, np.ndarray):
                return int(val.item() if val.size > 0 else 0)
            return int(val)
        return 0

    def _extract_sources_from_concept(self, concept_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source configurations from a concept.

        Handles both new format (mimic_iv as list) and legacy format.

        Args:
            concept_config: Concept configuration dictionary

        Returns:
            List of source configurations with keys: table, itemid, value_col
        """
        sources = []

        # New format: mimic_iv is a list of sources
        mimic_iv = concept_config.get("mimic_iv")
        if mimic_iv is None:
            return sources

        if isinstance(mimic_iv, list):
            for source in mimic_iv:
                table = source.get("table")
                itemid = source.get("itemid", [])
                value_col = source.get("value_col", "valuenum")
                transform = source.get("transform")

                # Normalize itemid to list
                if isinstance(itemid, int):
                    itemid = [itemid]
                elif not isinstance(itemid, list):
                    itemid = list(itemid) if itemid else []

                if table and itemid:
                    sources.append(
                        {
                            "table": table,
                            "itemid": itemid,
                            "value_col": value_col,
                            "transform": transform,
                        }
                    )
        elif isinstance(mimic_iv, dict):
            # Legacy format: mimic_iv is a single dict with source key
            table = mimic_iv.get("source") or mimic_iv.get("table")
            itemid = mimic_iv.get("itemid") or mimic_iv.get("itemids", [])
            value_col = mimic_iv.get("value_col", "valuenum")

            # Normalize itemid to list
            if isinstance(itemid, int):
                itemid = [itemid]
            elif not isinstance(itemid, list):
                itemid = list(itemid) if itemid else []

            if table and itemid:
                sources.append(
                    {
                        "table": table,
                        "itemid": itemid,
                        "value_col": value_col,
                        "transform": None,
                    }
                )

        return sources

    def _get_concept_measurements(
        self, table: str, itemids: List[int], value_col: str
    ) -> Dict[str, Any]:
        """Get measurement statistics for a concept.

        Args:
            table: Table name (chartevents, labevents, outputevents)
            itemids: List of item IDs for this concept
            value_col: Column name for values

        Returns:
            Dictionary with measurement statistics
        """
        table_path = self._parquet_path(table)

        if not table_path.exists():
            return {
                "total_measurements": 0,
                "unique_stays": 0,
                "unique_patients": 0,
                "values": None,
            }

        itemids_str = ",".join(map(str, itemids))

        # Handle different table structures
        # chartevents and outputevents have stay_id directly
        # labevents has hadm_id, need to join with icustays
        if table == "labevents":
            icustays_path = self._parquet_path("icustays")
            # Get basic counts with join to icustays for stay_id
            sql = f"""
            SELECT
                COUNT(*) as total_measurements,
                COUNT(DISTINCT i.stay_id) as unique_stays,
                COUNT(DISTINCT l.subject_id) as unique_patients,
                MIN(l.{value_col}) as min_val,
                MAX(l.{value_col}) as max_val,
                AVG(l.{value_col}) as mean_val,
                quantile_cont(l.{value_col}, 0.05) as p5,
                quantile_cont(l.{value_col}, 0.25) as p25,
                quantile_cont(l.{value_col}, 0.50) as p50,
                quantile_cont(l.{value_col}, 0.75) as p75,
                quantile_cont(l.{value_col}, 0.95) as p95,
                quantile_cont(l.{value_col}, 0.99) as p99
            FROM read_parquet('{table_path}') AS l
            INNER JOIN read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE l.itemid IN ({itemids_str})
                AND l.{value_col} IS NOT NULL
            """

            # Get all values for distribution (sample if too many)
            sql_values = f"""
            SELECT l.{value_col} as value
            FROM read_parquet('{table_path}') AS l
            INNER JOIN read_parquet('{icustays_path}') AS i
                ON l.hadm_id = i.hadm_id
            WHERE l.itemid IN ({itemids_str})
                AND l.{value_col} IS NOT NULL
            LIMIT 100000
            """
        else:
            # chartevents and outputevents have stay_id directly
            sql = f"""
            SELECT
                COUNT(*) as total_measurements,
                COUNT(DISTINCT stay_id) as unique_stays,
                COUNT(DISTINCT subject_id) as unique_patients,
                MIN({value_col}) as min_val,
                MAX({value_col}) as max_val,
                AVG({value_col}) as mean_val,
                quantile_cont({value_col}, 0.05) as p5,
                quantile_cont({value_col}, 0.25) as p25,
                quantile_cont({value_col}, 0.50) as p50,
                quantile_cont({value_col}, 0.75) as p75,
                quantile_cont({value_col}, 0.95) as p95,
                quantile_cont({value_col}, 0.99) as p99
            FROM read_parquet('{table_path}')
            WHERE itemid IN ({itemids_str})
                AND {value_col} IS NOT NULL
            """

            # Get all values for distribution (sample if too many)
            sql_values = f"""
            SELECT {value_col} as value
            FROM read_parquet('{table_path}')
            WHERE itemid IN ({itemids_str})
                AND {value_col} IS NOT NULL
            LIMIT 100000
            """

        stats = self._query(sql)
        values_df = self._query(sql_values)
        values = values_df["value"].to_numpy() if len(values_df) > 0 else np.array([])

        if len(stats) > 0:
            row_dict = stats.row(0, named=True)

            def get_scalar(col_name):
                val = row_dict.get(col_name)
                if val is None:
                    return None
                if isinstance(val, pl.Series):
                    return val[0] if len(val) > 0 else None
                if isinstance(val, np.ndarray):
                    return val.item() if val.size > 0 else None
                return val

            return {
                "total_measurements": int(get_scalar("total_measurements") or 0),
                "unique_stays": int(get_scalar("unique_stays") or 0),
                "unique_patients": int(get_scalar("unique_patients") or 0),
                "min_val": get_scalar("min_val"),
                "max_val": get_scalar("max_val"),
                "mean_val": get_scalar("mean_val"),
                "p5": get_scalar("p5"),
                "p25": get_scalar("p25"),
                "p50": get_scalar("p50"),
                "p75": get_scalar("p75"),
                "p95": get_scalar("p95"),
                "p99": get_scalar("p99"),
                "values": values,
            }
        else:
            return {
                "total_measurements": 0,
                "unique_stays": 0,
                "unique_patients": 0,
                "values": None,
            }

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis of concepts.

        Returns:
            Dictionary with analysis results
        """
        console.print("[bold blue]Analyzing concepts...[/bold blue]")

        results = {}

        # Process each modality
        for modality, modality_concepts in self.concepts.items():
            if not modality_concepts or not isinstance(modality_concepts, dict):
                continue

            console.print(f"\n[bold]Analyzing {modality.upper()}[/bold]")

            # Determine primary table for this modality
            first_concept = list(modality_concepts.values())[0]
            sources = self._extract_sources_from_concept(first_concept)
            if not sources:
                console.print("  [yellow]No MIMIC-IV sources found[/yellow]")
                continue

            primary_table = sources[0]["table"]
            primary_value_col = sources[0]["value_col"]

            # Get total measurements for this source
            total_source_measurements = self._get_total_measurements(
                primary_table, primary_value_col
            )

            # Analyze each concept
            concept_stats = []
            total_concept_measurements = 0

            for concept_name, concept_config in modality_concepts.items():
                console.print(f"  Analyzing [cyan]{concept_name}[/cyan]...", end="")

                # Extract all sources and combine itemids
                sources = self._extract_sources_from_concept(concept_config)
                if not sources:
                    console.print(" [yellow]no sources[/yellow]")
                    continue

                # Combine all itemids across sources (they may come from same table)
                all_itemids = []
                table = sources[0]["table"]
                value_col = sources[0]["value_col"]
                for source in sources:
                    all_itemids.extend(source["itemid"])
                    # Use first source's table/value_col for stats
                    # (assumes same table for all sources of a concept)

                # Get statistics
                stats = self._get_concept_measurements(table, all_itemids, value_col)

                # Get expected min/max from config
                expected_min = concept_config.get("min")
                expected_max = concept_config.get("max")
                units = concept_config.get("units", "N/A")

                # Check for outliers (values outside expected range)
                outliers = 0
                if stats["values"] is not None and len(stats["values"]) > 0:
                    if expected_min is not None:
                        outliers += (stats["values"] < expected_min).sum()
                    if expected_max is not None:
                        outliers += (stats["values"] > expected_max).sum()

                concept_stat = {
                    "concept_name": concept_name,
                    "itemids": all_itemids,
                    "n_itemids": len(all_itemids),
                    "units": units,
                    "expected_min": expected_min,
                    "expected_max": expected_max,
                    "total_measurements": stats["total_measurements"],
                    "unique_stays": stats["unique_stays"],
                    "unique_patients": stats["unique_patients"],
                    "actual_min": stats.get("min_val"),
                    "actual_max": stats.get("max_val"),
                    "mean": stats.get("mean_val"),
                    "p5": stats.get("p5"),
                    "p25": stats.get("p25"),
                    "p50": stats.get("p50"),
                    "p75": stats.get("p75"),
                    "p95": stats.get("p95"),
                    "p99": stats.get("p99"),
                    "outliers": outliers,
                    "outlier_pct": (
                        (outliers / len(stats["values"]) * 100)
                        if stats["values"] is not None and len(stats["values"]) > 0
                        else 0
                    ),
                    "values": stats["values"],
                }

                concept_stats.append(concept_stat)
                total_concept_measurements += stats["total_measurements"]

                console.print(" [green]done[/green]")

            # Calculate coverage percentage
            coverage_pct = (
                (total_concept_measurements / total_source_measurements * 100)
                if total_source_measurements > 0
                else 0
            )

            results[modality] = {
                "table": primary_table,
                "n_concepts": len(concept_stats),
                "total_source_measurements": total_source_measurements,
                "total_concept_measurements": total_concept_measurements,
                "coverage_pct": coverage_pct,
                "concepts": concept_stats,
            }

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary table of analysis results."""
        console.print("\n[bold green]=== SUMMARY ===[/bold green]\n")

        # Overall summary
        summary_table = Table(title="Modality Coverage Summary")
        summary_table.add_column("Modality", style="cyan")
        summary_table.add_column("Table", style="magenta")
        summary_table.add_column("Concepts", justify="right")
        summary_table.add_column("Total Measurements", justify="right", style="green")
        summary_table.add_column("Concept Measurements", justify="right", style="yellow")
        summary_table.add_column("Coverage %", justify="right", style="bold")

        for modality, data in results.items():
            summary_table.add_row(
                modality.upper(),
                data["table"],
                str(data["n_concepts"]),
                f"{data['total_source_measurements']:,}",
                f"{data['total_concept_measurements']:,}",
                f"{data['coverage_pct']:.2f}%",
            )

        console.print(summary_table)

        # Detailed per-concept statistics
        for modality, data in results.items():
            console.print(f"\n[bold]{modality.upper()} - Detailed Statistics[/bold]")

            detail_table = Table(title=f"{modality.upper()} Concepts")
            detail_table.add_column("Concept", style="cyan", width=20)
            detail_table.add_column("ItemIDs", style="dim", width=15)
            detail_table.add_column("Measurements", justify="right", width=12)
            detail_table.add_column("Stays", justify="right", width=10)
            detail_table.add_column("Patients", justify="right", width=10)
            detail_table.add_column("Min", justify="right", width=10)
            detail_table.add_column("Max", justify="right", width=10)
            detail_table.add_column("Outliers", justify="right", width=10)

            for concept in data["concepts"]:
                itemids_str = ",".join(map(str, concept["itemids"][:3]))
                if len(concept["itemids"]) > 3:
                    itemids_str += "..."

                min_val = (
                    f"{concept['actual_min']:.2f}" if concept["actual_min"] is not None else "N/A"
                )
                max_val = (
                    f"{concept['actual_max']:.2f}" if concept["actual_max"] is not None else "N/A"
                )
                outliers_str = (
                    f"{concept['outliers']:,} ({concept['outlier_pct']:.1f}%)"
                    if concept["outliers"] > 0
                    else "0"
                )

                detail_table.add_row(
                    concept["concept_name"],
                    itemids_str,
                    f"{concept['total_measurements']:,}",
                    f"{concept['unique_stays']:,}",
                    f"{concept['unique_patients']:,}",
                    min_val,
                    max_val,
                    outliers_str,
                )

            console.print(detail_table)

    def plot_distributions(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create distribution plots for each concept.

        Args:
            results: Analysis results
            output_dir: Directory to save plots
        """
        if not PLOTTING_AVAILABLE:
            console.print(
                "[yellow]Skipping plots: matplotlib/seaborn not available. "
                "Install with: uv add matplotlib seaborn[/yellow]"
            )
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        console.print("\n[bold blue]Generating distribution plots...[/bold blue]")

        for modality, data in results.items():
            concepts = data["concepts"]
            n_concepts = len(concepts)

            if n_concepts == 0:
                continue

            # For large numbers of concepts, split into multiple pages
            concepts_per_page = 16  # 4x4 grid max
            n_pages = (n_concepts + concepts_per_page - 1) // concepts_per_page

            for page in range(n_pages):
                start_idx = page * concepts_per_page
                end_idx = min(start_idx + concepts_per_page, n_concepts)
                page_concepts = concepts[start_idx:end_idx]
                n_page_concepts = len(page_concepts)

                # Create subplots (arrange in grid)
                n_cols = min(4, n_page_concepts)
                n_rows = (n_page_concepts + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
                if n_page_concepts == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                title = f"{modality.upper()} - Value Distributions"
                if n_pages > 1:
                    title += f" (Page {page + 1}/{n_pages})"
                fig.suptitle(title, fontsize=16, fontweight="bold")

                for idx, concept in enumerate(page_concepts):
                    ax = axes[idx]

                    if concept["values"] is None or len(concept["values"]) == 0:
                        ax.text(
                            0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                        )
                        ax.set_title(concept["concept_name"], fontsize=10)
                        continue

                    values = concept["values"]

                    # Adaptive binning for histograms
                    non_zero_values = values[values != 0] if len(values) > 0 else values
                    zero_count = (values == 0).sum() if len(values) > 0 else 0
                    zero_pct = zero_count / len(values) if len(values) > 0 else 0

                    if len(non_zero_values) > 0:
                        n_bins_nonzero = min(40, max(15, int(np.sqrt(len(non_zero_values)))))

                        if zero_pct > 0.1:
                            non_zero_min = float(non_zero_values.min())
                            non_zero_max = float(non_zero_values.max())

                            if non_zero_max > non_zero_min:
                                if non_zero_max / max(non_zero_min, 1e-10) > 100:
                                    log_min = np.log10(max(non_zero_min, 1e-10))
                                    log_max = np.log10(non_zero_max)
                                    log_bins = np.logspace(log_min, log_max, n_bins_nonzero)
                                    log_bins = np.unique(np.sort(log_bins))
                                    small_epsilon = (
                                        min(non_zero_min * 0.1, 0.01) if non_zero_min > 0 else 0.01
                                    )
                                    log_bins = log_bins[log_bins > small_epsilon * 1.1]
                                    if len(log_bins) > 0:
                                        bins = np.concatenate([[0, small_epsilon], log_bins])
                                    else:
                                        bins = np.array([0, small_epsilon, non_zero_max * 1.1])
                                else:
                                    nonzero_bins = np.linspace(
                                        non_zero_min, non_zero_max, n_bins_nonzero
                                    )
                                    nonzero_bins = np.unique(np.sort(nonzero_bins))
                                    small_epsilon = non_zero_min * 0.1 if non_zero_min > 0 else 0.01
                                    nonzero_bins = nonzero_bins[nonzero_bins > small_epsilon * 1.1]
                                    if len(nonzero_bins) > 0:
                                        bins = np.concatenate([[0, small_epsilon], nonzero_bins])
                                    else:
                                        bins = np.array([0, small_epsilon, non_zero_max * 1.1])
                            else:
                                if non_zero_max > 0:
                                    bins = np.array([0, non_zero_max * 0.5, non_zero_max * 1.1])
                                else:
                                    bins = np.array([0, 0.1, 1])
                        else:
                            n_bins = min(40, max(15, int(np.sqrt(len(values)))))
                            bins = n_bins
                    else:
                        bins = np.array([0, 0.1, 1])

                    # Safety check for bins
                    if isinstance(bins, np.ndarray):
                        bins = np.unique(bins)
                        bins = np.sort(bins)
                        if len(bins) < 2:
                            bins = min(40, max(15, int(np.sqrt(len(values)))))
                        if len(bins) > 1 and np.any(np.diff(bins) <= 0):
                            bins = min(40, max(15, int(np.sqrt(len(values)))))

                    ax.hist(values, bins=bins, alpha=0.7, edgecolor="gray", linewidth=0.5)

                    # Add vertical lines for statistics
                    if concept["actual_min"] is not None:
                        ax.axvline(
                            concept["actual_min"],
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="Min",
                        )
                    if concept["actual_max"] is not None:
                        ax.axvline(
                            concept["actual_max"],
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="Max",
                        )
                    if concept["p50"] is not None:
                        ax.axvline(
                            concept["p50"],
                            color="blue",
                            linestyle="-",
                            linewidth=1.5,
                            label="Median",
                            alpha=0.8,
                        )

                    # Add expected range if available
                    if concept["expected_min"] is not None:
                        ax.axvline(
                            concept["expected_min"],
                            color="orange",
                            linestyle=":",
                            linewidth=1,
                            alpha=0.7,
                            label="Expected Min",
                        )
                    if concept["expected_max"] is not None:
                        ax.axvline(
                            concept["expected_max"],
                            color="orange",
                            linestyle=":",
                            linewidth=1,
                            alpha=0.7,
                            label="Expected Max",
                        )

                    # Title with stats
                    title = f"{concept['concept_name']}\n"
                    title += f"n={concept['total_measurements']:,}"
                    if concept["outliers"] > 0:
                        title += f" | {concept['outliers']:,} outliers"
                    ax.set_title(title, fontsize=8)
                    ax.set_xlabel(concept["units"], fontsize=7)
                    ax.set_ylabel("Frequency", fontsize=7)
                    ax.tick_params(labelsize=6)
                    if idx == 0:
                        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
                    ax.grid(True, alpha=0.2)

                # Hide unused subplots
                for idx in range(n_page_concepts, len(axes)):
                    axes[idx].axis("off")

                plt.tight_layout()

                # Save plot
                if n_pages > 1:
                    plot_path = output_dir / f"{modality}_distributions_page{page + 1}.png"
                else:
                    plot_path = output_dir / f"{modality}_distributions.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                console.print(f"  Saved: {plot_path}")
                plt.close()

    def save_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save detailed report to files.

        Args:
            results: Analysis results
            output_dir: Directory to save reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary CSV
        summary_rows = []
        for modality, data in results.items():
            for concept in data["concepts"]:
                summary_rows.append(
                    {
                        "modality": modality,
                        "concept": concept["concept_name"],
                        "n_itemids": concept["n_itemids"],
                        "itemids": ",".join(map(str, concept["itemids"])),
                        "units": concept["units"],
                        "total_measurements": concept["total_measurements"],
                        "unique_stays": concept["unique_stays"],
                        "unique_patients": concept["unique_patients"],
                        "actual_min": concept["actual_min"],
                        "actual_max": concept["actual_max"],
                        "mean": concept["mean"],
                        "p5": concept["p5"],
                        "p25": concept["p25"],
                        "p50": concept["p50"],
                        "p75": concept["p75"],
                        "p95": concept["p95"],
                        "p99": concept["p99"],
                        "expected_min": concept["expected_min"],
                        "expected_max": concept["expected_max"],
                        "outliers": concept["outliers"],
                        "outlier_pct": concept["outlier_pct"],
                    }
                )

        summary_df = pl.DataFrame(summary_rows)
        csv_path = output_dir / "concept_analysis_summary.csv"
        summary_df.write_csv(csv_path)
        console.print(f"\n[green]Saved summary CSV: {csv_path}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze concept files against MIMIC-IV data")
    parser.add_argument(
        "--concepts-dir",
        type=str,
        default=None,
        help="Path to concepts directory (e.g., configs/concepts)",
    )
    parser.add_argument(
        "--concept-file",
        type=str,
        default=None,
        help="[DEPRECATED] Path to single concept YAML file",
    )
    parser.add_argument(
        "--parquet-root",
        type=str,
        required=True,
        help="Path to MIMIC-IV Parquet files directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/concept_analysis",
        help="Output directory for reports and plots (default: reports/concept_analysis)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.concepts_dir is None and args.concept_file is None:
        # Default to configs/concepts if exists
        default_dir = Path("configs/concepts")
        if default_dir.exists():
            args.concepts_dir = str(default_dir)
        else:
            parser.error("Must provide either --concepts-dir or --concept-file")

    if args.concept_file:
        console.print(
            "[yellow]Warning: --concept-file is deprecated. "
            "Use --concepts-dir instead.[/yellow]\n"
        )

    try:
        analyzer = ConceptAnalyzer(
            parquet_root=Path(args.parquet_root),
            concepts_dir=Path(args.concepts_dir) if args.concepts_dir else None,
            concept_file=Path(args.concept_file) if args.concept_file else None,
        )

        # Run analysis
        results = analyzer.analyze()

        # Print summary
        analyzer.print_summary(results)

        # Generate plots
        output_dir = Path(args.output_dir)
        analyzer.plot_distributions(results, output_dir)

        # Save report
        analyzer.save_report(results, output_dir)

        console.print("\n[bold green]Analysis complete![/bold green]")
        console.print(f"Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
