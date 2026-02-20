"""RICU-based extractor — reads pre-extracted parquet files from the RICU R package.

RICU handles concept lookup, unit harmonization, hourly binning, and gap filling
across multiple ICU datasets (MIMIC-III/IV, eICU, HiRID, AUMCdb, SICdb).

Usage:
    # Step 1: Run R extraction
    Rscript scripts/preprocessing/extract_with_ricu.R \
        --dataset miiv --output_dir data/ricu_output/miiv

    # Step 2: Run Python processing
    uv run python scripts/preprocessing/extract_ricu.py \
        data.ricu_output_dir=data/ricu_output/miiv
"""

from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)

from slices.data.config_schemas import TimeSeriesConceptConfig

from .base import BaseExtractor, ExtractorConfig

console = Console()


class RicuExtractor(BaseExtractor):
    """Reads pre-extracted RICU parquet output.

    Expects parquet_root to contain:
      ricu_timeseries.parquet, ricu_stays.parquet,
      ricu_mortality.parquet, ricu_diagnoses.parquet,
      ricu_metadata.yaml

    Unlike other extractors, this class does NOT use DuckDB or concept YAML
    files. All extraction and binning is handled by the R script; this class
    simply reads the results and converts them to SLICES format.
    """

    # RICU returns some concepts as categorical strings. Map to ordinal floats.
    CATEGORICAL_ENCODINGS: Dict[str, Dict[str, float]] = {
        "avpu": {"A": 0.0, "V": 1.0, "P": 2.0, "U": 3.0},
        "mech_vent": {"noninvasive": 1.0, "invasive": 2.0},
    }

    def __init__(self, config: ExtractorConfig) -> None:
        super().__init__(config)

        self._stays_cache: Optional[pl.DataFrame] = None
        self._metadata: Optional[dict] = None

        if not self.parquet_root.exists():
            raise ValueError(f"RICU output directory not found: {self.parquet_root}")

        # Load RICU metadata
        metadata_path = self.parquet_root / "ricu_metadata.yaml"
        if not metadata_path.exists():
            raise ValueError(f"ricu_metadata.yaml not found in {self.parquet_root}")
        with open(metadata_path) as f:
            self._metadata = yaml.safe_load(f)

    def _get_dataset_name(self) -> str:
        return self._metadata["dataset"]

    def extract_stays(self) -> pl.DataFrame:
        path = self.parquet_root / "ricu_stays.parquet"
        return pl.scan_parquet(path).collect()

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        """Read pre-binned hourly timeseries from RICU output.

        Bypasses _extract_raw_events() and _bin_to_hourly_grid() entirely.
        Returns same schema as _bin_to_hourly_grid() output:
          stay_id, hour, {feature}, {feature}_mask, ...

        Categorical string columns (e.g. avpu, mech_vent) are mapped to
        ordinal floats using CATEGORICAL_ENCODINGS.
        """
        path = self.parquet_root / "ricu_timeseries.parquet"
        df = pl.scan_parquet(path).filter(pl.col("stay_id").is_in(stay_ids)).collect()
        return self._encode_categorical_columns(df)

    def _encode_categorical_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert non-numeric RICU columns to floats.

        - Categorical strings (avpu, mech_vent) → ordinal floats
        - Duration columns (dobu_dur, etc.) → hours as Float64
        """
        for col_name, mapping in self.CATEGORICAL_ENCODINGS.items():
            if col_name not in df.columns:
                continue
            if df[col_name].dtype in (pl.Utf8, pl.String):
                df = df.with_columns(
                    pl.col(col_name).replace_strict(mapping, default=None, return_dtype=pl.Float64)
                )

        # Convert Duration columns to hours (float)
        for col_name in df.columns:
            if df[col_name].dtype == pl.Duration or str(df[col_name].dtype).startswith("Duration"):
                df = df.with_columns(
                    (pl.col(col_name).dt.total_milliseconds() / 3_600_000.0).alias(col_name)
                )

        return df

    def _extract_raw_events(
        self,
        stay_ids: List[int],
        feature_mapping: Dict[str, TimeSeriesConceptConfig],
    ) -> pl.DataFrame:
        raise NotImplementedError(
            "RicuExtractor reads pre-binned data. "
            "Raw event extraction is handled by the R script."
        )

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        dispatch = {
            "mortality_info": "ricu_mortality.parquet",
            "diagnoses": "ricu_diagnoses.parquet",
            "timeseries": "ricu_timeseries.parquet",
        }
        if source_name not in dispatch:
            raise ValueError(
                f"Unknown data source '{source_name}'. " f"Available: {list(dispatch.keys())}"
            )
        path = self.parquet_root / dispatch[source_name]
        df = pl.scan_parquet(path).filter(pl.col("stay_id").is_in(stay_ids)).collect()
        if source_name == "timeseries":
            df = self._encode_categorical_columns(df)
        return df

    def run(self) -> None:
        """Execute extraction pipeline for RICU data.

        Simplified version of BaseExtractor.run() — no batch loop needed
        since all data is already materialized in parquet files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing extraction
        existing_data = self._check_existing_extraction()
        existing_stay_ids = set()
        if existing_data is not None:
            existing_stay_ids = set(existing_data["stays"]["stay_id"].to_list())

        console.print("\n[bold blue]SLICES Data Extraction (RICU)[/bold blue]")
        console.print(f"RICU source: {self._get_dataset_name()}")
        console.print(f"Output directory: {self.output_dir}")
        console.print(f"Sequence length: {self.config.seq_length_hours} hours")
        console.print(f"Min stay length: {self.config.min_stay_hours} hours")
        if existing_stay_ids:
            console.print(f"Resuming: {len(existing_stay_ids)} stays already extracted\n")
        else:
            console.print("")

        # -----------------------------------------------------------------
        # Step 1: Load stays
        # -----------------------------------------------------------------
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading RICU stays...", total=None)
            stays = self.extract_stays()
            progress.update(task, completed=True)

        # Filter by minimum stay length
        min_los_days = self.config.min_stay_hours / 24.0
        stays_filtered = stays.filter(pl.col("los_days") >= min_los_days)

        console.print(f"Found {len(stays)} ICU stays")
        console.print(
            f"After filtering (>={self.config.min_stay_hours}h): {len(stays_filtered)} stays"
        )

        # Filter out already-extracted stays if resuming
        if existing_stay_ids:
            stays_filtered = stays_filtered.filter(
                ~pl.col("stay_id").is_in(list(existing_stay_ids))
            )
            console.print(
                f"After excluding already-extracted: {len(stays_filtered)} stays to process"
            )

        stay_ids = stays_filtered["stay_id"].to_list()

        if len(stay_ids) == 0:
            if existing_stay_ids:
                console.print("[green]All stays already extracted. Nothing to do![/green]")
                return
            else:
                console.print("[red]Error: No stays remaining after filtering![/red]")
                return

        self._validate_stays(stays_filtered)
        self._stays_cache = stays_filtered

        # -----------------------------------------------------------------
        # Step 2: Load timeseries (single read, no batching)
        # -----------------------------------------------------------------
        feature_names = self._metadata["feature_names"]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading RICU timeseries...", total=None)
            sparse_timeseries = self.extract_timeseries(stay_ids)
            progress.update(task, completed=True)

        console.print(f"Loaded {len(feature_names)} features: {feature_names}")
        self._validate_timeseries(sparse_timeseries, stay_ids, feature_names)

        # Convert to dense representation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Converting to dense timeseries...", total=None)
            dense_timeseries = self._create_dense_timeseries(
                sparse_timeseries, stays_filtered, feature_names
            )
            progress.update(task, completed=True)

        # -----------------------------------------------------------------
        # Step 3: Extract labels
        # -----------------------------------------------------------------
        task_configs = self._load_task_configs(self.config.tasks)
        task_names = [tc.task_name for tc in task_configs]

        if task_configs:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Extracting labels for {len(task_configs)} tasks...", total=None
                )
                labels = self.extract_labels(stay_ids, task_configs)
                progress.update(task, completed=True)

            console.print(f"Extracted labels for tasks: {task_names}")
        else:
            labels = pl.DataFrame({"stay_id": stay_ids})
            console.print("[yellow]No task configs found - labels will be empty[/yellow]")

        # -----------------------------------------------------------------
        # Step 4: Merge with existing data and save (with file locking)
        # -----------------------------------------------------------------
        static_path = self.output_dir / "static.parquet"
        timeseries_path = self.output_dir / "timeseries.parquet"
        labels_path = self.output_dir / "labels.parquet"

        with self._with_file_lock(static_path):
            existing_data_locked = self._check_existing_extraction()

            if existing_data_locked is not None:
                console.print("\n[bold]Merging with existing data...[/bold]")

                stays_filtered = pl.concat([existing_data_locked["stays"], stays_filtered])
                console.print(f"  Merged stays: {len(stays_filtered)} total")

                dense_timeseries = pl.concat([existing_data_locked["timeseries"], dense_timeseries])
                console.print(f"  Merged timeseries: {len(dense_timeseries)} total")

                labels = pl.concat([existing_data_locked["labels"], labels])
                console.print(f"  Merged labels: {len(labels)} total")

            console.print("\n[bold]Saving to Parquet...[/bold]")

            self._atomic_write(static_path, lambda tmp: stays_filtered.write_parquet(tmp))
            console.print(f"  Static features: {static_path} ({len(stays_filtered)} stays)")

            self._atomic_write(timeseries_path, lambda tmp: dense_timeseries.write_parquet(tmp))
            console.print(f"  Timeseries: {timeseries_path} ({len(dense_timeseries)} stays)")

            self._atomic_write(labels_path, lambda tmp: labels.write_parquet(tmp))
            console.print(f"  Labels: {labels_path} ({len(labels)} stays)")

            self._validate_labels(labels, stay_ids)

            # Save sliding-window labels (e.g. decompensation) to separate files
            sliding_window_tasks = []
            if hasattr(self, "_sliding_window_labels"):
                for sw_task_name, sw_labels in self._sliding_window_labels.items():
                    sw_path = self.output_dir / f"labels_{sw_task_name}.parquet"

                    # Merge with existing sliding-window labels if resuming
                    if existing_data_locked is not None:
                        existing_sw_path = self.output_dir / f"labels_{sw_task_name}.parquet"
                        if existing_sw_path.exists():
                            existing_sw = pl.read_parquet(existing_sw_path)
                            sw_labels = pl.concat([existing_sw, sw_labels])

                    self._atomic_write(sw_path, lambda tmp, df=sw_labels: df.write_parquet(tmp))
                    console.print(
                        f"  Sliding-window labels ({sw_task_name}): "
                        f"{sw_path} ({len(sw_labels)} samples)"
                    )
                    sliding_window_tasks.append(sw_task_name)

            metadata = {
                "dataset": self._get_dataset_name(),
                "feature_set": self.config.feature_set,
                "categories": self.config.categories,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "seq_length_hours": self.config.seq_length_hours,
                "min_stay_hours": self.config.min_stay_hours,
                "task_names": task_names,
                "sliding_window_tasks": sliding_window_tasks,
                "n_stays": len(stays_filtered),
                "extraction_config": {
                    "parquet_root": str(self.parquet_root),
                    "output_dir": str(self.output_dir),
                    "feature_set": self.config.feature_set,
                    "categories": self.config.categories,
                    "seq_length_hours": self.config.seq_length_hours,
                    "min_stay_hours": self.config.min_stay_hours,
                    "tasks": self.config.tasks,
                    "extraction_timestamp": datetime.now().isoformat(),
                },
                "ricu_metadata": self._metadata,
            }

            metadata_path = self.output_dir / "metadata.yaml"

            def write_metadata(tmp: str) -> None:
                with open(tmp, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            self._atomic_write(metadata_path, write_metadata, suffix=".yaml")
            console.print(f"  Metadata: {metadata_path}")

        console.print("\n[bold green]Extraction complete![/bold green]")
