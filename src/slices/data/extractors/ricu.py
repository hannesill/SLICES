"""RICU-based extractor — reads pre-extracted parquet files from the RICU R package.

RICU handles concept lookup, unit harmonization, hourly binning, and gap filling
across multiple ICU datasets (MIMIC-III/IV, eICU, HiRID, AUMCdb, SICdb).

Usage:
    # Step 1: Run R extraction
    Rscript scripts/preprocessing/extract_with_ricu.R \
        --dataset miiv --output_dir data/ricu_output/miiv \
        --raw_export_horizon_hours 48

    # Step 2: Run Python processing
    uv run python scripts/preprocessing/extract_ricu.py \
        data.ricu_output_dir=data/ricu_output/miiv
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)

from slices.constants import FEATURE_BLOCKLIST, LABEL_HORIZON_HOURS
from slices.data.config_schemas import TimeSeriesConceptConfig
from slices.data.labels import LabelBuilder, LabelBuilderFactory, LabelConfig

from .base import BaseExtractor, ExtractorConfig


def _resolve_parquet_path(path: Path) -> Union[Path, str]:
    """Resolve a parquet path that may be a single file or a directory of parts.

    The R extraction script writes timeseries as a directory of part files
    to avoid exceeding memory limits. Polars scan_parquet accepts both a
    single file and a glob pattern.
    """
    if path.is_file():
        return path
    if path.is_dir():
        return str(path / "*.parquet")
    raise FileNotFoundError(f"Parquet path not found: {path}")


console = Console()


class RicuExtractor(BaseExtractor):
    """Reads pre-extracted RICU parquet output.

    Expects parquet_root to contain:
      ricu_timeseries.parquet (file or directory of part files),
      ricu_stays.parquet, ricu_mortality.parquet,
      ricu_diagnoses.parquet, ricu_metadata.yaml

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
        self._task_configs_cache: Optional[List[LabelConfig]] = None

        if not self.parquet_root.exists():
            raise ValueError(f"RICU output directory not found: {self.parquet_root}")

        # Load RICU metadata
        metadata_path = self.parquet_root / "ricu_metadata.yaml"
        if not metadata_path.exists():
            raise ValueError(f"ricu_metadata.yaml not found in {self.parquet_root}")
        with open(metadata_path) as f:
            self._metadata = yaml.safe_load(f)
        self._validate_ricu_horizon()

    def _get_raw_export_horizon_hours(self) -> int:
        """Return the upstream RICU export horizon in hours.

        New exports should store ``raw_export_horizon_hours`` explicitly. Older
        exports only have ``seq_length_hours``; keep supporting that field for
        backward compatibility.
        """
        raw_horizon = self._metadata.get("raw_export_horizon_hours")
        if raw_horizon is None:
            raw_horizon = self._metadata.get("seq_length_hours")

        if raw_horizon is None:
            raise ValueError(
                "ricu_metadata.yaml is missing both raw_export_horizon_hours and "
                "seq_length_hours; cannot validate the export horizon safely."
            )

        return int(raw_horizon)

    def _get_task_configs_cached(self) -> List[LabelConfig]:
        """Load and cache task configs for horizon validation and metadata."""
        if self._task_configs_cache is None:
            self._task_configs_cache = self._load_task_configs(self.config.tasks)
        return self._task_configs_cache

    def _get_required_raw_export_horizon_hours(
        self, task_configs: Optional[List[LabelConfig]] = None
    ) -> int:
        """Return the minimum upstream raw export horizon required by this extraction."""
        required_horizon = int(self.config.seq_length_hours)
        for task_config in task_configs or self._get_task_configs_cached():
            builder = LabelBuilderFactory.create(task_config)
            required_horizon = max(
                required_horizon,
                int(builder.required_raw_timeseries_horizon_hours()),
            )

        return required_horizon

    def _validate_ricu_horizon(self) -> None:
        """Validate that the Python extraction horizon does not exceed the R export."""
        raw_export_horizon = self._get_raw_export_horizon_hours()
        if self.config.seq_length_hours > raw_export_horizon:
            raise ValueError(
                "Python extraction requests seq_length_hours="
                f"{self.config.seq_length_hours}, but the upstream RICU export only "
                f"contains {raw_export_horizon} hours. Re-run the R export with a longer "
                "horizon or lower the Python extraction seq_length_hours."
            )

        task_requirements = []
        for task_config in self._get_task_configs_cached():
            builder = LabelBuilderFactory.create(task_config)
            required_horizon = int(builder.required_raw_timeseries_horizon_hours())
            if required_horizon > self.config.seq_length_hours:
                task_requirements.append((task_config.task_name, required_horizon))

        required_raw_export_horizon = self._get_required_raw_export_horizon_hours(
            self._get_task_configs_cached()
        )
        if raw_export_horizon < required_raw_export_horizon:
            requirement_str = ", ".join(
                f"{task_name}={required_horizon}h"
                for task_name, required_horizon in task_requirements
            )
            raise ValueError(
                "Active task labels require a longer upstream raw timeseries horizon than "
                f"the current RICU export provides. Upstream export: {raw_export_horizon}h; "
                f"required: {required_raw_export_horizon}h. "
                f"Task-specific requirements: {requirement_str}. "
                "Keep the model input at 24h if desired, but re-run the R export with a "
                "longer raw horizon so forward-looking labels can be built safely."
            )

    def _iter_upstream_files(self) -> List[Path]:
        """Return the upstream files that define the extraction contents."""
        files: List[Path] = []
        for path in sorted(self.parquet_root.glob("ricu_*")):
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        return files

    def _get_upstream_source_signature(self) -> dict:
        """Fingerprint upstream RICU inputs for safe resume behavior."""
        files = []
        for path in self._iter_upstream_files():
            stat = path.stat()
            files.append(
                {
                    "path": str(path.relative_to(self.parquet_root)),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

        return {
            "dataset": self._metadata.get("dataset"),
            "ricu_seq_length_hours": int(self._metadata["seq_length_hours"]),
            "ricu_raw_export_horizon_hours": self._get_raw_export_horizon_hours(),
            "files": files,
        }

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
        path = _resolve_parquet_path(self.parquet_root / "ricu_timeseries.parquet")
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
                # Fix mask: unseen categorical values mapped to null must have mask=False
                mask_col = f"{col_name}_mask"
                if mask_col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col_name).is_null() & pl.col(mask_col))
                        .then(pl.lit(False))
                        .otherwise(pl.col(mask_col))
                        .alias(mask_col)
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
        path = _resolve_parquet_path(self.parquet_root / dispatch[source_name])
        df = pl.scan_parquet(path).filter(pl.col("stay_id").is_in(stay_ids)).collect()
        if source_name == "timeseries":
            df = self._encode_categorical_columns(df)
        elif source_name == "mortality_info":
            df = self._migrate_mortality_schema(df)
        return df

    @staticmethod
    def _migrate_mortality_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Migrate legacy mortality parquet (date_of_death only) to new schema.

        If the parquet has the old single-column format, infer precision from
        the Polars dtype so the label builder can use precision-aware logic.
        """
        if "death_time_precision" in df.columns:
            return df  # already new schema

        if "date_of_death" not in df.columns:
            return df

        dod_dtype = df["date_of_death"].dtype
        if dod_dtype == pl.Date:
            # Date-only column — treat as date precision
            df = df.with_columns(
                pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("death_time"),
                pl.col("date_of_death").cast(pl.Date).alias("death_date"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("date"))
                .otherwise(pl.lit(None))
                .alias("death_time_precision"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("legacy"))
                .otherwise(pl.lit(None))
                .alias("death_source"),
            )
        else:
            # Legacy datetimes are ambiguous: older exports may have stored
            # date-only values as midnight-cast timestamps. Treat them as date
            # precision unless explicit precision metadata is present.
            df = df.with_columns(
                pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("death_time"),
                pl.col("date_of_death").cast(pl.Date).alias("death_date"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("date"))
                .otherwise(pl.lit(None))
                .alias("death_time_precision"),
                pl.when(pl.col("date_of_death").is_not_null())
                .then(pl.lit("legacy"))
                .otherwise(pl.lit(None))
                .alias("death_source"),
            )
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
        feature_names = [f for f in self._metadata["feature_names"] if f not in FEATURE_BLOCKLIST]

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
        task_configs = self._get_task_configs_cached()
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

            # Build label manifest for freshness checking at training time
            label_manifest = {}
            for tc in task_configs:
                builder = LabelBuilderFactory.create(tc)
                label_manifest[tc.task_name] = {
                    "builder_version": builder.SEMANTIC_VERSION,
                    "config_hash": LabelBuilder.config_hash(tc),
                }

            metadata = {
                "dataset": self._get_dataset_name(),
                "feature_set": self.config.feature_set,
                "categories": self.config.categories,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "seq_length_hours": self.config.seq_length_hours,
                "input_seq_length_hours": self.config.seq_length_hours,
                "label_horizon_hours": LABEL_HORIZON_HOURS,
                "raw_export_horizon_hours": self._get_raw_export_horizon_hours(),
                "required_raw_export_horizon_hours": self._get_required_raw_export_horizon_hours(
                    task_configs
                ),
                "min_stay_hours": self.config.min_stay_hours,
                "task_names": task_names,
                "n_stays": len(stays_filtered),
                "label_manifest": label_manifest,
                "label_quality_stats": self._label_quality_stats,
                "upstream_source_signature": self._get_upstream_source_signature(),
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
