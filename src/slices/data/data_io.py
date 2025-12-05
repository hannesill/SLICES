"""Utilities for converting CSV files to Parquet format.

This module provides functionality to convert raw CSV.gz files to Parquet format
using DuckDB for efficient streaming conversion with low memory footprint.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Optional

import duckdb

# Use rich for better logging if available, fallback to standard logging
try:
    from rich.console import Console

    console = Console()
    _has_rich = True
    _logger = None
except ImportError:
    _has_rich = False
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _logger = logging.getLogger(__name__)


def _get_logger():
    """Get logger instance."""
    if _has_rich:
        return console
    return _logger


def _csv_to_parquet_all(src_root: Path, parquet_root: Path) -> bool:
    """Convert all CSV files in the source directory to Parquet files.
    
    Streams via DuckDB COPY to keep memory low. Low concurrency to avoid
    parallel memory spikes. Tunable via environment variables:
        - SLICES_CONVERT_MAX_WORKERS (default: 4)
        - SLICES_DUCKDB_MEM (default: 3GB)
        - SLICES_DUCKDB_THREADS (default: 2)
    
    Args:
        src_root: Root directory containing CSV.gz files.
        parquet_root: Destination root for Parquet files (mirrors structure).
        
    Returns:
        True if conversion succeeded, False otherwise.
    """
    log = _get_logger()
    parquet_paths: list[Path] = []
    csv_files = list(src_root.rglob("*.csv.gz"))
    
    if not csv_files:
        if _has_rich:
            log.print(f"[red]No CSV files found in {src_root}[/red]")
        else:
            log.error(f"No CSV files found in {src_root}")
        return False

    # Optional: process small files first so progress moves smoothly
    try:
        csv_files.sort(key=lambda p: p.stat().st_size)
    except Exception:
        pass

    def _convert_one(csv_gz: Path) -> tuple[Optional[Path], float]:
        """Convert one CSV file and return the output path and time taken.
        
        Args:
            csv_gz: Path to the CSV.gz file to convert.
            
        Returns:
            Tuple of (output_path, elapsed_time). output_path is None on failure.
        """
        start = time.time()
        rel = csv_gz.relative_to(src_root)
        out = parquet_root / rel.with_suffix("").with_suffix(".parquet")
        out.parent.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect()
        try:
            mem_limit = os.environ.get("SLICES_DUCKDB_MEM", "3GB")
            threads = int(os.environ.get("SLICES_DUCKDB_THREADS", "2"))
            con.execute(f"SET memory_limit='{mem_limit}'")
            con.execute(f"PRAGMA threads={threads}")

            # Streamed CSV -> Parquet conversion with robust parsing
            sql = f"""
                COPY (
                  SELECT * FROM read_csv_auto(
                    '{csv_gz.as_posix()}',
                    sample_size=-1,
                    auto_detect=true,
                    nullstr=['', 'NULL', 'NA', 'N/A', '___'],
                    ignore_errors=false
                  )
                )
                TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
            con.execute(sql)
            elapsed = time.time() - start
            return out, elapsed
        except Exception as e:
            # Log error in outer scope, just return None here
            return None, time.time() - start
        finally:
            con.close()

    start_time = time.time()
    max_workers = max(1, int(os.environ.get("SLICES_CONVERT_MAX_WORKERS", "4")))

    total_files = len(csv_files)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_convert_one, f): f for f in csv_files}

        if _has_rich:
            log.print(
                f"[cyan]Converting {total_files} CSV files to Parquet "
                f"using {max_workers} workers...[/cyan]"
            )
        else:
            log.info(
                f"Converting {total_files} CSV files to Parquet using {max_workers} workers..."
            )

        for fut in as_completed(futures):
            try:
                result_path, _ = fut.result()
                if result_path is not None:
                    parquet_paths.append(result_path)
                    completed += 1

                    elapsed = time.time() - start_time
                    progress_pct = 100 * completed / total_files
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    
                    if _has_rich:
                        log.print(
                            f"[green]Progress:[/green] {completed}/{total_files} files "
                            f"({progress_pct:.1f}%) - "
                            f"Elapsed: {elapsed_str}"
                        )
                    else:
                        log.info(
                            f"Progress: {completed}/{total_files} files "
                            f"({progress_pct:.1f}%) - "
                            f"Elapsed: {elapsed_str}"
                        )
            except Exception as e:
                csv_file = futures[fut]
                if _has_rich:
                    log.print(
                        f"[red]Parquet conversion failed for {csv_file}: {e}[/red]"
                    )
                else:
                    log.error(f"Parquet conversion failed for {csv_file}: {e}")
                ex.shutdown(cancel_futures=True)
                return False

    elapsed_time = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    if _has_rich:
        log.print(
            f"[green]✓[/green] Converted {len(parquet_paths)} files to Parquet "
            f"under {parquet_root} in {elapsed_str}"
        )
    else:
        log.info(
            f"✓ Converted {len(parquet_paths)} files to Parquet "
            f"under {parquet_root} in {elapsed_str}"
        )
    return True


def convert_csv_to_parquet(
    csv_root: Path, parquet_root: Path, dataset_name: Optional[str] = None
) -> bool:
    """Public wrapper to convert CSV.gz files to Parquet for a dataset.
    
    Args:
        csv_root: Root folder containing hosp/ and icu/ CSV.gz files.
        parquet_root: Destination root for Parquet files mirroring structure.
        dataset_name: Optional dataset name for logging (e.g., 'mimic_iv').
        
    Returns:
        True if conversion succeeded, False otherwise.
        
    Example:
        >>> from pathlib import Path
        >>> convert_csv_to_parquet(
        ...     Path("/data/mimic-iv-csv"),
        ...     Path("/data/mimic-iv-parquet")
        ... )
    """
    log = _get_logger()
    
    if not csv_root.exists():
        if _has_rich:
            log.print(f"[red]CSV root not found: {csv_root}[/red]")
        else:
            log.error(f"CSV root not found: {csv_root}")
        return False
    
    parquet_root.mkdir(parents=True, exist_ok=True)
    
    if dataset_name:
        if _has_rich:
            log.print(f"[cyan]Converting {dataset_name} CSV files to Parquet...[/cyan]")
        else:
            log.info(f"Converting {dataset_name} CSV files to Parquet...")
    
    return _csv_to_parquet_all(csv_root, parquet_root)
