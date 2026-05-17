"""Build processed SLICES datasets from existing RICU parquet exports.

This is a convenience orchestrator for the common local rebuild path:

    uv run python scripts/preprocessing/build_processed_data.py --datasets miiv eicu --combined

It assumes ``data/ricu_output/{dataset}`` already exists. Use
``scripts/setup_and_extract.sh`` when you also need dependency installation or
the upstream RICU R export step.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

SUPPORTED_DATASETS = ("miiv", "eicu")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Python extraction, preparation, and optional combined-dataset build.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
        help="Base datasets to process. Defaults to miiv eicu.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also build data/processed/combined from miiv and eicu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for split generation. Defaults to 42.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args(argv)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def build_commands(
    datasets: Sequence[str],
    *,
    build_combined: bool,
    seed: int,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Build the subprocess command list for the requested processing plan."""
    ordered_datasets: list[str] = []
    for dataset in datasets:
        _append_unique(ordered_datasets, dataset)

    if build_combined:
        for dataset in SUPPORTED_DATASETS:
            _append_unique(ordered_datasets, dataset)

    commands: list[list[str]] = []
    for dataset in ordered_datasets:
        commands.append(
            [
                python_executable,
                "scripts/preprocessing/extract_ricu.py",
                f"dataset={dataset}",
            ]
        )
        commands.append(
            [
                python_executable,
                "scripts/preprocessing/prepare_dataset.py",
                f"dataset={dataset}",
                f"seed={seed}",
            ]
        )

    if build_combined:
        commands.append(
            [
                python_executable,
                "scripts/preprocessing/create_combined_dataset.py",
                "--source",
                "data/processed/miiv",
                "data/processed/eicu",
                "--names",
                "miiv",
                "eicu",
                "--output",
                "data/processed/combined",
                "--seed",
                str(seed),
            ]
        )

    return commands


def run_commands(commands: Sequence[Sequence[str]], *, cwd: Path, dry_run: bool) -> None:
    """Print and optionally execute commands."""
    for command in commands:
        print(f"$ {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=cwd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    commands = build_commands(
        args.datasets,
        build_combined=args.combined,
        seed=args.seed,
    )
    run_commands(commands, cwd=project_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
