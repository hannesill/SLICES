#!/usr/bin/env python3
"""Audit the committed SLICES result packages.

Checks every package under ``results/`` that has an ``evidence_manifest.json``
or ``checksum_manifest.json``:

  - each listed file exists and matches its SHA256
  - row and column counts match where the manifest records them
  - no file in the package is missing from its manifest (README.md excepted)
  - no text file or parquet string column contains a local absolute path or a
    retired package name

Needs no raw ICU data, W&B access, or encoder weights.

Usage:
    uv run python scripts/audit_results.py
    uv run python scripts/audit_results.py --results-dir results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

MANIFEST_NAMES = ("evidence_manifest.json", "checksum_manifest.json")
UNLISTED_OK = {"README.md", *MANIFEST_NAMES}
LEAK_PATTERNS = {
    "local absolute path": re.compile(r"/Users/[^/\s]+|/home/[^/\s]+"),
    "sibling thesis repo path": re.compile(r"\.\./SLICES-thesis"),
    "retired package name": re.compile(r"slices-(?:final|thesis)_thesis-v1"),
}
TEXT_SUFFIXES = {".md", ".json", ".csv", ".txt", ".yaml", ".yml"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame | None:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def scan_text(label: str, text: str) -> list[str]:
    return [
        f"{label}: contains {name} ({match.group(0)!r})"
        for name, pattern in LEAK_PATTERNS.items()
        if (match := pattern.search(text))
    ]


def audit_package(package: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = json.loads(manifest_path.read_text())
    files: dict[str, dict] = manifest.get("files", {})
    if not files:
        return [f"{manifest_path}: lists no files"]

    for name, frozen in files.items():
        path = package / name
        if not path.exists():
            errors.append(f"{path}: missing")
            continue
        digest = sha256_file(path)
        if digest != frozen.get("sha256"):
            errors.append(f"{path}: sha256 {digest} != {frozen.get('sha256')}")
        table = read_table(path)
        if table is not None:
            if "rows" in frozen and len(table) != frozen["rows"]:
                errors.append(f"{path}: rows {len(table)} != {frozen['rows']}")
            if "columns" in frozen and len(table.columns) != frozen["columns"]:
                errors.append(f"{path}: columns {len(table.columns)} != {frozen['columns']}")

    for path in sorted(p for p in package.iterdir() if p.is_file()):
        if path.name not in files and path.name not in UNLISTED_OK:
            errors.append(f"{path}: not listed in {manifest_path.name}")
    return errors


def scan_package(package: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(p for p in package.iterdir() if p.is_file()):
        if path.suffix in TEXT_SUFFIXES:
            errors.extend(scan_text(str(path), path.read_text(errors="replace")))
        elif path.suffix == ".parquet":
            table = pd.read_parquet(path)
            for column in table.select_dtypes(include=["object", "string"]).columns:
                values = table[column].dropna().astype(str)
                blob = "\n".join(values.unique())
                errors.extend(scan_text(f"{path}[{column}]", blob))
    return errors


def audit(results_dir: Path) -> tuple[list[str], list[str]]:
    checked: list[str] = []
    errors: list[str] = []
    for package in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        manifests = [package / name for name in MANIFEST_NAMES if (package / name).exists()]
        if not manifests:
            errors.append(f"{package}: no evidence_manifest.json or checksum_manifest.json")
            continue
        for manifest in manifests:
            errors.extend(audit_package(package, manifest))
        errors.extend(scan_package(package))
        checked.append(package.name)
    index = results_dir / "README.md"
    if index.exists():
        errors.extend(scan_text(str(index), index.read_text()))
    return checked, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    checked, errors = audit(args.results_dir)
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} problem(s) in {args.results_dir}", file=sys.stderr)
        return 1
    print(f"OK: {len(checked)} result packages verified ({', '.join(checked)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
