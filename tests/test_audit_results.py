"""Tests for the committed result-package audit."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.audit_results import audit, main

REPO_RESULTS = Path(__file__).resolve().parents[1] / "results"


def _write_package(root: Path, frame: pd.DataFrame) -> Path:
    package = root / "pkg-v1"
    package.mkdir(parents=True)
    path = package / "table.parquet"
    frame.to_parquet(path, index=False)
    manifest = {
        "files": {
            "table.parquet": {
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        }
    }
    (package / "checksum_manifest.json").write_text(json.dumps(manifest))
    (root / "README.md").write_text("# results\n")
    return package


def test_committed_result_packages_pass_audit():
    assert main(["--results-dir", str(REPO_RESULTS)]) == 0


def test_audit_flags_checksum_drift(tmp_path):
    package = _write_package(tmp_path, pd.DataFrame({"a": [1, 2]}))
    pd.DataFrame({"a": [1, 3]}).to_parquet(package / "table.parquet", index=False)

    _, errors = audit(tmp_path)

    assert any("sha256" in error for error in errors)


def test_audit_flags_local_paths_in_parquet_strings(tmp_path):
    _write_package(tmp_path, pd.DataFrame({"path": ["/Users/someone/run/last.ckpt"]}))

    _, errors = audit(tmp_path)

    assert any("local absolute path" in error for error in errors)


def test_audit_flags_unlisted_files(tmp_path):
    package = _write_package(tmp_path, pd.DataFrame({"a": [1]}))
    shutil.copy(package / "table.parquet", package / "extra.parquet")

    _, errors = audit(tmp_path)

    assert any("not listed" in error for error in errors)
