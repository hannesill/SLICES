# Reproducibility

SLICES has three reproducibility paths with different access requirements.

## 1. Audit The Bundled Thesis Export

This path is public and local. It checks the committed metric export and
diagnostic packages under `results/` without raw ICU data, W&B credentials, or
encoder weights.

The frozen thesis metric package is `results/slices-thesis-v1/`.
Historical provenance: W&B `hannes-ill/slices-final`, revision `thesis-v1`.

Run from the repository root:

```bash
BAD_PATH_RE='/Users/hannesill|/home/hannesill|\.\./SLICES-thesis'
BAD_RESULT_RE='slices-(final|thesis)_thesis-v1'
BAD_SCRIPT_RE='scripts/(debug)/analyze_encoder_epochs\.py'
rg -n "${BAD_PATH_RE}|${BAD_RESULT_RE}|${BAD_SCRIPT_RE}" results/
```

Expected: no matches.

Verify the primary evidence manifest:

```bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

root = Path("results/slices-thesis-v1")
manifest = json.loads((root / "evidence_manifest.json").read_text())
errors = []

for name, frozen in manifest["files"].items():
    path = root / name
    if not path.exists():
        errors.append(f"{name}: missing")
        continue

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != frozen["sha256"]:
        errors.append(f"{name}: sha256 {digest} != {frozen['sha256']}")

    df = pd.read_parquet(path)
    if len(df) != frozen["rows"]:
        errors.append(f"{name}: rows {len(df)} != {frozen['rows']}")
    if len(df.columns) != frozen["columns"]:
        errors.append(f"{name}: columns {len(df.columns)} != {frozen['columns']}")

if errors:
    raise SystemExit("\n".join(errors))
print("primary evidence manifest ok")
PY
```

Verify the encoder audit and JEPA diagnostic checksum manifests:

```bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

import pandas as pd

packages = [
    Path("results/slices-thesis-encoder-audit-v1"),
    Path("results/slices-thesis-jepa-collapse-diagnostic-v1"),
]
errors = []

for root in packages:
    manifest = json.loads((root / "checksum_manifest.json").read_text())
    for name, frozen in manifest["files"].items():
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != frozen["sha256"]:
            errors.append(f"{path}: sha256 {digest} != {frozen['sha256']}")

        if name.endswith(".csv"):
            df = pd.read_csv(path)
        elif name.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            continue

        if "rows" in frozen and len(df) != frozen["rows"]:
            errors.append(f"{path}: rows {len(df)} != {frozen['rows']}")
        if "columns" in frozen and len(df.columns) != frozen["columns"]:
            errors.append(f"{path}: columns {len(df.columns)} != {frozen['columns']}")

if errors:
    raise SystemExit("\n".join(errors))
print("diagnostic checksum manifests ok")
PY
```

Run the source checks for export and diagnostics code:

```bash
uv run python -m py_compile \
  scripts/export_results.py \
  scripts/diagnostics/analyze_encoder_epochs.py \
  scripts/diagnostics/export_jepa_collapse_diagnostics.py \
  scripts/diagnostics/export_random_probe_diagnostic.py

uv run ruff check \
  scripts/export_results.py \
  scripts/diagnostics/analyze_encoder_epochs.py \
  scripts/diagnostics/export_jepa_collapse_diagnostics.py \
  scripts/diagnostics/export_random_probe_diagnostic.py

uv run pytest tests/test_export_results.py
```

Render thesis tables and figures into `/tmp` without touching tracked thesis
outputs:

```bash
(cd ../SLICES-thesis && uv run python scripts/viz/render_all.py \
  --results-dir ../SLICES/results/slices-thesis-v1 \
  --manifest ../SLICES/results/slices-thesis-v1/evidence_manifest.json \
  --figure-out-dir /tmp/slices-thesis-render/figures/generated \
  --figure-import-out /tmp/slices-thesis-render/content/generated/viz/main_figures.typ \
  --table-out-dir /tmp/slices-thesis-render/content/generated/tables \
  --manual-spec-out /tmp/slices-thesis-render/content/generated/viz/manual_figure_specs.md \
  --repo-root ../SLICES)
```

## 2. Future Public Smoke Rerun

This path is planned. It should use a small synthetic RICU-shaped fixture and
run on CPU without raw ICU access, W&B, or thesis encoder weights. Metrics from
synthetic data will be functional checks only, not scientific evidence.

Planned interface:

```bash
uv run slices reproduce smoke
```

## 3. Full Credentialed ICU And W&B Rerun

This path is private/credentialed. It rebuilds data, reruns training, exports
metrics, and optionally regenerates thesis artifacts.

Requirements:

- Credentialed MIMIC-IV and eICU access.
- R/RICU extraction environment and source dataset paths.
- W&B access to the target project, or a replacement local run registry.
- Sufficient training compute for the full fixed-seed benchmark.
- Encoder archives if auditing checkpoint provenance against local weights.

Raw ICU data, W&B access, and encoder weights are not required to audit the
committed metric export in path 1. They are required for full end-to-end reruns
and checkpoint-level weight audits.
