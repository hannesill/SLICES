#!/usr/bin/env python3
"""
Canonical results export pipeline for SLICES.

Pulls all experiment runs from W&B, extracts config + test metrics, and produces
two structured parquet files:

  - results/per_seed_results.parquet   — one row per W&B run (~3000 rows)
  - results/aggregated_results.parquet — one row per unique config (~600 rows),
                                         with mean/std/min/max across seeds

Both files include wandb run IDs for traceability back to W&B.

Usage:
    uv run python scripts/export_results.py
    uv run python scripts/export_results.py --sprint 1 --validate-seeds 3
    uv run python scripts/export_results.py --paradigm mae jepa --dataset miiv
    uv run python scripts/export_results.py --output-dir results/sprint1 --sprint 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sprint → experiment type mapping.
# Core sprints share the same config and only add seeds; they merge across sprints.
# Non-core sprints have ablation-specific pretrain configs that aren't visible
# in the finetune W&B config, so sprint must remain a grouping key.
EXPERIMENT_TYPES = {
    "1": "core",
    "2": "core",
    "3": "core",
    "4": "core",
    "5": "core",
    "10": "core",
    "1b": "lr_ablation",
    "1c": "mask_ablation",
    "8": "hp_ablation",
    "6": "label_efficiency",
    "7": "transfer",
    "7p": "capacity_pilot",
    "11": "classical_baselines",
    "12": "smart_reference",
}

# For core runs: sprint is noise — same config across sprints, different seeds.
# lr/mask_ratio excluded because they're determined by protocol (redundant) or
# only in the pretrain config (NaN for finetune runs).
CORE_FINGERPRINT = [
    "experiment_type",
    "paradigm",
    "dataset",
    "task",
    "protocol",
    "label_fraction",
    "model_size",
    "source_dataset",
    "phase",
]

# For non-core runs: sprint distinguishes pretrain configs that aren't captured
# in the finetune W&B config (e.g., which LR/mask_ratio the pretrain used).
ABLATION_FINGERPRINT = CORE_FINGERPRINT + ["sprint"]

# Legacy fingerprint used for deduplication (includes sprint + all config dims).
DEDUP_FINGERPRINT = [
    "sprint",
    "paradigm",
    "dataset",
    "task",
    "protocol",
    "label_fraction",
    "lr",
    "mask_ratio",
    "model_size",
    "source_dataset",
    "phase",
]

# Metrics to extract from run summaries.
TEST_METRICS = [
    # Binary classification
    "test/auroc",
    "test/auprc",
    "test/accuracy",
    "test/f1",
    "test/precision",
    "test/recall",
    "test/specificity",
    "test/brier_score",
    "test/ece",
    # Regression (LOS)
    "test/mse",
    "test/mae",
    "test/r2",
    # Universal
    "test/loss",
    # Fairness (populated by scripts/eval/evaluate_fairness.py)
    "fairness/gender/worst_group_auroc",
    "fairness/gender/worst_group_auprc",
    "fairness/gender/auroc_gap",
    "fairness/gender/auprc_gap",
    "fairness/gender/demographic_parity_diff",
    "fairness/gender/equalized_odds_diff",
    "fairness/gender/disparate_impact_ratio",
    "fairness/age_group/worst_group_auroc",
    "fairness/age_group/worst_group_auprc",
    "fairness/age_group/auroc_gap",
    "fairness/age_group/auprc_gap",
    "fairness/age_group/demographic_parity_diff",
    "fairness/age_group/equalized_odds_diff",
    "fairness/age_group/disparate_impact_ratio",
    "fairness/race/worst_group_auroc",
    "fairness/race/worst_group_auprc",
    "fairness/race/auroc_gap",
    "fairness/race/auprc_gap",
    "fairness/race/demographic_parity_diff",
    "fairness/race/equalized_odds_diff",
    "fairness/race/disparate_impact_ratio",
]

VAL_METRICS = [
    "val/auroc",
    "val/auprc",
    "val/loss",
]

ALL_METRICS = TEST_METRICS + VAL_METRICS

# d_model values that indicate non-default capacity (Sprint 7p pilot).
# Default d_model=128 maps to "default"; other values are named here.
SIZED_MODELS = {128: "default", 256: "large"}

# Phases that correspond to evaluation runs (not pretraining).
EVAL_PHASES = ["finetune", "supervised", "gru_d", "xgboost"]


# ---------------------------------------------------------------------------
# W&B Fetching
# ---------------------------------------------------------------------------


def fetch_all_runs(
    project: str,
    entity: str | None = None,
    state: str = "finished",
    sprint: list[str] | None = None,
    paradigm: list[str] | None = None,
    dataset: list[str] | None = None,
    phase: list[str] | None = None,
) -> list:
    """Fetch runs from W&B with server-side filtering.

    Returns a list of wandb.Run objects.
    """
    api = wandb.Api(timeout=300)
    path = f"{entity}/{project}" if entity else project

    # Build server-side filters
    filters: dict = {}
    if state:
        filters["state"] = state

    # Tag-based filters — use $all so runs must match ALL specified tags.
    tag_filters: list[str] = []

    if sprint:
        if len(sprint) == 1:
            tag_filters.append(f"sprint:{sprint[0]}")
        # Multiple sprints: fetch all, filter client-side (W&B $all requires ALL tags)

    if paradigm and len(paradigm) == 1:
        tag_filters.append(f"paradigm:{paradigm[0]}")

    if dataset and len(dataset) == 1:
        tag_filters.append(f"dataset:{dataset[0]}")

    if phase and len(phase) == 1:
        tag_filters.append(f"phase:{phase[0]}")

    if tag_filters:
        filters["tags"] = {"$all": tag_filters}

    print(f"Fetching runs from {path}...", file=sys.stderr)
    print(f"  Server-side filters: {json.dumps(filters, default=str)}", file=sys.stderr)

    runs_iter = api.runs(path, filters=filters or {}, order="-created_at")

    # Client-side filtering for multi-value filters
    sprint_set = set(sprint) if sprint and len(sprint) > 1 else None
    paradigm_set = set(paradigm) if paradigm and len(paradigm) > 1 else None
    dataset_set = set(dataset) if dataset and len(dataset) > 1 else None
    phase_set = set(phase) if phase and len(phase) > 1 else None

    runs = []
    for run in runs_iter:
        tags = set(run.tags)

        if sprint_set:
            if not any(f"sprint:{s}" in tags for s in sprint_set):
                continue
        if paradigm_set:
            if not any(f"paradigm:{p}" in tags for p in paradigm_set):
                continue
        if dataset_set:
            if not any(f"dataset:{d}" in tags for d in dataset_set):
                continue
        if phase_set:
            if not any(f"phase:{p}" in tags for p in phase_set):
                continue

        runs.append(run)

    print(f"  Fetched {len(runs)} runs.", file=sys.stderr)
    return runs


# ---------------------------------------------------------------------------
# Config & Metric Extraction
# ---------------------------------------------------------------------------


def _get_nested(config: dict, dotted_key: str, default=None):
    """Get a value from a nested dict using a dotted key path."""
    parts = dotted_key.split(".")
    val = config
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val


def _retry(fn, max_retries=3, base_delay=5):
    """Retry a function with exponential backoff on timeout/connection errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(
                kw in err_str
                for kw in ["timeout", "timed out", "connection", "429", "500", "502", "503"]
            )
            if not is_retryable or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"    Retry {attempt + 1}/{max_retries} after {delay}s: {e!r}",
                file=sys.stderr,
            )
            time.sleep(delay)


def _load_run_data(run) -> tuple[dict, dict, str, str, str, list[str], str, str]:
    """Load all data from a W&B run in one call (minimizes lazy-load API hits).

    Returns (config, summary_dict, run_id, run_url, run_name, tags, group, created_at).
    """
    # Access all lazy-loaded properties together so retries cover them all
    config = dict(run.config)
    summary = dict(run.summary_metrics or {})
    return (
        config,
        summary,
        run.id,
        run.url,
        run.name or "",
        list(run.tags),
        run.group,
        run.created_at or "",
    )


def extract_run(run, metric_keys: list[str]) -> dict:
    """Extract config + metrics from a W&B run in a single retry-protected call."""
    config, summary, run_id, run_url, run_name, tags, group, created_at = _retry(
        lambda: _load_run_data(run)
    )

    # Derive protocol from freeze_encoder
    freeze = _get_nested(config, "training.freeze_encoder")
    if freeze is True:
        protocol = "A"
    elif freeze is False:
        protocol = "B"
    else:
        protocol = None

    # Derive model_size from encoder.d_model
    d_model = _get_nested(config, "encoder.d_model")
    if d_model is not None and d_model != 128:
        # Sprint 7p capacity pilot: medium=128, large=256
        model_size = SIZED_MODELS.get(d_model, f"d{d_model}")
    else:
        model_size = "default"

    # Detect source_dataset for transfer runs from run name or config
    source_dataset = config.get("source_dataset", None)
    if source_dataset is None:
        if "_from_" in run_name:
            parts = run_name.split("_from_")
            if len(parts) >= 2:
                source_part = parts[1].split("_")[0]
                if source_part in ("miiv", "eicu", "combined"):
                    source_dataset = source_part

    # Detect phase from tags
    phase = None
    for tag in tags:
        if tag.startswith("phase:"):
            phase = tag.split(":", 1)[1]
            break
    if phase is None:
        p = config.get("paradigm", "")
        if p in ("supervised", "gru_d", "xgboost"):
            phase = p
        else:
            phase = "finetune"

    # Extract metrics from summary
    metrics = {}
    for key in metric_keys:
        val = summary.get(key, None)
        if val is not None and isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                metrics[key] = float("nan")
            else:
                metrics[key] = float(val)
        else:
            metrics[key] = float("nan")

    sprint_str = str(config.get("sprint", ""))
    experiment_type = EXPERIMENT_TYPES.get(sprint_str, "unknown")

    row = {
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        "wandb_run_name": run_name,
        "wandb_group": group,
        "created_at": created_at,
        "experiment_type": experiment_type,
        "sprint": sprint_str,
        "paradigm": config.get("paradigm", None),
        "dataset": config.get("dataset", None),
        "task": _get_nested(config, "task.task_name", default=None),
        "seed": config.get("seed", None),
        "protocol": protocol,
        "label_fraction": config.get("label_fraction", 1.0),
        "lr": _get_nested(config, "optimizer.lr", default=None),
        "mask_ratio": _get_nested(config, "ssl.mask_ratio", default=None),
        "model_size": model_size,
        "source_dataset": source_dataset,
        "revision": config.get("revision", None),
        "phase": phase,
    }
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# DataFrame Construction
# ---------------------------------------------------------------------------


def build_per_seed_df(runs: list) -> pd.DataFrame:
    """Build the per-seed DataFrame from raw W&B runs.

    One row per run with config columns + metric columns + wandb IDs.
    """
    rows = []
    failed = []
    for i, run in enumerate(runs):
        if (i + 1) % 100 == 0:
            print(f"  Processing run {i + 1}/{len(runs)}...", file=sys.stderr)
        try:
            row = extract_run(run, ALL_METRICS)
            rows.append(row)
        except Exception as e:
            run_id = getattr(run, "id", "unknown")
            failed.append(run_id)
            print(f"  FAILED to extract run {run_id}: {e!r}", file=sys.stderr)
    if failed:
        print(
            f"  WARNING: {len(failed)} runs failed extraction and were skipped.",
            file=sys.stderr,
        )

    df = pd.DataFrame(rows)

    # Ensure correct dtypes
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    if "label_fraction" in df.columns:
        df["label_fraction"] = pd.to_numeric(df["label_fraction"], errors="coerce")
        df["label_fraction"] = df["label_fraction"].fillna(1.0)
    if "lr" in df.columns:
        df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    if "mask_ratio" in df.columns:
        df["mask_ratio"] = pd.to_numeric(df["mask_ratio"], errors="coerce")

    # Deduplicate: keep only the most recent run per (fingerprint + seed).
    # This handles reruns/revisions — the latest finished run is canonical.
    # Uses conditional fingerprints: core runs dedup by CORE_FINGERPRINT (which
    # excludes sprint/lr/mask_ratio), non-core by ABLATION_FINGERPRINT.
    df = df.sort_values("created_at", ascending=False)
    before = len(df)

    core_mask = df["experiment_type"] == "core"
    core_dedup_cols = [c for c in CORE_FINGERPRINT + ["seed"] if c in df.columns]
    ablation_dedup_cols = [c for c in ABLATION_FINGERPRINT + ["seed"] if c in df.columns]

    core_df = df[core_mask].drop_duplicates(subset=core_dedup_cols, keep="first")
    non_core_df = df[~core_mask].drop_duplicates(subset=ablation_dedup_cols, keep="first")
    df = pd.concat([core_df, non_core_df], ignore_index=True)

    after = len(df)
    if before != after:
        print(
            f"  Deduplicated: {before} -> {after} rows "
            f"({before - after} older duplicate runs removed).",
            file=sys.stderr,
        )

    # Sort for reproducibility
    sort_cols = [c for c in ["sprint", "paradigm", "dataset", "task", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return df


def _aggregate_group(df: pd.DataFrame, fingerprint_cols: list[str]) -> pd.DataFrame:
    """Aggregate a subset of runs by the given fingerprint columns.

    Returns a DataFrame with one row per unique fingerprint, containing
    mean/std/min/max of metrics and lists of run IDs/seeds/sprints.
    """
    metric_cols = [c for c in ALL_METRICS if c in df.columns]

    # Sentinel for NaN in groupby
    work = df.copy()
    for col in fingerprint_cols:
        if col in work.columns:
            work[col] = work[col].fillna("__none__")

    grouped = work.groupby(fingerprint_cols, dropna=False)
    agg_rows = []

    for fingerprint, group in grouped:
        if isinstance(fingerprint, str):
            fingerprint = (fingerprint,)
        row = dict(zip(fingerprint_cols, fingerprint))

        # Restore None from sentinel
        for col in fingerprint_cols:
            if row.get(col) == "__none__":
                row[col] = None

        # Run metadata
        row["n_seeds"] = len(group)
        row["wandb_run_ids"] = json.dumps(group["wandb_run_id"].tolist())
        row["seed_list"] = json.dumps(sorted(group["seed"].dropna().astype(int).unique().tolist()))
        row["sprint_list"] = json.dumps(sorted(group["sprint"].dropna().unique().tolist()))

        # Metric aggregation
        for metric in metric_cols:
            values = group[metric].dropna()
            if len(values) > 0:
                row[f"{metric}/mean"] = values.mean()
                row[f"{metric}/std"] = values.std(ddof=1) if len(values) > 1 else 0.0
                row[f"{metric}/min"] = values.min()
                row[f"{metric}/max"] = values.max()
            else:
                row[f"{metric}/mean"] = float("nan")
                row[f"{metric}/std"] = float("nan")
                row[f"{metric}/min"] = float("nan")
                row[f"{metric}/max"] = float("nan")

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def build_aggregated_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Group by fingerprint, compute mean/std/min/max across seeds.

    Uses conditional grouping:
    - Core runs (Sprints 1-5, 10): group WITHOUT sprint to merge seeds across sprints
    - Non-core runs (ablations, label efficiency, etc.): group WITH sprint
    """
    core_mask = per_seed_df["experiment_type"] == "core"
    core = per_seed_df[core_mask]
    non_core = per_seed_df[~core_mask]

    parts = []
    if len(core) > 0:
        print(f"  Aggregating {len(core)} core runs (cross-sprint)...", file=sys.stderr)
        parts.append(_aggregate_group(core, CORE_FINGERPRINT))
    if len(non_core) > 0:
        print(f"  Aggregating {len(non_core)} non-core runs (per-sprint)...", file=sys.stderr)
        parts.append(_aggregate_group(non_core, ABLATION_FINGERPRINT))

    if not parts:
        return pd.DataFrame()

    agg_df = pd.concat(parts, ignore_index=True)

    # Sort
    sort_cols = [
        c for c in ["experiment_type", "paradigm", "dataset", "task"] if c in agg_df.columns
    ]
    if sort_cols:
        agg_df = agg_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return agg_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    per_seed_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    expected_core_seeds: int = 5,
) -> list[str]:
    """Validate results and return warning strings.

    Only warns about core configs with fewer seeds than expected.
    Non-core configs have variable seed counts by design.
    """
    warnings = []

    # Check core configs for expected seed count
    if "experiment_type" in aggregated_df.columns:
        core = aggregated_df[aggregated_df["experiment_type"] == "core"]
        low_seed = core[core["n_seeds"] < expected_core_seeds]
        if len(low_seed) > 0:
            warnings.append(
                f"WARNING: {len(low_seed)}/{len(core)} core configs have fewer "
                f"than {expected_core_seeds} seeds:"
            )
            for _, row in low_seed.iterrows():
                desc = ", ".join(
                    f"{c}={row[c]}"
                    for c in ["paradigm", "dataset", "task", "protocol"]
                    if c in row and row[c] is not None
                )
                seeds = json.loads(row["seed_list"]) if pd.notna(row.get("seed_list")) else []
                warnings.append(f"  {desc} — n_seeds={row['n_seeds']}, seeds={seeds}")

    # Check for runs with no test metrics at all
    test_cols = [c for c in TEST_METRICS if c in per_seed_df.columns]
    if test_cols:
        all_nan = per_seed_df[test_cols].isna().all(axis=1)
        if all_nan.any():
            n_empty = all_nan.sum()
            warnings.append(
                f"WARNING: {n_empty} runs have no test metrics at all. "
                "These may be pretraining runs or crashed evaluations."
            )

    # Summary by experiment type
    n_runs = len(per_seed_df)
    n_configs = len(aggregated_df)
    print("\nValidation summary:", file=sys.stderr)
    print(f"  Total runs: {n_runs}", file=sys.stderr)
    print(f"  Unique configs: {n_configs}", file=sys.stderr)
    if "experiment_type" in aggregated_df.columns:
        for etype, group in aggregated_df.groupby("experiment_type"):
            seed_dist = dict(group["n_seeds"].value_counts().sort_index())
            print(f"  {etype}: {len(group)} configs, seeds: {seed_dist}", file=sys.stderr)
    print(f"  Warnings: {len(warnings)}", file=sys.stderr)

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export SLICES experiment results from W&B to parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "slices"),
        help="W&B project name (default: WANDB_PROJECT env var or 'slices')",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity name (default: WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--sprint",
        nargs="+",
        help="Filter to specific sprint(s), e.g. --sprint 1 1b 2",
    )
    parser.add_argument(
        "--paradigm",
        nargs="+",
        help="Filter to specific paradigm(s), e.g. --paradigm mae jepa",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Filter to specific dataset(s), e.g. --dataset miiv eicu",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=EVAL_PHASES,
        help=f"Filter to specific phase(s) (default: {EVAL_PHASES})",
    )
    parser.add_argument(
        "--state",
        default="finished",
        help="Run state filter (default: finished)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--expected-core-seeds",
        type=int,
        default=5,
        help="Expected seed count for core configs (default: 5)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch runs
    runs = fetch_all_runs(
        project=args.project,
        entity=args.entity,
        state=args.state,
        sprint=args.sprint,
        paradigm=args.paradigm,
        dataset=args.dataset,
        phase=args.phase,
    )

    if not runs:
        print("No runs found matching filters. Exiting.", file=sys.stderr)
        sys.exit(0)

    # Build DataFrames
    print(f"\nBuilding per-seed DataFrame from {len(runs)} runs...", file=sys.stderr)
    per_seed_df = build_per_seed_df(runs)
    print(f"  Shape: {per_seed_df.shape}", file=sys.stderr)

    print("\nBuilding aggregated DataFrame...", file=sys.stderr)
    aggregated_df = build_aggregated_df(per_seed_df)
    print(f"  Shape: {aggregated_df.shape}", file=sys.stderr)

    # Validate
    warnings = validate(per_seed_df, aggregated_df, expected_core_seeds=args.expected_core_seeds)
    for w in warnings:
        print(w, file=sys.stderr)

    # Save
    per_seed_path = output_dir / "per_seed_results.parquet"
    aggregated_path = output_dir / "aggregated_results.parquet"

    per_seed_df.to_parquet(per_seed_path, index=False)
    aggregated_df.to_parquet(aggregated_path, index=False)

    print("\nSaved:", file=sys.stderr)
    print(f"  {per_seed_path} ({len(per_seed_df)} rows)", file=sys.stderr)
    print(f"  {aggregated_path} ({len(aggregated_df)} rows)", file=sys.stderr)

    # Print quick summary to stdout for piping
    print("\n--- Quick Summary ---")
    print(f"Runs: {len(per_seed_df)}, Configs: {len(aggregated_df)}")
    if "experiment_type" in aggregated_df.columns:
        for etype, group in aggregated_df.groupby("experiment_type"):
            seed_dist = dict(group["n_seeds"].value_counts().sort_index())
            print(f"  {etype}: {len(group)} configs, seeds: {seed_dist}")
    if "paradigm" in aggregated_df.columns:
        print(f"Paradigms: {sorted(aggregated_df['paradigm'].dropna().unique())}")
    if "dataset" in aggregated_df.columns:
        print(f"Datasets: {sorted(aggregated_df['dataset'].dropna().unique())}")
    if "task" in aggregated_df.columns:
        print(f"Tasks: {sorted(aggregated_df['task'].dropna().unique())}")


if __name__ == "__main__":
    main()
