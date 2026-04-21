#!/usr/bin/env python3
"""
Canonical results export pipeline for SLICES.

Pulls all experiment runs from W&B, extracts config + test metrics, and produces
three structured parquet files:

  - results/per_seed_results.parquet   — one row per W&B run (~3000 rows)
  - results/aggregated_results.parquet — one row per unique config (~600 rows),
                                         with mean/std/min/max across seeds
  - results/statistical_tests.parquet  — pairwise Wilcoxon + Bonferroni +
                                         Cohen's d significance table

Both files include wandb run IDs for traceability back to W&B.

Usage:
    uv run python scripts/export_results.py
    uv run python scripts/export_results.py --sprint 1 --validate-seeds 3
    uv run python scripts/export_results.py --paradigm mae jepa --dataset miiv
    uv run python scripts/export_results.py --output-dir results/sprint1 --sprint 1
    uv run python scripts/export_results.py --project slices-benchmark --revision benchmark-v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
from slices.eval.statistical import (
    bonferroni_correction,
    cohens_d,
    paired_wilcoxon_signed_rank,
)

import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Historical sprint tag to benchmark experiment type mapping.
# Core groups share the same config and only add seeds; they merge across groups.
# Non-core groups have ablation-specific pretrain configs that are not visible
# in the finetune W&B config, so the sprint tag remains a grouping key.
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
    "13": "temporal_contrastive",
}

CROSS_SPRINT_TYPES = {"core", "label_efficiency", "transfer"}
FIXED_SEED_EXPERIMENT_TYPES = {
    "core",
    "label_efficiency",
    "transfer",
    "hp_ablation",
    "capacity_pilot",
    "classical_baselines",
    "smart_reference",
    "temporal_contrastive",
}

# For core runs: the sprint tag is not part of the config identity.
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

# HP ablations merge across groups, but must preserve the upstream pretrain
# config and subtype so LR and mask-ratio families cannot collapse together.
HP_ABLATION_FINGERPRINT = CORE_FINGERPRINT + [
    "experiment_subtype",
    "upstream_pretrain_lr",
    "upstream_pretrain_mask_ratio",
]

# For per-group experiment families, the sprint tag remains part of the identity.
ABLATION_FINGERPRINT = HP_ABLATION_FINGERPRINT + [
    "sprint",
]

FINGERPRINT_AUDIT_COLUMNS = [
    "experiment_type",
    "experiment_subtype",
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
    "sprint",
    "upstream_pretrain_lr",
    "upstream_pretrain_mask_ratio",
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

# Canonical model variants used in the benchmark matrix.
MODEL_VARIANTS = {
    (64, 2): "default",
    (128, 4): "medium",
    (256, 4): "large",
}

PRIMARY_TEST_METRIC_BY_TASK = {
    "mortality_24h": "test/auprc",
    "mortality_hospital": "test/auprc",
    "mortality": "test/auprc",
    "aki_kdigo": "test/auprc",
    "los_remaining": "test/mae",
}

LOWER_IS_BETTER_METRICS = {
    "test/loss",
    "test/mae",
    "test/mse",
}

# Phases that correspond to evaluation runs (not pretraining).
EVAL_PHASES = ["finetune", "supervised", "gru_d", "xgboost", "baseline"]


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
    revision: list[str] | None = None,
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

    if revision and len(revision) == 1:
        tag_filters.append(f"revision:{revision[0]}")

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
    revision_set = set(revision) if revision and len(revision) > 1 else None

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
        if revision_set:
            if not any(f"revision:{r}" in tags for r in revision_set):
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


# Lookup tables for decoding run-name-encoded HP values.
# run_experiments.py encodes: str(v).replace(".", "")
#   str(2e-4)="0.0002" -> "00002", str(5e-4)="0.0005" -> "00005",
#   str(2e-3)="0.002" -> "0002", str(0.3)="0.3" -> "03", str(0.75)="0.75" -> "075"
_LR_DECODE = {
    "00002": 2e-4,
    "00005": 5e-4,
    "0002": 2e-3,
}
_MR_DECODE = {"03": 0.3, "075": 0.75}


def _recover_source_dataset(run_name: str, config: dict) -> str | None:
    """Recover transfer provenance from explicit config or historical names."""
    source_dataset = config.get("source_dataset")
    if source_dataset in {"miiv", "eicu", "combined"}:
        return source_dataset

    for candidate in [config.get("output_dir", ""), run_name]:
        match = re.search(r"_from_(miiv|eicu|combined)(?:_|$)", candidate or "")
        if match:
            return match.group(1)

    return None


def _recover_pretrain_metadata(
    run_name: str, config: dict
) -> tuple[float | None, float | None, str | None]:
    """Recover upstream pretrain metadata from config (new runs) or run name (historical).

    Returns (upstream_pretrain_lr, upstream_pretrain_mask_ratio, experiment_subtype).
    """
    # New runs have explicit fields
    up_lr = config.get("upstream_pretrain_lr")
    up_mr = config.get("upstream_pretrain_mask_ratio")
    subtype = config.get("experiment_subtype")
    if up_lr is not None or up_mr is not None:
        return up_lr, up_mr, subtype

    # Historical recovery: parse from run name / output_dir
    output_dir = config.get("output_dir", run_name)

    lr_match = re.search(r"_lr([^_]+)", output_dir)
    if lr_match:
        lr_str = lr_match.group(1)
        up_lr = _LR_DECODE.get(lr_str)
        if up_lr is not None:
            subtype = "lr_ablation"

    mr_match = re.search(r"_mask_ratio([^_]+)", output_dir)
    if mr_match:
        mr_str = mr_match.group(1)
        up_mr = _MR_DECODE.get(mr_str)
        if up_mr is not None:
            subtype = subtype or "mask_ablation"

    return up_lr, up_mr, subtype


def _infer_model_size(config: dict) -> str:
    """Infer model-size label from the encoder config."""
    d_model = _get_nested(config, "encoder.d_model")
    n_layers = _get_nested(config, "encoder.n_layers")
    sprint = str(config.get("sprint", ""))

    if d_model is None:
        return "default"

    variant = MODEL_VARIANTS.get((d_model, n_layers))
    if variant is not None:
        return variant

    if sprint == "7p":
        if d_model == 128:
            return "medium"
        if d_model == 256:
            return "large"

    if d_model == 64:
        return "default"
    if n_layers is None:
        return f"d{d_model}"
    return f"d{d_model}_L{n_layers}"


def extract_run(run, metric_keys: list[str]) -> dict:
    """Extract config + metrics from a W&B run in a single retry-protected call."""
    config, summary, run_id, run_url, run_name, tags, group, created_at = _retry(
        lambda: _load_run_data(run)
    )

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

    # Derive protocol from freeze_encoder. XGBoost runs do not use this field,
    # but they belong with the full-training / Protocol-B comparison family.
    freeze = _get_nested(config, "training.freeze_encoder")
    if freeze is True:
        protocol = "A"
    elif freeze is False:
        protocol = "B"
    elif config.get("paradigm") == "xgboost" or phase == "baseline":
        protocol = "B"
    else:
        protocol = None

    model_size = _infer_model_size(config)

    # Detect source_dataset for transfer runs from config, output_dir, or run name.
    source_dataset = _recover_source_dataset(run_name, config)

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

    # Historical group 10 contains both core (label_fraction=1.0) and
    # label-efficiency runs that merge with group 6 during aggregation.
    label_frac = config.get("label_fraction", 1.0)
    if experiment_type == "core" and label_frac is not None and float(label_frac) < 1.0:
        experiment_type = "label_efficiency"

    # Recover upstream pretrain metadata (from config for new runs, run name for historical)
    up_lr, up_mr, experiment_subtype = _recover_pretrain_metadata(run_name, config)

    if experiment_type in ("lr_ablation", "mask_ablation"):
        experiment_subtype = experiment_subtype or experiment_type
        experiment_type = "hp_ablation"

    # Reclassify HP ablation runs that would otherwise stay as "core"
    if experiment_type == "core" and experiment_subtype in ("lr_ablation", "mask_ablation"):
        experiment_type = "hp_ablation"

    # Historical group 10 also adds transfer seeds for group 7 scope.
    if experiment_type == "core" and source_dataset is not None:
        experiment_type = "transfer"

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
        "upstream_pretrain_lr": up_lr,
        "upstream_pretrain_mask_ratio": up_mr,
        "experiment_subtype": experiment_subtype,
        "_eval_checkpoint_source": summary.get("_eval_checkpoint_source", None),
        "_best_ckpt_path": summary.get("_best_ckpt_path", None),
        "_best_ckpt_load_ok": summary.get("_best_ckpt_load_ok", None),
        "_best_ckpt_error": summary.get("_best_ckpt_error", None),
    }
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# DataFrame Construction
# ---------------------------------------------------------------------------


def _fingerprint_for_experiment_type(experiment_type: str) -> list[str]:
    """Return the canonical fingerprint for a given experiment family."""
    if experiment_type in CROSS_SPRINT_TYPES:
        return CORE_FINGERPRINT
    if experiment_type == "hp_ablation":
        return HP_ABLATION_FINGERPRINT
    return ABLATION_FINGERPRINT


def _allowed_varying_columns(experiment_type: str) -> set[str]:
    """Columns allowed to vary within one canonical experiment family."""
    if experiment_type in CROSS_SPRINT_TYPES or experiment_type == "hp_ablation":
        return {"sprint"}
    return set()


def _fillna_for_grouping(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill NaNs so pandas groupby/drop_duplicates treat missing values deterministically."""
    work = df.copy()
    for col in columns:
        if col in work.columns:
            work[col] = work[col].fillna("__none__")
    return work


def _assert_no_ambiguous_fingerprint_collisions(df: pd.DataFrame) -> None:
    """Fail if the chosen fingerprint would collapse non-equivalent configurations."""
    collisions = []

    for experiment_type in sorted(df["experiment_type"].dropna().unique()):
        subset = df[df["experiment_type"] == experiment_type]
        if subset.empty:
            continue

        fingerprint = [*(_fingerprint_for_experiment_type(experiment_type)), "seed"]
        fingerprint = [c for c in fingerprint if c in subset.columns]
        if not fingerprint:
            continue

        allowed_vary = _allowed_varying_columns(experiment_type)
        identity_cols = [
            c for c in FINGERPRINT_AUDIT_COLUMNS if c in subset.columns and c not in allowed_vary
        ]

        work = _fillna_for_grouping(subset, list(dict.fromkeys(fingerprint + identity_cols)))
        grouped = work.groupby(fingerprint, dropna=False)

        for key, group in grouped:
            if len(group) <= 1:
                continue

            unique_identities = group[identity_cols].drop_duplicates()
            if len(unique_identities) > 1:
                if not isinstance(key, tuple):
                    key = (key,)
                collisions.append((experiment_type, dict(zip(fingerprint, key)), len(group)))

    if collisions:
        preview = []
        for experiment_type, fingerprint, count in collisions[:5]:
            desc = ", ".join(f"{k}={v}" for k, v in fingerprint.items())
            preview.append(f"{experiment_type}: {desc} ({count} rows)")
        raise RuntimeError(
            "Ambiguous export fingerprint would collapse distinct runs:\n  " + "\n  ".join(preview)
        )


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
    # Different experiment families use different fingerprints.
    df = df.sort_values("created_at", ascending=False)
    before = len(df)
    _assert_no_ambiguous_fingerprint_collisions(df)

    parts = []
    for experiment_type in sorted(df["experiment_type"].dropna().unique()):
        subset = df[df["experiment_type"] == experiment_type]
        dedup_cols = [*(_fingerprint_for_experiment_type(experiment_type)), "seed"]
        dedup_cols = [c for c in dedup_cols if c in subset.columns]
        parts.append(subset.drop_duplicates(subset=dedup_cols, keep="first"))
    df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]

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

    # Row count validation by experiment_type for auditability
    if "experiment_type" in df.columns:
        counts = df["experiment_type"].value_counts().sort_index()
        print("\n  Row counts by experiment_type after dedup:", file=sys.stderr)
        for etype, count in counts.items():
            print(f"    {etype}: {count}", file=sys.stderr)
        print(f"    TOTAL: {len(df)}", file=sys.stderr)

    return df


def _aggregate_group(df: pd.DataFrame, fingerprint_cols: list[str]) -> pd.DataFrame:
    """Aggregate a subset of runs by the given fingerprint columns.

    Returns a DataFrame with one row per unique fingerprint, containing
    mean/std/min/max of metrics and lists of run IDs/seeds/groups.
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
    - Core / label-efficiency / transfer runs: group without the sprint tag
    - HP ablations: group without the sprint tag, but include subtype + upstream pretrain config
    - Other experiment families: group with the sprint tag
    """
    parts = []
    cross_sprint = per_seed_df[per_seed_df["experiment_type"].isin(CROSS_SPRINT_TYPES)]
    if len(cross_sprint) > 0:
        print(
            f"  Aggregating {len(cross_sprint)} cross-sprint runs (merged experiment families)...",
            file=sys.stderr,
        )
        parts.append(_aggregate_group(cross_sprint, CORE_FINGERPRINT))

    hp_ablation = per_seed_df[per_seed_df["experiment_type"] == "hp_ablation"]
    if len(hp_ablation) > 0:
        print(
            "  Aggregating "
            f"{len(hp_ablation)} HP-ablation runs "
            "(merged by subtype + upstream config)...",
            file=sys.stderr,
        )
        parts.append(_aggregate_group(hp_ablation, HP_ABLATION_FINGERPRINT))

    per_sprint = per_seed_df[
        ~per_seed_df["experiment_type"].isin(CROSS_SPRINT_TYPES | {"hp_ablation"})
    ]
    if len(per_sprint) > 0:
        print(
            f"  Aggregating {len(per_sprint)} per-sprint runs (ablations, etc.)...", file=sys.stderr
        )
        parts.append(_aggregate_group(per_sprint, ABLATION_FINGERPRINT))

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


def _primary_metric_for_task(task_name: str | None) -> str | None:
    """Return the benchmark-primary test metric for a downstream task."""
    if task_name is None:
        return None
    if task_name in PRIMARY_TEST_METRIC_BY_TASK:
        return PRIMARY_TEST_METRIC_BY_TASK[task_name]
    if task_name.startswith("los"):
        return "test/mae"
    return "test/auprc"


def build_statistical_tests_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise paradigm significance tables from per-seed results.

    Comparisons are scoped by the canonical experiment fingerprint with
    `paradigm` and `task` removed, then pooled across matched `(task, seed)`
    pairs for tasks that share the same primary metric. This matches the benchmark
    plan's paired-across-seeds-and-tasks intent without mixing incompatible
    scales such as AUPRC and MAE in the same test.
    """
    if per_seed_df.empty or "experiment_type" not in per_seed_df.columns:
        return pd.DataFrame()

    rows = []

    for experiment_type in sorted(per_seed_df["experiment_type"].dropna().unique()):
        subset = per_seed_df[per_seed_df["experiment_type"] == experiment_type].copy()
        if subset.empty or "paradigm" not in subset.columns or "task" not in subset.columns:
            continue

        subset["primary_metric_name"] = subset["task"].map(_primary_metric_for_task)
        subset["primary_metric_value"] = [
            row.get(metric_name, float("nan")) if metric_name is not None else float("nan")
            for _, row, metric_name in zip(
                subset.index,
                subset.to_dict("records"),
                subset["primary_metric_name"],
                strict=True,
            )
        ]

        fingerprint = _fingerprint_for_experiment_type(experiment_type)
        scope_cols = [
            c for c in fingerprint if c in subset.columns and c not in {"paradigm", "task", "phase"}
        ]
        if "primary_metric_name" not in scope_cols:
            scope_cols.append("primary_metric_name")

        if not scope_cols:
            continue

        work = _fillna_for_grouping(subset, scope_cols)
        for scope_key, scope_group in work.groupby(scope_cols, dropna=False):
            if not isinstance(scope_key, tuple):
                scope_key = (scope_key,)

            scope = dict(zip(scope_cols, scope_key))
            for col, value in list(scope.items()):
                if value == "__none__":
                    scope[col] = None

            paradigms = sorted(
                paradigm
                for paradigm in scope_group["paradigm"].dropna().unique().tolist()
                if paradigm != "__none__"
            )
            if len(paradigms) < 2:
                continue

            higher_is_better = scope.get("primary_metric_name") not in LOWER_IS_BETTER_METRICS
            family_rows = []

            for paradigm_a, paradigm_b in combinations(paradigms, 2):
                pairs_a = scope_group[scope_group["paradigm"] == paradigm_a][
                    ["task", "seed", "primary_metric_value"]
                ].rename(columns={"primary_metric_value": "value_a"})
                pairs_b = scope_group[scope_group["paradigm"] == paradigm_b][
                    ["task", "seed", "primary_metric_value"]
                ].rename(columns={"primary_metric_value": "value_b"})

                paired = pairs_a.merge(pairs_b, on=["task", "seed"], how="inner")
                paired = paired.dropna(subset=["value_a", "value_b"])
                if paired.empty:
                    continue

                values_a = paired["value_a"].astype(float).tolist()
                values_b = paired["value_b"].astype(float).tolist()
                if higher_is_better:
                    improvement = [a - b for a, b in zip(values_a, values_b, strict=True)]
                else:
                    improvement = [b - a for a, b in zip(values_a, values_b, strict=True)]

                wilcoxon = paired_wilcoxon_signed_rank(improvement, [0.0] * len(improvement))
                effect_size = cohens_d(improvement, [0.0] * len(improvement), paired=True)
                mean_improvement = sum(improvement) / len(improvement)
                median_improvement = float(pd.Series(improvement).median())
                better_paradigm = (
                    paradigm_a
                    if mean_improvement > 0
                    else paradigm_b if mean_improvement < 0 else None
                )

                row = {
                    **scope,
                    "paradigm_a": paradigm_a,
                    "paradigm_b": paradigm_b,
                    "n_pairs": int(len(paired)),
                    "n_tasks": int(paired["task"].nunique()),
                    "task_list": json.dumps(sorted(paired["task"].unique().tolist())),
                    "seed_list": json.dumps(
                        sorted(paired["seed"].dropna().astype(int).unique().tolist())
                    ),
                    "score_a_mean": float(sum(values_a) / len(values_a)),
                    "score_b_mean": float(sum(values_b) / len(values_b)),
                    "mean_improvement": float(mean_improvement),
                    "median_improvement": median_improvement,
                    "better_paradigm": better_paradigm,
                    "wilcoxon_statistic": wilcoxon["statistic"],
                    "wilcoxon_z": wilcoxon["z_score"],
                    "p_value": wilcoxon["p_value"],
                    "n_nonzero_pairs": int(wilcoxon["n_nonzero_pairs"]),
                    "cohens_d": float(effect_size),
                    "significant_at_005": bool(
                        not math.isnan(wilcoxon["p_value"]) and wilcoxon["p_value"] < 0.05
                    ),
                }
                family_rows.append(row)

            if not family_rows:
                continue

            corrected = bonferroni_correction([row["p_value"] for row in family_rows])
            for row, corrected_p in zip(family_rows, corrected, strict=True):
                row["p_value_bonferroni"] = corrected_p
                row["significant_bonferroni_005"] = bool(
                    not math.isnan(corrected_p) and corrected_p < 0.05
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in [
            "experiment_type",
            "dataset",
            "protocol",
            "phase",
            "label_fraction",
            "primary_metric_name",
            "paradigm_a",
            "paradigm_b",
        ]
        if col in stats_df.columns
    ]
    if sort_cols:
        stats_df = stats_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return stats_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    per_seed_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    expected_core_seeds: int = 5,
) -> list[str]:
    """Validate results and return warning strings.

    Warns when fixed-seed experiment families have fewer seeds than expected.
    """
    warnings = []

    # Check fixed-seed experiment families for expected seed count
    if "experiment_type" in aggregated_df.columns:
        fixed_seed = aggregated_df[
            aggregated_df["experiment_type"].isin(FIXED_SEED_EXPERIMENT_TYPES)
        ]
        low_seed = fixed_seed[fixed_seed["n_seeds"] < expected_core_seeds]
        if len(low_seed) > 0:
            warnings.append(
                f"WARNING: {len(low_seed)}/{len(fixed_seed)} fixed-seed configs have fewer "
                f"than {expected_core_seeds} seeds:"
            )
            for _, row in low_seed.iterrows():
                desc = ", ".join(
                    f"{c}={row[c]}"
                    for c in ["paradigm", "dataset", "task", "protocol"]
                    if c in row and row[c] is not None
                )
                seeds = json.loads(row["seed_list"]) if pd.notna(row.get("seed_list")) else []
                warnings.append(
                    f"  experiment_type={row['experiment_type']}, {desc} — "
                    f"n_seeds={row['n_seeds']}, seeds={seeds}"
                )

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


def parse_args() -> argparse.Namespace:
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
        help="Filter to specific experiment group(s), e.g. --sprint 1 1b 2",
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
        "--revision",
        nargs="+",
        help="Filter to specific revision tag(s), e.g. --revision benchmark-v1",
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

    if not args.revision:
        env_revision = os.environ.get("REVISION") or os.environ.get("WANDB_REVISION")
        if env_revision:
            args.revision = [env_revision]
        else:
            parser.error(
                "--revision is required to avoid mixing reruns. "
                "Pass --revision <name> or set REVISION/WANDB_REVISION."
            )

    return args


def main():
    args = parse_args()
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
        revision=args.revision,
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

    print("\nBuilding statistical significance table...", file=sys.stderr)
    statistical_df = build_statistical_tests_df(per_seed_df)
    print(f"  Shape: {statistical_df.shape}", file=sys.stderr)

    # Validate
    warnings = validate(per_seed_df, aggregated_df, expected_core_seeds=args.expected_core_seeds)
    for w in warnings:
        print(w, file=sys.stderr)

    # Save
    per_seed_path = output_dir / "per_seed_results.parquet"
    aggregated_path = output_dir / "aggregated_results.parquet"
    statistical_path = output_dir / "statistical_tests.parquet"

    per_seed_df.to_parquet(per_seed_path, index=False)
    aggregated_df.to_parquet(aggregated_path, index=False)
    statistical_df.to_parquet(statistical_path, index=False)

    print("\nSaved:", file=sys.stderr)
    print(f"  {per_seed_path} ({len(per_seed_df)} rows)", file=sys.stderr)
    print(f"  {aggregated_path} ({len(aggregated_df)} rows)", file=sys.stderr)
    print(f"  {statistical_path} ({len(statistical_df)} rows)", file=sys.stderr)

    # Print quick summary to stdout for piping
    print("\n--- Quick Summary ---")
    print(
        f"Runs: {len(per_seed_df)}, Configs: {len(aggregated_df)}, "
        f"StatTests: {len(statistical_df)}"
    )
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
