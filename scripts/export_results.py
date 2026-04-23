#!/usr/bin/env python3
"""Canonical class-based results export pipeline for SLICES.

Pulls final rerun-corpus W&B runs, extracts config plus test metrics, and writes
publication-oriented parquet files:

  - results/per_seed_results.parquet
  - results/aggregated_results.parquet
  - results/statistical_tests.parquet
  - results/label_efficiency_curves.parquet
  - results/capacity_study_comparison.parquet
  - results/classical_context.parquet
  - results/ts2vec_vs_core_contrastive.parquet

Aggregated metric columns include mean, standard deviation, min/max, and 95%
confidence intervals across finite seed values.

Usage:
    uv run python scripts/export_results.py \
        --project slices-thesis --revision thesis-v1 --entity <entity>
    uv run python scripts/export_results.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> \
        --experiment-class core_ssl_benchmark label_efficiency
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
import wandb
from scipy import stats as scipy_stats
from slices.eval.fairness_metadata import (
    FAIRNESS_ARTIFACT_PATH_KEY,
    FAIRNESS_ARTIFACT_SOURCE_KEY,
    FAIRNESS_CHECKPOINT_SOURCE_KEY,
    FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE,
    FAIRNESS_DEFAULT_PROTECTED_ATTRIBUTES,
    FAIRNESS_METADATA_COLUMNS,
    FAIRNESS_MIN_SUBGROUP_SIZE_KEY,
    FAIRNESS_PROTECTED_ATTRIBUTES_KEY,
    FAIRNESS_SCHEMA_VERSION_KEY,
    FAIRNESS_SCRIPT_VERSION,
    FAIRNESS_SCRIPT_VERSION_KEY,
    FAIRNESS_SUMMARY_SCHEMA_VERSION,
    canonical_artifact_id,
    decode_protected_attributes,
    normalize_protected_attributes,
)
from slices.eval.statistical import (
    bonferroni_correction,
    cohens_d,
    paired_wilcoxon_signed_rank,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_CLASSES = [
    "core_ssl_benchmark",
    "label_efficiency",
    "cross_dataset_transfer",
    "hp_robustness",
    "capacity_study",
    "classical_baselines",
    "ts2vec_extension",
    "smart_external_reference",
]

CONTEXTUAL_STAT_CLASSES = {
    "classical_context_full",
    "classical_context_label_efficiency",
}

FIXED_SEED_EXPERIMENT_CLASSES = set(EXPERIMENT_CLASSES)
EXPECTED_FIXED_SEEDS = {42, 123, 456, 789, 1011}
CAPACITY_LABEL_FRACTIONS = {0.01, 0.1, 0.5}
CAPACITY_PARADIGMS = {"mae", "supervised"}
THESIS_TASKS = {
    "mortality_24h",
    "mortality_hospital",
    "aki_kdigo",
    "los_remaining",
}

FINGERPRINT = [
    "experiment_class",
    "experiment_type",
    "experiment_subtype",
    "paradigm",
    "dataset",
    "task",
    "protocol",
    "label_fraction",
    "model_size",
    "source_dataset",
    "phase",
    "upstream_pretrain_lr",
    "upstream_pretrain_mask_ratio",
]

FINGERPRINT_AUDIT_COLUMNS = [
    *FINGERPRINT,
    "lr",
    "mask_ratio",
]

TEST_METRICS = [
    "test/auroc",
    "test/auprc",
    "test/accuracy",
    "test/f1",
    "test/precision",
    "test/recall",
    "test/specificity",
    "test/brier_score",
    "test/ece",
    "test/mse",
    "test/mae",
    "test/r2",
    "test/loss",
    "fairness/gender/worst_group_auroc",
    "fairness/gender/worst_group_auprc",
    "fairness/gender/auroc_gap",
    "fairness/gender/auprc_gap",
    "fairness/gender/demographic_parity_diff",
    "fairness/gender/equalized_odds_diff",
    "fairness/gender/disparate_impact_ratio",
    "fairness/gender/worst_group_mse",
    "fairness/gender/worst_group_mae",
    "fairness/gender/mse_gap",
    "fairness/gender/mae_gap",
    "fairness/gender/n_valid_groups",
    "fairness/gender/n_metric_valid_groups",
    "fairness/age_group/worst_group_auroc",
    "fairness/age_group/worst_group_auprc",
    "fairness/age_group/auroc_gap",
    "fairness/age_group/auprc_gap",
    "fairness/age_group/demographic_parity_diff",
    "fairness/age_group/equalized_odds_diff",
    "fairness/age_group/disparate_impact_ratio",
    "fairness/age_group/worst_group_mse",
    "fairness/age_group/worst_group_mae",
    "fairness/age_group/mse_gap",
    "fairness/age_group/mae_gap",
    "fairness/age_group/n_valid_groups",
    "fairness/age_group/n_metric_valid_groups",
    "fairness/race/worst_group_auroc",
    "fairness/race/worst_group_auprc",
    "fairness/race/auroc_gap",
    "fairness/race/auprc_gap",
    "fairness/race/demographic_parity_diff",
    "fairness/race/equalized_odds_diff",
    "fairness/race/disparate_impact_ratio",
    "fairness/race/worst_group_mse",
    "fairness/race/worst_group_mae",
    "fairness/race/mse_gap",
    "fairness/race/mae_gap",
    "fairness/race/n_valid_groups",
    "fairness/race/n_metric_valid_groups",
]

VAL_METRICS = [
    "val/auroc",
    "val/auprc",
    "val/loss",
]

ALL_METRICS = TEST_METRICS + VAL_METRICS
PERFORMANCE_TEST_METRICS = [metric for metric in TEST_METRICS if metric.startswith("test/")]

MODEL_VARIANTS = {
    (64, 2): "default",
    (128, 4): "medium",
    (256, 4): "large",
}

PRIMARY_TEST_METRIC_BY_TASK = {
    "mortality_24h": "test/auprc",
    "mortality_hospital": "test/auprc",
    "aki_kdigo": "test/auprc",
    "los_remaining": "test/mae",
}

LOWER_IS_BETTER_METRICS = {
    "test/loss",
    "test/mae",
    "test/mse",
}

EVAL_PHASES = ["finetune", "supervised", "baseline"]
FAIRNESS_ATTRIBUTES = FAIRNESS_DEFAULT_PROTECTED_ATTRIBUTES
FAIRNESS_SUMMARY_KEY_COLUMN = "_fairness_summary_keys"
BINARY_FAIRNESS_REQUIRED_METRICS = [
    "n_valid_groups",
    "n_metric_valid_groups",
    "worst_group_auroc",
    "worst_group_auprc",
    "auroc_gap",
    "auprc_gap",
    "demographic_parity_diff",
    "equalized_odds_diff",
    "disparate_impact_ratio",
]
REGRESSION_FAIRNESS_REQUIRED_METRICS = [
    "n_valid_groups",
    "worst_group_mse",
    "worst_group_mae",
    "mse_gap",
    "mae_gap",
]


# ---------------------------------------------------------------------------
# W&B Fetching
# ---------------------------------------------------------------------------


def fetch_all_runs(
    project: str,
    entity: str | None = None,
    state: str = "finished",
    experiment_class: list[str] | None = None,
    paradigm: list[str] | None = None,
    dataset: list[str] | None = None,
    phase: list[str] | None = None,
    revision: list[str] | None = None,
) -> list:
    """Fetch runs from W&B with server-side filters where W&B supports them."""
    api = wandb.Api(timeout=300)
    path = f"{entity}/{project}" if entity else project

    filters: dict = {}
    if state:
        filters["state"] = state

    tag_filters: list[str] = []
    if experiment_class and len(experiment_class) == 1:
        tag_filters.append(f"experiment_class:{experiment_class[0]}")
    if paradigm and len(paradigm) == 1:
        tag_filters.append(f"paradigm:{paradigm[0]}")
    if dataset and len(dataset) == 1:
        tag_filters.append(f"dataset:{dataset[0]}")
    if phase and len(phase) == 1:
        tag_filters.append(f"phase:{phase[0]}")
    if revision and len(revision) == 1:
        tag_filters.append(f"revision:{revision[0]}")
    if tag_filters:
        filters["$and"] = [{"tags": tag} for tag in tag_filters]

    print(f"Fetching runs from {path}...", file=sys.stderr)
    print(f"  Server-side filters: {json.dumps(filters, default=str)}", file=sys.stderr)
    runs_iter = api.runs(path, filters=filters or {}, order="-created_at")

    class_set = set(experiment_class) if experiment_class and len(experiment_class) > 1 else None
    paradigm_set = set(paradigm) if paradigm and len(paradigm) > 1 else None
    dataset_set = set(dataset) if dataset and len(dataset) > 1 else None
    phase_set = set(phase) if phase and len(phase) > 1 else None
    revision_set = set(revision) if revision and len(revision) > 1 else None

    runs = []
    for run in runs_iter:
        tags = set(run.tags)
        if class_set and not any(f"experiment_class:{value}" in tags for value in class_set):
            continue
        if paradigm_set and not any(f"paradigm:{value}" in tags for value in paradigm_set):
            continue
        if dataset_set and not any(f"dataset:{value}" in tags for value in dataset_set):
            continue
        if phase_set and not any(f"phase:{value}" in tags for value in phase_set):
            continue
        if revision_set and not any(f"revision:{value}" in tags for value in revision_set):
            continue
        runs.append(run)

    print(f"  Fetched {len(runs)} runs.", file=sys.stderr)
    return runs


# ---------------------------------------------------------------------------
# Config & Metric Extraction
# ---------------------------------------------------------------------------


def _get_nested(config: dict, dotted_key: str, default=None):
    parts = dotted_key.split(".")
    val = config
    for part in parts:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            return default
    return val


def _retry(fn, max_retries=3, base_delay=5):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            err_str = str(exc).lower()
            is_retryable = any(
                word in err_str
                for word in ["timeout", "timed out", "connection", "429", "500", "502", "503"]
            )
            if not is_retryable or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"    Retry {attempt + 1}/{max_retries} after {delay}s: {exc!r}",
                file=sys.stderr,
            )
            time.sleep(delay)


def _load_run_data(run) -> tuple[dict, dict, str, str, str, list[str], str, str]:
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


_LR_DECODE = {
    "00002": 2e-4,
    "00005": 5e-4,
    "0002": 2e-3,
}
_MR_DECODE = {"03": 0.3, "075": 0.75}


def _tag_value(tags: list[str], prefix: str) -> str | None:
    full_prefix = f"{prefix}:"
    for tag in tags:
        if tag.startswith(full_prefix):
            return tag.split(":", 1)[1]
    return None


def _recover_source_dataset(run_name: str, config: dict) -> str | None:
    source_dataset = config.get("source_dataset")
    if source_dataset in {"miiv", "eicu", "combined"}:
        return source_dataset
    for candidate in [config.get("output_dir", ""), run_name]:
        match = re.search(r"_from_(miiv|eicu|combined)(?:_|$)", candidate or "")
        if match:
            return match.group(1)
    return None


def _recover_pretrain_metadata(
    run_name: str,
    config: dict,
) -> tuple[float | None, float | None, str | None]:
    """Recover upstream pretrain metadata from explicit config or encoded names."""
    up_lr = config.get("upstream_pretrain_lr")
    up_mr = config.get("upstream_pretrain_mask_ratio")
    subtype = config.get("experiment_subtype")

    if up_lr is None and subtype in {"lr_sensitivity", "lr_ablation"}:
        up_lr = _get_nested(config, "optimizer.lr")
    if up_mr is None and subtype in {
        "mask_ratio_sensitivity",
        "view_mask_sensitivity",
        "mask_ablation",
    }:
        up_mr = _get_nested(config, "ssl.mask_ratio")
    if up_lr is not None or up_mr is not None:
        return up_lr, up_mr, subtype

    output_dir = config.get("output_dir", run_name)
    lr_match = re.search(r"_lr([^_]+)", output_dir)
    if lr_match:
        up_lr = _LR_DECODE.get(lr_match.group(1))
        if up_lr is not None:
            subtype = "lr_sensitivity"

    mr_match = re.search(r"_mask_?ratio([^_]+)", output_dir)
    if mr_match:
        up_mr = _MR_DECODE.get(mr_match.group(1))
        if up_mr is not None:
            subtype = subtype or "mask_ratio_sensitivity"

    return up_lr, up_mr, subtype


def _infer_model_size(config: dict) -> str:
    explicit = config.get("model_size")
    if explicit:
        return explicit
    model_markers = {
        str(value).lower()
        for value in [
            config.get("paradigm"),
            _get_nested(config, "ssl.name"),
            _get_nested(config, "encoder.name"),
        ]
        if value is not None
    }
    if "smart" in model_markers:
        return "default"
    d_model = _get_nested(config, "encoder.d_model")
    n_layers = _get_nested(config, "encoder.n_layers")
    if d_model is None:
        return "default"
    variant = MODEL_VARIANTS.get((d_model, n_layers))
    if variant is not None:
        return variant
    if d_model == 64:
        return "default"
    if n_layers is None:
        return f"d{d_model}"
    return f"d{d_model}_L{n_layers}"


def _metric_value_or_nan(value) -> float:
    """Coerce numeric W&B summary values to export floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    if math.isnan(value) or math.isinf(value):
        return float("nan")
    return float(value)


def extract_run(run, metric_keys: list[str]) -> dict:
    """Extract one W&B run to a class-based result row."""
    config, summary, run_id, run_url, run_name, tags, group, created_at = _retry(
        lambda: _load_run_data(run)
    )

    experiment_class = config.get("experiment_class") or _tag_value(tags, "experiment_class")
    if not experiment_class:
        raise RuntimeError(
            f"Run {run_id} has no experiment_class in config or tags; final exports fail closed."
        )

    phase = config.get("phase") or _tag_value(tags, "phase")
    if phase is None:
        paradigm = config.get("paradigm", "")
        if paradigm in {"supervised", "gru_d", "xgboost"}:
            phase = "baseline" if paradigm in {"gru_d", "xgboost"} else "supervised"
        else:
            phase = "finetune"

    freeze = _get_nested(config, "training.freeze_encoder")
    protocol = config.get("protocol") or _tag_value(tags, "protocol")
    if protocol is None:
        if freeze is True:
            protocol = "A"
        elif freeze is False or config.get("paradigm") == "xgboost" or phase == "baseline":
            protocol = "B"

    source_dataset = _recover_source_dataset(run_name, config)
    up_lr, up_mr, experiment_subtype = _recover_pretrain_metadata(run_name, config)

    metrics = {}
    for key in metric_keys:
        metrics[key] = _metric_value_or_nan(summary.get(key, None))
    for key, value in summary.items():
        if key.startswith("fairness/") and isinstance(value, (int, float)):
            metrics[key] = _metric_value_or_nan(value)
    fairness_summary_keys = sorted(
        key for key, value in summary.items() if key.startswith("fairness/") and value is not None
    )

    paradigm = config.get("paradigm") or _get_nested(config, "ssl.name")
    lr = _get_nested(config, "optimizer.lr", default=None)
    if paradigm == "xgboost":
        lr = _get_nested(config, "xgboost.learning_rate", default=lr)
    row = {
        "wandb_run_id": run_id,
        "wandb_run_url": run_url,
        "wandb_run_name": run_name,
        "wandb_group": group,
        "created_at": created_at,
        "experiment_class": experiment_class,
        "experiment_type": experiment_class,
        "experiment_subtype": experiment_subtype,
        "paradigm": paradigm,
        "dataset": config.get("dataset", None),
        "task": _get_nested(config, "task.task_name", default=None),
        "seed": config.get("seed", None),
        "protocol": protocol,
        "label_fraction": config.get("label_fraction", 1.0),
        "lr": lr,
        "mask_ratio": _get_nested(config, "ssl.mask_ratio", default=None),
        "model_size": _infer_model_size(config),
        "source_dataset": source_dataset,
        "revision": config.get("revision", None) or _tag_value(tags, "revision"),
        "launch_commit": config.get("launch_commit", None) or _tag_value(tags, "commit"),
        "phase": phase,
        "upstream_pretrain_lr": up_lr,
        "upstream_pretrain_mask_ratio": up_mr,
        "_output_dir": config.get("output_dir", None),
        "_eval_checkpoint_source": summary.get("_eval_checkpoint_source", None),
        "_best_ckpt_path": summary.get("_best_ckpt_path", None),
        "_best_ckpt_load_ok": summary.get("_best_ckpt_load_ok", None),
        "_best_ckpt_error": summary.get("_best_ckpt_error", None),
        FAIRNESS_SUMMARY_KEY_COLUMN: json.dumps(fairness_summary_keys),
    }
    for key in FAIRNESS_METADATA_COLUMNS:
        row[key] = summary.get(key, None)
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# DataFrame Construction
# ---------------------------------------------------------------------------


def _fillna_for_grouping(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in columns:
        if col in work.columns:
            work[col] = work[col].fillna("__none__")
    return work


def _assert_no_ambiguous_fingerprint_collisions(df: pd.DataFrame) -> None:
    collisions = []
    fingerprint = [col for col in [*FINGERPRINT, "seed"] if col in df.columns]
    identity_cols = [col for col in FINGERPRINT_AUDIT_COLUMNS if col in df.columns]
    if not fingerprint:
        return

    work = _fillna_for_grouping(df, list(dict.fromkeys(fingerprint + identity_cols)))
    for key, group in work.groupby(fingerprint, dropna=False):
        if len(group) <= 1:
            continue
        if len(group[identity_cols].drop_duplicates()) > 1:
            if not isinstance(key, tuple):
                key = (key,)
            collisions.append((dict(zip(fingerprint, key)), len(group)))

    if collisions:
        preview = []
        for fingerprint_values, count in collisions[:5]:
            desc = ", ".join(f"{key}={value}" for key, value in fingerprint_values.items())
            preview.append(f"{desc} ({count} rows)")
        raise RuntimeError(
            "Ambiguous export fingerprint would collapse distinct runs:\n  " + "\n  ".join(preview)
        )


def build_per_seed_df(
    runs: list,
    allow_extraction_failures: bool = False,
    allow_duplicate_fingerprints: bool = False,
) -> pd.DataFrame:
    rows = []
    failed = []
    for index, run in enumerate(runs):
        if (index + 1) % 100 == 0:
            print(f"  Processing run {index + 1}/{len(runs)}...", file=sys.stderr)
        try:
            rows.append(extract_run(run, ALL_METRICS))
        except Exception as exc:
            run_id = getattr(run, "id", "unknown")
            failed.append(run_id)
            print(f"  FAILED to extract run {run_id}: {exc!r}", file=sys.stderr)

    if failed:
        print(f"  WARNING: {len(failed)} runs failed extraction and were skipped.", file=sys.stderr)
        if not allow_extraction_failures:
            preview = ", ".join(failed[:10])
            raise RuntimeError(
                f"{len(failed)} W&B runs failed extraction and were skipped: {preview}. "
                "Pass --allow-extraction-failures only for exploratory exports."
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    if "label_fraction" in df.columns:
        df["label_fraction"] = pd.to_numeric(df["label_fraction"], errors="coerce").fillna(1.0)
    if "lr" in df.columns:
        df["lr"] = pd.to_numeric(df["lr"], errors="coerce")
    if "mask_ratio" in df.columns:
        df["mask_ratio"] = pd.to_numeric(df["mask_ratio"], errors="coerce")

    df = df.sort_values("created_at", ascending=False)
    before = len(df)
    _assert_no_ambiguous_fingerprint_collisions(df)
    dedup_cols = [col for col in [*FINGERPRINT, "seed"] if col in df.columns]
    duplicate_rows = df[df.duplicated(subset=dedup_cols, keep=False)]
    if not duplicate_rows.empty and not allow_duplicate_fingerprints:
        preview = []
        for _, row in duplicate_rows.head(10).iterrows():
            desc = ", ".join(f"{col}={row.get(col)}" for col in dedup_cols)
            run_id = row.get("wandb_run_id", "unknown")
            preview.append(f"{desc}, wandb_run_id={run_id}")
        raise RuntimeError(
            "Duplicate exact export fingerprints would be silently collapsed:\n  "
            + "\n  ".join(preview)
            + "\nPass --allow-duplicate-fingerprints only for exploratory cleanup exports."
        )
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    after = len(df)
    if before != after:
        print(
            f"  Deduplicated: {before} -> {after} rows "
            f"({before - after} older duplicate runs removed).",
            file=sys.stderr,
        )

    sort_cols = [
        col
        for col in ["experiment_class", "paradigm", "dataset", "task", "seed"]
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    counts = df["experiment_class"].value_counts().sort_index()
    print("\n  Row counts by experiment_class after dedup:", file=sys.stderr)
    for experiment_class, count in counts.items():
        print(f"    {experiment_class}: {count}", file=sys.stderr)
    print(f"    TOTAL: {len(df)}", file=sys.stderr)
    return df


def filter_thesis_tasks(
    per_seed_df: pd.DataFrame,
    include_extension_tasks: bool = False,
) -> pd.DataFrame:
    """Restrict publication exports to the fixed thesis task set by default."""
    if include_extension_tasks or per_seed_df.empty or "task" not in per_seed_df.columns:
        return per_seed_df

    task_mask = per_seed_df["task"].isna() | per_seed_df["task"].isin(THESIS_TASKS)
    dropped = per_seed_df.loc[~task_mask]
    if not dropped.empty:
        counts = dropped["task"].value_counts(dropna=False).to_dict()
        print(
            "  Dropped extension-task rows outside the thesis task set: "
            f"{len(dropped)} ({counts})",
            file=sys.stderr,
        )
    return per_seed_df.loc[task_mask].reset_index(drop=True)


def _mean_ci95(values: pd.Series) -> tuple[float, float]:
    """Return a two-sided 95% CI for a seed mean using Student's t interval."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    std = values.std(ddof=1)
    if pd.isna(std):
        return float("nan"), float("nan")
    if std == 0:
        mean = float(values.mean())
        return mean, mean

    mean = float(values.mean())
    half_width = float(scipy_stats.t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    return mean - half_width, mean + half_width


def _aggregate_group(df: pd.DataFrame, fingerprint_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        col
        for col in df.columns
        if col in ALL_METRICS or (isinstance(col, str) and col.startswith("fairness/"))
    ]
    work = _fillna_for_grouping(df, fingerprint_cols)
    grouped = work.groupby(fingerprint_cols, dropna=False)
    rows = []

    for fingerprint, group in grouped:
        if isinstance(fingerprint, str):
            fingerprint = (fingerprint,)
        row = dict(zip(fingerprint_cols, fingerprint))
        for col in fingerprint_cols:
            if row.get(col) == "__none__":
                row[col] = None

        row["n_seeds"] = len(group)
        row["wandb_run_ids"] = json.dumps(group["wandb_run_id"].tolist())
        row["seed_list"] = json.dumps(sorted(group["seed"].dropna().astype(int).unique().tolist()))
        row["experiment_class_list"] = json.dumps(
            sorted(group["experiment_class"].dropna().unique().tolist())
        )

        for metric in metric_cols:
            values = group[metric].dropna()
            row[f"{metric}/n"] = int(len(values))
            if len(values) > 0:
                row[f"{metric}/mean"] = values.mean()
                row[f"{metric}/std"] = values.std(ddof=1) if len(values) > 1 else 0.0
                row[f"{metric}/min"] = values.min()
                row[f"{metric}/max"] = values.max()
                ci_lower, ci_upper = _mean_ci95(values)
                row[f"{metric}/ci95_lower"] = ci_lower
                row[f"{metric}/ci95_upper"] = ci_upper
            else:
                row[f"{metric}/mean"] = float("nan")
                row[f"{metric}/std"] = float("nan")
                row[f"{metric}/min"] = float("nan")
                row[f"{metric}/max"] = float("nan")
                row[f"{metric}/ci95_lower"] = float("nan")
                row[f"{metric}/ci95_upper"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def build_aggregated_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()
    fingerprint_cols = [col for col in FINGERPRINT if col in per_seed_df.columns]
    agg_df = _aggregate_group(per_seed_df, fingerprint_cols)
    sort_cols = [
        col for col in ["experiment_class", "paradigm", "dataset", "task"] if col in agg_df.columns
    ]
    if sort_cols:
        agg_df = agg_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return agg_df


# ---------------------------------------------------------------------------
# Derived Comparison Views
# ---------------------------------------------------------------------------


def _copy_for_view(df: pd.DataFrame, view_name: str, target_class: str) -> pd.DataFrame:
    view = df.copy()
    view["source_experiment_class"] = view["experiment_class"]
    view["experiment_class"] = target_class
    view["experiment_type"] = target_class
    view["comparison_view"] = view_name
    return view


def build_label_efficiency_curve_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()
    label_fraction = pd.to_numeric(per_seed_df["label_fraction"], errors="coerce").fillna(1.0)
    own = per_seed_df[per_seed_df["experiment_class"] == "label_efficiency"].copy()
    endpoint = per_seed_df[
        (per_seed_df["experiment_class"] == "core_ssl_benchmark")
        & (label_fraction == 1.0)
        & (per_seed_df["phase"].isin(["finetune", "supervised"]))
        & (per_seed_df["source_dataset"].isna())
        & (per_seed_df["model_size"] == "default")
    ].copy()
    parts = []
    if not own.empty:
        parts.append(_copy_for_view(own, "label_efficiency_with_core_endpoint", "label_efficiency"))
    if not endpoint.empty:
        parts.append(
            _copy_for_view(endpoint, "label_efficiency_with_core_endpoint", "label_efficiency")
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_capacity_study_comparison_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()
    label_fraction = pd.to_numeric(per_seed_df["label_fraction"], errors="coerce").fillna(1.0)
    own = per_seed_df[per_seed_df["experiment_class"] == "capacity_study"].copy()
    default_baseline = per_seed_df[
        (per_seed_df["experiment_class"].isin(["core_ssl_benchmark", "label_efficiency"]))
        & (per_seed_df["dataset"] == "miiv")
        & (per_seed_df["task"] == "mortality_24h")
        & (per_seed_df["paradigm"].isin(CAPACITY_PARADIGMS))
        & (per_seed_df["model_size"] == "default")
        & (label_fraction.isin(CAPACITY_LABEL_FRACTIONS | {1.0}))
    ].copy()
    parts = []
    if not own.empty:
        parts.append(_copy_for_view(own, "capacity_with_default_baseline", "capacity_study"))
    if not default_baseline.empty:
        parts.append(
            _copy_for_view(default_baseline, "capacity_with_default_baseline", "capacity_study")
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_classical_context_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty:
        return pd.DataFrame()
    label_fraction = pd.to_numeric(per_seed_df["label_fraction"], errors="coerce").fillna(1.0)
    protocol_b = per_seed_df["protocol"] == "B"
    full = per_seed_df[
        protocol_b
        & (label_fraction == 1.0)
        & (per_seed_df["experiment_class"].isin(["core_ssl_benchmark", "classical_baselines"]))
    ].copy()
    low_label = per_seed_df[
        protocol_b
        & (label_fraction < 1.0)
        & (per_seed_df["experiment_class"].isin(["label_efficiency", "classical_baselines"]))
    ].copy()

    parts = []
    if not full.empty:
        parts.append(_copy_for_view(full, "classical_context_full", "classical_context_full"))
    if not low_label.empty:
        parts.append(
            _copy_for_view(
                low_label,
                "classical_context_label_efficiency",
                "classical_context_label_efficiency",
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistical Tables
# ---------------------------------------------------------------------------


def _primary_metric_for_task(task_name: str | None) -> str | None:
    if task_name is None:
        return None
    if task_name in PRIMARY_TEST_METRIC_BY_TASK:
        return PRIMARY_TEST_METRIC_BY_TASK[task_name]
    if task_name.startswith("los"):
        return "test/mae"
    return "test/auprc"


def _format_task_seed_pairs(keys: set[tuple[str, int]]) -> str:
    return json.dumps([{"task": task, "seed": int(seed)} for task, seed in sorted(keys)])


def _metric_pairs_for_paradigm(
    scope_group: pd.DataFrame,
    paradigm: str,
    value_col: str = "primary_metric_value",
) -> pd.DataFrame:
    pairs = scope_group[scope_group["paradigm"] == paradigm][["task", "seed", value_col]].rename(
        columns={value_col: "value"}
    )
    pairs = pairs.dropna(subset=["task", "seed", "value"]).copy()
    if not pairs.empty:
        pairs["seed"] = pairs["seed"].astype(int)
        pairs["value"] = pairs["value"].astype(float)
    return pairs


def _paired_metric_frame(
    scope_group: pd.DataFrame,
    paradigm_a: str,
    paradigm_b: str,
    value_col: str = "primary_metric_value",
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    pairs_a = _metric_pairs_for_paradigm(scope_group, paradigm_a, value_col=value_col)
    pairs_b = _metric_pairs_for_paradigm(scope_group, paradigm_b, value_col=value_col)
    keys_a = set(zip(pairs_a["task"], pairs_a["seed"], strict=True))
    keys_b = set(zip(pairs_b["task"], pairs_b["seed"], strict=True))
    shared_keys = keys_a & keys_b
    union_keys = keys_a | keys_b
    paired = pairs_a.rename(columns={"value": "value_a"}).merge(
        pairs_b.rename(columns={"value": "value_b"}),
        on=["task", "seed"],
        how="inner",
    )
    coverage = {
        "n_task_seed_pairs_a": len(keys_a),
        "n_task_seed_pairs_b": len(keys_b),
        "n_shared_task_seed_pairs": len(shared_keys),
        "n_union_task_seed_pairs": len(union_keys),
        "missing_task_seed_pairs_a": _format_task_seed_pairs(keys_b - keys_a),
        "missing_task_seed_pairs_b": _format_task_seed_pairs(keys_a - keys_b),
    }
    return paired, coverage


def _paired_stat_row(
    scope: dict,
    paired: pd.DataFrame,
    coverage: dict[str, int | str],
    paradigm_a: str,
    paradigm_b: str,
    higher_is_better: bool,
) -> dict:
    values_a = paired["value_a"].astype(float).tolist()
    values_b = paired["value_b"].astype(float).tolist()
    if higher_is_better:
        improvement = [a - b for a, b in zip(values_a, values_b, strict=True)]
    else:
        improvement = [b - a for a, b in zip(values_a, values_b, strict=True)]
    wilcoxon = paired_wilcoxon_signed_rank(improvement, [0.0] * len(improvement))
    effect_size = cohens_d(improvement, [0.0] * len(improvement), paired=True)
    mean_improvement = sum(improvement) / len(improvement)
    return {
        **scope,
        "paradigm_a": paradigm_a,
        "paradigm_b": paradigm_b,
        "n_pairs": int(len(paired)),
        "n_tasks": int(paired["task"].nunique()),
        "task_list": json.dumps(sorted(paired["task"].unique().tolist())),
        "seed_list": json.dumps(sorted(paired["seed"].dropna().astype(int).unique().tolist())),
        **coverage,
        "score_a_mean": float(sum(values_a) / len(values_a)),
        "score_b_mean": float(sum(values_b) / len(values_b)),
        "mean_improvement": float(mean_improvement),
        "median_improvement": float(pd.Series(improvement).median()),
        "better_paradigm": (
            paradigm_a if mean_improvement > 0 else paradigm_b if mean_improvement < 0 else None
        ),
        "wilcoxon_statistic": wilcoxon["statistic"],
        "wilcoxon_z": wilcoxon["z_score"],
        "p_value": wilcoxon["p_value"],
        "n_nonzero_pairs": int(wilcoxon["n_nonzero_pairs"]),
        "cohens_d": float(effect_size),
        "significant_at_005": bool(
            not math.isnan(wilcoxon["p_value"]) and wilcoxon["p_value"] < 0.05
        ),
    }


def _add_contextual_classical_stat_rows(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment_class", "protocol", "label_fraction", "paradigm"}
    if per_seed_df.empty or not required.issubset(per_seed_df.columns):
        return per_seed_df

    parts = [per_seed_df]
    context = build_classical_context_df(per_seed_df)
    if not context.empty:
        parts.append(context)
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else per_seed_df


def build_statistical_tests_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    if per_seed_df.empty or "experiment_class" not in per_seed_df.columns:
        return pd.DataFrame()

    per_seed_df = _add_contextual_classical_stat_rows(per_seed_df)
    rows = []

    for experiment_class in sorted(per_seed_df["experiment_class"].dropna().unique()):
        subset = per_seed_df[per_seed_df["experiment_class"] == experiment_class].copy()
        if subset.empty or "paradigm" not in subset.columns or "task" not in subset.columns:
            continue
        subset["primary_metric_name"] = subset["task"].map(_primary_metric_for_task)
        subset["primary_metric_value"] = [
            row.get(metric_name, float("nan")) if metric_name is not None else float("nan")
            for row, metric_name in zip(
                subset.to_dict("records"),
                subset["primary_metric_name"],
                strict=True,
            )
        ]

        base_scope_cols = [
            col
            for col in FINGERPRINT
            if col in subset.columns and col not in {"paradigm", "task", "phase"}
        ]
        if "primary_metric_name" not in base_scope_cols:
            base_scope_cols.append("primary_metric_name")

        scope_defs = [
            ("omnibus_primary_metric", base_scope_cols, 2),
            ("per_task", [*base_scope_cols, "task"], 1),
        ]
        for comparison_scope, raw_scope_cols, min_tasks in scope_defs:
            scope_cols = list(dict.fromkeys(col for col in raw_scope_cols if col in subset.columns))
            work = _fillna_for_grouping(subset, scope_cols)
            for scope_key, scope_group in work.groupby(scope_cols, dropna=False):
                if scope_group["task"].nunique() < min_tasks:
                    continue
                if not isinstance(scope_key, tuple):
                    scope_key = (scope_key,)
                scope = dict(zip(scope_cols, scope_key))
                for col, value in list(scope.items()):
                    if value == "__none__":
                        scope[col] = None
                scope["comparison_scope"] = comparison_scope

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
                    if experiment_class in CONTEXTUAL_STAT_CLASSES:
                        a_classical = paradigm_a in {"gru_d", "xgboost"}
                        b_classical = paradigm_b in {"gru_d", "xgboost"}
                        if a_classical == b_classical:
                            continue
                    paired, coverage = _paired_metric_frame(scope_group, paradigm_a, paradigm_b)
                    if paired.empty:
                        continue
                    family_rows.append(
                        _paired_stat_row(
                            scope=scope,
                            paired=paired,
                            coverage=coverage,
                            paradigm_a=paradigm_a,
                            paradigm_b=paradigm_b,
                            higher_is_better=higher_is_better,
                        )
                    )

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
            "experiment_class",
            "comparison_scope",
            "dataset",
            "task",
            "protocol",
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


def build_ts2vec_vs_core_contrastive_df(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment_class", "paradigm", "task", "seed"}
    if per_seed_df.empty or not required.issubset(per_seed_df.columns):
        return pd.DataFrame()

    subset = per_seed_df[
        (
            (per_seed_df["experiment_class"] == "ts2vec_extension")
            & (per_seed_df["paradigm"] == "ts2vec")
        )
        | (
            (per_seed_df["experiment_class"] == "core_ssl_benchmark")
            & (per_seed_df["paradigm"] == "contrastive")
        )
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["primary_metric_name"] = subset["task"].map(_primary_metric_for_task)
    subset["primary_metric_value"] = [
        row.get(metric_name, float("nan")) if metric_name is not None else float("nan")
        for row, metric_name in zip(
            subset.to_dict("records"),
            subset["primary_metric_name"],
            strict=True,
        )
    ]

    scope_cols = [
        col
        for col in [
            "dataset",
            "protocol",
            "label_fraction",
            "model_size",
            "source_dataset",
            "primary_metric_name",
        ]
        if col in subset.columns
    ]
    rows = []
    work = _fillna_for_grouping(subset, scope_cols)
    for scope_key, scope_group in work.groupby(scope_cols, dropna=False):
        if not isinstance(scope_key, tuple):
            scope_key = (scope_key,)
        scope = dict(zip(scope_cols, scope_key))
        for col, value in list(scope.items()):
            if value == "__none__":
                scope[col] = None
        if not {"contrastive", "ts2vec"}.issubset(set(scope_group["paradigm"].dropna())):
            continue
        paired, coverage = _paired_metric_frame(scope_group, "ts2vec", "contrastive")
        if paired.empty:
            continue
        higher_is_better = scope.get("primary_metric_name") not in LOWER_IS_BETTER_METRICS
        rows.append(
            _paired_stat_row(
                scope={
                    **scope,
                    "comparison_type": "ts2vec_vs_core_contrastive",
                    "experiment_class_a": "ts2vec_extension",
                    "experiment_class_b": "core_ssl_benchmark",
                },
                paired=paired,
                coverage=coverage,
                paradigm_a="ts2vec",
                paradigm_b="contrastive",
                higher_is_better=higher_is_better,
            )
        )

    if not rows:
        return pd.DataFrame()
    corrected = bonferroni_correction([row["p_value"] for row in rows])
    for row, corrected_p in zip(rows, corrected, strict=True):
        row["p_value_bonferroni"] = corrected_p
        row["significant_bonferroni_005"] = bool(not math.isnan(corrected_p) and corrected_p < 0.05)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_missing_export_value(value) -> bool:
    """Return whether a scalar export value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _is_true_export_value(value) -> bool:
    """Return whether a W&B summary value represents boolean true."""
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return False


def _row_requires_checkpoint_provenance(row: pd.Series) -> bool:
    """Return whether an exported evaluation row should carry checkpoint provenance."""
    if row.get("phase") not in EVAL_PHASES:
        return False
    if str(row.get("paradigm", "")).lower() == "xgboost":
        return False
    return True


def _checkpoint_provenance_issues(per_seed_df: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """Find evaluation rows whose logged test metrics lack reproducible checkpoint metadata."""
    if per_seed_df.empty:
        return []

    issues = []
    for _, row in per_seed_df.iterrows():
        if not _row_requires_checkpoint_provenance(row):
            continue

        source = row.get("_eval_checkpoint_source")
        if _is_missing_export_value(source):
            issues.append((row, "missing _eval_checkpoint_source"))
            continue

        if source == "failed":
            error = row.get("_best_ckpt_error")
            reason = "recorded checkpoint-selection failure"
            if not _is_missing_export_value(error):
                reason += f": {error}"
            issues.append((row, reason))
            continue

        if source not in {"best", "final"}:
            issues.append((row, f"unrecognized _eval_checkpoint_source={source!r}"))
            continue

        if source == "best":
            if _is_missing_export_value(row.get("_best_ckpt_path")):
                issues.append((row, "best-checkpoint evaluation missing _best_ckpt_path"))
                continue
            if not _is_true_export_value(row.get("_best_ckpt_load_ok")):
                issues.append((row, "best-checkpoint evaluation did not record load success"))

    return issues


def _fairness_attributes_for_row(row: pd.Series) -> list[str]:
    """Return protected attributes expected in publication exports for a row."""
    dataset = str(row.get("dataset", "")).lower()
    attrs = list(FAIRNESS_ATTRIBUTES)
    if dataset == "eicu":
        attrs = [attr for attr in attrs if attr != "race"]
    return attrs


def _fairness_required_metrics_for_row(row: pd.Series) -> list[str]:
    """Return required aggregate fairness metrics for the row's task family."""
    task = str(row.get("task", "") or "").lower()
    if task.startswith("los"):
        return REGRESSION_FAIRNESS_REQUIRED_METRICS
    return BINARY_FAIRNESS_REQUIRED_METRICS


def _fairness_present_key_set(row: pd.Series) -> set[str] | None:
    """Return W&B fairness summary keys known to have existed, if available."""
    value = row.get(FAIRNESS_SUMMARY_KEY_COLUMN)
    if _is_missing_export_value(value):
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    elif isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        return None
    return {str(key) for key in parsed}


def _row_has_fairness_metric(row: pd.Series, key: str) -> bool:
    """Check whether a required fairness metric was written for this row."""
    present_keys = _fairness_present_key_set(row)
    if present_keys is not None:
        return key in present_keys
    if key not in row.index:
        return False
    return not _is_missing_export_value(row.get(key))


def _fairness_completeness_issues(per_seed_df: pd.DataFrame) -> list[tuple[pd.Series, list[str]]]:
    """Find evaluation rows missing dataset/task-appropriate fairness summaries."""
    if per_seed_df.empty:
        return []

    issues = []
    for _, row in per_seed_df.iterrows():
        if row.get("phase") not in EVAL_PHASES:
            continue
        if _is_missing_export_value(row.get("task")):
            continue

        missing_keys = []
        for attr in _fairness_attributes_for_row(row):
            for metric_name in _fairness_required_metrics_for_row(row):
                key = f"fairness/{attr}/{metric_name}"
                if not _row_has_fairness_metric(row, key):
                    missing_keys.append(key)

        if missing_keys:
            issues.append((row, missing_keys))

    return issues


def _expected_fairness_checkpoint_source_for_row(row: pd.Series) -> str | None:
    """Return checkpoint provenance expected by fairness metadata for a row."""
    if str(row.get("paradigm", "")).lower() == "xgboost":
        return "xgboost_model"

    source = row.get("_eval_checkpoint_source")
    return str(source) if source in {"best", "final"} else None


def _expected_fairness_artifact_source_for_row(row: pd.Series) -> str | None:
    """Return evaluated artifact source expected by fairness metadata for a row."""
    if str(row.get("paradigm", "")).lower() == "xgboost":
        return "xgboost_model"

    source = row.get("_eval_checkpoint_source")
    if source == "best":
        return "recorded_best"
    if source == "final":
        return "recorded_final"
    return None


def _expected_fairness_artifact_id_for_row(row: pd.Series) -> str | None:
    """Return expected artifact identity when export metadata makes it knowable."""
    output_dir = row.get("_output_dir")
    paradigm = str(row.get("paradigm", "")).lower()

    if paradigm == "xgboost":
        if _is_missing_export_value(output_dir):
            return None
        return canonical_artifact_id(Path(str(output_dir)) / "xgboost_model.json")

    source = row.get("_eval_checkpoint_source")
    if source == "best":
        best_path = row.get("_best_ckpt_path")
        if _is_missing_export_value(best_path):
            return None
        return canonical_artifact_id(best_path)
    if source == "final" and not _is_missing_export_value(output_dir):
        return canonical_artifact_id(Path(str(output_dir)) / "checkpoints" / "last.ckpt")
    return None


def _coerce_int_metadata(value) -> int | None:
    if _is_missing_export_value(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fairness_metadata_staleness_issues(
    per_seed_df: pd.DataFrame,
) -> list[tuple[pd.Series, list[str]]]:
    """Find evaluation rows whose fairness summaries lack current provenance metadata."""
    if per_seed_df.empty:
        return []

    issues = []
    for _, row in per_seed_df.iterrows():
        if row.get("phase") not in EVAL_PHASES:
            continue
        if _is_missing_export_value(row.get("task")):
            continue

        row_issues: list[str] = []
        if row.get(FAIRNESS_SCHEMA_VERSION_KEY) != FAIRNESS_SUMMARY_SCHEMA_VERSION:
            row_issues.append("missing or stale fairness schema version")
        if row.get(FAIRNESS_SCRIPT_VERSION_KEY) != FAIRNESS_SCRIPT_VERSION:
            row_issues.append("missing or stale fairness script version")

        actual_attrs = decode_protected_attributes(row.get(FAIRNESS_PROTECTED_ATTRIBUTES_KEY))
        expected_attrs = normalize_protected_attributes(FAIRNESS_ATTRIBUTES)
        if actual_attrs != expected_attrs:
            row_issues.append(
                f"protected attributes mismatch: expected={expected_attrs}, actual={actual_attrs}"
            )

        actual_min_subgroup_size = _coerce_int_metadata(row.get(FAIRNESS_MIN_SUBGROUP_SIZE_KEY))
        if actual_min_subgroup_size != FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE:
            row_issues.append(
                "min subgroup size mismatch: "
                f"expected={FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE}, "
                f"actual={actual_min_subgroup_size}"
            )

        expected_artifact_source = _expected_fairness_artifact_source_for_row(row)
        actual_artifact_source = row.get(FAIRNESS_ARTIFACT_SOURCE_KEY)
        if (
            expected_artifact_source is not None
            and actual_artifact_source != expected_artifact_source
        ):
            row_issues.append(
                "artifact source mismatch: "
                f"expected={expected_artifact_source}, actual={actual_artifact_source}"
            )

        expected_checkpoint_source = _expected_fairness_checkpoint_source_for_row(row)
        actual_checkpoint_source = row.get(FAIRNESS_CHECKPOINT_SOURCE_KEY)
        if (
            expected_checkpoint_source is not None
            and actual_checkpoint_source != expected_checkpoint_source
        ):
            row_issues.append(
                "checkpoint source mismatch: "
                f"expected={expected_checkpoint_source}, actual={actual_checkpoint_source}"
            )

        actual_artifact_id = canonical_artifact_id(row.get(FAIRNESS_ARTIFACT_PATH_KEY))
        expected_artifact_id = _expected_fairness_artifact_id_for_row(row)
        if not actual_artifact_id:
            row_issues.append("missing fairness artifact path")
        elif expected_artifact_id is not None and actual_artifact_id != expected_artifact_id:
            row_issues.append(
                "artifact path mismatch: "
                f"expected={expected_artifact_id}, actual={actual_artifact_id}"
            )

        if row_issues:
            issues.append((row, row_issues))

    return issues


def _mixed_revision_group_warnings(per_seed_df: pd.DataFrame) -> list[str]:
    """Warn when one aggregated scientific config draws seeds from multiple revisions."""
    if per_seed_df.empty or "revision" not in per_seed_df.columns:
        return []

    fingerprint_cols = [col for col in FINGERPRINT if col in per_seed_df.columns]
    if not fingerprint_cols:
        return []

    warnings = []
    work = _fillna_for_grouping(per_seed_df, fingerprint_cols + ["revision"])
    for key, group in work.groupby(fingerprint_cols, dropna=False):
        revisions = sorted(
            revision
            for revision in group["revision"].dropna().astype(str).unique().tolist()
            if revision != "__none__"
        )
        if len(revisions) <= 1:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        desc = _format_matrix_key(key, fingerprint_cols)
        warnings.append(
            "WARNING: export group mixes multiple revisions; publication exports should "
            f"use one revision. {desc}; revisions={revisions}"
        )

    return warnings


def _export_text_or_none(value) -> str | None:
    if _is_missing_export_value(value):
        return None
    text = str(value).strip()
    return text or None


def _launch_commit_homogeneity_warnings(per_seed_df: pd.DataFrame) -> list[str]:
    """Warn when one revision is exported from missing or mixed launch commits."""
    if per_seed_df.empty or "revision" not in per_seed_df.columns:
        return []

    if "launch_commit" not in per_seed_df.columns:
        return [
            "WARNING: export rows lack launch_commit metadata; final revision provenance "
            "cannot be audited."
        ]

    warnings = []
    for revision_value, group in per_seed_df.groupby("revision", dropna=False):
        revision = _export_text_or_none(revision_value)
        if revision is None:
            warnings.append(
                f"WARNING: {len(group)} exported rows are missing revision metadata; "
                "launch-commit homogeneity cannot be scoped."
            )
            continue

        commits = sorted(
            {
                commit
                for commit in (_export_text_or_none(value) for value in group["launch_commit"])
                if commit is not None
            }
        )
        missing = int(group["launch_commit"].map(_export_text_or_none).isna().sum())
        if missing:
            warnings.append(
                f"WARNING: revision={revision} has {missing} runs missing launch_commit; "
                "final provenance cannot be audited."
            )
        if len(commits) > 1:
            warnings.append(
                f"WARNING: revision={revision} mixes multiple launch commits; " f"commits={commits}"
            )

    return warnings


def _expected_row_from_run(run) -> dict:
    return {
        "experiment_class": run.experiment_class,
        "experiment_type": run.experiment_class,
        "experiment_subtype": run.experiment_subtype,
        "paradigm": run.paradigm,
        "dataset": run.dataset,
        "task": run.task,
        "seed": run.seed,
        "protocol": run.protocol,
        "label_fraction": run.label_fraction,
        "model_size": run.model_size or "default",
        "source_dataset": run.source_dataset,
        "phase": run.phase,
        "upstream_pretrain_lr": run.upstream_pretrain_lr,
        "upstream_pretrain_mask_ratio": run.upstream_pretrain_mask_ratio,
    }


def build_expected_matrix_df(
    experiment_class: list[str] | None = None,
    paradigm: list[str] | None = None,
    dataset: list[str] | None = None,
    phase: list[str] | None = None,
) -> pd.DataFrame:
    """Build the expected per-seed evaluation matrix for the selected export scope."""
    from scripts.internal.run_experiments import generate_all_runs

    experiment_class_set = set(experiment_class) if experiment_class else None
    paradigm_set = set(paradigm) if paradigm else None
    dataset_set = set(dataset) if dataset else None
    phase_set = set(phase) if phase else set(EVAL_PHASES)

    rows = []
    for run in generate_all_runs():
        if run.phase not in EVAL_PHASES:
            continue
        row = _expected_row_from_run(run)
        if experiment_class_set and row["experiment_class"] not in experiment_class_set:
            continue
        if paradigm_set and row["paradigm"] not in paradigm_set:
            continue
        if dataset_set and row["dataset"] not in dataset_set:
            continue
        if phase_set and row["phase"] not in phase_set:
            continue
        rows.append(row)

    return pd.DataFrame(rows)


def _matrix_key_value(column: str, value):
    if _is_missing_export_value(value):
        return "__none__"
    if column == "seed":
        return int(value)
    if column in {"label_fraction", "upstream_pretrain_lr", "upstream_pretrain_mask_ratio"}:
        return round(float(value), 12)
    return str(value)


def _matrix_keys(df: pd.DataFrame, columns: list[str]) -> set[tuple]:
    if df.empty:
        return set()
    keys = set()
    for _, row in df.iterrows():
        keys.add(tuple(_matrix_key_value(col, row.get(col)) for col in columns))
    return keys


def _format_matrix_key(key: tuple, columns: list[str]) -> str:
    return ", ".join(f"{col}={value}" for col, value in zip(columns, key, strict=True))


def _matrix_coverage_warnings(
    per_seed_df: pd.DataFrame,
    expected_matrix_df: pd.DataFrame | None,
) -> list[str]:
    if expected_matrix_df is None or expected_matrix_df.empty:
        return []

    key_columns = [*FINGERPRINT, "seed"]
    missing_columns = [col for col in key_columns if col not in per_seed_df.columns]
    if missing_columns:
        return [
            "WARNING: export is missing matrix key columns needed for coverage validation: "
            + ", ".join(missing_columns)
        ]

    expected_keys = _matrix_keys(expected_matrix_df, key_columns)
    observed_keys = _matrix_keys(per_seed_df, key_columns)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)

    warnings = []
    if missing:
        warnings.append(
            f"WARNING: {len(missing)}/{len(expected_keys)} expected matrix evaluation rows "
            "are absent from export:"
        )
        for key in missing[:20]:
            warnings.append(f"  missing {_format_matrix_key(key, key_columns)}")
        if len(missing) > 20:
            warnings.append(f"  ... {len(missing) - 20} more missing rows")

    if unexpected:
        warnings.append(
            f"WARNING: {len(unexpected)} exported evaluation rows are outside the expected matrix:"
        )
        for key in unexpected[:20]:
            warnings.append(f"  unexpected {_format_matrix_key(key, key_columns)}")
        if len(unexpected) > 20:
            warnings.append(f"  ... {len(unexpected) - 20} more unexpected rows")

    return warnings


def validate(
    per_seed_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    statistical_df: pd.DataFrame | None = None,
    expected_seeds: set[int] | None = None,
    expected_matrix_df: pd.DataFrame | None = None,
) -> list[str]:
    warnings = []
    expected_seeds = set(EXPECTED_FIXED_SEEDS if expected_seeds is None else expected_seeds)

    warnings.extend(_matrix_coverage_warnings(per_seed_df, expected_matrix_df))

    if "experiment_class" in aggregated_df.columns:
        fixed_seed = aggregated_df[
            aggregated_df["experiment_class"].isin(FIXED_SEED_EXPERIMENT_CLASSES)
        ]
        bad_seed_rows = []
        for _, row in fixed_seed.iterrows():
            seeds = set(json.loads(row["seed_list"])) if pd.notna(row.get("seed_list")) else set()
            if seeds != expected_seeds:
                bad_seed_rows.append((row, seeds))
        if bad_seed_rows:
            warnings.append(
                f"WARNING: {len(bad_seed_rows)}/{len(fixed_seed)} fixed-seed configs do not "
                f"have the expected seed set {sorted(expected_seeds)}:"
            )
            for row, seeds in bad_seed_rows:
                desc = ", ".join(
                    f"{col}={row[col]}"
                    for col in ["paradigm", "dataset", "task", "protocol"]
                    if col in row and row[col] is not None
                )
                warnings.append(
                    f"  experiment_class={row['experiment_class']}, {desc} - "
                    f"seeds={sorted(seeds)}, missing={sorted(expected_seeds - seeds)}, "
                    f"unexpected={sorted(seeds - expected_seeds)}"
                )

    if statistical_df is not None and not statistical_df.empty:
        coverage_cols = {"n_shared_task_seed_pairs", "n_union_task_seed_pairs"}
        if coverage_cols.issubset(statistical_df.columns):
            incomplete = statistical_df[
                statistical_df["n_shared_task_seed_pairs"]
                < statistical_df["n_union_task_seed_pairs"]
            ]
            if len(incomplete) > 0:
                warnings.append(
                    f"WARNING: {len(incomplete)}/{len(statistical_df)} statistical comparisons "
                    "have incomplete paired task/seed coverage:"
                )

    test_cols = [col for col in PERFORMANCE_TEST_METRICS if col in per_seed_df.columns]
    if test_cols:
        all_nan = per_seed_df[test_cols].isna().all(axis=1)
        if all_nan.any():
            warnings.append(
                f"WARNING: {all_nan.sum()} runs have no test metrics at all. "
                "These may be pretraining runs or crashed evaluations."
            )

    if not per_seed_df.empty and "task" in per_seed_df.columns:
        missing_primary = []
        for _, row in per_seed_df.iterrows():
            if row.get("phase") not in EVAL_PHASES:
                continue
            primary_metric = _primary_metric_for_task(row.get("task"))
            if primary_metric is None or primary_metric not in per_seed_df.columns:
                continue
            if pd.isna(row.get(primary_metric)):
                missing_primary.append((row, primary_metric))
        if missing_primary:
            warnings.append(
                f"WARNING: {len(missing_primary)} evaluation runs are missing their "
                "primary test metric."
            )

    checkpoint_issues = _checkpoint_provenance_issues(per_seed_df)
    if checkpoint_issues:
        warnings.append(
            f"WARNING: {len(checkpoint_issues)} evaluation runs have missing or failed "
            "checkpoint provenance. Publication exports fail closed by default; use "
            "--allow-incomplete only for exploratory exports."
        )
        for row, reason in checkpoint_issues[:10]:
            warnings.append(
                "  "
                + ", ".join(
                    f"{col}={row.get(col)}"
                    for col in [
                        "wandb_run_id",
                        "paradigm",
                        "dataset",
                        "task",
                        "seed",
                        "phase",
                    ]
                    if col in row
                )
                + f" - {reason}"
            )

    fairness_issues = _fairness_completeness_issues(per_seed_df)
    if fairness_issues:
        warnings.append(
            f"WARNING: {len(fairness_issues)} evaluation runs are missing required "
            "fairness summary metrics. Run scripts/eval/evaluate_fairness.py before "
            "publication export; use --allow-incomplete only for exploratory exports."
        )
        for row, missing_keys in fairness_issues[:10]:
            preview = ", ".join(missing_keys[:8])
            if len(missing_keys) > 8:
                preview += f", ... {len(missing_keys) - 8} more"
            warnings.append(
                "  "
                + ", ".join(
                    f"{col}={row.get(col)}"
                    for col in [
                        "wandb_run_id",
                        "paradigm",
                        "dataset",
                        "task",
                        "seed",
                        "phase",
                    ]
                    if col in row
                )
                + f" - missing={preview}"
            )

    fairness_metadata_issues = _fairness_metadata_staleness_issues(per_seed_df)
    if fairness_metadata_issues:
        warnings.append(
            f"WARNING: {len(fairness_metadata_issues)} evaluation runs have missing or "
            "stale fairness summary metadata. Re-run scripts/eval/evaluate_fairness.py "
            "with the publication defaults before export; use --allow-incomplete only "
            "for exploratory exports."
        )
        for row, row_issues in fairness_metadata_issues[:10]:
            preview = "; ".join(row_issues[:6])
            if len(row_issues) > 6:
                preview += f"; ... {len(row_issues) - 6} more"
            warnings.append(
                "  "
                + ", ".join(
                    f"{col}={row.get(col)}"
                    for col in [
                        "wandb_run_id",
                        "paradigm",
                        "dataset",
                        "task",
                        "seed",
                        "phase",
                    ]
                    if col in row
                )
                + f" - {preview}"
            )

    warnings.extend(_mixed_revision_group_warnings(per_seed_df))
    warnings.extend(_launch_commit_homogeneity_warnings(per_seed_df))

    print("\nValidation summary:", file=sys.stderr)
    print(f"  Total runs: {len(per_seed_df)}", file=sys.stderr)
    print(f"  Unique configs: {len(aggregated_df)}", file=sys.stderr)
    if "experiment_class" in aggregated_df.columns:
        for experiment_class, group in aggregated_df.groupby("experiment_class"):
            seed_dist = dict(group["n_seeds"].value_counts().sort_index())
            print(
                f"  {experiment_class}: {len(group)} configs, seeds: {seed_dist}", file=sys.stderr
            )
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
        default=os.environ.get("WANDB_PROJECT", "slices-thesis"),
        help="W&B project name (default: WANDB_PROJECT env var or 'slices-thesis')",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity name (default: WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--experiment-class",
        nargs="+",
        choices=EXPERIMENT_CLASSES,
        help="Filter to specific experiment class(es)",
    )
    parser.add_argument("--paradigm", nargs="+", help="Filter to specific paradigm(s)")
    parser.add_argument("--dataset", nargs="+", help="Filter to specific dataset(s)")
    parser.add_argument(
        "--phase",
        nargs="+",
        default=EVAL_PHASES,
        help=f"Filter to specific phase(s) (default: {EVAL_PHASES})",
    )
    parser.add_argument("--revision", nargs="+", help="Filter to specific revision tag(s)")
    parser.add_argument(
        "--allow-multiple-revisions",
        action="store_true",
        help=(
            "Permit exporting more than one revision in a single exploratory run. "
            "Publication exports should use exactly one revision."
        ),
    )
    parser.add_argument("--state", default="finished", help="Run state filter (default: finished)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        type=int,
        default=sorted(EXPECTED_FIXED_SEEDS),
        help=(
            "Expected exact seed set for fixed-seed configs "
            f"(default: {sorted(EXPECTED_FIXED_SEEDS)})"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully even if no runs match or validation warnings are emitted.",
    )
    parser.add_argument(
        "--allow-extraction-failures",
        action="store_true",
        help=(
            "Skip W&B runs that fail row extraction. Use only for exploratory exports; "
            "publication exports fail closed by default."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-fingerprints",
        action="store_true",
        help=(
            "Keep the newest duplicate exact scientific fingerprint. Use only for "
            "exploratory cleanup exports; publication exports fail closed by default."
        ),
    )
    parser.add_argument(
        "--include-extension-tasks",
        action="store_true",
        help=(
            "Include task rows outside the fixed thesis task set. By default, "
            f"primary thesis exports are restricted to {sorted(THESIS_TASKS)}."
        ),
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
    if len(args.revision) > 1 and not args.allow_multiple_revisions:
        parser.error(
            "Multiple --revision values are disabled for publication exports. "
            "Run one revision at a time, or pass --allow-multiple-revisions for exploratory export."
        )

    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = fetch_all_runs(
        project=args.project,
        entity=args.entity,
        state=args.state,
        experiment_class=args.experiment_class,
        paradigm=args.paradigm,
        dataset=args.dataset,
        phase=args.phase,
        revision=args.revision,
    )
    if not runs:
        print("No runs found matching filters. Exiting.", file=sys.stderr)
        sys.exit(0 if args.allow_incomplete else 1)

    print(f"\nBuilding per-seed DataFrame from {len(runs)} runs...", file=sys.stderr)
    per_seed_df = build_per_seed_df(
        runs,
        allow_extraction_failures=args.allow_extraction_failures,
        allow_duplicate_fingerprints=args.allow_duplicate_fingerprints,
    )
    per_seed_df = filter_thesis_tasks(
        per_seed_df,
        include_extension_tasks=args.include_extension_tasks,
    )
    print(f"  Shape: {per_seed_df.shape}", file=sys.stderr)

    if per_seed_df.empty:
        print("No rows remain after task filtering. Exiting.", file=sys.stderr)
        sys.exit(0 if args.allow_incomplete else 1)

    print("\nBuilding aggregated DataFrame...", file=sys.stderr)
    aggregated_df = build_aggregated_df(per_seed_df)
    print(f"  Shape: {aggregated_df.shape}", file=sys.stderr)

    print("\nBuilding derived comparison views...", file=sys.stderr)
    label_efficiency_df = build_label_efficiency_curve_df(per_seed_df)
    capacity_df = build_capacity_study_comparison_df(per_seed_df)
    classical_context_df = build_classical_context_df(per_seed_df)

    print("\nBuilding statistical significance table...", file=sys.stderr)
    statistical_df = build_statistical_tests_df(per_seed_df)
    print(f"  Shape: {statistical_df.shape}", file=sys.stderr)

    print("\nBuilding TS2Vec vs core contrastive table...", file=sys.stderr)
    ts2vec_contrastive_df = build_ts2vec_vs_core_contrastive_df(per_seed_df)
    print(f"  Shape: {ts2vec_contrastive_df.shape}", file=sys.stderr)

    expected_matrix_df = build_expected_matrix_df(
        experiment_class=args.experiment_class,
        paradigm=args.paradigm,
        dataset=args.dataset,
        phase=args.phase,
    )

    coverage_parts = [df for df in [statistical_df, ts2vec_contrastive_df] if not df.empty]
    statistical_coverage_df = (
        pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    )
    warnings = validate(
        per_seed_df,
        aggregated_df,
        statistical_df=statistical_coverage_df,
        expected_seeds=set(args.expected_seeds),
        expected_matrix_df=expected_matrix_df,
    )
    for warning in warnings:
        print(warning, file=sys.stderr)

    if warnings and not args.allow_incomplete:
        print("Validation failed; no parquet outputs were written.", file=sys.stderr)
        sys.exit(1)

    outputs = {
        "per_seed_results.parquet": per_seed_df,
        "aggregated_results.parquet": aggregated_df,
        "statistical_tests.parquet": statistical_df,
        "label_efficiency_curves.parquet": label_efficiency_df,
        "capacity_study_comparison.parquet": capacity_df,
        "classical_context.parquet": classical_context_df,
        "ts2vec_vs_core_contrastive.parquet": ts2vec_contrastive_df,
    }
    for name, df in outputs.items():
        path = output_dir / name
        df.to_parquet(path, index=False)
        print(f"  {path} ({len(df)} rows)", file=sys.stderr)

    print("\n--- Quick Summary ---")
    print(
        f"Runs: {len(per_seed_df)}, Configs: {len(aggregated_df)}, "
        f"StatTests: {len(statistical_df)}, "
        f"TS2VecContrastiveTests: {len(ts2vec_contrastive_df)}"
    )
    if "experiment_class" in aggregated_df.columns:
        for experiment_class, group in aggregated_df.groupby("experiment_class"):
            seed_dist = dict(group["n_seeds"].value_counts().sort_index())
            print(f"  {experiment_class}: {len(group)} configs, seeds: {seed_dist}")


if __name__ == "__main__":
    main()
