#!/usr/bin/env python3
"""Post-run fairness evaluation for SLICES experiment runs.

Queries W&B for finished downstream runs, reconstructs the exact evaluation
artifact used for each run, runs inference on the test set, computes fairness
metrics via FairnessEvaluator, and writes results back to the same W&B run's
summary.

Designed for batch evaluation of the benchmark fairness corpus across finetune,
supervised, and classical baseline runs. Supports resumability via
--skip-existing (default) and scoping via --experiment-class/--paradigm/--dataset filters.

Usage:
    # Evaluate the benchmark fairness corpus for one explicit revision
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity>

    # Scope to specific experiment class/dataset
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> \
        --experiment-class core_ssl_benchmark --dataset miiv

    # Preview which runs would be evaluated
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> --dry-run

    # Override paths (e.g., different machine than training)
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> \
        --outputs-root /mnt/data/outputs --data-root /mnt/data

    # Recompute fairness for runs that already have metrics
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> --force

    # Debug with a single run
    uv run python scripts/eval/evaluate_fairness.py \
        --project slices-thesis --revision thesis-v1 --entity <entity> --max-runs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from slices.eval.fairness_evaluator import flatten_fairness_report
from slices.eval.fairness_metadata import (
    FAIRNESS_ARTIFACT_PATH_KEY,
    FAIRNESS_ARTIFACT_SOURCE_KEY,
    FAIRNESS_CHECKPOINT_SOURCE_KEY,
    FAIRNESS_CLEAR_PREFIXES,
    FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE,
    FAIRNESS_DEFAULT_PROTECTED_ATTRIBUTES,
    FAIRNESS_MIN_SUBGROUP_SIZE_KEY,
    FAIRNESS_PROTECTED_ATTRIBUTES_KEY,
    FAIRNESS_SCHEMA_VERSION_KEY,
    FAIRNESS_SCRIPT_VERSION,
    FAIRNESS_SCRIPT_VERSION_KEY,
    FAIRNESS_SUMMARY_SCHEMA_VERSION,
    canonical_artifact_id,
    decode_protected_attributes,
    encode_protected_attributes,
    normalize_protected_attributes,
)

log = logging.getLogger("evaluate_fairness")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_CLASSES = [
    "core_ssl_benchmark",
    "label_efficiency",
    "cross_dataset_transfer",
    "hp_robustness",
    "capacity_study",
    "classical_baselines",
    "ts2vec_extension",
    "smart_external_reference",
]
DEFAULT_PHASES = ["finetune", "supervised", "baseline"]
DEFAULT_PROTECTED_ATTRIBUTES = FAIRNESS_DEFAULT_PROTECTED_ATTRIBUTES
THESIS_TASKS = {
    "mortality_24h",
    "mortality_hospital",
    "aki_kdigo",
    "los_remaining",
}
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
# W&B helpers
# ---------------------------------------------------------------------------


def _retry(fn, max_retries=3, base_delay=5):
    """Retry with exponential backoff on transient errors."""
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
            log.warning("Retry %d/%d after %ds: %r", attempt + 1, max_retries, delay, e)
            time.sleep(delay)


def _run_config_filter_value(run, key: str):
    config = dict(getattr(run, "config", {}) or {})
    if key == "paradigm":
        ssl_cfg = config.get("ssl") or {}
        return config.get("paradigm") or ssl_cfg.get("name")
    return config.get(key)


def _run_matches_filter(run, key: str, tag_prefix: str, values: Optional[list[str]]) -> bool:
    """Match a W&B run by config first, falling back to tags."""
    if not values:
        return True

    allowed = {str(value) for value in values}
    config_value = _run_config_filter_value(run, key)
    if config_value is not None:
        return str(config_value) in allowed

    tags = set(getattr(run, "tags", []) or [])
    return any(f"{tag_prefix}:{value}" in tags for value in allowed)


def fetch_eval_runs(
    project: str,
    entity: Optional[str],
    experiment_classes: Optional[list[str]],
    paradigms: Optional[list[str]],
    datasets: Optional[list[str]],
    phases: list[str],
    revisions: Optional[list[str]],
) -> list:
    """Fetch finished evaluation runs from W&B matching filters."""
    import wandb

    api = wandb.Api(timeout=300)
    path = f"{entity}/{project}" if entity else project

    filters: dict = {"state": "finished"}

    log.info("Fetching runs from %s with filters: %s", path, json.dumps(filters, default=str))
    runs_iter = api.runs(path, filters=filters or {}, order="-created_at")

    runs = []
    for run in runs_iter:
        if not _run_matches_filter(run, "experiment_class", "experiment_class", experiment_classes):
            continue
        if not _run_matches_filter(run, "paradigm", "paradigm", paradigms):
            continue
        if not _run_matches_filter(run, "dataset", "dataset", datasets):
            continue
        if not _run_matches_filter(run, "phase", "phase", phases):
            continue
        if not _run_matches_filter(run, "revision", "revision", revisions):
            continue

        runs.append(run)

    log.info("Fetched %d runs.", len(runs))
    return runs


def _expected_fairness_attributes(run, protected_attributes: list[str]) -> list[str]:
    """Return requested fairness attributes that are meaningful for the run dataset."""
    dataset = str(run.config.get("dataset", "")).lower()
    attrs = list(dict.fromkeys(protected_attributes))
    if dataset == "eicu":
        attrs = [attr for attr in attrs if attr != "race"]
    return attrs


def _has_summary_value(summary: dict[str, Any], key: str) -> bool:
    """Return true when a fairness summary key was written, even if undefined."""
    return key in summary and summary.get(key) is not None


def _task_type_for_run(run) -> str:
    """Infer fairness metric family from W&B config."""
    config = run.config or {}
    task_type = _get_nested(config, "task.task_type")
    if task_type:
        return str(task_type).lower()

    task_name = _get_nested(config, "task.task_name", "")
    if str(task_name).startswith("los"):
        return "regression"
    return "binary"


def _required_fairness_metrics_for_run(run) -> list[str]:
    """Return required aggregate fairness metrics for this run's task type."""
    if _task_type_for_run(run) == "regression":
        return REGRESSION_FAIRNESS_REQUIRED_METRICS
    return BINARY_FAIRNESS_REQUIRED_METRICS


def _expected_fairness_artifact_source(run) -> str | None:
    """Return the artifact-source metadata expected for a run."""
    paradigm = str((run.config or {}).get("paradigm", "")).lower()
    if paradigm == "xgboost":
        return "xgboost_model"

    eval_source = (run.summary_metrics or {}).get("_eval_checkpoint_source")
    if eval_source == "best":
        return "recorded_best"
    if eval_source == "final":
        return "recorded_final"
    return None


def _expected_fairness_checkpoint_source(run) -> str | None:
    """Return checkpoint-provenance metadata expected for a run."""
    paradigm = str((run.config or {}).get("paradigm", "")).lower()
    if paradigm == "xgboost":
        return "xgboost_model"
    source = (run.summary_metrics or {}).get("_eval_checkpoint_source")
    return str(source) if source in {"best", "final"} else None


def _expected_fairness_artifact_id(run) -> str | None:
    """Return expected evaluated-artifact identity when it can be inferred."""
    config = run.config or {}
    summary = run.summary_metrics or {}
    output_dir = config.get("output_dir", "")
    paradigm = str(config.get("paradigm", "")).lower()

    if paradigm == "xgboost":
        if not output_dir:
            return None
        return canonical_artifact_id(Path(output_dir) / "xgboost_model.json")

    eval_source = summary.get("_eval_checkpoint_source")
    if eval_source == "best":
        best_path = summary.get("_best_ckpt_path")
        return canonical_artifact_id(best_path) if best_path else None
    if eval_source == "final" and output_dir:
        return canonical_artifact_id(Path(output_dir) / "checkpoints" / "last.ckpt")
    return None


def build_fairness_summary_metadata(
    run,
    artifact_path: Path,
    artifact_source: str,
    protected_attributes: list[str],
    min_subgroup_size: int,
) -> dict[str, Any]:
    """Build versioned metadata that makes fairness summaries auditable."""
    checkpoint_source = _expected_fairness_checkpoint_source(run) or artifact_source
    return {
        FAIRNESS_SCHEMA_VERSION_KEY: FAIRNESS_SUMMARY_SCHEMA_VERSION,
        FAIRNESS_SCRIPT_VERSION_KEY: FAIRNESS_SCRIPT_VERSION,
        FAIRNESS_ARTIFACT_PATH_KEY: str(artifact_path),
        FAIRNESS_ARTIFACT_SOURCE_KEY: artifact_source,
        FAIRNESS_CHECKPOINT_SOURCE_KEY: checkpoint_source,
        FAIRNESS_PROTECTED_ATTRIBUTES_KEY: encode_protected_attributes(protected_attributes),
        FAIRNESS_MIN_SUBGROUP_SIZE_KEY: int(min_subgroup_size),
    }


def fairness_summary_metadata_issues(
    run,
    protected_attributes: list[str],
    min_subgroup_size: int = FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE,
) -> list[str]:
    """Return freshness problems for existing fairness summary metadata."""
    summary = run.summary_metrics or {}
    issues: list[str] = []

    if summary.get(FAIRNESS_SCHEMA_VERSION_KEY) != FAIRNESS_SUMMARY_SCHEMA_VERSION:
        issues.append("missing or stale fairness schema version")
    if summary.get(FAIRNESS_SCRIPT_VERSION_KEY) != FAIRNESS_SCRIPT_VERSION:
        issues.append("missing or stale fairness script version")

    actual_attrs = decode_protected_attributes(summary.get(FAIRNESS_PROTECTED_ATTRIBUTES_KEY))
    expected_attrs = normalize_protected_attributes(protected_attributes)
    if actual_attrs != expected_attrs:
        issues.append(
            f"protected attributes mismatch: expected={expected_attrs}, actual={actual_attrs}"
        )

    try:
        actual_min_subgroup_size = int(float(summary.get(FAIRNESS_MIN_SUBGROUP_SIZE_KEY)))
    except (TypeError, ValueError):
        actual_min_subgroup_size = None
    if actual_min_subgroup_size != int(min_subgroup_size):
        issues.append(
            "min subgroup size mismatch: "
            f"expected={int(min_subgroup_size)}, actual={actual_min_subgroup_size}"
        )

    expected_artifact_source = _expected_fairness_artifact_source(run)
    actual_artifact_source = summary.get(FAIRNESS_ARTIFACT_SOURCE_KEY)
    if expected_artifact_source is not None and actual_artifact_source != expected_artifact_source:
        issues.append(
            "artifact source mismatch: "
            f"expected={expected_artifact_source}, actual={actual_artifact_source}"
        )

    expected_checkpoint_source = _expected_fairness_checkpoint_source(run)
    actual_checkpoint_source = summary.get(FAIRNESS_CHECKPOINT_SOURCE_KEY)
    if (
        expected_checkpoint_source is not None
        and actual_checkpoint_source != expected_checkpoint_source
    ):
        issues.append(
            "checkpoint source mismatch: "
            f"expected={expected_checkpoint_source}, actual={actual_checkpoint_source}"
        )

    actual_artifact_id = canonical_artifact_id(summary.get(FAIRNESS_ARTIFACT_PATH_KEY))
    expected_artifact_id = _expected_fairness_artifact_id(run)
    if not actual_artifact_id:
        issues.append("missing fairness artifact path")
    elif expected_artifact_id is not None and actual_artifact_id != expected_artifact_id:
        issues.append(
            "artifact path mismatch: "
            f"expected={expected_artifact_id}, actual={actual_artifact_id}"
        )

    return issues


def has_fairness_metrics(
    run,
    protected_attributes: list[str],
    min_subgroup_size: int = FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE,
) -> bool:
    """Check whether all requested dataset-appropriate fairness keys exist.

    Uses ``summary_metrics`` (populated from the batch query) instead of
    ``summary._json_dict`` which triggers a per-run GraphQL reload. Existing
    summaries must also carry current metadata so stale fairness outputs are
    recomputed instead of skipped.
    """
    try:
        sm = run.summary_metrics or {}
        expected_attrs = _expected_fairness_attributes(run, protected_attributes)
        if not expected_attrs:
            return False
        required_metrics = _required_fairness_metrics_for_run(run)
        for attr in expected_attrs:
            prefix = f"fairness/{attr}/"
            for metric_name in required_metrics:
                if not _has_summary_value(sm, f"{prefix}{metric_name}"):
                    return False
        return not fairness_summary_metadata_issues(run, protected_attributes, min_subgroup_size)
    except Exception:
        return False


def missing_fairness_report_requirements(
    run,
    report: dict[str, Any],
    protected_attributes: list[str],
) -> list[str]:
    """Return missing requested fairness attributes or required aggregate metrics."""
    expected_attrs = _expected_fairness_attributes(run, protected_attributes)
    required_metrics = _required_fairness_metrics_for_run(run)
    missing: list[str] = []

    for attr in expected_attrs:
        if attr not in report:
            missing.append(f"{attr}: no valid fairness groups")
            continue

        flat = flatten_fairness_report({attr: report[attr]})
        prefix = f"fairness/{attr}/"
        for metric_name in required_metrics:
            key = f"{prefix}{metric_name}"
            if not _has_summary_value(flat, key):
                missing.append(f"{attr}: missing {metric_name}")

    return missing


def write_fairness_to_wandb(
    run_path: str,
    fairness_flat: dict[str, Any],
    dry_run: bool = False,
    clear_existing: bool = False,
) -> None:
    """Write fairness metrics to W&B run summary."""
    import wandb

    if dry_run:
        mode = "replace" if clear_existing else "write"
        log.info(
            "  [DRY RUN] Would %s %d fairness keys on %s",
            mode,
            len(fairness_flat),
            run_path,
        )
        return

    def _do_update():
        api = wandb.Api(timeout=120)
        run = api.run(run_path)
        if clear_existing:
            try:
                existing_summary = dict(run.summary)
            except (TypeError, ValueError):
                existing_summary = dict(getattr(run.summary, "_json_dict", {}) or {})
            for key in existing_summary:
                if key.startswith(FAIRNESS_CLEAR_PREFIXES) and key not in fairness_flat:
                    run.summary[key] = None
        run.summary.update(fairness_flat)
        run.summary.save()

    _retry(_do_update)
    log.info("  Wrote %d fairness keys to W&B run %s", len(fairness_flat), run_path)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _resolve_ckpt_dir(output_dir: str, outputs_root: Optional[str] = None) -> Path:
    """Resolve the checkpoint directory, applying outputs_root rebase if needed."""
    return _resolve_output_dir(output_dir, outputs_root) / "checkpoints"


def _resolve_output_dir(output_dir: str, outputs_root: Optional[str] = None) -> Path:
    """Resolve a run output directory, applying outputs_root rebase if needed."""
    if outputs_root:
        rel = output_dir
        if rel.startswith("outputs/"):
            rel = rel[len("outputs/") :]
        elif rel.startswith("/") and "/outputs/" in rel:
            rel = rel.split("/outputs/", 1)[1]
        return Path(outputs_root) / rel
    return Path(output_dir)


def _resolve_logged_checkpoint_path(
    checkpoint_path: str,
    outputs_root: Optional[str] = None,
) -> Path:
    """Resolve a checkpoint path recorded in W&B summary metadata."""
    path = Path(checkpoint_path)
    if outputs_root is None:
        return path

    checkpoint_str = str(checkpoint_path)
    if checkpoint_str.startswith("outputs/"):
        return Path(outputs_root) / checkpoint_str[len("outputs/") :]
    if checkpoint_str.startswith("/") and "/outputs/" in checkpoint_str:
        return Path(outputs_root) / checkpoint_str.split("/outputs/", 1)[1]
    if path.is_absolute():
        return path
    return Path(outputs_root) / checkpoint_str


def find_best_checkpoint(
    output_dir: str,
    outputs_root: Optional[str] = None,
    task_type: str = "binary",
) -> Optional[Path]:
    """Find the best .ckpt file in the run's checkpoint directory.

    Search strategy:
    1. Look for metric-named .ckpt files directly in checkpoints/ (standard case)
    2. Look for .ckpt files in subdirectories (Lightning creates subdirs when the
       monitor metric contains '/' — e.g. val/auprc becomes a directory separator)
    3. Fall back to last.ckpt

    Args:
        output_dir: Run output directory (from W&B config).
        outputs_root: Optional rebase root for checkpoint paths.
        task_type: Task type — determines whether higher or lower metric
            values are better. Classification monitors val/auprc (higher=better),
            regression monitors val/mse (lower=better).
    """
    ckpt_dir = _resolve_ckpt_dir(output_dir, outputs_root)

    if not ckpt_dir.exists():
        log.warning("  Checkpoint dir not found: %s", ckpt_dir)
        return None

    # Strategy 1: Direct .ckpt files (excluding last*.ckpt variants)
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if not p.name.startswith("last")]

    # Strategy 2: .ckpt files in subdirectories (Lightning '/' in metric name issue)
    if not ckpts:
        ckpts = list(ckpt_dir.glob("*/*.ckpt"))

    # Parse metric values from filenames if we found any candidates
    if ckpts:
        metric_pattern = re.compile(r"[-=](\d+\.\d+)\.ckpt$")
        scored = []
        for p in ckpts:
            match = metric_pattern.search(p.name)
            if match:
                scored.append((float(match.group(1)), p))

        if scored:
            # Regression: lower is better (val/mse); classification: higher (val/auprc)
            pick_lowest = task_type == "regression"
            scored.sort(key=lambda x: x[0], reverse=not pick_lowest)
            best = scored[0][1]
            log.info("  Best checkpoint: %s (metric=%.4f)", best.name, scored[0][0])
            return best

        # Non-last checkpoints exist but no parseable metric — pick most recent
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        log.info("  Using most recent checkpoint: %s", ckpts[0].name)
        return ckpts[0]

    # Strategy 3: Fall back to last.ckpt
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        log.info("  Using last.ckpt (no best checkpoint found)")
        return last

    log.warning("  No checkpoints found in %s", ckpt_dir)
    return None


def resolve_evaluation_checkpoint(
    run,
    outputs_root: Optional[str] = None,
    task_type: str = "binary",
) -> tuple[Optional[Path], str]:
    """Resolve the checkpoint that was actually used for logged test metrics.

    Uses the run's recorded evaluation provenance when present:
    - `_eval_checkpoint_source=best`  -> recorded `_best_ckpt_path`
    - `_eval_checkpoint_source=final` -> `last.ckpt`

    Fails closed for runs that lack recorded provenance entirely.
    """
    summary = dict(run.summary_metrics or {})
    output_dir = run.config.get("output_dir", "")
    eval_source = summary.get("_eval_checkpoint_source")

    if eval_source == "best":
        best_path = summary.get("_best_ckpt_path", "")
        if not best_path:
            raise FileNotFoundError(
                "Run recorded _eval_checkpoint_source=best but did not persist _best_ckpt_path."
            )
        ckpt_path = _resolve_logged_checkpoint_path(best_path, outputs_root)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Recorded best checkpoint not found: {ckpt_path} "
                f"(from summary path {best_path!r})"
            )
        return ckpt_path, "recorded_best"

    if eval_source == "final":
        ckpt_path = _resolve_ckpt_dir(output_dir, outputs_root) / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Run recorded final-model evaluation but last.ckpt was not found at {ckpt_path}"
            )
        return ckpt_path, "recorded_final"

    if eval_source == "failed":
        best_path = summary.get("_best_ckpt_path", "")
        error = summary.get("_best_ckpt_error", "unknown error")
        raise RuntimeError(
            "Training recorded a checkpoint-selection failure "
            f"(best_ckpt={best_path!r}, error={error!r})."
        )

    raise RuntimeError(
        "Run "
        f"{run.id} lacks recorded checkpoint provenance (_eval_checkpoint_source). "
        "Fairness evaluation now requires explicit provenance so it cannot silently "
        "re-evaluate a different checkpoint than the logged test metrics."
    )


def resolve_evaluation_artifact(
    run,
    outputs_root: Optional[str] = None,
    task_type: str = "binary",
) -> tuple[Path, str]:
    """Resolve the saved artifact needed to reproduce a run's test-time predictions."""
    paradigm = str(run.config.get("paradigm", "")).lower()
    if paradigm == "xgboost":
        model_path = _resolve_output_dir(run.config.get("output_dir", ""), outputs_root)
        model_path = model_path / "xgboost_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved XGBoost model not found: {model_path}")
        return model_path, "xgboost_model"

    ckpt_path, ckpt_source = resolve_evaluation_checkpoint(run, outputs_root, task_type)
    if ckpt_path is None:
        raise FileNotFoundError(f"Could not resolve evaluation checkpoint for run {run.id}")
    return ckpt_path, ckpt_source


# ---------------------------------------------------------------------------
# Model + data reconstruction
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


def _run_task_name(run) -> str | None:
    return _get_nested(run.config or {}, "task.task_name")


def filter_thesis_task_runs(runs: list) -> list:
    """Keep only fixed thesis tasks for publication fairness evaluation."""
    return [run for run in runs if _run_task_name(run) in THESIS_TASKS]


def build_datamodule(
    wandb_config: dict,
    batch_size: int = 64,
    data_root: Optional[str] = None,
):
    """Build ICUDataModule from W&B run config."""
    from slices.data.datamodule import ICUDataModule

    processed_dir = _get_nested(wandb_config, "data.processed_dir", "")
    if data_root:
        parts = Path(processed_dir).parts
        if len(parts) >= 2 and parts[-2] == "processed":
            processed_dir = str(Path(data_root) / parts[-2] / parts[-1])
        else:
            processed_dir = str(Path(data_root) / parts[-1])

    task_name = _get_nested(wandb_config, "task.task_name", "mortality_24h")
    seed = wandb_config.get("seed", 42)
    label_fraction = wandb_config.get("label_fraction", 1.0)

    datamodule = ICUDataModule(
        processed_dir=processed_dir,
        task_name=task_name,
        batch_size=batch_size,
        num_workers=4,
        seed=seed,
        label_fraction=label_fraction,
    )
    datamodule.setup()
    return datamodule


def build_model(wandb_config: dict, checkpoint_path: Path, datamodule):
    """Build FineTuneModule from W&B config and load checkpoint weights."""
    from omegaconf import OmegaConf
    from slices.training import FineTuneModule

    cfg = OmegaConf.create(wandb_config)
    OmegaConf.set_struct(cfg, False)
    cfg.encoder.d_input = datamodule.get_feature_dim()
    cfg.encoder.max_seq_length = datamodule.get_seq_length()

    class_weight = _get_nested(wandb_config, "training.class_weight")
    if class_weight == "balanced" or class_weight is None:
        cfg.training.class_weight = None
    OmegaConf.set_struct(cfg, True)

    model = FineTuneModule(
        config=cfg,
        checkpoint_path=None,
        pretrain_checkpoint_path=None,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Fairness evaluation
# ---------------------------------------------------------------------------


def evaluate_run_fairness(
    model: torch.nn.Module,
    datamodule,
    protected_attributes: list[str],
    min_subgroup_size: int,
    device: str,
) -> dict[str, Any]:
    """Run fairness evaluation on a single run."""
    from slices.eval.inference import run_inference

    model = model.to(device)
    predictions, labels, stay_ids = run_inference(
        model,
        datamodule.test_dataloader(),
        device=device,
    )

    task_type = getattr(model, "task_type", "binary")
    report = evaluate_predictions_fairness(
        predictions,
        labels,
        stay_ids,
        datamodule,
        protected_attributes,
        min_subgroup_size,
        task_type,
    )
    return report


def evaluate_predictions_fairness(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    stay_ids: list[int],
    datamodule,
    protected_attributes: list[str],
    min_subgroup_size: int,
    task_type: str,
) -> dict[str, Any]:
    """Run fairness evaluation from materialized predictions and labels."""
    from slices.eval.fairness_evaluator import FairnessEvaluator

    evaluator = FairnessEvaluator(
        static_df=datamodule.dataset.static_df,
        protected_attributes=protected_attributes,
        min_subgroup_size=min_subgroup_size,
        task_type=task_type,
        dataset_name=getattr(getattr(datamodule, "processed_dir", None), "name", None),
    )
    return evaluator.evaluate(predictions, labels, stay_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-run fairness evaluation for SLICES experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "slices-thesis"),
        help="W&B project name (default: $WANDB_PROJECT or 'slices-thesis')",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (default: $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--experiment-class",
        nargs="+",
        help="Filter to experiment class(es). Default: downstream-producing benchmark classes",
    )
    parser.add_argument("--paradigm", nargs="+", help="Filter to paradigm(s)")
    parser.add_argument("--dataset", nargs="+", help="Filter to dataset(s)")
    parser.add_argument(
        "--revision",
        nargs="+",
        help=(
            "Filter to revision tag(s). Required unless REVISION or "
            "WANDB_REVISION is set in the environment."
        ),
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        default=DEFAULT_PHASES,
        help=f"Filter to phase(s) (default: {DEFAULT_PHASES})",
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Override outputs directory root (rebase checkpoint paths)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data directory root (rebase data.processed_dir)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", default="auto", help="Device for inference (auto/cpu/cuda/mps)")
    parser.add_argument(
        "--protected-attributes",
        nargs="+",
        default=DEFAULT_PROTECTED_ATTRIBUTES,
        help=f"Attributes to evaluate (default: {DEFAULT_PROTECTED_ATTRIBUTES})",
    )
    parser.add_argument(
        "--min-subgroup-size",
        type=int,
        default=FAIRNESS_DEFAULT_MIN_SUBGROUP_SIZE,
        help="Min patients per subgroup",
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs without processing")
    parser.add_argument(
        "--include-extension-tasks",
        action="store_true",
        help="Include task runs outside the fixed thesis task set.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully even if no runs match, fairness groups are skipped, or runs fail.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs that already have fairness metrics (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if fairness metrics already exist",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit runs (for debugging)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
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


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Default to the benchmark fairness classes if not specified.
    experiment_classes = args.experiment_class or DEFAULT_EXPERIMENT_CLASSES
    log.info("Experiment classes: %s", experiment_classes)

    # Fetch runs from W&B
    runs = _retry(
        lambda: fetch_eval_runs(
            project=args.project,
            entity=args.entity,
            experiment_classes=experiment_classes,
            paradigms=args.paradigm,
            datasets=args.dataset,
            phases=args.phase,
            revisions=args.revision,
        )
    )

    if not args.include_extension_tasks:
        before = len(runs)
        runs = filter_thesis_task_runs(runs)
        dropped = before - len(runs)
        if dropped:
            log.info("Dropped %d runs outside thesis tasks: %s", dropped, sorted(THESIS_TASKS))

    if not runs:
        print("No runs found matching filters.", file=sys.stderr)
        sys.exit(0 if args.allow_incomplete else 1)

    # Filter out runs that already have fairness metrics (unless --force)
    if args.skip_existing and not args.force:
        before = len(runs)
        runs = [
            r
            for r in runs
            if not has_fairness_metrics(
                r,
                args.protected_attributes,
                args.min_subgroup_size,
            )
        ]
        skipped = before - len(runs)
        if skipped:
            log.info("Skipped %d runs with fresh existing fairness metrics.", skipped)
        if not runs:
            log.info("No pending runs after --skip-existing filtering.")
            return

    if args.max_runs:
        runs = runs[: args.max_runs]

    # Sort by (dataset, task, seed) to maximize datamodule reuse
    def _sort_key(r):
        cfg = r.config
        return (
            cfg.get("dataset", ""),
            _get_nested(cfg, "task.task_name", ""),
            cfg.get("seed", 0),
        )

    runs.sort(key=_sort_key)

    print(f"\nRuns to evaluate: {len(runs)}")
    if args.dry_run:
        print("\n[DRY RUN] Listing runs:\n")
        for i, r in enumerate(runs):
            cfg = r.config
            task_type = _get_nested(cfg, "task.task_type", "binary")
            try:
                artifact, artifact_source = resolve_evaluation_artifact(
                    r, args.outputs_root, task_type
                )
                ckpt_str = f"{artifact} [{artifact_source}]"
            except Exception as e:
                ckpt_str = f"ERROR: {e}"
            print(
                f"  {i + 1:3d}. {r.name or r.id}  "
                f"[{cfg.get('dataset', '?')}/{_get_nested(cfg, 'task.task_name', '?')}/"
                f"seed{cfg.get('seed', '?')}]  "
                f"ckpt={ckpt_str}"
            )
        print(f"\nTotal: {len(runs)} runs")
        return

    # Process runs
    results = {"processed": 0, "skipped": 0, "failed": 0, "errors": []}
    prev_dm_key: Optional[tuple] = None
    datamodule = None

    for i, run in enumerate(runs):
        cfg = run.config
        run_desc = run.name or run.id
        ds = cfg.get("dataset", "?")
        task = _get_nested(cfg, "task.task_name", "?")
        seed = cfg.get("seed", "?")
        log.info("[%d/%d] %s (%s/%s/seed%s)", i + 1, len(runs), run_desc, ds, task, seed)

        try:
            # 1. Resolve the saved evaluation artifact
            task_type = _get_nested(cfg, "task.task_type", "binary")
            artifact_path, artifact_source = resolve_evaluation_artifact(
                run, args.outputs_root, task_type
            )
            log.info("  Evaluation artifact: %s (%s)", artifact_path, artifact_source)

            # 2. Reconstruct model + data (reuse datamodule if same dataset/task/seed)
            dm_key = (ds, task, seed, cfg.get("label_fraction", 1.0))
            if dm_key != prev_dm_key:
                if datamodule is not None:
                    del datamodule
                datamodule = build_datamodule(cfg, args.batch_size, args.data_root)
                prev_dm_key = dm_key

            paradigm = str(cfg.get("paradigm", "")).lower()
            model = None
            if paradigm == "xgboost":
                from slices.eval.inference import run_xgboost_inference

                predictions, labels, stay_ids = run_xgboost_inference(
                    artifact_path,
                    task_type,
                    datamodule.dataset,
                    datamodule.test_indices,
                )
            else:
                from slices.eval.inference import run_inference

                model = build_model(cfg, artifact_path, datamodule)
                model = model.to(device)
                predictions, labels, stay_ids = run_inference(
                    model,
                    datamodule.test_dataloader(),
                    device=device,
                )

            # 3. Evaluate fairness
            report = evaluate_predictions_fairness(
                predictions,
                labels,
                stay_ids,
                datamodule,
                args.protected_attributes,
                args.min_subgroup_size,
                task_type,
            )

            if not report:
                log.warning("  No fairness results (no valid attribute groups)")
                results["skipped"] += 1
                continue

            missing_requirements = missing_fairness_report_requirements(
                run,
                report,
                args.protected_attributes,
            )
            if missing_requirements:
                message = "Incomplete fairness results: " + "; ".join(missing_requirements)
                log.warning("  %s", message)
                results["skipped"] += 1
                results["errors"].append((run.id, run_desc, message))
                continue

            # 4. Flatten and write back
            fairness_flat = flatten_fairness_report(report)
            fairness_flat.update(
                build_fairness_summary_metadata(
                    run,
                    artifact_path,
                    artifact_source,
                    args.protected_attributes,
                    args.min_subgroup_size,
                )
            )
            run_path = (
                f"{args.entity}/{args.project}/{run.id}"
                if args.entity
                else f"{args.project}/{run.id}"
            )
            write_fairness_to_wandb(
                run_path,
                fairness_flat,
                args.dry_run,
                clear_existing=args.force,
            )
            results["processed"] += 1

            # Free model memory
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            results["failed"] += 1
            results["errors"].append((run.id, run_desc, str(e)))
            log.error("  FAILED: %s", e, exc_info=args.verbose)
            continue

    # Summary
    print("\n" + "=" * 60)
    print("Fairness Evaluation Summary")
    print("=" * 60)
    print(f"  Processed: {results['processed']}")
    print(f"  Skipped:   {results['skipped']}")
    print(f"  Failed:    {results['failed']}")
    if results["errors"]:
        print("\n  Errors:")
        for entry in results["errors"]:
            run_id, run_name, err = entry
            print(f"    {run_name} ({run_id}): {err}")
    print("=" * 60)

    if not args.allow_incomplete and (results["failed"] > 0 or results["skipped"] > 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
