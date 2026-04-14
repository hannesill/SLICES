#!/usr/bin/env python3
"""Post-run fairness evaluation for SLICES experiment runs.

Queries W&B for finished finetune/supervised runs, loads their best checkpoints,
runs inference on the test set, computes fairness metrics via FairnessEvaluator,
and writes results back to the same W&B run's summary.

Designed for batch evaluation of the thesis fairness corpus. Supports
resumability via --skip-existing (default) and scoping via --sprint/--paradigm/
--dataset filters.

Usage:
    # Evaluate the thesis fairness corpus for one explicit revision
    uv run python scripts/eval/evaluate_fairness.py --revision thesis-v1

    # Scope to specific sprint/dataset
    uv run python scripts/eval/evaluate_fairness.py --revision thesis-v1 --sprint 1 --dataset miiv

    # Preview which runs would be evaluated
    uv run python scripts/eval/evaluate_fairness.py --revision thesis-v1 --dry-run

    # Override paths (e.g., different machine than training)
    uv run python scripts/eval/evaluate_fairness.py \
        --revision thesis-v1 \
        --outputs-root /mnt/data/outputs --data-root /mnt/data

    # Recompute fairness for runs that already have metrics
    uv run python scripts/eval/evaluate_fairness.py --revision thesis-v1 --force

    # Debug with a single run
    uv run python scripts/eval/evaluate_fairness.py --revision thesis-v1 --max-runs 1
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

log = logging.getLogger("evaluate_fairness")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_SPRINTS = ["1", "2", "3", "4", "5", "7p", "10", "12", "13"]
DEFAULT_PHASES = ["finetune", "supervised"]
DEFAULT_PROTECTED_ATTRIBUTES = ["gender", "age_group", "race"]


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


def fetch_eval_runs(
    project: str,
    entity: Optional[str],
    sprints: Optional[list[str]],
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
    tag_filters: list[str] = []

    if sprints and len(sprints) == 1:
        tag_filters.append(f"sprint:{sprints[0]}")
    if paradigms and len(paradigms) == 1:
        tag_filters.append(f"paradigm:{paradigms[0]}")
    if datasets and len(datasets) == 1:
        tag_filters.append(f"dataset:{datasets[0]}")
    if phases and len(phases) == 1:
        tag_filters.append(f"phase:{phases[0]}")
    if revisions and len(revisions) == 1:
        tag_filters.append(f"revision:{revisions[0]}")

    if tag_filters:
        filters["tags"] = {"$all": tag_filters}

    log.info("Fetching runs from %s with filters: %s", path, json.dumps(filters, default=str))
    runs_iter = api.runs(path, filters=filters or {}, order="-created_at")

    # Client-side filtering for multi-value filters
    sprint_set = set(sprints) if sprints and len(sprints) > 1 else None
    paradigm_set = set(paradigms) if paradigms and len(paradigms) > 1 else None
    dataset_set = set(datasets) if datasets and len(datasets) > 1 else None
    phase_set = set(phases) if phases and len(phases) > 1 else None
    revision_set = set(revisions) if revisions and len(revisions) > 1 else None

    runs = []
    for run in runs_iter:
        tags = set(run.tags)

        if sprint_set and not any(f"sprint:{s}" in tags for s in sprint_set):
            continue
        if paradigm_set and not any(f"paradigm:{p}" in tags for p in paradigm_set):
            continue
        if dataset_set and not any(f"dataset:{d}" in tags for d in dataset_set):
            continue
        if phase_set and not any(f"phase:{p}" in tags for p in phase_set):
            continue
        if revision_set and not any(f"revision:{r}" in tags for r in revision_set):
            continue

        runs.append(run)

    log.info("Fetched %d runs.", len(runs))
    return runs


def has_fairness_metrics(run) -> bool:
    """Check if run summary already contains fairness/* keys.

    Uses ``summary_metrics`` (populated from the batch query) instead of
    ``summary._json_dict`` which triggers a per-run GraphQL reload.
    """
    try:
        sm = run.summary_metrics or {}
        return any(k.startswith("fairness/") for k in sm)
    except Exception:
        return False


def write_fairness_to_wandb(
    run_path: str,
    fairness_flat: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Write fairness metrics to W&B run summary (additive only)."""
    import wandb

    if dry_run:
        log.info("  [DRY RUN] Would write %d fairness keys to %s", len(fairness_flat), run_path)
        return

    def _do_update():
        api = wandb.Api(timeout=120)
        run = api.run(run_path)
        run.summary.update(fairness_flat)
        run.summary.save()

    _retry(_do_update)
    log.info("  Wrote %d fairness keys to W&B run %s", len(fairness_flat), run_path)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _resolve_ckpt_dir(output_dir: str, outputs_root: Optional[str] = None) -> Path:
    """Resolve the checkpoint directory, applying outputs_root rebase if needed."""
    if outputs_root:
        rel = output_dir
        if rel.startswith("outputs/"):
            rel = rel[len("outputs/") :]
        elif rel.startswith("/") and "/outputs/" in rel:
            rel = rel.split("/outputs/", 1)[1]
        return Path(outputs_root) / rel / "checkpoints"
    return Path(output_dir) / "checkpoints"


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
    from slices.eval.fairness_evaluator import FairnessEvaluator
    from slices.eval.inference import run_inference

    model = model.to(device)
    predictions, labels, stay_ids = run_inference(
        model,
        datamodule.test_dataloader(),
        device=device,
    )

    task_type = getattr(model, "task_type", "binary")
    evaluator = FairnessEvaluator(
        static_df=datamodule.dataset.static_df,
        protected_attributes=protected_attributes,
        min_subgroup_size=min_subgroup_size,
        task_type=task_type,
        dataset_name=getattr(getattr(datamodule, "processed_dir", None), "name", None),
    )
    report = evaluator.evaluate(predictions, labels, stay_ids)
    return report


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
        default=os.environ.get("WANDB_PROJECT", "slices"),
        help="W&B project name (default: $WANDB_PROJECT or 'slices')",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (default: $WANDB_ENTITY)",
    )
    parser.add_argument(
        "--sprint",
        nargs="+",
        help="Filter to sprint(s). Default: thesis fairness sprints",
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
        "--min-subgroup-size", type=int, default=50, help="Min samples per subgroup"
    )
    parser.add_argument("--dry-run", action="store_true", help="List runs without processing")
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

    # Default to the thesis fairness sprint set if not specified
    sprints = args.sprint or CORE_SPRINTS
    log.info("Sprints: %s", sprints)

    # Fetch runs from W&B
    runs = _retry(
        lambda: fetch_eval_runs(
            project=args.project,
            entity=args.entity,
            sprints=sprints,
            paradigms=args.paradigm,
            datasets=args.dataset,
            phases=args.phase,
            revisions=args.revision,
        )
    )

    if not runs:
        print("No runs found matching filters.", file=sys.stderr)
        sys.exit(0)

    # Filter out runs that already have fairness metrics (unless --force)
    if args.skip_existing and not args.force:
        before = len(runs)
        runs = [r for r in runs if not has_fairness_metrics(r)]
        skipped = before - len(runs)
        if skipped:
            log.info("Skipped %d runs with existing fairness metrics.", skipped)

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
                ckpt, ckpt_source = resolve_evaluation_checkpoint(r, args.outputs_root, task_type)
                ckpt_str = f"{ckpt} [{ckpt_source}]" if ckpt else "NOT FOUND"
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
            # 1. Find checkpoint
            task_type = _get_nested(cfg, "task.task_type", "binary")
            ckpt_path, ckpt_source = resolve_evaluation_checkpoint(
                run, args.outputs_root, task_type
            )
            if ckpt_path is None:
                results["skipped"] += 1
                results["errors"].append((run.id, run_desc, "no checkpoint found"))
                continue
            log.info("  Checkpoint: %s (%s)", ckpt_path, ckpt_source)

            # 2. Reconstruct model + data (reuse datamodule if same dataset/task/seed)
            dm_key = (ds, task, seed, cfg.get("label_fraction", 1.0))
            if dm_key != prev_dm_key:
                if datamodule is not None:
                    del datamodule
                datamodule = build_datamodule(cfg, args.batch_size, args.data_root)
                prev_dm_key = dm_key

            model = build_model(cfg, ckpt_path, datamodule)

            # 3. Evaluate fairness
            report = evaluate_run_fairness(
                model,
                datamodule,
                args.protected_attributes,
                args.min_subgroup_size,
                device,
            )

            if not report:
                log.warning("  No fairness results (no valid attribute groups)")
                results["skipped"] += 1
                continue

            # 4. Flatten and write back
            fairness_flat = flatten_fairness_report(report)
            run_path = (
                f"{args.entity}/{args.project}/{run.id}"
                if args.entity
                else f"{args.project}/{run.id}"
            )
            write_fairness_to_wandb(run_path, fairness_flat, args.dry_run)
            results["processed"] += 1

            # Free model memory
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


if __name__ == "__main__":
    main()
