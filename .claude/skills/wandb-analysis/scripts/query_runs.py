#!/usr/bin/env python3
"""
Query and Filter Wandb Runs

Search for runs in a wandb project using tags, config filters, groups, and
state. Supports aggregation across matching runs for experiment analysis.

Usage:
    # Find all Sprint 1 finetune runs
    python query_runs.py --tags "sprint:1" "phase:finetune"

    # Find all MAE runs that finished
    python query_runs.py --tags "paradigm:mae" --state finished

    # Filter by config values (wandb MongoDB-style filter JSON)
    python query_runs.py --run-filters '{"config.optimizer.lr": 0.001}'

    # Find runs in a specific group
    python query_runs.py --group "pretrain_miiv_mae"

    # Aggregate test AUROC across seeds, grouped by paradigm
    python query_runs.py --tags "sprint:1" "phase:finetune" \
        --aggregate --group-by paradigm --metrics test/auroc

    # Show only latest 10 runs
    python query_runs.py --latest 10

    # Use a specific project (overrides env vars)
    python query_runs.py --project slices --entity myteam --tags "sprint:1"

Options:
    --tags TAG1 TAG2      Filter by wandb tags (runs must have ALL specified tags)
    --group GROUP         Filter by wandb group name
    --state STATE         Filter by run state: finished, running, crashed, failed
    --run-filters JSON    Raw wandb MongoDB-style filter JSON
    --run-names PATTERN   Glob pattern for run names (client-side fnmatch)
    --latest N            Only return N most recent runs
    --metrics M1,M2       Specific summary metrics to show (comma-separated)
    --aggregate           Compute mean/std across matching runs
    --group-by KEY        Config key to group by when aggregating (e.g., paradigm, dataset)
    --sort-by METRIC      Sort runs by this metric
    --output FORMAT       Output format: 'text' or 'json' (default: text)
    --project PROJECT     W&B project name (default: WANDB_PROJECT env var)
    --entity ENTITY       W&B entity name (default: WANDB_ENTITY env var)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from fnmatch import fnmatch
from math import sqrt
from typing import Any

try:
    import pandas as pd
    import wandb
except ImportError as e:
    print(f"Error: Required package not installed: {e}")
    print("Install with: pip install wandb pandas")
    sys.exit(1)


def fetch_runs(
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    state: str | None = None,
    run_filters: dict | None = None,
    run_names: str | None = None,
    latest: int | None = None,
) -> list:
    """
    Fetch runs from wandb with server-side and client-side filtering.

    Args:
        project: W&B project name
        entity: W&B entity (username or team)
        tags: List of tags to filter by (runs must have ALL tags)
        group: Group name to filter by
        state: Run state filter (finished, running, crashed, failed)
        run_filters: Raw MongoDB-style filter dict
        run_names: Glob pattern for run names (client-side)
        latest: Only return N most recent runs

    Returns:
        List of wandb Run objects
    """
    api = wandb.Api()

    # Build server-side filters
    filters = {}
    if tags:
        # All specified tags must be present
        filters["tags"] = {"$all": tags}
    if group:
        filters["group"] = group
    if state:
        filters["state"] = state
    if run_filters:
        filters.update(run_filters)

    path = f"{entity}/{project}" if entity else project
    order = "-created_at"

    print(f"Querying {path} with filters: {json.dumps(filters, default=str)}...", file=sys.stderr)

    runs = api.runs(path, filters=filters or {}, order=order)

    # Client-side filtering
    result = []
    for run in runs:
        if run_names and not fnmatch(run.name, run_names):
            continue
        result.append(run)
        if latest and len(result) >= latest:
            break

    print(f"Found {len(result)} matching runs.", file=sys.stderr)
    return result


def extract_run_data(run, metrics: list[str] | None = None) -> dict[str, Any]:
    """Extract key data from a wandb Run object."""
    summary = {}
    for k, v in run.summary._json_dict.items():
        if k.startswith("_") or isinstance(v, dict):
            continue
        if metrics and k not in metrics:
            continue
        if isinstance(v, float):
            if pd.isna(v):
                summary[k] = None
            else:
                summary[k] = round(v, 6)
        else:
            summary[k] = v

    # Extract key config values for display
    config = dict(run.config)
    key_config = {}
    for key in [
        "sprint",
        "paradigm",
        "dataset",
        "seed",
        "label_fraction",
        "protocol",
        "revision",
        "rerun_reason",
    ]:
        if key in config:
            key_config[key] = config[key]
    # Also extract nested config keys
    for dotted_key in [
        "optimizer.lr",
        "ssl.mask_ratio",
        "ssl.name",
        "training.freeze_encoder",
        "training.max_epochs",
        "task.task_name",
    ]:
        parts = dotted_key.split(".")
        val = config
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                val = None
                break
        if val is not None:
            key_config[dotted_key] = val

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "tags": list(run.tags),
        "group": run.group,
        "created_at": run.created_at,
        "url": run.url,
        "config": key_config,
        "summary": summary,
    }


def aggregate_runs(
    runs_data: list[dict],
    group_by: str,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    Aggregate metrics across runs grouped by a config key.

    Args:
        runs_data: List of run data dicts from extract_run_data
        group_by: Config key to group by (e.g., 'paradigm', 'dataset')
        metrics: Specific metrics to aggregate (None = all numeric summary metrics)

    Returns:
        Dict with groups and their aggregated stats
    """
    groups = defaultdict(list)
    for run in runs_data:
        key = run["config"].get(group_by, "unknown")
        groups[key].append(run)

    result = {}
    for group_key, group_runs in sorted(groups.items()):
        # Collect all numeric metrics across runs in this group
        all_metrics = set()
        for run in group_runs:
            for k, v in run["summary"].items():
                if isinstance(v, (int, float)) and v is not None:
                    if metrics is None or k in metrics:
                        all_metrics.add(k)

        stats = {}
        for metric in sorted(all_metrics):
            values = [
                run["summary"][metric]
                for run in group_runs
                if metric in run["summary"]
                and isinstance(run["summary"][metric], (int, float))
                and run["summary"][metric] is not None
            ]
            if values:
                mean = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
                    std = sqrt(variance)
                else:
                    std = 0.0
                stats[metric] = {
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                    "n": len(values),
                }

        result[str(group_key)] = {
            "n_runs": len(group_runs),
            "run_ids": [r["run_id"] for r in group_runs],
            "stats": stats,
        }

    return result


def format_text_output(
    runs_data: list[dict],
    aggregate_result: dict | None = None,
    group_by: str | None = None,
    metrics: list[str] | None = None,
    sort_by: str | None = None,
) -> str:
    """Format results as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("WANDB RUN QUERY RESULTS")
    lines.append("=" * 80)

    if aggregate_result:
        lines.append(f"\n## Aggregated Results (grouped by: {group_by})")
        for group_key, group_data in aggregate_result.items():
            lines.append(f"\n### {group_by}={group_key}  (n={group_data['n_runs']})")
            if group_data["stats"]:
                # Build table
                header = (
                    f"  {'Metric':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>4}"
                )
                lines.append(header)
                lines.append("  " + "-" * 83)
                for metric, stat in group_data["stats"].items():
                    lines.append(
                        f"  {metric:<35} {stat['mean']:>10.4f} {stat['std']:>10.4f} "
                        f"{stat['min']:>10.4f} {stat['max']:>10.4f} {stat['n']:>4d}"
                    )
            else:
                lines.append("  No numeric metrics found.")
    else:
        lines.append(f"\nFound {len(runs_data)} runs\n")

        # Sort if requested
        if sort_by:
            runs_data = sorted(
                runs_data,
                key=lambda r: (
                    r["summary"].get(sort_by, float("-inf"))
                    if isinstance(r["summary"].get(sort_by), (int, float))
                    else float("-inf")
                ),
                reverse=True,
            )

        lines.append("## Runs")
        for i, run in enumerate(runs_data):
            lines.append(f"\n--- Run {i + 1}: {run['run_name']} ---")
            lines.append(f"  ID: {run['run_id']}")
            lines.append(f"  State: {run['state']}")
            lines.append(f"  Tags: {', '.join(run['tags'])}")
            if run["group"]:
                lines.append(f"  Group: {run['group']}")
            lines.append(f"  Created: {run['created_at']}")

            # Key config
            if run["config"]:
                config_str = ", ".join(f"{k}={v}" for k, v in run["config"].items())
                lines.append(f"  Config: {config_str}")

            # Summary metrics
            if run["summary"]:
                if metrics:
                    shown = {k: v for k, v in run["summary"].items() if k in metrics}
                else:
                    shown = run["summary"]
                if shown:
                    metrics_str = ", ".join(f"{k}={v}" for k, v in shown.items())
                    lines.append(f"  Metrics: {metrics_str}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query and filter wandb runs")
    parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        help="Filter by wandb tags (runs must have ALL specified tags)",
    )
    parser.add_argument(
        "--group",
        "-g",
        help="Filter by wandb group name",
    )
    parser.add_argument(
        "--state",
        choices=["finished", "running", "crashed", "failed"],
        help="Filter by run state",
    )
    parser.add_argument(
        "--run-filters",
        help="Raw wandb MongoDB-style filter JSON",
    )
    parser.add_argument(
        "--run-names",
        help="Glob pattern for run names (client-side fnmatch)",
    )
    parser.add_argument(
        "--latest",
        "-n",
        type=int,
        help="Only return N most recent runs",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        help="Comma-separated list of specific summary metrics to show",
    )
    parser.add_argument(
        "--aggregate",
        "-a",
        action="store_true",
        help="Compute mean/std across matching runs",
    )
    parser.add_argument(
        "--group-by",
        help="Config key to group by when aggregating (e.g., paradigm, dataset)",
    )
    parser.add_argument(
        "--sort-by",
        "-s",
        help="Sort runs by this metric (descending)",
    )
    parser.add_argument(
        "--output",
        "-o",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--project",
        "-p",
        default=os.environ.get("WANDB_PROJECT", "slices"),
        help="W&B project name (default: WANDB_PROJECT env var or 'slices')",
    )
    parser.add_argument(
        "--entity",
        "-e",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity name (default: WANDB_ENTITY env var)",
    )

    args = parser.parse_args()

    # Parse metrics
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]

    # Parse run-filters
    run_filters = None
    if args.run_filters:
        try:
            run_filters = json.loads(args.run_filters)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --run-filters: {e}", file=sys.stderr)
            sys.exit(1)

    # Validate aggregate options
    if args.aggregate and not args.group_by:
        print("Error: --aggregate requires --group-by", file=sys.stderr)
        sys.exit(1)

    # Fetch runs
    runs = fetch_runs(
        project=args.project,
        entity=args.entity,
        tags=args.tags,
        group=args.group,
        state=args.state,
        run_filters=run_filters,
        run_names=args.run_names,
        latest=args.latest,
    )

    if not runs:
        print("No matching runs found.", file=sys.stderr)
        sys.exit(0)

    # Extract data
    runs_data = []
    for run in runs:
        runs_data.append(extract_run_data(run, metrics=metrics))

    # Aggregate if requested
    aggregate_result = None
    if args.aggregate:
        aggregate_result = aggregate_runs(runs_data, args.group_by, metrics=metrics)

    # Output
    if args.output == "json":
        output = {
            "query": {
                "project": args.project,
                "entity": args.entity,
                "tags": args.tags,
                "group": args.group,
                "state": args.state,
                "run_filters": run_filters,
            },
            "n_runs": len(runs_data),
            "runs": runs_data,
        }
        if aggregate_result:
            output["aggregated"] = aggregate_result
        print(json.dumps(output, indent=2, default=str))
    else:
        print(
            format_text_output(
                runs_data,
                aggregate_result=aggregate_result,
                group_by=args.group_by,
                metrics=metrics,
                sort_by=args.sort_by,
            )
        )


if __name__ == "__main__":
    main()
