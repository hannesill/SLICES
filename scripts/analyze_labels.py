#!/usr/bin/env python
"""Analyze label distributions in processed datasets.

This script provides comprehensive statistics about task labels including:
- Available tasks and their types
- Class distributions (counts and percentages)
- Missing label counts
- Per-split statistics (train/val/test)
- Class imbalance metrics

Example usage:
    # Analyze labels in processed directory
    uv run python scripts/analyze_labels.py data/processed/mimic-iv-demo

    # Analyze specific task
    uv run python scripts/analyze_labels.py data/processed/mimic-iv-demo --task mortality_24h

    # Show per-split statistics
    uv run python scripts/analyze_labels.py data/processed/mimic-iv-demo --splits

    # Export to JSON
    uv run python scripts/analyze_labels.py data/processed/mimic-iv-demo --output stats.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import yaml


def load_labels(processed_dir: Path) -> pl.DataFrame:
    """Load labels from parquet file.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        DataFrame with stay_id and task label columns.

    Raises:
        FileNotFoundError: If labels.parquet doesn't exist.
    """
    labels_path = processed_dir / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    return pl.read_parquet(labels_path)


def load_metadata(processed_dir: Path) -> Dict[str, Any]:
    """Load metadata from YAML file.

    Args:
        processed_dir: Path to processed data directory.

    Returns:
        Metadata dictionary.
    """
    metadata_path = processed_dir / "metadata.yaml"
    if not metadata_path.exists():
        return {}

    with open(metadata_path) as f:
        return yaml.safe_load(f) or {}


def get_task_columns(labels_df: pl.DataFrame) -> List[str]:
    """Get task column names (excluding stay_id).

    Args:
        labels_df: Labels DataFrame.

    Returns:
        List of task column names.
    """
    return [col for col in labels_df.columns if col != "stay_id"]


def analyze_task(
    labels_df: pl.DataFrame,
    task_name: str,
) -> Dict[str, Any]:
    """Analyze a single task's label distribution.

    Args:
        labels_df: Labels DataFrame.
        task_name: Name of the task column.

    Returns:
        Dictionary with task statistics.
    """
    if task_name not in labels_df.columns:
        return {"error": f"Task '{task_name}' not found in labels"}

    total_stays = len(labels_df)
    task_labels = labels_df[task_name]

    # Count missing labels
    missing_count = task_labels.null_count()
    valid_count = total_stays - missing_count

    # Get non-null labels for class distribution
    valid_labels = task_labels.drop_nulls()

    stats = {
        "task_name": task_name,
        "total_stays": total_stays,
        "valid_labels": valid_count,
        "missing_labels": missing_count,
        "missing_percentage": (missing_count / total_stays * 100) if total_stays > 0 else 0,
    }

    if valid_count == 0:
        stats["class_distribution"] = {}
        stats["n_classes"] = 0
        return stats

    # Get unique classes and their counts
    class_counts = valid_labels.value_counts().sort("count", descending=True)

    class_distribution = {}
    for row in class_counts.iter_rows():
        class_value, count = row
        percentage = (count / valid_count * 100) if valid_count > 0 else 0
        class_distribution[int(class_value)] = {
            "count": count,
            "percentage": round(percentage, 2),
        }

    stats["class_distribution"] = class_distribution
    stats["n_classes"] = len(class_distribution)

    # Binary classification specific metrics
    if stats["n_classes"] == 2 and 0 in class_distribution and 1 in class_distribution:
        positive_count = class_distribution[1]["count"]
        negative_count = class_distribution[0]["count"]

        stats["positive_count"] = positive_count
        stats["negative_count"] = negative_count
        stats["positive_rate"] = round(positive_count / valid_count * 100, 2)
        stats["negative_rate"] = round(negative_count / valid_count * 100, 2)

        # Imbalance ratio (majority / minority)
        majority = max(positive_count, negative_count)
        minority = min(positive_count, negative_count)
        stats["imbalance_ratio"] = round(majority / minority, 2) if minority > 0 else float("inf")

        # Class weight for balanced training
        stats["suggested_class_weights"] = {
            0: round(valid_count / (2 * negative_count), 4) if negative_count > 0 else 1.0,
            1: round(valid_count / (2 * positive_count), 4) if positive_count > 0 else 1.0,
        }

    return stats


def analyze_splits(
    processed_dir: Path,
    task_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Analyze label distribution across train/val/test splits.

    Args:
        processed_dir: Path to processed data directory.
        task_name: Task to analyze.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed for splits.

    Returns:
        Dictionary with per-split statistics.
    """
    from slices.data.datamodule import ICUDataModule

    datamodule = ICUDataModule(
        processed_dir=str(processed_dir),
        task_name=task_name,
        batch_size=32,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        normalize=False,
    )

    datamodule.setup()

    split_stats = {}

    for split_name, dataset in [
        ("train", datamodule.train_dataset),
        ("val", datamodule.val_dataset),
        ("test", datamodule.test_dataset),
    ]:
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if "label" in sample and sample["label"] is not None:
                labels.append(int(sample["label"]))

        if not labels:
            split_stats[split_name] = {
                "n_samples": len(dataset),
                "n_valid_labels": 0,
                "class_distribution": {},
            }
            continue

        labels_series = pl.Series(labels)
        class_counts = labels_series.value_counts().sort("count", descending=True)

        class_distribution = {}
        for row in class_counts.iter_rows():
            class_value, count = row
            percentage = (count / len(labels) * 100) if labels else 0
            class_distribution[int(class_value)] = {
                "count": count,
                "percentage": round(percentage, 2),
            }

        split_stats[split_name] = {
            "n_samples": len(dataset),
            "n_valid_labels": len(labels),
            "class_distribution": class_distribution,
        }

        # Add positive rate for binary tasks
        if 0 in class_distribution and 1 in class_distribution:
            split_stats[split_name]["positive_rate"] = class_distribution[1]["percentage"]

    return split_stats


def print_task_stats(stats: Dict[str, Any]) -> None:
    """Print formatted task statistics.

    Args:
        stats: Task statistics dictionary.
    """
    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"\n  Task: {stats['task_name']}")
    print(f"  {'─' * 50}")
    print(f"  Total stays:     {stats['total_stays']:,}")
    print(f"  Valid labels:    {stats['valid_labels']:,}")
    print(f"  Missing labels:  {stats['missing_labels']:,} ({stats['missing_percentage']:.1f}%)")
    print(f"  Number of classes: {stats['n_classes']}")

    if stats["class_distribution"]:
        print("\n  Class Distribution:")
        for class_val, class_stats in stats["class_distribution"].items():
            print(
                f"    Class {class_val}: {class_stats['count']:,} "
                f"({class_stats['percentage']:.1f}%)"
            )

    # Binary classification specific
    if "imbalance_ratio" in stats:
        print("\n  Binary Classification Metrics:")
        print(f"    Positive (1): {stats['positive_count']:,} ({stats['positive_rate']:.1f}%)")
        print(f"    Negative (0): {stats['negative_count']:,} ({stats['negative_rate']:.1f}%)")
        print(f"    Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
        print(f"    Suggested class weights: {stats['suggested_class_weights']}")


def print_split_stats(split_stats: Dict[str, Dict[str, Any]], task_name: str) -> None:
    """Print formatted split statistics.

    Args:
        split_stats: Per-split statistics.
        task_name: Name of the task.
    """
    print(f"\n  Per-Split Statistics for '{task_name}':")
    print(f"  {'─' * 50}")

    # Header
    print(f"  {'Split':<10} {'Samples':>10} {'Positive':>12} {'Negative':>12} {'Pos Rate':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}")

    for split_name in ["train", "val", "test"]:
        if split_name not in split_stats:
            continue

        stats = split_stats[split_name]
        n_samples = stats["n_samples"]

        if stats["class_distribution"]:
            pos = stats["class_distribution"].get(1, {}).get("count", 0)
            neg = stats["class_distribution"].get(0, {}).get("count", 0)
            pos_rate = stats.get("positive_rate", 0)
            print(f"  {split_name:<10} {n_samples:>10,} {pos:>12,} {neg:>12,} {pos_rate:>9.1f}%")
        else:
            print(f"  {split_name:<10} {n_samples:>10,} {'N/A':>12} {'N/A':>12} {'N/A':>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze label distributions in processed datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "processed_dir",
        type=Path,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Analyze specific task only (default: all tasks)",
    )
    parser.add_argument(
        "--splits",
        "-s",
        action="store_true",
        help="Show per-split statistics (train/val/test)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Export statistics to JSON file",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio for split analysis (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split analysis (default: 42)",
    )

    args = parser.parse_args()

    # Validate directory
    if not args.processed_dir.exists():
        print(f"Error: Directory not found: {args.processed_dir}")
        return 1

    print("=" * 60)
    print("Label Distribution Analysis")
    print("=" * 60)
    print(f"\nProcessed directory: {args.processed_dir}")

    # Load labels
    try:
        labels_df = load_labels(args.processed_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1

    # Load metadata
    load_metadata(args.processed_dir)

    print(f"Total stays in labels.parquet: {len(labels_df):,}")

    # Get tasks to analyze
    task_columns = get_task_columns(labels_df)

    if not task_columns:
        print("\nNo task columns found in labels.parquet")
        return 1

    print(f"Available tasks: {', '.join(task_columns)}")

    # Filter to specific task if requested
    if args.task:
        if args.task not in task_columns:
            print(f"\nError: Task '{args.task}' not found. Available: {task_columns}")
            return 1
        task_columns = [args.task]

    # Analyze each task
    all_stats = {}
    print("\n" + "=" * 60)
    print("Task Statistics")
    print("=" * 60)

    for task_name in task_columns:
        stats = analyze_task(labels_df, task_name)
        all_stats[task_name] = stats
        print_task_stats(stats)

    # Per-split analysis
    if args.splits:
        print("\n" + "=" * 60)
        print("Per-Split Analysis")
        print("=" * 60)

        val_ratio = (1.0 - args.train_ratio) / 2
        test_ratio = val_ratio

        for task_name in task_columns:
            try:
                split_stats = analyze_splits(
                    args.processed_dir,
                    task_name,
                    train_ratio=args.train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    seed=args.seed,
                )
                all_stats[task_name]["splits"] = split_stats
                print_split_stats(split_stats, task_name)
            except Exception as e:
                print(f"\n  Could not analyze splits for '{task_name}': {e}")

    # Export to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStatistics exported to: {args.output}")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
