#!/usr/bin/env python3
"""Export cached random-Transformer probe diagnostics.

This reads local cached-probe JSON outputs and compares them with the frozen
SSL linear-probe rows from the thesis export. It writes a separate diagnostic
artifact; it does not modify the frozen thesis result export.

Usage:
    uv run python scripts/diagnostics/export_random_probe_diagnostic.py

Outputs (under --out):
    random_probe_per_seed.parquet
    random_probe_ssl_comparison_per_seed.parquet
    random_probe_task_summary.parquet
    random_probe_overall.json
    checksum_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SSL_PARADIGMS = ("mae", "jepa", "contrastive")
DEFAULT_DATASETS = ("miiv", "eicu")
DEFAULT_TASKS = ("mortality_24h", "mortality_hospital", "aki_kdigo", "los_remaining")
DEFAULT_SEEDS = (42, 123, 456, 789, 1011)
LOWER_IS_BETTER_METRICS = {"test/mae", "test/mse", "test/rmse", "test/loss"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="outputs/diagnostics/random_transformer_probe_cached",
        help="Cached random-probe output root.",
    )
    p.add_argument(
        "--thesis-per-seed",
        default="results/slices-thesis-v1/per_seed_results.parquet",
        help="Frozen thesis per-seed result parquet.",
    )
    p.add_argument(
        "--out",
        default="results/slices-thesis-random-probe-diagnostic-v1",
        help="Diagnostic artifact output directory.",
    )
    p.add_argument("--revision", default="random-probe-cached-v1")
    p.add_argument("--thesis-revision", default="thesis-v1")
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def primary_metric(task: str) -> str:
    return "test/mae" if task == "los_remaining" else "test/auprc"


def metric_lower_is_better(metric_name: str) -> bool:
    return metric_name in LOWER_IS_BETTER_METRICS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tabular_shape(path: Path) -> tuple[int | None, int | None]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return int(len(df)), int(len(df.columns))
    return None, None


def write_checksum_manifest(out_dir: Path, files: list[Path], args: argparse.Namespace) -> None:
    manifest = {
        "package": out_dir.name,
        "source_cached_probe_dir": args.input,
        "source_thesis_per_seed": args.thesis_per_seed,
        "revision": args.revision,
        "thesis_revision": args.thesis_revision,
        "generation_note": (
            "Local cached random-Transformer linear-probe diagnostic. "
            "Separate from the frozen thesis export; no model weights are included."
        ),
        "files": {},
    }
    for path in sorted(files):
        rows, columns = tabular_shape(path)
        entry: dict[str, Any] = {"sha256": sha256_file(path)}
        if rows is not None:
            entry["rows"] = rows
            entry["columns"] = columns
        manifest["files"][path.name] = entry
    (out_dir / "checksum_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_random_probe_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/seed*/test_results.json")):
        payload = json.loads(path.read_text())
        task = str(payload["task"])
        metric_name = primary_metric(task)
        metrics = payload["metrics"]
        split_info = payload.get("split_info", {})
        timings = payload.get("timings_sec", {})
        support = payload.get("train_label_support", {})
        row = {
            "dataset": payload["dataset"],
            "task": task,
            "seed": int(payload["seed"]),
            "paradigm": "random_transformer",
            "protocol": "linear_probe_cached_embeddings",
            "experiment_class": payload.get("experiment_class", "random_probe_diagnostic"),
            "experiment_subtype": payload.get(
                "experiment_subtype",
                "random_transformer_cached_linear_probe",
            ),
            "revision": payload.get("revision"),
            "metric_name": metric_name,
            "metric_value": float(metrics[metric_name]),
            "output_dir": payload.get("output_dir"),
            "embedding_cache": payload.get("embedding_cache"),
            "embedding_cache_hit": payload.get("embedding_cache_hit"),
            "best_checkpoint": payload.get("best_checkpoint"),
            "best_monitor": payload.get("best_monitor"),
            "best_model_score": payload.get("best_model_score"),
            "train_patients": split_info.get("train_patients"),
            "train_stays": split_info.get("train_stays"),
            "val_patients": split_info.get("val_patients"),
            "val_stays": split_info.get("val_stays"),
            "test_patients": split_info.get("test_patients"),
            "test_stays": split_info.get("test_stays"),
            "embedding_sec": timings.get("embedding_sec"),
            "train_sec": timings.get("train_sec"),
            "test_sec": timings.get("test_sec"),
            "total_sec": timings.get("total_sec"),
            "train_subset_total": support.get("train_subset_total"),
            "train_subset_positive": support.get("train_subset_positive"),
            "train_subset_negative": support.get("train_subset_negative"),
            "train_subset_prevalence": support.get("train_subset_prevalence"),
        }
        for key, value in metrics.items():
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_complete(random_df: pd.DataFrame, allow_incomplete: bool) -> None:
    expected = {
        (dataset, task, seed)
        for dataset in DEFAULT_DATASETS
        for task in DEFAULT_TASKS
        for seed in DEFAULT_SEEDS
    }
    observed = set(random_df[["dataset", "task", "seed"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if (missing or extra) and not allow_incomplete:
        raise SystemExit(
            "Cached random-probe corpus is incomplete or has unexpected jobs. "
            f"missing={missing}; extra={extra}. Use --allow-incomplete to export anyway."
        )


def load_ssl_rows(path: Path, thesis_revision: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ssl = df[
        (df["experiment_class"] == "core_ssl_benchmark")
        & (df["revision"] == thesis_revision)
        & (df["paradigm"].isin(SSL_PARADIGMS))
        & (df["protocol"] == "linear_probe")
        & (df["label_fraction"] == 1.0)
        & (df["model_size"] == "default")
        & (df["dataset"].isin(DEFAULT_DATASETS))
        & (df["task"].isin(DEFAULT_TASKS))
    ].copy()
    ssl["metric_name"] = ssl["task"].map(primary_metric)
    ssl["ssl_metric_value"] = ssl.apply(lambda row: row[row["metric_name"]], axis=1)
    return ssl[
        [
            "dataset",
            "task",
            "seed",
            "paradigm",
            "protocol",
            "metric_name",
            "ssl_metric_value",
            "wandb_run_id",
            "wandb_run_name",
            "wandb_run_url",
        ]
    ].rename(columns={"paradigm": "ssl_paradigm", "protocol": "ssl_protocol"})


def compare_to_ssl(random_df: pd.DataFrame, ssl_df: pd.DataFrame) -> pd.DataFrame:
    comp = ssl_df.merge(
        random_df[
            [
                "dataset",
                "task",
                "seed",
                "metric_name",
                "metric_value",
                "output_dir",
                "embedding_cache",
            ]
        ],
        on=["dataset", "task", "seed", "metric_name"],
        how="inner",
    ).rename(columns={"metric_value": "random_metric_value"})
    comp["delta_random_minus_ssl"] = comp["random_metric_value"] - comp["ssl_metric_value"]
    comp["random_worse_than_ssl"] = comp.apply(
        lambda row: (
            row["random_metric_value"] > row["ssl_metric_value"]
            if metric_lower_is_better(row["metric_name"])
            else row["random_metric_value"] < row["ssl_metric_value"]
        ),
        axis=1,
    )
    comp["bad_delta"] = comp.apply(
        lambda row: (
            row["delta_random_minus_ssl"]
            if metric_lower_is_better(row["metric_name"])
            else -row["delta_random_minus_ssl"]
        ),
        axis=1,
    )
    return comp


def task_summary(random_df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    rand_agg = random_df.groupby(["dataset", "task", "metric_name"], as_index=False).agg(
        random_mean=("metric_value", "mean"),
        random_sd=("metric_value", "std"),
        n_random_seeds=("metric_value", "count"),
    )
    ssl_mean = comp.groupby(["dataset", "task", "metric_name"], as_index=False).agg(
        ssl_mean_across_paradigms_and_seeds=("ssl_metric_value", "mean")
    )

    best_rows = []
    for keys, group in comp.groupby(["dataset", "task", "seed", "metric_name"]):
        metric_name = keys[-1]
        idx = (
            group["ssl_metric_value"].idxmin()
            if metric_lower_is_better(metric_name)
            else group["ssl_metric_value"].idxmax()
        )
        best_rows.append(group.loc[idx])
    best = pd.DataFrame(best_rows)
    best_agg = best.groupby(["dataset", "task", "metric_name"], as_index=False).agg(
        best_ssl_mean=("ssl_metric_value", "mean")
    )

    counts = comp.groupby(["dataset", "task", "metric_name"], as_index=False).agg(
        ssl_pairwise_rows=("ssl_metric_value", "count"),
        random_worse_pairwise_count=("random_worse_than_ssl", "sum"),
        mean_bad_delta_pairwise=("bad_delta", "mean"),
    )

    summary = (
        rand_agg.merge(ssl_mean, on=["dataset", "task", "metric_name"])
        .merge(
            best_agg,
            on=["dataset", "task", "metric_name"],
        )
        .merge(counts, on=["dataset", "task", "metric_name"])
    )
    summary["delta_random_minus_ssl_mean"] = (
        summary["random_mean"] - summary["ssl_mean_across_paradigms_and_seeds"]
    )
    summary["delta_random_minus_best_ssl"] = summary["random_mean"] - summary["best_ssl_mean"]
    summary["bad_delta_vs_ssl_mean"] = summary.apply(
        lambda row: (
            row["delta_random_minus_ssl_mean"]
            if metric_lower_is_better(row["metric_name"])
            else -row["delta_random_minus_ssl_mean"]
        ),
        axis=1,
    )
    summary["bad_delta_vs_best_ssl"] = summary.apply(
        lambda row: (
            row["delta_random_minus_best_ssl"]
            if metric_lower_is_better(row["metric_name"])
            else -row["delta_random_minus_best_ssl"]
        ),
        axis=1,
    )
    return summary.sort_values(["dataset", "task"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_df = load_random_probe_rows(input_root)
    if random_df.empty:
        raise SystemExit(f"No test_results.json files found under {input_root}")
    validate_complete(random_df, args.allow_incomplete)

    ssl_df = load_ssl_rows(Path(args.thesis_per_seed), args.thesis_revision)
    comp = compare_to_ssl(random_df, ssl_df)
    summary = task_summary(random_df, comp)

    per_seed_path = out_dir / "random_probe_per_seed.parquet"
    comparison_path = out_dir / "random_probe_ssl_comparison_per_seed.parquet"
    summary_path = out_dir / "random_probe_task_summary.parquet"
    random_df.sort_values(["dataset", "task", "seed"]).to_parquet(per_seed_path, index=False)
    comp.sort_values(["dataset", "task", "seed", "ssl_paradigm"]).to_parquet(
        comparison_path,
        index=False,
    )
    summary.to_parquet(summary_path, index=False)

    overall = {
        "n_random_probe_rows": int(len(random_df)),
        "n_ssl_comparison_rows": int(len(comp)),
        "n_task_summaries": int(len(summary)),
        "datasets": sorted(random_df["dataset"].unique().tolist()),
        "tasks": sorted(random_df["task"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in random_df["seed"].unique()),
        "ssl_paradigms": list(SSL_PARADIGMS),
        "random_worse_pairwise_count": int(comp["random_worse_than_ssl"].sum()),
        "pairwise_count": int(len(comp)),
        "mean_bad_delta_by_metric": {
            metric: float(value)
            for metric, value in comp.groupby("metric_name")["bad_delta"].mean().items()
        },
        "interpretation": (
            "Positive bad_delta means the random Transformer probe is worse: lower AUPRC "
            "for classification tasks or higher MAE for LOS. This is a diagnostic sanity "
            "check, not part of the frozen thesis benchmark evidence."
        ),
    }
    overall_path = out_dir / "random_probe_overall.json"
    overall_path.write_text(json.dumps(overall, indent=2) + "\n")

    files = [per_seed_path, comparison_path, summary_path, overall_path]
    write_checksum_manifest(out_dir, files, args)

    print(f"Wrote random-probe diagnostic package to {out_dir}")
    print(json.dumps(overall, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
