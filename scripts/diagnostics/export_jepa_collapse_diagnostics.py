#!/usr/bin/env python3
"""Export JEPA anti-collapse diagnostics from the W&B `slices-final` project.

JEPA's latent-prediction validation loss is a moving target (the EMA teacher
co-evolves), so a low early-epoch val loss is expected and is NOT evidence of
representation collapse. The direct collapse signals are the target-embedding
standard deviation (collapse -> 0) and the mean pairwise cosine similarity of
target embeddings (collapse -> 1). This script pulls those signals for the core
SSL benchmark JEPA pretraining runs and writes a citable diagnostic artifact.

It is read-only against W&B and writes only a new results artifact. It does not
touch the frozen thesis export.

Usage:
    uv run python scripts/diagnostics/export_jepa_collapse_diagnostics.py \
        --entity hannes-ill --project slices-final --revision thesis-v1 \
        --out results/slices-thesis-jepa-collapse-diagnostic-v1

Outputs (under --out):
    jepa_collapse_history.parquet   per-run, per-logged-step std + mean_cos
    jepa_collapse_summary.csv       one row per run: min/last std, max/last cos
    jepa_collapse_overall.json      corpus-level min std, max cos, run count
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

STD_KEY = "train/jepa_target_repr_std"
COS_KEY = "train/jepa_target_repr_mean_cos"
VAL_STD_KEY = "val/jepa_target_repr_std"
VAL_COS_KEY = "val/jepa_target_repr_mean_cos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="hannes-ill")
    p.add_argument("--project", default="slices-final")
    p.add_argument("--revision", default="thesis-v1")
    p.add_argument(
        "--experiment-class",
        default="core_ssl_benchmark",
        help="config.experiment_class filter for the core pretraining runs.",
    )
    p.add_argument("--out", default="results/slices-thesis-jepa-collapse-diagnostic-v1")
    p.add_argument("--history-samples", type=int, default=1000)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api(timeout=args.timeout)
    runs = api.runs(
        f"{args.entity}/{args.project}",
        filters={
            "config.phase": "pretrain",
            "config.experiment_class": args.experiment_class,
            "config.revision": args.revision,
            "config.paradigm": "jepa",
        },
    )
    runs = list(runs)
    if not runs:
        raise SystemExit(
            "No JEPA pretraining runs matched. Check --entity/--project/--revision "
            "and that config.phase == 'pretrain'."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_rows: list[pd.DataFrame] = []
    summary: list[dict] = []
    for r in runs:
        cfg = r.config
        dataset = cfg.get("dataset")
        seed = cfg.get("seed")
        h = r.history(keys=[STD_KEY, COS_KEY], samples=args.history_samples)
        std_min = std_last = cos_max = cos_last = None
        if h is not None and len(h) and STD_KEY in h:
            std = h[STD_KEY].dropna()
            cos = h[COS_KEY].dropna() if COS_KEY in h else pd.Series(dtype=float)
            if len(std):
                std_min = float(std.min())
                std_last = float(std.iloc[-1])
            if len(cos):
                cos_max = float(cos.max())
                cos_last = float(cos.iloc[-1])
            hh = h.copy()
            hh["run"] = r.name
            hh["dataset"] = dataset
            hh["seed"] = seed
            hist_rows.append(hh)

        summary.append(
            {
                "run": r.name,
                "dataset": dataset,
                "seed": seed,
                "std_min_over_training": std_min,
                "std_last": std_last,
                "mean_cos_max_over_training": cos_max,
                "mean_cos_last": cos_last,
                "summary_val_std": r.summary.get(VAL_STD_KEY),
                "summary_val_mean_cos": r.summary.get(VAL_COS_KEY),
            }
        )

    summary_df = pd.DataFrame(summary).sort_values(["dataset", "seed"])
    summary_df.to_csv(out_dir / "jepa_collapse_summary.csv", index=False)

    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_parquet(
            out_dir / "jepa_collapse_history.parquet", index=False
        )

    overall = {
        "n_runs": int(len(runs)),
        "n_runs_with_history": int(len(hist_rows)),
        "std_min_over_all_runs": (
            float(summary_df["std_min_over_training"].min())
            if summary_df["std_min_over_training"].notna().any()
            else None
        ),
        "std_last_mean": (
            float(summary_df["std_last"].mean()) if summary_df["std_last"].notna().any() else None
        ),
        "mean_cos_max_over_all_runs": (
            float(summary_df["mean_cos_max_over_training"].max())
            if summary_df["mean_cos_max_over_training"].notna().any()
            else None
        ),
        "interpretation": (
            "std bounded away from 0 and mean_cos bounded away from 1 across all "
            "runs indicates no representation collapse; the fixed-budget policy "
            "uses the final encoder.pt regardless."
        ),
    }
    (out_dir / "jepa_collapse_overall.json").write_text(json.dumps(overall, indent=2))

    print(f"Wrote diagnostics for {len(runs)} JEPA runs to {out_dir}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
