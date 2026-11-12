#!/usr/bin/env python3
"""Render the rebuttal-diagnostic paper tables. Standard library only.

Run from the artifact root: uv run --no-project python render_rebuttal.py [--out DIR]

Inputs: the post-hoc package results/ml4h-rebuttal-v1/, the frozen thesis-v1 runs in
results/per_seed.csv and, when present, results/camera-ready-v1/low_label_train_positives.csv.
Table values are recomputed from per-seed rows (means, sample SDs, differences paired by
seed) and cross-checked against the package's aggregate CSVs/JSON and README tables; any
disagreement beyond float noise or display rounding raises. Prints the summary facts the
paper text cites. Output fragments are bare booktabs tabulars in the style of render.py.
The table builders live in rebuttal_tables/; this file is the command-line entry point.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from audit import analyze, metric, read_csv
from rebuttal_tables import (
    classification_main,
    compute,
    converged_probe,
    label_fraction_delta,
    los_matched,
    low_label_positives,
    per_source,
    reproduction,
    transfer_context,
)
from rebuttal_tables.common import LOS, PKG, POSITIVES, load, readme_tables, seeds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, help="output directory (default ROOT/paper/generated)")
    p.add_argument(
        "--positives", type=Path, help=f"low-label positives CSV (default ROOT/{POSITIVES})"
    )
    a = p.parse_args()
    out = a.out or a.root / "paper/generated"
    out.mkdir(parents=True, exist_ok=True)
    pkg = a.root / PKG
    rows = read_csv(a.root / "results/per_seed.csv")
    analysis = analyze(rows)  # validates the frozen grid as audit.py does
    ref = defaultdict(dict)
    for r in rows:
        if r["experiment_class"] == "core_ssl_benchmark":
            ref[r["dataset"], r["task"], r["protocol"], r["paradigm"]][r["seed"]] = r[metric(r)]
    for r in rows:
        if (
            r["experiment_class"] == "classical_baselines"
            and r["label_fraction"] == 1
            and r["model_size"] == "default"
        ):
            ref[r["dataset"], r["task"], r["protocol"], r["paradigm"]][r["seed"]] = r[metric(r)]
    docs = readme_tables(pkg / "README.md")
    p1 = load(pkg / "p1_converged_probe_per_seed.csv")
    p1_conv = {
        (r["dataset"], r["task"], r["paradigm"], r["probe"]): seeds(
            p1,
            "converged/test_" + ("mae" if r["task"] == LOS else "auprc"),
            dataset=r["dataset"],
            task=r["task"],
            paradigm=r["paradigm"],
            probe=r["probe"],
        )
        for r in p1
    }
    converged_probe(out, pkg, ref, docs)
    classification_main(out, ref, p1_conv, analysis)
    los_matched(out, pkg, ref, analysis, docs)
    per_source(out, pkg, ref, p1_conv, docs)
    compute(out, pkg, docs)
    reproduction(out, pkg, ref, docs)
    label_fraction_delta(out, rows, analysis)
    transfer_context(out, rows, ref, analysis)
    low_label_positives(out, a.positives or a.root / POSITIVES, rows, pkg)
    print(f"Rendered rebuttal tables in {out}")


if __name__ == "__main__":
    main()
