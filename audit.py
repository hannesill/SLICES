#!/usr/bin/env python3
"""Credential-free audit of anonymous *run-level* aggregates. Standard library only.

Run from the extracted supplement root: uv run --no-project python audit.py
Results are descriptive across five matched split/training seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = {42, 123, 456, 789, 1011}
DATASETS = ("combined", "eicu", "miiv")
TASKS = ("mortality_24h", "mortality_hospital", "aki_kdigo", "los_remaining")
SSL = ("mae", "jepa", "contrastive")
GROUP = (
    "experiment_class",
    "experiment_subtype",
    "dataset",
    "task",
    "protocol",
    "paradigm",
    "label_fraction",
    "model_size",
    "source_dataset",
    "upstream_pretrain_lr",
    "upstream_pretrain_mask_ratio",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for key in (
            "label_fraction",
            "test/auprc",
            "test/mae",
            "baseline/median_predictor_mae",
            "test/n_samples",
            "test/n_positive",
            "test/positive_ratio",
        ):
            if key in r:
                r[key] = float(r[key]) if r[key] else None
    return rows


def metric(row):
    return "test/mae" if row["task"] == "los_remaining" else "test/auprc"


def summary(values):
    return dict(
        mean=st.mean(values),
        sd=st.stdev(values),
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )


def check_rows(rows):
    """Require complete seed sets and unique scientific cells, not just row count."""
    if all("scientific_id" in r for r in rows) and len({r["scientific_id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Duplicate scientific row identity")
    groups = defaultdict(list)
    for r in rows:
        if r["dataset"] not in DATASETS or r["task"] not in TASKS:
            raise ValueError("Unexpected dataset/task")
        if r["revision"] != "thesis-v1":
            raise ValueError("Mixed result revision")
        value = r[metric(r)]
        if value is None or not math.isfinite(value):
            raise ValueError("Missing or nonfinite primary metric")
        if metric(r) == "test/auprc" and not 0 <= value <= 1:
            raise ValueError("AUPRC outside [0, 1]")
        if metric(r) == "test/mae" and value < 0:
            raise ValueError("Negative absolute error")
        groups[tuple(r.get(k) for k in GROUP)].append(r)
    for key, group in groups.items():
        if len(group) != 5 or {r["seed"] for r in group} != SEEDS:
            raise ValueError(f"Incomplete/duplicate seed cell: {key}")
    core = [r for r in rows if r["experiment_class"] == "core_ssl_benchmark"]
    expected = set()
    for d, t, seed in itertools.product(DATASETS, TASKS, SEEDS):
        for p, methods in [("linear_probe", SSL), ("full_finetune", SSL + ("supervised",))]:
            for m in methods:
                expected.add((d, t, seed, p, m))
    actual = [(r["dataset"], r["task"], r["seed"], r["protocol"], r["paradigm"]) for r in core]
    if len(actual) != 420 or set(actual) != expected:
        raise ValueError("Core grid does not contain exactly 420 declared seed rows")
    if any(r["label_fraction"] != 1.0 or r["model_size"] != "default" for r in core):
        raise ValueError("Core fraction/capacity mismatch")
    if any(
        r.get("_eval_checkpoint_source") != "best"
        or str(r.get("_best_ckpt_load_ok")).lower() != "true"
        for r in core
    ):
        raise ValueError("Core evaluation checkpoint mismatch")
    return core


def analyze(rows):
    core = check_rows(rows)
    result = {
        "core_rows": len(core),
        "core_cells": len(core) // 5,
        "replication": "five matched split/training seeds; descriptive only",
    }
    result["core"] = []
    for d, t, p, m in itertools.product(
        DATASETS, TASKS, ("linear_probe", "full_finetune"), SSL + ("supervised",)
    ):
        group = [
            r
            for r in core
            if (r["dataset"], r["task"], r["protocol"], r["paradigm"]) == (d, t, p, m)
        ]
        if group:
            result["core"].append(
                dict(
                    dataset=d,
                    task=t,
                    protocol=p,
                    paradigm=m,
                    **summary([r[metric(r)] for r in group]),
                )
            )
    result["paired"] = []
    for d, t, p, m in itertools.product(
        DATASETS, TASKS, ("linear_probe", "full_finetune"), ("jepa", "contrastive", "supervised")
    ):
        left = {
            r["seed"]: r
            for r in core
            if (r["dataset"], r["task"], r["protocol"], r["paradigm"]) == (d, t, p, "mae")
        }
        right = {
            r["seed"]: r
            for r in core
            if (r["dataset"], r["task"], r["protocol"], r["paradigm"]) == (d, t, p, m)
        }
        if right:
            deltas = [left[s][metric(left[s])] - right[s][metric(right[s])] for s in sorted(SEEDS)]
            result["paired"].append(
                dict(
                    dataset=d,
                    task=t,
                    protocol=p,
                    comparator=m,
                    orientation="MAE minus comparator",
                    differences=deltas,
                    **summary(deltas),
                )
            )
    # Baseline logs occur twice per seed, in core-supervised and classical runs.
    result["constant_los"] = []
    for d in DATASETS:
        vals = []
        for seed in sorted(SEEDS):
            logs = [
                r["baseline/median_predictor_mae"]
                for r in rows
                if r["dataset"] == d
                and r["seed"] == seed
                and r["task"] == "los_remaining"
                and r["label_fraction"] == 1.0
                and r["model_size"] == "default"
                and r.get("baseline/median_predictor_mae") is not None
            ]
            if len(logs) != 2 or len(set(logs)) != 1:
                raise ValueError("LOS duplicate logs do not collapse uniquely")
            vals.append(logs[0])
        mae = [
            r
            for r in result["core"]
            if r["dataset"] == d
            and r["task"] == "los_remaining"
            and r["protocol"] == "linear_probe"
            and r["paradigm"] == "mae"
        ][0]
        result["constant_los"].append(
            dict(dataset=d, **summary(vals), frozen_mae_minus_constant=mae["mean"] - st.mean(vals))
        )
    # Use one source table. Add full-label core rows once only when requested.
    reduced = [
        r
        for r in rows
        if r["experiment_class"] == "label_efficiency"
        and r["label_fraction"] < 1
        and r["model_size"] == "default"
    ]
    result["label_winners"] = []
    for full, p in itertools.product((False, True), ("linear_probe", "full_finetune")):
        for methods in (SSL, SSL + ("supervised",)):
            if p == "linear_probe" and "supervised" in methods:
                continue
            relevant = [
                r
                for r in reduced + (core if full else [])
                if r["protocol"] == p and r["paradigm"] in methods
            ]
            cells = defaultdict(lambda: defaultdict(list))
            for r in relevant:
                cells[(r["dataset"], r["task"], r["label_fraction"])][r["paradigm"]].append(r)
            wins = Counter()
            for cell, methods_rows in cells.items():
                if set(methods_rows) != set(methods):
                    raise ValueError(f"Label comparator coverage mismatch: {cell}")
                means = {m: st.mean(r[metric(r)] for r in rs) for m, rs in methods_rows.items()}
                winner = (min if cell[1] == "los_remaining" else max)(means, key=means.get)
                wins[winner] += 1
            result["label_winners"].append(
                dict(
                    include_full_labels=full,
                    protocol=p,
                    comparators=list(methods),
                    cells=len(cells),
                    wins=dict(wins),
                )
            )
    # Compare each SSL implementation with supervised at the same fraction/seed.
    result["label_vs_supervised"] = []
    for d, t, m in itertools.product(DATASETS, TASKS, SSL):
        for frac in sorted({r["label_fraction"] for r in reduced if r["task"] == t}):
            subset = [
                r
                for r in reduced
                if r["dataset"] == d
                and r["task"] == t
                and r["label_fraction"] == frac
                and r["protocol"] == "full_finetune"
            ]
            a = {r["seed"]: r[metric(r)] for r in subset if r["paradigm"] == m}
            b = {r["seed"]: r[metric(r)] for r in subset if r["paradigm"] == "supervised"}
            if set(a) != SEEDS or set(b) != SEEDS:
                raise ValueError("Label efficiency missing paired supervised row")
            dif = [a[s] - b[s] for s in sorted(SEEDS)]
            result["label_vs_supervised"].append(
                dict(
                    dataset=d,
                    task=t,
                    paradigm=m,
                    label_fraction=frac,
                    ssl_mean=st.mean(a.values()),
                    supervised_mean=st.mean(b.values()),
                    **summary(dif),
                )
            )
    result["hp"] = []
    hp = defaultdict(list)
    for r in rows:
        if r["experiment_class"] == "hp_robustness":
            hp[
                (
                    r["experiment_subtype"],
                    r["paradigm"],
                    r["upstream_pretrain_lr"],
                    r["upstream_pretrain_mask_ratio"],
                )
            ].append(r["test/auprc"])
    for (kind, m, lr, mask), vals in sorted(hp.items()):
        result["hp"].append(dict(subtype=kind, paradigm=m, lr=lr, mask_ratio=mask, **summary(vals)))
    result["test_support"] = []
    for d, t in itertools.product(DATASETS, TASKS):
        group = [
            r
            for r in core
            if r["dataset"] == d and r["task"] == t and r["paradigm"] == "supervised"
        ]
        support = dict(dataset=d, task=t)
        for name in ("n_samples", "n_positive", "positive_ratio"):
            values = [r["test/" + name] for r in group]
            if t == "los_remaining" and name != "n_samples":
                support[name] = None
            else:
                if any(v is None or not math.isfinite(v) for v in values):
                    raise ValueError("Missing test support")
                support[name] = st.mean(values)
        result["test_support"].append(support)
    result["transfer"] = []
    for source, target in (("eicu", "miiv"), ("miiv", "eicu")):
        for t, m in itertools.product(TASKS, SSL):
            group = [
                r
                for r in rows
                if r["experiment_class"] == "cross_dataset_transfer"
                and r["source_dataset"] == source
                and r["dataset"] == target
                and r["task"] == t
                and r["paradigm"] == m
            ]
            if len(group) != 5 or {r["seed"] for r in group} != SEEDS:
                raise ValueError("Transfer grid mismatch")
            result["transfer"].append(
                dict(
                    source=source,
                    target=target,
                    task=t,
                    paradigm=m,
                    **summary([r[metric(r)] for r in group]),
                )
            )
    result["ts2vec"] = []
    for p in ("full_finetune", "linear_probe"):
        wins = Counter({"ts2vec": 0, "contrastive": 0, "ties": 0})
        for d, t in itertools.product(DATASETS, TASKS):
            group = [
                r
                for r in rows
                if r["experiment_class"] == "ts2vec_extension"
                and (r["dataset"], r["task"], r["protocol"]) == (d, t, p)
            ]
            if len(group) != 5 or {r["seed"] for r in group} != SEEDS:
                raise ValueError("TS2Vec extension grid mismatch")
            extension = st.mean(r[metric(r)] for r in group)
            reference = next(
                r["mean"]
                for r in result["core"]
                if (r["dataset"], r["task"], r["protocol"], r["paradigm"])
                == (d, t, p, "contrastive")
            )
            delta = extension - reference
            wins[
                (
                    "ties"
                    if delta == 0
                    else (
                        "ts2vec"
                        if (delta < 0 if t == "los_remaining" else delta > 0)
                        else "contrastive"
                    )
                )
            ] += 1
        result["ts2vec"].append(dict(protocol=p, cells=12, wins=dict(wins)))
    return result


def verify_manifest(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, info in manifest["files"].items():
        path = root / name
        if not path.is_file() or digest(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    return manifest


def analyze_diagnostics(root, rows):
    """Recompute separately sourced controls without adding them to the core grid."""
    random = read_csv(root / "results/random.csv")
    if len(random) != 40 or len({(r["dataset"], r["task"], r["seed"]) for r in random}) != 40:
        raise ValueError("Random diagnostic grid mismatch")
    core = [
        r
        for r in rows
        if r["experiment_class"] == "core_ssl_benchmark"
        and r["protocol"] == "linear_probe"
        and r["paradigm"] in SSL
    ]
    better = 0
    comparisons = 0
    for r in random:
        if r["dataset"] not in {"miiv", "eicu"} or r["task"] not in TASKS or r["seed"] not in SEEDS:
            raise ValueError("Unexpected random diagnostic cell")
        match = [
            v
            for v in core
            if (v["dataset"], v["task"], v["seed"]) == (r["dataset"], r["task"], r["seed"])
        ]
        if len(match) != 3:
            raise ValueError("Random diagnostic pairing failure")
        for ref in match:
            delta = ref[metric(r)] - r[metric(r)]
            better += int(delta < 0 if r["task"] == "los_remaining" else delta > 0)
            comparisons += 1
    with (root / "results/jepa.csv").open() as stream:
        jepa = list(csv.DictReader(stream))
    if len(jepa) != 15 or {(r["dataset"], int(r["seed"])) for r in jepa} != set(
        itertools.product(DATASETS, SEEDS)
    ):
        raise ValueError("JEPA diagnostic coverage mismatch")
    jepa_result = {
        "std_min": min(float(r["std_min_over_training"]) for r in jepa),
        "cos_max": max(float(r["mean_cos_max_over_training"]) for r in jepa),
        "std_last_mean": st.mean(float(r["std_last"]) for r in jepa),
    }
    raw = json.loads((root / "results/raw_mask.json").read_text())
    if not raw.get("complete") or len(raw["results"]) != 10:
        raise ValueError("Incomplete raw/mask control")
    raw_summary = {}
    for method in ("mask_only", "values_and_mask"):
        group = [r for r in raw["results"] if r["representation"] == method]
        if len(group) != 5 or {r["seed"] for r in group} != SEEDS:
            raise ValueError("Incomplete raw/mask seed set")
        for r in group:
            ref = next(
                v
                for v in rows
                if v["experiment_class"] == "core_ssl_benchmark"
                and v["paradigm"] == "supervised"
                and v["dataset"] == "miiv"
                and v["task"] == "mortality_hospital"
                and v["seed"] == r["seed"]
            )
            if (r["n_test"], r["n_test_positive"]) != (
                ref["test/n_samples"],
                ref["test/n_positive"],
            ):
                raise ValueError("Raw/mask test support differs from frozen core")
        raw_summary[method] = summary([r["test_auprc"] for r in group])
    return {
        "random": {"ssl_better": better, "comparisons": comparisons},
        "jepa": jepa_result,
        "raw_mask": raw_summary,
        "raw_mask_seconds": raw["elapsed_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("audit-output"))
    args = parser.parse_args()
    verify_manifest(args.root)
    rows = read_csv(args.root / "results/per_seed.csv")
    configs = json.loads((args.root / "results/scientific_configs.json").read_text())
    if len(configs) != len(rows) or len({c["scientific_id"] for c in configs}) != len(rows):
        raise ValueError("Scientific configuration identity coverage mismatch")
    config_map = {c["scientific_id"]: c for c in configs}
    if set(config_map) != {r["scientific_id"] for r in rows}:
        raise ValueError("Run/config projection join mismatch")
    for r in rows:
        cell = config_map[r["scientific_id"]]["scientific_cell"]
        for key in (*GROUP, "seed"):
            actual = r.get(key)
            expected = cell.get(key)
            if actual in ("", None) and expected is None:
                continue
            if isinstance(expected, (int, float)):
                if float(actual) != expected:
                    raise ValueError("Scientific configuration mismatch")
            elif actual != expected:
                raise ValueError(f"Scientific configuration mismatch: {key}")
    results = analyze(rows)
    expected = json.loads((args.root / "results/expected_analysis.json").read_text())
    if results != expected:
        raise ValueError("Recomputed paper analysis differs from packaged reference")
    diagnostics = analyze_diagnostics(args.root, rows)
    if diagnostics != json.loads((args.root / "results/expected_diagnostics.json").read_text()):
        raise ValueError("Diagnostic recomputation mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    from checkpoint_report import check_checkpoint_results

    checkpoint = check_checkpoint_results(args.root)
    (args.out / "checkpoint-analysis.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    print(
        json.dumps(
            {k: results[k] for k in ("core_rows", "core_cells", "constant_los", "label_winners")},
            indent=2,
        )
    )
    print("PASS: checksums, complete matched-seed grid, recomputed paper analysis")


if __name__ == "__main__":
    main()
